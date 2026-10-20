"""Read-only CPU measurements for a candidate collision-sphere YAML.

Example: python scripts/audit_collision_model.py --config candidate.yml --output audit.json
Optional --poses accepts [{"name": "frame-0", "joint_positions_rad": {"Joint1_L": ...}}].
It may instead contain {"poses": [...], "table": {"position": [...], "size": [...]}}.
Every external pose must specify all unlocked cspace joints; configured gripper locks prevail.
Exit zero means measurements were produced, not that a candidate is safe or approved.
No fitting, mesh repair, rtree, cuRobo import, GPU, or runtime service is used.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf"
DEFAULT_TABLE = {"position": [.55, 0., .56], "size": [.7, 1., .08]}
STAND_WITNESS = [-.060, -.010785, .163894]
SEED = 20260930
SURFACE_SAMPLES = 10_000
COVERAGE_TOLERANCE_M = 1e-9


def origin_transform(element):
    transform = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        transform[:3, 3] = [float(value) for value in origin.get("xyz", "0 0 0").split()]
    return transform


def load_visual_meshes(urdf_path):
    """Read visual surfaces in their owning link frames, preserving source files."""
    urdf_path = Path(urdf_path)
    meshes, assets = {}, []
    for link in ET.parse(urdf_path).getroot().findall("link"):
        parts = []
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh_node = geometry.find("mesh")
            if mesh_node is not None:
                path = (urdf_path.parent / mesh_node.get("filename")).resolve()
                mesh = trimesh.load_mesh(path, process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(f"expected triangle mesh: {path}")
                scale = np.array([float(value) for value in mesh_node.get("scale", "1 1 1").split()])
                mesh.vertices = np.asarray(mesh.vertices) * scale
                assets.append({"link": link.get("name"), "path": str(path),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "scale": scale.tolist(), "origin": origin_transform(visual).tolist()})
            elif geometry.find("box") is not None:
                size = [float(value) for value in geometry.find("box").get("size").split()]
                mesh = trimesh.creation.box(extents=size)
            else:
                raise ValueError(f"unsupported visual geometry on {link.get('name')}")
            mesh.apply_transform(origin_transform(visual))
            parts.append(mesh)
        if parts:
            meshes[link.get("name")] = trimesh.util.concatenate(parts)
    return meshes, assets


def retain_central_base(mesh):
    """Exclude reviewed floor components without cutting away low central ribs."""
    connected = mesh.copy()
    # Welding equal-position vertices only establishes connectivity. Never fill holes.
    connected.merge_vertices(merge_tex=True, merge_norm=True)
    components = connected.split(only_watertight=False, repair=False)
    retained, records = [], []
    for index, component in enumerate(components):
        radial_extent = float(np.linalg.norm(component.vertices[:, :2], axis=1).max())
        keep = radial_extent <= .161
        if not keep and component.bounds[1, 2] > .075:
            raise ValueError(f"unclassified Link_Base component {index}; review the changed source geometry")
        if keep:
            retained.append(component)
        records.append({"component_index": index, "retained": keep, "faces": len(component.faces),
                        "bounds_m": component.bounds.tolist(), "maximum_radius_xy_m": radial_extent})
    if not retained:
        raise ValueError("Link_Base has no retained central geometry")
    return trimesh.util.concatenate(retained), {
        "rule": "retain components with maximum XY radius <=161 mm; exclude other components only below 75 mm",
        "component_count": len(components), "retained_components": len(retained),
        "excluded_components": len(components) - len(retained), "components": records,
    }


def sphere_modes(config, link):
    raw = np.array([[*sphere["center"], sphere["radius"]]
                    for sphere in config["collision_spheres"][link]], dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all() or np.any(raw[:, 3] <= 0):
        raise ValueError(f"invalid collision spheres for {link}")
    world_buffer = config.get("collision_sphere_buffer", 0.)
    if isinstance(world_buffer, dict):
        world_buffer = world_buffer.get(link, 0.)
    self_buffer = config.get("self_collision_buffer", {}).get(link, 0.)
    world = raw.copy()
    world[:, 3] += float(world_buffer)
    effective = world.copy()
    effective[:, 3] += float(self_buffer)
    return {"raw": raw, "world_buffered": world, "self_effective": effective}


def sphere_gaps(points, spheres):
    points = np.asarray(points, dtype=float)
    result = np.empty(len(points))
    for start in range(0, len(points), 2048):
        distances = np.linalg.norm(points[start:start + 2048, None] - spheres[None, :, :3], axis=2)
        result[start:start + 2048] = (distances - spheres[None, :, 3]).min(axis=1)
    return result


def gap_statistics(points, spheres):
    gaps = sphere_gaps(points, spheres)
    worst = int(np.argmax(gaps))
    positive = np.maximum(gaps, 0.)
    return {"count": len(points), "covered_fraction": float(np.mean(gaps <= COVERAGE_TOLERANCE_M)),
            "mean_gap_m": float(positive.mean()), "p95_gap_m": float(np.percentile(positive, 95)),
            "maximum_gap_m": float(positive.max()), "worst_point_m": points[worst].tolist(),
            "maximum_signed_gap_m": float(gaps[worst])}


def audit_surface(mesh, modes, *, sample_count=SURFACE_SAMPLES, seed=SEED):
    if not len(mesh.vertices) or not len(mesh.faces) or mesh.area <= 0:
        raise ValueError("visual surface has no nondegenerate triangles")
    samples, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=seed)
    return {name: {"vertices": gap_statistics(mesh.vertices, spheres),
                   "samples": gap_statistics(samples, spheres),
                   "nonpositive_effective_radii": int(np.count_nonzero(spheres[:, 3] <= 0))}
            for name, spheres in modes.items()}


def strict_containment(spheres):
    distances = np.linalg.norm(spheres[:, None, :3] - spheres[None, :, :3], axis=2)
    slack = spheres[None, :, 3] - spheres[:, None, 3] - distances
    np.fill_diagonal(slack, -np.inf)
    result = []
    for index, row in enumerate(slack):
        container = int(row.argmax())
        if row[container] > 0:
            result.append({"sphere_index": index, "contained_by": container,
                           "slack_m": float(row[container])})
    return result


def forward_kinematics(robot, config, joint_positions_rad):
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    values = dict(joint_positions_rad)
    values.update(config.get("lock_joints", {}))

    def joint_value(name, ancestors=()):
        if name in values:
            return float(values[name])
        if name in ancestors:
            raise ValueError("cyclic URDF mimic joints")
        mimic = joints[name].find("mimic")
        if mimic is None:
            return 0.
        return (joint_value(mimic.get("joint"), (*ancestors, name)) * float(mimic.get("multiplier", "1"))
                + float(mimic.get("offset", "0")))

    transforms = {config["base_link"]: np.eye(4)}
    pending = list(joints.values())
    while pending:
        previous_count = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            motion = np.eye(4)
            kind = joint.get("type")
            if kind in {"revolute", "continuous", "prismatic"}:
                axis_node = joint.find("axis")
                axis = np.array([float(value) for value in
                                 (axis_node.get("xyz") if axis_node is not None else "1 0 0").split()])
                axis /= np.linalg.norm(axis)
                position = joint_value(joint.get("name"))
                if kind == "prismatic":
                    motion[:3, 3] = axis * position
                else:
                    motion[:3, :3] = Rotation.from_rotvec(axis * position).as_matrix()
            elif kind != "fixed":
                raise ValueError(f"unsupported URDF joint type: {kind}")
            transforms[joint.find("child").get("link")] = transforms[parent] @ origin_transform(joint) @ motion
            pending.remove(joint)
        if len(pending) == previous_count:
            raise ValueError("URDF joints are disconnected from the configured base or contain a cycle")
    return transforms


def box_sdf(points, position, size):
    size = np.asarray(size, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("table size must have three positive finite values")
    delta = np.abs(np.asarray(points) - np.asarray(position)) - size / 2
    return np.linalg.norm(np.maximum(delta, 0.), axis=-1) + np.minimum(delta.max(axis=-1), 0.)


def audit_pose(robot, config, joint_positions_rad, table):
    transforms = forward_kinematics(robot, config, joint_positions_rad)
    spheres, table_clearance = {}, {}
    for link in config["collision_link_names"]:
        modes = sphere_modes(config, link)
        transform = transforms[link]
        for values in modes.values():
            values[:, :3] = values[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        spheres[link] = modes["self_effective"]
        world = modes["world_buffered"]
        gaps = box_sdf(world[:, :3], table["position"], table["size"]) - world[:, 3]
        index = int(gaps.argmin())
        table_clearance[link] = {"sphere_index": index, "clearance_m": float(gaps[index])}
    overlaps, closest, pair_count = [], None, 0
    ignored = config.get("self_collision_ignore", {})
    for first, second in combinations(config["collision_link_names"], 2):
        if second in ignored.get(first, []) or first in ignored.get(second, []):
            continue
        a, b = spheres[first], spheres[second]
        pair_count += len(a) * len(b)
        gaps = np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=2) - a[:, None, 3] - b[None, :, 3]
        i, j = np.unravel_index(gaps.argmin(), gaps.shape)
        record = {"links": [first, second], "sphere_indices": [int(i), int(j)], "gap_m": float(gaps[i, j])}
        if closest is None or record["gap_m"] < closest["gap_m"]:
            closest = record
        if record["gap_m"] < -COVERAGE_TOLERANCE_M:
            overlaps.append(record)
    minimum = min(value["clearance_m"] for value in table_clearance.values())
    return {"applied_joint_positions_rad": {**joint_positions_rad, **config.get("lock_joints", {})},
            "self_collision": {"potential_sphere_pairs": pair_count, "closest_pair": closest,
                               "overlapping_link_pairs": overlaps},
            "table": {"minimum_clearance_m": minimum, "clearance_by_link": table_clearance,
                      "meets_10mm_clearance": minimum >= .01 - COVERAGE_TOLERANCE_M}}


def read_poses(path, config):
    defaults = dict(zip(config["cspace"]["joint_names"], config["cspace"]["default_joint_position"], strict=True))
    poses = [{"name": "default", "joint_positions_rad": defaults},
             {"name": "zero", "joint_positions_rad": {name: 0. for name in defaults}}]
    table = DEFAULT_TABLE.copy()
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            table = payload.get("table", table)
            payload = payload["poses"]
        if not isinstance(payload, list):
            raise ValueError("external poses must be a JSON list")
        required = set(defaults).difference(config.get("lock_joints", {}))
        for index, pose in enumerate(payload):
            positions = pose["joint_positions_rad"]
            if not isinstance(positions, dict) or not required.issubset(positions):
                raise ValueError(f"external pose {index} must contain every unlocked cspace joint")
            if set(positions).difference(defaults) or not np.isfinite(list(positions.values())).all():
                raise ValueError(f"external pose {index} has unknown joints or nonfinite positions")
            poses.append({"name": pose.get("name", f"external-{index}"), "joint_positions_rad": positions})
    return poses, table


def audit_model(config_path, urdf_path=DEFAULT_URDF, poses_path=None):
    config_path, urdf_path = Path(config_path), Path(urdf_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    for gripper in ("gripL_finger_joint", "gripR_finger_joint"):
        if gripper in config["cspace"]["joint_names"] and config.get("lock_joints", {}).get(gripper) != 0.:
            raise ValueError(f"candidate must preserve the open-gripper zero lock: {gripper}")
    robot = ET.parse(urdf_path).getroot()
    meshes, assets = load_visual_meshes(urdf_path)
    links = {}
    for link in config["collision_link_names"]:
        mesh = meshes[link]
        classification = None
        if link == "Link_Base":
            mesh, classification = retain_central_base(mesh)
        modes = sphere_modes(config, link)
        record = {"sphere_count": len(modes["raw"]), "visual_vertices": len(mesh.vertices),
                  "visual_faces": len(mesh.faces), "visual_bounds_m": mesh.bounds.tolist(),
                  "surface": audit_surface(mesh, modes, seed=SEED + zlib.crc32(link.encode())),
                  "strictly_contained": strict_containment(modes["raw"])}
        if classification is not None:
            record["static_geometry_classification"] = classification
        if link == "Link_Stand":
            record["old_gap_witness"] = {
                "point_link_m": STAND_WITNESS, "historical_visual_gap_mm": 16.448,
                "signed_gap_m": {name: float(sphere_gaps(np.array([STAND_WITNESS]), values)[0])
                                 for name, values in modes.items()},
            }
        links[link] = record
    poses, table = read_poses(poses_path, config)
    return {"config_path": str(config_path.resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "urdf_path": str(urdf_path.resolve()), "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
            "units": "metres and radians unless explicitly named otherwise", "seed": SEED,
            "surface_samples_per_link": SURFACE_SAMPLES, "coverage_tolerance_m": COVERAGE_TOLERANCE_M,
            "sphere_count": sum(value["sphere_count"] for value in links.values()),
            "self_radius_semantics": "raw + collision_sphere_buffer + self_collision_buffer; pinned SDK 8e734f3",
            "lock_joints": config.get("lock_joints", {}), "table": table, "assets": assets, "links": links,
            "poses": [{**pose, **audit_pose(robot, config, pose["joint_positions_rad"], table)} for pose in poses],
            "limitations": ["Measurements are not a cuRobo planning or execution safety proof.",
                            "Surface coverage samples and vertices do not prove complete solid-volume coverage.",
                            "Containment uses float64 and does not prove equivalent optimization costs or gradients.",
                            "Pose checks use the full configured robot ignore matrix, not a frozen-peer reduction.",
                            "Link_Base floor components are deliberately excluded from the reviewed fitting surface."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--poses", type=Path)
    arguments = parser.parse_args()
    inputs = {arguments.config.resolve(), arguments.urdf.resolve()}
    if arguments.poses:
        inputs.add(arguments.poses.resolve())
    if arguments.output.resolve() in inputs:
        parser.error("output must not replace an input file")
    report = audit_model(arguments.config, arguments.urdf, arguments.poses)
    if arguments.output.resolve() in {Path(asset["path"]) for asset in report["assets"]}:
        parser.error("output must not replace a visual mesh input")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audited {len(report['links'])} links, {report['sphere_count']} spheres, "
          f"{len(report['poses'])} poses: {arguments.output}")


if __name__ == "__main__":
    main()
