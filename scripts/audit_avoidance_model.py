"""Compare obstacle-avoidance sphere models without fitting or running the robot.

Exit zero means measurements were produced, not approval. Surface diagnostics include
CAD holes and internal faces; they are not instructions to add spheres at every gap.
Use the existing geometry audit and fixed rendering views to review relevant witnesses.
"""

import argparse
import copy
import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import trimesh
import yaml
from audit_collision_model import (
    DEFAULT_URDF,
    SEED,
    audit_pose,
    audit_surface,
    box_sdf,
    forward_kinematics,
    load_visual_meshes,
    read_poses,
    retain_central_base,
    sphere_modes,
)
from scipy.spatial.transform import Rotation

from jev_mujoco.geometry import primitive_sdf, rigid_matrix


def validate_configuration(baseline, candidate):
    """Allow sphere replacements only; keep all 38 links and safety settings."""
    copies = [copy.deepcopy(value) for value in (baseline, candidate)]
    sphere_maps = [value["robot_cfg"]["kinematics"].pop("collision_spheres") for value in copies]
    if set(sphere_maps[0]) != set(sphere_maps[1]):
        raise ValueError("candidate sphere links differ from the baseline")
    if copies[0] != copies[1]:
        raise ValueError("candidate changed non-sphere configuration")
    config = candidate["robot_cfg"]["kinematics"]
    for link in config["collision_link_names"]:
        sphere_modes(config, link)


def axis_envelope(mesh, spheres):
    """Report axis extents only, not exact geometric protrusion or coverage."""
    mesh_bounds = np.asarray(mesh.bounds)
    bounds = np.array(
        [
            (spheres[:, :3] - spheres[:, 3, None]).min(axis=0),
            (spheres[:, :3] + spheres[:, 3, None]).max(axis=0),
        ]
    )
    return {
        "mesh_bounds_m": mesh_bounds.tolist(),
        "sphere_bounds_m": bounds.tolist(),
        "missing_extent_m": np.maximum(
            [bounds[0] - mesh_bounds[0], mesh_bounds[1] - bounds[1]], 0.0
        ).tolist(),
        "outward_extent_m": np.maximum(
            [mesh_bounds[0] - bounds[0], bounds[1] - mesh_bounds[1]], 0.0
        ).tolist(),
    }


def target_box_clearances(spheres_tcp, target_from_tcp, target_size):
    transform = np.asarray(target_from_tcp, dtype=float)
    centers = spheres_tcp[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    return box_sdf(centers, [0.0, 0.0, 0.0], target_size) - spheres_tcp[:, 3]


def read_audit_poses(path, config):
    poses, table = read_poses(None, config)
    if path is None:
        return poses, table
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    payload = json.loads(data)
    if isinstance(payload, dict):
        table = payload.get("table", table)
        payload = payload["poses"]
    if not isinstance(payload, list):
        raise TypeError("external poses must be a JSON list")
    names = set(config["cspace"]["joint_names"])
    required = names.difference(config.get("lock_joints", {}))
    for index, pose in enumerate(payload):
        positions = pose["joint_positions_rad"]
        if not isinstance(positions, dict) or not required.issubset(positions):
            raise ValueError(f"external pose {index} must contain every unlocked cspace joint")
        if set(positions).difference(names) or not np.isfinite(list(positions.values())).all():
            raise ValueError(f"external pose {index} has unknown joints or nonfinite positions")
        poses.append({"name": pose.get("name", f"external-{index}"), "joint_positions_rad": positions})
    return poses, table


def compact_pose(result):
    table = result["table"]
    closest_link = min(
        table["clearance_by_link"], key=lambda link: table["clearance_by_link"][link]["clearance_m"]
    )
    return {
        "self_collision": result["self_collision"],
        "table": {
            "minimum_clearance_m": table["minimum_clearance_m"],
            "closest_link": closest_link,
            "sphere_index": table["clearance_by_link"][closest_link]["sphere_index"],
        },
    }


def task_channel(robot, config, fixture):
    """Diagnostic open-gripper/camera clearance along the recorded linear approach.

    The planner intentionally excludes the target. This independent query does not
    alter that policy and does not substitute for finger contact or grasp simulation.
    """
    request = fixture["request"]
    arm = request["arm"]
    suffix, grip = {"A": ("L", "gripL_"), "B": ("R", "gripR_")}[arm]
    target = next(
        obj
        for obj in fixture["simulation_snapshot_after_scene_build"]["objects"]
        if obj["id"] == fixture["scene"]["target_id"]
    )
    if target["kind"] not in {"cube", "box"}:
        raise ValueError("task-channel audit currently requires a recorded box target")
    target_from_world = np.linalg.inv(np.asarray(target["T_world_object"]))
    transforms = forward_kinematics(robot, config, {})
    tcp = f"TCP_Link_{suffix}"
    tcp_from_world = np.linalg.inv(transforms[tcp])
    arrays, owners = {"raw": [], "world_buffered": []}, []
    for link in config["collision_link_names"]:
        if link != tcp and not link.startswith(grip):
            continue
        link_to_tcp = tcp_from_world @ transforms[link]
        modes = sphere_modes(config, link)
        owners.extend({"link": link, "sphere_index": index} for index in range(len(modes["raw"])))
        for mode, group in arrays.items():
            values = modes[mode].copy()
            values[:, :3] = values[:, :3] @ link_to_tcp[:3, :3].T + link_to_tcp[:3, 3]
            group.append(values)
    arrays = {mode: np.concatenate(values) for mode, values in arrays.items()}
    axis = np.eye(3)["xyz".index(request["approach_axis"])]
    records = []
    for candidate in fixture["selected_candidates"]:
        goal = np.asarray(candidate["T_world_tcp"])
        direction = goal[:3, :3] @ axis if request["approach_in_tool_frame"] else axis
        minima = {mode: None for mode in arrays}
        for fraction in np.linspace(0.0, 1.0, 11):
            pose = goal.copy()
            pose[:3, 3] += direction * request["approach_offset_m"] * (1.0 - fraction)
            for mode, spheres in arrays.items():
                gaps = target_box_clearances(spheres, target_from_world @ pose, target["size"])
                index = int(gaps.argmin())
                if minima[mode] is None or gaps[index] < minima[mode]["clearance_m"]:
                    minima[mode] = {
                        **owners[index],
                        "clearance_m": float(gaps[index]),
                        "approach_fraction": float(fraction),
                    }
        records.append(
            {"candidate_index": candidate["candidate_index"], "score": candidate["score"], **minima}
        )
    return {
        "arm": arm,
        "target_id": target["id"],
        "target_size_m": target["size"],
        "open_gripper_lock": config.get("lock_joints", {}),
        "approach_samples": 11,
        "goals": records,
        "meaning": "Sphere-to-target diagnostic; negative clearance requires geometry/contact review.",
    }


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def obstacle_transform(obstacle):
    if "T_world_object" in obstacle:
        return rigid_matrix(obstacle["T_world_object"])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(obstacle["quaternion"], scalar_first=True).as_matrix()
    transform[:3, 3] = obstacle["position"]
    return rigid_matrix(transform)


def audit_visual_world(robot, config, poses, scene, meshes, *, samples=1000):
    """Check actual visual vertices/samples against known non-target primitives.

    This deliberately does not use collision spheres. It can detect sampled mesh
    intersections but is not continuous or exact mesh/mesh collision certification.
    """
    points = {}
    for link in config["collision_link_names"]:
        mesh = meshes[link]
        surface, _ = trimesh.sample.sample_surface(mesh, samples, seed=SEED + zlib.crc32(link.encode()))
        points[link] = np.vstack([np.unique(np.asarray(mesh.vertices), axis=0), surface])
    table_transform = np.eye(4)
    table_transform[:3, 3] = scene["table"]["position"]
    obstacles = [
        {"id": "table", "kind": "box", "size": scene["table"]["size"], "T_world_object": table_transform}
    ]
    obstacles.extend(obj for obj in scene["objects"] if obj["id"] != scene["target_id"])
    inverses = {obj["id"]: np.linalg.inv(obstacle_transform(obj)) for obj in obstacles}
    records = []
    for pose in poses:
        transforms = forward_kinematics(robot, config, pose["joint_positions_rad"])
        nearest = None
        for link, local_points in points.items():
            transform = transforms[link]
            world_points = local_points @ transform[:3, :3].T + transform[:3, 3]
            for obstacle in obstacles:
                inverse = inverses[obstacle["id"]]
                query = world_points @ inverse[:3, :3].T + inverse[:3, 3]
                distances = primitive_sdf(query, obstacle["kind"], obstacle["size"])
                index = int(distances.argmin())
                if nearest is None or distances[index] < nearest["clearance_m"]:
                    nearest = {
                        "link": link,
                        "obstacle_id": obstacle["id"],
                        "clearance_m": float(distances[index]),
                        "point_link_m": local_points[index].tolist(),
                        "point_world_m": world_points[index].tolist(),
                    }
        records.append({"name": pose["name"], "nearest": nearest})
    return {
        "excluded_target_id": scene["target_id"],
        "links": list(points),
        "point_count_by_link": {link: len(values) for link, values in points.items()},
        "area_samples_per_link": samples,
        "poses": records,
        "limitation": "All unique visual vertices plus fixed area samples; discrete supplied configurations. "
        "No sphere buffers, no target contact, no continuous or exact triangle intersection proof.",
    }


def audit(baseline_path, config_path, urdf_path, *, poses_path=None, fixture_path=None, samples=1000):
    models = [yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in (baseline_path, config_path)]
    validate_configuration(*models)
    configs = [model["robot_cfg"]["kinematics"] for model in models]
    robot = ET.parse(urdf_path).getroot()
    meshes, assets = load_visual_meshes(urdf_path)
    links = {}
    for link in configs[0]["collision_link_names"]:
        mesh = meshes[link]
        classification = None
        if link == "Link_Base":
            mesh, classification = retain_central_base(mesh)
        comparison = {}
        for name, config in zip(("baseline", "candidate"), configs, strict=True):
            modes = sphere_modes(config, link)
            modes = {mode: modes[mode] for mode in ("raw", "world_buffered")}
            comparison[name] = {
                "sphere_count": len(modes["raw"]),
                "axis_envelope": axis_envelope(mesh, modes["raw"]),
                "all_mesh_diagnostic": audit_surface(
                    mesh, modes, sample_count=samples, seed=SEED + zlib.crc32(link.encode())
                ),
            }
        links[link] = comparison
        if classification is not None:
            links[link]["geometry_classification"] = classification
    poses, table = read_audit_poses(poses_path, configs[0])
    pose_records = []
    same_models = models[0] == models[1]
    for pose in poses:
        baseline = compact_pose(audit_pose(robot, configs[0], pose["joint_positions_rad"], table))
        candidate = (
            baseline
            if same_models
            else compact_pose(audit_pose(robot, configs[1], pose["joint_positions_rad"], table))
        )
        pose_records.append({"name": pose["name"], "baseline": baseline, "candidate": candidate})
    report = {
        "schema_version": 1,
        "baseline": file_record(baseline_path),
        "candidate": file_record(config_path),
        "urdf": file_record(urdf_path),
        "assets": assets,
        "non_sphere_configuration_unchanged": True,
        "quality_verdict": "requires_geometric_and_task_review",
        "units": "metres and radians",
        "surface_samples_per_link": samples,
        "seed": SEED,
        "links": links,
        "pose_count": len(poses),
        "table": table,
        "poses": pose_records,
        "limitations": [
            "All-mesh surface gaps include internal CAD faces, holes and fasteners; not a coverage gate.",
            "Axis bounds detect gross extent changes, not exact mesh distance or all protrusions.",
            "Positive surface gaps need body-level geometric review, not automatic sphere patches.",
            "Sphere self/table checks cover supplied configurations, not continuous motion or all poses.",
            "New old-path collisions require reviewing real geometry; neither automatic rejection nor approval.",
            "Only the recorded table is checked as a world obstacle; other scene objects are not included.",
            "Target-channel queries use the locked open gripper and discrete linear approach, not grasp contact.",
            "This CPU audit does not establish cuRobo planning success or physical execution safety.",
        ],
    }
    if poses_path:
        report["poses_source"] = file_record(poses_path)
    if fixture_path:
        fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        report["task_fixture"] = file_record(fixture_path)
        report["task_channel"] = {
            name: task_channel(robot, config, fixture)
            for name, config in zip(("baseline", "candidate"), configs, strict=True)
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.baseline, args.config, args.urdf, args.poses, args.fixture]
    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.output.resolve() in {path.resolve() for path in inputs if path}:
        parser.error("output must not replace an input")
    report = audit(
        args.baseline,
        args.config,
        args.urdf,
        poses_path=args.poses,
        fixture_path=args.fixture,
        samples=args.samples,
    )
    if args.output.resolve() in {Path(asset["path"]) for asset in report["assets"]}:
        parser.error("output must not replace a visual asset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Compared {len(report['links'])} links and {report['pose_count']} poses: {args.output}")


if __name__ == "__main__":
    main()
