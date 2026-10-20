"""Read-only sampling of exposed robot surfaces and the collision-sphere union.

Requires the optional trimesh spatial-query dependency rtree; no SDK or GPU is used.
Use --links to limit an expensive full-robot geometric query to representative links.
Raw geometry is the default. Buffers must be requested explicitly with --radius-mode.
Output contains conditional sampled statistics, never a complete geometric proof.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml
from audit_collision_model import (
    DEFAULT_URDF,
    forward_kinematics,
    load_visual_meshes,
    read_poses,
    retain_central_base,
    sphere_gaps,
    sphere_modes,
)

TOLERANCE_M = 1e-8
RAY_OFFSET_M = 2e-6
QUERY_SCALE = 1000.


def millimetre_query_mesh(mesh, cache, key):
    """Use a cached query-only copy so absolute library tolerances do not erase small triangles."""
    if key not in cache:
        query = mesh.copy()
        query.apply_scale(QUERY_SCALE)
        cache[key] = query
    return cache[key]


def exposed_sphere_points(points, spheres):
    """A sphere surface hidden strictly inside another sphere is not a union boundary."""
    return sphere_gaps(points, spheres) >= -TOLERANCE_M


def sample_sphere_surfaces(spheres, samples_per_sphere):
    if samples_per_sphere < 8:
        raise ValueError("at least eight samples per sphere are required")
    indices = np.arange(samples_per_sphere, dtype=float)
    z = 1. - 2. * (indices + .5) / samples_per_sphere
    angle = indices * np.pi * (3. - np.sqrt(5.))
    radial = np.sqrt(1. - z * z)
    directions = np.column_stack([radial * np.cos(angle), radial * np.sin(angle), z])
    points = spheres[:, None, :3] + spheres[:, None, 3, None] * directions
    owners = np.repeat(np.arange(len(spheres)), samples_per_sphere)
    weights = np.repeat(4. * np.pi * spheres[:, 3] ** 2 / samples_per_sphere, samples_per_sphere)
    return points.reshape(-1, 3), owners, weights


def distance_summary(distances, points, weights=None):
    distances = np.maximum(np.asarray(distances, dtype=float), 0.)
    if not len(distances):
        return {"sample_count": 0, "p95_mm": None, "maximum_mm": None,
                "within_1mm_fraction": None, "worst_point_world_m": None}
    weights = np.ones(len(distances)) if weights is None else np.asarray(weights)
    order = np.argsort(distances, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    p95 = distances[order[np.searchsorted(cumulative, .95)]]
    worst = int(distances.argmax())
    return {"sample_count": len(distances), "p95_mm": float(p95 * 1000),
            "maximum_mm": float(distances[worst] * 1000),
            "mean_mm": float(np.average(distances, weights=weights) * 1000),
            "within_1mm_fraction": float(weights[distances <= .001 + TOLERANCE_M].sum() / weights.sum()),
            "within_3mm_fraction": float(weights[distances <= .003 + TOLERANCE_M].sum() / weights.sum()),
            "within_5mm_fraction": float(weights[distances <= .005 + TOLERANCE_M].sum() / weights.sum()),
            "worst_point_world_m": np.asarray(points)[worst].tolist()}


def points_in_bounds(points, bounds, padding=TOLERANCE_M):
    return ((points >= bounds[0] - padding) & (points <= bounds[1] + padding)).all(axis=1)


def solid_components(meshes):
    parts = []
    for owner, mesh in meshes.items():
        welded = mesh.copy()
        # Establish equal-position connectivity; do not fill holes or alter the surface.
        welded.merge_vertices(merge_tex=True, merge_norm=True)
        for component in welded.split(only_watertight=False, repair=False):
            parts.append({"owner": owner, "mesh": component, "closed": bool(component.is_watertight)})
    return parts


def classify_points(parts, points):
    """Inside closed solids, ambiguous in open-component bounds, or outside all bounds/solids."""
    inside = np.zeros(len(points), dtype=bool)
    unknown = np.zeros(len(points), dtype=bool)
    for part in parts:
        mesh = part["mesh"]
        indices = np.flatnonzero(~inside & points_in_bounds(points, mesh.bounds))
        if not len(indices):
            continue
        if not part["closed"]:
            unknown[indices] = True
            continue
        query = millimetre_query_mesh(mesh, part, "query_mesh")
        for start in range(0, len(indices), 256):
            selected = indices[start:start + 256]
            inside[selected] |= query.contains(points[selected] * QUERY_SCALE)
    unknown[inside] = False
    return inside, unknown


def nearest_surfaces(meshes, points, query_meshes=None):
    """Exact closest triangles; mesh bounds only prune impossible improvements."""
    distance = np.full(len(points), np.inf)
    nearest = np.zeros_like(points)
    owners = [None] * len(points)
    query_meshes = {} if query_meshes is None else query_meshes
    for owner, mesh in meshes.items():
        delta = np.maximum(np.maximum(mesh.bounds[0] - points, points - mesh.bounds[1]), 0.)
        indices = np.flatnonzero(np.linalg.norm(delta, axis=1) <= distance + TOLERANCE_M)
        if not len(indices):
            continue
        query = millimetre_query_mesh(mesh, query_meshes, owner)
        for start in range(0, len(indices), 256):
            selected = indices[start:start + 256]
            closest, values, _ = trimesh.proximity.closest_point(query, points[selected] * QUERY_SCALE)
            closest, values = closest / QUERY_SCALE, values / QUERY_SCALE
            if not np.isfinite(values).all() or not np.isfinite(closest).all():
                raise ValueError(f"nonfinite closest-triangle query for {owner}")
            improved = values < distance[selected]
            improved_indices = selected[improved]
            distance[improved_indices] = values[improved]
            nearest[improved_indices] = closest[improved]
            for index in improved_indices:
                owners[int(index)] = owner
    return nearest, distance, owners


def rays_cross_bounds(origins, directions, bounds):
    near = np.zeros(len(origins))
    far = np.full(len(origins), np.inf)
    valid = np.ones(len(origins), dtype=bool)
    for axis in range(3):
        moving = np.abs(directions[:, axis]) > 1e-14
        valid &= moving | ((origins[:, axis] >= bounds[0, axis]) & (origins[:, axis] <= bounds[1, axis]))
        first = np.full(len(origins), -np.inf)
        second = np.full(len(origins), np.inf)
        first[moving] = (bounds[0, axis] - origins[moving, axis]) / directions[moving, axis]
        second[moving] = (bounds[1, axis] - origins[moving, axis]) / directions[moving, axis]
        near = np.maximum(near, np.minimum(first, second))
        far = np.minimum(far, np.maximum(first, second))
    return valid & (far >= near)


def clear_rays(meshes, origins, directions, query_meshes=None):
    clear = np.ones(len(origins), dtype=bool)
    query_meshes = {} if query_meshes is None else query_meshes
    for owner, mesh in meshes.items():
        indices = np.flatnonzero(clear & rays_cross_bounds(origins, directions, mesh.bounds))
        if not len(indices):
            continue
        query = millimetre_query_mesh(mesh, query_meshes, owner)
        for start in range(0, len(indices), 256):
            selected = indices[start:start + 256]
            clear[selected] &= ~query.ray.intersects_any(origins[selected] * QUERY_SCALE, directions[selected])
    return clear


def visible_surface_points(meshes, points, normals, query_meshes=None):
    """A clear outward or inward normal ray certifies visibility in the literal input mesh."""
    query_meshes = {} if query_meshes is None else query_meshes
    visible = clear_rays(meshes, points + RAY_OFFSET_M * normals, normals, query_meshes)
    remaining = np.flatnonzero(~visible)
    if len(remaining):
        visible[remaining] = clear_rays(meshes, points[remaining] - RAY_OFFSET_M * normals[remaining],
                                        -normals[remaining], query_meshes)
    return visible


def exterior_visible_points(meshes, points, query_meshes=None):
    """Finite directions can certify visibility, but failure is unknown, not proof of enclosure."""
    visible = np.zeros(len(points), dtype=bool)
    query_meshes = {} if query_meshes is None else query_meshes
    for direction in np.vstack([np.eye(3), -np.eye(3)]):
        remaining = np.flatnonzero(~visible)
        if not len(remaining):
            break
        directions = np.broadcast_to(direction, (len(remaining), 3))
        visible[remaining] = clear_rays(meshes, points[remaining], directions, query_meshes)
    return visible


def measure_sphere_boundaries(link, meshes, spheres_by_link, parts, sphere_samples=128, query_meshes=None):
    """Report sampled per-sphere contributions to the present union; deletion requires remeasurement."""
    all_spheres = np.concatenate(list(spheres_by_link.values()))
    query_meshes = {} if query_meshes is None else query_meshes
    boundary, owners, weights = sample_sphere_surfaces(spheres_by_link[link], sphere_samples)
    exposed = exposed_sphere_points(boundary, all_spheres)
    boundary, owners, weights = boundary[exposed], owners[exposed], weights[exposed]
    inside, unknown = classify_points(parts, boundary)
    outside = ~inside & ~unknown
    visible_outside = np.zeros(len(boundary), dtype=bool)
    visible_outside[outside] = exterior_visible_points(meshes, boundary[outside], query_meshes)
    selected = np.flatnonzero(visible_outside)
    closest, distances, closest_owners = nearest_surfaces({link: meshes[link], **meshes}, boundary[selected],
                                                         query_meshes)
    outward = distance_summary(distances, boundary[selected], weights[selected])
    if len(selected):
        worst = int(distances.argmax())
        outward["worst_sphere_index"] = int(owners[selected[worst]])
        outward["nearest_mesh_link"] = closest_owners[worst]
        outward["nearest_point_world_m"] = closest[worst].tolist()
    total_area = weights.sum()
    per_sphere = []
    for sphere_index in range(len(spheres_by_link[link])):
        owned = owners == sphere_index
        selected_owned = np.flatnonzero(owners[selected] == sphere_index)
        record = distance_summary(distances[selected_owned], boundary[selected[selected_owned]])
        if len(selected_owned):
            worst = selected_owned[int(distances[selected_owned].argmax())]
            record["nearest_mesh_link"] = closest_owners[worst]
            record["nearest_point_world_m"] = closest[worst].tolist()
        per_sphere.append({"sphere_index": sphere_index, "sampled_points": sphere_samples,
                           "union_boundary_samples": int(owned.sum()),
                           "inside_closed_geometry_samples": int((inside & owned).sum()),
                           "open_geometry_ambiguous_samples": int((unknown & owned).sum()),
                           "outward": record})
    return {"sphere_union_boundary_samples": len(boundary),
            "boundary_inside_closed_geometry_samples": int(inside.sum()),
            "boundary_open_geometry_ambiguous_samples": int(unknown.sum()),
            "boundary_exterior_but_not_visible_samples": int((outside & ~visible_outside).sum()),
            "certified_exterior_boundary_area_fraction": float(weights[selected].sum() / total_area)
            if total_area else None,
            "outward_distance_conditional_on_certified_exterior": outward, "per_sphere": per_sphere}


def measure_link(link, meshes, spheres_by_link, parts, *, surface_samples=4000, sphere_samples=128, seed=20261001):
    mesh = meshes[link]
    query_meshes = {}
    all_spheres = np.concatenate(list(spheres_by_link.values()))
    points, faces = trimesh.sample.sample_surface(mesh, surface_samples, seed=seed)
    visible = visible_surface_points(meshes, points, mesh.face_normals[faces], query_meshes)
    visible_points = points[visible]
    local_gaps = sphere_gaps(visible_points, spheres_by_link[link])
    global_gaps = sphere_gaps(visible_points, all_spheres)
    coverage = distance_summary(global_gaps, visible_points)
    coverage["covered_fraction"] = float(np.mean(global_gaps <= TOLERANCE_M)) if len(global_gaps) else None
    coverage["owning_link_only"] = distance_summary(local_gaps, visible_points)
    boundary = measure_sphere_boundaries(link, meshes, spheres_by_link, parts, sphere_samples, query_meshes)
    return {"sphere_count": len(spheres_by_link[link]), "mesh_surface_samples": surface_samples,
            "visible_surface_samples": int(visible.sum()), "visible_surface_coverage": coverage, **boundary}


def measure_model(config_path, urdf_path, links, poses_path, radius_mode, surface_samples, sphere_samples):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    local_meshes, assets = load_visual_meshes(urdf_path)
    local_meshes["Link_Base"], classification = retain_central_base(local_meshes["Link_Base"])
    robot = ET.parse(urdf_path).getroot()
    poses, _ = read_poses(poses_path, config)
    links = links or config["collision_link_names"]
    records = []
    for pose in poses:
        transforms = forward_kinematics(robot, config, pose["joint_positions_rad"])
        meshes, spheres = {}, {}
        for link, mesh in local_meshes.items():
            meshes[link] = mesh.copy()
            meshes[link].apply_transform(transforms[link])
        for link in config["collision_link_names"]:
            values = sphere_modes(config, link)[radius_mode]
            transform = transforms[link]
            values[:, :3] = values[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            spheres[link] = values
        parts = solid_components(meshes)
        measurements = {}
        for link in links:
            print(f"Measuring {pose['name']}: {link}", flush=True)
            measurements[link] = measure_link(link, meshes, spheres, parts, surface_samples=surface_samples,
                                               sphere_samples=sphere_samples)
        records.append({"name": pose["name"], "joint_positions_rad": pose["joint_positions_rad"],
                        "closed_components": sum(part["closed"] for part in parts),
                        "open_components": sum(not part["closed"] for part in parts), "links": measurements})
    return {"config": str(config_path.resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(), "trimesh_version": trimesh.__version__,
            "assets": assets,
            "radius_mode": radius_mode, "distance_units": "mm; witness coordinates are world metres",
            "spatial_query_units": "mm query copies for nearest, contains and rays; source meshes unchanged",
            "spatial_query_scale": QUERY_SCALE,
            "sample_seed": 20261001, "sphere_samples_per_sphere": sphere_samples,
            "base_component_classification": classification, "poses": records,
            "limitations": [
                "Percentiles are conditional area-weighted samples, not continuous extrema or full coverage proofs.",
                "Outward statistics exclude internal sphere surfaces, closed-solid interiors, and open-mesh ambiguity.",
                "Outward percentiles cover only the reported certified exterior boundary fraction.",
                "Visibility uses literal mesh ray intersections; finite ray directions can miss exposed concavities.",
                "Open-component bounds conservatively mark sign ambiguity; no mesh holes are repaired or hulls used.",
                "Surface coverage uses areas visible along at least one face-normal ray, not all manufactured surfaces.",
                "Pose-dependent neighbouring meshes and all robot spheres participate in visibility and distances.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--links", nargs="+")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--radius-mode", choices=["raw", "world_buffered", "self_effective"], default="raw")
    parser.add_argument("--surface-samples", type=int, default=4000)
    parser.add_argument("--sphere-samples", type=int, default=128)
    args = parser.parse_args()
    inputs = {path.resolve() for path in [args.config, args.urdf, args.poses] if path}
    if args.output.resolve() in inputs:
        parser.error("output must not overwrite an input")
    if args.surface_samples < 1 or args.sphere_samples < 8:
        parser.error("surface samples must be positive and sphere samples at least eight")
    report = measure_model(args.config, args.urdf, args.links, args.poses, args.radius_mode,
                           args.surface_samples, args.sphere_samples)
    if args.output.resolve() in {Path(asset["path"]) for asset in report["assets"]}:
        parser.error("output must not overwrite a source mesh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
