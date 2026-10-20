"""Reproduce the measured Link4 correction with project-side CPU sphere fitting.

This is an offline geometry tool, not a cuRobo fitting implementation. Its fixed
reference attachment came from recorded visible RGBD points. Runtime attachment
fitting, collision buffers and collision exclusions are not changed here.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "resources/curobo/link4-refit-20261016.json"


def enclosing_sphere(points, attachment, generation):
    """Contain every patch vertex, retaining the measured attachment clearance."""
    points = np.unique(points, axis=0) * 1000
    if len(points) > 4:
        try:
            points = points[ConvexHull(points).vertices]
        except QhullError:
            pass  # A planar patch still has valid enclosing-sphere constraints.
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    initial = np.r_[center, np.linalg.norm(points - center, axis=1).max()]
    obstacles = attachment * 1000
    open_halfspace = generation["open_gripper_base_halfspace"]
    constraints = [
        {"type": "ineq", "fun": lambda value:
         value[3] ** 2 - np.sum((points - value[:3]) ** 2, axis=1)},
        {"type": "ineq", "fun": lambda value:
         np.linalg.norm(value[:3] - obstacles[:, :3], axis=1) - value[3] - obstacles[:, 3]
         - generation["attachment_reference_minimum_gap_m"] * 1000},
        {"type": "ineq", "fun": lambda value:
         value[:3] @ np.asarray(open_halfspace["normal"])
         + open_halfspace["offset_m"] * 1000 - value[3] - open_halfspace["minimum_gap_m"] * 1000},
    ]
    result = minimize(
        lambda value: value[3], initial, jac=lambda value: np.array([0., 0., 0., 1.]),
        constraints=constraints, bounds=[(None, None)] * 3 + [(.001, generation["radius_bound_m"] * 1000)],
        method="SLSQP", options={"ftol": 1e-8, "maxiter": 100},
    )
    center = result.x[:3] / 1000
    # Recompute the radius from all extremal vertices; optimizer tolerances
    # cannot turn an almost-contained surface into a false coverage claim.
    sphere = np.r_[center, np.linalg.norm(points / 1000 - center, axis=1).max() + 1e-7]
    return sphere, bool(result.success)


def fit_template(mesh, attachment, generation):
    vertices, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=generation["mesh_subdivision_max_edge_m"])
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    _, _, directions = np.linalg.svd(mesh.vertices - mesh.vertices.mean(axis=0), full_matrices=False)
    long_axis = directions[0].copy()
    long_axis[2] = 0
    long_axis /= np.linalg.norm(long_axis)
    coordinate = centers @ long_axis
    long_bins = np.minimum(((coordinate - coordinate.min()) / np.ptp(coordinate) * 5).astype(int), 4)
    z_bins = np.minimum(((centers[:, 2] - centers[:, 2].min()) / np.ptp(centers[:, 2]) * 2).astype(int), 1)
    groups = [np.flatnonzero((long_bins == i) & (z_bins == j)) for i in range(5) for j in range(2)]
    groups = [group for group in groups if len(group)]
    fitted = [enclosing_sphere(triangles[group].reshape(-1, 3), attachment, generation) for group in groups]
    max_radius = generation["split_until_radius_m"]
    while (len(groups) < generation["maximum_template_spheres"]
           and (not all(ok for _, ok in fitted) or max(s[3] for s, _ in fitted) > max_radius)):
        index = int(np.argmax([sphere[3] + (.1 if not ok else 0.) for sphere, ok in fitted]))
        group = groups[index]
        choices = []
        for axis in (long_axis, np.array([0., 0., 1.]), np.cross(long_axis, [0., 0., 1.])):
            values = centers[group] @ axis
            middle = (values.min() + values.max()) / 2
            children = [group[values <= middle], group[values > middle]]
            if not all(len(child) for child in children):
                continue
            spheres = [enclosing_sphere(triangles[child].reshape(-1, 3), attachment, generation)
                       for child in children]
            score = max(sphere[3] + (.1 if not ok else 0.) for sphere, ok in spheres)
            choices.append((score, children, spheres))
        if not choices:
            raise ValueError("Cannot split an infeasible Link4 patch")
        _, children, spheres = min(choices, key=lambda choice: choice[0])
        groups[index:index + 1] = children
        fitted[index:index + 1] = spheres
    if not all(ok for _, ok in fitted) or max(sphere[3] for sphere, _ in fitted) > max_radius:
        raise ValueError("No feasible Link4 template within the recorded sphere budget")
    return np.array([sphere for sphere, _ in fitted])


def validate_final_template(spheres, attachment, generation):
    """Postprocessing may expand spheres; recheck the actual returned envelope."""
    if (not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)
            or spheres[:, 3].max() > generation["split_until_radius_m"]):
        raise ValueError("Final Link4 radius exceeds the recorded limit")
    gaps = (np.linalg.norm(spheres[:, None, :3] - attachment[None, :, :3], axis=-1)
            - spheres[:, None, 3] - attachment[None, :, 3])
    if gaps.min() <= 0:
        raise ValueError("Final Link4 envelope intersects the reference attachment")
    plane = generation["open_gripper_base_halfspace"]
    gaps = spheres[:, :3] @ np.asarray(plane["normal"]) + plane["offset_m"] - spheres[:, 3]
    if gaps.min() <= 0:
        raise ValueError("Final Link4 envelope crosses the open gripper midplane")


def build_candidate(source, record, project=ROOT):
    for name, digest in record["geometry_sha256"].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Link4 input geometry changed: {name}")
    mesh_root = project / "assets/marvin_urdf/meshes/collision"
    meshes = {side: trimesh.load_mesh(mesh_root / f"gripL_Link4_{side}.stl", process=False) for side in "LR"}
    attachment = np.asarray(record["regression"]["attachment_spheres_m"])
    generation = record["generation"]
    right = fit_template(meshes["R"], attachment, generation)
    transform = np.asarray(record["left_from_right_link"])
    left = right.copy()
    left[:, :3] = right[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    radii = right[:, 3].copy()
    # The mirrored CAD exports differ by micrometers. Increase only radii
    # needed to contain both complete meshes, preserving the shared template.
    for side, spheres in (("L", left), ("R", right)):
        vertices, faces = trimesh.remesh.subdivide_to_size(
            meshes[side].vertices, meshes[side].faces,
            max_edge=generation["mirror_mesh_refinement_max_edge_m"])
        distances = np.linalg.norm(vertices[faces][:, :, None] - spheres[None, None, :, :3], axis=-1).max(axis=1)
        nearest = np.argmin(distances - spheres[None, :, 3], axis=1)
        for index in range(len(spheres)):
            assigned = distances[nearest == index, index]
            if len(assigned):
                radii[index] = max(radii[index], assigned.max() + 1e-7)
    radii += record["generation"]["asset_numeric_guard_m"]
    right[:, 3] = radii
    validate_final_template(right, attachment, generation)
    candidate = copy.deepcopy(source)
    for hand in "LR":
        for side, spheres in (("L", left), ("R", right)):
            candidate["robot_cfg"]["kinematics"]["collision_spheres"][f"grip{hand}_Link4_{side}"] = [
                {"center": center.tolist(), "radius": float(radius)} for center, radius in zip(spheres[:, :3], radii)
            ]
    if sum(map(len, candidate["robot_cfg"]["kinematics"]["collision_spheres"].values())) > 350:
        raise ValueError("Final collision model exceeds the 350-sphere budget")
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "resources/curobo/marvin_gui_dual.yml")
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output file")
    candidate = build_candidate(yaml.safe_load(args.source.read_bytes()), json.loads(args.record.read_text()))
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        yaml.safe_dump(candidate, output, sort_keys=False)
    print(f"Reproduced Link4 triangle coverage candidate: {args.output}")


if __name__ == "__main__":
    main()
