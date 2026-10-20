"""Read-only final-output provenance and independent geometric measurements.

This audits the project manifest, not the optimizer's truthfulness. Frozen outputs
and source files must also be archived by the generator. Passing provenance never
means that a sphere model covers the real robot or is safe for a trajectory.
"""

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import audit_collision_geometry as geometry
import audit_collision_model as collision
import numpy as np
import trimesh
import yaml

SDK_COMMIT = "8e734f3ced1df898990bcd92de40abce475907db"


def scopes_overlap(first, second):
    return (
        first == "whole_link"
        or second == "whole_link"
        or first == second
        or first.startswith(second + "/")
        or second.startswith(first + "/")
    )


def sphere_counts(spheres):
    return Counter((*sphere["center"], sphere["radius"]) for sphere in spheres)


def check_final_transform(part):
    """Replay only a rigid frame change; sphere radii are never adjusted here."""
    inputs = part.get("input_geometry", {})
    rotation = inputs.get("link_to_fit_rotation")
    transform = part.get("fit_from_link")
    if rotation is None and transform is None:
        return []
    if rotation is not None and transform is not None:
        return ["ambiguous input frame transformations"]
    if transform is None:
        matrix = np.eye(4)
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            return ["input transform is not a rigid rotation"]
        matrix[:3, :3] = rotation
        original = part.get("official_final_in_fit_frame")
        if original is None:
            return ["rotated input requires the original official final"]
        centers = np.asarray([sphere["center"] for sphere in original], dtype=float)
        radii = np.asarray([sphere["radius"] for sphere in original], dtype=float)
    else:
        matrix = np.asarray(transform, dtype=float)
        centers = np.asarray(part.get("final_centers_fit_frame_m", []), dtype=float)
        radii = np.asarray(part.get("final_radii_m", []), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return ["input transform is not a finite 4-by-4 rigid matrix"]
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-10)
        or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-10)
    ):
        return ["input transform is not a proper rigid rotation and translation"]
    spheres = part.get("spheres", [])
    if centers.shape != (len(spheres), 3) or radii.shape != (len(spheres),):
        return ["transformed input requires complete original official final arrays"]
    expected = (centers - matrix[:3, 3]) @ rotation
    recorded = np.asarray([sphere["center"] for sphere in spheres], dtype=float)
    recorded_radii = np.asarray([sphere["radius"] for sphere in spheres], dtype=float)
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(radii).all()
        or not np.array_equal(radii, recorded_radii)
        or not np.allclose(expected, recorded, rtol=0.0, atol=1e-12)
    ):
        return ["mapped sphere differs from the official final"]
    return []


def check_link_sources(candidate, source, record):
    """Check exact assembly of disjoint declared scopes; scopes need human review."""
    errors, assembled, scopes = [], [], []
    for part in record.get("parts", []):
        inputs = part.get("input_geometry", {})
        scope = inputs.get("scope_id")
        if not scope:
            errors.append("missing geometry scope_id")
        else:
            scopes.append(scope)
        if part.get("result_source") != "official_final":
            errors.append("result must be an official final, not optimizer history")
        if part.get("method") not in ("MORPHIT", "SURFACE"):
            errors.append("unsupported fitting method")
        if inputs.get("selection") in ("gap_patch", "uncovered_faces") or part.get("patch_stage"):
            errors.append("gap patches are not a replacement fit of a physical part")
        spheres = part.get("spheres", [])
        budget = part.get("parameters", {}).get("num_spheres")
        if not isinstance(budget, int) or budget < len(spheres) or budget < 1:
            errors.append("missing or exceeded explicit sphere budget")
        errors.extend(check_final_transform(part))
        assembled.extend(spheres)
    retained = record.get("retained_source", [])
    retained = [retained] if isinstance(retained, dict) else retained
    for group in retained:
        scope = group.get("scope_id")
        if not scope or not group.get("reason"):
            errors.append("retained source needs a reviewed scope and reason")
        else:
            scopes.append(scope)
        indices = group.get("indices", [])
        if len(indices) != len(set(indices)) or any(
            not isinstance(i, int) or i < 0 or i >= len(source) for i in indices
        ):
            errors.append("invalid retained source indices")
        else:
            assembled.extend(source[index] for index in indices)
    for index, scope in enumerate(scopes):
        if any(scopes_overlap(scope, other) for other in scopes[index + 1 :]):
            errors.append(f"overlapping geometry scopes: {scope}")
    if sphere_counts(assembled) != sphere_counts(candidate) or record.get("spheres", candidate) != candidate:
        errors.append("candidate contains edited, missing or unexplained spheres")
    return errors


def audit_provenance(candidate, source, manifest):
    errors = []
    before, after = copy.deepcopy(source), copy.deepcopy(candidate)
    original = before["robot_cfg"]["kinematics"].pop("collision_spheres")
    current = after["robot_cfg"]["kinematics"].pop("collision_spheres")
    if before != after:
        errors.append("non-sphere configuration changed")
    if manifest.get("schema_version") != 1:
        errors.append("unsupported manifest schema")
    if manifest.get("sdk_commit") != SDK_COMMIT:
        errors.append("manifest does not identify the pinned SDK")
    if manifest.get("history_used") or manifest.get("patch_stage"):
        errors.append("history accumulation or patch stage is not the approved workflow")
    records = manifest.get("links", {})
    for link, spheres in current.items():
        if link not in records:
            if spheres != original.get(link):
                errors.append(f"{link}: changed spheres lack a provenance record")
            continue
        errors.extend(
            f"{link}: {error}" for error in check_link_sources(spheres, original.get(link, []), records[link])
        )
    for link in set(records).difference(current):
        errors.append(f"{link}: manifest link absent from configuration")
    return {
        "procedural_verified": not errors,
        "geometry_verified": False,
        "errors": errors,
        "limitations": [
            "Declared physical scopes require review against the original mesh.",
            "This verifies recorded arrays, not a rerun or independent authentication of the SDK.",
            "Unlisted links are permitted only when their source spheres are exactly unchanged.",
        ],
    }


def surface_gap_report(mesh, points, visible, spheres):
    """Keep full geometry and unproven samples; a blocked normal is not an internal face."""
    result = {"geometry_verified": False, "visibility_unproven_samples": int((~visible).sum())}
    for name, values in (
        ("all_vertices", mesh.vertices),
        ("all_area_samples", points),
        ("certified_visible_samples", points[visible]),
        ("visibility_unproven", points[~visible]),
    ):
        gaps = collision.sphere_gaps(values, spheres)
        result[name] = geometry.distance_summary(gaps, values)
        result[name]["covered_fraction"] = float(np.mean(gaps <= geometry.TOLERANCE_M)) if len(gaps) else None
    return result


def measure_geometry(config, urdf, links, poses_path, surface_samples, sphere_samples, seed):
    local_meshes, assets = collision.load_visual_meshes(urdf)
    local_meshes["Link_Base"], classification = collision.retain_central_base(local_meshes["Link_Base"])
    robot = ET.parse(urdf).getroot()
    poses, table = collision.read_poses(poses_path, config)
    records = []
    for pose in poses:
        transforms = collision.forward_kinematics(robot, config, pose["joint_positions_rad"])
        meshes, modes = {}, {name: {} for name in ("raw", "world_buffered", "self_effective")}
        for link, mesh in local_meshes.items():
            meshes[link] = mesh.copy()
            meshes[link].apply_transform(transforms[link])
        for link in config["collision_link_names"]:
            transform = transforms[link]
            for mode, values in collision.sphere_modes(config, link).items():
                values[:, :3] = values[:, :3] @ transform[:3, :3].T + transform[:3, 3]
                modes[mode][link] = values
        parts, query_meshes = geometry.solid_components(meshes), {}
        measurements = {}
        for link in links:
            print(f"Auditing {pose['name']}: {link}", flush=True)
            mesh = meshes[link]
            points, faces = trimesh.sample.sample_surface(mesh, surface_samples, seed=seed)
            visible = geometry.visible_surface_points(meshes, points, mesh.face_normals[faces], query_meshes)
            # Additional clear directions certify more surfaces; no failure is labelled internal.
            for direction in np.vstack([np.eye(3), -np.eye(3)]):
                remaining = np.flatnonzero(~visible)
                if not len(remaining):
                    break
                directions = np.broadcast_to(direction, (len(remaining), 3))
                origins = points[remaining] + geometry.RAY_OFFSET_M * directions
                visible[remaining] = geometry.clear_rays(meshes, origins, directions, query_meshes)
            coverage = {}
            for mode, by_link in modes.items():
                coverage[mode] = surface_gap_report(
                    mesh, points, visible, np.concatenate(list(by_link.values()))
                )
                coverage[mode]["owning_link_only"] = surface_gap_report(mesh, points, visible, by_link[link])
            measurements[link] = {
                "sphere_count": len(modes["raw"][link]),
                "coverage": coverage,
                "raw_union_boundary": geometry.measure_sphere_boundaries(
                    link, meshes, modes["raw"], parts, sphere_samples, query_meshes
                ),
            }
        records.append(
            {
                **pose,
                **collision.audit_pose(robot, config, pose["joint_positions_rad"], table),
                "links": measurements,
            }
        )
    return {
        "assets": assets,
        "base_component_classification": classification,
        "poses": records,
        "surface_samples": surface_samples,
        "sphere_samples": sphere_samples,
        "seed": seed,
        "distance_units": "mm; witness coordinates in world metres",
        "query_scale": 1000,
        "limitations": [
            "Finite surface and boundary samples do not prove continuous coverage or extrema.",
            "Visible means a tested normal or axis ray escaped the literal mesh; others remain unproven.",
            "Raw exterior distances exclude sphere-union interiors and preserve open-mesh ambiguity.",
            "Geometry, provenance and GPU planning success are separate acceptance evidence.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=collision.DEFAULT_URDF)
    parser.add_argument("--links", nargs="+")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--surface-samples", type=int, default=4000)
    parser.add_argument("--sphere-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--provenance-only", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() in {
        p.resolve() for p in (args.config, args.source, args.manifest, args.urdf, args.poses) if p
    }:
        parser.error("output must not replace an input")
    if args.surface_samples < 1 or args.sphere_samples < 8:
        parser.error("surface samples must be positive; sphere samples at least eight")
    candidate = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance = audit_provenance(candidate, source, manifest)
    if manifest.get("source_config_sha256") != hashlib.sha256(args.source.read_bytes()).hexdigest():
        provenance["errors"].append("source configuration SHA does not match")
        provenance["procedural_verified"] = False
    report = {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "provenance": provenance}
    if not args.provenance_only:
        config = candidate["robot_cfg"]["kinematics"]
        report["geometry"] = measure_geometry(
            config,
            args.urdf,
            args.links or config["collision_link_names"],
            args.poses,
            args.surface_samples,
            args.sphere_samples,
            args.seed,
        )
        if args.output.resolve() in {Path(asset["path"]).resolve() for asset in report["geometry"]["assets"]}:
            parser.error("output must not replace a source mesh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    raise SystemExit(0 if provenance["procedural_verified"] else 1)


if __name__ == "__main__":
    main()
