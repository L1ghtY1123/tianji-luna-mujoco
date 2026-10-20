"""Fit bounded static candidates with public cuRobo final outputs only.

This offline tool writes a new directory. It never changes the running model.
Thin parts are rigidly rotated for voxel initialization, not thickened or scaled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
import yaml

SDK_COMMIT = "8e734f3ced1df898990bcd92de40abce475907db"
LOCK_PATH = "/tmp/jev-sphere-fit-guangyu.lock"


def as_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def final_spheres(result, fit_from_link, budget):
    """Invert a rigid fit frame; preserve every final sphere and its radius."""
    transform = np.asarray(fit_from_link, dtype=float)
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("Fit frame must be a rigid 4 by 4 transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-10
    ):
        raise ValueError("Fit frame must be rigid")
    if result.debug_info.get("fallback_used"):
        raise ValueError("Official MORPHIT fallback requires review; no candidate is emitted")
    centers, radii = as_array(result.centers), as_array(result.radii).reshape(-1)
    if len(centers) != len(radii) or not 0 < len(radii) <= budget:
        raise ValueError("Final sphere count violates the explicit budget")
    if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("Official final output contains invalid spheres")
    centers = (centers - transform[:3, 3]) @ rotation
    return [{"center": c.tolist(), "radius": float(r)} for c, r in zip(centers, radii)]


def retained_column(source_spheres, archive):
    """Find frozen public-final column balls by their values, never an assumed row order."""
    by_name = {call["label"]: call for call in archive["calls"]}
    spheres, indices = [], []
    for index in range(16):
        sphere = by_name[f"column_segment_{index}"]["spheres"][0]
        matches = [i for i, value in enumerate(source_spheres) if value == sphere]
        if len(matches) != 1:
            raise ValueError("Frozen column sphere must occur exactly once in the source model")
        spheres.append(deepcopy(sphere))
        indices.append(matches[0])
    return spheres, indices


def load_archived_body_parts(project, archive_path):
    """Replay the reviewed 30-cell final outputs, with their input and asset hashes."""
    archive = json.loads(archive_path.read_text())
    if archive["sdk_commit"] != SDK_COMMIT:
        raise ValueError("Archived body uses a different SDK commit")
    lock = archive["geometry_input_lock"]
    for source in [lock["urdf"], *lock["visual_assets"]]:
        if hashlib.sha256((project / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Archived body source geometry hash changed")
    input_path = archive_path.parent / lock["fit_inputs"]["path"]
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != lock["fit_inputs"]["sha256"]:
        raise ValueError("Archived body input hash changed")
    calls = [call for call in archive["calls"] if call["label"].startswith("body_cell_")]
    if len(calls) != 30 or len({call["label"] for call in calls}) != 30:
        raise ValueError("Archived body must contain exactly thirty unique physical cells")
    parts = []
    with np.load(input_path) as inputs:
        for call in calls:
            if call["fit_type"] != "MORPHIT" or call["budget"] != 1 or len(call["spheres"]) != 1:
                raise ValueError("Archived cell must contain one public MORPHIT final sphere")
            name = call["label"]
            mesh = trimesh.Trimesh(inputs[name + "_vertices"], inputs[name + "_faces"], process=False)
            result = SimpleNamespace(
                centers=[sphere["center"] for sphere in call["spheres"]],
                radii=[sphere["radius"] for sphere in call["spheres"]],
                debug_info=call["debug_info"],
            )
            record = {
                "link": "Link_Stand", "name": name, "budget": 1,
                "input_geometry": {"scope_id": "Link_Stand/" + name, **call["geometry"]},
                "fit_from_link": np.eye(4).tolist(), "method": "MORPHIT",
                "result_source": "official_final", "generation_mode": "verified_archived_public_final",
                "parameters": {"num_spheres": 1, "iterations": call["iterations"],
                               "coverage_weight": call["coverage_weight"],
                               "protrusion_weight": call["protrusion_weight"], "seed": call["seed"]},
                "debug_info": call["debug_info"],
                "final_centers_fit_frame_m": result.centers, "final_radii_m": result.radii,
                "spheres": final_spheres(result, np.eye(4), 1),
                "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "source_archive_call": name,
            }
            parts.append({**record, "mesh": mesh, "fit_mesh": mesh.copy(), "archived_record": record})
    return parts


def part_record(link, name, mesh, budget, geometry, rotate_thin=False):
    geometry = {"scope_id": f"{link}/{name}", **geometry}
    transform = np.eye(4)
    if rotate_thin:
        centered = mesh.vertices - mesh.vertices.mean(axis=0)
        _, directions = np.linalg.eigh(centered.T @ centered)
        transform = trimesh.geometry.align_vectors(directions[:, 0], np.ones(3) / np.sqrt(3))
    transform[:3, 3] = -transform[:3, :3] @ mesh.bounds.mean(axis=0)
    fit_mesh = mesh.copy()
    fit_mesh.apply_transform(transform)
    return {
        "link": link, "name": name, "mesh": mesh, "fit_mesh": fit_mesh,
        "budget": budget, "input_geometry": geometry,
        "fit_from_link": transform.tolist(),
    }


def static_parts(project):
    from fit_collision_model import load_visual_parts

    base_visuals, base_sources = load_visual_parts(project, "Link_Base")
    stand_visuals, stand_sources = load_visual_parts(project, "Link_Stand")
    base = trimesh.util.concatenate(base_visuals).split(only_watertight=False)
    stand = stand_visuals[0].split(only_watertight=False)
    if len(base) != 46 or len(stand) != 20:
        raise ValueError("The audited component inventory changed")
    # Fill only each physical tube/flange's internal bore; retain the actual outer step.
    column_end = base[21].convex_hull.slice_plane([0, 0, .9], [0, 0, 1], cap=True)
    upper = trimesh.boolean.union([column_end, base[28].convex_hull], engine="manifold")
    parts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            sector = upper.slice_plane([0, 0, 0], [sx, 0, 0], cap=True)
            sector = sector.slice_plane([0, 0, 0], [0, sy, 0], cap=True)
            parts.append(part_record(
                "Link_Base", f"column_end_sector_{sx}_{sy}", sector, 12,
                {"components": [21, 28], "column_z_min_m": .9,
                 "quadrant_signs": [sx, sy],
                 "construction": "one closed radial quarter of the exact union of connected single-component outer solids; preserves 82.5/90 mm radial step; fills their bores only"},
            ))
    for i in range(22, 28):
        parts.append(part_record(
            "Link_Base", f"rib_{i}", base[i], 12,
            {"components": [i], "construction": "complete original physical rib, no thickening"},
            rotate_thin=True,
        ))
    parts.extend([
        part_record("Link_Stand", "body", stand[19].convex_hull, 24,
                    {"components": [19], "construction": "single physical body outer hull fills through-bore only"}),
        part_record("Link_Stand", "flange", trimesh.util.concatenate(stand[:19]), 32,
                    {"components": list(range(19)), "construction": "complete original flange assembly including all bolts and locating pins"}, rotate_thin=True),
        part_record("Link_Stand", "camera_mount", stand_visuals[1], 12,
                    {"visual": 1, "construction": "complete original mount visual"}, rotate_thin=True),
        part_record("Link_Stand", "camera_body", trimesh.util.concatenate(stand_visuals[2:]), 12,
                    {"visuals": [2, 3], "construction": "complete camera housing and window; same physical camera"}),
    ])
    return parts, {"Link_Base": base_sources, "Link_Stand": stand_sources}


def fit_part(part, output, seed, coverage, protrusion):
    if "archived_record" in part:
        record = deepcopy(part["archived_record"])
        (output / f'{part["link"]}-{part["name"]}.json').write_text(json.dumps(record, indent=2))
        return record
    import fcntl

    import torch
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.monotonic()
    with open(LOCK_PATH, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with urllib.request.urlopen("http://192.168.3.4:8340/api/state", timeout=10) as response:
                state = json.load(response)
            if state.get("run", {}).get("algorithm_pending", False):
                raise RuntimeError("Live algorithm is pending; retry this offline fit when idle")
            result = fit_spheres_to_mesh(
                part["fit_mesh"], num_spheres=part["budget"], fit_type=SphereFitType.MORPHIT,
                iterations=200, coverage_weight=coverage, protrusion_weight=protrusion,
                compute_metrics=False,
            )
            torch.cuda.synchronize()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    record = {key: value for key, value in part.items() if key not in {"mesh", "fit_mesh"}}
    record.update({
        "method": "MORPHIT", "result_source": "official_final",
        "parameters": {"num_spheres": part["budget"], "iterations": 200,
                       "coverage_weight": coverage, "protrusion_weight": protrusion, "seed": seed},
        "debug_info": result.debug_info, "elapsed_s": time.monotonic() - started,
        "final_centers_fit_frame_m": as_array(result.centers).tolist(),
        "final_radii_m": as_array(result.radii).reshape(-1).tolist(),
    })
    try:
        record["spheres"] = final_spheres(result, part["fit_from_link"], part["budget"])
    except ValueError as error:
        record["rejected"] = str(error)
    (output / f'{part["link"]}-{part["name"]}.json').write_text(json.dumps(record, indent=2))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/opt/jev"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", nargs="+", help="Selected representatives only; no complete YAML emitted")
    parser.add_argument("--plan", type=Path, help="Frozen per-part parameters for reproducing a reviewed candidate")
    parser.add_argument("--body-archive", type=Path,
                        help="Replay exactly the archived 30 body-cell public finals with recorded parameters; replaces the whole-body fit")
    parser.add_argument("--column-archive", type=Path,
                        help="Frozen public-final source for the retained 16-sphere column chain")
    parser.add_argument("--coverage", type=float, default=1000.)
    parser.add_argument("--protrusion", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    output, project = args.output.resolve(), args.project.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh offline output directory")
    if output.is_relative_to(project / "resources") or output.is_relative_to(project / "assets"):
        raise ValueError("Offline output must not overwrite runtime assets")
    source = args.source or project / "resources/curobo/marvin_gui_dual.yml"
    config = yaml.safe_load(source.read_text())
    recipe_path = project / "data/collision-correction-2026-09-30/recipe.json"
    recipe = json.loads(recipe_path.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != recipe["sha256"]:
        raise ValueError("Retained column evidence does not match the supplied source configuration")
    parts, geometry_sources = static_parts(project)
    if args.body_archive:
        parts = [part for part in parts if part["name"] != "body"]
        parts.extend(load_archived_body_parts(project, args.body_archive))
    plan = json.loads(args.plan.read_text()) if args.plan else {}
    if any(name not in {p["name"] for p in parts} for name in plan):
        raise ValueError("Frozen plan contains an unknown physical part")
    for part in parts:
        part["budget"] = plan.get(part["name"], {}).get("num_spheres", part["budget"])
        if not isinstance(part["budget"], int) or part["budget"] <= 0:
            raise ValueError("Every physical part requires a positive integer budget")
    if plan:
        order = {name: index for index, name in enumerate(plan)}
        parts.sort(key=lambda part: order.get(part["name"], len(order)))
    selected = [part for part in parts if not args.parts or part["name"] in args.parts]
    if args.parts and set(args.parts) != {p["name"] for p in selected}:
        raise ValueError("Unknown representative part")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1, "sdk_commit": SDK_COMMIT, "candidate_only": True,
        "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256((project / "assets/marvin_urdf/marvin_with_grippers.urdf").read_bytes()).hexdigest(),
        "geometry_sources": geometry_sources, "sdk_defaults": {"coverage_weight": 1000., "protrusion_weight": 10., "iterations": 200},
        "omissions": "Approved feet and unreachable bottom flange/bolts only; existing arbitrary-joint reachability certificate remains required",
        "links": {},
    }
    inputs = {}
    for part in selected:
        key = part["link"] + "-" + part["name"]
        inputs[key + "-vertices"] = part["fit_mesh"].vertices
        inputs[key + "-faces"] = part["fit_mesh"].faces
        inputs[key + "-original-vertices"] = part["mesh"].vertices
        inputs[key + "-original-faces"] = part["mesh"].faces
    np.savez_compressed(output / "fit-inputs.npz", **inputs)
    manifest["input_archive_sha256"] = hashlib.sha256((output / "fit-inputs.npz").read_bytes()).hexdigest()
    if args.prepare_only:
        manifest["prepared_parts"] = [{k: v for k, v in p.items() if k not in {"mesh", "fit_mesh"}} for p in selected]
    else:
        from curobo.logging import setup_logger
        setup_logger("warn")
        for index, part in enumerate(selected):
            parameters = plan.get(part["name"], {})
            record = fit_part(
                part, output, parameters.get("seed", args.seed + index),
                parameters.get("coverage_weight", args.coverage),
                parameters.get("protrusion_weight", args.protrusion),
            )
            manifest["links"].setdefault(part["link"], {"parts": [], "spheres": []})["parts"].append(record)
            manifest["links"][part["link"]]["spheres"].extend(record.get("spheres", []))
            print(json.dumps({"part": part["name"], "count": len(record.get("spheres", [])), "debug": record["debug_info"], "rejected": record.get("rejected")}), flush=True)
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if not args.parts and not any(p.get("rejected") for r in manifest["links"].values() for p in r["parts"]):
            column_archive = args.column_archive or project / "data/collision-correction-2026-09-30/static-fit.json"
            static_archive = json.loads(column_archive.read_text())
            column, column_indices = retained_column(
                config["robot_cfg"]["kinematics"]["collision_spheres"]["Link_Base"], static_archive
            )
            manifest["links"]["Link_Base"]["retained_source"] = {
                "indices": column_indices, "spheres": deepcopy(column),
                "scope_id": "Link_Base/approved_regular_column_chain",
                "reason": "Complete 16-sphere congruent axial-template public-final column chain; the new connected end assembly overlaps its junction",
            }
            manifest["links"]["Link_Base"]["spheres"] = deepcopy(column) + manifest["links"]["Link_Base"]["spheres"]
            candidate = deepcopy(config)
            for name, record in manifest["links"].items():
                candidate["robot_cfg"]["kinematics"]["collision_spheres"][name] = record["spheres"]
            (output / "candidate.yml").write_text(yaml.safe_dump(candidate, sort_keys=False))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
