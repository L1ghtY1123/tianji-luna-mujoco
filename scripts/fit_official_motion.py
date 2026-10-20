"""Offline, bounded official sphere fits; writes candidates, never runtime assets.

The plan lists links and explicit per-part budgets. A part may select complete URDF
visuals; omitted visual_indices means the entire link, matching RobotBuilder.
Only final public results are retained. No history union or patch stage is used.
"""

import argparse
import dataclasses
import hashlib
import json
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh
import yaml
from fit_collision_model import SDK_COMMIT, load_visual_parts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_spheres(result, budget):
    centers = np.asarray(result.centers.tolist(), dtype=float)
    radii = np.asarray(result.radii.tolist(), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or radii.shape != (len(centers),):
        raise ValueError("Official final result has invalid array shapes")
    if not 0 < len(radii) <= budget or result.num_spheres != len(radii):
        raise ValueError("Official final result violates its explicit sphere budget")
    if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("Official final result contains invalid spheres")
    return [{"center": point.tolist(), "radius": float(radius)} for point, radius in zip(centers, radii)]


def check_live_state(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        state = json.load(response)
    run = state.get("run") if isinstance(state, dict) else None
    if not isinstance(run, dict) or type(run.get("algorithm_pending")) is not bool:
        raise RuntimeError("Unrecognised live state; offline fitting stopped")
    if run["algorithm_pending"]:
        raise RuntimeError("Live algorithm request is pending; retry offline fit after it finishes")


def validate_output_path(project, source, output, sdk_root=None):
    if output.exists():
        raise FileExistsError("Output must be a new offline directory")
    if output == project or any(
        output.is_relative_to(project / name) for name in ["resources", "assets", "src", "skills"]
    ):
        raise ValueError("Output may not overwrite project runtime files")
    if source == output / "marvin_gui_dual.yml":
        raise ValueError("Output must not overwrite the input configuration")
    if sdk_root is not None and output.is_relative_to(sdk_root):
        raise ValueError("Output may not write inside the official SDK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-url", default="http://192.168.3.4:8340/api/state")
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    source = args.source_config.resolve()
    validate_output_path(project, source, output)
    plan = json.loads(args.plan.read_text())
    config = yaml.safe_load(source.read_text())
    kinematics = config["robot_cfg"]["kinematics"]
    for link, spec in plan.items():
        if link in {"Link_Base", "Link_Stand"} or link not in kinematics["collision_link_names"]:
            raise ValueError(f"Not a motion collision link: {link}")
        if type(spec.get("num_spheres")) is not int or spec["num_spheres"] < 1:
            raise ValueError("Every link requires a positive integer link budget")
        parts = spec.get("parts", [{"num_spheres": spec["num_spheres"]}])
        if not isinstance(parts, list) or not parts:
            raise ValueError("Every selected link requires at least one fitted part")
        if any(type(part["num_spheres"]) is not int or part["num_spheres"] < 1 for part in parts):
            raise ValueError("Every part requires a positive integer sphere budget")
        if sum(part["num_spheres"] for part in parts) > spec["num_spheres"]:
            raise ValueError("Part budgets exceed the explicit per-link total")

    import fcntl

    import curobo
    import torch
    from curobo.logging import setup_logger
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

    validate_output_path(project, source, output, Path(curobo.__file__).resolve().parent)
    setup_logger("warn")
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    (output / "generator.py").write_bytes(Path(__file__).read_bytes())
    manifest = {
        "schema_version": 1,
        "sdk_commit": SDK_COMMIT,
        "source_config_sha256": sha256(source),
        "generator_sha256": sha256(__file__),
        "api": "curobo.sphere_fit.fit_spheres_to_mesh",
        "candidate_only": True,
        "geometry_reviewed": False,
        "history_used": False,
        "patch_stage": False,
        "seed": args.seed,
        "links": {},
    }
    candidate = deepcopy(config)
    for link, spec in plan.items():
        meshes, sources = load_visual_parts(project, link)
        # This preparation and visual preference follow the pinned RobotBuilder.
        for mesh in meshes:
            mesh.fill_holes()
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_winding(mesh)
        record = {
            "sphere_budget": spec["num_spheres"],
            "spheres": [],
            "parts": [],
            "geometry_sources": sources,
        }
        parts = spec.get("parts", [{"num_spheres": spec["num_spheres"]}])
        for index, part in enumerate(parts):
            indices = part.get("visual_indices", list(range(len(meshes))))
            mesh = trimesh.util.concatenate([meshes[i] for i in indices])
            if "component_indices" in part:
                components = mesh.split(only_watertight=False, repair=False)
                mesh = trimesh.util.concatenate([components[i] for i in part["component_indices"]])
            for cut in part.get("cuts", []):
                normal = np.zeros(3)
                normal[cut["axis"]] = cut["sign"]
                origin = np.zeros(3)
                origin[cut["axis"]] = cut["offset"]
                mesh = mesh.slice_plane(origin, normal, cap=True)
            if part.get("convex_hull", False):
                mesh = mesh.convex_hull
            if not len(mesh.faces):
                raise ValueError(f"{link}: selected part {index} contains no geometry")
            rotation = trimesh.transformations.euler_matrix(
                *np.deg2rad(part.get("fit_frame_rpy_deg", [0, 0, 0])), axes="sxyz"
            )
            mesh.apply_transform(rotation)
            input_path = output / f"{link}-part{index}.ply"
            mesh.export(input_path)
            parameters = {"num_spheres": part["num_spheres"], "iterations": 200, "compute_metrics": True}
            for key in ["iterations", "coverage_weight", "protrusion_weight", "surface_radius", "clip_plane"]:
                if key in spec:
                    parameters[key] = spec[key]
                if key in part:
                    parameters[key] = part[key]
            method = part.get("method", spec.get("method", "MORPHIT"))
            with open("/tmp/jev-sphere-fit-guangyu.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                check_live_state(args.runtime_url)
                free, _ = torch.cuda.mem_get_info()
                if free < 4 * 1024**3:
                    raise RuntimeError("Less than 4 GiB GPU memory is free; offline fit stopped")
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                started = time.monotonic()
                result = fit_spheres_to_mesh(mesh, fit_type=SphereFitType[method], **parameters)
                official_spheres = final_spheres(result, part["num_spheres"])
                spheres = deepcopy(official_spheres)
                for sphere in spheres:
                    sphere["center"] = (rotation[:3, :3].T @ sphere["center"]).tolist()
                elapsed = time.monotonic() - started
            record["spheres"].extend(spheres)
            record["parts"].append(
                {
                    "name": f"visual-part-{index}",
                    "scope_id": f"{link}:part-{index}",
                    "method": method,
                    "input_geometry": {
                        "path": input_path.name,
                        "sha256": sha256(input_path),
                        "visual_indices": indices,
                        "frame": link,
                        "cuts": part.get("cuts", []),
                        "convex_hull": part.get("convex_hull", False),
                        "component_indices": part.get("component_indices"),
                        "link_to_fit_rotation": rotation[:3, :3].tolist(),
                    },
                    "parameters": parameters,
                    "result_source": "official_final",
                    "spheres": spheres,
                    "official_final_in_fit_frame": official_spheres,
                    "debug_info": result.debug_info,
                    "elapsed_s": elapsed,
                    "official_metrics": dataclasses.asdict(result.metrics),
                }
            )
        record["sphere_count"] = len(record["spheres"])
        manifest["links"][link] = record
        candidate["robot_cfg"]["kinematics"]["collision_spheres"][link] = record["spheres"]
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (output / "marvin_gui_dual.yml").write_text(yaml.safe_dump(candidate, sort_keys=False))
        print(
            json.dumps({"link": link, "count": record["sphere_count"], "budget": spec["num_spheres"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
