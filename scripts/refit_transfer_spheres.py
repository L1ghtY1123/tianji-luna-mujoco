"""Fit frozen physical parts using the pinned public MORPHIT final API.

Input: a plan with hashed meshes and explicit budgets. Output: a new offline
directory with final arrays and provenance. Runtime assets are never written.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from fit_official_motion import check_live_state, validate_output_path
from fit_official_static import final_spheres


def load_job_mesh(archive, job):
    archive = Path(archive).resolve()
    path = (archive / job["mesh"]).resolve()
    if not path.is_relative_to(archive):
        raise ValueError("Input mesh escapes the archive")
    if hashlib.sha256(path.read_bytes()).hexdigest() != job["sha256"]:
        raise ValueError("Frozen input geometry hash changed")
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Input must be a nonempty triangle mesh")
    return mesh


def convert_final(result, transform, budget):
    return final_spheres(result, transform, budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("/opt/jev"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-url", default="http://192.168.3.4:8340/api/state")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    meshes = [load_job_mesh(args.plan.parent, job) for job in plan["jobs"]]
    if len({job["name"] for job in plan["jobs"]}) != len(meshes):
        raise ValueError("Job names must be unique")
    for job in plan["jobs"]:
        if type(job["num_spheres"]) is not int or job["num_spheres"] < 1:
            raise ValueError("Every physical part needs a positive integer budget")
    import fcntl

    import curobo
    import torch
    from curobo.logging import setup_logger
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

    project, output = args.project.resolve(), args.output.resolve()
    sdk = Path(curobo.__file__).resolve().parent
    validate_output_path(project, args.plan.resolve(), output, sdk)
    manifest = json.loads((project / "resources/curobo/sdk-source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((sdk / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned SDK source differs: {name}")
    setup_logger("warn")
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    (output / "plan.json").write_bytes(args.plan.read_bytes())
    report = {"sdk_commit": manifest["commit"], "sdk_verified_files": len(manifest["files"]),
              "api": "curobo.sphere_fit.fit_spheres_to_mesh", "method": "MORPHIT",
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "history_used": False, "radius_postprocessing": False, "jobs": {}}
    for job, mesh in zip(plan["jobs"], meshes, strict=True):
        transform = trimesh.transformations.euler_matrix(*np.deg2rad(job.get("fit_rpy_deg", [0, 0, 0])))
        mesh.apply_transform(transform)
        parameters = {"num_spheres": job["num_spheres"], "iterations": 200,
                      "coverage_weight": job["coverage_weight"], "protrusion_weight": job["protrusion_weight"]}
        seed = job.get("seed", 20261012)
        with open("/tmp/jev-sphere-fit-guangyu.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_live_state(args.runtime_url)
            if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
                raise RuntimeError("Less than 16 GiB GPU memory free")
            np.random.seed(seed)
            torch.manual_seed(seed)
            started = time.perf_counter()
            result = fit_spheres_to_mesh(mesh, fit_type=SphereFitType.MORPHIT, **parameters)
            torch.cuda.synchronize()
            spheres = convert_final(result, transform, job["num_spheres"])
        record = {"input": job, "parameters": parameters, "seed": seed,
                  "fit_from_link": transform.tolist(), "debug_info": result.debug_info,
                  "final_centers_fit_frame_m": result.centers.tolist(), "final_radii_m": result.radii.tolist(),
                  "spheres": spheres, "elapsed_s": time.perf_counter() - started}
        report["jobs"][job["name"]] = record
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"job": job["name"], "count": len(spheres), "elapsed_s": record["elapsed_s"]}),
              flush=True)


if __name__ == "__main__":
    main()
