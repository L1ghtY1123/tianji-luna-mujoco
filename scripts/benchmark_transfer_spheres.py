"""Offline native IK/batch regression in an isolated existing-image container."""

import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import torch
import yaml
from tianji_curobo.curobo_service import materialize_urdf

from luna_mujoco.geometry_service import SDK_COMMIT, GeometryRuntime, run_operation, sdk_identity


def check_idle():
    """Read live state; the isolated process never sends production mutations."""
    with urllib.request.urlopen("http://192.168.3.4:8340/api/state", timeout=10) as response:
        state = json.load(response)
    run = state.get("run", {})
    if (run.get("status") not in ("idle", "succeeded", "failed", "cancelled")
            or run.get("algorithm_pending") is not False
            or state.get("simulation", {}).get("active_commands") != []):
        raise RuntimeError("Live application became active; isolated benchmark stopped")


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--mode", choices=("fixed-plan", "ik-plan"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(args.fixture.read_text())
    arm_fixture = fixture["arms"][args.arm]
    request = arm_fixture["request"]
    config = yaml.safe_load(args.config.read_text())["robot_cfg"]["kinematics"]
    if args.mode == "fixed-plan" and len(request["goal_matrices_world"]) != 16:
        raise ValueError("Fixed planning comparison requires sixteen recorded targets")
    check_idle()
    with (args.output.parent / "gpu-benchmark-guangyu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = sdk_identity()
        if source[0] != SDK_COMMIT:
            raise RuntimeError(f"SDK identity mismatch: {source}")
        free, _ = torch.cuda.mem_get_info()
        if free < 16 * 1024**3:
            raise RuntimeError("Less than 16 GiB free; benchmark not started")
        root = Path("/opt/jev")
        urdf = materialize_urdf(
            root / "assets/marvin_urdf/marvin_with_grippers.urdf",
            marvin_meshes=root / "assets/marvin_urdf/meshes",
            rag_meshes=root / "assets/marvin_urdf/rag100_050/meshes",
        )
        runtime = GeometryRuntime(robot_config_path=args.config, urdf_path=urdf, asset_root_path=urdf.parent)
        started = time.perf_counter()
        runtime.build_geometry(fixture["scene"])
        torch.cuda.synchronize()
        report = {
            "config_sha256": digest(args.config), "fixture_sha256": digest(args.fixture),
            "script_sha256": digest(Path(__file__)), "sdk": source,
            "sphere_count": sum(len(values) for values in config["collision_spheres"].values()),
            "arm": args.arm, "mode": args.mode, "motion_executed": False,
            "candidate_indices": arm_fixture["candidate_indices"], "scores": arm_fixture["scores"],
            "settings": request["planning_settings"], "scene_build_s_excluded": time.perf_counter() - started,
            "timing_scope": "Synchronized run_operation wall time; cold includes backend construction. Imports, scene build, serialization, HTTP, and motion excluded.",
            "limitations": [
                "Shared GPU, finite repetitions, historical scene approximation.",
                "Fixed plan inputs retain historical candidates, but native plan_grasp recomputes current-model IK.",
                "B56 is the old-model reachable subset, not all 200 original GraspGen candidates.",
                "IK-filtered planning subsets can differ between models; those timings are not same-input comparisons.",
            ], "passes": [],
        }

        def execute(label, route, payload):
            check_idle()
            before = gpu_snapshot()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = run_operation(runtime, route, payload)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            result_path = args.output / f"{label}.json"
            result_path.write_text(json.dumps(result))
            row = {
                "label": label, "wall_s": elapsed, "batch_size": len(result["items"]),
                "success_count": sum(item["success"] for item in result["items"]),
                "gpu_before": before, "gpu_after": gpu_snapshot(),
                "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "result": result_path.name, "result_sha256": digest(result_path),
            }
            report["passes"].append(row)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
            return result

        if args.mode == "fixed-plan":
            for label in ("plan_cold", "plan_warm_1", "plan_warm_2"):
                execute(label, "/tools/plan_grasp_batch", request)
        else:
            for label in ("ik_cold", "ik_warm"):
                ik_result = execute(label, "/tools/solve_ik_batch", request)
            selected = [index for index, item in enumerate(ik_result["items"]) if item["success"]]
            selected.sort(key=lambda index: -arm_fixture["scores"][index])
            report["warm_survivors_original_indices"] = [arm_fixture["candidate_indices"][index] for index in selected]
            selected = selected[:16]
            if selected:
                chosen = copy.deepcopy(fixture)
                chosen_arm = chosen["arms"][args.arm]
                chosen_arm["request"]["goal_matrices_world"] = [request["goal_matrices_world"][index] for index in selected]
                chosen_arm["candidate_indices"] = [arm_fixture["candidate_indices"][index] for index in selected]
                chosen_arm["scores"] = [arm_fixture["scores"][index] for index in selected]
                (args.output / "selected-plan-fixture.json").write_text(json.dumps(chosen, indent=2))
                for label in ("plan_cold", "plan_warm"):
                    execute(label, "/tools/plan_grasp_batch", chosen_arm["request"])
        check_idle()
        report["production_idle_at_checks"] = True
        (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
