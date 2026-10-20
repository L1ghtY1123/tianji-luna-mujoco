"""One isolated recorded-map world query; no product or request modifications."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tianji_curobo.curobo_service import materialize_urdf

from jev_mujoco.carrying import CarryingBinding
from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity
from jev_mujoco.planning_settings import PlanningSettings

parser = argparse.ArgumentParser()
parser.add_argument("--pack", default="/tmp/b7-opening-world-pack.npz")
parser.add_argument("--output", default="/tmp/b7-opening-world-gpu-probe.json")
options = parser.parse_args()
folder = ROOT / "data/visible-cloud-acceptance/grasp-place-right-7/gpu-requests"
files = [folder / "020-perception-build.json", folder / "026-carrying-validate.json"]
mapping, request = [json.loads(path.read_text()) for path in files]
payload = request["body"]
pack = np.load(options.pack, allow_pickle=False)
spheres = pack["spheres"].copy()
metadata = json.loads(str(pack["metadata"]))
urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
    marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
report = {"sdk": sdk_identity()[0], "metadata": metadata,
    "request_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
try:
    scene = run_operation(runtime, mapping["path"], mapping["body"])
    assert scene["scene_token"] == payload["scene_token"]
    settings = PlanningSettings.model_validate(payload["planning_settings"])
    runtime.configure_planning(settings)
    binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm=payload["arm"])
    with runtime.carrying_scope(binding, payload["gripper_joint_positions_rad"]), runtime.gpu_lock:
        snapshot = runtime._prepare_scene_cache_unlocked()
        backend = runtime._compound_backend_unlocked("B", settings.trajectory_collision_clearance_m,
            settings.optimizer_collision_activation_distance_m)
        runtime._refresh_scene_unlocked(arm="B", backend=backend, snapshot=snapshot)
        runtime._refresh_peer_collision_unlocked(arm="B", backend=backend,
            active_joints_deg=np.rad2deg(payload["positions_rad"][0]).tolist(), other_joints_deg=payload["other_joints_deg"])
        checker = backend._motion_clearance_checker
        kwargs = backend.motion_planner.device_cfg.as_torch_dict()
        state = checker.get_kinematics(torch.as_tensor(np.array(payload["positions_rad"])[None], **kwargs))
        native = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
        native_indices = {}
        for label in metadata["labels"]:
            link = label["link"]
            if link not in native_indices:
                native_indices[link] = checker.kinematics.kinematics_config.get_sphere_index_from_link_name(
                    link).detach().cpu().numpy().reshape(-1)
        order = [int(native_indices[label["link"]][label["link_sphere_index"]]) for label in metadata["labels"]]
        delta = np.abs(spheres[0] - native[order])
        report["cpu_native_start_max_component_delta_m"] = float(delta.max())
        assert float(delta.max()) < 5e-6, "CPU original world spheres must match actual native geometry"
        tensor = torch.as_tensor(spheres[None], **kwargs)
        costs = checker.get_collision_distance(tensor).detach().cpu().numpy().copy()[0]
        report["scene_token"] = scene["scene_token"]
        report["hard_world_activation_distance_m"] = settings.trajectory_collision_clearance_m
        report["world_layer_names"] = list(runtime.geometry_model.get("voxel", {}))
        report["queries"] = []
        for i, group in enumerate(metadata["groups"]):
            positive = np.flatnonzero(costs[i] > 0)
            row = {**group, "maximum_cost": float(costs[i].max()), "positive_count": len(positive),
                "world_costs": costs[i].tolist(), "world_spheres": spheres[i].tolist(),
                "positive_spheres": [{"cpu_index": int(k), "native_sphere_index": order[k],
                    **metadata["labels"][k], "cost": float(costs[i, k]), "sphere_world": spheres[i, k].tolist()}
                    for k in positive]}
            report["queries"].append(row)
        runtime._require_scene_token_unchanged_unlocked(scene["scene_token"])
        report["completed"] = True
        print("COUNTS " + json.dumps([{k: row[k] for k in ("kind", "positive_count", "maximum_cost")}
            for row in report["queries"]]), flush=True)
except Exception as error:
    report["completed"] = False
    report["error"] = {"message": str(error), "type": type(error).__name__}
    raise
finally:
    Path(options.output).write_text(json.dumps(report, indent=2))
    runtime.clear_carrying()
    print("GPU_DIAGNOSTIC_FINISHED " + str(report.get("completed")), flush=True)
