"""Replay exact physical13 requests; observe public results without moving hardware."""

import json
import sys
import time

import numpy as np
import torch
import yaml
from tianji_curobo.contracts import CuroboToolError
from tianji_curobo.curobo_service import materialize_urdf

from luna_mujoco.carrying import CarryingBinding
from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation


def summary(value):
    if hasattr(value, "detach"):
        a = value.detach().cpu().numpy()
        return {"shape": list(a.shape), "values": a.tolist() if a.size <= 20 else None,
                "min": float(a.min()) if a.size else None, "max": float(a.max()) if a.size else None}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


folder = ROOT / "data/visible-cloud-work/grasp-place-left-13"
mapping = json.loads((folder / "gpu-requests/022-perception-build.json").read_text())
request = json.loads((folder / "gpu-requests/023-tools-plan_place.json").read_text())
payload = request["body"]
urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
    marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
report = {"physical_motion": False, "oracle_used": False, "exact_requests": ["022", "023"]}
try:
    scene = run_operation(runtime, mapping["path"], mapping["body"])
    assert scene["scene_token"] == payload["scene_token"]
    report["prepare"] = run_operation(runtime, "/carrying/prepare", payload)
    binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm="A")
    with runtime.carrying_scope(binding, payload["gripper_joint_positions_rad"]):
        backend = next(iter(runtime._compound_backends.values()))
        planner, checker = backend.motion_planner, backend._motion_clearance_checker
        positions = torch.as_tensor(np.deg2rad(payload["start_joints_deg"])[None, None],
                                    **planner.device_cfg.as_torch_dict())
        state = checker.get_kinematics(positions)
        spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
        world = checker.get_collision_distance(state).detach().cpu().numpy().reshape(-1)
        self_cost = checker.get_self_collision_distance(state.robot_spheres).detach().cpu().numpy()
        config = checker.kinematics.kinematics_config
        robot = yaml.safe_load(runtime._robot_config_path.read_bytes())["robot_cfg"]["kinematics"]
        labels = {}
        for link in robot["collision_link_names"]:
            try:
                indices = config.get_sphere_index_from_link_name(link).detach().cpu().numpy().reshape(-1)
            except KeyError:
                continue
            labels.update({int(index): link for index in indices})
        report["held"] = {"world_positive_count": int(np.sum(world > 0)),
            "self_positive_count": int(np.sum(self_cost > 0)), "self_max_cost": float(self_cost.max()),
            "sphere_count": len(spheres),
            "positive": [{"index": int(index), "link": labels.get(int(index)),
                          "sphere_world": spheres[index].tolist(), "cost": float(world[index])}
                         for index in np.flatnonzero(world > 0)]}
        print("HELD " + json.dumps(report["held"]), flush=True)
        if "--held-only" in sys.argv:
            self_config = checker.kinematics.get_self_collision_config()
            report["self_config"] = {k: summary(v) for k, v in vars(self_config).items() if not k.startswith("_")}
            pairs = self_config.collision_pairs.detach().cpu().numpy().astype(int)
            padding = self_config.sphere_padding.detach().cpu().numpy()
            pair_gaps = np.linalg.norm(spheres[pairs[:, 0], :3] - spheres[pairs[:, 1], :3], axis=-1)
            pair_gaps -= spheres[pairs[:, 0], 3] + spheres[pairs[:, 1], 3] + padding[pairs[:, 0]] + padding[pairs[:, 1]]
            report["negative_self_pairs"] = [{"indices": pairs[i].tolist(), "links": [labels.get(int(k)) for k in pairs[i]],
                "sphere_world": spheres[pairs[i]].tolist(), "sphere_padding": padding[pairs[i]].tolist(),
                "self_surface_gap_m": float(pair_gaps[i])} for i in np.flatnonzero(pair_gaps < 0)]
            print("SELF_PAIRS " + json.dumps(report["negative_self_pairs"]), flush=True)
            report["sphere_labels"] = labels
            report["world_spheres"] = spheres.tolist()
            attached = config.get_sphere_index_from_link_name("jev_carried_object_A").detach().cpu().numpy().reshape(-1)
            body = np.arange(int(attached.min()))
            gaps = np.linalg.norm(spheres[attached, None, :3] - spheres[None, body, :3], axis=-1)
            gaps -= spheres[attached, None, 3] + spheres[None, body, 3]
            candidates = [(int(attached[a]), int(body[b]), float(gaps[a, b]))
                          for a, b in np.argwhere(gaps < .01)
                          if labels.get(int(body[b])) not in ("gripL_Link3_L", "gripL_Link3_R")]
            report["nonpad_near_pairs"] = sorted(candidates, key=lambda x: x[2])[:20]
            print("PAIR " + json.dumps({"near": report["nonpad_near_pairs"], "self_config": report["self_config"]}), flush=True)
            print("RESULT " + json.dumps(report), flush=True)
            sys.exit(0)
        report["pose_calls"] = []
        original = planner.plan_pose

        def observed_plan_pose(*args, **kwargs):
            started = time.perf_counter()
            result = original(*args, **kwargs)
            entry = {"wall_s": time.perf_counter() - started, "returned_none": result is None}
            if result is not None:
                for name in ("success", "status", "valid_query", "ik_success", "graph_success",
                             "trajopt_success", "planning_time", "position_error", "rotation_error"):
                    if hasattr(result, name):
                        entry[name] = summary(getattr(result, name))
            report["pose_calls"].append(entry)
            print("POSE " + json.dumps(entry), flush=True)
            return result

        planner.plan_pose = observed_plan_pose
    started = time.perf_counter()
    try:
        plan = run_operation(runtime, request["path"], payload)
        report["native_success"] = True
        report["native_phase_lengths"] = {k: len(v["positions_rad"]) for k, v in plan["phases"].items()}
    except (CuroboToolError, RuntimeError, ValueError) as error:
        report["native_success"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error),
            "code": getattr(error, "code", None), "stage": getattr(error, "stage", None),
            "details": getattr(error, "details", None)}
    report["native_wall_s"] = time.perf_counter() - started
    print("RESULT " + json.dumps(report), flush=True)
finally:
    runtime.clear_carrying()
