"""Observe exact A15 native planning and unchanged hard checks; no physical motion."""

import json
import time

import numpy as np
import torch
import yaml
from tianji_curobo.contracts import CuroboToolError
from tianji_curobo.curobo_service import materialize_urdf

from jev_mujoco.carrying import CarryingBinding
from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation

folder = ROOT / "data/visible-cloud-work/grasp-place-left-15/gpu-requests"
mapping = json.loads((folder / "020-perception-build.json").read_text())
payload = json.loads((folder / "021-tools-plan_place.json").read_text())["body"]
urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
    marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
report = {"physical_motion": False, "oracle_used": False, "exact_requests": ["020", "021"],
          "validations": [], "world_queries": []}
context = {"phase": "held", "joints_rad": None}
try:
    scene = run_operation(runtime, mapping["path"], mapping["body"])
    assert scene["scene_token"] == payload["scene_token"]
    report["prepare"] = run_operation(runtime, "/carrying/prepare", payload)
    binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm="A")
    with runtime.carrying_scope(binding, payload["gripper_joint_positions_rad"]):
        backend = next(iter(runtime._compound_backends.values()))
        planner, checker = backend.motion_planner, backend._motion_clearance_checker
        robot = yaml.safe_load(runtime._robot_config_path.read_bytes())["robot_cfg"]["kinematics"]
        labels = {}
        for link in [*robot["collision_link_names"], "jev_carried_object_A"]:
            try:
                ids = checker.kinematics.kinematics_config.get_sphere_index_from_link_name(link).detach().cpu().numpy().reshape(-1)
            except KeyError:
                continue
            labels.update({int(index): link for index in ids})
        original_kinematics = checker.get_kinematics
        original_world = checker.get_collision_distance
        original_validate = backend.validate_trajectory_native
        original_grasp = planner.plan_grasp

        def observed_kinematics(q, *args, **kwargs):
            context["joints_rad"] = q.detach().cpu().numpy().tolist()
            return original_kinematics(q, *args, **kwargs)

        def observed_world(x, *args, **kwargs):
            result = original_world(x, *args, **kwargs)
            costs = result.detach().cpu().numpy()
            state_input = hasattr(x, "robot_spheres")
            spheres = x.robot_spheres if state_input else x
            spheres = spheres.detach().cpu().numpy()
            positive = []
            for batch, sample, index in np.argwhere(costs > 1e-9):
                positive.append({"sample": int(sample), "index": int(index), "link": labels.get(int(index)),
                    "sphere_world": spheres[batch, sample, index].tolist(), "cost": float(costs[batch, sample, index])})
            query = {"phase": context["phase"], "state_input": state_input, "cost_shape": list(costs.shape),
                "positive": positive, "world_costs": costs.tolist(), "world_spheres": spheres.tolist(),
                "joints_rad": context["joints_rad"] if state_input else None}
            report["world_queries"].append(query)
            print("WORLD " + json.dumps({k: query[k] for k in ("phase", "state_input", "cost_shape", "positive")}), flush=True)
            return result

        def observed_validate(request):
            phase = ("approach", "place", "retreat")[len(report["validations"])]
            context["phase"] = phase
            entry = {"phase": phase, "native_positions_deg": request.points_deg}
            report["validations"].append(entry)
            try:
                entry["checked_point_count"] = original_validate(request)
                entry["passed"] = True
                return entry["checked_point_count"]
            except CuroboToolError as error:
                entry.update(passed=False, code=error.code, details=error.details)
                raise

        def observed_grasp(*args, **kwargs):
            result = original_grasp(*args, **kwargs)
            if result is not None:
                report["native_public_result"] = {"status": str(result.status)}
                for phase in ("approach", "grasp", "lift"):
                    flag = getattr(result, phase + "_success")
                    report["native_public_result"][phase + "_success"] = (
                        flag.detach().cpu().numpy().tolist() if hasattr(flag, "detach") else bool(flag))
                    trajectory = getattr(result, phase + "_interpolated_trajectory")
                    if trajectory is not None:
                        report["native_public_result"][phase + "_positions_rad"] = trajectory.position.detach().cpu().numpy().tolist()
            return result

        checker.get_kinematics = observed_kinematics
        checker.get_collision_distance = observed_world
        backend.validate_trajectory_native = observed_validate
        planner.plan_grasp = observed_grasp
        q = torch.as_tensor(np.deg2rad(payload["start_joints_deg"])[None, None], **planner.device_cfg.as_torch_dict())
        state = checker.get_kinematics(q)
        world = checker.get_collision_distance(state).detach().cpu().numpy()
        self_cost = checker.get_self_collision_distance(state.robot_spheres).detach().cpu().numpy()
        report["held"] = {"world_positive_count": int(np.sum(world > 1e-9)),
                          "self_positive_count": int(np.sum(self_cost > 1e-9))}
        print("HELD " + json.dumps(report["held"]), flush=True)
    context["phase"] = "release_endpoint_precheck"
    started = time.perf_counter()
    try:
        report["plan"] = run_operation(runtime, "/tools/plan_place", payload)
        report["native_request_passed"] = True
    except (CuroboToolError, ValueError, RuntimeError) as error:
        report["native_request_passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error),
            "code": getattr(error, "code", None), "details": getattr(error, "details", None)}
    report["native_wall_s"] = time.perf_counter() - started
    print("FINAL " + json.dumps(report), flush=True)
finally:
    runtime.clear_carrying()
