"""Diagnose placement from recorded RGBD/feedback without physical actions."""

import argparse
import json
import time
from copy import deepcopy

from scipy.spatial.transform import Rotation
from tianji_curobo.contracts import CuroboToolError
from tianji_curobo.curobo_service import materialize_urdf

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.carrying import CarryingBinding
from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity
from luna_mujoco.placement import release_goal
from luna_mujoco.trajectory_postprocessing import build_execution_phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroid-control", action="store_true", help="Explicit 40mm control using measured bin visible centroid XY")
    parser.add_argument("--input-dir", default="data/visible-cloud-work/grasp-place-left-8")
    parser.add_argument("--forty-only", action="store_true", help="Explicit authorized 40mm probe only; retain100mm default approach")
    parser.add_argument("--held-check", action="store_true", help="Record actual-held hard world/self validity before planning")
    options = parser.parse_args()
    folder = ROOT / options.input_dir
    stage = json.loads((folder / "stage-1.json").read_text(encoding="utf-8"))
    mapping = json.loads((folder / "gpu-requests/017-perception-build.json").read_text(encoding="utf-8"))
    preparing = json.loads((folder / "gpu-requests/018-carrying-prepare.json").read_text(encoding="utf-8"))
    payload = preparing["body"]
    holding = stage["holding"]
    holding["points_tool_m"] = payload["carrying_binding"]["points_tool_m"]
    observation = holding["observation"]
    target = next(obj for obj in observation["objects"] if obj["id"] == holding["instance_id"])
    destination = next(obj for obj in observation["objects"] if obj["kind"] == "bin" and obj["color"] == "blue")
    if options.centroid_control:
        destination = deepcopy(destination)
        destination["placement_reference"]["position_world_m"][:2] = destination["visible_centroid_world"][:2]
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    report = {"sdk": sdk_identity()[0], "physical_motion": False, "oracle_used": False,
              "centroid_xy_control": options.centroid_control,
              "identity_assumptions": stage.get("identity_assumptions"), "probes": []}
    report["scene"] = run_operation(runtime, mapping["path"], mapping["body"])
    assert report["scene"]["scene_token"] == payload["scene_token"]
    report["prepare"] = run_operation(runtime, preparing["path"], payload)
    if options.held_check:
        import numpy as np
        import torch

        binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=report["scene"], arm=payload["arm"])
        with runtime.carrying_scope(binding, payload["gripper_joint_positions_rad"]):
            backend = next(iter(runtime._compound_backends.values()))
            planner, checker = backend.motion_planner, backend._motion_clearance_checker
            positions = torch.as_tensor(np.deg2rad(payload["start_joints_deg"])[None, None], **planner.device_cfg.as_torch_dict())
            state = checker.get_kinematics(positions)
            spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
            world = checker.get_collision_distance(state).detach().cpu().numpy().reshape(-1)
            self_cost = checker.get_self_collision_distance(state.robot_spheres).detach().cpu().numpy()
            report["actual_held_validity"] = {"world_positive_count": int(np.sum(world > 0)),
                "world_max_cost": float(world.max()), "self_positive_count": int(np.sum(self_cost > 0)),
                "self_max_cost": float(self_cost.max()), "sphere_count": len(spheres),
                "positive_world_spheres": [{"index": int(index), "world_sphere": spheres[index].tolist(), "cost": float(world[index])}
                                           for index in np.flatnonzero(world > 0)]}
            print("HELD_CHECK " + json.dumps(report["actual_held_validity"]), flush=True)
    algorithms = Algorithms.__new__(Algorithms)
    algorithms.request = lambda path, body: run_operation(runtime, path, body)
    for height in ((.04,) if options.centroid_control or options.forty_only else (.01, .04)):
        started = time.perf_counter()
        goal = release_goal(target, destination, holding, release_height_m=height)
        matrix = goal["T_world_tcp"]
        quaternion = Rotation.from_matrix([row[:3] for row in matrix[:3]]).as_quat()
        settings = deepcopy(payload["planning_settings"])
        settings["place_release_height_m"] = height
        request = deepcopy(payload) | {"goal_position_m": [row[3] for row in matrix[:3]],
            "goal_quaternion_wxyz": [float(quaternion[3]), *quaternion[:3].tolist()], "planning_settings": settings,
            "approach_axis": settings["place_approach_axis"], "approach_offset_m": settings["place_approach_offset_m"],
            "approach_in_tool_frame": settings["place_approach_in_tool_frame"],
            "lift_axis": settings["place_retreat_axis"], "lift_offset_m": settings["place_retreat_offset_m"],
            "lift_in_tool_frame": settings["place_retreat_in_tool_frame"]}
        result = {"release_height_m": height, "goal": goal}
        try:
            plan = run_operation(runtime, "/tools/plan_place", request)
            result["native_plan_succeeded"] = True
            result["collision_evidence"] = plan["collision_evidence"]
            phases = build_execution_phases(Algorithms._native_phases(plan["phases"], payload["arm"]), settings["time_scale"])
            checking = deepcopy(observation)
            result["checks"] = {}
            for phase in ("approach", "place"):
                result["checks"][phase] = algorithms.validate_execution(checking, payload["arm"], report["scene"], phase,
                    phases[phase], payload["carrying_binding"], planning_settings=settings)
                checking["arms"][payload["arm"]]["joints_rad"] = phases[phase]["positions_rad"][-1]
            result["actual_command_checks_succeeded"] = True
        except (AlgorithmError, CuroboToolError, ValueError, RuntimeError) as error:
            result.update(error_type=type(error).__name__, error_code=getattr(error, "code", None),
                          message=str(error), details=getattr(error, "details", None))
        result["wall_s"] = time.perf_counter()-started
        report["probes"].append(result)
        print(json.dumps({"probe": result}, indent=2), flush=True)
        if height == .01 and result.get("error_code") != "release_clearance_conflict":
            break
    runtime.clear_carrying()
    print("FINAL " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
