"""Actual registered frames, recorded SAM3 and real GraspGen through GPU APIs."""

import json
from pathlib import Path

import numpy as np


def main():
    from tianji_curobo.curobo_service import materialize_urdf

    from luna_mujoco.geometry import grasp_targets
    from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity
    from luna_mujoco.trajectory_postprocessing import build_execution_phases

    directory = Path("/opt/jev/tests/artifacts/task3")
    payload = json.loads((directory / "real-rgbd-map-request.json").read_text())
    observation = json.loads((directory / "real-rgbd-observation.json").read_text())
    grasps = json.loads((directory / "real-rgbd-graspgen.json").read_text())
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    scene = run_operation(runtime, "/perception/build", payload)
    report = {"sdk": sdk_identity()[0], "input": "actual_registered_rgbd_recorded_official_sam3_real_graspgen",
              "map": scene, "visible_source_point_count": grasps["cloud"]["point_count"],
              "real_graspgen_candidate_count": grasps["candidate_count"], "arm_results": {}}
    target = next(obj for obj in observation["objects"] if obj["id"] == scene["target_id"])
    for arm, peer in (("A", "B"), ("B", "A")):
        initial = observation["arms"][arm]["joints_rad"]
        names = [f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)]
        base = {"arm": arm, "planning_settings": payload["planning_settings"], "scene_token": scene["scene_token"],
            "other_joints_deg": np.rad2deg(observation["arms"][peer]["joints_rad"]).tolist(),
            "gripper_joint_positions_rad": payload["gripper_joint_positions_rad"]}
        result = {"initial_open": run_operation(runtime, "/carrying/validate", base | {
            "phase": "open_grasp", "positions_rad": [initial], "current_joints_rad": initial, "joint_names": names,
            "gripper_start_position_rad": payload["gripper_joint_positions_rad"][arm], "gripper_end_position_rad": 0.})}
        all_goals = grasp_targets([np.asarray(target["T_world_object"]) @ candidate["T_object_grasp"]
                                  for candidate in grasps["candidates"]], arm)
        valid_indices = []
        for start in range(0, len(all_goals), 100):
            ik = run_operation(runtime, "/tools/solve_ik_batch", base | {
                "start_joints_deg": np.rad2deg(initial).tolist(), "goal_matrices_world": all_goals[start:start + 100].tolist()})
            valid_indices.extend(start + index for index, item in enumerate(ik["items"]) if item["success"])
        result["real_pool_ik_success_count"] = len(valid_indices)
        if not valid_indices:
            report["arm_results"][arm] = result
            continue
        valid_indices = valid_indices[:16]
        goals = all_goals[valid_indices]
        native = run_operation(runtime, "/tools/plan_grasp_batch", base | {
            "start_joints_deg": np.rad2deg(initial).tolist(), "goal_matrices_world": goals.tolist()})
        result["native_success_count"] = sum(item["success"] for item in native["items"])
        result["native_statuses"] = [{key: item[key] for key in ("candidate_index", "success", "status")}
                                     for item in native["items"]]
        result["candidate_checks"] = []
        for index, item in enumerate(native["items"]):
            if not item["success"]:
                continue
            phases = build_execution_phases(item["phases"], payload["planning_settings"]["time_scale"])
            checks = {"candidate": grasps["candidates"][valid_indices[index]], "T_world_tcp": goals[index].tolist()}
            for phase, actual in (("approach_grasp", "approach"), ("grasp", "grasp")):
                checks[phase] = run_operation(runtime, "/carrying/validate", base | {
                    "phase": phase, "positions_rad": phases[actual]["positions_rad"], "joint_names": names,
                    "contact_goal_pose_world": goals[index].tolist()})
            end = item["phases"]["grasp"]["positions_rad"][-1]
            checks["close"] = run_operation(runtime, "/carrying/validate", base | {
                "phase": "close", "positions_rad": [end], "joint_names": names,
                "contact_goal_pose_world": goals[index].tolist(), "gripper_start_position_rad": 0., "gripper_end_position_rad": .8})
            result["candidate_checks"].append(checks)
            # One real native successful candidate per arm diagnoses the actual
            # coverage boundary without pretending later stages were executed.
            break
        report["arm_results"][arm] = result
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
