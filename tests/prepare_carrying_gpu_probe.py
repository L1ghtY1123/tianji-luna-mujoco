"""Isolated held preparation/IK lifecycle probe; never calls a live service."""

import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    import torch
    from curobo.types import GoalToolPose, Pose
    from tianji_curobo.curobo_service import materialize_urdf

    from jev_mujoco.carrying import CarryingBinding, carrying_spheres
    from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity

    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    scene = {"scene_token": "prepare-probe", "target_id": "surface", "observation_id": "saved-probe",
             "generation": 1, "exclusion_state": "held", "disabled_obstacle_names": ()}
    runtime.geometry_scene, runtime.geometry_grids = scene, []
    runtime.geometry_model = {"cuboid": {}, "voxel": {}}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    binding = {"arm": "A", "owner_id": "prepare-probe", "observation_id": "saved-probe", "generation": 1,
        "scene_token": "prepare-probe", "instance_id": "surface", "frame": "tool", "state": "held",
        "points_tool_m": [[.23, -.02, .02], [.27, .02, .02]], "uncertainty_m": .002}
    feedback = {"A": .55, "B": 0.}
    base = {"arm": "A", "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
        "goal_position_m": [.4, .1, .7], "goal_quaternion_wxyz": [1., 0., 0., 0.],
        "scene_token": scene["scene_token"], "planning_settings": runtime.planning_settings.model_dump(),
        "carrying_binding": binding, "gripper_joint_positions_rad": feedback}
    report = {"sdk": sdk_identity()[0], "fixture": "two_measured_points_from_visible_surface_gpu_smoke",
        "scene": "empty_world_with_robot_self_collision", "physical_execution": False,
        "process_cold_disk_cache_preserved": True}
    started = time.perf_counter()
    first = run_operation(runtime, "/carrying/prepare", base)
    report["first_prepare_s"] = time.perf_counter() - started
    report["prepare_evidence"] = first["collision_evidence"]
    report["trajectory_backends_after_prepare"] = len(runtime._batch_backends)
    second = run_operation(runtime, "/carrying/prepare", base)
    report["repeat_prepare_s"] = second["prepare_wall_ms"] / 1000
    backend = next(iter(runtime._ik_batch_backends.values()), None)
    if backend is None:
        backend = next(iter(runtime._batch_backends.values()))
    planner = backend.motion_planner
    state = planner.default_joint_state.clone().unsqueeze(0)
    q = state.position.reshape(-1).detach().cpu().numpy()
    pose = planner.compute_kinematics(state).tool_poses.get_link_pose("TCP_Link_L")
    matrix = pose.get_matrix().reshape(-1, 4, 4)[0].detach().cpu().numpy()
    ik_payload = {key: value for key, value in base.items()
                  if key not in ("goal_position_m", "goal_quaternion_wxyz")}
    ik_payload.update(start_joints_deg=np.rad2deg(q).tolist(), other_joints_deg=np.rad2deg(q).tolist(),
                      goal_matrices_world=[matrix.tolist()])
    started = time.perf_counter()
    solved = run_operation(runtime, "/tools/solve_ik_batch", ik_payload)
    report["first_ik_s"] = time.perf_counter() - started
    report["first_ik"] = solved["items"]
    report["prepare_plus_ik_s"] = report["first_prepare_s"] + report["first_ik_s"]
    held = CarryingBinding.from_mapping(binding, scene=scene, arm="A")
    expected = np.asarray(carrying_spheres(held), dtype=np.float32)
    ik_backend = next(iter(runtime._ik_batch_backends.values()))
    ik = ik_backend.motion_planner
    models = {"ik": ik.kinematics, "hard_checker": ik_backend._motion_clearance_checker.kinematics}
    assert models['ik'].config.kinematics_config is models['hard_checker'].config.kinematics_config
    report['ik_and_checker_share_robot_model'] = True
    for name, model in models.items():
        params = model.config.kinematics_config
        indices = params.get_sphere_index_from_link_name("jev_carried_object_A")
        actual = params.link_spheres[0, indices].detach().cpu().numpy()
        np.testing.assert_array_equal(actual, expected)
        report[name + "_spheres_exact"] = True
    np.testing.assert_array_equal(np.asarray(first["attachment_spheres_tool_m"]), expected)
    report["uploaded_spheres_exact"] = True
    # Read full public IK result metrics, without replacing solver methods.
    target = Pose(position=pose.position.reshape(-1, 3).repeat(100, 1),
                  quaternion=pose.quaternion.reshape(-1, 4).repeat(100, 1))
    goal = GoalToolPose.from_poses({"TCP_Link_L": target}, ordered_tool_frames=["TCP_Link_L"], num_goalset=1)
    raw = ik.solve_pose(goal, return_seeds=ik.config.num_seeds)
    report["public_ik_metrics"] = {name: getattr(raw, name).detach().cpu().tolist()
        for name in ("success", "position_error", "rotation_error")}
    started = time.perf_counter()
    with runtime.carrying_scope(held, feedback), runtime.gpu_lock:
        batch = runtime._batch_backend_unlocked("A", 16, .01, .03)
        runtime._refresh_scene_unlocked(arm="A", backend=batch)
        runtime._refresh_peer_collision_unlocked(arm="A", backend=batch,
            active_joints_deg=np.rad2deg(q), other_joints_deg=np.rad2deg(q))
        for name, model in (("trajectory_ik", batch.motion_planner.ik_solver.kinematics),
                            ("trajopt", batch.motion_planner.trajopt_solver.kinematics),
                            ("graph", batch.motion_planner.graph_planner.kinematics),
                            ("trajectory_checker", batch._motion_clearance_checker.kinematics)):
            params = model.config.kinematics_config
            assert params is batch.motion_planner.kinematics.config.kinematics_config
            indices = params.get_sphere_index_from_link_name("jev_carried_object_A")
            np.testing.assert_array_equal(params.link_spheres[0, indices].detach().cpu().numpy(), expected)
            report[name + "_spheres_exact"] = True
    report["ensure_trajectory_backend_s"] = time.perf_counter() - started
    report["all_needed_backend_cost_s"] = report["prepare_plus_ik_s"] + report["ensure_trajectory_backend_s"]
    report["allocated_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["reserved_peak_mib"] = torch.cuda.max_memory_reserved() / 2**20
    report["holding_preserved"] = runtime.logical_carrying_binding == held
    output = Path(os.environ.get('JEV_PREPARE_PROBE_OUTPUT', '/tmp/jev-prepare-latency-20261016/result.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "public_ik_metrics"}), flush=True)
    runtime.clear_carrying()


if __name__ == "__main__":
    main()
