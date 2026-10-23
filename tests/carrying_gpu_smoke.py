"""Run directly inside the pinned, isolated cuRobo image; no robot hardware."""

import json

import numpy as np


def main():
    import torch
    import yaml
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    from tianji_curobo.curobo_robot_config import create_dual_arm_robot_config
    from tianji_curobo.curobo_service import materialize_urdf
    from tianji_curobo.gripper_collision_envelope import create_gripper_closing_envelope

    from luna_mujoco.carrying import CarryingBinding, configure_gripper_feedback, create_carrying_robot_config
    from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity
    from luna_mujoco.planning_settings import PlanningSettings

    assert sdk_identity()[0] == "8e734f3ced1df898990bcd92de40abce475907db"
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    scene = {"scene_token": "gpu-test", "observation_id": "synthetic-test", "generation": 1,
             "target_id": "surface", "disabled_obstacle_names": ()}
    binding = CarryingBinding.from_mapping({"arm": "A", "owner_id": "test-only", "observation_id": "synthetic-test",
        "generation": 1, "scene_token": "gpu-test", "instance_id": "surface", "frame": "tool", "state": "preflight",
        "points_tool_m": [[.23, -.02, -.02], [.27, .02, .02]],
        "estimated_bounds_tool_m": {"min_m": [.23, -.02, -.02], "max_m": [.27, .02, .02]},
        "shape_prior_provenance": {"source": "registered_rgbd_shape_estimate", "reference_observation_id": "synthetic-test",
                                   "instance_id": "surface", "assumptions": ["synthetic rigid box fixture"]},
        "uncertainty_m": .002, "coverage": .5}, scene=scene, arm="A")
    runtime.geometry_scene = scene
    runtime.geometry_grids = []
    runtime.geometry_model = {"cuboid": {}, "voxel": {}}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    report = {"sdk": sdk_identity()[0], "torch": torch.__version__, "cuda": torch.version.cuda}
    with runtime.carrying_scope(binding, {"A": 0., "B": 0.}):  # noqa: SIM117
        with runtime.gpu_lock:
            runtime._prepare_scene_cache_unlocked()
            backend = runtime._compound_backend_unlocked("A", .01, .03)
            q = backend.motion_planner.default_joint_state.position.reshape(-1).detach().cpu().numpy()
            backend.attach_measured_carrying(binding, np.rad2deg(q))
            planner = backend.motion_planner
            manager = planner.attachment_manager
            indices = manager.kinematics_params.get_sphere_index_from_link_name("jev_carried_object_A")
            spheres = manager.kinematics_params.link_spheres[0, indices].detach().cpu().numpy()
            assert len(spheres) > 0 and np.all(spheres[:, 3] > 0)
            report["optimizer_attachment"] = runtime.effective_config["A"]["plan"]["optimizer_attachment"]
            report["sphere_count"] = len(spheres)
            original = yaml.safe_load(runtime._robot_config_path.read_bytes())
            robot = create_dual_arm_robot_config(original, urdf_path=urdf, asset_root_path=urdf.parent)
            kinematics = Kinematics(KinematicsCfg.from_data_dict(robot, tool_frames=["TCP_Link_L"], device_cfg=planner.device_cfg))
            both = np.concatenate((q, q))
            joint_state = JointState.from_position(torch.as_tensor(both[None], **planner.device_cfg.as_torch_dict()),
                                                   joint_names=kinematics.joint_names)
            pose = kinematics.compute_kinematics(joint_state).tool_poses.get_link_pose("TCP_Link_L")
            point = pose.transform_points(torch.as_tensor([[.25, 0., 0.]], **planner.device_cfg.as_torch_dict())).reshape(-1, 3)[0]
            world = {"cuboid": {"bin_wall": {"pose": [*point.detach().cpu().tolist(), 1, 0, 0, 0], "dims": [.025, .025, .025]}}}
            def checker(config):
                return RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=config,
                    device_cfg=planner.device_cfg, scene_model=world, collision_activation_distance=.01))
            robot_checker = checker(robot)
            carried_robot, _link = create_carrying_robot_config(robot, binding)
            carried_checker = checker(carried_robot)
            positions = torch.as_tensor(both[None, None], **planner.device_cfg.as_torch_dict())
            robot_cost = robot_checker.get_collision_distance(robot_checker.get_kinematics(positions)).detach().cpu().numpy()
            carried_cost = carried_checker.get_collision_distance(carried_checker.get_kinematics(positions)).detach().cpu().numpy()
            assert np.all(robot_cost == 0), robot_cost
            assert np.any(carried_cost > 0), carried_cost
            report["robot_clears_wall"] = True
            report["carried_body_hits_wall"] = True
    assert runtime.carrying_binding is None and not runtime._compound_backends
    report["scoped_detach_cleanup"] = True
    settings = PlanningSettings(place_approach_offset_m=.06, place_retreat_offset_m=.06)
    runtime.geometry_solver_profile = settings.solver_payload()
    matrix = pose.get_matrix().reshape(-1, 4, 4)[0].detach().cpu().numpy()
    goal_position = matrix[:3, 3].copy()
    goal_position[2] += .03
    runtime.geometry_scene = scene | {"exclusion_state": "held"}
    held_binding = {name: getattr(binding, name) for name in binding.__dataclass_fields__} | {"state": "held"}
    carried_pose = run_operation(runtime, "/tools/plan_carried_pose", {
        "arm": "A", "start_joints_deg": np.rad2deg(q).tolist(), "other_joints_deg": np.rad2deg(q).tolist(),
        "goal_position_m": goal_position.tolist(), "goal_quaternion_wxyz": pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist(),
        "scene_token": "gpu-test", "planning_settings": settings.model_dump(),
        "carrying_binding": held_binding, "gripper_joint_positions_rad": {"A": .55, "B": 0.}})
    assert carried_pose["collision_evidence"]["optimizer_attachment"] is True and carried_pose["attached"] is True
    assert runtime.carrying_binding is None and not runtime._compound_backends
    held_trajectory = carried_pose["trajectory"]
    assert len(held_trajectory["positions_rad"]) == len(held_trajectory["velocities_rad_s"]) == len(held_trajectory["accelerations_rad_s2"])
    lift_check = run_operation(runtime, "/carrying/validate", {
        "arm": "A", "phase": "lift", "positions_rad": held_trajectory["positions_rad"],
        "joint_names": held_trajectory["joint_names"], "other_joints_deg": np.rad2deg(q).tolist(),
        "scene_token": "gpu-test", "planning_settings": settings.model_dump(),
        "carrying_binding": held_binding, "gripper_joint_positions_rad": {"A": .55, "B": 0.}})
    assert lift_check["collision_free"] is True and lift_check["attached"] is True
    report["carried_pose_optimizer_attachment"] = True
    report["carried_pose_native_points"] = len(held_trajectory["positions_rad"])
    report["carried_pose_qva_retained"] = True
    report["carried_lift_actual_command_checked_points"] = lift_check["checked_point_count"]
    report["carried_lift_actual_command_collision_free"] = True
    result = run_operation(runtime, "/tools/plan_place", {
        "arm": "A", "start_joints_deg": np.rad2deg(q).tolist(), "other_joints_deg": np.rad2deg(q).tolist(),
        "goal_position_m": goal_position.tolist(), "goal_quaternion_wxyz": pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist(),
        "scene_token": "gpu-test", "planning_settings": settings.model_dump(),
        "carrying_binding": {name: getattr(binding, name) for name in binding.__dataclass_fields__},
        "gripper_joint_positions_rad": {"A": 0., "B": 0.}})
    assert result["collision_evidence"]["optimizer_attachment"] is True
    assert result["retreat_provisional"] and result["retreat_requires_replan"]
    for phase in result["phases"].values():
        assert len(phase["velocities_rad_s"]) == len(phase["positions_rad"])
        assert len(phase["accelerations_rad_s2"]) == len(phase["positions_rad"])
    report["native_place_phase_points"] = {name: len(phase["positions_rad"]) for name, phase in result["phases"].items()}
    report["place_qva_retained"] = True
    runtime.geometry_model = world | {"voxel": {}}
    carrying_payload = {name: getattr(binding, name) for name in binding.__dataclass_fields__}
    check_payload = {"arm": "A", "phase": "place", "positions_rad": [q.tolist(), q.tolist()],
        "joint_names": backend.joint_names, "other_joints_deg": np.rad2deg(q).tolist(),
        "scene_token": "gpu-test", "planning_settings": settings.model_dump(),
        "gripper_joint_positions_rad": {"A": 0., "B": 0.}, "carrying_binding": carrying_payload}
    checked = run_operation(runtime, "/carrying/validate", check_payload)
    assert checked["collision_free"] is False and checked["attached"] is True
    report["actual_command_carried_wall_rejected"] = True
    runtime.geometry_scene = scene | {"exclusion_state": "released", "disabled_obstacle_names": ()}
    retreat_payload = check_payload | {"phase": "retreat"}
    retreat_payload.pop("carrying_binding")
    checked = run_operation(runtime, "/carrying/validate", retreat_payload)
    assert checked["collision_free"] is True and checked["attached"] is False
    report["released_retreat_robot_only"] = True
    retreat_goal = goal_position.copy()
    retreat_goal[2] += settings.place_retreat_offset_m
    fresh = run_operation(runtime, "/tools/plan_to_pose", {
        "arm": "A", "start_joints_deg": np.rad2deg(result["phases"]["place"]["positions_rad"][-1]).tolist(),
        "other_joints_deg": np.rad2deg(q).tolist(), "goal_position_m": retreat_goal.tolist(),
        "goal_quaternion_wxyz": pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist(),
        "scene_token": "gpu-test", "planning_settings": settings.model_dump(),
        "gripper_joint_positions_rad": {"A": 0., "B": 0.}})
    assert fresh["attached"] is False
    assert len(fresh["trajectory"]["velocities_rad_s"]) == len(fresh["trajectory"]["positions_rad"])
    report["fresh_detached_retreat_points"] = len(fresh["trajectory"]["positions_rad"])
    driver_diagnostics = {}
    free_driver_positions = []
    for driver_position in (.1, .2, .3, .4, .55, .8):
        candidate_robot = configure_gripper_feedback(robot, {"A": driver_position, "B": 0.}, urdf)
        candidate_robot, _ = create_carrying_robot_config(candidate_robot, binding)
        candidate_checker = checker(candidate_robot)
        state = candidate_checker.get_kinematics(positions)
        self_cost = float(candidate_checker.get_self_collision_distance(state.robot_spheres).max().detach().cpu())
        driver_diagnostics[str(driver_position)] = self_cost
        if self_cost == 0:
            free_driver_positions.append(driver_position)
    assert free_driver_positions, driver_diagnostics
    closed_driver = .55
    assert closed_driver in free_driver_positions
    report["authored_driver_self_costs"] = driver_diagnostics
    report["tested_closed_driver_rad"] = closed_driver
    closed_robot = configure_gripper_feedback(robot, {"A": closed_driver, "B": 0.}, urdf)
    swept_robot = create_gripper_closing_envelope(closed_robot, "A", urdf)
    def world_spheres(config):
        model = Kinematics(KinematicsCfg.from_data_dict(config, device_cfg=planner.device_cfg))
        return model, model.compute_kinematics(joint_state).robot_spheres[0, 0].detach().cpu().numpy()
    _closed_model, closed_spheres = world_spheres(closed_robot)
    swept_model, swept_spheres = world_spheres(swept_robot)
    finger_indices = []
    for name in robot["robot_cfg"]["kinematics"]["collision_link_names"]:
        if name.startswith("gripL_") and name != "gripL_Base":
            finger_indices.extend(swept_model.kinematics_config.get_sphere_index_from_link_name(name).detach().cpu().tolist())
    best_gap, candidate_point = -np.inf, None
    for sphere in swept_spheres[finger_indices]:
        for direction in np.concatenate((np.eye(3), -np.eye(3))):
            sample = sphere[:3] + sphere[3] * .85 * direction
            gap = np.min(np.linalg.norm(closed_spheres[:, :3] - sample, axis=1) - closed_spheres[:, 3])
            if gap > best_gap:
                best_gap, candidate_point = gap, sample
    assert best_gap > .02, best_gap
    runtime.geometry_model = {"cuboid": {"bin_wall": {"pose": [*candidate_point.tolist(), 1, 0, 0, 0], "dims": [.008] * 3}}, "voxel": {}}
    runtime.geometry_scene = scene
    closed_payload = check_payload | {"gripper_joint_positions_rad": {"A": closed_driver, "B": 0.}}
    closed_check = run_operation(runtime, "/carrying/validate", closed_payload)
    if not closed_check["collision_free"]:
        world = runtime.geometry_model
        diagnostic_robot, _ = create_carrying_robot_config(closed_robot, binding)
        diagnostic_checker = checker(diagnostic_robot)
        diagnostic_state = diagnostic_checker.get_kinematics(positions)
        print(json.dumps({"closed_check": closed_check, "candidate_gap_m": float(best_gap),
            "wall_center": candidate_point.tolist(),
            "world_cost": diagnostic_checker.get_collision_distance(diagnostic_state).detach().cpu().tolist(),
            "self_cost": diagnostic_checker.get_self_collision_distance(diagnostic_state.robot_spheres).detach().cpu().tolist(),
            "bound_cost": diagnostic_checker.get_bound(positions).detach().cpu().tolist()}), flush=True)
    assert closed_check["collision_free"] is True
    open_check = run_operation(runtime, "/carrying/validate", closed_payload | {
        "phase": "open", "gripper_start_position_rad": closed_driver, "gripper_end_position_rad": 0.})
    assert open_check["collision_free"] is False and open_check["gripper_sweep_checked"] is True
    report["closed_gripper_clears_wall"] = True
    report["opening_gripper_wall_rejected"] = True
    assert open_check["gripper_sweep_self_checked"] is False and open_check["gripper_internal_self_checked"]
    report["opening_driver_samples_checked"] = open_check["gripper_driver_samples_checked"]
    # The published source is excluded from optimization, then restored for
    # actual grasp/close checks with only narrowly allowed pad contact.
    source_scene = scene | {"exclusion_state": "selected", "disabled_obstacle_names": ("semantic_instance::surface",)}
    nonpad = list(range(len(closed_spheres)))
    pad_indices = []
    for link in ("gripL_Link3_L", "gripL_Link3_R"):
        pad_indices.extend(_closed_model.kinematics_config.get_sphere_index_from_link_name(link).detach().cpu().tolist())
    nonpad = [index for index in nonpad if index not in pad_indices]
    pad_gap, pad_point = -np.inf, None
    for sphere in closed_spheres[pad_indices]:
        for direction in np.concatenate((np.eye(3), -np.eye(3))):
            sample = sphere[:3] + sphere[3] * .8 * direction
            gap = np.min(np.linalg.norm(closed_spheres[nonpad, :3] - sample, axis=1) - closed_spheres[nonpad, 3])
            if gap > pad_gap:
                pad_gap, pad_point = gap, sample
    assert pad_gap > .015, pad_gap
    source_payload = {key: value for key, value in closed_payload.items() if key != "carrying_binding"} | {
        "phase": "grasp", "contact_goal_pose_world": matrix.tolist()}
    runtime.geometry_scene = source_scene
    runtime.geometry_model = {"cuboid": {"semantic_instance::surface": {"pose": [*pad_point.tolist(), 1, 0, 0, 0], "dims": [.008] * 3}}, "voxel": {}}
    pad_grasp = run_operation(runtime, "/carrying/validate", source_payload)
    assert pad_grasp["collision_free"] and pad_grasp["selected_source_layer_checked"]
    assert pad_grasp["target_contact_links"] == ["gripL_Link3_L", "gripL_Link3_R"]
    pad_approach = run_operation(runtime, "/carrying/validate", source_payload | {"phase": "approach_grasp"})
    assert not pad_approach["collision_free"]
    pad_close = run_operation(runtime, "/carrying/validate", source_payload | {
        "phase": "close", "gripper_start_position_rad": 0., "gripper_end_position_rad": .8})
    assert pad_close["collision_free"] and pad_close["gripper_internal_self_checked"], pad_close
    report["full_close_driver_position_rad"] = .8
    report["full_close_driver_samples_checked"] = pad_close["gripper_driver_samples_checked"]
    report["full_close_source_pad_contact_collision_free"] = True
    forearm_index = int(_closed_model.kinematics_config.get_sphere_index_from_link_name("Link5_L")[0].detach().cpu())
    runtime.geometry_model = {"cuboid": {"semantic_instance::surface": {"pose": [*closed_spheres[forearm_index, :3].tolist(), 1, 0, 0, 0], "dims": [.008] * 3}}, "voxel": {}}
    forearm_grasp = run_operation(runtime, "/carrying/validate", source_payload)
    assert not forearm_grasp["collision_free"]
    report["source_pad_contact_allowed_only_at_grasp"] = True
    report["source_forearm_contact_rejected"] = True
    # Reviewer-observed real witness: closed A is self-free, opening contacts B.
    peer_q_a = [2.708805799, -1.741319180, 1.038784146, -2.396264076, -1.943896055, .322357625, 1.068867803]
    peer_q_b = [-1.178343534, 1.690245390, -2.540313721, -.543797612, -2.417025566, .727467895, 1.110873342]
    runtime.geometry_scene = scene | {"exclusion_state": "held"}
    runtime.geometry_model = {"cuboid": {}, "voxel": {}}
    peer_binding = carrying_payload | {"points_tool_m": [[.48, -.005, -.005], [.49, .005, .005]], "state": "held",
        "estimated_bounds_tool_m": {"min_m": [.48, -.005, -.005], "max_m": [.49, .005, .005]}}
    peer_payload = closed_payload | {"phase": "place", "positions_rad": [peer_q_a],
        "other_joints_deg": np.rad2deg(peer_q_b).tolist(), "carrying_binding": peer_binding,
        "gripper_joint_positions_rad": {"A": .55, "B": 0.}}
    peer_closed = run_operation(runtime, "/carrying/validate", peer_payload)
    assert peer_closed["collision_free"], peer_closed
    peer_open = run_operation(runtime, "/carrying/validate", peer_payload | {
        "phase": "open", "gripper_start_position_rad": .55, "gripper_end_position_rad": 0.})
    assert not peer_open["collision_free"] and peer_open["gripper_internal_self_checked"]
    assert peer_open["gripper_sweep_self_checked"] is False
    report["peer_witness_closed_gripper_collision_free"] = True
    report["peer_witness_opening_rejected"] = True
    report["maximum_cuda_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
