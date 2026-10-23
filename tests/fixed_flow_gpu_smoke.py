"""Pinned GPU adapter checks in an isolated /tmp checkout; no physical robot."""
import json
import time
from dataclasses import asdict

import numpy as np


def main():
    import torch
    from curobo.types import JointState
    from tianji_curobo.curobo_service import materialize_urdf

    from luna_mujoco.carrying import CarryingBinding, carrying_spheres
    from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity
    from luna_mujoco.planning_settings import PlanningSettings

    report = {"sdk": sdk_identity()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "physical_success_claimed": False}
    assert report["sdk"] == "8e734f3ced1df898990bcd92de40abce475907db"
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf", marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    settings = PlanningSettings(place_approach_offset_m=.03, place_retreat_offset_m=.03)
    runtime.configure_planning(settings)
    scene = {"scene_token": "gpu-test", "observation_id": "test", "generation": 1, "target_id": "surface", "disabled_obstacle_names": (), "exclusion_state": "held"}
    runtime.geometry_scene = scene
    runtime.geometry_model = {"cuboid": {}, "voxel": {}}
    runtime.geometry_grids = []
    runtime.geometry_solver_profile = settings.solver_payload()
    binding = CarryingBinding.from_mapping({"arm": "A", "owner_id": "gpu-test-owner", "observation_id": "test", "generation": 1, "scene_token": "gpu-test", "instance_id": "surface", "frame": "tool", "state": "held", "points_tool_m": [[.23, -.02, .02], [.27, .02, .02]], "uncertainty_m": .002}, scene=scene, arm="A")
    feedback = {"A": .55, "B": 0.}
    with runtime.carrying_scope(binding, feedback):
        runtime._prepare_scene_cache_unlocked()
        backend = runtime._compound_backend_unlocked("A", .01, .03)
        planner = backend.motion_planner
        q = planner.default_joint_state.position.reshape(-1).detach().cpu().numpy()
        backend.attach_measured_carrying(binding, np.rad2deg(q))
        state = JointState.from_position(torch.as_tensor(q[None], **planner.device_cfg.as_torch_dict()), joint_names=planner.joint_names)
        pose = planner.compute_kinematics(state).tool_poses.get_link_pose("TCP_Link_L")
        link = "jev_carried_object_A"
        expected = np.asarray(carrying_spheres(binding))
        for name, model in (("ik", planner.ik_solver.kinematics), ("trajopt", planner.trajopt_solver.kinematics)):
            params = model.config.kinematics_config
            indices = params.get_sphere_index_from_link_name(link)
            actual = params.link_spheres[0, indices].detach().cpu().numpy()
            assert np.allclose(actual, expected, atol=1e-7), name
        report["ik_trajopt_measured_spheres_match"] = True
        point = pose.transform_points(torch.as_tensor([binding.points_tool_m[0]], **planner.device_cfg.as_torch_dict())).reshape(-1, 3)[0].detach().cpu().tolist()
        runtime.geometry_model = {"cuboid": {"test_wall": {"pose": [*point, 1, 0, 0, 0], "dims": [.008]*3}}, "voxel": {}}
        runtime._refresh_scene_unlocked(arm="A", backend=backend)
        feasible = planner.graph_planner.check_samples_feasibility(state.position).detach().cpu().numpy()
        assert not feasible.all(), feasible
        report["graph_carried_surface_wall_rejected"] = True
    runtime.geometry_model = {"cuboid": {}, "voxel": {}}
    base = {"arm": "A", "start_joints_deg": np.rad2deg(q).tolist(), "other_joints_deg": np.rad2deg(q).tolist(), "goal_position_m": pose.position.reshape(-1, 3)[0].detach().cpu().tolist(), "goal_quaternion_wxyz": pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist(), "scene_token": "gpu-test", "planning_settings": settings.model_dump(), "carrying_binding": asdict(binding), "gripper_joint_positions_rad": feedback}
    print(json.dumps({"stage": "geometry_verified", "report": report}), flush=True)
    first = run_operation(runtime, "/carrying/prepare", base)
    cached_backend = next(iter(runtime._compound_backends.values()))
    second = run_operation(runtime, "/carrying/prepare", base)
    assert next(iter(runtime._compound_backends.values())) is cached_backend
    assert runtime.logical_carrying_binding == binding
    report["prepare_ms"] = [first["prepare_wall_ms"], second["prepare_wall_ms"]]
    report["exact_configuration_cache_reused"] = True
    checker = cached_backend._motion_clearance_checker
    raw = np.asarray(carrying_spheres(binding))
    world = raw.copy()
    matrix = pose.get_matrix()[0].detach().cpu().numpy()
    world[:, :3] = raw[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    subset = torch.as_tensor(world[None, None], **planner.device_cfg.as_torch_dict())
    subset_cost = checker.get_collision_distance(subset)
    assert tuple(subset_cost.shape) == (1, 1, len(raw))
    positions = torch.as_tensor(q[None, None], **planner.device_cfg.as_torch_dict())
    full_state = checker.get_kinematics(positions)
    full_cost = checker.get_collision_distance(full_state)
    assert tuple(full_cost.shape) == (1, 1, full_state.robot_spheres.shape[2])
    assert np.all(full_cost.detach().cpu().numpy() == 0)
    report["raw_subset_then_full_checker_shapes"] = [list(subset_cost.shape), list(full_cost.shape)]
    target = base["goal_position_m"].copy()
    target[2] += .03
    started = time.perf_counter()
    print(json.dumps({"stage": "prepare_verified", "report": report}), flush=True)
    placed = run_operation(runtime, "/tools/plan_place", base | {"goal_position_m": target})
    report["native_place_wall_ms"] = (time.perf_counter()-started)*1000
    report["native_place_points"] = {name: len(value["positions_rad"]) for name, value in placed["phases"].items()}
    report["native_place_optimizer_attachment"] = placed["collision_evidence"]["optimizer_attachment"]
    from tianji_curobo.contracts import CuroboToolError
    wall = world[0, :3].copy() + [0, 0, .03]
    runtime.geometry_model = {"cuboid": {"release_endpoint_wall": {"pose": [*wall.tolist(), 1, 0, 0, 0], "dims": [.008] * 3}}, "voxel": {}}
    try:
        run_operation(runtime, "/tools/plan_place", base | {"goal_position_m": target})
    except CuroboToolError as error:
        assert error.code == "release_clearance_conflict", str(error)
        report["release_endpoint_positive_cost_classified"] = True
        report["release_endpoint_conflicting_sphere_count"] = len(error.details["conflicting_sphere_indices"])
    else:
        raise AssertionError("positive endpoint world cost must reject before native planning")
    runtime.geometry_model = {"cuboid": {}, "voxel": {}}
    path = placed["phases"]["place"]
    checked = run_operation(runtime, "/carrying/validate", base | {"phase": "place", "joint_names": path["joint_names"], "positions_rad": path["positions_rad"]})
    assert checked["collision_free"]
    report["place_actual_checker_passed"] = True
    held_backend = next(iter(runtime._compound_backends.values()))
    retreat_payload = dict(base)
    retreat_payload.pop("carrying_binding")
    retreat_payload["start_joints_deg"] = np.rad2deg(path["positions_rad"][-1]).tolist()
    retreat_payload["goal_position_m"] = (np.asarray(target) + [0, 0, .03]).tolist()
    retreat_payload["gripper_joint_positions_rad"] = {"A": 0., "B": 0.}
    retreat = run_operation(runtime, "/tools/plan_to_pose", retreat_payload)
    assert retreat["attached"] is False and runtime.logical_carrying_binding == binding
    retreat_path = retreat["trajectory"]
    checked = run_operation(runtime, "/carrying/validate", retreat_payload | {"phase": "retreat",
        "joint_names": retreat_path["joint_names"], "positions_rad": retreat_path["positions_rad"]})
    assert checked["collision_free"] and runtime.logical_carrying_binding == binding
    report["retreat_validated_before_logical_release"] = True
    open_backend = next(iter(runtime._compound_backends.values()))
    run_operation(runtime, "/carrying/prepare", base)
    assert next(iter(runtime._compound_backends.values())) is held_backend
    report["held_open_held_cache_reused"] = True
    report["retreat_planned_before_logical_release"] = True
    current_backend = held_backend
    params = current_backend.motion_planner.attachment_manager.kinematics_params
    indices = params.get_sphere_index_from_link_name(link)
    run_operation(runtime, "/carrying/detach", {})
    assert runtime.geometry_scene == scene and runtime.logical_carrying_binding is None
    assert np.all(params.link_spheres[0, indices, 3].detach().cpu().numpy() < 0)
    report["detach_inactive_slots_and_scene_retained"] = True
    assert next(iter(runtime._compound_backends.values())) is open_backend
    report["detach_keeps_preplanned_open_backend"] = True
    runtime.geometry_scene = scene | {"exclusion_state": "observation"}
    observation = {key: value for key, value in base.items() if key != "carrying_binding"}
    started = time.perf_counter()
    observed = run_operation(runtime, "/tools/plan_observation", observation | {"goal_position_m": target})
    assert observed["attached"] is False
    report["native_observation_wall_ms"] = (time.perf_counter()-started)*1000
    report["native_observation_points"] = len(observed["trajectory"]["positions_rad"])
    trajectory = observed["trajectory"]
    actual_observation = run_operation(runtime, "/carrying/validate", observation | {"phase": "observation", "positions_rad": trajectory["positions_rad"], "joint_names": trajectory["joint_names"]})
    assert actual_observation["collision_free"]
    assert actual_observation["gripper_configuration_source"] == "holding_or_open_feedback"
    report["observation_actual_checker_passed"] = True
    validated = run_operation(runtime, "/carrying/validate", observation | {"phase": "approach_grasp", "gripper_geometry_mode": "fixed_open", "positions_rad": trajectory["positions_rad"], "joint_names": trajectory["joint_names"]})
    assert validated["gripper_configuration_source"] == "configured_open_gripper"
    assert "gripper_sweep_checked" not in validated
    report["fixed_open_configuration_checked"] = True
    runtime.clear_carrying()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
