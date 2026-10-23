"""Shared-capacity held placement retains independent public SDK results."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_carrying import binding
from test_geometry_service import request_payload
from test_release_clearance import Tensor

from luna_mujoco import backend_builder, geometry_service
from luna_mujoco.planning_settings import PlanningSettings


def payload():
    value = request_payload(PlanningSettings())
    value["goal_matrices_world"] *= 16
    value["carrying_binding"] = binding(scene_token="scene")
    value["gripper_joint_positions_rad"] = {"A": .4, "B": 0.}
    return value


def request():
    from tianji_curobo.contracts import PlanGraspBatchRequest

    return PlanGraspBatchRequest.from_mapping(payload() | {"freq_hz": 50})


def raw_result():
    q = np.zeros((16, 1, 4, 7))
    q[:, 0, 1, :] = .1
    trajectory = SimpleNamespace(position=Tensor(q), velocity=Tensor(np.ones_like(q)),
        acceleration=Tensor(np.full_like(q, 2)), joint_names=[f"Joint{i}_L" for i in range(1, 8)])
    return SimpleNamespace(approach_success=Tensor([1] * 16), grasp_success=Tensor([1] * 16),
        status="completed", planning_time=.2, approach_interpolated_trajectory=trajectory,
        grasp_interpolated_trajectory=trajectory, approach_interpolated_last_tstep=Tensor([2] * 16),
        grasp_interpolated_last_tstep=Tensor([3] * 16),
        approach_trajectory=SimpleNamespace(**vars(trajectory), dt=Tensor([[.047]] * 16)),
        grasp_trajectory=SimpleNamespace(**vars(trajectory), dt=Tensor([[.047]] * 16)))


def backend(raw):
    value = backend_builder.ValidatedBatchBackend.__new__(backend_builder.ValidatedBatchBackend)
    value.arm, value.tool_frame = "A", "tcp"
    value.joint_names = tuple(f"Joint{i}_L" for i in range(1, 8))
    value._torch = SimpleNamespace(as_tensor=lambda values, **kwargs: Tensor(values))
    value._device, value._dtype = "cpu", "float32"
    value._state = lambda joints: SimpleNamespace(position=np.zeros((1, 7)))
    value._joint_state_type = SimpleNamespace(from_position=lambda positions, **kwargs: positions)
    value._pose_type = lambda **kwargs: kwargs
    value._goal_tool_pose_type = SimpleNamespace(from_poses=lambda poses, **kwargs: poses)
    value._tool_pose_criteria_type = lambda: "default"
    value.carrying_link, value.carrying_scene_token = "jev_carried_object_A", "scene"
    value.carrying_spheres_tool = np.array([[.01, 0., 0., .01]], dtype=np.float32)
    queries, calls, checked, restored = [], [], [], []
    def cost(spheres):
        queries.append(spheres.values.copy())
        return Tensor([[[.001 if len(queries) == 1 else 0.]]])
    value._motion_clearance_checker = SimpleNamespace(get_collision_distance=cost)
    def plan(goal, state, **kwargs):
        calls.append((goal, state, kwargs))
        return raw
    value.motion_planner = SimpleNamespace(plan_grasp=plan, update_tool_pose_criteria=restored.append,
        trajopt_solver=SimpleNamespace(config=SimpleNamespace(interpolation_dt=.02)), last_grasp_result=None)
    value.validate_trajectory_native = checked.append
    return value, calls, checked, restored


def test_release_conflict_rejects_only_one_candidate_and_two_phase_batch_copies_derivatives():
    raw = raw_result()
    value, calls, checked, restored = backend(raw)
    # Match the public tensor repeat operation while keeping this test CPU-only.
    class Position:
        def repeat(self, count, axis):
            return np.zeros((count, 7))
    value._state = lambda joints: SimpleNamespace(position=Position())
    result = value.plan_place_batch_native(request())
    assert len(calls) == 1 and calls[0][1].shape == (16, 7)
    assert calls[0][2]["plan_grasp_to_lift"] is False
    assert calls[0][2]["plan_approach_to_grasp"] is True
    assert calls[0][2]["disable_collision_links"] == []
    assert restored == [{"tcp": "default"}]
    assert result["batch_size"] == result["batch_capacity"] == 16
    assert [item["index"] for item in result["items"]] == list(range(16))
    failed, passed = result["items"][:2]
    assert failed["success"] is False and failed["status"] == "release_clearance_conflict"
    assert failed["release_clearance"]["conflicting_sphere_indices"] == [0]
    assert passed["success"] is True and set(passed["phases"]) == {"approach", "place"}
    assert checked == []  # Only final interpolated commands get the extra hard check.
    raw.grasp_interpolated_trajectory.velocity.values[:] = 99.
    assert len(passed["phases"]["place"]["positions_rad"]) == 3
    assert passed["phases"]["place"]["velocities_rad_s"][0] == (1.,) * 7
    assert passed["phases"]["place"]["interpolation_dt_s"] == .02
    assert value.motion_planner.last_grasp_result is None


def test_runtime_reuses_capacity_sixteen_backend_and_preserves_logical_hold_after_rejections(monkeypatch):
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene", "target_id": "sam-source",
        "observation_id": "rgbd-2", "generation": 2, "exclusion_state": "held"}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    created = []
    def create(arm, batch_size, **kwargs):
        created.append(batch_size)
        runtime.effective_config = {"A": {"plan": {"optimizer_attachment": True}}}
        return SimpleNamespace(update_voxel_scene=lambda *args, **kwargs: None,
            update_peer_collision_model=lambda *args: None, attach_measured_carrying=lambda *args: None,
            plan_place_batch_native=lambda request: {"items": [{"index": i, "success": False,
                "approach_success": False, "place_success": False, "phases": {}, "status": "unreachable"}
                for i in range(16)], "batch_size": 16, "batch_capacity": 16})
    monkeypatch.setattr(runtime, "_create_real_backend_unlocked", create)
    monkeypatch.setattr(runtime, "_prepare_scene_cache_unlocked", lambda: runtime.geometry_scene.copy())
    runtime.geometry_grids, runtime.geometry_model = [], {"cuboid": {}}
    held = object()
    runtime.logical_carrying_binding = held
    for _ in range(2):
        result = geometry_service.run_operation(runtime, "/tools/plan_place_batch", payload())
        assert result["ok"] is True and result["batch_capacity"] == 16
        assert result["collision_evidence"]["internal_batched_ik"] is True
        assert runtime.logical_carrying_binding is held
        assert runtime.carrying_binding is None
    assert created == [16]


@pytest.mark.parametrize("changes,match", [
    ({"goal_position_m": [0., 0., 0.]}, "singular"),
    ({"goal_matrices_world": [np.eye(4).tolist()]}, "exactly 16"),
    ({"gripper_joint_positions_rad": None}, "feedback"),
    ({"carrying_binding": None}, "carrying"),
])
def test_batch_endpoint_rejects_ambiguous_or_incomplete_input(changes, match):
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene", "target_id": "sam-source",
        "observation_id": "rgbd-2", "generation": 2}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    with pytest.raises(ValueError, match=match):
        geometry_service.run_operation(runtime, "/tools/plan_place_batch", payload() | changes)


def test_place_ik_uses_closed_gripper_and_held_attachment_scope(monkeypatch):
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene", "target_id": "sam-source",
        "observation_id": "rgbd-2", "generation": 2, "exclusion_state": "held"}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    seen = []

    def solve(request, lock_timeout_s):
        seen.append((runtime.carrying_binding.instance_id, runtime.gripper_feedback.copy(),
                     len(request.goal_matrices_world)))
        runtime.effective_config = {"A": {"ik": {"optimizer_attachment": True}}}
        return SimpleNamespace(to_payload=lambda: {"scene_token": "scene", "batch_size": 100,
            "items": [{"candidate_index": index, "success": True} for index in range(100)]})

    monkeypatch.setattr(runtime, "solve_ik_batch_native", solve)
    result = geometry_service.run_operation(runtime, "/tools/solve_ik_batch", payload())
    assert seen == [("sam-source", {"A": .4, "B": 0.}, 100)]
    assert result['batch_size'] == 16 and len(result['items']) == 16
    assert result["attached"] is True
    assert result["collision_evidence"]["optimizer_attachment"] is True
    assert runtime.carrying_binding is None

    with pytest.raises(ValueError, match="gripper feedback"):
        geometry_service.run_operation(runtime, "/tools/solve_ik_batch",
                                       payload() | {"gripper_joint_positions_rad": None})


def test_prepare_returns_exact_uploaded_cpu_spheres_without_building_trajectory_solver(monkeypatch):
    from luna_mujoco.carrying import CarryingBinding

    runtime = geometry_service.GeometryRuntime()
    scene = {"scene_token": "scene", "target_id": "sam-source", "observation_id": "rgbd-2", "generation": 2}
    value = CarryingBinding.from_mapping(payload()["carrying_binding"], scene=scene, arm="A")
    uploaded = np.array([[.0123, .0234, -.0567, .0089]], dtype=np.float32)
    backend_value = SimpleNamespace(carrying_spheres_tool=uploaded)
    def prepare():
        runtime.effective_config = {"A": {"ik": {"optimizer_attachment": True, "max_batch_size": 100},
                                          "plan": {"optimizer_attachment": True, "max_batch_size": 16}}}
    monkeypatch.setattr(runtime, "_prepare_scene_cache_unlocked", prepare)
    created = []
    def create(arm, batch_size, *, ik_only=False, **kwargs):
        created.append((batch_size, ik_only))
        return backend_value
    monkeypatch.setattr(runtime, "_create_real_backend_unlocked", create)
    monkeypatch.setattr(runtime, "_refresh_scene_unlocked", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "_refresh_peer_collision_unlocked", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "_require_scene_token_unchanged_unlocked", lambda token: None)
    result = runtime.prepare_carrying(request(), value, {"A": .4, "B": 0.})
    assert result["attachment_spheres_tool_m"] == uploaded.tolist()
    assert created == [(100, True)]
    assert result["collision_evidence"] == {"optimizer_attachment": True, "max_batch_size": 100}
    assert not runtime._batch_backends
    uploaded[:] = 99
    assert result["attachment_spheres_tool_m"][0][0] != 99


def test_prepare_and_failed_place_ik_reuse_one_backend_and_preserve_hold(monkeypatch):
    from tianji_curobo.contracts import NativeIkBatchResult, NativeIkResult

    from luna_mujoco.carrying import CarryingBinding

    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene", "target_id": "sam-source",
        "observation_id": "rgbd-2", "generation": 2, "exclusion_state": "held"}
    runtime.geometry_grids, runtime.geometry_model = [], {"cuboid": {}, "voxel": {}}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    held = CarryingBinding.from_mapping(payload()["carrying_binding"], scene=runtime.geometry_scene, arm="A")
    rejected = NativeIkResult(False, None, tuple(f"Joint{i}_L" for i in range(1, 8)), .1, .1)
    created = []
    def create(arm, batch_size, *, ik_only=False, **kwargs):
        created.append((batch_size, ik_only))
        evidence = {"optimizer_attachment": True, "max_batch_size": batch_size}
        runtime.effective_config.setdefault(arm, {})["ik" if ik_only else "plan"] = evidence
        return SimpleNamespace(carrying_spheres_tool=np.array([[.01, .02, .03, .04]], dtype=np.float32),
            update_voxel_scene=lambda *args, **kwargs: None,
            update_peer_collision_model=lambda *args: None, attach_measured_carrying=lambda *args: None,
            solve_ik_batch_native=lambda request: NativeIkBatchResult((rejected,) * len(request.goal_matrices_world),
                "scene", .01))
    monkeypatch.setattr(runtime, "_create_real_backend_unlocked", create)

    for _ in range(2):
        prepared = runtime.prepare_carrying(request(), held, {"A": .4, "B": 0.})
        result = geometry_service.run_operation(runtime, "/tools/solve_ik_batch", payload())
        assert prepared["attached"] is True and result["attached"] is True
        assert not any(item["success"] for item in result["items"])
        assert result["collision_evidence"] == prepared["collision_evidence"]
        assert runtime.logical_carrying_binding == held
    assert created == [(100, True)]
    assert not runtime._batch_backends


def test_all_failed_batch_needs_no_phase_samples():
    from luna_mujoco.place_batch import place_batch_items_from_result

    raw = SimpleNamespace(approach_success=Tensor([0] * 16), grasp_success=Tensor([0] * 16),
        status="No grasp in goal set was reachable.")
    items = place_batch_items_from_result(raw, [f"Joint{i}_L" for i in range(1, 8)], .02)
    assert len(items) == 16 and all(not item["success"] and item["phases"] == {} for item in items)


@pytest.mark.parametrize("length", [1, 1.5, 5, float("nan")])
def test_invalid_successful_candidate_length_is_a_service_failure(length):
    from luna_mujoco.place_batch import place_batch_items_from_result

    raw = raw_result()
    raw.grasp_interpolated_last_tstep.values[2] = length
    with pytest.raises(ValueError, match="length"):
        place_batch_items_from_result(raw, [f"Joint{i}_L" for i in range(1, 8)], .02)


@pytest.mark.parametrize("ik_only", [True, False])
def test_attachment_upload_uses_fixed_tool_spheres_and_ik_has_no_attachment_manager(ik_only):
    from luna_mujoco.carrying import CarryingBinding, carrying_spheres

    value, _, _, _ = backend(raw_result())
    value.ik_only = ik_only
    calls = []
    if not ik_only:
        value.motion_planner.attachment_manager = SimpleNamespace(update=lambda *args, **kwargs: calls.append(args))
    scene = {"scene_token": "scene", "target_id": "sam-source", "observation_id": "rgbd-2", "generation": 2}
    held = CarryingBinding.from_mapping(payload()["carrying_binding"], scene=scene, arm="A")
    value.attach_measured_carrying(held, [0.] * 7)
    np.testing.assert_array_equal(value.carrying_spheres_tool, np.asarray(carrying_spheres(held), dtype=np.float32))
    assert len(calls) == int(not ik_only)


@pytest.mark.parametrize("message,expected", [
    ("cuRobo trajectory optimization found no valid trajectory", "CUROBO_NO_VALID_TRAJECTORY"),
    ("cuRobo trajectory optimization failed", "CUROBO_TRAJOPT_FAILED"),
])
def test_retreat_distinguishes_normal_no_solution_from_internal_sdk_failure(monkeypatch, message, expected):
    from tianji_curobo.contracts import CuroboToolError

    value = backend_builder.ValidatedSingleBackend.__new__(backend_builder.ValidatedSingleBackend)
    value.motion_planner = SimpleNamespace(last_pose_result="old")
    def fail(self, request):
        raise CuroboToolError(message, code="CUROBO_TRAJOPT_FAILED", stage="trajopt", status=422,
            details={"arm": "A", "status": "failed"})
    monkeypatch.setattr(backend_builder._NativeCuroboBackend, "plan_to_pose_native", fail)
    with pytest.raises(CuroboToolError) as caught:
        value.plan_to_pose_native(object())
    assert caught.value.code == expected
    assert caught.value.details == {"arm": "A", "status": "failed"}
    assert value.motion_planner.last_pose_result is None


@pytest.mark.parametrize("ik_only", [True, False])
def test_held_backend_construction_reserves_shared_geometry_before_measured_update(monkeypatch, ik_only):
    import sys
    from types import ModuleType

    from test_backend_builder import ConfigFactory

    from luna_mujoco.carrying import CarryingBinding

    captured = {}
    class CheckerReached(Exception):
        pass
    class CheckerConfig:
        @staticmethod
        def load_from_config(**kwargs):
            captured["checker"] = kwargs["robot_config"]
            raise CheckerReached
    class Planner:
        def __init__(self, config):
            self.config, self.device_cfg, self.scene_collision_checker = config, object(), object()
            captured["config"] = config
    modules = {
        "curobo": {}, "torch": {},
        "curobo.collision_checking": {"RobotCollisionChecker": object, "RobotCollisionCheckerCfg": CheckerConfig},
        "curobo.inverse_kinematics": {"InverseKinematics": Planner, "InverseKinematicsCfg": ConfigFactory},
        "curobo.kinematics": {"Kinematics": object, "KinematicsCfg": object},
        "curobo.motion_planner": {"MotionPlannerCfg": ConfigFactory, "MotionPlanner": Planner},
        "curobo.batch_motion_planner": {"BatchMotionPlanner": Planner}, "curobo.scene": {"Cuboid": object},
        "curobo.types": dict.fromkeys(("GoalToolPose", "JointState", "Pose", "ToolPoseCriteria"), object),
        "tianji_curobo.grasp_planner": {"GraspBatchMotionPlanner": Planner},
    }
    for name, exports in modules.items():
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(backend_builder, "_graph_planner_rollout_config", dict)
    root = backend_builder.ROOT
    scene = {"scene_token": "scene", "target_id": "sam-source", "observation_id": "rgbd-2", "generation": 2}
    held = CarryingBinding.from_mapping(payload()["carrying_binding"], scene=scene, arm="A")
    runtime = SimpleNamespace(planning_settings=PlanningSettings(), carrying_binding=held,
        gripper_feedback={"A": .4, "B": 0.}, _robot_config_path=root / "resources/curobo/marvin_gui_dual.yml",
        _urdf_path=root / "assets/marvin_urdf/marvin_with_grippers.urdf", _asset_root_path=root / "assets",
        _excluded_collision_links=(), _scene_model=None, _collision_cache={})
    with pytest.raises(CheckerReached):
        backend_builder.build_geometry_backend(runtime, "A", batch_size=100 if ik_only else 16, ik_only=ik_only)
    config = captured["config"]
    assert config.max_batch_size == (100 if ik_only else 16)
    assert config.robot is captured["checker"]
    if not ik_only:
        assert config.ik_solver_config.robot is config.robot
    kin = config.robot["robot_cfg"]["kinematics"]
    assert kin["lock_joints"]["gripL_finger_joint"] == 0.
    for arm in ('A', 'B'):
        reserved = kin['collision_spheres']['jev_carried_object_' + arm]
        assert len(reserved) == 31 and all(row['radius'] < 0 for row in reserved)


def test_batch_leaves_final_command_check_to_caller_and_sdk_exception_is_fatal():
    class Position:
        def repeat(self, count, axis):
            return np.zeros((count, 7))
    value, _, _, _ = backend(raw_result())
    value._state = lambda joints: SimpleNamespace(position=Position())
    checked = []
    def check(phase):
        checked.append(phase)
        raise AssertionError('raw path checked before execution interpolation')
    value.validate_trajectory_native = check
    result = value.plan_place_batch_native(request())
    assert result["items"][1]["success"] is True
    assert checked == []
    assert result["items"][2]["success"] is True
    def crash(*args, **kwargs):
        raise RuntimeError("SDK broken")
    value.motion_planner.plan_grasp = crash
    with pytest.raises(RuntimeError, match="SDK broken"):
        value.plan_place_batch_native(request())
    assert value.motion_planner.last_grasp_result is None
