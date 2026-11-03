"""Pre-release placement never requires an attached retreat."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from test_release_clearance import Tensor, backend, request

from luna_mujoco.trajectory_postprocessing import build_execution_phases


def raw_result(**changes):
    names = [f"Joint{i}_L" for i in range(1, 8)]
    q = np.zeros((1, 1, 4, 7))
    q[0, 0, 1] = .1
    trajectory = SimpleNamespace(position=Tensor(q), velocity=Tensor(np.ones_like(q)),
        acceleration=Tensor(np.zeros_like(q)), joint_names=names)
    return SimpleNamespace(approach_success=True, grasp_success=True, lift_success=False,
        status="success", planning_time=.2, approach_interpolated_trajectory=trajectory,
        grasp_interpolated_trajectory=trajectory, approach_interpolated_last_tstep=Tensor([2]),
        grasp_interpolated_last_tstep=Tensor([2]),
        approach_trajectory=SimpleNamespace(**vars(trajectory), dt=Tensor([[.047]])),
        grasp_trajectory=SimpleNamespace(**vars(trajectory), dt=Tensor([[.047]])), **changes)


def planner_backend(result=None, error=None):
    value = backend([[[0.]]])
    calls, restored = [], []
    value.tool_frame = "tcp"
    value.joint_names = tuple(f"Joint{i}_L" for i in range(1, 8))
    value._state_and_goal = lambda *args: ("state", "goal")
    value._tool_pose_criteria_type = lambda: "default_criteria"
    def plan(goal, state, **kwargs):
        calls.append((goal, state, kwargs))
        # An attached lift is deliberately infeasible in this regression.
        if kwargs.get("plan_grasp_to_lift"):
            raise RuntimeError("attached retreat blocked")
        if error is not None:
            raise error
        return result
    value.motion_planner.plan_grasp = plan
    value.motion_planner.update_tool_pose_criteria = restored.append
    value.motion_planner.trajopt_solver = SimpleNamespace(config=SimpleNamespace(interpolation_dt=.02,
        position_tolerance=.005, orientation_tolerance=.05))
    value._state = lambda joints: np.deg2rad(joints)
    fk_positions = [[.5, .2, .6], [.5, .2, .7]]
    def compute(state):
        pose = SimpleNamespace(position=Tensor([fk_positions.pop(0)]), quaternion=Tensor([[1., 0., 0., 0.]]))
        return SimpleNamespace(tool_poses=SimpleNamespace(get_link_pose=lambda frame: pose))
    value.motion_planner.compute_kinematics = compute
    checked = []
    value.validate_trajectory_native = checked.append
    return value, calls, restored, checked


def test_attached_retreat_failure_does_not_gate_real_approach_place_or_create_a_retreat():
    raw = raw_result()
    value, calls, restored, checked = planner_backend(raw)
    plan = value.plan_place_native(request())
    assert len(calls) == 1
    assert calls[0][2]["plan_approach_to_grasp"] is True
    assert calls[0][2]["plan_grasp_to_lift"] is False
    assert calls[0][2]["disable_collision_links"] == []
    assert restored == [{"tcp": "default_criteria"}]
    assert not checked and not hasattr(plan, "retreat")  # Check final interpolated commands in Algorithms.
    assert plan.scene_token is None
    assert len(plan.place.positions_rad) == 2  # SDK padding is not execution.
    assert plan.place.terminal_state['dt'] == pytest.approx(.047)  # Native optimization dt, not command dt.
    assert plan.place.terminal_state['velocity'] == [1.] * 7
    raw.grasp_interpolated_trajectory.velocity.values[:] = 99
    assert plan.place.velocities_rad_s == ((1.,) * 7,) * 2
    assert set(build_execution_phases({name: asdict(getattr(plan, name))
        for name in ("approach", "place")}, 3)) == {"approach", "place"}
    assert value.motion_planner.last_grasp_result is None
    assert value.motion_planner.last_pose_result is None


@pytest.mark.parametrize("sdk_phase,compound_phase", [("approach", "approach"), ("grasp", "place")])
def test_public_phase_failure_remains_a_hard_failure(sdk_phase, compound_phase):
    raw = raw_result()
    setattr(raw, sdk_phase + "_success", False)
    value, _, restored, checked = planner_backend(raw)
    with pytest.raises(Exception) as caught:
        value.plan_place_native(request())
    assert caught.value.code == "CUROBO_GRASP_FAILED"
    assert caught.value.details["phase"] == sdk_phase
    assert caught.value.details["compound_phase"] == compound_phase
    assert restored and not checked
    assert value.motion_planner.last_grasp_result is None


def test_public_exception_still_restores_criteria_and_clears_result_buffers():
    value, _, restored, _ = planner_backend(error=RuntimeError("native error"))
    with pytest.raises(Exception) as caught:
        value.plan_place_native(request())
    assert caught.value.code == "CUROBO_GRASP_FAILED" and restored
    assert value.motion_planner.last_grasp_result is None
    assert value.motion_planner.last_pose_result is None


@pytest.mark.parametrize('status', ['Goalset planning returned None.', 'No grasp in goal set was reachable.'])
def test_release_goal_preflight_failure_is_not_reported_as_approach_failure(status):
    raw = raw_result()
    raw.status, raw.approach_success, raw.grasp_success = status, False, False
    value, _, restored, checked = planner_backend(raw)
    with pytest.raises(Exception) as caught:
        value.plan_place_native(request())
    assert caught.value.code == 'CUROBO_GRASP_FAILED'
    assert caught.value.details['phase'] == 'goalset'
    assert caught.value.details['compound_phase'] == 'release_goal_preflight'
    assert caught.value.details['status'] == status
    assert restored and not checked


@pytest.mark.parametrize("count", [0, 1, 5, 1.5, float("nan")])
def test_invalid_sdk_phase_length_is_not_a_successful_place(count):
    raw = raw_result()
    raw.grasp_interpolated_last_tstep = Tensor([count])
    value, _, _, _ = planner_backend(raw)
    with pytest.raises(Exception, match="length"):
        value.plan_place_native(request())
    assert value.motion_planner.last_grasp_result is None


def test_criteria_restore_failure_is_fatal_even_after_public_plan_success():
    from tianji_curobo.contracts import CuroboToolError

    value, _, _, checked = planner_backend(raw_result())
    def failed_restore(criteria):
        raise RuntimeError("restore failed")
    value.motion_planner.update_tool_pose_criteria = failed_restore
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value.code == "CUROBO_GRASP_RESTORE_FAILED"
    assert not checked and value.motion_planner.last_grasp_result is None


def test_reported_conversion_error_does_not_retain_the_borrowed_sdk_result():
    import gc
    import weakref

    from tianji_curobo.contracts import CuroboToolError

    class BorrowedResult:
        pass
    raw = BorrowedResult()
    raw.__dict__.update(vars(raw_result()))
    raw.grasp_interpolated_last_tstep = Tensor([5])
    reference = weakref.ref(raw)
    pending = [raw]
    del raw
    value, _, _, _ = planner_backend()
    value.motion_planner.plan_grasp = lambda *args, **kwargs: pending.pop()
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value.code == "CUROBO_INVALID_TRAJECTORY"
    gc.collect()
    assert reference() is None  # Keep the reported error alive during this check.


def runtime_fixture():
    from luna_mujoco.geometry_service import GeometryRuntime

    planned, _, _, _ = planner_backend(raw_result())
    plan = planned.plan_place_native(request())
    runtime = GeometryRuntime()
    calls = []
    def prepare():
        assert runtime.gpu_lock.locked()
        calls.append("prepare")
        return {"scene_token": "current"}
    runtime._prepare_scene_cache_unlocked = prepare
    def refresh(**kwargs):
        assert kwargs["snapshot"]["scene_token"] == "current"
        calls.append("refresh")
        return "current"
    runtime._refresh_scene_unlocked = refresh
    def peer(**kwargs):
        assert kwargs["active_joints_deg"] == request().start_joints_deg
        assert kwargs["other_joints_deg"] == request().other_joints_deg
        calls.append("peer")
    runtime._refresh_peer_collision_unlocked = peer
    def native(value):
        assert runtime.gpu_lock.locked()
        calls.append("native")
        return plan
    backend_value = SimpleNamespace(plan_place_native=native)
    runtime._compound_backend_unlocked = lambda *args: backend_value
    def unchanged(token):
        assert token == "current"
        calls.append("unchanged")
    runtime._require_scene_token_unchanged_unlocked = unchanged
    runtime._discard_compound_backend_unlocked = lambda **kwargs: calls.append("discard")
    return runtime, backend_value, plan, calls


def test_runtime_refreshes_scene_peer_and_returns_copied_two_phase_scene_identity():
    runtime, _, original, calls = runtime_fixture()
    result = runtime.plan_pre_release_place_native(request())
    assert calls == ["prepare", "refresh", "peer", "native", "unchanged"]
    assert result.scene_token == result.approach.scene_token == result.place.scene_token == "current"
    assert original.scene_token is None and not hasattr(result, "retreat")
    assert not runtime.gpu_lock.locked()


@pytest.mark.parametrize("scene_changed", [False, True])
def test_runtime_failure_never_returns_plan_and_always_releases_gpu_lock(scene_changed):
    from tianji_curobo.contracts import CuroboToolError

    runtime, native, _, calls = runtime_fixture()
    error = CuroboToolError("failure", stage="validation",
        code="CUROBO_SCENE_CHANGED" if scene_changed else "CUROBO_GRASP_FAILED")
    def fail(*args):
        raise error
    if scene_changed:
        runtime._require_scene_token_unchanged_unlocked = fail
    else:
        native.plan_place_native = fail
    with pytest.raises(CuroboToolError) as caught:
        runtime.plan_pre_release_place_native(request())
    assert caught.value is error
    assert ("discard" in calls) is not scene_changed  # Match the established runtime contract.
    assert not runtime.gpu_lock.locked()


def test_runtime_rejects_three_phase_or_placeholder_backend_contract_and_discards():
    from tianji_curobo.contracts import CuroboToolError

    runtime, native, _, calls = runtime_fixture()
    native.plan_place_native = lambda *args: SimpleNamespace(approach="fake", place="fake", retreat="fake")
    with pytest.raises(CuroboToolError) as caught:
        runtime.plan_pre_release_place_native(request())
    assert caught.value.code == "CUROBO_BACKEND_FAILED"
    assert "discard" in calls and not runtime.gpu_lock.locked()


def test_busy_runtime_cannot_mutate_scene_or_release_another_operations_lock():
    from tianji_curobo.contracts import CuroboToolError

    runtime, _, _, calls = runtime_fixture()
    with runtime.gpu_lock:
        with pytest.raises(CuroboToolError) as caught:
            runtime.plan_pre_release_place_native(request(), lock_timeout_s=0)
        assert caught.value.code == "CUROBO_BUSY" and runtime.gpu_lock.locked()
    assert calls == []


@pytest.mark.parametrize("timeout", [True, "600", -1., float("nan"), float("inf")])
def test_runtime_preserves_typed_timeout_validation(timeout):
    from tianji_curobo.contracts import CuroboToolError

    runtime, _, _, calls = runtime_fixture()
    with pytest.raises(CuroboToolError) as caught:
        runtime.plan_pre_release_place_native(request(), lock_timeout_s=timeout)
    assert caught.value.code == "CUROBO_INVALID_TIMEOUT" and not calls


