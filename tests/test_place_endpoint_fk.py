"""A successful SDK flag cannot replace FK of the returned release samples."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_pre_release_place import planner_backend, raw_result
from test_release_clearance import Tensor, request

from luna_mujoco.backend_builder import CuroboToolError


def with_fk(value, poses):
    calls = []
    pending = list(poses)
    value._state = lambda joints: np.deg2rad(joints)
    def compute(state):
        calls.append(np.array(state))
        position, quaternion = pending.pop(0)
        pose = SimpleNamespace(position=Tensor([position]), quaternion=Tensor([quaternion]))
        return SimpleNamespace(tool_poses=SimpleNamespace(get_link_pose=lambda frame: pose))
    value.motion_planner.compute_kinematics = compute
    config = value.motion_planner.trajopt_solver.config
    config.position_tolerance, config.orientation_tolerance = .005, .05
    return calls


def test_successful_sdk_flag_with_stationary_place_at_approach_is_rejected():
    value, _, _, _ = planner_backend(raw_result())
    with_fk(value, [([.5, .2, .6], [1., 0, 0, 0])] * 2)
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value.code == "CUROBO_INVALID_TRAJECTORY"
    assert caught.value.details["compound_phase"] == "place"
    assert caught.value.details["position_error_m"] == pytest.approx(.1)
    assert caught.value.details["position_tolerance_m"] == .005
    assert value.motion_planner.last_grasp_result is None
    assert value.motion_planner.last_pose_result is None


def test_endpoint_errors_are_measured_from_last_valid_interpolated_samples():
    value, _, _, _ = planner_backend(raw_result())
    quaternion = Rotation.from_rotvec([0, 0, .02]).as_quat()[[3, 0, 1, 2]]
    calls = with_fk(value, [([.501, .2, .6], [1., 0, 0, 0]), ([.504, .2, .7], quaternion)])
    result = value.plan_place_native(request())
    assert result.approach.position_error_m == pytest.approx(.001, abs=1e-7)
    assert result.place.position_error_m == pytest.approx(.004, abs=1e-7)
    assert result.place.rotation_error_rad == pytest.approx(.02, abs=1e-7)
    assert len(calls) == 2
    np.testing.assert_allclose(calls, [[.1] * 7] * 2, atol=1e-7)


@pytest.mark.parametrize("phase,position_delta,rotation_delta", [
    ("approach", .0051, 0), ("place", .0051, 0), ("place", 0, .0501),
])
def test_each_phase_keeps_original_position_and_rotation_limits(phase, position_delta, rotation_delta):
    value, _, _, _ = planner_backend(raw_result())
    poses = [([.5, .2, .6], [1., 0, 0, 0]), ([.5, .2, .7], [1., 0, 0, 0])]
    index = 0 if phase == "approach" else 1
    poses[index][0][0] += position_delta
    poses[index] = (poses[index][0], Rotation.from_rotvec([0, rotation_delta, 0]).as_quat()[[3, 0, 1, 2]])
    with_fk(value, poses)
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value.details["compound_phase"] == phase
    assert caught.value.details["rotation_tolerance_rad"] == .05


@pytest.mark.parametrize("axis,in_tool_frame", [(axis, local) for axis in "xyz" for local in (False, True)])
def test_approach_endpoint_uses_request_axis_frame_and_full_rotation(axis, in_tool_frame):
    value, _, _, _ = planner_backend(raw_result())
    rotation = Rotation.from_euler("xyz", [.3, -.4, .8])
    quaternion = rotation.as_quat()[[3, 0, 1, 2]]
    goal = request(goal_quaternion_wxyz=quaternion.tolist(), approach_axis=axis,
        approach_offset_m=.12, approach_in_tool_frame=in_tool_frame)
    offset = np.zeros(3)
    offset["xyz".index(axis)] = .12
    if in_tool_frame:
        offset = rotation.apply(offset)
    with_fk(value, [(np.array(goal.goal_position_m) + offset, quaternion), (goal.goal_position_m, -quaternion)])
    result = value.plan_place_native(goal)
    assert result.approach.position_error_m < 1e-6
    assert result.place.rotation_error_rad < 1e-6


@pytest.mark.parametrize("temperature,accepted", [("cold", False), ("warm", True)])
def test_frozen_b3_cold_error_is_rejected_and_warm_measured_error_is_retained(temperature, accepted):
    fixture = json.loads((Path(__file__).parent / "fixtures/black_b3_release_endpoint.json").read_text())
    value, _, _, _ = planner_backend(raw_result())
    goal = request(**fixture["request"])
    poses = []
    for name in ("approach", "place"):
        matrix = np.array(fixture[temperature][name]["T_world_tcp"])
        poses.append((matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]))
    with_fk(value, poses)
    if accepted:
        result = value.plan_place_native(goal)
        assert result.place.position_error_m == pytest.approx(.004944565, abs=1e-7)
        assert result.place.rotation_error_rad == pytest.approx(.020434121, abs=1e-7)
    else:
        with pytest.raises(CuroboToolError) as caught:
            value.plan_place_native(goal)
        assert caught.value.details["position_error_m"] == pytest.approx(.119870851, abs=1e-7)


def test_fk_service_failure_cannot_return_a_plan_or_retain_borrowed_result_buffers():
    import gc
    import weakref

    class BorrowedResult:
        pass
    raw = BorrowedResult()
    raw.__dict__.update(vars(raw_result()))
    reference = weakref.ref(raw)
    pending = [raw]
    del raw
    value, _, _, _ = planner_backend()
    value.motion_planner.plan_grasp = lambda *args, **kwargs: pending.pop()
    def unavailable(state):
        raise RuntimeError("FK unavailable")
    value.motion_planner.compute_kinematics = unavailable
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value.code == "CUROBO_FORWARD_KINEMATICS_FAILED"
    gc.collect()
    assert reference() is None


def test_effective_solver_tolerance_is_used_without_a_new_looser_constant():
    value, _, _, _ = planner_backend(raw_result())
    with_fk(value, [([.5015, .2, .6], [1., 0, 0, 0])])
    value.motion_planner.trajopt_solver.config.position_tolerance = .001
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value.details["position_tolerance_m"] == .001
