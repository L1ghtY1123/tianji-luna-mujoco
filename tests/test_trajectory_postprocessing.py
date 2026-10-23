from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_builder import native_batch

from luna_mujoco.backend_builder import ValidatedBatchBackend, validate_grasp_batch
from luna_mujoco.public_planner import PublicPlanner as PlannerAdapter
from luna_mujoco.trajectory_postprocessing import (
    add_trajectory_derivatives,
    build_execution_phases,
)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()


def sdk_result():
    q = np.array([[[[0.] * 7, [.1] * 7]]] * 2)
    trajectory = SimpleNamespace(position=Tensor(q), velocity=Tensor(np.zeros_like(q)),
                                 acceleration=Tensor(np.zeros_like(q)),
                                 joint_names=[f"joint{i}" for i in range(7)])
    return SimpleNamespace(approach_interpolated_trajectory=trajectory,
                           grasp_interpolated_trajectory=trajectory,
                           lift_interpolated_trajectory=trajectory)


def execution_phases(item, scale=3):
    return build_execution_phases(
        {name: asdict(getattr(item, name)) for name in ("approach", "grasp", "lift")}, scale,
    )


def test_batch_preserves_only_raw_samples_and_copies_derivatives():
    original = native_batch()
    raw = sdk_result()
    result = add_trajectory_derivatives(original, raw)
    phase = result.items[0].approach
    assert phase.positions_rad == original.items[0].approach.positions_rad
    assert phase.interpolation_dt_s == .02
    assert phase.command_positions_rad is None
    assert phase.trajectory_point_scale is None
    assert phase.velocities_rad_s == ((0.,) * 7,) * 2
    assert phase.accelerations_rad_s2 == ((0.,) * 7,) * 2
    raw.approach_interpolated_trajectory.velocity.values[:] = 20.
    assert phase.velocities_rad_s == ((0.,) * 7,) * 2
    assert asdict(phase)["velocities_rad_s"] == phase.velocities_rad_s


@pytest.mark.parametrize("field", ["velocities_rad_s", "accelerations_rad_s2"])
@pytest.mark.parametrize("invalid", [((0.,) * 6,) * 2, ((float("nan"),) * 7,) * 2])
def test_raw_trajectory_rejects_mismatched_or_nonfinite_derivatives(field, invalid):
    phase = add_trajectory_derivatives(native_batch(), sdk_result()).items[0].approach
    with pytest.raises(ValueError, match="native trajectory"):
        replace(phase, **{field: invalid})


def test_scene_token_rebinding_keeps_copied_derivatives():
    phase = add_trajectory_derivatives(native_batch(), sdk_result()).items[0].approach
    rebound = replace(phase, scene_token="next-scene")
    assert rebound.scene_token == "next-scene"
    assert rebound.velocities_rad_s == phase.velocities_rad_s
    assert rebound.accelerations_rad_s2 == phase.accelerations_rad_s2


def test_derivative_collection_does_not_interpolate_and_selection_interpolates_only_three_phases(monkeypatch):
    from luna_mujoco import trajectory_postprocessing

    original = trajectory_postprocessing.quintic_hermite_points
    calls = []

    def interpolate(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(trajectory_postprocessing, "quintic_hermite_points", interpolate)
    batch = add_trajectory_derivatives(native_batch(), sdk_result())
    assert calls == []
    execution_phases(batch.items[1])
    assert len(calls) == 3


def test_selected_trajectory_hermite_preserves_endpoints_and_original_dt():
    raw = add_trajectory_derivatives(native_batch(), sdk_result())
    phase = execution_phases(raw.items[0])["approach"]
    assert len(phase["positions_rad"]) == 4
    assert phase["positions_rad"][0] == [0.] * 7
    assert phase["positions_rad"][-1] == [.1] * 7
    # Zero endpoint velocity/acceleration gives smoothstep, not a linear chord.
    u = 1 / 3
    assert phase["positions_rad"][1][0] == pytest.approx(.1 * (10*u**3 - 15*u**4 + 6*u**5))
    assert phase["raw_dt"] == phase["dt"] == .02
    assert phase["raw_point_count"] == 2
    assert phase["command_point_count"] == 4
    assert phase["raw_duration_s"] == .02
    assert phase["command_duration_s"] == .06


def test_hermite_uses_matching_derivatives_and_joint_order():
    raw = sdk_result()
    phase = raw.approach_interpolated_trajectory
    phase.joint_names.reverse()
    phase.velocity.values[0, 0, 0, 6] = 1.
    result = add_trajectory_derivatives(native_batch(), raw)
    assert result.items[0].approach.velocities_rad_s[0][0] == 1.
    points = execution_phases(result.items[0])["approach"]["positions_rad"]
    assert points[1][0] > points[1][1]


@pytest.mark.parametrize("field", ["velocity", "acceleration"])
def test_missing_derivatives_never_silently_fall_back_to_linear(field):
    raw = sdk_result()
    setattr(raw.approach_interpolated_trajectory, field, None)
    with pytest.raises(ValueError, match="derivative"):
        add_trajectory_derivatives(native_batch(), raw)


def test_mismatched_official_positions_are_rejected():
    raw = sdk_result()
    raw.approach_interpolated_trajectory.position.values[0, 0, 0, 0] = .3
    with pytest.raises(ValueError, match="positions"):
        add_trajectory_derivatives(native_batch(), raw)


def test_failed_batch_does_not_require_nonexistent_trajectories():
    native = native_batch()
    failed = replace(native.items[0], success=False, approach_success=False,
                     grasp_success=False, lift_success=False,
                     approach=None, grasp=None, lift=None, grasp_end_joints_rad=None)
    result = add_trajectory_derivatives(replace(native, items=(failed,), batch_size=1), None)
    assert result.items == (failed,)


@pytest.mark.parametrize("field", ["position", "velocity", "acceleration"])
@pytest.mark.parametrize("malformation", ["nan", "shape"])
def test_malformed_successful_phase_samples_are_rejected(field, malformation):
    raw = sdk_result()
    tensor = getattr(raw.approach_interpolated_trajectory, field)
    if malformation == "nan":
        tensor.values[0, 0, 0, 0] = np.nan
    else:
        tensor.values = tensor.values[:, :, :, :6]
    with pytest.raises(ValueError, match="positions|derivative"):
        add_trajectory_derivatives(native_batch(), raw)


@pytest.mark.parametrize("names", [None, ["joint0"] * 7, [f"missing{i}" for i in range(7)]])
def test_missing_duplicate_or_wrong_joint_names_are_rejected(names):
    raw = sdk_result()
    raw.approach_interpolated_trajectory.joint_names = names
    with pytest.raises(ValueError):
        add_trajectory_derivatives(native_batch(), raw)


def test_only_successful_candidate_valid_samples_are_used():
    native = native_batch()
    failed = replace(native.items[1], success=False, approach_success=False,
                     grasp_success=False, lift_success=False,
                     approach=None, grasp=None, lift=None, grasp_end_joints_rad=None)
    native = replace(native, items=(native.items[0], failed))
    raw = sdk_result()
    phase = raw.approach_interpolated_trajectory
    for field in ("position", "velocity", "acceleration"):
        tensor = getattr(phase, field)
        tensor.values[1] = np.nan
        tensor.values = np.concatenate((tensor.values, np.full((2, 1, 3, 7), np.nan)), axis=2)
    result = add_trajectory_derivatives(native, raw)
    assert len(result.items[0].approach.velocities_rad_s) == 2
    assert result.items[0].approach.command_positions_rad is None
    assert result.items[1] is failed


def test_noninteger_scale_rounds_intervals_up_with_original_dt():
    result = add_trajectory_derivatives(native_batch(), sdk_result())
    phase = execution_phases(result.items[0], 1.5)["approach"]
    assert len(phase["positions_rad"]) == 3
    assert phase["dt"] == .02
    assert phase["positions_rad"][1][0] == pytest.approx(.05)


def test_hard_clearance_checks_raw_paths_only_not_hermite_execution_points():
    from tianji_curobo.contracts import CuroboToolError

    raw = sdk_result()
    raw.approach_interpolated_trajectory.velocity.values[0, 0, 0, :] = 100.
    result = add_trajectory_derivatives(native_batch(), raw)
    assert np.max(execution_phases(result.items[0])["approach"]["positions_rad"]) > .1
    checked_points = []

    def validate(request):
        checked_points.append(request.points_deg)
        if np.max(np.deg2rad(request.points_deg)) > .11:
            raise CuroboToolError("inserted point collides", stage="validation",
                                 code="CUROBO_TRAJECTORY_COLLISION", status=422)

    checked = validate_grasp_batch(result, validate)
    assert checked.items[0].success
    assert checked.items[1].success
    assert [len(points) for points in checked_points] == [6, 6]


def test_public_grasp_composition_preserves_result_for_postprocessing():
    raw = sdk_result()

    class PublicPlanner:
        def plan_grasp(self, goal, state):
            assert goal == "goal" and state == "state"
            return raw

    planner = PlannerAdapter(PublicPlanner(), SimpleNamespace(), batch=True)
    assert planner.plan_grasp("goal", "state") is raw
    assert planner.last_grasp_result is raw


@pytest.mark.parametrize("failure", [None, "native", "derivative"])
def test_backend_releases_public_result_after_success_or_failure(monkeypatch, failure):
    from tianji_curobo._curobo_impl import _NativeCuroboBackend

    raw = sdk_result()
    if failure == "derivative":
        raw.approach_interpolated_trajectory.velocity = None
    backend = object.__new__(ValidatedBatchBackend)
    backend.arm = 'A'
    backend.motion_planner = SimpleNamespace(last_grasp_result="stale")

    def native_call(self, request):
        assert self.motion_planner.last_grasp_result is None
        self.motion_planner.last_grasp_result = raw
        if failure == "native":
            raise RuntimeError("native planner failed")
        return native_batch()

    monkeypatch.setattr(_NativeCuroboBackend, "plan_grasp_batch_native", native_call)
    checked = []
    backend.validate_trajectory_native = lambda request: checked.append(request.points_deg)
    backend.tool_fk_evidence = lambda joints, scene_token: {
        "joints_rad": list(joints), "scene_token": scene_token}
    request = SimpleNamespace(scene_token="scene")
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            backend.plan_grasp_batch_native(request)
    else:
        result = backend.plan_grasp_batch_native(request)
        assert result.items[0].approach.command_positions_rad is None
        assert len(result.items[0].approach.velocities_rad_s) == 2
        assert [len(points) for points in checked] == [6, 6]
    assert backend.motion_planner.last_grasp_result is None
