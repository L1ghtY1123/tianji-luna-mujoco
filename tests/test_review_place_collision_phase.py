"""Compound hard failures retain their evidence and identify the failed phase."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_release_clearance import backend, request
from tianji_curobo.contracts import CuroboToolError

from jev_mujoco import backend_builder


@pytest.mark.parametrize("failed_phase", ["approach", "place"])
def test_compound_phase_is_added_without_losing_hard_failure_or_continuing(monkeypatch, failed_phase):
    names = ("approach", "place")
    phases = SimpleNamespace(**{name: SimpleNamespace(positions_rad=[[0.] * 7, [.1 * (index + 1)] * 7])
                               for index, name in enumerate(names)})
    monkeypatch.setattr(backend_builder.ValidatedSingleBackend, "_plan_pre_release_native", lambda *args: phases)
    original_details = {"arm": "A", "reason": "motion_world_clearance", "first_invalid_index": 123,
                        "checked_point_count": 125, "phase": "original_checker_detail",
                        "costs": [1.1738e-6, 2.3538e-6]}
    failure = CuroboToolError("the active arm violates the motion world clearance",
        code="CUROBO_TRAJECTORY_COLLISION", stage="validation", status=422,
        retryable=True, details=original_details)
    value = backend([[[0.]]])
    checked, execution_calls = [], []

    def validate_unchanged_samples(command):
        name = names[len(checked)]
        np.testing.assert_array_equal(command.points_deg, np.rad2deg(getattr(phases, name).positions_rad))
        checked.append(name)
        if name == failed_phase:
            raise failure

    value.validate_trajectory_native = validate_unchanged_samples
    with pytest.raises(CuroboToolError) as caught:
        plan = value.plan_place_native(request())
        execution_calls.append(plan)

    error = caught.value
    assert error.details == original_details | {"compound_phase": failed_phase}
    assert (str(error), error.code, error.stage, error.status, error.retryable) == (
        str(failure), failure.code, failure.stage, failure.status, failure.retryable)
    assert error.__cause__ is failure
    assert failure.details == original_details
    assert checked == list(names[:names.index(failed_phase) + 1])
    assert execution_calls == []
    assert value.motion_planner.last_grasp_result is None
    assert value.motion_planner.last_pose_result is None
