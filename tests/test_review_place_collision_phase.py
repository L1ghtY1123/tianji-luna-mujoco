"""Public SDK phase failures propagate without rewriting or executing a path."""

import pytest
from test_release_clearance import backend, request
from tianji_curobo.contracts import CuroboToolError

from luna_mujoco import backend_builder


@pytest.mark.parametrize('failed_phase', ['approach', 'grasp'])
def test_place_preserves_sdk_phase_failure_and_releases_result(monkeypatch, failed_phase):
    failure = CuroboToolError('public planner returned no solution', code='CUROBO_GRASP_FAILED',
                             stage='trajopt', status=422, details={'phase': failed_phase})

    def fail(*args):
        raise failure

    monkeypatch.setattr(backend_builder.ValidatedSingleBackend, '_plan_pre_release_native', fail)
    value = backend([[[0.]]])
    checked = []
    value.validate_trajectory_native = checked.append
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value is failure
    assert checked == []
    assert value.motion_planner.last_grasp_result is None
    assert value.motion_planner.last_pose_result is None
