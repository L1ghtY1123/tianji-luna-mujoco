from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from jev_mujoco import geometry_service


@pytest.mark.parametrize('cache,key', [('_backends', 'A'), ('_batch_backends', ('A', 8, .01, .03))])
def test_validation_reuses_current_planner_world_but_still_checks_exact_command(monkeypatch, cache, key):
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {'exclusion_state': 'selected', 'scene_token': 'scene'}
    planner = object()
    backend = SimpleNamespace(motion_planner=planner)
    getattr(runtime, cache)[key] = backend
    monkeypatch.setattr(runtime, 'carrying_scope', lambda *args: nullcontext())
    monkeypatch.setattr(runtime, '_prepare_scene_cache_unlocked', lambda: None)
    def unexpected(*args):
        raise AssertionError('validation must not create another full trajectory planner')
    monkeypatch.setattr(runtime, '_compound_backend_unlocked', unexpected)
    refreshed, checked = [], []
    monkeypatch.setattr(runtime, '_refresh_scene_unlocked', lambda **kwargs: refreshed.append(kwargs['backend']))
    payload = {'arm': 'A', 'phase': 'retreat', 'scene_token': 'scene'}
    def check(owner, actual_planner, command, binding):
        assert owner is runtime and actual_planner is planner
        assert command is payload and binding is None
        checked.append(True)
        return {'collision_free': True}
    monkeypatch.setattr(geometry_service, 'check_command_path', check)
    assert runtime.validate_execution(payload, None)['collision_free']
    assert refreshed == [backend] and checked == [True]


def test_actual_feedback_change_does_not_destroy_world_borrowed_for_validation(monkeypatch):
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {'exclusion_state': 'selected', 'scene_token': 'scene'}
    planner = object()
    def unexpected(*args):
        raise AssertionError('world-only validation must not destroy or build trajectory planners')
    backend = SimpleNamespace(motion_planner=planner, destroy=unexpected)
    with runtime.carrying_scope(None, {'A': 0., 'B': 0.}):
        runtime._backends['A'] = backend
    monkeypatch.setattr(runtime, '_prepare_scene_cache_unlocked', lambda: None)
    monkeypatch.setattr(runtime, '_compound_backend_unlocked', unexpected)
    monkeypatch.setattr(runtime, '_refresh_scene_unlocked', lambda **kwargs: None)
    payload = {'arm': 'A', 'phase': 'retreat', 'scene_token': 'scene',
               'gripper_joint_positions_rad': {'A': .00028, 'B': .000038}}
    seen = []
    def check(owner, actual_planner, command, binding):
        assert actual_planner is planner and binding is None
        seen.append(command['gripper_joint_positions_rad'])
        return {'collision_free': True}
    monkeypatch.setattr(geometry_service, 'check_command_path', check)
    assert runtime.validate_execution(payload, None)['collision_free']
    assert seen == [payload['gripper_joint_positions_rad']]
    assert runtime._backends['A'] is backend


def test_dual_validation_uses_warm_world_and_measured_feedback(monkeypatch):
    from tianji_curobo.dual_contracts import DualValidationRequest

    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {'exclusion_state': 'observation', 'scene_token': 'scene'}
    planner = object()
    def unexpected(*args):
        raise AssertionError('checking both arms must not replace the warm motion planner')
    backend = SimpleNamespace(motion_planner=planner, destroy=unexpected)
    with runtime.carrying_scope(None, {'A': 0., 'B': 0.}):
        runtime._dual_backends[(.01, .03)] = backend
    monkeypatch.setattr(runtime, '_prepare_scene_cache_unlocked', lambda: None)
    monkeypatch.setattr(runtime, '_refresh_scene_unlocked', lambda **kwargs: None)
    command = {'points_deg': {'A': [[0.] * 7, [1.] * 7], 'B': [[0.] * 7, [-2.] * 7]},
               'expected_scene_token': 'scene'}
    feedback = {'A': .00028, 'B': .000038}
    seen = []
    def check(owner, actual_planner, payload, binding):
        import numpy as np
        assert actual_planner is planner and binding is None
        assert payload['gripper_joint_positions_rad'] == feedback
        np.testing.assert_allclose(payload['positions_rad'][-1], np.deg2rad([1.] * 7 + [-2.] * 7))
        seen.append(payload['arm'])
        return {'collision_free': True, 'scene_token': 'scene'}
    monkeypatch.setattr(geometry_service, 'check_command_path', check)
    result = runtime.validate_dual_execution(DualValidationRequest.from_mapping(command), feedback)
    assert result['collision_free'] and seen == ['AB']
    assert runtime._dual_backends[(.01, .03)] is backend


def test_dual_validation_reports_collision_with_original_endpoint_contract(monkeypatch):
    from tianji_curobo.contracts import CuroboToolError
    from tianji_curobo.dual_contracts import DualValidationRequest

    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'scene', 'exclusion_state': 'observation'}
    runtime._dual_backends[(.01, .03)] = SimpleNamespace(motion_planner=object())
    monkeypatch.setattr(runtime, '_prepare_scene_cache_unlocked', lambda: None)
    monkeypatch.setattr(runtime, '_refresh_scene_unlocked', lambda **kwargs: None)
    monkeypatch.setattr(geometry_service, 'check_command_path',
                        lambda *args: {'collision_free': False, 'first_invalid_index': 2})
    request = DualValidationRequest.from_mapping({'points_deg': {
        'A': [[0.] * 7, [1.] * 7], 'B': [[0.] * 7, [1.] * 7]}, 'expected_scene_token': 'scene'})
    with pytest.raises(CuroboToolError) as caught:
        runtime.validate_dual_execution(request, {'A': 0., 'B': 0.})
    assert caught.value.code == 'CUROBO_TRAJECTORY_COLLISION'
    assert caught.value.details['first_invalid_index'] == 2
