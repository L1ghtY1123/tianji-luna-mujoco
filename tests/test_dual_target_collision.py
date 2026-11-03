"""An empty-hand dual motion must include the selected, already released object."""
from types import SimpleNamespace

import pytest

from luna_mujoco import geometry_service


def runtime_with_selected_target():
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'scene', 'exclusion_state': 'selected', 'target_id': 'pink',
                              'disabled_obstacle_names': ('semantic_instance::pink',)}
    runtime.geometry_model = {'voxel': {'semantic_instance::pink': {}}}
    return runtime


@pytest.mark.parametrize('raise_inside', [False, True])
def test_dual_planner_uploads_target_inclusive_world_and_restores_selection(monkeypatch, raise_inside):
    runtime = runtime_with_selected_target()

    def plan(owner, request, **kwargs):
        assert owner._scene_snapshot_unlocked()['disabled_obstacle_names'] == ()
        if raise_inside:
            raise RuntimeError('planning failed')
        return {'success': True}

    monkeypatch.setattr(geometry_service.CuroboRuntime, 'dual_operation_native', plan)
    if raise_inside:
        with pytest.raises(RuntimeError, match='planning failed'):
            runtime.dual_operation_native(object())
    else:
        assert runtime.dual_operation_native(object())['success']
    assert runtime._scene_snapshot_unlocked()['disabled_obstacle_names'] == ('semantic_instance::pink',)


def test_dual_command_validation_includes_released_target(monkeypatch):
    from tianji_curobo.dual_contracts import DualValidationRequest

    runtime = runtime_with_selected_target()
    runtime._dual_backends['test'] = SimpleNamespace(motion_planner=object())
    monkeypatch.setattr(runtime, '_prepare_scene_cache_unlocked', lambda: None)
    uploads = []
    monkeypatch.setattr(runtime, '_refresh_scene_unlocked', lambda **kwargs:
                        uploads.append(runtime._scene_snapshot_unlocked()['disabled_obstacle_names']))

    def check(*args):
        assert uploads == [()]
        return {'collision_free': True}

    monkeypatch.setattr(geometry_service, 'check_command_path', check)
    request = DualValidationRequest.from_mapping({'expected_scene_token': 'scene',
        'points_deg': {arm: [[0.] * 7, [1.] * 7] for arm in ('A', 'B')}})
    assert runtime.validate_dual_execution(request, {'A': 0., 'B': 0.})['collision_free']
    assert runtime._scene_snapshot_unlocked()['disabled_obstacle_names'] == ('semantic_instance::pink',)
