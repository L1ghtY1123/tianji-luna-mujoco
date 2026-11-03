from types import SimpleNamespace

import pytest

from luna_mujoco.geometry_service import measured_target_collision_scope


def target_world():
    states = {'semantic_instance::target': False, 'semantic_instance::peer': True}
    calls = []
    def set_enabled(name, *, enabled, env_idx):
        assert env_idx == 0
        calls.append((name, enabled))
        states[name] = enabled
    model = {'voxel': {name: {} for name in states}}
    runtime = SimpleNamespace(geometry_model=model, geometry_scene={
        'target_id': 'target', 'exclusion_state': 'selected',
        'disabled_obstacle_names': [name for name, enabled in states.items() if not enabled]})
    data = SimpleNamespace(**{kind: SimpleNamespace(set_enabled=set_enabled) for kind in ('voxels', 'cuboids', 'meshes')})
    return runtime, SimpleNamespace(scene_collision_checker=SimpleNamespace(data=data)), states, calls


@pytest.mark.parametrize('phase', ['approach_grasp', 'grasp', 'grasp_closure'])
@pytest.mark.parametrize('raise_inside', [False, True])
def test_open_grasp_target_layers_are_enabled_and_selection_restored(phase, raise_inside):
    runtime, planner, states, calls = target_world()
    try:
        with measured_target_collision_scope(runtime, planner, phase) as proof:
            assert all(states.values())
            assert len(proof['target_layers']) == 1
            if raise_inside:
                raise RuntimeError('checker failed')
    except RuntimeError:
        assert raise_inside
    assert states == {'semantic_instance::target': False,
                      'semantic_instance::peer': True}
    assert len(calls) == 2


@pytest.mark.parametrize('phase', ['close', 'lift', 'place', 'observation'])
def test_contact_and_held_phases_do_not_change_target_layer_policy(phase):
    runtime, planner, states, calls = target_world()
    with measured_target_collision_scope(runtime, planner, phase):
        assert not states['semantic_instance::target']
    assert not calls


def test_missing_selected_target_geometry_cannot_pass_open_grasp_check():
    runtime, planner, _, _ = target_world()
    runtime.geometry_model = {}
    with (pytest.raises(ValueError, match='target collision layers missing'),
          measured_target_collision_scope(runtime, planner, 'grasp')):
        pytest.fail('missing geometry admitted')


def test_validation_checks_world_with_target_enabled_then_restores_selection(monkeypatch):
    from luna_mujoco import geometry_service
    source, planner, states, _ = target_world()
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = source.geometry_scene | {'scene_token': 'scene'}
    runtime.geometry_model = source.geometry_model
    runtime._backends['A'] = SimpleNamespace(motion_planner=planner)
    monkeypatch.setattr(runtime, '_prepare_scene_cache_unlocked', lambda: None)
    monkeypatch.setattr(runtime, '_refresh_scene_unlocked', lambda **kwargs: None)
    def check(owner, checked_planner, payload, binding):
        assert owner is runtime and checked_planner is planner
        assert all(states.values())
        return {'collision_free': False, 'failure_reason': 'collision_or_joint_bounds'}
    monkeypatch.setattr(geometry_service, 'check_command_path', check)
    result = runtime.validate_execution({'arm': 'A', 'phase': 'approach_grasp', 'scene_token': 'scene'}, None)
    assert not result['collision_free']
    assert result['target_collision_scope']['policy'] == 'measured_target_retained_during_open_approach'
    assert not states['semantic_instance::target']


@pytest.mark.parametrize('raise_inside', [False, True])
def test_grasp_planning_retains_target_only_in_its_snapshot_and_restores_on_error(monkeypatch, raise_inside):
    from luna_mujoco import geometry_service
    source, _, _, _ = target_world()
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = source.geometry_scene | {'scene_token': 'scene',
        'disabled_obstacle_names': [*source.geometry_scene['disabled_obstacle_names'], 'unrelated_disabled']}
    runtime.geometry_model = source.geometry_model
    original = runtime.geometry_scene.copy()
    request = object()
    def plan(owner, passed, lock_timeout_s):
        from concurrent.futures import ThreadPoolExecutor

        assert owner is runtime and passed is request and lock_timeout_s == 12
        assert runtime._carrying_scope_lock._is_owned()
        snapshot = owner._scene_snapshot_unlocked()
        assert snapshot['disabled_obstacle_names'] == ('unrelated_disabled',)
        assert snapshot['scene_token'] == 'scene'
        assert owner.geometry_scene == original
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(owner._scene_snapshot_unlocked).result() == original
        if raise_inside:
            raise RuntimeError('SDK planning failed')
        return 'planned'
    monkeypatch.setattr(geometry_service.CuroboRuntime, 'plan_grasp_batch_native', plan)
    if raise_inside:
        with pytest.raises(RuntimeError, match='SDK planning failed'):
            runtime.plan_grasp_batch_native(request, lock_timeout_s=12)
    else:
        assert runtime.plan_grasp_batch_native(request, lock_timeout_s=12) == 'planned'
    assert runtime._scene_snapshot_unlocked() == original


def test_grasp_planning_missing_measured_target_fails_before_sdk_query(monkeypatch):
    from luna_mujoco import geometry_service
    source, _, _, _ = target_world()
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene, runtime.geometry_model = source.geometry_scene, {}
    monkeypatch.setattr(geometry_service.CuroboRuntime, 'plan_grasp_batch_native',
        lambda owner, request, lock_timeout_s=120: owner._scene_snapshot_unlocked())
    with pytest.raises(ValueError, match='target collision layers missing'):
        runtime.plan_grasp_batch_native(object())


@pytest.mark.parametrize('raise_inside', [False, True])
def test_grasp_upload_cache_tracks_scope_and_restores_selection_on_next_refresh(monkeypatch, raise_inside):
    from luna_mujoco import geometry_service
    source, _, _, _ = target_world()
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = source.geometry_scene | {'scene_token': 'scene'}
    runtime.geometry_model, runtime.geometry_grids = source.geometry_model, []
    uploads = []
    backend = SimpleNamespace(update_voxel_scene=lambda grids, model, disabled_obstacle_names:
        uploads.append(tuple(disabled_obstacle_names)))
    monkeypatch.setattr(geometry_service, 'sdk_scene_model', lambda model, **kwargs: model)
    def refresh():
        return runtime._refresh_scene_unlocked(arm='A', backend=backend)
    assert refresh() == 'scene'
    def plan(owner, request, lock_timeout_s):
        assert refresh() == 'scene'
        refresh()
        assert uploads[-1] == ()
        if raise_inside:
            raise RuntimeError('SDK planning failed')
    monkeypatch.setattr(geometry_service.CuroboRuntime, 'plan_grasp_batch_native', plan)
    if raise_inside:
        with pytest.raises(RuntimeError, match='SDK planning failed'):
            runtime.plan_grasp_batch_native(object())
    else:
        runtime.plan_grasp_batch_native(object())
    assert refresh() == 'scene'
    refresh()
    original = tuple(source.geometry_scene['disabled_obstacle_names'])
    assert uploads == [original, (), original]
