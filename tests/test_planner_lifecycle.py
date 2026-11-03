from types import SimpleNamespace

import pytest

from luna_mujoco.geometry_service import GeometryRuntime
from luna_mujoco.planning_settings import PlanningSettings


def test_feedback_changes_retain_resident_backend_for_public_geometry_update():
    runtime = GeometryRuntime()
    destroyed = []
    backend = SimpleNamespace(destroy=lambda: destroyed.append(True))
    with runtime.carrying_scope(None, {"A": 0., "B": 0.}):
        runtime._backends['A'] = backend
    with runtime.carrying_scope(None, {"A": 3e-19, "B": -2e-19}):
        assert runtime._backends['A'] is backend
        assert runtime.gripper_feedback == {"A": 0., "B": 0.}
    with runtime.carrying_scope(None):
        assert runtime._backends['A'] is backend
    with runtime.carrying_scope(None, {"A": .1, "B": 0.}):
        assert runtime._backends['A'] is backend
    assert destroyed == []


def test_reset_retains_resident_solver_but_clears_attachment_and_world():
    runtime = GeometryRuntime()
    destroyed = []
    opened = SimpleNamespace(destroy=lambda: destroyed.append('open'))
    cleared = []
    opened.carrying_link = 'jev_carried_object_A'
    opened.resident_geometry = SimpleNamespace(detach=lambda backend: cleared.append(backend))
    with runtime.carrying_scope(None, {"A": 0., "B": 0.}):
        runtime._backends['A'] = opened
        runtime._scene_snapshot_key_by_arm['old'] = object()
        runtime._command_checkers['old'] = object()
    runtime.geometry_scene = {'scene_token': 'old'}
    runtime.clear_carrying()
    assert runtime._backends['A'] is opened
    assert destroyed == [] and cleared == [opened]
    assert not runtime._scene_snapshot_key_by_arm
    assert not runtime._command_checkers
    assert runtime.geometry_scene is None
    with pytest.raises(ValueError, match='build a registered'):
        runtime._scene_snapshot_unlocked()
    runtime.clear_carrying()
    assert runtime._backends['A'] is opened
    assert destroyed == []


def test_startup_warms_cached_instances_with_saved_settings(monkeypatch):
    from luna_mujoco import geometry_service, reachability
    runtime = GeometryRuntime()
    warmed = []
    created = []
    memory_reads = []
    def memory_snapshot():
        memory_reads.append(True)
        return {'allocated_bytes': len(memory_reads) * 100, 'reserved_bytes': 2000,
                'free_bytes': 3000, 'total_bytes': 5000}
    monkeypatch.setattr(geometry_service, 'cuda_memory_snapshot', memory_snapshot)
    def create(arm, batch_size=None, ik_only=False, **kwargs):
        assert runtime.planning_settings.ik_num_seeds == 24
        created.append((arm, batch_size, ik_only))
        return SimpleNamespace(warmup=lambda: warmed.append((arm, batch_size, ik_only)), destroy=lambda: None)
    monkeypatch.setattr(runtime, '_create_real_backend_unlocked', create)
    reachability_warmed = []
    monkeypatch.setattr(reachability, 'PositionReachability', lambda *args:
        SimpleNamespace(warmup=lambda: reachability_warmed.append(args[-1])))
    runtime.warm_start(PlanningSettings(ik_num_seeds=24))
    assert runtime.warmup_status['ready']
    assert len(warmed) == 9
    assert reachability_warmed == ['A', 'B']
    assert len(runtime._reachability_solvers) == 2
    assert ('A', 100, True) in warmed and ('B', 2, False) in warmed
    assert all(profile[1] != 16 for profile in runtime.warmup_status['profiles'])
    assert len(runtime.warmup_status['profile_memory']) == 11
    for profile, memory in zip(runtime.warmup_status['profiles'], runtime.warmup_status['profile_memory'], strict=True):
        assert memory['profile'] == profile
        assert memory['allocated_bytes'] > 0
        assert memory['reserved_bytes'] == 2000 and memory['free_bytes'] == 3000
    for arm in ('A', 'B'):
        compound = runtime._compound_backend_unlocked(arm, .01, .03)
        assert compound is runtime._compound_backend_unlocked(arm, .01, .03)
    assert len(warmed) == 9
    # A later explicit larger request still constructs and caches the same
    # public batch backend, without eagerly allocating it at service startup.
    large = runtime._batch_backend_unlocked('A', 16, .01, .03)
    assert large is runtime._batch_backend_unlocked('A', 16, .01, .03)
    assert created.count(('A', 16, False)) == 1
    assert runtime._collision_cache['mesh'] == 4
    assert runtime._collision_cache['cuboid'] == 5
    assert runtime._collision_cache['voxel']['layers'] == 16
    before = dict(runtime._batch_backends)
    runtime.clear_carrying()
    assert runtime._batch_backends == before
    runtime.geometry_scene = {'scene_token': 'new measured scene'}
    runtime.geometry_model = {'voxel': {name: {'dims': [1., 1., 1.], 'voxel_size': .02} for name in ('first', 'second')}}
    runtime._prepare_scene_cache_unlocked()
    assert runtime._batch_backends == before
    runtime.geometry_model = {'voxel': {str(index): {'dims': [1., 1., 1.], 'voxel_size': .02} for index in range(9)}}
    runtime._prepare_scene_cache_unlocked()
    assert runtime._collision_cache['voxel']['layers'] == 16
    assert len(runtime._scene_model['voxel']) == 9
    assert runtime._batch_backends == before
    runtime.geometry_model['voxel'].update({str(index): {'dims': [1., 1., 1.], 'voxel_size': .02}
                                           for index in range(17)})
    runtime._prepare_scene_cache_unlocked()
    assert runtime._collision_cache['voxel']['layers'] == 32
    assert not runtime._batch_backends and not runtime._compound_backends


@pytest.mark.parametrize('backend_name', ['ValidatedSingleBackend', 'ValidatedBatchBackend'])
def test_motion_warmup_includes_official_graph_planner(backend_name):
    from luna_mujoco import backend_builder

    backend = object.__new__(getattr(backend_builder, backend_name))
    calls = []
    backend.motion_planner = SimpleNamespace(warmup=lambda **kwargs: calls.append(kwargs))
    backend.ik_only = False
    backend._warmup_iterations = 1
    backend._warmup_joint_space = False
    backend.warmup()
    assert calls == [{'enable_graph': True, 'num_warmup_iterations': 1}]


def test_reused_backend_reports_its_own_capacity_not_last_constructed():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'scene', 'target_id': 'target'}
    runtime.geometry_model = {'cuboid': {}}
    runtime.geometry_grids = []
    runtime.effective_config = {'A': {'plan': {'max_batch_size': 16}}}
    backend = SimpleNamespace(effective_config={'max_batch_size': 4}, ik_only=False,
                              update_voxel_scene=lambda *args, **kwargs: None)
    runtime._prepare_scene_cache_unlocked()
    runtime._refresh_scene_unlocked(arm='A', backend=backend)
    assert runtime.effective_config['A']['plan']['max_batch_size'] == 4
    assert runtime.effective_config['A']['plan']['collision_cache_capacity'] == {
        'mesh': 4, 'cuboid': 5, 'voxel_layers': 16}


def test_new_geometry_after_startup_is_warmed_once_before_cache_use(monkeypatch):
    from luna_mujoco import geometry_service

    runtime = GeometryRuntime()
    runtime.warmup_status['ready'] = True
    calls = []
    backend = SimpleNamespace(warmup=lambda: calls.append('warm'),
                              destroy=lambda: calls.append('destroy'))
    monkeypatch.setattr(geometry_service, 'build_geometry_backend', lambda *args, **kwargs: backend)
    assert runtime._backend_unlocked('A') is backend
    assert runtime._backend_unlocked('A') is backend
    assert calls == ['warm']


@pytest.mark.parametrize('kind', ['place', 'pose'])
def test_attachment_uses_next_motion_backend_without_unused_ik(monkeypatch, kind):
    from contextlib import nullcontext

    import numpy as np

    runtime = GeometryRuntime()
    backend = SimpleNamespace(carrying_spheres_tool=np.zeros((24, 4)))
    used = []
    monkeypatch.setattr(runtime, 'carrying_scope', lambda *args: nullcontext())
    monkeypatch.setattr(runtime, '_prepare_scene_cache_unlocked', lambda: None)
    monkeypatch.setattr(runtime, '_compound_backend_unlocked', lambda *args: used.append('place') or backend)
    monkeypatch.setattr(runtime, '_backend_unlocked', lambda *args: used.append('pose') or backend)
    monkeypatch.setattr(runtime, '_refresh_scene_unlocked', lambda **kwargs: None)
    monkeypatch.setattr(runtime, '_refresh_peer_collision_unlocked', lambda **kwargs: None)
    monkeypatch.setattr(runtime, '_require_scene_token_unchanged_unlocked', lambda *args: None)
    runtime.effective_config = {'A': {'plan': {'optimizer_attachment': True}}}
    request = SimpleNamespace(arm='A', start_joints_deg=[0.] * 7, other_joints_deg=[0.] * 7)
    binding = SimpleNamespace(state='held', owner_id='run', scene_token='map')
    result = runtime.prepare_carrying(request, binding, {'A': .3, 'B': 0.}, kind)
    assert result['attached'] and used == [kind]
    assert not runtime._ik_batch_backends
