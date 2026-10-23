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
    from luna_mujoco import reachability
    runtime = GeometryRuntime()
    warmed = []
    def create(arm, batch_size=None, ik_only=False, **kwargs):
        assert runtime.planning_settings.ik_num_seeds == 24
        return SimpleNamespace(warmup=lambda: warmed.append((arm, batch_size, ik_only)))
    monkeypatch.setattr(runtime, '_create_real_backend_unlocked', create)
    reachability_warmed = []
    monkeypatch.setattr(reachability, 'PositionReachability', lambda *args:
        SimpleNamespace(warmup=lambda: reachability_warmed.append(args[-1])))
    runtime.warm_start(PlanningSettings(ik_num_seeds=24))
    assert runtime.warmup_status['ready']
    assert len(warmed) == 11
    assert reachability_warmed == ['A', 'B']
    assert len(runtime._reachability_solvers) == 2
    assert ('A', 100, True) in warmed and ('B', 16, False) in warmed
    for arm in ('A', 'B'):
        compound = runtime._compound_backend_unlocked(arm, .01, .03)
        assert compound is runtime._compound_backend_unlocked(arm, .01, .03)
    assert len(warmed) == 11
    assert runtime._collision_cache['mesh'] >= 98
    before = dict(runtime._batch_backends)
    runtime.clear_carrying()
    assert runtime._batch_backends == before
    compound_before = dict(runtime._compound_backends)
    runtime.geometry_scene = {'scene_token': 'new CAD scene'}
    runtime.geometry_model = {'mesh': {str(index): {} for index in range(98)}, 'cuboid': {}}
    runtime._prepare_scene_cache_unlocked()
    assert runtime._batch_backends == before
    assert runtime._compound_backends == compound_before


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
    runtime._refresh_scene_unlocked(arm='A', backend=backend)
    assert runtime.effective_config['A']['plan']['max_batch_size'] == 4


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
