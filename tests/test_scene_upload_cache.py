from types import SimpleNamespace

import pytest

from luna_mujoco.geometry_service import GeometryRuntime


def test_scene_upload_reuses_only_same_backend_revision_and_exclusion():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'one', 'target_id': 'target'}
    runtime.geometry_grids = []
    runtime.geometry_model = {'cuboid': {}}
    calls = []
    first = SimpleNamespace(update_voxel_scene=lambda *a, **k: calls.append('first'))
    second = SimpleNamespace(update_voxel_scene=lambda *a, **k: calls.append('second'))
    for backend in (first, first, second, first):
        runtime._refresh_scene_unlocked(arm='A', backend=backend)
    assert calls == ['first', 'second']
    runtime.geometry_scene['disabled_obstacle_names'] = []
    runtime._refresh_scene_unlocked(arm='A', backend=first)
    runtime.geometry_scene['scene_token'] = 'two'
    runtime._refresh_scene_unlocked(arm='A', backend=first)
    assert calls == ['first', 'second', 'first', 'first']


def test_failed_upload_is_never_cached():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'one', 'target_id': 'target'}
    runtime.geometry_grids = []
    runtime.geometry_model = {'cuboid': {}}
    calls = []
    def upload(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('upload failed')
    backend = SimpleNamespace(update_voxel_scene=upload)
    with pytest.raises(RuntimeError):
        runtime._refresh_scene_unlocked(arm='A', backend=backend)
    runtime._refresh_scene_unlocked(arm='A', backend=backend)
    runtime._refresh_scene_unlocked(arm='A', backend=backend)
    assert len(calls) == 2
