from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from jev_mujoco.geometry_service import GeometryRuntime, run_operation


@dataclass
class Grid:
    name: str
    pose: list
    feature_tensor: object


def mapped():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'old', 'generation': 1, 'observation_id': 'initial',
        'target_id': 'one', 'exclusion_state': 'selected', 'disabled_obstacle_names': ('semantic_instance::one',)}
    runtime.geometry_grids = [Grid('semantic_instance::' + name, [x, 0, 0, 1, 0, 0, 0], np.ones(2))
                              for name, x in [('one', 0), ('two', 1)]]
    runtime.geometry_model = {'voxel': {grid.name: {'pose': grid.pose, 'centers_base_m': [[grid.pose[0], 0, 0]]}
                                      for grid in runtime.geometry_grids}, 'cuboid': {'table': {'pose': [0]}}}
    runtime.observed_free_guard = SimpleNamespace(scene_token='old', observation_id='initial', generation=1,
        views={'fixed': 'rays'}, selected_instance_id='one', selected_surface_world_m=[[0, 0, 0]])
    runtime.pointcloud_preview = None
    delta = np.eye(4)
    delta[0, 3] = .5
    request = {'scene_token': 'old', 'generation': 1, 'operation': 'transfer', 'target_id': 'one',
        'T_new_world_from_old_world': delta.tolist(), 'after_capture_id': 'released',
        'source': 'release_feedback_rigid_prediction', 'exclusion_state': 'released'}
    return runtime, request


def test_update_moves_only_selected_layer_reuses_tensor_and_publishes_new_token():
    runtime, request = mapped()
    first, other = runtime.geometry_grids
    result = run_operation(runtime, '/perception/update', request)
    assert result['scene_token'] != 'old'
    assert runtime.geometry_grids[0].pose[:3] == [.5, 0, 0]
    assert runtime.geometry_grids[0].feature_tensor is first.feature_tensor
    assert runtime.geometry_grids[1] is other
    assert first.pose[:3] == [0, 0, 0]
    assert runtime.geometry_model['voxel'][first.name]['centers_base_m'] == [[.5, 0, 0]]
    assert result['layer_evidence']['position_source'] == 'action_prediction'
    assert result['disabled_obstacle_names'] == ()


def test_update_does_not_move_or_relabel_free_space_capture():
    runtime, request = mapped()
    rays = runtime.observed_free_guard.views
    run_operation(runtime, '/perception/update', request)
    assert runtime.observed_free_guard.views is rays
    assert runtime.observed_free_guard.observation_id == 'initial'
    assert runtime.geometry_scene['observation_id'] == 'initial'


@pytest.mark.parametrize('field,value', [('scene_token', 'stale'), ('generation', 2), ('target_id', 'absent')])
def test_invalid_update_is_atomic(field, value):
    runtime, request = mapped()
    initial = deepcopy(runtime.geometry_scene)
    with pytest.raises(ValueError):
        run_operation(runtime, '/perception/update', request | {field: value})
    assert runtime.geometry_scene == initial
    assert runtime.geometry_grids[0].pose[:3] == [0, 0, 0]


def test_next_target_exclusion_recomputed_without_remapping():
    runtime, request = mapped()
    moved = run_operation(runtime, '/perception/update', request)
    selected = run_operation(runtime, '/perception/update', {'operation': 'select',
        'scene_token': moved['scene_token'], 'generation': 1, 'target_id': 'two',
        'exclusion_state': 'selected', 'selected_surface_world_m': [[1, 0, 0]]})
    assert selected['disabled_obstacle_names'] == ('semantic_instance::two',)
    assert selected['scene_token'] != moved['scene_token']
    assert runtime.geometry_grids[0].pose[:3] == [.5, 0, 0]


def test_old_planners_cannot_reuse_cached_scene_after_update():
    runtime, request = mapped()
    runtime._scene_snapshot_key_by_arm['A'] = 'old'
    runtime._command_checkers['old'] = object()
    run_operation(runtime, '/perception/update', request)
    assert not runtime._scene_snapshot_key_by_arm and not runtime._command_checkers
