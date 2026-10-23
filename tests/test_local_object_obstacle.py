from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.local_object_obstacle import object_bounds


def test_bounds_keep_only_table_roi_and_cover_known_height():
    points = np.array([[.4, .1, .61], [.45, .14, .63], [1., 1., 2.]])
    bounds = object_bounds(points, [.3, 0.], [.6, .3], .55, .11)
    assert bounds['point_count'] == 2
    assert bounds['minimum_world_m'][2] == pytest.approx(.55)
    assert bounds['maximum_world_m'][2] >= .66


def test_missing_visible_surface_cannot_publish_old_pose():
    with pytest.raises(ValueError, match='local_object_depth_missing'):
        object_bounds(np.array([[1., 1., 2.]]), [.3, 0.], [.6, .3], .55, .11)


def test_dual_uses_official_direct_then_graph_attempt_order():
    from luna_mujoco import backend_builder  # noqa: F401 - initializes the vendor boundary
    from luna_mujoco.dual_planning import ValidatedDualBackend

    backend = object.__new__(ValidatedDualBackend)
    backend._ordered_dual_joints = lambda values: values
    backend._state = lambda values: values
    calls = []
    backend.motion_planner = SimpleNamespace(plan_cspace=lambda *args, **kwargs: calls.append(kwargs))
    request = SimpleNamespace(start_joints_deg={}, goal_joints_deg={}, max_attempts=2)
    with pytest.raises(Exception, match='no collision-free'):
        backend.plan_dual_native(request)
    assert calls == [{'max_attempts': 2, 'enable_graph_attempt': 1}]


def test_default_budget_allows_official_graph_fallback():
    from luna_mujoco.planning_settings import PlanningSettings

    assert PlanningSettings().plan_max_attempts == 2


def test_black_obstacle_uses_wrist_depth_when_head_has_no_surface(monkeypatch):
    from luna_mujoco.demo_algorithms import DemoAlgorithms

    cameras, published = [], []
    def measure(capture, detector, area, appearance, *, camera_id):
        cameras.append(camera_id)
        if camera_id == '3':
            raise ValueError('local_object_depth_missing')
        return {'camera_id': camera_id, 'source': 'wrist_rgbd_object_surface_with_height_prior',
                'bounds': {'minimum_world_m': [.4, -.1, .56], 'maximum_world_m': [.5, 0., .62]}}
    monkeypatch.setattr('luna_mujoco.local_object_obstacle.measure_object_obstacle', measure)
    adapter = object.__new__(DemoAlgorithms)
    adapter.perception = SimpleNamespace(detector=object())
    adapter.request = lambda path, payload: published.append(payload) or {'scene_token': 'new'}
    result = DemoAlgorithms.publish_local_object(adapter,
        {'scene_token': 'old', 'generation': 1, 'target_id': 'pink'},
        {'generation': 1, 'observation_id': 'current'}, 'pink', {}, 'unpacked')
    assert cameras == ['3', '2']
    assert published[0]['camera_id'] == '2'
    assert result['scene']['scene_token'] == 'new'


def test_wrist_obstacle_passes_scene_update_without_remapping_background():
    from luna_mujoco.scene_update import prepare_scene_update

    scene = {'scene_token': 'old', 'generation': 1, 'observation_id': 'initial', 'target_id': 'pink'}
    background = SimpleNamespace(name='semantic_instance::box')
    identity, grids, model = prepare_scene_update(scene, [background],
        {'voxel': {background.name: {}}, 'cuboid': {'table': {}, 'local_object::pink': {}}},
        {'operation': 'local_object', 'scene_token': 'old', 'generation': 1, 'target_id': 'pink',
         'replaces_entity_id': 'pink', 'exclusion_state': 'released', 'camera_id': '2',
         'source': 'wrist_rgbd_object_surface_with_height_prior', 'after_capture_id': 'current',
         'bounds': {'minimum_world_m': [.4, -.1, .56], 'maximum_world_m': [.5, 0., .62]}})
    assert grids == [background]
    assert 'table' in model['cuboid']
    assert identity['layer_evidence']['camera_id'] == '2'
    assert identity['layer_evidence']['background_remapped'] is False
