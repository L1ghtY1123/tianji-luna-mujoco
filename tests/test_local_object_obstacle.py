from types import SimpleNamespace

import numpy as np
import pytest


def test_plain_black_object_prompt_recovers_surface_when_qualified_prompts_are_empty(monkeypatch):
    from luna_mujoco.local_object_obstacle import measure_object_obstacle

    size = (10, 10)
    rgb = np.zeros((*size, 3), np.uint8)
    view = {'rgb': rgb, 'depth_m': np.ones(size), 'valid_depth': np.ones(size, bool)}
    calls = []
    def segment(image, prompt, timestamp):
        calls.append(prompt)
        return [{'mask': np.ones(size, bool), 'score': .9}] if prompt == 'black object' else []
    monkeypatch.setattr('luna_mujoco.local_object_obstacle.own_robot_surface_pixels',
        lambda capture, masks: {'status': 'measured', 'masks': {'3': np.zeros(size, bool)}})
    monkeypatch.setattr('luna_mujoco.local_object_obstacle.deproject_view',
        lambda view, mask: np.tile([.5, 0., .59], (int(mask.sum()), 1)))
    result = measure_object_obstacle({'views': {'3': view}, 'time': 1., 'generation': 1,
        'observation_id': 'frozen'}, SimpleNamespace(segment=segment),
        {'region_polygon_world_xy_m': [[.3, -.2], [.7, -.2], [.7, .2], [.3, .2]], 'floor_z_m': .57},
        'unpacked')
    assert len(result['points_world_m']) == 100
    assert result['mask'].sum() == 100
    assert 'bounds' not in result
    assert 'black object' in calls


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


def test_black_measurement_uses_wrist_depth_without_publishing_map(monkeypatch):
    from luna_mujoco.demo_algorithms import DemoAlgorithms

    cameras = []
    def measure(capture, detector, area, appearance, *, camera_id):
        cameras.append(camera_id)
        if camera_id == '3':
            raise ValueError('local_object_depth_missing')
        return {'camera_id': camera_id, 'source': 'current_rgbd_object_surface',
                'points_world_m': np.tile([.4, -.1, .62], (30, 1))}
    monkeypatch.setattr('luna_mujoco.local_object_obstacle.measure_object_obstacle', measure)
    adapter = object.__new__(DemoAlgorithms)
    adapter.perception = SimpleNamespace(detector=object())
    adapter.request = lambda *args: pytest.fail('measurement cannot publish geometry')
    result = adapter.measure_local_object({'generation': 1, 'observation_id': 'current'}, {}, 'unpacked')
    assert cameras == ['3', '2']
    assert result['camera_id'] == '2'
    assert not hasattr(adapter, 'publish_local_object')


def test_local_measurement_retains_high_measured_points_and_exact_filtered_mask(monkeypatch):
    from luna_mujoco.local_object_obstacle import measure_object_obstacle

    shape = (10, 10)
    view = {'rgb': np.zeros((*shape, 3), np.uint8), 'depth_m': np.ones(shape),
            'valid_depth': np.ones(shape, bool)}
    points = np.tile([.4, 0., .9], (100, 1))
    points[60:] = [4., 0., .9]
    detector = SimpleNamespace(segment=lambda *a: [{'mask': np.ones(shape, bool), 'score': .9}])
    monkeypatch.setattr('luna_mujoco.local_object_obstacle.own_robot_surface_pixels',
        lambda capture, masks: {'status': 'measured', 'masks': {'3': np.zeros(shape, bool)}})
    monkeypatch.setattr('luna_mujoco.local_object_obstacle.deproject_view',
        lambda view, mask: points[mask.ravel()])
    result = measure_object_obstacle({'views': {'3': view}, 'time': 1., 'generation': 1,
        'observation_id': 'current'}, detector,
        {'region_polygon_world_xy_m': [[.3, -.2], [.5, -.2], [.5, .2], [.3, .2]], 'floor_z_m': .5},
        'wrapped')
    np.testing.assert_array_equal(result['points_world_m'], points[:60])
    np.testing.assert_array_equal(np.flatnonzero(result['mask']), np.arange(60))
    assert result['source'] == 'current_rgbd_object_surface'
    assert 'bounds' not in result and 'height_prior_m' not in result


def test_invalidation_request_has_no_invented_geometry():
    from luna_mujoco.demo_algorithms import DemoAlgorithms

    calls = []
    adapter = DemoAlgorithms()
    adapter.request = lambda path, payload: calls.append((path, payload)) or {'scene_token': 'new'}
    result = adapter.invalidate_scene_entity({'scene_token': 'old', 'generation': 1}, 'object', 'action')
    assert result['scene_token'] == 'new'
    assert calls == [('/perception/update', {'scene_token': 'old', 'generation': 1,
        'target_id': 'object', 'operation': 'invalidate', 'exclusion_state': 'released',
        'action_id': 'action', 'source': 'mock_vla'})]
