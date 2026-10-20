import json

import numpy as np
from fastapi.testclient import TestClient

from jev_mujoco.app import create_app
from jev_mujoco.pointcloud_preview import pointcloud_preview
from jev_mujoco.simulation import Simulation


def test_preview_preserves_rejected_samples_without_changing_points():
    points = np.arange(300, dtype=float).reshape(-1, 3) / 1000
    keep = np.ones(100, bool)
    keep[[4, 99]] = False
    result = pointcloud_preview(points, keep, np.arange(100), maximum_points=20)
    indices = result['original_point_indices']
    assert {4, 99}.issubset(indices)
    np.testing.assert_array_equal(result['points_world_m'], points[indices])
    np.testing.assert_array_equal(result['kept_mask'], keep[indices])
    assert result['counts'] == {'input': 100, 'kept': 98, 'rejected': 2}
    assert len(indices) == 20
    json.dumps(result, allow_nan=False)


def test_preview_api_is_cache_only_and_rejects_prior_generation(tmp_path):
    simulation = Simulation()
    class Algorithms:
        generation = simulation.robot_snapshot()['generation']
        def request(self, path):
            assert path == '/perception/preview'
            return {'generation': self.generation, 'scene_token': 'measured', 'points_world_m': []}

    algorithms = Algorithms()
    with TestClient(create_app(simulation=simulation, algorithms=algorithms, data_dir=tmp_path)) as client:
        assert simulation.latest_rgbd() is None
        assert client.get('/api/perception-preview').json()['scene_token'] == 'measured'
        assert simulation.latest_rgbd() is None
        algorithms.generation -= 1
        assert client.get('/api/perception-preview').status_code == 409
