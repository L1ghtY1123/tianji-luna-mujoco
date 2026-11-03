"""Archived inferred surfaces are rejected before inference; measured samples survive."""
import hashlib
from copy import deepcopy

import numpy as np
import pytest
from test_demo_planning import GraspClient, grasp_placement
from test_rgbd_algorithms import observed_scene

from luna_mujoco.demo_algorithms import DemoAlgorithms


def completed_scene():
    scene = observed_scene()
    target = scene["objects"][0]
    measured = target["points_world_m"]
    synthetic = measured[[0, 4, 20, 24]].copy()
    synthetic[:, 2] -= .1
    target["completed_geometry_points_world_m"] = np.vstack((measured, synthetic))
    target["geometry_completion"] = {
        "source": "measured_surface_and_user_height_prior",
        "observation_id": scene["observation_id"],
        "height_prior_m": .1, "height_prior_source": "user_specified_100mm",
        "support_z_m": .54, "support_source": "measured_bin_floor",
        "measured_point_count": len(measured), "synthetic_point_count": len(synthetic),
        "synthetic_points_are_measured": False,
    }
    return scene


@pytest.mark.parametrize("arm", ["A", "B"])
def test_old_completion_stops_before_inference_even_with_valid_provenance(arm):
    scene = completed_scene()
    before = deepcopy(scene["objects"][0])
    client = GraspClient()
    with pytest.raises(ValueError, match="synthetic_scene_geometry_unsupported"):
        DemoAlgorithms(graspgen_client=client).generate_grasps(
            scene, "seen-red", arm=arm, placement_request=grasp_placement())
    assert not client.calls
    np.testing.assert_array_equal(scene['objects'][0]['points_world_m'], before['points_world_m'])


@pytest.mark.parametrize("arm", ["A", "B"])
def test_obb_inference_samples_and_measured_reference_remain_separate_and_exact(arm):
    scene = observed_scene()
    target = scene['objects'][0]
    before = deepcopy(target)
    client = GraspClient()
    result = DemoAlgorithms(graspgen_client=client).generate_grasps(
        scene, "seen-red", arm=arm, placement_request=grasp_placement())
    expected = client.calls[0]['object_points_camera_m'].copy()
    assert len(expected) == 4096
    for request in client.calls:
        np.testing.assert_array_equal(request['object_points_camera_m'], expected)
    assert result['cloud']['id'] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert result['cloud']['synthetic_point_count'] == 4096
    assert result['candidate_replay']['graspgen_input_source'] == 'observed_obb_mesh_surface_samples'
    np.testing.assert_array_equal(result['candidate_replay']['measured_points_world_m'], before['points_world_m'])
    assert 'completed_geometry_points_world_m' not in result['candidate_replay']
    np.testing.assert_array_equal(target['points_world_m'], before['points_world_m'])


def test_failed_direction_filter_archives_exact_obb_input():
    class Upward(GraspClient):
        def generate(self, payload):
            self.calls.append(payload)
            return {'ok': True, 'model': 'upward', 'candidates': [
                {'index': 0, 'score': .9, 'T_camera_grasp': np.eye(4).tolist()}]}
    client = Upward()
    pool = DemoAlgorithms(graspgen_client=client).generate_grasps(
        observed_scene(), 'seen-red', arm='A', placement_request=grasp_placement(), planning_settings={'grasp_insertion_m': .01, 'grasp_maximum_tilt_deg': 45., 'grasp_require_away_from_robot': True})
    assert pool['candidates'] == []
    replay = pool['candidate_replay']
    assert replay['graspgen_input_source'] == 'observed_obb_mesh_surface_samples'
    np.testing.assert_array_equal(replay['graspgen_request']['object_points_camera_m'],
                                  client.calls[0]['object_points_camera_m'])
