"""Observed OBB approximation, model input frames and post-grasp attachment."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_rgbd_algorithms import observed_scene

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.carrying import CarryingBinding, carrying_geometry, held_collision_binding
from luna_mujoco.placement import carrying_binding


def box_observation():
    observation = observed_scene()
    rotation = Rotation.from_euler('z', 31, degrees=True).as_matrix()
    points = np.array(list(product(np.linspace(-.06, .06, 7), [-.025, .025], [-.02, .02])))
    target = observation['objects'][0]
    target['points_world_m'] = points @ rotation.T + [.5, .1, .64]
    target['visible_centroid_world'] = target['points_world_m'].mean(0).tolist()
    target['position'] = target['visible_centroid_world']
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler('xyz', [15, 170, 25], degrees=True).as_matrix()
    tcp[:3, 3] = [.49, .11, .7]
    observation['arms']['A']['tcp'] = tcp.tolist()
    return observation


def test_graspgen_receives_sampled_obb_surface_in_existing_cloud_frame():
    calls = []
    class Client:
        def generate(self, request):
            calls.append(request)
            return {'ok': True, 'model': 'fixture', 'candidates': [
                {'index': 0, 'score': .9, 'T_camera_grasp': np.eye(4).tolist()}]}
    observation = box_observation()
    measured = observation['objects'][0]['points_world_m'].copy()
    result = Algorithms(graspgen_client=Client()).generate_grasps(observation, 'seen-red', arm='A')
    submitted = calls[0]['object_points_camera_m']
    assert len(submitted) == 4096
    obb = result['object_obb']
    transform = np.asarray(obb['T_frame_obb'])
    cloud_frame = np.asarray(result['T_world_cloud'])
    world = submitted @ cloud_frame[:3, :3].T + cloud_frame[:3, 3]
    local = (world - transform[:3, 3]) @ transform[:3, :3]
    half = np.array(obb['half_extents_m'])
    assert np.all(np.abs(local) <= half + 1e-7)
    assert np.all(np.min(np.abs(np.abs(local) - half), axis=1) < 1e-7)
    assert obb['instance_id'] == 'seen-red'
    assert obb['frame'] == 'world'
    assert obb['source'] == 'observed_points_obb_approximation'
    assert result['graspgen_input_source'] == 'observed_obb_mesh_surface_samples'
    np.testing.assert_array_equal(observation['objects'][0]['points_world_m'], measured)
    np.testing.assert_array_equal(calls[0]['object_points_camera_m'], calls[1]['object_points_camera_m'])


def test_attachment_covers_obb_volume_and_binds_actual_tcp_without_replacing_measurements():
    observation = box_observation()
    value = carrying_binding(observation, 'seen-red', 'A', 'map-1', owner_id='owner')
    assert 'obb_tool' in value
    scene = {'scene_token': 'map-1', 'observation_id': 'capture-4', 'generation': 1, 'target_id': 'seen-red'}
    bound = CarryingBinding.from_mapping(value, scene=scene, arm='A')
    geometry = carrying_geometry(bound)
    assert geometry['fit_method'] == 'obb_volume_partition_cover'
    assert 1 <= geometry['sphere_count'] <= 24
    obb = value['obb_tool']
    transform = np.asarray(obb['T_frame_obb'])
    half = np.asarray(obb['half_extents_m'])
    samples = np.array(list(product(*(np.linspace(-h, h, 11) for h in half))))
    points = samples @ transform[:3, :3].T + transform[:3, 3]
    spheres = np.array(geometry['spheres'])
    gaps = np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=-1) - spheres[None, :, 3]
    assert np.all(gaps.min(axis=1) <= -value['uncertainty_m'] + 1e-9)
    tcp = np.array(observation['arms']['A']['tcp'])
    measured = np.array(value['points_tool_m']) @ tcp[:3, :3].T + tcp[:3, 3]
    np.testing.assert_allclose(measured, observation['objects'][0]['points_world_m'])
    assert held_collision_binding(value, scene, 'A', 'seen-red')['obb_tool'] == obb


def test_attachment_rejects_obb_that_does_not_cover_observed_points():
    observation = box_observation()
    value = carrying_binding(observation, 'seen-red', 'A', 'map-1', owner_id='owner')
    assert 'obb_tool' in value
    value['obb_tool']['half_extents_m'] = [.001] * 3
    scene = {'scene_token': 'map-1', 'observation_id': 'capture-4', 'generation': 1, 'target_id': 'seen-red'}
    with pytest.raises(ValueError, match='obb'):
        CarryingBinding.from_mapping(value, scene=scene, arm='A')


def test_perception_attaches_obb_to_confirmed_id_with_original_measurement_provenance():
    from geometry_perception_fixture import GeometryPerception
    from test_perception import FrozenSegmenter, capture_fixture

    capture, masks = capture_fixture()
    result = GeometryPerception(FrozenSegmenter(masks)).observe(capture, source_hint={'kind': 'cube', 'color': 'any'})
    assert result['objects']
    for entity in result['objects']:
        obb = entity['obb']
        assert obb['instance_id'] == entity['id']
        assert obb['observation_provenance'] == entity['observation_provenance']
        assert obb['unobserved_surfaces_are_measured'] is False


def test_cached_obb_cannot_be_reused_for_changed_measurements():
    from luna_mujoco.object_obb import observed_object_obb

    target = box_observation()['objects'][0]
    target['obb'] = observed_object_obb(target)
    target['points_world_m'][0, 0] += .01
    with pytest.raises(ValueError, match='obb_measurement_binding_changed'):
        observed_object_obb(target)


def test_obb_grasp_footprint_and_projection_cover_the_same_box():
    from luna_mujoco.object_obb import held_geometry_points

    observation = box_observation()
    value = carrying_binding(observation, 'seen-red', 'A', 'map-1', owner_id='owner')
    corners = held_geometry_points(value)
    assert corners.shape == (8, 3)
    # The held box remains bound even when TCP axes are tilted.
    tcp = np.asarray(observation['arms']['A']['tcp'])
    world = corners @ tcp[:3, :3].T + tcp[:3, 3]
    original = observation['objects'][0]['points_world_m']
    assert np.all(world.min(0) <= original.min(0) + 1e-9)
    assert np.all(world.max(0) >= original.max(0) - 1e-9)


def test_recorded_float32_tcp_preserves_obb_enclosure_and_attachment():
    from luna_mujoco.object_obb import observed_object_obb, points_digest

    fixture = json.loads((Path(__file__).parent / 'fixtures/obb_float32_fk_boundary.json').read_text())
    observation = box_observation()
    target = observation['objects'][0]
    target.update(id=fixture['obb']['instance_id'], obb=fixture['obb'],
                  points_world_m=np.asarray(fixture['points_world_m']))
    observation['arms']['A']['tcp'] = fixture['T_world_tcp']
    world_obb = observed_object_obb(target)
    value = carrying_binding(observation, target['id'], 'A', 'map-1', owner_id='owner')
    scene = {'scene_token': 'map-1', 'observation_id': observation['observation_id'],
             'generation': observation['generation'], 'target_id': target['id']}
    bound = CarryingBinding.from_mapping(value, scene=scene, arm='A')
    geometry = carrying_geometry(bound)
    tool_obb = value['obb_tool']
    assert tool_obb['frame'] == 'gripper_front'
    assert tool_obb['instance_id'] == target['id']
    assert tool_obb['points_sha256'] == points_digest(value['points_tool_m'])
    assert target['obb'] == world_obb

    # Every corner of the original box, not just the measured subset, stays enclosed.
    half = np.asarray(world_obb['half_extents_m'])
    corners = np.array(list(product(*[[-h, h] for h in half])))
    world_transform = np.asarray(world_obb['T_frame_obb'])
    world_corners = corners @ world_transform[:3, :3].T + world_transform[:3, 3]
    inverse_tcp = np.linalg.inv(fixture['T_world_tcp'])
    tool_corners = world_corners @ inverse_tcp[:3, :3].T + inverse_tcp[:3, 3]
    tool_transform = np.asarray(tool_obb['T_frame_obb'])
    tool_rotation = tool_transform[:3, :3]
    np.testing.assert_allclose(tool_rotation.T @ tool_rotation, np.eye(3), atol=1e-14, rtol=0)
    local_corners = (tool_corners - tool_transform[:3, 3]) @ tool_rotation
    new_half = np.asarray(tool_obb['half_extents_m'])
    assert np.all(np.abs(local_corners) <= new_half + 1e-14)
    assert np.all(new_half >= half - 1e-14)
    assert np.max(new_half - half) < 2e-8
    spheres = np.asarray(geometry['spheres'])
    gaps = np.linalg.norm(tool_corners[:, None] - spheres[None, :, :3], axis=-1) - spheres[None, :, 3]
    assert np.all(gaps.min(axis=1) <= -value['uncertainty_m'] + 1e-9)


def test_obb_transform_does_not_resize_to_accept_uncovered_measurements():
    from luna_mujoco.object_obb import obb_in_tool, observed_object_obb, validate_obb

    target = box_observation()['objects'][0]
    obb = observed_object_obb(target)
    # Rebinding the point digest at a frame conversion must not hide bad geometry.
    changed = target['points_world_m'].copy()
    changed[0] -= [.01, .01, .01]
    converted = obb_in_tool(obb, np.eye(4), changed)
    with pytest.raises(ValueError, match='obb_does_not_cover_observed_points'):
        validate_obb(converted, changed, frame='gripper_front', instance_id=target['id'])
