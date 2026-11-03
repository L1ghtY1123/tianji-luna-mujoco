"""Scene CAD and inferred hidden surfaces cannot enter physical consumers."""
from copy import deepcopy

import numpy as np
import pytest

from luna_mujoco.carrying import CarryingBinding, carrying_geometry
from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.placement import carrying_binding


def observation():
    points = np.array([[x, y, .6] for x in np.linspace(.4, .44, 5)
                       for y in np.linspace(.1, .14, 4)])
    completion = np.vstack((points, points - [0, 0, .1]))
    target = {'id': 'object', 'points_world_m': points,
        'completed_geometry_points_world_m': completion,
        'geometry_completion': {'source': 'measured_surface_and_user_height_prior',
            'observation_id': 'capture', 'measured_point_count': len(points),
            'synthetic_point_count': len(points), 'synthetic_points_are_measured': False}}
    return {'observation_id': 'capture', 'generation': 1, 'objects': [target],
            'arms': {'A': {'tcp': np.eye(4).tolist()}}}


def test_old_valid_height_completion_is_rejected_by_grasp_and_attachment():
    snapshot = observation()
    target = snapshot['objects'][0]
    for consumer in (lambda: DemoAlgorithms._graspgen_object_points(snapshot, target),
                     lambda: carrying_binding(snapshot, 'object', 'A', 'scene', owner_id='hold')):
        with pytest.raises(ValueError, match='synthetic_scene_geometry_unsupported'):
            consumer()


def test_measured_attachment_preserves_every_sample_and_rejects_stored_completion():
    snapshot = observation()
    target = snapshot['objects'][0]
    target.pop('completed_geometry_points_world_m')
    target.pop('geometry_completion')
    value = carrying_binding(snapshot, 'object', 'A', 'scene', owner_id='hold')
    scene = {'scene_token': 'scene', 'observation_id': 'capture', 'generation': 1, 'target_id': 'object'}
    binding = CarryingBinding.from_mapping(value, scene=scene, arm='A')
    spheres = np.array(carrying_geometry(binding)['spheres'])
    points = np.array(value['points_tool_m'])
    np.testing.assert_allclose(points, target['points_world_m'])
    assert np.all((np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=2)
                   - spheres[None, :, 3]).min(axis=1) <= 0)
    invalid = deepcopy(value)
    invalid['measurement_provenance'].update(
        source='registered_rgbd_surface_with_explicit_height_completion',
        measured_point_count=16, synthetic_point_count=4,
        measured_reference_tool_m=points[:16].mean(axis=0).tolist(),
        geometry_completion={'source': 'measured_surface_and_user_height_prior',
                             'synthetic_points_are_measured': False})
    with pytest.raises(ValueError, match='synthetic_scene_geometry_unsupported'):
        CarryingBinding.from_mapping(invalid, scene=scene, arm='A')


def test_container_never_expands_measured_region_from_cad(monkeypatch):
    from luna_mujoco.robot_skills import container_geometry as transfer

    polygon = [[.3, .1], [.5, .1], [.5, .2], [.3, .2]]
    destination = {'id': 'bin', 'placement_reference': {
        'observation_id': 'capture', 'region_polygon_world_xy_m': polygon}}
    scene = {'mapping_diagnostics': {'bin_shell_completions': {'bin': {
        'source': 'known_cad_measured_registration', 'asset_id': 'not-a-real-asset'}}}}
    monkeypatch.setattr(transfer, 'container_occupied_bounds', lambda *a: {'maximum_world_m': [.5, .2, .7]})
    result = transfer.container_transport_region(destination, scene, {'observation_id': 'capture'})
    assert result['polygon_world_xy_m'] == polygon
    assert result['source'] == 'measured_visible_floor_patch'
