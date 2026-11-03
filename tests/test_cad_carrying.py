"""Registered CAD carrying preserves measurements and covers full model volume."""
from copy import deepcopy

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation
from test_rgbd_algorithms import observed_scene

from luna_mujoco.cad_perception import CadCatalog, point_cloud_fingerprint
from luna_mujoco.carrying import CarryingBinding, carrying_geometry, held_collision_binding
from luna_mujoco.object_obb import held_geometry_points
from luna_mujoco.placement import carrying_binding


def cad_observation(mesh_key='black_core', residual_m=0.):
    geometry = CadCatalog.from_file().geometry(mesh_key)
    observation = observed_scene()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    pose[:3, 3] = [.5, .1, .7]
    vertices = geometry['vertices_m']
    points = vertices[np.linspace(0, len(vertices) - 1, 128, dtype=int)] @ pose[:3, :3].T + pose[:3, 3]
    points[0] += [residual_m, 0., 0.]
    fingerprint = point_cloud_fingerprint(points)
    target = observation['objects'][0]
    target['points_world_m'] = points
    target['cad'] = {
        'asset_id': geometry['asset_id'], 'mesh_key': mesh_key, 'instance_id': target['id'],
        'source_sha256': geometry['source_sha256'], 'mesh_sha256': geometry['mesh_sha256'],
        'T_world_object': pose.tolist(),
        'T_world_source': (pose @ np.linalg.inv(geometry['T_source_canonical_m'])).tolist(),
        'observation_id': observation['observation_id'], 'generation': observation['generation'],
        'pose_validation': {'accepted': True, 'status': 'validated', 'surface': {
            'full_point_count': fingerprint['point_count'], 'full_point_sha256': fingerprint['sha256'],
            'full_max_residual_m': residual_m}},
    }
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler('xyz', [5, 170, -15], degrees=True).as_matrix()
    tcp[:3, 3] = [.49, .11, .72]
    observation['arms']['A']['tcp'] = tcp.tolist()
    scene = {'scene_token': 'map-1', 'observation_id': observation['observation_id'],
             'generation': observation['generation'], 'target_id': target['id']}
    return observation, scene, geometry


def test_registered_cad_keeps_measured_points_and_uses_cad_vertices_for_placement():
    observation, scene, geometry = cad_observation()
    target = observation['objects'][0]
    measured = target['points_world_m'].copy()
    value = carrying_binding(observation, target['id'], 'A', 'map-1', owner_id='owner')
    assert 'obb_tool' not in value
    assert value['cad_tool']['asset_id'] == geometry['asset_id']
    assert value['measurement_provenance']['point_count'] == len(measured)
    np.testing.assert_array_equal(target['points_world_m'], measured)
    tcp = np.asarray(observation['arms']['A']['tcp'])
    recovered = np.asarray(value['points_tool_m']) @ tcp[:3, :3].T + tcp[:3, 3]
    np.testing.assert_allclose(recovered, measured, atol=1e-14)
    held_vertices = held_geometry_points(value)
    world_vertices = held_vertices @ tcp[:3, :3].T + tcp[:3, 3]
    pose = np.asarray(target['cad']['T_world_object'])
    expected = geometry['vertices_m'] @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(world_vertices.min(0), expected.min(0), atol=1e-14)
    np.testing.assert_allclose(world_vertices.max(0), expected.max(0), atol=1e-14)
    assert CarryingBinding.from_mapping(value, scene=scene, arm='A').cad_tool == value['cad_tool']


@pytest.mark.parametrize('mesh_key,watertight', [('black_core', False), ('blue_box', True)])
def test_cad_attachment_spheres_cover_faces_and_anchor_tetrahedra(mesh_key, watertight):
    observation, scene, geometry = cad_observation(mesh_key)
    value = carrying_binding(observation, scene['target_id'], 'A', 'map-1', owner_id='owner')
    binding = CarryingBinding.from_mapping(value, scene=scene, arm='A')
    result = carrying_geometry(binding)
    assert result['source_mesh_watertight'] is watertight
    assert result['fit_method'] == ('cad_triangle_anchor_tetrahedra_cover' if watertight
                                    else 'cad_convex_volume_partition_cover')
    assert result['geometry_source'] == 'registered_cad'
    assert result['covers_model_volume'] is True
    assert result['fills_concavities'] is True
    assert 1 <= result['sphere_count'] <= 24
    mesh = trimesh.Trimesh(vertices=geometry['vertices_m'], faces=geometry['faces'], process=True)
    if not watertight:
        mesh = mesh.convex_hull
    anchor = np.asarray(mesh.vertices).mean(axis=0)
    # Face centres and interior points supplement every vertex, including large triangles.
    samples = np.vstack([mesh.vertices, mesh.triangles_center,
                         .5 * mesh.triangles_center + .5 * anchor, anchor[None]])
    transform = np.asarray(value['cad_tool']['T_tool_object'])
    samples = samples @ transform[:3, :3].T + transform[:3, 3]
    spheres = np.asarray(result['spheres'])
    gaps = np.linalg.norm(samples[:, None] - spheres[None, :, :3], axis=-1) - spheres[None, :, 3]
    assert np.all(gaps.min(axis=1) <= -value['uncertainty_m'] + 1e-10)


@pytest.mark.parametrize('field,value', [
    ('observation_id', 'stale'), ('generation', -1), ('instance_id', 'other'),
    ('source_sha256', 'wrong'), ('mesh_sha256', 'wrong'), ('T_world_source', np.eye(4).tolist()),
    ('pose_validation', {'accepted': False}),
])
def test_cad_binding_rejects_stale_or_changed_registration(field, value):
    observation, scene, _ = cad_observation()
    observation['objects'][0]['cad'][field] = value
    with pytest.raises(ValueError, match='cad_'):
        carrying_binding(observation, scene['target_id'], 'A', 'map-1', owner_id='owner')


@pytest.mark.parametrize('field,value', [('full_point_count', 1), ('full_point_sha256', 'wrong'),
                                        ('full_max_residual_m', .101), ('full_max_residual_m', float('nan'))])
def test_cad_binding_requires_full_measured_residual_evidence(field, value):
    observation, scene, _ = cad_observation()
    observation['objects'][0]['cad']['pose_validation']['surface'][field] = value
    with pytest.raises(ValueError, match='cad_'):
        carrying_binding(observation, scene['target_id'], 'A', 'map-1', owner_id='owner')


def test_cad_residual_expands_sphere_padding_and_preserves_mesh_and_measurements():
    observation, scene, geometry = cad_observation(residual_m=.02)
    value = carrying_binding(observation, scene['target_id'], 'A', 'map-1', owner_id='owner')
    result = carrying_geometry(CarryingBinding.from_mapping(value, scene=scene, arm='A'))
    assert result['measurement_uncertainty_m'] == value['uncertainty_m']
    assert result['verified_surface_residual_max_m'] == .02
    assert result['geometry_padding_m'] == pytest.approx(.021)
    spheres = np.asarray(result['spheres'])
    points = np.asarray(value['points_tool_m'])
    gaps = np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=-1) - spheres[None, :, 3]
    assert np.all(gaps.min(axis=1) <= -result['margin_m'] + 1e-10)
    assert value['cad_tool']['mesh_sha256'] == geometry['mesh_sha256']


def test_cad_holding_rebind_preserves_original_registration_and_rejects_tampering():
    observation, scene, _ = cad_observation()
    value = carrying_binding(observation, scene['target_id'], 'A', 'map-1', owner_id='owner')
    original = deepcopy(value['cad_tool'])
    new_scene = scene | {'observation_id': 'later-observation', 'scene_token': 'map-2'}
    rebound = held_collision_binding(value, new_scene, 'A', scene['target_id'])
    assert rebound['cad_tool'] == original
    CarryingBinding.from_mapping(rebound, scene=new_scene, arm='A')
    rebound['cad_tool']['instance_id'] = 'other'
    assert value['cad_tool'] == original
    with pytest.raises(ValueError, match='cad_'):
        CarryingBinding.from_mapping(rebound, scene=new_scene, arm='A')


def test_cad_attachment_rejects_uncovered_measurements_despite_claimed_pose_residual():
    observation, scene, _ = cad_observation()
    target = observation['objects'][0]
    target['points_world_m'][-1] += [.25, 0., 0.]
    fingerprint = point_cloud_fingerprint(target['points_world_m'])
    target['cad']['pose_validation']['surface']['full_point_sha256'] = fingerprint['sha256']
    value = carrying_binding(observation, scene['target_id'], 'A', 'map-1', owner_id='owner')
    bound = CarryingBinding.from_mapping(value, scene=scene, arm='A')
    with pytest.raises(ValueError, match='cad_attachment_does_not_cover_observed_points'):
        carrying_geometry(bound)


def test_cad_attachment_cannot_mix_obb_or_rebind_changed_measured_points():
    observation, scene, _ = cad_observation()
    value = carrying_binding(observation, scene['target_id'], 'A', 'map-1', owner_id='owner')
    conflicting = value | {'obb_tool': {}}
    with pytest.raises(ValueError, match='cad_and_obb'):
        CarryingBinding.from_mapping(conflicting, scene=scene, arm='A')
    value['points_tool_m'][0][0] += .001
    with pytest.raises(ValueError, match='cad_holding_measurement_binding_changed'):
        CarryingBinding.from_mapping(value, scene=scene, arm='A')


@pytest.mark.parametrize('mesh_key', ['black_core', 'blue_box'])
def test_cad_placement_support_vertices_preserve_extents_across_rotations(mesh_key):
    observation, scene, geometry = cad_observation(mesh_key)
    value = carrying_binding(observation, scene['target_id'], 'A', 'map-1', owner_id='owner')
    support = held_geometry_points(value)
    assert len(support) < len(geometry['vertices_m']) / 10
    transform = np.asarray(value['cad_tool']['T_tool_object'])
    raw = geometry['vertices_m'] @ transform[:3, :3].T + transform[:3, 3]
    rotations = Rotation.random(36, random_state=42).as_matrix()
    for rotation in rotations:
        actual, expected = support @ rotation.T, raw @ rotation.T
        np.testing.assert_allclose(actual.min(0), expected.min(0), atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual.max(0), expected.max(0), atol=1e-12, rtol=0)
