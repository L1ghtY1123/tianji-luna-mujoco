import importlib
import importlib.util
from copy import deepcopy

import numpy as np
import pytest


def module():
    name = 'luna_mujoco.robot_skills.placement_zone'
    assert importlib.util.find_spec(name) is not None, 'measured placement zones are required'
    return importlib.import_module(name)


def observation():
    return {'observation_id': 'obs', 'generation': 1, 'objects': [
        {'id': 'area', 'kind': 'marker', 'placement_reference': {
            'observation_id': 'obs', 'floor_z_m': .6, 'support_entity_id': 'table',
            'region_polygon_world_xy_m': [[.2, -.3], [.8, -.3], [.8, .3], [.2, .3]]}},
        {'id': 'cup', 'kind': 'object', 'points_world_m': [[.4, -.03, .6], [.46, .03, .7]]},
        {'id': 'bowl', 'kind': 'object', 'points_world_m': [[.65, -.03, .6], [.7, .03, .7]]}]}


def intent(**kwargs):
    return {'relation': 'on', 'relative': 'none', 'zone': 'whole', 'reference_ids': [],
            'frame': None, 'preference': 'clearance', **kwargs}


def test_left_of_uses_whole_carried_footprint_not_tcp_or_pixel_center():
    api = module()
    zone = api.perceive_placement_zone(observation(), 'area', intent(
        relative='left_of', reference_ids=['cup'], frame='robot_base'))
    # Grasped off-center: the object's footprint lies to the TCP's right (-Y).
    footprint = np.array([[0., -.08], [.02, -.04]])
    keep = api.placement_constraint_mask(np.array([[.5, .06], [.5, .15]]), footprint, zone)
    assert keep.tolist() == [False, True]


def test_between_requires_gap_between_reference_extents():
    api = module()
    zone = api.perceive_placement_zone(observation(), 'area', intent(
        relative='between', reference_ids=['cup', 'bowl']))
    keep = api.placement_constraint_mask(np.array([[.42, 0], [.55, 0], [.75, 0]]),
                                        np.array([[-.02, -.02], [.02, .02]]), zone)
    assert keep.tolist() == [False, True, False]


def test_reference_without_current_depth_does_not_become_free_space():
    api = module()
    obs = observation()
    del obs['objects'][1]['points_world_m']
    with pytest.raises(ValueError, match='placement_reference_geometry_missing'):
        api.perceive_placement_zone(obs, 'area', intent(
            relative='left_of', reference_ids=['cup'], frame='robot_base'))


def test_camera_left_uses_calibrated_camera_axes():
    api = module()
    obs = observation()
    transform = np.eye(4)
    transform[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    obs['views'] = {'2': {'T_world_camera': transform.tolist()}}
    zone = api.perceive_placement_zone(obs, 'area', intent(
        relative='left_of', reference_ids=['cup'], frame='camera_2'))
    keep = api.placement_constraint_mask(np.array([[.5, -.1], [.5, .1]]), np.zeros((1, 2)), zone)
    assert keep.tolist() == [True, False]


def test_container_requires_measured_interior_not_outer_bounds():
    api = module()
    obs = observation()
    obs['objects'][0] = {'id': 'box', 'kind': 'bin', 'points_world_m': [[.2, -.3, .6], [.8, .3, .8]]}
    with pytest.raises(ValueError, match='placement_support_geometry_missing'):
        api.perceive_placement_zone(obs, 'box', intent(relation='inside'))


def test_stale_region_is_not_refreshed_by_changing_timestamp():
    api = module()
    obs = observation()
    obs['objects'][0]['placement_reference']['observation_id'] = 'old'
    before = deepcopy(obs)
    with pytest.raises(ValueError, match='placement_support_geometry_stale'):
        api.perceive_placement_zone(obs, 'area', intent())
    assert obs == before


def test_object_top_uses_measured_surface_and_retains_identity():
    api = module()
    xy = np.stack(np.meshgrid(np.linspace(.3, .6, 31), np.linspace(-.15, .15, 31)), axis=-1).reshape(-1, 2)
    obs = observation()
    obj = {'id': 'block', 'kind': 'object', 'points_world_m': np.c_[xy, np.full(len(xy), .75)].tolist()}
    obs['objects'].append(obj)
    before = deepcopy(obj)
    region = api.measured_object_region(obs, 'block')
    assert region['parent_entity_id'] == 'block'
    assert region['placement_reference']['floor_z_m'] == pytest.approx(.75)
    assert obj == before
    assert region['source'] == 'measured_object_surface'


def test_hole_in_measured_surface_is_not_filled_by_bounding_polygon():
    api = module()
    grid = np.stack(np.meshgrid(np.linspace(.3, .6, 61), np.linspace(-.15, .15, 61)), axis=-1).reshape(-1, 2)
    grid = grid[np.linalg.norm(grid - [.45, 0.], axis=1) > .045]
    obs = observation()
    obs['objects'].append({'id': 'block', 'kind': 'object',
                          'points_world_m': np.c_[grid, np.full(len(grid), .75)].tolist()})
    region = api.measured_object_region(obs, 'block')
    obs['objects'].append(region)
    zone = api.perceive_placement_zone(obs, region['id'], intent())
    keep = api.placement_constraint_mask(np.array([[.45, 0], [.35, .1]]),
                                        np.array([[-.01, -.01], [.01, .01]]), zone)
    assert keep.tolist() == [False, True]
