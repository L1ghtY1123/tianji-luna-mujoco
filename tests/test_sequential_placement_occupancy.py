"""The next placement excludes current measured objects, including earlier deliveries."""
from copy import deepcopy

import numpy as np
import pytest
from test_resolve_placement import setup

from luna_mujoco.algorithms import AlgorithmError


def resolve(algorithm, obs, scene, held, request, zone):
    result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    return [item['request'] for item in result['candidates']], result['selection_evidence']


def retained_scene(scene):
    scene['collision_layer_bounds'] = {key: scene[key] for key in ('scene_token', 'observation_id', 'generation')}
    scene['collision_layer_bounds']['layers'] = {'semantic_instance::already-delivered': {
        'kind': 'voxel', 'enabled': True, 'source': 'native_surface_voxel_cell_bounds',
        'bounds_source': 'all_native_surface_cells_before_preview_sampling', 'cell_count': 100,
        'voxel_size_m': .02, 'minimum_world_m': [.3, -.5, .65], 'maximum_world_m': [.9, .2, .75]}}
    scene['inventory_map'] = {'entities': {'already-delivered': {'status': 'retained',
        'last_observation_id': 'previous-placement'}}}


def test_occluded_delivered_object_remains_occupied_from_retained_native_map(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    retained_scene(scene)
    service.world_boxes = [([.3, -.5, .60], [.9, .2, .75])]
    with pytest.raises(AlgorithmError) as error:
        resolve(algorithm, obs, scene, held, request, zone)
    assert error.value.code == 'placement_no_candidate'
    assert error.value.details['selection_evidence']['occupied_candidate_count'] > 0
    assert error.value.details['selection_evidence']['retained_occupancy_ids'] == ['already-delivered']
    assert {payload['scene_token'] for _, payload in service.calls} == {scene['scene_token']}


def test_stale_map_bound_cannot_be_used_as_current_occupancy(monkeypatch):
    algorithm, _, obs, scene, held, request, zone = setup(monkeypatch)
    retained_scene(scene)
    scene['collision_layer_bounds']['scene_token'] = 'previous-map'
    with pytest.raises(AlgorithmError) as error:
        resolve(algorithm, obs, scene, held, request, zone)
    assert error.value.code == 'planning_stale_scene'


def test_large_occupied_region_rejects_footprint_even_with_no_points_inside_it(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    # Four measured outer corners surround the entire destination. Point-in-footprint
    # alone misses every point and incorrectly treats the interior as unoccupied.
    obs['objects'].append({'id': 'large-object', 'kind': 'object',
        'points_world_m': [[.3, -.5, .7], [.9, -.5, .7], [.9, .2, .7], [.3, .2, .7]]})
    service.world_boxes = [([.3, -.5, .60], [.9, .2, .7])]
    with pytest.raises(AlgorithmError) as error:
        resolve(algorithm, obs, scene, held, request, zone)
    assert error.value.code == 'placement_no_candidate'
    assert error.value.details['selection_evidence']['occupied_candidate_count'] > 0


def test_three_deliveries_use_current_measured_occupancy_and_distinct_clear_spaces(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    original = deepcopy(held)
    # A measured surface with room for three items, without preallocated slots.
    polygon = [[.3, -.4], [.85, -.4], [.85, .2], [.3, .2]]
    obs['objects'][-1]['placement_reference']['region_polygon_world_xy_m'] = polygon
    zone['polygon_world_xy_m'] = polygon
    request['layout']['cells'][0]['polygon_world_xy_m'] = polygon
    request['layout']['region_polygon_world_xy_m'] = polygon
    placed = []
    for index in range(3):
        obs['observation_id'] = 'after-' + str(index)
        scene['observation_id'] = obs['observation_id']
        scene['scene_token'] = 'map-' + str(index)
        held = deepcopy(original)
        held['attachment_evidence']['scene_token'] = scene['scene_token']
        zone['observation_id'] = obs['observation_id']
        obs['objects'][-1]['placement_reference']['observation_id'] = obs['observation_id']
        requests, evidence = resolve(algorithm, obs, scene, held, request, zone)
        pose = np.asarray(held['grasp_support_reference']['T_world_tcp_at_grasp'], float).copy()
        pose[:2, 3] = requests[0]['selected_reference_world_xy_m']
        points = np.asarray(held['points_tool_m']) @ pose[:3, :3].T + pose[:3, 3]
        points[:, 2] += zone['floor_z_m'] + .01 - points[:, 2].min()
        low, high = points[:, :2].min(0), points[:, :2].max(0)
        for old_low, old_high in placed:
            gap = np.maximum(np.maximum(old_low - high, low - old_high), 0.)
            assert np.linalg.norm(gap) >= .005 - 1e-8
        placed.append((low, high))
        if index:
            # A low blocker may be cleared at a higher legal sample; only
            # actual world collisions count, not IK failures or all such XY.
            assert evidence['world_colliding_sample_count'] > 0
        # This represents the next identity-bound measurement, not a production
        # reservation generated from the command. Insert before the destination.
        obs['objects'].insert(-1, {'id': 'delivered-' + str(index), 'kind': 'object',
            'points_world_m': points.tolist()})
        service.world_boxes.append((points.min(axis=0), points.max(axis=0)))
    assert held['points_tool_m'] == original['points_tool_m']
