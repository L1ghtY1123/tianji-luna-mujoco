"""A stationary measured box retains its known region through partial views."""
from copy import deepcopy

import numpy as np
import pytest
from test_demo_container_tracking import case, module


def example():
    observation, initial, current, detections, entities = case()
    initial_region = {'position_world_m': [.4, -.6, .5], 'floor_z_m': .5,
        'region_polygon_world_xy_m': [[.3, -.8], [.6, -.8], [.6, -.5], [.3, -.5]],
        'source': 'sam3_bin_mask_and_measured_horizontal_floor_patch', 'observation_id': 'initial'}
    observation['objects'] = [{'id': 'right', 'kind': 'bin', 'placement_reference': initial_region}]
    reference = module().reference_with_container_tracks(observation, initial)
    proof = module().container_correspondence(current, detections, reference, entities)
    patch = deepcopy(initial_region) | {'observation_id': 'later',
        'region_polygon_world_xy_m': [[.3, -.8], [.6, -.8], [.6, -.75], [.3, -.75]]}
    result = {**current, 'visual_reference': reference, 'visual_inventory': {'container_correspondence': proof},
        'objects': [{'id': 'right', 'kind': 'bin', 'placement_reference': patch,
            'points_world_m': np.array([[.3, -.8, .5], [.6, -.8, .5]]),
            'identity_evidence': [{'source': module().SOURCE, 'binding_hash': proof['binding_hash']}]}]}
    return result, initial_region, patch


def test_narrow_current_patch_keeps_prior_measured_extent_without_adding_collision_points():
    observation, initial, patch = example()
    obj = observation['objects'][0]
    points = obj['points_world_m'].copy()
    module().retain_container_placement_regions(observation)
    region = obj['placement_reference']
    assert region['region_polygon_world_xy_m'] == initial['region_polygon_world_xy_m']
    assert region['source'] == 'retained_measured_floor_patch'
    assert region['source_observation_id'] == 'initial' and region['observation_id'] == 'later'
    assert region['current_visible_reference'] == patch and not region['full_container_extent_claimed']
    np.testing.assert_array_equal(obj['points_world_m'], points)


@pytest.mark.parametrize('depth_changed', [False, True])
def test_retention_checks_all_visible_anchor_depth_not_sam_boundary_jitter(depth_changed):
    original, initial, current, detections, entities = case()
    original['objects'] = [{'id': 'right', 'kind': 'bin', 'placement_reference': example()[1]}]
    reference = module().reference_with_container_tracks(original, initial)
    # SAM omits 10% of the same visible surface. Independent registered depth
    # distinguishes segmentation jitter from a surface that actually moved.
    omitted = detections[0]['mask'].copy()
    omitted[8:, :] = False
    detections[0]['silhouette_mask'][omitted] = False
    detections[0]['mask'][omitted] = False
    if depth_changed:
        current['views']['1']['depth_m'][omitted] += .01
    proof = module().container_correspondence(current, detections, reference, entities)
    assert proof['candidate_mapping'] == {'0': 'right'}
    observation = {**current, 'visual_reference': reference,
        'visual_inventory': {'container_correspondence': proof}, 'objects': [{'id': 'right', 'kind': 'bin',
            'identity_evidence': [{'source': module().SOURCE, 'binding_hash': proof['binding_hash']}]}]}
    module().retain_container_placement_regions(observation)
    assert ('placement_reference' in observation['objects'][0]) is not depth_changed


@pytest.mark.parametrize('fault', ['moved_depth', 'stationarity_change', 'height_change', 'tampered_region',
    'stale_capture', 'wrong_generation', 'identity_only', 'bad_hash', 'owner', 'no_anchor'])
def test_old_region_never_expands_a_view_without_matching_stationary_evidence(fault):
    observation, _, _ = example()
    reference = observation['visual_reference']; tracks = reference['container_identity_tracks']
    proof = observation['visual_inventory']['container_correspondence']
    if fault == 'moved_depth':
        proof['matches']['0']['depth_residual_p95_m'] = .006
    elif fault == 'stationarity_change':
        proof['matches']['0']['stationary_depth']['within_5mm_fraction'] = .90
    elif fault == 'height_change':
        observation['objects'][0]['placement_reference']['floor_z_m'] += .01
    elif fault == 'tampered_region':
        tracks['containers']['right']['placement_reference']['region_polygon_world_xy_m'][0][0] -= .1
    elif fault == 'stale_capture':
        observation['observation_id'] = 'another'
    elif fault == 'wrong_generation':
        observation['generation'] += 1
    elif fault == 'identity_only':
        observation['objects'][0]['identity_evidence'] = []
    elif fault == 'owner':
        tracks['owner_id'] = 'another-owner'
    elif fault == 'no_anchor':
        tracks['containers']['right'].pop('placement_reference')
    else:
        proof['binding_hash'] = 'incorrect'
    if fault in {'moved_depth', 'stationarity_change'}:
        proof['binding_hash'] = module()._digest({k: v for k, v in proof.items() if k != 'binding_hash'})
        observation['objects'][0]['identity_evidence'][0]['binding_hash'] = proof['binding_hash']
    before = deepcopy(observation['objects'][0]['placement_reference'])
    module().retain_container_placement_regions(observation)
    assert observation['objects'][0]['placement_reference'] == before
