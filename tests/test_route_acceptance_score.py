from copy import deepcopy

import numpy as np
import pytest
from route_acceptance_score import score


def sample(x=.3, z=.62):
    transform = np.eye(4)
    transform[:3, 3] = [x, 0., z]
    obj = {'id': 'source-0', 'position': [x, 0., z], 'size': [.04] * 3,
           'T_world_object': transform.tolist()}
    oracle = {'before': {'objects': [obj | {'position': [.2, .2, .62]}]}, 'after': {'objects': [obj]}}
    progress = {'objects': [{'id': 'measured', 'display_id': 1, 'status': 'completed',
        'initial_source': {'position': [.2, .2, .64]}, 'transfers': [
            {'transfer_index': 0, 'destination_id': 'bin', 'arm': 'B', 'observation_id': 'after'}]}], 'placement_regions': [
        {'transfer_index': 0, 'destination_id': 'bin', 'layout': {'floor_z_m': .6,
            'region_polygon_world_xy_m': [[.1, -.2], [.9, -.2], [.9, .2], [.1, .2]],
            'cells': [{'slot_id': 1, 'polygon_world_xy_m': [[.1, -.2], [.5, -.2], [.5, .2], [.1, .2]]}]}}]}
    return oracle, progress


@pytest.mark.parametrize('x,z,passed', [(.3, .62, True), (.7, .62, True), (.3, .8, False)])
def test_final_physical_score_checks_settling_without_imposing_object_xy(x, z, passed):
    oracle, progress = sample(x, z)
    result = score(oracle, progress, 1)
    assert result['passed'] is passed


def test_three_explicit_verified_transfers_are_supported_without_a_route_enum():
    oracle, progress = sample()
    progress['objects'][0]['transfers'] = [
        {'transfer_index': index, 'destination_id': destination, 'arm': arm, 'observation_id': f'frame-{index}'}
        for index, (destination, arm) in enumerate([('buffer', 'A'), ('bench', 'B'), ('bin', 'A')])]
    progress['placement_regions'][0]['transfer_index'] = 2
    before = deepcopy(progress)
    result = score(oracle, progress, 1, required_transfer_count=3)
    assert result['passed']
    assert result['checks'][0]['transfer_arms'] == ['A', 'B', 'A']
    assert progress == before
    assert not score(oracle, progress, 1, required_transfer_count=2)['passed']


@pytest.mark.parametrize('fault', ['missing_observation', 'duplicate_observation', 'missing_slot', 'missing_physical_object'])
def test_physical_score_rejects_incomplete_transfer_evidence(fault):
    oracle, progress = sample()
    if fault == 'missing_observation':
        del progress['objects'][0]['transfers'][0]['observation_id']
    elif fault == 'duplicate_observation':
        progress['objects'][0]['transfers'].append({**progress['objects'][0]['transfers'][0], 'transfer_index': 1})
    elif fault == 'missing_slot':
        progress['placement_regions'][0]['layout']['cells'] = []
    else:
        oracle['after']['objects'] = []
    count = len(progress['objects'][0]['transfers'])
    assert not score(oracle, progress, 1, required_transfer_count=count)['passed']
