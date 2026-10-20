import numpy as np
import pytest
from route_acceptance_score import score


@pytest.mark.parametrize('x,z,passed', [(.3, .62, True), (.7, .62, True), (.3, .8, False)])
def test_final_physical_score_checks_settling_without_imposing_object_xy(x, z, passed):
    transform = np.eye(4)
    transform[:3, 3] = [x, 0., z]
    obj = {'id': 'source-0', 'position': [x, 0., z], 'size': [.04] * 3,
           'T_world_object': transform.tolist()}
    oracle = {'before': {'objects': [obj | {'position': [.2, .2, .62]}]}, 'after': {'objects': [obj]}}
    progress = {'objects': [{'id': 'measured', 'display_id': 1, 'status': 'completed',
        'initial_source': {'position': [.2, .2, .64]}, 'completed_stages': [
            {'stage_index': 0, 'destination_id': 'bin', 'arm': 'B'}]}], 'placement_regions': [
        {'stage_index': 0, 'destination_id': 'bin', 'layout': {'floor_z_m': .6,
            'region_polygon_world_xy_m': [[.1, -.2], [.9, -.2], [.9, .2], [.1, .2]],
            'cells': [{'slot_id': 1, 'polygon_world_xy_m': [[.1, -.2], [.5, -.2], [.5, .2], [.1, .2]]}]}}]}
    result = score(oracle, progress, 1, 'table_box')
    assert result['passed'] is passed
