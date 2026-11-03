"""The bounded support grid must remain available after its first sample fails."""
from copy import deepcopy

import numpy as np
import pytest
from test_resolve_placement import setup

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.robot_skills import placement_resolution


def test_binary_subdivision_keeps_both_sides_and_compensates_grasp_offset():
    from luna_mujoco.placement_candidates import subdivision_xy_candidates
    polygon = [[0., 0.], [4., 0.], [4., 2.], [0., 2.]]
    offset = np.array([[.1, -.1], [.3, .1]])
    candidates = subdivision_xy_candidates(polygon, offsets_world_xy_m=offset)
    np.testing.assert_allclose(candidates[:3], [[1.8, 1.], [.8, 1.], [2.8, 1.]])
    assert len(candidates) == 15
    assert len({tuple(x) for x in candidates}) == 15


def test_resolver_keeps_three_clear_centers_without_checking_all_candidates(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    zone['semantic_scope'] = 'entire_region'
    result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    assert len(result['candidates']) == 3
    evidence = result['selection_evidence']
    assert evidence['xy_search_method'] == 'long_axis_binary_subdivision'
    assert evidence['sampled_candidate_count'] == 3
    assert evidence['maximum_xy_candidates'] == 15
    assert len(service.calls) == 3


@pytest.mark.parametrize('first_pass_succeeds', [False, True])
def test_resolver_only_searches_remaining_grid_when_first_pass_has_no_valid_endpoint(monkeypatch, first_pass_succeeds):
    algorithm, _, obs, scene, held, request, zone = setup(monkeypatch)
    baseline = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)['candidates'][0]
    requests = [dict(baseline['request'], test_candidate_index=i) for i in range(105)]
    monkeypatch.setattr(placement_resolution, 'candidate_requests', lambda *a, **k: (
        requests, {'primary_candidate_count': 96, 'remaining_candidate_count': 9}))
    checked = []

    def resolve_goal(*args, placement_request, **kwargs):
        index = placement_request['test_candidate_index']
        checked.append(index)
        if not first_pass_succeeds and index != 100:
            raise AlgorithmError('held IK failed', code='release_height_no_held_ik_goal', details={})
        return {key: deepcopy(value) for key, value in baseline.items() if key not in ('candidate_id', 'request')}

    monkeypatch.setattr(algorithm, '_resolve_place_goal', resolve_goal)
    result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    assert checked == list(range(12 if first_pass_succeeds else 105))
    evidence = result['selection_evidence']
    assert evidence['sampled_candidate_count'] == len(checked)
    assert evidence['remaining_grid_searched'] is (not first_pass_succeeds)
    assert evidence['unsearched_candidate_count'] == (93 if first_pass_succeeds else 0)
    if not first_pass_succeeds:
        assert [item['candidate_id'] for item in result['candidates']] == ['placement-100']
        assert len(result['candidate_failures']) == 104
