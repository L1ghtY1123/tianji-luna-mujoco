"""Selection preferences cannot change destinations or replace collision proofs."""
import numpy as np
import pytest

from luna_mujoco.robot_skills.placement_selection import check_goal_workspace, rank_support_candidates


def test_clearance_prefers_unoccupied_candidate_without_changing_available_points():
    candidates = np.array([[.3, .2], [.5, .2], [.7, .2]])
    observation = {'observation_id': 'capture', 'objects': [
        {'id': 'other', 'kind': 'object', 'points_world_m': [[.30, .20, .65], [.32, .22, .68]]},
        {'id': 'target', 'kind': 'object', 'points_world_m': [[.7, .2, .65]]},
        {'id': 'table', 'kind': 'table', 'points_world_m': [[.5, .2, .6]]}]}
    ordered, evidence = rank_support_candidates(candidates, observation, 'target',
        footprint_offsets=[[-.03, -.03], [.03, .03]], floor_z_m=.6,
        preference='clearance', desired_clearance_m=.03)
    assert ordered[0].tolist() == [.5, .2]
    assert set(map(tuple, ordered)) == set(map(tuple, candidates))
    assert evidence['obstacle_ids'] == ['other']
    assert evidence['collision_proof'] is False
    assert evidence['observation_id'] == 'capture'


def test_reach_preference_preserves_reach_order_and_records_occupancy():
    candidates = np.array([[.3, .2], [.5, .2]])
    obs = {'objects': [{'id': 'blocker', 'points_world_m': [[.3, .2, .7]]}]}
    ordered, evidence = rank_support_candidates(candidates, obs, 'held',
        footprint_offsets=[[0, 0]], floor_z_m=.6,
        preference='reach', desired_clearance_m=.03)
    np.testing.assert_array_equal(ordered, candidates)
    assert evidence['candidates'][0]['measured_obstacle_gap_m'] == 0


def test_missing_geometry_does_not_mean_region_is_known_empty():
    ordered, evidence = rank_support_candidates(np.array([[.3, .2]]),
        {'objects': [{'id': 'unseen', 'kind': 'object'}],
         'visual_inventory': {'missing_geometry_ids': ['occluded', 'held']}}, 'held',
        footprint_offsets=[[0, 0]], floor_z_m=.6,
        preference='clearance', desired_clearance_m=.03)
    assert len(ordered) == 1
    assert evidence['missing_geometry_ids'] == ['occluded', 'unseen']
    assert evidence['candidates'][0]['measured_obstacle_gap_m'] is None


def test_workspace_checks_exact_final_tcp_and_rejects_mismatched_evidence():
    from luna_mujoco.algorithms import AlgorithmError
    position = [.3, .2, .7]
    def outside(arm, positions):
        assert arm == 'B' and positions == [position]
        return [{'position_world_m': position, 'reachable': False}]
    with pytest.raises(AlgorithmError, match='outside') as result:
        check_goal_workspace(outside, 'B', position)
    assert result.value.code == 'place_goal_workspace_rejected'
    with pytest.raises(AlgorithmError) as result:
        check_goal_workspace(lambda *_: [{'position_world_m': [0, 0, 0], 'reachable': True}], 'B', position)
    assert result.value.code == 'algorithm_invalid_response'


def test_support_preference_is_typed_and_does_not_change_destination():
    from luna_mujoco.runtime.controller import DestinationArguments
    arguments = DestinationArguments(destination_id='table', destination_kind='support_region',
                                     placement_preference='clearance')
    assert arguments.destination_id == 'table'
    with pytest.raises(ValueError):
        DestinationArguments(destination_id='table', destination_kind='support_region',
                             placement_preference='ignore_obstacles')
