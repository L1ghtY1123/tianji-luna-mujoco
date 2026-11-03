from copy import deepcopy

import numpy as np
from test_demo_grasp_reach import inputs

from luna_mujoco.demo_grasp import filter_fixed_slot_reach, fixed_slot_reach_context
from luna_mujoco.planning_settings import PlanningSettings


def test_region_reach_keeps_grasp_when_center_fails_but_measured_near_edge_passes():
    observation, candidates, request = inputs()
    context = fixed_slot_reach_context(observation, 'target', 'A', request, PlanningSettings())
    original = deepcopy(candidates[0])
    assert filter_fixed_slot_reach([original], context)[0] == []
    context['candidate_world_xy_m'] = [[.30, .45], request['layout']['cells'][0]['center_world_xy_m']]
    accepted, evidence = filter_fixed_slot_reach([original], context)
    assert accepted == [original]
    assert evidence['candidate_evidence'][0]['target_reference_world_xy_m'] == [.30, .45]

def test_arm_selection_accepts_measured_region_edge_and_keeps_failure_evidence():
    from luna_mujoco.demo_algorithms import DemoAlgorithms
    destination = {'id': 'area', 'kind': 'marker', 'position': [.7, .2, .6],
                   'placement_reference': {'floor_z_m': .6, 'source': 'rgbd', 'observation_id': 'capture',
                       'region_polygon_world_xy_m': [[.2, .1], [.8, .1], [.8, .3], [.2, .3]]}}
    observation = {'objects': [{'id': 'object', 'kind': 'cube', 'position': [.3, .2, .6]}, destination]}
    def lookup(arm, positions):
        return [{'position_world_m': position, 'reachable': arm == 'A' and position[0] < .4}
                for position in positions]
    result = DemoAlgorithms(workspace_lookup=lookup).evaluate_reachability(observation, 'object', 'area')
    assert result['A']['reachable']
    assert not result['B']['reachable']
    assert result['A']['destination_candidates']
    assert all(row['position_world_m'][0] < .4 for row in result['A']['destination_candidates'])
    assert result['A']['collision_checked'] is False


def test_support_candidates_keep_complete_footprint_inside_rotated_measured_region():
    from luna_mujoco.placement import region_contains_xy
    from luna_mujoco.placement_candidates import support_xy_candidates
    polygon = np.array([[0., .2], [.3, 0.], [.6, .2], [.3, .4]])
    offsets = np.array([[-.04, -.03], [.04, -.03], [.04, .03], [-.04, .03]])
    points = support_xy_candidates(polygon, offsets_world_xy_m=offsets)
    assert 1 < len(points) <= 626
    assert region_contains_xy((points[:, None, :] + offsets).reshape(-1, 2), polygon).all()
    assert len(support_xy_candidates(polygon, offsets_world_xy_m=offsets * 100)) == 0


def test_concave_support_never_fills_unmeasured_notch():
    from luna_mujoco.placement_candidates import support_xy_candidates
    polygon = [[0, 0], [.4, 0], [.4, .1], [.1, .1], [.1, .4], [0, .4]]
    points = support_xy_candidates(polygon, offsets_world_xy_m=[[-.02, -.02], [.02, .02]])
    assert len(points) > 1
    assert not np.any((points[:, 0] > .1) & (points[:, 1] > .1))

def test_bin_floor_rejection_does_not_claim_above_container_pose_is_unreachable():
    from luna_mujoco.demo_algorithms import DemoAlgorithms
    source = {'id': 'object', 'kind': 'cube', 'position': [.4, .3, .6]}
    destination = {'id': 'bin', 'kind': 'bin', 'position': [.5, -.6, .5],
        'placement_reference': {'floor_z_m': .5,
            'region_polygon_world_xy_m': [[.4, -.8], [.6, -.8], [.6, -.4], [.4, -.4]]}}
    queried = []
    def lookup(arm, positions):
        queried.extend(positions)
        return [{'position_world_m': p, 'reachable': p == source['position']} for p in positions]
    result = DemoAlgorithms(workspace_lookup=lookup).evaluate_reachability(
        {'objects': [source, destination]}, 'object', 'bin')
    assert all(row['reachable'] for row in result.values())
    assert all(row['destination']['reachable'] is None for row in result.values())
    assert all(row['feasibility_scope'] == 'source_position_workspace_only' for row in result.values())
    assert queried == [source['position'], source['position']]


def test_limited_release_search_covers_region_instead_of_only_nearest_cluster():
    from luna_mujoco.demo_placement import distributed_release_candidates
    near = np.column_stack((np.linspace(0., .02, 20), np.zeros(20)))
    candidates = np.vstack((near, [[.4, 0.], [.4, .4], [0., .4], [.2, .2]]))
    selected = distributed_release_candidates(candidates, limit=5)
    np.testing.assert_array_equal(selected[0], candidates[0])
    assert len(selected) == 5
    assert len(np.unique(selected, axis=0)) == 5
    for point in ([.4, 0.], [.4, .4], [0., .4], [.2, .2]):
        assert np.any(np.all(selected == point, axis=1))

