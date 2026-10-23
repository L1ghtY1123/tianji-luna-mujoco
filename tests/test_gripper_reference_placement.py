"""The placement reference is the TCP origin in world coordinates."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from luna_mujoco.placement import release_candidates, release_endpoint_evidence


def tilted_marker_inputs(arm='A'):
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler('y', 60, degrees=True).as_matrix()
    grasp[:3, 3] = [.4, .3, .7]
    holding = {'arm': arm, 'instance_id': 'held', 'points_tool_m': [[.01, 0., .19]],
               'grasp_support_reference': {'T_world_tcp_at_grasp': grasp.tolist(), 'support_height_m': .6}}
    destination = {'id': 'marker', 'kind': 'marker',
        'points_world_m': [[.3, -.1, .6], [.5, -.1, .6], [.5, .1, .6], [.3, .1, .6]],
        'placement_reference': {'position_world_m': [.4, 0., .6], 'floor_z_m': .6,
                               'region_polygon_world_xy_m': [[.3, -.1], [.5, -.1], [.5, .1], [.3, .1]]}}
    return destination, holding, [[0., 0., .185, .025]]


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_tilted_tcp_origin_uses_only_bound_partition_for_all_yaws(arm):
    destination, holding, spheres = tilted_marker_inputs(arm)
    goals = release_candidates({}, destination, holding, spheres_tool_m=spheres, clearance_m=.01,
                               placement_request={'target_count': 2, 'slot_id': 2})
    assert len(goals) == 18
    positions = set()
    for goal in goals:
        tcp = np.asarray(goal['T_world_tcp'])
        endpoint = (tcp @ [0., 0., .185, 1.])[:3]
        reference = goal['release_reference']
        np.testing.assert_allclose(endpoint[:2], reference['position_world_m'][:2], atol=1e-12)
        np.testing.assert_allclose(reference['tool_reference_point_m'], [0., 0., .185])
        np.testing.assert_allclose(reference['target_reference_world_xy_m'], endpoint[:2])
        assert reference['tcp_frame'] == 'gripper_front_' + arm
        assert reference['position_constraint'] == 'tcp_world_xy'
        assert (tcp[:3, :3] @ spheres[0][:3] + tcp[:3, 3])[2] - spheres[0][3] - .6 == pytest.approx(.012)
        positions.add(tuple(np.round(endpoint[:2], 6)))
    assert positions == {(.4, -.05)}
    assert [goal['yaw_delta_world_rad'] for goal in goals[:18]] == pytest.approx(np.deg2rad(range(0, 360, 20)))


def test_small_yaw_error_at_mount_can_displace_front_tcp():
    destination, holding, spheres = tilted_marker_inputs()
    destination['kind'] = 'bin'
    goal = release_candidates({}, destination, holding, spheres_tool_m=spheres, clearance_m=.01)[0]
    assert release_endpoint_evidence(goal, holding, goal['T_world_tcp'])['at_release_goal']
    actual = np.asarray(goal['T_world_tcp'])
    actual[:3, :3] = Rotation.from_euler('z', 1.5, degrees=True).as_matrix() @ actual[:3, :3]
    evidence = release_endpoint_evidence(goal, holding, actual)
    assert evidence['position_error_m'] > .002
    assert evidence['rotation_error_rad'] < np.deg2rad(2.)
    assert evidence['attachment_floor_gap_m'] == pytest.approx(.012)
    assert evidence['reference_xy_error_m'] > .002
    assert not evidence['at_release_goal']


@pytest.mark.parametrize('arm', [None, 'invalid'])
def test_placement_requires_explicit_valid_arm_for_tool_reference(arm):
    destination, holding, spheres = tilted_marker_inputs()
    if arm is None:
        holding.pop('arm')
    else:
        holding['arm'] = arm
    with pytest.raises(ValueError, match='release_requires_valid_arm'):
        release_candidates({}, destination, holding, spheres_tool_m=spheres, clearance_m=.01)
