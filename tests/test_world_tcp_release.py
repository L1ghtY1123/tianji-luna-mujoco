import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_region_placement import region_inputs

from jev_mujoco.placement import release_candidates, release_endpoint_evidence
from jev_mujoco.placement_layout import partition_region


@pytest.mark.parametrize('kind', ['marker', 'bin'])
@pytest.mark.parametrize('frozen', [False, True])
def test_all_yaws_keep_world_tcp_xy_without_projected_sphere_constraints(kind, frozen):
    target, destination, holding, spheres = region_inputs(kind)
    grasp = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    grasp[:3, :3] = Rotation.from_euler('xy', [25., 60.], degrees=True).as_matrix()
    holding['grasp_support_reference']['T_world_tcp_at_grasp'] = grasp.tolist()
    spheres = [[.12, -.03, -.17, .15]]
    layout = partition_region(destination['placement_reference']['region_polygon_world_xy_m'], 2)
    request = {'target_count': 2, 'slot_id': 2}
    if frozen:
        request['layout'] = layout
    observation = {'objects': [destination, {'id': 'obstacle', 'points_world_m': [[.7, -.2, 1.5]]}]}
    goals = release_candidates(target, destination, holding, spheres_tool_m=spheres,
        clearance_m=.01, observation=observation, placement_request=request)
    assert len(goals) == 18
    expected_xy = layout['cells'][1]['center_world_xy_m']
    for degrees, goal in zip(range(0, 360, 20), goals, strict=True):
        tcp = np.asarray(goal['T_world_tcp'])
        np.testing.assert_allclose((tcp @ [0., 0., .185, 1.])[:2], expected_xy, atol=1e-12)
        expected_rotation = Rotation.from_euler('z', degrees, degrees=True).as_matrix() @ grasp[:3, :3]
        np.testing.assert_allclose(tcp[:3, :3], expected_rotation, atol=1e-12)
        sphere_array = np.asarray(spheres)
        bottom = np.min((sphere_array[:, :3] @ tcp[:3, :3].T + tcp[:3, 3])[:, 2] - sphere_array[:, 3])
        assert bottom - destination['placement_reference']['floor_z_m'] == pytest.approx(.012)
        assert goal['release_reference']['position_constraint'] == 'tcp_world_xy'
        assert release_endpoint_evidence(goal, holding, tcp)['at_release_goal']


def test_release_height_uses_collision_threshold_despite_legacy_requested_height():
    target, destination, holding, spheres = region_inputs()
    goals = release_candidates(target, destination, holding, spheres_tool_m=spheres,
        clearance_m=.008, release_height_m=.015)
    assert len(goals) == 18
    assert all(goal['release_reference']['attachment_floor_gap_m'] == .010 for goal in goals)


@pytest.mark.parametrize('z_error_m, accepted', [(-.001, True), (0., True), (.001, True)])
def test_tracking_tolerance_stays_above_collision_threshold(z_error_m, accepted):
    target, destination, holding, spheres = region_inputs()
    goal = release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)[0]
    actual = np.asarray(goal['T_world_tcp']).copy()
    actual[2, 3] += z_error_m
    evidence = release_endpoint_evidence(goal, holding, actual)
    assert evidence['position_error_m'] < evidence['position_tolerance_m']
    assert evidence['at_release_goal'] is accepted
