import numpy as np
import pytest
from test_region_placement import region_inputs

from jev_mujoco.placement import release_candidates, release_endpoint_evidence


@pytest.mark.parametrize('kind', ['bin', 'marker'])
def test_uploaded_sphere_bottom_is_collision_threshold_above_support_for_every_yaw(kind):
    target, destination, holding, spheres = region_inputs(kind)
    destination['points_world_m'] = np.vstack((destination['points_world_m'],
        [[x, -.32, .70] for x in np.linspace(.58, .82, 80)]))
    spheres[0][2] -= .03  # conservative attachment extends below measured support
    goals = release_candidates(target, destination, holding, spheres_tool_m=spheres,
                               clearance_m=.01, map_resolution_m=.01)
    spheres = np.asarray(spheres)
    grasp = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    for goal in goals:
        tcp = np.asarray(goal['T_world_tcp'])
        centers = spheres[:, :3] @ tcp[:3, :3].T + tcp[:3, 3]
        assert np.min(centers[:, 2] - spheres[:, 3]) - .615 == pytest.approx(.012)
        assert goal['release_reference']['nominal_object_floor_gap_m'] == pytest.approx(
            .6 + tcp[2, 3] - grasp[2, 3] - .615)
        assert goal['release_reference']['height_source'] == 'uploaded_attachment_sphere_bottom'


def test_release_feedback_rejects_high_endpoint_and_accepts_arrival():
    target, destination, holding, spheres = region_inputs()
    goal = release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)[0]
    actual = np.asarray(goal['T_world_tcp'])
    actual[2, 3] += .1
    evidence = release_endpoint_evidence(goal, holding, actual)
    assert not evidence['at_release_goal']
    assert evidence['attachment_floor_gap_m'] == pytest.approx(.112)
    actual[2, 3] -= .1
    assert release_endpoint_evidence(goal, holding, actual)['at_release_goal']


def test_release_feedback_rejects_silently_raised_plan():
    target, destination, holding, spheres = region_inputs()
    goal = release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)[0]
    goal['T_world_tcp'][2][3] += .1
    assert not release_endpoint_evidence(goal, holding, goal['T_world_tcp'])['at_release_goal']


def test_release_feedback_requires_uploaded_geometry_not_tcp_alone():
    target, destination, holding, spheres = region_inputs()
    goal = release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)[0]
    goal['release_reference'].pop('attachment_spheres_tool_m', None)
    with pytest.raises(ValueError, match='release_requires_uploaded_attachment_spheres'):
        release_endpoint_evidence(goal, holding, goal['T_world_tcp'])
