from copy import deepcopy

import numpy as np
import pytest
from test_demo_planning import PlaceService, placement_inputs

from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.demo_placement import fixed_release_goal, release_endpoint_evidence
from luna_mujoco.geometry import front_tcp_pose


def precise_inputs():
    observation, scene, holding, request = placement_inputs()
    destination = observation['objects'][-1]
    destination['placement_reference'].update(precise_xy=True, xy_tolerance_m=.002)
    request.update(free_placement=False, selected_reference_world_xy_m=[.6, -.2], xy_tolerance_m=.002)
    return observation, scene, holding, request


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_precise_xy_places_measured_object_reference_instead_of_finger_front(arm):
    observation, _, holding, request = precise_inputs()
    holding['arm'] = arm
    # The full attachment includes synthetic completion; its centroid must not
    # replace the separately recorded centroid of the measured surface.
    measured = np.array([.04, -.025, .14])
    holding['measurement_provenance'] = {'measured_reference_tool_m': measured.tolist()}
    before = deepcopy((observation['objects'][-1], holding, request))
    goal = fixed_release_goal(observation['objects'][-1], holding, request,
        holding['attachment_evidence']['attachment_spheres_tool_m'], [[0, 0, 0, .01]],
        clearance_m=.01, observation_id=observation['observation_id'])
    pose = np.asarray(goal['T_world_tcp'])
    np.testing.assert_array_equal(pose[:3, :3], np.asarray(
        holding['grasp_support_reference']['T_world_tcp_at_grasp'])[:3, :3])
    np.testing.assert_allclose((pose[:3, :3] @ measured + pose[:3, 3])[:2], [.6, -.2])
    assert not np.allclose(front_tcp_pose(pose, arm)[:2, 3], [.6, -.2])
    reference = goal['release_reference']
    assert reference['position_constraint'] == 'measured_object_reference_world_xy'
    np.testing.assert_allclose(reference['visible_reference_tool_m'], measured)
    np.testing.assert_allclose(reference['object_reference_world_xy_m'], [.6, -.2])
    assert reference['object_reference_definition'] == 'measured_visible_surface_centroid_not_volume_center'
    assert (observation['objects'][-1], holding, request) == before


def test_precise_place_retains_object_xy_through_height_resolution_and_checks_actual_error(monkeypatch):
    observation, scene, holding, request = precise_inputs()
    service = PlaceService()
    service.resolved_height_increase = .006
    algorithm = DemoAlgorithms(curobo_request=service)
    monkeypatch.setattr(algorithm, 'gripper_release_spheres', lambda *args: [[0., 0., 0., .01]])
    result = algorithm._plan_place_at_reference(observation, 'seen-red', 'A', scene, holding,
                                 'demo-destination', placement_request=request)
    goal = result['goal']
    reference = goal['release_reference']
    np.testing.assert_allclose(reference['object_reference_world_xy_m'], [.6, -.2])
    np.testing.assert_allclose(reference['visible_reference_tool_m'], np.mean(holding['points_tool_m'], axis=0))
    assert reference['height_resolution']['height_increase_m'] == pytest.approx(.006)
    assert release_endpoint_evidence(goal, holding, goal['T_world_tcp'])['at_release_goal']
    actual = np.asarray(goal['T_world_tcp'])
    actual[0, 3] += .003
    evidence = release_endpoint_evidence(goal, holding, actual)
    assert evidence['reference_xy_error_m'] == pytest.approx(.003)
    assert evidence['front_inside_region']
    assert not evidence['at_release_goal']


def test_precise_request_cannot_override_configured_xy():
    observation, _, holding, request = precise_inputs()
    request['selected_reference_world_xy_m'] = [.61, -.2]
    with pytest.raises(ValueError, match='precise_placement_reference_changed'):
        fixed_release_goal(observation['objects'][-1], holding, request,
            holding['attachment_evidence']['attachment_spheres_tool_m'], [[0, 0, 0, .01]],
            clearance_m=.01, observation_id=observation['observation_id'])


@pytest.mark.parametrize('reference', [[np.nan, 0., 0.], [0., 0.]])
def test_precise_reference_rejects_invalid_measured_provenance(reference):
    observation, _, holding, request = precise_inputs()
    holding['measurement_provenance'] = {'measured_reference_tool_m': reference}
    with pytest.raises(ValueError, match='holding_requires_measured_visible_reference'):
        fixed_release_goal(observation['objects'][-1], holding, request,
            holding['attachment_evidence']['attachment_spheres_tool_m'], [[0, 0, 0, .01]],
            clearance_m=.01, observation_id=observation['observation_id'])


@pytest.mark.parametrize('arm', ['A', 'B'])
@pytest.mark.parametrize('reference_flag', ['precise_xy', 'fixed_xy'])
def test_precise_grasp_reach_uses_object_offset_and_exact_xy_instead_of_slot_center(arm, reference_flag):
    from luna_mujoco.demo_grasp import filter_fixed_slot_reach, fixed_slot_reach_context
    from luna_mujoco.planning_settings import PlanningSettings

    observation, _, holding, request = precise_inputs()
    grasp = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    points = np.asarray(holding['points_tool_m']) @ grasp[:3, :3].T + grasp[:3, 3]
    source = next(obj for obj in observation['objects'] if obj['id'] == holding['instance_id'])
    source['points_world_m'] = points.tolist()
    request.update(precise_xy=True, selected_reference_world_xy_m=[.61, -.21])
    request.pop('precise_xy')
    request[reference_flag] = True
    context = fixed_slot_reach_context(observation, holding['instance_id'], arm, request, PlanningSettings())
    np.testing.assert_allclose(context['target_reference_world_xy_m'], [.61, -.21])
    expected_xy = np.array([.61, -.21]) - (points.mean(axis=0) - grasp[:3, 3])[:2]
    mount_z = grasp[2, 3] - points[:, 2].min() + .6 + .01
    wrist = np.r_[expected_xy, mount_z] + grasp[:3, :3] @ context['tcp_to_wrist_tool_m']
    # Isolate the two coordinate interpretations at a tight synthetic bound.
    context.update(shoulder_world_m=wrist.tolist(), maximum_shoulder_wrist_distance_m=.02,
                   approach_offset_m=0.)
    candidate = {'index': 1, 'T_world_tcp': grasp.tolist()}
    accepted, evidence = filter_fixed_slot_reach([candidate], context)
    assert accepted == [candidate]
    np.testing.assert_allclose(evidence['candidate_evidence'][0]['target_mount_world_xy_m'], expected_xy)
    assert filter_fixed_slot_reach([candidate], context | {'precise_xy': False})[0] == []


def test_free_region_release_records_actual_measured_object_reference_without_fixing_xy():
    observation, _, holding, request = precise_inputs()
    destination = observation['objects'][-1]
    destination['placement_reference'].pop('precise_xy')
    destination['placement_reference'].pop('xy_tolerance_m')
    request.update(free_placement=True)
    request.pop('xy_tolerance_m')
    measured = np.array([.01, -.01, .05])
    holding['measurement_provenance'] = {'measured_reference_tool_m': measured.tolist()}
    goal = fixed_release_goal(destination, holding, request,
        holding['attachment_evidence']['attachment_spheres_tool_m'], [[0., 0., 0., .01]],
        clearance_m=.01, observation_id=observation['observation_id'])
    actual = np.asarray(goal['T_world_tcp']).copy()
    actual[0, 3] += .003
    check = release_endpoint_evidence(goal, holding, actual)
    np.testing.assert_allclose(check['actual_object_reference_world_xy_m'],
        (actual[:3, :3] @ measured + actual[:3, 3])[:2])
    assert check['position_tolerance_m'] is None
