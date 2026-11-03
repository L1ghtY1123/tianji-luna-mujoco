from copy import deepcopy

import numpy as np
import pytest
from test_demo_planning import PlaceService, placement_inputs

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.robot_skills.placement_resolution import digest
from luna_mujoco.robot_skills.placement_zone import perceive_placement_zone


def setup(monkeypatch, *, container=False):
    observation, scene, holding, request = placement_inputs()
    request['free_placement'] = True
    destination = observation['objects'][-1]
    if container:
        destination['kind'] = 'bin'
        # Measured rim, not a fabricated cavity or CAD shell.
        destination['points_world_m'].extend([[.5, -.3, .9], [.7, -.1, .9]])
    zone = perceive_placement_zone(observation, destination['id'],
                                   {'relation': 'inside' if container else 'on'})
    service = PlaceService()
    algorithm = DemoAlgorithms(curobo_request=service)
    monkeypatch.setattr(algorithm, 'gripper_release_spheres', lambda *args: [[0., 0., 0., .01]])
    assert hasattr(algorithm, 'resolve_placement'), 'placement resolution must be an explicit capability'
    return algorithm, service, observation, scene, holding, request, zone


def test_plan_place_requires_resolution_without_searching_or_planning(monkeypatch):
    algorithm, service, obs, scene, held, request, _ = setup(monkeypatch)
    with pytest.raises(AlgorithmError) as caught:
        algorithm.plan_place(obs, 'seen-red', 'A', scene, held, 'demo-destination', placement_request=request)
    assert caught.value.code == 'placement_resolution_required'
    assert service.calls == []


@pytest.mark.parametrize('container', [False, True])
def test_resolution_computes_poses_without_trajectory_or_motion(monkeypatch, container):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch, container=container)
    result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
                                        placement_request=request, placement_zone=zone)
    assert result['candidates']
    assert {path for path, _ in service.calls} == {'/tools/resolve_release_height'}
    for candidate in result['candidates']:
        pose = np.asarray(candidate['goal']['T_world_tcp'])
        np.testing.assert_array_equal(pose[:3, :3], np.diag([1, -1, -1]))
        assert candidate['goal']['release_reference']['goal_workspace']['reachable']
    assert result['trajectory_checked'] is False


@pytest.mark.parametrize('blocked', [False, True])
def test_configured_xy_is_checked_once_and_never_replaced(monkeypatch, blocked):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    xy = [.6, -.2]
    obs['objects'][-1]['placement_reference'].update(fixed_xy=True, position_world_m=[*xy, .6])
    request.update(free_placement=False, fixed_xy=True, selected_reference_world_xy_m=xy)
    zone['semantic_scope'] = 'entire_region'
    if blocked:
        service.world_boxes = [([.4, -.4, .5], [.8, 0., 1.])]
        with pytest.raises(AlgorithmError) as error:
            algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
                placement_request=request, placement_zone=zone)
        assert error.value.code == 'placement_no_candidate'
    else:
        result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_zone=zone)
        assert result['selection_evidence']['xy_search_method'] == 'configured_xy'
        assert len(result['candidates']) == 1
        np.testing.assert_allclose(result['candidates'][0]['goal']['object_reference_world_m'][:2], xy)
        from luna_mujoco.demo_placement import release_endpoint_evidence
        goal = result['candidates'][0]['goal']
        endpoint = release_endpoint_evidence(goal, held, goal['T_world_tcp'])
        assert endpoint['at_release_goal']
        np.testing.assert_allclose(endpoint['actual_object_reference_world_xy_m'], xy)
    assert len(service.calls) == 1


def test_same_three_phase_planner_accepts_resolved_container_poses(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch, container=True)
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
                                             placement_request=request, placement_zone=zone)
    service.calls.clear()
    result = algorithm.plan_place(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_resolution=resolution)
    assert set(result['phases']) == {'approach', 'place', 'retreat'}
    assert '/tools/plan_place' in [path for path, _ in service.calls]
    assert '/tools/plan_carried_pose' not in [path for path, _ in service.calls]
    assert '/tools/resolve_release_height' not in [path for path, _ in service.calls]
    assert result['placement_request'] == request
    assert result['placement_zone']['intent']['relation'] == 'inside'


@pytest.mark.parametrize('change', ['scene', 'holding', 'candidate', 'minimum_partition_size_m',
                                    'voxel_size_m', 'margin_m', 'sphere_budget', 'verified', 'attached'])
def test_stale_or_modified_resolution_cannot_reach_planner(monkeypatch, change):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
                                             placement_request=request, placement_zone=zone)
    if change == 'scene':
        scene['scene_token'] = 'new-map'
    elif change == 'holding':
        held['points_tool_m'][0][0] += .03
    elif change == 'candidate':
        resolution['candidates'][0]['goal']['T_world_tcp'][0][3] += .05
    elif change in {'verified', 'attached'}:
        held[change] = False
    else:
        held[change] = 12 if change == 'sphere_budget' else .03
    service.calls.clear()
    with pytest.raises(ValueError, match='placement_resolution_changed'):
        algorithm.plan_place(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_resolution=resolution)
    assert service.calls == []


def test_known_occupied_center_is_filtered_and_other_positions_remain(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    obs['objects'].append({'id': 'blocker', 'kind': 'object',
                          'points_world_m': [[.61, -.23, .65], [.65, -.21, .67]]})
    service.world_boxes = [([.61, -.23, .60], [.65, -.21, .70])]
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
                                             placement_request=request, placement_zone=zone)
    assert resolution['candidates']
    assert resolution['selection_evidence']['occupied_candidate_count'] > 0


@pytest.mark.parametrize('selection', [None, {'status': 'success', 'candidate_ids': ['not-approved']}])
def test_signed_resolution_without_candidate_permission_cannot_reach_planner(monkeypatch, selection):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    zone['semantic_scope'] = 'candidate_selection'
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
                                             placement_request=request, placement_zone=zone)
    if selection is not None:
        resolution['semantic_selection'] = selection
    resolution.pop('resolution_id')
    resolution['resolution_id'] = digest(resolution)
    service.calls.clear()
    with pytest.raises(ValueError, match='placement_semantic_selection_required'):
        algorithm.plan_place(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_resolution=resolution)
    assert service.calls == []


def test_fixed_xy_is_not_replaced_when_relative_constraint_rejects_it(monkeypatch):
    algorithm, _, obs, scene, held, request, zone = setup(monkeypatch)
    obs['objects'][-1]['placement_reference'].update(precise_xy=True, xy_tolerance_m=.002)
    request.update(free_placement=False, selected_reference_world_xy_m=[.6, -.2], xy_tolerance_m=.002)
    zone['halfplanes_world_xy'] = [[0., -1., -.1]]  # object must have y >= -.1
    before = deepcopy(request)
    with pytest.raises(AlgorithmError) as error:
        algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
                                    placement_request=request, placement_zone=zone)
    assert error.value.code == 'placement_no_candidate'
    assert request == before
