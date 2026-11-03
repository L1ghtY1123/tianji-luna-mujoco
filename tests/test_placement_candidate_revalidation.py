"""Stale feedback permits fresh endpoint checks, never reuse of old proofs."""
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from test_resolve_placement import setup
from test_unified_placement_controller import controller

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.robot_skills import placement_resolution


def sign(resolution):
    resolution.pop('resolution_id', None)
    resolution['resolution_id'] = placement_resolution.digest(resolution)


def accept(resolution):
    resolution['semantic_selection'] = {'status': 'success', 'scope': 'entire_region',
        'candidate_ids': [item['candidate_id'] for item in resolution['candidates']]}
    sign(resolution)


def resolved_setup(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    request['placement_preference'] = 'clearance'
    zone.update(semantic_scope='entire_region', semantic_evidence={'current': True})
    # Exact C5 first resolve/plan gripper values; other geometry uses CPU fixtures.
    obs['arms']['A']['gripper_position_rad'] = .38148022503653994
    previous = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    accept(previous)
    obs['arms']['A']['gripper_position_rad'] = .38194976045478873
    service.calls.clear()
    return algorithm, service, obs, scene, held, request, zone, previous


def inputs(obs, scene, held, request):
    return placement_resolution.resolution_input(obs, 'seen-red', 'A', scene, held,
        'demo-destination', request, PlanningSettings().model_dump())


def test_feedback_only_change_rechecks_prior_xy_and_builds_current_height_proofs(monkeypatch):
    algorithm, service, obs, scene, held, request, zone, previous = resolved_setup(monkeypatch)
    old_digest = placement_resolution.digest(previous)
    with pytest.raises(ValueError, match='placement_resolution_changed'):
        placement_resolution.validate_resolution(previous, inputs(obs, scene, held, request))
    service.resolved_height_increase = .001
    gripper = Mock(wraps=algorithm.gripper_release_spheres)
    monkeypatch.setattr(algorithm, 'gripper_release_spheres', gripper)
    result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone, previous_resolution=previous)
    assert len(service.calls) == len(previous['candidates']) == 3
    gripper.assert_called_once_with(obs, 'A')
    assert {path for path, _ in service.calls} == {'/tools/resolve_release_height'}
    assert 'semantic_selection' not in result
    assert result['resolution_id'] != previous['resolution_id']
    assert placement_resolution.digest(previous) == old_digest
    assert placement_resolution.validate_resolution(result, inputs(obs, scene, held, request))['consistent']
    old = {item['candidate_id']: item for item in previous['candidates']}
    for candidate in result['candidates']:
        prior = old[candidate['candidate_id']]
        pose, old_pose = np.asarray(candidate['goal']['T_world_tcp']), np.asarray(prior['goal']['T_world_tcp'])
        np.testing.assert_array_equal(pose[:2], old_pose[:2])
        assert pose[2, 3] > old_pose[2, 3]
        assert candidate['request'] == prior['request']
        reference = candidate['goal']['release_reference']
        assert reference['height_search']['input_sha256'] != prior['goal']['release_reference']['height_search']['input_sha256']
        assert reference['height_resolution']['scope'] == 'held_ik_release_endpoint'
        assert reference['goal_workspace']['reachable']
        assert reference['object_floor_gap_m'] <= PlanningSettings().place_max_release_gap_m
        assert candidate['payload']['gripper_joint_positions_rad']['A'] == obs['arms']['A']['gripper_position_rad']
    evidence = result['selection_evidence']['candidate_revalidation']
    assert evidence['source_resolution_id'] == previous['resolution_id']
    assert evidence['sampled_candidate_count'] == evidence['accepted_candidate_count'] == 3
    assert evidence['full_grid_fallback'] is False
    assert evidence['wall_s'] >= 0
    obs['arms']['A']['gripper_position_rad'] += 2e-6
    with pytest.raises(ValueError, match='placement_resolution_changed'):
        placement_resolution.validate_resolution(result, inputs(obs, scene, held, request))


@pytest.mark.parametrize('change', ['scene', 'observation', 'generation', 'destination', 'holding',
    'attachment', 'settings', 'preference', 'zone', 'zone_semantics', 'signature', 'permission',
    'missing_permission', 'proof', 'grid', 'same_feedback', 'sub_tolerance_feedback'])
def test_seed_requires_unchanged_complete_binding_and_accepted_original_candidates(monkeypatch, change):
    _, _, obs, scene, held, request, zone, previous = resolved_setup(monkeypatch)
    current = inputs(obs, scene, held, request)
    requests, _ = placement_resolution.candidate_requests(obs, 'seen-red', held, request, zone,
        PlanningSettings().model_dump(), scene=scene)
    if change == 'scene':
        current['scene_token'] = 'new-map'
    elif change == 'observation':
        current['observation_id'] = 'new-capture'
    elif change == 'generation':
        current['generation'] += 1
    elif change == 'destination':
        current['destination']['points_world_m'][0][2] += .001
    elif change == 'holding':
        current['holding']['points_tool_m'][0][0] += .001
    elif change == 'attachment':
        current['carrying_binding']['owner_id'] = 'new-attachment'
    elif change == 'settings':
        current['planning_settings']['place_max_release_gap_m'] += .001
    elif change == 'preference':
        current['placement_request']['placement_preference'] = 'reach'
    elif change == 'zone':
        zone['floor_z_m'] += .001
    elif change == 'zone_semantics':
        zone['semantic_evidence'] = {'current': False}
    elif change == 'signature':
        previous['candidates'][0]['goal']['T_world_tcp'][0][3] += .001
    elif change == 'permission':
        previous['semantic_selection']['candidate_ids'] = []
        sign(previous)
    elif change == 'missing_permission':
        previous.pop('semantic_selection')
        sign(previous)
    elif change == 'proof':
        previous['candidates'][0]['goal']['release_reference'].pop('height_search')
        sign(previous)
    elif change == 'grid':
        requests = []
    elif change == 'same_feedback':
        current['robot_feedback'] = deepcopy(previous['robot_feedback'])
    else:
        current['robot_feedback'] = deepcopy(previous['robot_feedback'])
        current['robot_feedback']['A']['gripper_position_rad'] += .9e-6
    assert placement_resolution.revalidation_candidate_indices(previous, current, zone, requests) == []


@pytest.mark.parametrize('seeds_succeed', [False, True])
def test_seed_checks_precede_original_grid_and_failed_seeds_are_not_repeated(monkeypatch, seeds_succeed):
    algorithm, _, obs, scene, held, request, zone, previous = resolved_setup(monkeypatch)
    template = previous['candidates'][0]
    requests = [dict(template['request'], test_index=index) for index in range(105)]
    previous['candidates'] = [dict(deepcopy(template), candidate_id=f'placement-{index}', request=requests[index])
                              for index in (100, 2)]
    accept(previous)
    monkeypatch.setattr(placement_resolution, 'candidate_requests', lambda *a, **k: (
        requests, {'primary_candidate_count': 96, 'remaining_candidate_count': 9}))
    checked = []

    def resolve_goal(*args, placement_request, **kwargs):
        index = placement_request['test_index']
        checked.append(index)
        assert kwargs['workspace_prefilter'] is True
        if not seeds_succeed and index != 102:
            raise AlgorithmError('held IK failed', code='release_height_no_held_ik_goal', details={})
        return {key: deepcopy(value) for key, value in template.items() if key not in ('candidate_id', 'request')}

    monkeypatch.setattr(algorithm, '_resolve_place_goal', resolve_goal)
    result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone, previous_resolution=previous)
    assert checked == [100, 2] + ([] if seeds_succeed else [i for i in range(105) if i not in (100, 2)])
    evidence = result['selection_evidence']
    assert evidence['tested_candidate_indices'] == checked
    assert evidence['sampled_candidate_count'] == len(checked)
    assert evidence['unsearched_candidate_count'] == 105 - len(checked)
    assert evidence['remaining_grid_searched'] is True  # index 100, despite only two seed queries
    assert evidence['candidate_revalidation']['full_grid_fallback'] is (not seeds_succeed)
    if not seeds_succeed:
        assert [item['candidate_id'] for item in result['candidates']] == ['placement-102']
        assert len(result['candidate_failures']) == 104


@pytest.mark.parametrize('preference', ['decide', 'clearance', 'reach'])
async def test_normal_failed_plan_retains_resolution_for_next_explicit_resolve(tmp_path, monkeypatch, preference):
    algorithm, service, obs, scene, held, request, zone, previous = resolved_setup(monkeypatch)
    runner = controller(tmp_path)
    runner.algorithms = algorithm
    runner.state.update(target_id='seen-red', destination_id='demo-destination', arm='A',
        generation=runner.simulation.robot_snapshot()['generation'])
    runner.state['facts'].update(runner.registry.contracts['plan_place'].requires)
    runner.state['facts'].update(arm_selected=True, placement_zone_ready=True)
    runner.emit = Mock()
    runner.sync_inventory = Mock()
    runner.verify_scene = AsyncMock()
    runner.prepare_attachment = AsyncMock()
    runner.measured_state = lambda context: context['observation']
    runner.task_progress.records = {'seen-red': {'placement_preference': 'clearance'}}
    runner._bind_region = lambda observation: dict(request,
        placement_preference=runner.task_progress.records['seen-red']['placement_preference'])
    context = {'observation': obs, 'scene': scene, 'holding': held, 'placement_zone': zone,
        'placement_resolution': previous, 'planning_settings': PlanningSettings().model_dump()}
    # A real failed current path retains its candidate proposal for an explicit
    # graph-selected resolve, regardless of whether feedback also changed.
    def fail_current_plan(*args, **kwargs):
        raise AlgorithmError('no checked path', code='place_no_valid_candidate', details={'candidate_failures': []})

    monkeypatch.setattr(algorithm, 'plan_place', fail_current_plan)
    with pytest.raises(AlgorithmError, match='no checked path'):
        await runner._step('plan_place', context)
    assert context['placement_resolution'] is previous
    assert 'place_plan' not in context
    assert runner.state['facts']['place_plan_ready'] is False
    assert service.calls == []
    zone['original_requirement'] = 'put onto support'
    async def choose_xy(context, requests, **kwargs):
        return {'judgments': {'placement_xy': {'status': 'proposed', 'value': requests[0].choices[0]}}}
    runner._judge = choose_xy
    result = await runner._step('resolve_placement', context, {'preference': preference})
    assert len(service.calls) == 1
    assert result['selection_evidence']['search_order'] == 'decisions_xy_then_numeric_z'
    assert result is context['placement_resolution']
    assert runner.state['facts']['placement_resolved'] is True
    assert result['semantic_selection']['status'] == 'success'


def test_seed_service_error_does_not_fall_through_to_full_search(monkeypatch):
    algorithm, _, obs, scene, held, request, zone, previous = resolved_setup(monkeypatch)
    request_service = Mock(side_effect=AlgorithmError('service unavailable', code='algorithm_request_timeout'))
    monkeypatch.setattr(algorithm, 'request', request_service)
    with pytest.raises(AlgorithmError, match='service unavailable'):
        algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_zone=zone, previous_resolution=previous)
    assert request_service.call_count == 1
