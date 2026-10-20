from copy import deepcopy

import numpy as np
import pytest
from test_visible_placement import measured_inputs

from jev_mujoco.algorithms import AlgorithmError, Algorithms


def setup_plan(monkeypatch, *, valid=(0, 1, 2)):
    target, destination, holding = measured_inputs()
    holding.update(owner_id='held', verified=True, attached=True,
        attachment_evidence={'attachment_spheres_tool_m': [[.02, -.03, -.20, .01]]})
    observation = {'source': 'rgbd', 'observation_id': 'capture', 'generation': 1,
        'objects': [destination], 'arms': {arm: {'joints_rad': [0.] * 7,
        'tcp': np.eye(4).tolist(), 'gripper_position_rad': .4} for arm in ('A', 'B')}}
    scene = {'target_id': target['id'], 'observation_id': 'capture', 'generation': 1, 'scene_token': 'scene'}
    requests, checks, retreats = [], [], []
    items = []
    for index in range(16):
        phase = {'positions_rad': [[0.] * 7, [.01 * (index + 1)] * 7],
            'velocities_rad_s': [[0.] * 7] * 2, 'accelerations_rad_s2': [[0.] * 7] * 2,
            'interpolation_dt_s': .02, 'joint_names': [f'Joint{i}_L' for i in range(1, 8)]}
        items.append({'index': index, 'success': index in valid,
            'approach_success': index in valid, 'place_success': index in valid,
            'phases': {'approach': phase, 'place': phase}, 'status': 'ok' if index in valid else 'unreachable'})
    def request(path, body):
        requests.append((path, body))
        if path == '/tools/solve_ik_batch':
            return {'scene_token': 'scene', 'batch_size': len(body['goal_matrices_world']),
                'items': [{'candidate_index': index, 'success': True}
                          for index in range(len(body['goal_matrices_world']))],
                'attached': True, 'collision_evidence': {'optimizer_attachment': True}}
        assert path == '/tools/plan_place_batch'
        return {'scene_token': 'scene', 'items': deepcopy(items), 'collision_evidence': {'optimizer_attachment': True}}
    algorithms = Algorithms.__new__(Algorithms)
    algorithms.request = request
    def validate(observed, arm, scene, phase, command, binding=None, **kwargs):
        checks.append((phase, observed['arms'][arm]['gripper_position_rad'], binding is not None))
        return {'collision_free': True}
    def retreat(observed, arm, scene, goal, **kwargs):
        retreats.append(deepcopy(observed))
        return {'command': {'positions_rad': [observed['arms'][arm]['joints_rad']] * 2},
                'check': {'collision_free': True}}
    algorithms.validate_execution = validate
    algorithms.plan_retreat = retreat
    args = (observation, target['id'], 'A', scene, holding, destination['id'])
    return algorithms, args, requests, checks, retreats


def test_batch_place_uses_fixed_world_reference_and_closed_attachment_then_open_retreat(monkeypatch):
    algorithms, args, requests, checks, retreats = setup_plan(monkeypatch)
    before = deepcopy(args)
    events = []
    result = algorithms.plan_place(*args, progress=lambda kind, title, data: events.append((kind, data)))
    assert [path for path, _ in requests] == ['/tools/solve_ik_batch', '/tools/plan_place_batch']
    assert len(requests[0][1]['goal_matrices_world']) == 18
    assert len(requests[1][1]['goal_matrices_world']) == 16
    assert requests[0][1]['carrying_binding']['state'] == 'held'
    assert 'goal_position_m' not in requests[1][1] and 'goal_quaternion_wxyz' not in requests[1][1]
    assert requests[1][1]['gripper_joint_positions_rad'] == {'A': .4, 'B': .4}
    assert result['goal']['candidate_index'] == 0
    assert result['goal']['frame'] == 'world'
    assert set(result['phases']) == {'approach', 'place', 'retreat'}
    assert checks == [('approach', .4, True), ('place', .4, True)]
    assert retreats[0]['arms']['A']['gripper_position_rad'] == 0.
    assert before == args
    evidence = result['selection_evidence']
    assert evidence['successful_candidates_in_batch'] == 3
    assert evidence['absolute_yaw_change_deg'] == 0
    assert evidence['retreat_checked'] and not evidence['later_batches_evaluated']
    assert events[-1] == ('placement_candidate_selected', evidence)


def test_infeasible_open_retreat_rejects_only_that_yaw_and_tries_next(monkeypatch):
    algorithms, args, requests, _checks, _retreats = setup_plan(monkeypatch)
    original = algorithms.plan_retreat
    attempts = []
    def retreat(*values, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise AlgorithmError('open gripper collides with bin', code='execution_collision')
        return original(*values, **kwargs)
    algorithms.plan_retreat = retreat
    result = algorithms.plan_place(*args)
    assert result['goal']['candidate_index'] == 1
    assert len(requests) == 2 and len(attempts) == 2
    rejected = next(item for item in result['candidate_failures'] if item['candidate_index'] == 0)
    assert rejected['code'] == 'execution_collision'


def test_complete_place_sends_attached_batch_then_detached_open_retreat(monkeypatch):
    from test_planning_batch import trajectory

    algorithms, args, requests, checks, _ = setup_plan(monkeypatch)
    original = algorithms.request
    algorithms.plan_retreat = Algorithms.plan_retreat.__get__(algorithms)

    def request(path, body):
        if path == '/tools/plan_to_pose':
            requests.append((path, deepcopy(body)))
            return {'scene_token': 'scene', 'attached': False, 'trajectory': trajectory()}
        return original(path, body)

    algorithms.request = request
    result = algorithms.plan_place(*args)
    assert [path for path, _ in requests] == [
        '/tools/solve_ik_batch', '/tools/plan_place_batch', '/tools/plan_to_pose']
    assert requests[1][1]['carrying_binding']['state'] == 'held'
    assert len(requests[1][1]['goal_matrices_world']) == 16
    assert 'carrying_binding' not in requests[2][1]
    assert requests[2][1]['gripper_joint_positions_rad'] == {'A': 0., 'B': .4}
    assert checks == [('approach', .4, True), ('place', .4, True), ('retreat', 0., False)]
    assert set(result['phases']) == {'approach', 'place', 'retreat'}
    assert args[4]['attached'] is True
    assert args[0]['arms']['A']['gripper_position_rad'] == .4


def test_service_error_during_retreat_is_not_hidden_as_an_infeasible_candidate(monkeypatch):
    algorithms, args, _requests, checks, _retreats = setup_plan(monkeypatch)
    failure = AlgorithmError('GPU unavailable', code='algorithm_service_unavailable')
    def retreat(*values, **kwargs):
        raise failure
    algorithms.plan_retreat = retreat
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_place(*args)
    assert caught.value is failure
    assert len(checks) == 2


def test_all_yaws_infeasible_returns_evidence_without_a_partial_plan(monkeypatch):
    algorithms, args, _requests, checks, retreats = setup_plan(monkeypatch, valid=())
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_place(*args)
    assert caught.value.code == 'place_no_valid_candidate'
    assert len(caught.value.details['candidate_failures']) == 18
    assert not checks and not retreats


def test_place_ik_filters_yaws_before_planning_and_preserves_original_indices(monkeypatch):
    algorithms, args, requests, _checks, _retreats = setup_plan(monkeypatch, valid=(0,))
    original_request = algorithms.request

    def request(path, body):
        if path == '/tools/solve_ik_batch':
            requests.append((path, body))
            return {'scene_token': 'scene', 'batch_size': 18,
                'items': [{'candidate_index': index, 'success': index in (2, 9)} for index in range(18)],
                'attached': True, 'collision_evidence': {'optimizer_attachment': True}}
        return original_request(path, body)

    algorithms.request = request
    result = algorithms.plan_place(*args)
    assert len(requests) == 2
    assert requests[1][1]['goal_matrices_world'][0] == requests[0][1]['goal_matrices_world'][2]
    assert requests[1][1]['goal_matrices_world'][1] == requests[0][1]['goal_matrices_world'][9]
    assert result['goal']['candidate_index'] == 2
    assert any(item['candidate_index'] == 0 and item['code'] == 'place_ik_unreachable'
               for item in result['candidate_failures'])


def test_place_ik_without_survivors_reports_all_yaws_without_trajectory_planning(monkeypatch):
    algorithms, args, requests, _checks, _retreats = setup_plan(monkeypatch)

    def request(path, body):
        requests.append((path, body))
        assert path == '/tools/solve_ik_batch'
        return {'scene_token': 'scene', 'batch_size': 18,
            'items': [{'candidate_index': index, 'success': False,
                       'position_error_m': .002, 'rotation_error_rad': .03} for index in range(18)],
            'attached': True, 'collision_evidence': {'optimizer_attachment': True}}

    algorithms.request = request
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_place(*args)
    assert caught.value.code == 'place_no_valid_candidate'
    assert len(caught.value.details['candidate_failures']) == 18
    assert all(item['code'] == 'place_ik_unreachable'
               for item in caught.value.details['candidate_failures'])
    assert len(requests) == 1
    rejected = caught.value.details['candidate_failures'][0]
    assert rejected['position_error_m'] == .002
    assert rejected['rotation_error_rad'] == .03
    assert rejected['failure_scope'] == 'collision_aware_ik_not_proof_of_kinematic_unreachability'


def test_last_batch_reports_real_successes_without_counting_padding(monkeypatch):
    algorithms, args, requests, _checks, _retreats = setup_plan(monkeypatch)
    original = algorithms.request
    batches = 0
    def request(path, body):
        nonlocal batches
        response = original(path, body)
        if path == '/tools/plan_place_batch':
            batches += 1
            for item in response['items']:
                item.update(success=batches == 2, approach_success=True, place_success=True)
        return response
    algorithms.request = request
    events = []
    result = algorithms.plan_place(*args, progress=lambda kind, title, data: events.append((kind, data)))
    completed = [data for kind, data in events if kind == 'planning_batch_complete']
    assert [path for path, _ in requests] == [
        '/tools/solve_ik_batch', '/tools/plan_place_batch', '/tools/plan_place_batch']
    assert len(completed) == 2
    assert [len(body['goal_matrices_world']) for _, body in requests] == [18, 16, 16]
    assert completed[-1]['candidate_indices'] == [16, 17]
    assert requests[-1][1]['goal_matrices_world'][2:] == [requests[-1][1]['goal_matrices_world'][1]] * 14
    assert completed[-1]['success_count'] == 2
    assert completed[-1]['tested_count'] == 2
    assert result['candidate_count'] == 18
    assert result['selection_evidence']['successful_candidates_in_batch'] == 2
    assert result['goal']['candidate_index'] == 17
    assert [item['candidate_index'] for item in result['candidate_failures']] == list(range(16))
