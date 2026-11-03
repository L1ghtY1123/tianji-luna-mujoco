"""Arm choice uses measured workspace references without claiming a planned path."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.runtime.controller import PolicyController


def test_container_samples_inform_choice_without_becoming_release_goals_or_rejecting_arms():
    source = {'id': 'part', 'kind': 'cube', 'position': [.4, .1, .6]}
    points = np.column_stack((np.linspace(.3, .6, 120), np.full(120, -.6), np.full(120, .64)))
    destination = {'id': 'box', 'kind': 'bin', 'points_world_m': points}
    calls = []

    def lookup(arm, positions):
        calls.append(positions)
        return [{'position_world_m': p, 'reachable': p == source['position'] or arm == 'B'}
                for p in positions]

    result = DemoAlgorithms(workspace_lookup=lookup).evaluate_reachability(
        {'objects': [source, destination]}, 'part', 'box')
    assert all(row['reachable'] for row in result.values())
    assert all(row['destination']['reachable'] is None for row in result.values())
    assert all(row['destination_candidates'] == [] for row in result.values())
    assert result['A']['destination_workspace_context']['reachable_sample_count'] == 0
    assert result['B']['destination_workspace_context']['reachable_sample_count'] == 64
    assert all(len(call) <= 2 for call in calls)
    for row in result.values():
        evidence = row['destination_workspace_context']
        assert evidence['reference_only_not_release_goals'] and evidence['absence_is_not_infeasibility_proof']
        assert all(any(np.array_equal(p, sample) for sample in points) for p in evidence['reachable_examples'])


@pytest.mark.parametrize('choice', ['A', 'B', 'unknown', 'invalid'])
async def test_model_chooses_current_transfer_and_explicit_unknown_never_falls_back(choice):
    evaluations = {arm: {'reachable': True} for arm in ('A', 'B')}
    runner = SimpleNamespace(registry=SimpleNamespace(handlers={'choose_arm': None}),
        state={'target_id': 'product', 'destination_id': 'box'},
        algorithms=SimpleNamespace(evaluate_reachability=Mock()),
        call_algorithm=AsyncMock(return_value=evaluations),
        task_progress=SimpleNamespace(bind_arm=Mock()),
        _judge=AsyncMock(return_value={'judgments': {'execution_arm': {
            'value': choice, 'status': 'proposed' if choice in ('A', 'B') else 'unknown', 'confidence': .58}},
            'evidence': {}}))
    context = {'observation': {'source': 'rgbd', 'observation_id': 'current', 'objects': [
        {'id': 'product', 'kind': 'cube', 'position': [.4, 0, .6]},
        {'id': 'box', 'kind': 'bin', 'position': [.5, -.6, .6]},
        {'id': 'unrelated', 'kind': 'cube'}]}}
    result = await PolicyController.execute(runner, 'choose_arm', context, {'arm': 'auto'})
    request = runner._judge.call_args.args[1][0]
    assert request.selection_mode == 'proposal' and request.choices == ('A', 'B', 'unknown')
    details = runner._judge.call_args.kwargs['extra']
    assert {e['id'] for e in details['current_measured_entities']} == {'product', 'box'}
    if choice in ('A', 'B'):
        assert result['arm'] == choice and result['success']
        runner.task_progress.bind_arm.assert_called_once_with(choice)
    else:
        assert result['status'] == 'unknown'
        runner.task_progress.bind_arm.assert_not_called()
