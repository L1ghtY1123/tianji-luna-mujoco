"""Resolve retries use the preference that was actually bound on first failure."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from test_unified_placement_controller import controller

from luna_mujoco.agent.schema import Graph
from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.runtime.execution_state import PolicyStopped
from luna_mujoco.runtime.recovery_evidence import blocked_operations, operation_evidence


@pytest.fixture(autouse=True)
def isolate_numerical_resolution_for_recovery_tests(monkeypatch):
    """These tests exercise retry evidence; XY semantics have separate integration tests."""
    async def resolve(controller, context, observation, request):
        try:
            return await controller.call_algorithm(controller.algorithms.resolve_placement,
                observation, placement_request=request)
        except AlgorithmError as error:
            if error.code == 'placement_no_candidate':
                error.details = {**(error.details or {}),
                    'effective_arguments': {'preference': request['placement_preference']}}
            raise
    monkeypatch.setattr('luna_mujoco.runtime.controller.resolve_xy_then_z', resolve)


def setup_runner(tmp_path, *, effective='clearance', error_code='placement_no_candidate'):
    runner = controller(tmp_path)
    runner.state['generation'] = runner.simulation.robot_snapshot()['generation']
    runner.state['facts'].update(runner.registry.contracts['resolve_placement'].requires)
    runner.task_progress = SimpleNamespace(records={'held': {'placement_preference': effective}},
        completed_input_ids=set, snapshot=dict, inventory_finalized=True)
    runner.task_inventory = {}
    runner.sync_inventory = Mock()
    runner.emit = Mock()
    runner.verify_scene = AsyncMock()
    runner.prepare_attachment = AsyncMock()
    runner.measured_state = lambda context: context['observation']
    runner._bind_region = lambda observation: {'placement_preference':
        runner.task_progress.records['held']['placement_preference']}
    runner.algorithms = Mock()
    runner.call_algorithm = AsyncMock(side_effect=AlgorithmError('No candidate', code=error_code,
        details={'candidate_failures': [], 'whole_destination_infeasible': False}))
    runner.policy_graph = Graph.model_validate({'version': 2, 'id': 'resolve-retry', 'entry': 'resolve',
        'subgraphs': [], 'nodes': [
            {'id': 'resolve', 'type': 'skill', 'skill': 'resolve_placement', 'arguments': {'preference': 'decide'}},
            {'id': 'done', 'type': 'end', 'status': 'success', 'reason': 'resolved'}],
        'edges': [{'id': 'resolved', 'source': 'resolve', 'target': 'done', 'on': 'success'}]})
    context = {'placement_zone': {}, 'scene': {'coverage': {'target': 'current'}}, 'holding': {},
        'planning_settings': {}, 'observation': {'objects': [{'id': 'held'}]},
        'latest_capture': {'observation_id': 'current', 'views': {'1': {'rgb': np.zeros((16, 16, 3), dtype=np.uint8)}}},
        'capture_epochs': {'current': runner.observation_epoch}}
    return runner, context


@pytest.mark.parametrize('effective,other', [('clearance', 'reach'), ('reach', 'clearance')])
async def test_initial_failure_blocks_resolved_aliases_before_recovery_choice(tmp_path, effective, other):
    runner, context = setup_runner(tmp_path, effective=effective)
    async def judge(ctx, requests, *, extra=None, **kwargs):
        question = requests[0]
        assert 'retry_failed_node' not in question.choices
        assert 'resolve_placement' in question.choices
        blocked = [row.get('arguments') for row in extra['rejected_proposals']
                   if row.get('skill') == 'resolve_placement']
        assert {'preference': 'decide'} in blocked
        assert {'preference': effective} in blocked
        assert {'preference': other} not in blocked
        assert extra['failed_result']['details']['effective_arguments'] == {'preference': effective}
        return {'judgments': {question.name: {'status': 'proposed', 'value': 'stop'}}}
    runner._judge = judge
    with pytest.raises(PolicyStopped):
        await runner._execute_graph(context)
    assert runner.call_algorithm.await_count == 1


async def test_explicit_different_preference_remains_selectable_after_first_failure(tmp_path):
    runner, context = setup_runner(tmp_path)
    count = 0
    async def judge(ctx, requests, *, extra=None, **kwargs):
        nonlocal count
        count += 1
        question = requests[0]
        if question.name == 'recovery_arguments':
            choice = next(key for key, values in extra['argument_options'].items()
                          if values == {'preference': 'reach', 'capture_mode': 'reuse_current'})
        else:
            assert question.name == 'recovery_capability'
            choice = 'resolve_placement' if count == 1 else 'stop'
        return {'judgments': {question.name: {'status': 'proposed', 'value': choice}}}
    runner._judge = judge
    with pytest.raises(PolicyStopped):
        await runner._execute_graph(context)
    assert runner.call_algorithm.await_count == 2
    assert [call.kwargs['placement_request']['placement_preference']
            for call in runner.call_algorithm.await_args_list] == ['clearance', 'reach']


async def test_transient_service_error_does_not_register_an_exhausted_resolve(tmp_path):
    runner, context = setup_runner(tmp_path, error_code='service_unavailable')
    async def judge(ctx, requests, *, extra=None, **kwargs):
        assert 'retry_failed_node' in requests[0].choices
        assert extra['rejected_proposals'] == []
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'stop'}}}
    runner._judge = judge
    with pytest.raises(PolicyStopped):
        await runner._execute_graph(context)


@pytest.mark.parametrize('proposal', [
    {'action': 'retry'},
    {'action': 'skill', 'skill': 'resolve_placement', 'arguments': {}},
    {'action': 'skill', 'skill': 'resolve_placement', 'arguments': {'preference': 'decide'}},
    {'action': 'skill', 'skill': 'resolve_placement', 'arguments': {'preference': 'clearance'}},
])
async def test_dispatch_guard_rejects_same_request_even_if_planner_returns_it(tmp_path, monkeypatch, proposal):
    runner, context = setup_runner(tmp_path)
    monkeypatch.setattr('luna_mujoco.agent.recovery.RecoveryPlanner.propose', AsyncMock(return_value=proposal))
    with pytest.raises(PolicyStopped):
        await runner._execute_graph(context)
    assert runner.call_algorithm.await_count == 1
    decisions = [call.args[2]['proposal'] for call in runner.emit.call_args_list
                 if call.args[0] == 'policy_recovery_decision']
    assert len(decisions) == 3
    assert all(row['action'] == 'rejected' and row['reason'] == 'identical_evidence_and_operation_no_progress'
               for row in decisions)


@pytest.mark.parametrize('change', ['pixels', 'map_coverage', 'holding_fact', 'measured_ids', 'completed_inputs'])
async def test_existing_generic_evidence_changes_still_restore_choices(tmp_path, change):
    runner, context = setup_runner(tmp_path)
    runner._judge = AsyncMock(side_effect=lambda ctx, requests, **kwargs:
        {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'stop'}}})
    with pytest.raises(PolicyStopped):
        await runner._execute_graph(context)
    facts = deepcopy(runner.state['facts'])
    completed = []
    if change == 'pixels':
        context['latest_capture']['views']['1']['rgb'][:] = 255
    elif change == 'map_coverage':
        context['scene']['coverage'] = {'target': 'new-published-coverage'}
    elif change == 'holding_fact':
        facts['attached'] = not facts.get('attached', False)
    elif change == 'measured_ids':
        context['observation']['objects'].append({'id': 'new-measured-entity'})
    else:
        completed = ['prior-target']
    evidence = operation_evidence(context, 'resolve', 'placement_no_candidate', facts, completed)
    assert blocked_operations(context['graph_recovery_signatures'], evidence) == []
