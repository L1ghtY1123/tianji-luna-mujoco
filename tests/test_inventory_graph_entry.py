"""Initial uncertainty belongs to a model-authored graph, before any transfer."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.agent.responses import BOOTSTRAP_FACTS
from luna_mujoco.runtime.controller import PolicyController
from luna_mujoco.runtime.diagnostics import DiagnosticError


def call(identifier, skill, **arguments):
    return {'type': 'skill', 'id': identifier, 'skill': skill, 'arguments': arguments}


@pytest.mark.parametrize('judgment,status', [('unknown', 'unknown'), ('no', 'failure')])
async def test_controller_routes_initial_binding_outcome_to_explicit_fresh_branch(tmp_path, judgment, status):
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.task_inventory = {'selected_ids': [], 'source_id': None, 'destination_id': None, 'work_area_id': None}
    runner.task_progress = SimpleNamespace(inventory_finalized=False, completed_input_ids=list)
    from graph_contract_fixtures import linear_graph
    from luna_mujoco.core.graph import Graph
    graph = linear_graph([call('view', 'observe_scene'), call('confirm', 'confirm_inventory'),
        call('fresh', 'observe_scene', capture_mode='fresh'), call('confirm_again', 'confirm_inventory')])
    graph['edges'][1]['on'] = status
    runner.policy_graph = Graph.model_validate(graph)
    calls = []

    async def step(skill, context, arguments):
        calls.append((skill, deepcopy(arguments)))
        if len(calls) == 2:
            raise DiagnosticError('task_binding_unconfirmed:target_0', {
                'diagnostic_checks': [{'name': 'target_0', 'status': judgment, 'required': True}]})
        return {'success': True}

    runner._step = step
    runner.emit = lambda *args: None
    runner.check_generation = lambda: None
    await runner._execute_graph({})
    assert calls == [('observe_scene', {}), ('confirm_inventory', {}),
                     ('observe_scene', {'capture_mode': 'fresh'}), ('confirm_inventory', {})]
    assert runner.compiler.budget.used == 0
    assert runner.graph_executor.results['confirm']['status'] == status


def test_raw_initial_capture_is_not_a_confirmed_inventory():
    for key in ('observed', 'inventory_ready', 'inventory_finalized', 'task_assessed', 'grasp_view_ready'):
        assert BOOTSTRAP_FACTS[key] is False


def test_fresh_observation_can_precede_reconfirmation_of_stale_task_source():
    from luna_mujoco.runtime.inventory import PolicyTaskProgress
    ledger = PolicyTaskProgress('Move two objects')
    ledger.records['a'] = {'id': 'a', 'status': 'observed', 'initial_source': {
        'measurement_observation_id': 'new', 'task_source_evidence': {
            'source': 'openai_decisions', 'status': 'yes', 'target_id': 'a',
            'source_id': 'bin', 'source_kind': 'bin', 'request_hash': 'hash', 'observation_id': 'old'}}}
    ledger.membership['a'] = 'unknown'
    ledger.assess_observed_membership({'a': 'required'})
    assert ledger.records['a']['candidate_membership'] == 'required'
    assert ledger.membership['a'] == 'unknown'
    assert ledger.inventory_finalized is False
    # Only the dedicated confirmation may authorize a selected task member.
    with pytest.raises(ValueError, match='task_source_region_unconfirmed'):
        ledger.assess_membership({'a': 'required'})
    ledger.records['a']['initial_source']['task_source_evidence']['observation_id'] = 'new'
    ledger.assess_membership({'a': 'required'})
    assert ledger.membership['a'] == 'required'
