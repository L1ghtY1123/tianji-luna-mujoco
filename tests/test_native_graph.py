import asyncio
from copy import deepcopy

import pytest
from test_object_task_policy import task

from luna_mujoco.agent.schema import Graph, object_task_specs
from luna_mujoco.runtime.native_graph import NativeGraphExecutor


def graph(nodes=None, edges=None):
    return {'version': 2, 'id': 'task', 'entry': 'first', 'subgraphs': [],
        'nodes': nodes or [
            {'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''},
            {'type': 'skill', 'id': 'second', 'skill': 'measure', 'arguments': {'number': 2}},
            {'type': 'skill', 'id': 'first', 'skill': 'measure', 'arguments': {'number': 1}}],
        'edges': edges or [
            {'id': 'e1', 'source': 'first', 'target': 'second', 'on': 'success'},
            {'id': 'e2', 'source': 'second', 'target': 'done', 'on': 'success'}]}


def test_native_prompt_deduplicates_only_verbatim_documented_sentences():
    import re

    from luna_mujoco.agent.graph_prompt import graph_context
    from luna_mujoco.skills import Registry
    registry = Registry()
    catalog = registry.describe()
    original = deepcopy(catalog)
    output = graph_context({'skill_catalog': catalog, 'skill_instructions': registry.documents})
    saved = 0
    for name, contract in catalog.items():
        compact = output['skill_catalog'][name]
        assert {k: v for k, v in compact.items() if k != 'description'} == {
            k: v for k, v in contract.items() if k != 'description'}
        available = ' '.join((compact['description'] + ' ' + output['skill_instructions'][name]).split())
        for sentence in re.split(r'(?<=[.!?])\s+', contract['description']):
            assert ' '.join(sentence.split()) in available
        saved += len(contract['description']) - len(compact['description'])
    assert saved > 0  # Short current descriptions need not recreate historical prompt bulk.
    assert catalog == original
    assert output['skill_instructions'] == registry.documents


def test_execution_follows_edges_not_node_order():
    calls, events = [], []
    async def invoke(skill, args):
        calls.append(args['number'])
        return {'success': True}
    runner = NativeGraphExecutor(invoke, target_ids=[], bindings={},
        emit=lambda kind, title, data: events.append((kind, data)))
    asyncio.run(runner.run(Graph.model_validate(graph())))
    assert calls == [1, 2]
    assert [data['edge_id'] for kind, data in events if kind == 'policy_edge'] == ['e1', 'e2']


def test_complete_graph_requires_explicit_scope_return_with_specific_diagnostic():
    from luna_mujoco.runtime.validation import validate_capabilities
    data = graph(nodes=[{'type': 'skill', 'id': 'first', 'skill': 'measure', 'arguments': {}}])
    data['edges'] = []
    original = deepcopy(data)
    with pytest.raises(ValueError, match='graph_scope_missing_return:task:leaves=first'):
        validate_capabilities(Graph.model_validate(data), {}, require_completion=True)
    assert data == original  # Validation never inserts a developer-authored return edge.


def test_router_predicates_do_not_have_to_be_completion_conditions_and_cannot_prove_completion():
    from luna_mujoco.runtime.validation import validate_capabilities
    data = Graph.model_validate(graph(nodes=[
        {'type': 'router', 'id': 'first', 'condition': 'Is the target visible now?'},
        {'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''}], edges=[
        {'id': 'yes', 'source': 'first', 'target': 'done', 'on': 'yes'}]))
    validate_capabilities(data, {}, judgments={'All objects delivered'}, require_completion=True)
    with pytest.raises(ValueError, match='completion_fact_not_established:task_verified'):
        validate_capabilities(data, {}, judgments={'All objects delivered'}, require_completion=True,
                              required_final_facts={'task_verified': True})


@pytest.mark.asyncio
@pytest.mark.parametrize('verdict', ['yes', 'no', 'unknown'])
async def test_controller_sends_graph_declared_router_condition_to_decisions(tmp_path, verdict):
    from types import SimpleNamespace

    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.runtime.controller import PolicyController
    controller = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    controller.task_spec = task('a', 'left')
    controller.task_inventory = {'selected_ids': []}
    controller.task_progress = SimpleNamespace(inventory_finalized=True, completed_input_ids=set, snapshot=dict)
    condition = 'Is the target visible now?'
    assert condition not in controller.task_spec.completion_conditions
    controller.policy_graph = Graph.model_validate(graph(nodes=[
        {'type': 'router', 'id': 'first', 'condition': condition},
        {'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''}], edges=[
        {'id': verdict, 'source': 'first', 'target': 'done', 'on': verdict}]))
    requests = []
    async def judge(context, questions, *, extra):
        requests.extend(questions)
        return {'judgments': {'condition': {'status': verdict}}}
    controller._judge = judge
    events = []
    controller.emit = lambda kind, title, data: events.append((kind, data))
    controller.check_generation = lambda: None
    await controller._execute_graph({})
    assert [request.instructions for request in requests] == [condition]
    assert [data['on'] for kind, data in events if kind == 'policy_edge'] == [verdict]


def test_dangling_and_ambiguous_edges_are_rejected():
    data = graph()
    data['edges'][0]['target'] = 'missing'
    with pytest.raises(ValueError, match='graph_edge_unknown_node'):
        Graph.model_validate(data)
    data = graph()
    data['edges'].append({'id': 'other', 'source': 'first', 'target': 'done', 'on': 'success'})
    with pytest.raises(ValueError, match='graph_ambiguous_edge'):
        Graph.model_validate(data)


def test_unexpected_unknown_gets_decisions_authored_graph_extension_before_execution():
    calls, events = [], []
    async def invoke(skill, args):
        calls.append(skill)
        if skill == 'measure' and calls.count('measure') == 1:
            return {'status': 'unknown', 'success': False, 'reason': 'occluded'}
        if skill == 'change_view':
            return {'observation_id': 'new'}  # A successful skill need not return a success flag.
        return {'success': True}
    async def recover(node, result, history, bindings):
        assert node['id'] == 'first'
        if history:
            assert history[-1]['status'] == 'success'
        return {'action': 'skill', 'skill': 'change_view', 'arguments': {}, 'evidence': {'request_hash': 'choice1'}} if not history else {
            'action': 'retry', 'evidence': {'request_hash': 'choice2'}}
    data = graph()
    runner = NativeGraphExecutor(invoke, target_ids=[], bindings={}, propose_recovery=recover,
        repeatable_skills={'measure'}, emit=lambda kind, title, payload: events.append((kind, deepcopy(payload))))
    asyncio.run(runner.run(Graph.model_validate(data)))
    assert calls == ['measure', 'change_view', 'measure', 'measure']
    patches = [payload for kind, payload in events if kind == 'policy_graph_patch']
    assert patches and patches[-1]['source'] == 'decisions'
    assert patches[-1]['policy_graph']['version'] == 2
    assert data == graph()  # initial proposal remains immutable
    assert object_task_specs(runner.graph) == {}


def test_decisions_cannot_replay_nonrepeatable_failed_action():
    async def invoke(skill, args):
        return {'status': 'failure', 'success': False}
    async def recover(*args):
        return {'action': 'retry', 'evidence': {}}
    runner = NativeGraphExecutor(invoke, target_ids=[], bindings={}, propose_recovery=recover,
        repeatable_skills=set())
    with pytest.raises(ValueError, match='recovery_attempt_not_repeatable'):
        asyncio.run(runner.run(Graph.model_validate(graph())))


@pytest.mark.parametrize(('initial_verdict', 'outcome'), [(None, 'unknown'), (False, 'no')])
def test_unhandled_router_unknown_uses_decisions_recovery_and_rechecks_condition(initial_verdict, outcome):
    from luna_mujoco.agent.recovery import RecoveryPlanner
    calls, events = [], []
    async def judge_condition(condition, bindings):
        calls.append('condition')
        return True if 'observe' in calls else initial_verdict
    async def invoke(skill, arguments):
        calls.append(skill)
        return {'success': True, 'observation_id': 'new-view'}
    async def decide(requests, context):
        request = requests[0]
        choice = 'retry_failed_node' if request.name == 'recovery_continuation' else 'observe'
        assert choice in request.choices
        return {'judgments': {request.name: {'status': 'proposed', 'value': choice}}}
    catalog = {'observe': {'arguments': {'type': 'object', 'properties': {}}}}
    planner = RecoveryPlanner(decide, catalog)
    async def recover(node, result, history, bindings):
        return await planner.propose(node, result, history, bindings, facts={}, inventory=[], context={})
    data = Graph.model_validate(graph(nodes=[
        {'type': 'router', 'id': 'first', 'condition': 'Can the next object be identified?'},
        {'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''}], edges=[
        {'id': 'yes', 'source': 'first', 'target': 'done', 'on': 'yes'}]))
    runner = NativeGraphExecutor(invoke, judge=judge_condition, target_ids=[], bindings={},
        propose_recovery=recover, emit=lambda kind, title, data: events.append((kind, deepcopy(data))))
    asyncio.run(runner.run(data))
    assert calls == ['condition', 'observe', 'condition']
    assert any(kind == 'policy_graph_patch' and data['source'] == 'decisions' for kind, data in events)
    assert [data['on'] for kind, data in events if kind == 'policy_edge'] == [
        outcome, 'selected', 'success', 'selected', 'yes']


def two_objects():
    data = graph()
    data['nodes'] = [{'type': 'subgraph', 'id': 'first', 'subgraph_id': 'item_a'},
        {'type': 'subgraph', 'id': 'second', 'subgraph_id': 'item_b'},
        {'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''}]
    for identifier, destination in [('a', 'left'), ('b', 'right')]:
        data['subgraphs'].append({'id': 'item_' + identifier, 'input_id': identifier,
            'task': task(identifier, destination).model_dump(mode='json'), 'entry': 'move_' + identifier,
            'nodes': [{'type': 'skill', 'id': 'move_' + identifier, 'skill': 'move',
                'arguments': {'target': '$target', 'destination': '$destination'}},
                {'type': 'end', 'id': 'end_' + identifier, 'status': 'success', 'reason': ''}],
            'edges': [{'id': 'edge_' + identifier, 'source': 'move_' + identifier,
                       'target': 'end_' + identifier, 'on': 'success'}]})
    return data


def test_subgraph_recovery_stop_preserves_original_skill_failure_details():
    from luna_mujoco.runtime.execution_state import UnhandledSkillResult

    calls = []
    failure = {'status': 'failure', 'success': False, 'reason': 'planner unavailable',
               'error_code': 'CUROBO_BACKEND_FAILED', 'details': {'arm': 'A'}}

    async def invoke(skill, arguments):
        calls.append(arguments['target'])
        return deepcopy(failure)

    async def recover(*args):
        return {'action': 'stop', 'reason': 'decisions_recovery_stop'}

    runner = NativeGraphExecutor(invoke, target_ids=['a', 'b'], bindings={}, propose_recovery=recover)
    with pytest.raises(UnhandledSkillResult) as caught:
        asyncio.run(runner.run(Graph.model_validate(two_objects())))
    result = caught.value.details['result']
    assert result['error_code'] == failure['error_code']
    assert result['details'] == failure['details']
    assert result['reason'] == failure['reason']
    assert result['recovery_stop_reason'] == 'decisions_recovery_stop'
    assert calls == ['a']


def test_inventory_owned_scopes_are_independent_and_completed_item_is_skipped():
    calls = []
    async def invoke(skill, args):
        calls.append(deepcopy(args))
        return {'success': True}
    runner = NativeGraphExecutor(invoke, target_ids=['a', 'b'], bindings={'$destination': 'outer'})
    model = Graph.model_validate(two_objects())
    asyncio.run(runner.run(model))
    assert calls == [{'target': 'a', 'destination': 'left'}, {'target': 'b', 'destination': 'right'}]
    assert runner.bindings == {'$destination': 'outer'}
    assert set(object_task_specs(model)) == {'a', 'b'}
    calls.clear()
    runner = NativeGraphExecutor(invoke, target_ids=['a', 'b'], bindings={}, is_target_pending=lambda identifier: identifier != 'a')
    asyncio.run(runner.run(model))
    assert calls == [{'target': 'b', 'destination': 'right'}]


def test_missing_completion_receipt_recovers_at_end_without_replaying_transfer():
    calls, checks, events = [], [], []
    async def invoke(skill, args):
        calls.append((skill, deepcopy(args)))
        return {'success': True}
    async def complete(identifier):
        checks.append(identifier)
        if identifier == 'a' and not any(skill == 'observe' for skill, _ in calls):
            return {'status': 'unknown', 'success': False, 'reason': 'observation_receipt_missing'}
    async def recover(node, result, history, bindings):
        assert node['type'] == 'end' and node['id'] == 'end_a'
        assert bindings['$target'] == 'a'
        return {'action': 'retry'} if history else {'action': 'skill', 'skill': 'observe', 'arguments': {}}
    runner = NativeGraphExecutor(invoke, target_ids=['a', 'b'], bindings={}, after_target=complete,
        propose_recovery=recover, repeatable_skills={'observe'},
        emit=lambda kind, title, data: events.append((kind, data)))
    asyncio.run(runner.run(Graph.model_validate(two_objects())))
    assert [skill for skill, _ in calls] == ['move', 'observe', 'move']
    assert checks == ['a', 'a', 'b']
    assert runner.results['end_a']['success']
    assert any(kind == 'policy_graph_patch' for kind, _ in events)
    Graph.model_validate(runner.graph)


def test_decisions_can_recheck_end_obligation_without_retrying_subgraph():
    from luna_mujoco.agent.recovery import RecoveryPlanner
    async def judge(requests, context):
        assert context['failed_node']['type'] == 'end'
        assert 'retry_failed_node' in requests[0].choices
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'retry_failed_node'}}}
    result = asyncio.run(RecoveryPlanner(judge, {}).propose(
        {'id': 'finish', 'type': 'end', 'status': 'success'}, {'status': 'unknown'},
        [{'status': 'success', 'result': {'observation_id': 'fresh'}}], {}, facts={}, inventory=[], context={}))
    assert result['action'] == 'retry'


@pytest.mark.parametrize('final_pose_verified', [False, True])
def test_single_initial_responses_emits_native_graph_and_binds_inventory_ids(final_pose_verified):
    import json

    from test_agent_model_layer import Client, capture, plan_data

    from luna_mujoco.agent.responses import ResponsesBudget, ResponsesCompiler
    from luna_mujoco.agent.schema import InitialPlan, response_schema
    output = plan_data()
    output['task'].update(selection_order='explicit', target_refs=['item'])
    item = task('item', 'tray').model_dump(mode='json')
    item['source_ref'] = 'table'
    native = two_objects()
    native['nodes'] = [native['nodes'][0], native['nodes'][-1]]
    native['edges'] = [{'id': 'complete', 'source': 'first', 'target': 'done', 'on': 'success'}]
    native['subgraphs'] = [native['subgraphs'][0]]
    native['subgraphs'][0].update(input_id='item', task=item)
    output['graph'] = native
    catalog = {'move': {'arguments': {'type': 'object', 'properties': {
        'target': {'type': 'string'}, 'destination': {'type': 'string'}}, 'required': ['target', 'destination'],
        'additionalProperties': False}}}
    if final_pose_verified:
        output['task']['completion_pose'] = 'initial'
        native['nodes'].append({'type': 'skill', 'id': 'verify', 'skill': 'verify_task_completion', 'arguments': {}})
        native['edges'][0]['target'] = 'verify'
        native['edges'].append({'id': 'verified', 'source': 'verify', 'target': 'done', 'on': 'success'})
        catalog['verify_task_completion'] = {'arguments': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                                            'provides': ['task_verified']}
    client = Client([output])
    compiler = ResponsesCompiler(registry=catalog, api_key='test', client=client,
        initial_facts={'inventory_finalized': True}, budget=ResponsesBudget())
    result = compiler.discover(capture(), instruction=output['task']['instruction'])
    assert len(client.requests) == 1
    assert result['policy_graph']['version'] == 2
    identifier = result['task_inventory']['source_object_ids'][0]
    assert result['task_inventory']['subgraph_by_input_id'] == {identifier: 'item_a'}
    assert result['policy_graph']['subgraphs'][0]['task']['target_refs'] == [identifier]
    prompt = json.loads(client.requests[0][1]['input'][0]['content'][0]['text'])
    assert 'NOT required to enumerate all' in prompt['rules']
    schema = response_schema(InitialPlan, catalog)
    assert 'RecoveryNode' not in json.dumps(schema)
    assert 'steps' not in json.dumps(schema['$defs']['Graph'])


def test_runtime_recovery_selects_skill_and_parameters_with_decisions():
    from luna_mujoco.agent.recovery import RecoveryPlanner
    calls = []
    async def judge(requests, context):
        calls.append(context)
        return {'judgments': {request.name: {'status': 'proposed', 'value':
            'view' if request.name == 'recovery_capability' else 'value_1'} for request in requests},
            'evidence': {'request_hash': str(len(calls))}}
    catalog = {'measure': {'arguments': {'type': 'object', 'properties': {}}, 'recovery_retry_safe': True},
        'view': {'arguments': {'type': 'object', 'properties': {'camera': {'enum': ['left', 'right']}},
            'required': ['camera']}, 'requires': {'map_ready': True}}}
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(
        {'id': 'measure', 'skill': 'measure'}, {'status': 'unknown'}, [], {}, facts={'map_ready': True}, inventory=[], context={}))
    assert result['skill'] == 'view' and result['arguments'] == {'camera': 'right'}
    assert len(calls) == 2 and calls[0]['verification_scope'] == 'graph_recovery'


@pytest.mark.parametrize('choice,action', [('retry_failed_node', 'retry'), ('stop', 'stop'), ('unknown', 'stop')])
def test_recovery_continuation_is_chosen_by_decisions_after_success(choice, action):
    from luna_mujoco.agent.recovery import RecoveryPlanner
    calls = []
    async def judge(requests, context):
        calls.append((requests, context))
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': choice}}}
    catalog = {'measure': {'arguments': {'type': 'object', 'properties': {}}, 'recovery_retry_safe': True}}
    history = [{'node': {'skill': 'observe'}, 'status': 'success', 'result': {'observation_id': 'new'}}]
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(
        {'id': 'measure', 'skill': 'measure'}, {'status': 'unknown'}, history, {},
        facts={'observed': True}, inventory=[], context={}))
    assert result['action'] == action
    assert len(calls) == 1
    assert calls[0][0][0].name == 'recovery_continuation'
    assert calls[0][1]['latest_recovery']['result']['observation_id'] == 'new'
    assert 'continuation' in result['evidence']


def test_new_object_cannot_consume_previous_objects_product():
    from luna_mujoco.agent.schema import validate_graph
    data = two_objects()
    data['subgraphs'][1]['nodes'][0]['arguments']['target'] = '$product'
    catalog = {'move': {'arguments': {'type': 'object', 'properties': {
        'target': {'type': 'string'}, 'destination': {'type': 'string'}}},
        'result_bindings': {'$product': 'product_id'}}}
    with pytest.raises(ValueError, match='unbound_policy_reference:move_b:.*product'):
        validate_graph(Graph.model_validate(data), catalog)


def test_actual_new_format_cannot_execute_steps_or_change_inventory_owner():
    with pytest.raises(ValueError):
        Graph.model_validate({'version': 1, 'steps': []})
    data = two_objects()
    data['subgraphs'][1]['input_id'] = 'a'
    with pytest.raises(ValueError, match='subgraph_task_owner_mismatch'):
        Graph.model_validate(data)


def test_nonrepeatable_completed_node_cannot_be_reentered_by_backedge():
    calls = []
    async def invoke(skill, args):
        calls.append(skill)
        return {'success': True}
    data = graph()
    data['edges'][1]['target'] = 'first'
    data['nodes'] = [node for node in data['nodes'] if node['id'] != 'done']
    runner = NativeGraphExecutor(invoke, target_ids=[], bindings={}, repeatable_skills=set())
    with pytest.raises(ValueError, match='completed_action_replay_refused:first'):
        asyncio.run(runner.run(Graph.model_validate(data)))
    assert len(calls) == 2


def test_native_processing_requirements_exclude_ordinary_motion_skills():
    from luna_mujoco.agent.schema import InitialPlan, response_schema
    catalog = {name: {'arguments': {'type': 'object', 'properties': {}}, 'processing_effect': effect}
               for name, effect in [('grasp', False), ('process', True)]}
    schema = response_schema(InitialPlan, catalog)
    assert schema['$defs']['TaskSpec']['properties']['processing']['items']['enum'] == ['process']


def test_responses_schema_pairs_shared_subgraph_owner_and_task_nulls():
    from jsonschema import Draft202012Validator

    from luna_mujoco.agent.schema import InitialPlan, response_schema
    catalog = {'move': {'arguments': {'type': 'object', 'properties': {
        'target': {'type': 'string'}, 'destination': {'type': 'string'}}}}}
    schema = response_schema(InitialPlan, catalog)
    validator = Draft202012Validator({'$ref': '#/$defs/Subgraph', '$defs': schema['$defs']})
    owned = two_objects()['subgraphs'][0]
    assert validator.is_valid(owned)
    shared = {**deepcopy(owned), 'input_id': None, 'task': None}
    assert validator.is_valid(shared)
    assert not validator.is_valid({**shared, 'task': owned['task']})
    assert not validator.is_valid({**owned, 'task': None})


async def test_controller_recovery_history_never_sends_numpy_geometry_to_decisions(tmp_path):
    import json
    from types import SimpleNamespace

    import numpy as np
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.runtime.controller import PolicyController
    controller = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    controller.task_spec = task('a', 'left')
    controller.task_inventory = {'selected_ids': []}
    controller.task_progress = SimpleNamespace(inventory_finalized=True, completed_input_ids=set, snapshot=dict)
    controller.policy_graph = Graph.model_validate(graph(nodes=[
        {'type': 'skill', 'id': 'first', 'skill': 'confirm_inventory', 'arguments': {}},
        {'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''}], edges=[
        {'id': 'exit', 'source': 'first', 'target': 'done', 'on': 'success'}]))
    controller.state['facts'].update(grasp_held=False, observed=True)
    calls, requests = [], []
    async def step(name, context, arguments):
        calls.append(name)
        if len(calls) == 1:
            return {'status': 'unknown', 'reason': 'ambiguous', 'geometry': np.array([1., 2., 3.])}
        return {'success': True, 'geometry': np.array([1., 2., 3.])}
    async def judge(context, questions, *, extra):
        json.dumps(extra)
        requests.append(questions)
        return {'judgments': {request.name: {'status': 'proposed', 'value':
            ('observe_scene' if len(calls) == 1 else 'retry_failed_node')
            if request.name in {'recovery_capability', 'recovery_continuation'} else 'value_1'} for request in questions}}
    controller._step, controller._judge = step, judge
    controller.emit = lambda *args: None
    controller.check_generation = lambda: None
    await controller._execute_graph({'latest_capture': {'views': {}}})
    assert calls == ['confirm_inventory', 'observe_scene', 'confirm_inventory']


def test_missing_repeatable_node_precondition_is_reported_for_decisions_not_silently_fixed():
    from luna_mujoco.agent.graph_validation import validate_capabilities
    data = Graph.model_validate(graph())
    catalog = {'measure': {'arguments': {'type': 'object', 'properties': {'number': {'type': 'integer'}}},
        'requires': {'observed': True}, 'recovery_retry_safe': True}}
    diagnostics = validate_capabilities(data, catalog, initial_facts={'observed': False}, require_completion=True)
    assert diagnostics[0]['missing_preconditions'] == ['observed']
    assert data.model_dump() == graph()
    catalog['measure']['recovery_retry_safe'] = False
    with pytest.raises(ValueError, match='planning_precondition_not_satisfied'):
        validate_capabilities(data, catalog, initial_facts={'observed': False})


def test_recovery_reports_new_geometry_separately_from_old_incomplete_map():
    from luna_mujoco.agent.recovery import recovery_measurements
    context = {'observation': {'observation_id': 'new', 'objects': [{'id': 'green', 'position': [1, 2, 3]}]},
        'scene': {'observation_id': 'old', 'coverage': {'missing_entity_ids': ['green']}},
        'latest_capture': {'observation_id': 'new'}, 'scene_refresh_required': True}
    summary = recovery_measurements(context)
    assert summary['measured_entities'][0]['id'] == 'green'
    assert summary['map_coverage']['missing_entity_ids'] == ['green']
    assert summary['observation_id'] != summary['map_observation_id']
    assert context['scene']['coverage']['missing_entity_ids'] == ['green']


def test_recovery_removes_rejected_combinations_and_filters_argument_preconditions():
    from luna_mujoco.agent.recovery import RecoveryPlanner
    requests_seen = []
    async def judge(requests, context):
        requests_seen.extend(requests)
        assert 'observe' not in requests[0].choices
        return {'judgments': {'recovery_capability': {'status': 'proposed', 'value': 'map'}}}
    catalog = {
        'identify': {'arguments': {'type': 'object', 'properties': {}}, 'recovery_retry_safe': True},
        'observe': {'arguments': {'type': 'object', 'properties': {'mode': {'enum': ['fresh', 'reuse']}}}},
        'map': {'arguments': {'type': 'object', 'properties': {'mode': {'enum': ['selected', 'observation']}}},
                'argument_rules': [{'when_arguments': {'mode': 'selected'}, 'requires': {'target_selected': True}, 'effects': {}}]}}
    rejected = [{'skill': 'observe', 'arguments': {'mode': mode}} for mode in ['fresh', 'reuse']]
    proposal = asyncio.run(RecoveryPlanner(judge, catalog).propose(
        {'id': 'id', 'skill': 'identify'}, {'status': 'unknown'}, [], {},
        facts={'target_selected': False}, inventory=[], context={'rejected_proposals': rejected}))
    assert proposal['skill'] == 'map' and proposal['arguments'] == {'mode': 'observation'}
    assert len(requests_seen) == 1


def test_recovery_keeps_other_arguments_of_a_rejected_capability():
    from luna_mujoco.agent.recovery import RecoveryPlanner
    async def judge(requests, context):
        assert 'observe' in requests[0].choices
        return {'judgments': {'recovery_capability': {'status': 'proposed', 'value': 'observe'}}}
    catalog = {'identify': {'arguments': {'type': 'object', 'properties': {}}},
        'observe': {'arguments': {'type': 'object', 'properties': {'mode': {'enum': ['fresh', 'reuse']}}}}}
    proposal = asyncio.run(RecoveryPlanner(judge, catalog).propose(
        {'id': 'id', 'skill': 'identify'}, {}, [], {}, facts={}, inventory=[],
        context={'rejected_proposals': [{'skill': 'observe', 'arguments': {'mode': 'fresh'}}]}))
    assert proposal['arguments'] == {'mode': 'reuse'}


def test_recovery_filters_registered_cross_parameter_rules_before_asking_model():
    from luna_mujoco.agent.recovery import RecoveryPlanner
    from luna_mujoco.runtime.controller import BuildSceneArguments
    async def judge(requests, context):
        request = requests[0]
        # Selecting global must leave entity_ids=[] as the only valid continuation.
        if request.name == 'recovery_arguments':
            options = context['argument_options']
            assert all(row['entity_ids'] == [] for row in options.values() if row['scope'] == 'global')
            selected = next(key for key, row in options.items() if row['scope'] == 'global')
            return {'judgments': {request.name: {'status': 'proposed', 'value': selected}}}
        return {'judgments': {request.name: {'status': 'proposed', 'value':
            'map' if request.name == 'recovery_capability' else 'value_1'}}}
    def validate(name, arguments):
        if name == 'map':
            BuildSceneArguments.model_validate(arguments)
    catalog = {'identify': {'arguments': {'type': 'object', 'properties': {}}},
        'map': {'arguments': BuildSceneArguments.model_json_schema()}}
    proposal = asyncio.run(RecoveryPlanner(judge, catalog, validate).propose(
        {'id': 'id', 'skill': 'identify'}, {}, [], {}, facts={}, inventory=['item'], context={}))
    assert proposal['arguments']['scope'] == 'global'
    assert proposal['arguments']['entity_ids'] == []
