import json

import pytest
from copy import deepcopy

from test_agent_model_layer import CATALOG, Client, capture, plan_data, task_data

from luna_mujoco.agent.responses import ResponsesCompiler


def test_initial_root_instruction_is_constrained_to_exact_user_text_in_wire_schema():
    from jsonschema import Draft202012Validator

    client = Client([plan_data()])
    instruction = task_data()['instruction']
    ResponsesCompiler(registry=CATALOG, api_key='test', client=client).compile(capture(), instruction=instruction)
    schema = client.requests[0][1]['text']['format']['schema']
    root = schema['$defs'][schema['properties']['task']['$ref'].split('/')[-1]]
    assert root['properties']['instruction']['enum'] == [instruction]
    # Object-owned task descriptions retain their own schema; only the immutable
    # aggregate instruction is constrained to the user's complete text.
    assert 'enum' not in schema['$defs']['TaskSpec']['properties']['instruction']
    changed = plan_data()
    changed['task']['instruction'] = 'abbreviated task that omits user constraints'
    assert any(error.validator == 'enum' for error in Draft202012Validator(schema).iter_errors(changed))


def test_first_call_exposes_contract_transitions_without_changing_model_graph():
    catalog = deepcopy(CATALOG)
    catalog['pick'].update(requires={'observed': True}, provides=['held'],
                           planning_effects={'observed': False, 'map_ready': None})
    catalog['inspect'].update(provides=['observed'])
    plan = plan_data()
    plan['graph']['nodes'].insert(0, {'type': 'skill', 'id': 'inspect_first', 'skill': 'inspect', 'arguments': {}})
    plan['graph']['edges'].append({'id': 'inspect_edge', 'source': 'inspect_first', 'target': plan['graph']['entry'], 'on': 'success'})
    plan['graph']['entry'] = 'inspect_first'
    client = Client([plan])
    result = ResponsesCompiler(registry=catalog, api_key='test', client=client).compile(
        capture(), instruction=task_data()['instruction'])
    context = json.loads(client.requests[0][1]['input'][0]['content'][0]['text'])
    transitions = context['planning_state_transitions']
    assert transitions['pick']['requires'] == {'observed': True}
    assert transitions['pick']['on_success'] == {'held': True, 'observed': False, 'map_ready': None}
    assert transitions['inspect']['on_success'] == {'observed': True}
    assert result.graph.model_dump() == plan['graph']
    assert len(client.requests) == 1
    assert client.requests[0][1]['reasoning']['effort'] == 'medium'
    failures = context['execution_failure_context']
    assert 'decisions_judgment_http_503' in failures['observed_service_errors']
    assert 'visual_decisions_http_503' in failures['observed_service_errors']
    assert failures['recovery_owner'] == 'decisions_runtime_graph_patch'
    assert 'Do not enumerate all failures' in context['initial_recovery_design']
    assert context['known_capability_outcomes']['pick'] == catalog['pick'].get('errors', [])
    assert 'without a verdict' in failures['transport_policy']


def test_invalid_initial_graph_records_error_without_second_responses_call():
    invalid = plan_data()
    invalid['graph']['nodes'][0]['skill'] = 'missing_skill'
    events = []
    client = Client([invalid, plan_data()])
    compiler = ResponsesCompiler(registry=CATALOG, api_key='test', client=client, audit=events.append)
    with pytest.raises(ValueError):
        compiler.compile(capture(), instruction=task_data()['instruction'])
    assert len(client.requests) == len(events) == 1
    assert events[0]['model_call']['kind'] == 'initial_compile'
    assert events[0]['validation_errors']
