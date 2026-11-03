import asyncio
from copy import deepcopy

import pytest
from test_native_graph import graph

from luna_mujoco.agent.recovery import RecoveryPlanner
from luna_mujoco.runtime.native_graph import NativeGraphExecutor


def test_decisions_can_choose_the_current_successful_same_operation_result():
    node = {'id': 'first', 'type': 'skill', 'skill': 'observe', 'arguments': {'fresh': True}}
    latest = {'node': {**node, 'id': 'repair'}, 'status': 'success',
              'result': {'observation_id': 'capture', 'generation': 2}}
    async def judge(requests, context):
        assert requests[0].name == 'recovery_continuation'
        assert 'use_recovery_result' in requests[0].choices
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'use_recovery_result'}}}
    catalog = {'observe': {'arguments': {'type': 'object', 'properties': {}},
                          'recovery_retry_safe': True, 'recovery_result_reusable': True}}
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(node, {'status': 'unknown'},
        [latest], {}, facts={}, inventory=[], context={'reusable_recovery_result': {
            'node_id': 'repair', 'observation_id': 'capture', 'generation': 2}}))
    assert result['action'] == 'use_recovery_result'
    assert result['source_node_id'] == 'repair'


def test_reused_result_follows_original_success_edge_without_third_observation():
    calls, events = [], []
    async def invoke(skill, arguments):
        calls.append(arguments['number'])
        return {'success': len(calls) > 1, 'observation_id': 'new', 'generation': 2}
    async def recover(node, result, history, bindings):
        if not history:
            return {'action': 'skill', 'skill': node['skill'], 'arguments': node['arguments']}
        return {'action': 'use_recovery_result', 'source_node_id': history[-1]['node']['id']}
    runner = NativeGraphExecutor(invoke, target_ids=[], bindings={}, propose_recovery=recover,
        repeatable_skills={'measure'}, reusable_result_skills={'measure'},
        emit=lambda kind, title, data: events.append((kind, deepcopy(data))))
    asyncio.run(runner.run(graph()))
    assert calls == [1, 1, 2]
    assert runner.results['first']['success'] is True
    assert any(kind == 'policy_recovery_result_reused' for kind, _ in events)


@pytest.mark.parametrize('damage', ['arguments', 'skill', 'failed', 'source', 'not_opted_in'])
def test_reuse_cannot_bypass_result_provenance_or_operation_equivalence(damage):
    calls = []
    async def invoke(skill, arguments):
        calls.append(skill)
        return {'success': len(calls) > 1 and damage != 'failed'}
    async def recover(node, result, history, bindings):
        if not history:
            return {'action': 'skill', 'skill': 'other' if damage == 'skill' else node['skill'],
                'arguments': {'number': 99} if damage == 'arguments' else node['arguments']}
        return {'action': 'use_recovery_result',
            'source_node_id': 'invented' if damage == 'source' else history[-1]['node']['id']}
    runner = NativeGraphExecutor(invoke, target_ids=[], bindings={}, propose_recovery=recover,
        repeatable_skills={'measure'}, reusable_result_skills=set() if damage == 'not_opted_in' else {'measure'})
    with pytest.raises(ValueError, match='recovery_result_not_reusable'):
        asyncio.run(runner.run(graph()))
    assert len(calls) == 2


@pytest.mark.parametrize('damage', [None, 'epoch', 'capture', 'observation', 'generation', 'failed', 'mutating'])
def test_only_current_nonmutating_observation_result_is_offered(damage):
    from types import SimpleNamespace

    from luna_mujoco.runtime.recovery_evidence import reusable_observation_result
    context = {'latest_capture': {'observation_id': 'current', 'generation': 2},
               'observation': {'observation_id': 'current', 'generation': 2},
               'capture_epochs': {'current': 7}}
    history = [{'node': {'id': 'repair', 'skill': 'observe'}, 'status': 'success',
                'result': {'observation_id': 'current', 'generation': 2}}]
    contract = SimpleNamespace(recovery_result_reusable=True, recovery_retry_safe=True,
                               invalidates_observation=False)
    if damage == 'epoch': context['capture_epochs']['current'] = 6
    if damage == 'capture': context['latest_capture']['observation_id'] = 'another'
    if damage == 'observation': context['observation']['observation_id'] = 'another'
    if damage == 'generation': history[0]['result']['generation'] = 1
    if damage == 'failed': history[0]['status'] = 'failure'
    if damage == 'mutating': contract.invalidates_observation = True
    result = reusable_observation_result(context, history, 7, {'observe': contract})
    if damage is None:
        assert result == {'node_id': 'repair', 'observation_id': 'current', 'generation': 2, 'action_epoch': 7}
    else:
        assert result is None


def test_reuse_option_is_absent_without_current_controller_evidence():
    node = {'id': 'first', 'type': 'skill', 'skill': 'observe', 'arguments': {}}
    history = [{'node': {**node, 'id': 'repair'}, 'status': 'success', 'result': {'observation_id': 'old'}}]
    async def judge(requests, context):
        assert 'use_recovery_result' not in requests[0].choices
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'stop'}}}
    catalog = {'observe': {'arguments': {'type': 'object', 'properties': {}},
                          'recovery_retry_safe': True, 'recovery_result_reusable': True}}
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(node, {'status': 'unknown'},
        history, {}, facts={}, inventory=[], context={}))
    assert result['action'] == 'stop'


@pytest.mark.parametrize('change', [{'invalidates_observation': True}, {'recovery_retry_safe': False}])
def test_mutating_or_nonrepeatable_contract_cannot_opt_in(change):
    from luna_mujoco.skills import Contract, Registry
    value = Registry().contracts['observe_scene'].model_dump() | change
    with pytest.raises(ValueError, match='recovery_result_reuse_requires_repeatable_sensing'):
        Contract.model_validate(value)


def test_current_fresh_recovery_can_satisfy_reuse_current_observation():
    calls = []
    async def invoke(skill, arguments):
        calls.append(arguments['capture_mode'])
        return {'success': len(calls) > 1, 'observation_id': 'current', 'generation': 2}
    async def recover(node, result, history, bindings):
        if not history:
            return {'action': 'skill', 'skill': node['skill'], 'arguments': {'capture_mode': 'fresh'}}
        return {'action': 'use_recovery_result', 'source_node_id': history[-1]['node']['id']}
    runner = NativeGraphExecutor(invoke, target_ids=[], bindings={}, propose_recovery=recover,
        repeatable_skills={'observe_scene'}, reusable_result_skills={'observe_scene'},
        reusable_result_ignored_arguments={'observe_scene': ['capture_mode']})
    data = graph(nodes=[{'type': 'skill', 'id': 'first', 'skill': 'observe_scene',
                        'arguments': {'capture_mode': 'reuse_current'}},
                       {'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''}],
                 edges=[{'id': 'ok', 'source': 'first', 'target': 'done', 'on': 'success'}])
    asyncio.run(runner.run(data))
    assert calls == ['reuse_current', 'fresh']
    assert runner.results['first']['observation_id'] == 'current'


def test_model_is_offered_fresh_recovery_result_for_current_capture_request():
    node = {'id': 'first', 'type': 'skill', 'skill': 'observe_scene',
            'arguments': {'capture_mode': 'reuse_current'}}
    latest = {'node': {**node, 'id': 'repair', 'arguments': {'capture_mode': 'fresh'}},
              'status': 'success', 'result': {'observation_id': 'current', 'generation': 2}}
    async def judge(requests, context):
        assert 'use_recovery_result' in requests[0].choices
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'use_recovery_result'}}}
    catalog = {'observe_scene': {'arguments': {'type': 'object', 'properties': {}},
        'recovery_retry_safe': True, 'recovery_result_reusable': True,
        'recovery_result_ignored_arguments': ['capture_mode']}}
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(node, {'status': 'unknown'},
        [latest], {}, facts={}, inventory=[], context={'reusable_recovery_result': {
            'node_id': 'repair', 'observation_id': 'current', 'generation': 2}}))
    assert result['action'] == 'use_recovery_result'


def test_destination_rebinding_is_available_after_recovery_observes_the_missing_box():
    from luna_mujoco.runtime.controller import DestinationArguments
    from luna_mujoco.skills import Registry
    contract = Registry().contracts['identify_destination'].model_dump()
    assert contract['invalidates_observation'] is False
    catalog = {'identify_destination': {**contract,
        'arguments': DestinationArguments.model_json_schema()}}
    node = {'id': 'destination', 'type': 'skill', 'skill': 'identify_destination',
            'arguments': {'destination_id': 'box'}}
    history = [{'node': {'id': 'reobserve', 'skill': 'observe_scene'}, 'status': 'success',
                'result': {'status': 'success', 'observation_id': 'new'}}]
    async def judge(requests, context):
        assert requests[0].name == 'recovery_continuation'
        assert 'retry_failed_node' in requests[0].choices
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'retry_failed_node'}}}
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(node, {'status': 'unknown'},
        history, {}, facts=contract['requires'], inventory=['box'], context={}))
    assert result['action'] == 'retry'
