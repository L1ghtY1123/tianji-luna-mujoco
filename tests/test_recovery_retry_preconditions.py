import asyncio

from luna_mujoco.agent.recovery import RecoveryPlanner


def test_recovery_does_not_offer_retry_until_original_argument_contract_is_satisfied():
    calls = []

    async def judge(requests, context):
        calls.append((requests, context))
        assert requests[0].name == 'recovery_capability'
        assert 'retry_failed_node' not in requests[0].choices
        assert context['retry_precondition_conflicts'] == {'grasps_ready': {'required': True, 'actual': False}}
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'generate'}}}

    catalog = {
        'plan': {'arguments': {'type': 'object', 'properties': {}}, 'recovery_retry_safe': True,
                 'requires': {'scene_ready': True, 'grasps_ready': True}},
        'generate': {'arguments': {'type': 'object', 'properties': {}}, 'requires': {'scene_ready': True}},
    }
    history = [{'node': {'skill': 'map'}, 'status': 'success', 'result': {'status': 'success'}}]
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(
        {'id': 'plan', 'skill': 'plan'}, {'status': 'failure'}, history, {},
        facts={'scene_ready': True, 'grasps_ready': False}, inventory=[], context={}))
    assert result['action'] == 'skill' and result['skill'] == 'generate'
    assert len(calls) == 1


def test_out_of_domain_retry_response_is_rejected():
    async def judge(requests, context):
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'retry_failed_node'}}}

    catalog = {'execute': {'arguments': {'type': 'object', 'properties': {}}, 'recovery_retry_safe': False}}
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(
        {'id': 'execute', 'skill': 'execute'}, {'status': 'failure'}, [], {}, facts={}, inventory=[], context={}))
    assert result['action'] == 'stop'
    assert result['reason'] == 'recovery_selection_outside_domain'


def test_small_legal_argument_tuples_are_chosen_in_one_typed_decision():
    calls = []

    async def judge(requests, context):
        question = requests[0]
        calls.append(question.name)
        if question.name == 'recovery_capability':
            value = 'observe'
        else:
            assert question.name == 'recovery_arguments'
            options = context['argument_options']
            assert all(row != {'fresh': False, 'scope': 'local'} for row in options.values())
            value = next(key for key, row in options.items() if row == {'fresh': True, 'scope': 'local'})
        return {'judgments': {question.name: {'status': 'proposed', 'value': value}}}

    def validate(name, arguments):
        if arguments == {'fresh': False, 'scope': 'local'}:
            raise ValueError('invalid_combination')

    catalog = {'observe': {'arguments': {'type': 'object', 'properties': {
        'fresh': {'type': 'boolean'}, 'scope': {'enum': ['global', 'local']}}}}}
    result = asyncio.run(RecoveryPlanner(judge, catalog, validate).propose(
        {'id': 'failed', 'type': 'skill', 'skill': 'absent'}, {'status': 'unknown'}, [], {},
        facts={}, inventory=[], context={}))
    assert calls == ['recovery_capability', 'recovery_arguments']
    assert result['arguments'] == {'fresh': True, 'scope': 'local'}
