from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from luna_mujoco.agent.schema import EntityProposal, TaskSpec
from luna_mujoco.runtime.inventory import PolicyTaskProgress


def task(identifier, destination, processing=()):
    return TaskSpec(instruction='Move the listed objects', supported=True, reason='', scope='ids',
        target_selector=identifier, target_refs=(identifier,), quantity=1, source_ref='source',
        destination_ref=destination, work_area_ref=None, selection_order='explicit',
        processing=processing, completion_conditions=(identifier + ' delivered',), constraints=(),
        source_inventory_complete='unknown')


def call(identifier, skill, **arguments):
    return {'type': 'skill', 'id': identifier, 'skill': skill, 'arguments': arguments}


def object_node(identifier, destination):
    return {'type': 'object_task', 'id': 'task_' + identifier,
        'task': task(identifier, destination).model_dump(mode='json'),
        'body': [call('move_' + identifier, 'move', target='$target', destination='$destination')]}


def test_product_completion_uses_its_input_task_not_another_objects_goal():
    ledger = PolicyTaskProgress('Move the listed objects')
    ledger.configure_task(task('a', 'left'), ['a', 'b'],
        object_tasks={'a': task('a', 'left'), 'b': task('b', 'right', ('vla_unpack',))})
    ledger.records = {
        'a': {'current_region': 'left', 'confirmed_observation_id': 'obs'},
        'b': {'output_entity_id': 'product'},
        'product': {'root_input_id': 'b', 'current_region': 'right',
                    'confirmed_observation_id': 'obs', 'processing_history': ['vla_unpack']}}
    assert ledger.completed_input_ids() == {'a', 'b'}
    ledger.records['product']['current_region'] = 'left'
    assert ledger.completed_input_ids() == {'a'}


def test_alias_queries_are_bounded_deduplicated_and_keep_distinct_visual_phrases():
    from luna_mujoco.demo_perception import DemoPerception
    entities = {'a': {'segmentation_query': {'kind': 'object', 'color': 'pink',
        'prompt': 'pink object', 'aliases': [' PINK  object ', 'pink rectangular block']}},
        'b': {'segmentation_query': {'kind': 'object', 'color': 'pink', 'prompt': 'pink object'}}}
    queries = DemoPerception._segmentation_queries(entities)
    assert [query['prompt'] for query in queries] == ['pink object', 'pink rectangular block']
    assert all(query['kind'] == 'object' and query['color'] == 'pink' for query in queries)


def test_appearance_proposals_do_not_allow_unbounded_or_empty_aliases():
    data = {'ref': 'a', 'kind': 'object', 'color': 'pink', 'description': 'pink rectangular block',
        'bbox': [.1,.1,.2,.2], 'reference_camera_id': '2', 'parent_ref': None, 'membership': 'required',
        'segmentation_prompt': 'pink object', 'appearance_keywords': ['pink', 'rectangular']}
    with pytest.raises(ValueError):
        EntityProposal(**data, segmentation_aliases=['a','b','c'])
    with pytest.raises(ValueError, match='invalid_appearance_description'):
        EntityProposal(**data, segmentation_aliases=[' '])


async def test_object_subgraph_cannot_select_another_inputs_entity(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.runtime.controller import PolicyController
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.active_object_task = task('a', 'left')
    runner.object_tasks = {'a': runner.active_object_task}
    runner.task_progress = SimpleNamespace(records={'b': {'root_input_id': 'b'}})
    with pytest.raises(ValueError, match='target_outside_active_object_task'):
        await runner.execute('identify_target', {}, {'target_id': 'b'})


def test_responses_transport_failure_keeps_exception_type_and_timeout():
    import httpx

    from luna_mujoco.agent.responses import ResponsesCompiler
    from luna_mujoco.visual_api import VisualInventoryError
    class TimeoutClient:
        def post(self, *args, **kwargs):
            raise httpx.ReadTimeout('test timeout')
    events = []
    compiler = ResponsesCompiler(registry={}, api_key='test', client=TimeoutClient(), timeout_s=180, audit=events.append)
    with pytest.raises(VisualInventoryError) as caught:
        compiler.request({'input': []}, {'observation_id': 'frozen'}, 'compile_task', lambda *args: None)
    assert caught.value.evidence['transport_error_type'] == 'ReadTimeout'
    assert events[0]['configured_timeout_s'] == 180
    assert len(events) == 1  # No extra Responses sampling or automatic replanning.


def test_bounded_audit_keeps_incomplete_reason_and_usage():
    from luna_mujoco.visual_api import audit_response
    body = {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
            'usage': {'output_tokens': 12000}, 'output': ['x' * 50000]}
    result = audit_response(body, '')
    assert result['status'] == 'incomplete'
    assert result['incomplete_details']['reason'] == 'max_output_tokens'
    assert result['usage']['output_tokens'] == 12000


async def test_reconfirming_frozen_inventory_preserves_original_selection_and_evidence(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.runtime.controller import PolicyController
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    frozen = {'selected_input_ids': ['original'], 'initial_entities': {'original': {'observation_id': 'initial'}}}
    runner.task_progress = SimpleNamespace(inventory_finalized=True, snapshot=lambda: deepcopy(frozen))
    runner._confirm_task_binding = AsyncMock(side_effect=AssertionError('must not reselect'))
    result = await runner.execute('confirm_inventory', {'observation': {'observation_id': 'new'}})
    assert result['success'] is True
    assert result['inventory'] == frozen
    assert result['confirmation_reused'] is True
    assert result['current_scene_reverified'] is False
    runner._confirm_task_binding.assert_not_awaited()


def test_recovery_schema_reference_has_no_api_unsupported_siblings():
    from test_agent_model_layer import CATALOG

    from luna_mujoco.agent.schema import InitialPlan, response_schema
    schema = response_schema(InitialPlan, CATALOG)

    def check(value):
        if isinstance(value, dict):
            if '$ref' in value:
                assert set(value) == {'$ref'}
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(schema)


def test_schema_rejection_records_redacted_api_validation_message():
    from luna_mujoco.visual_api import audit_response
    error = {'code': 'invalid_json_schema', 'message': 'bad schema secret-token'}
    assert audit_response({'error': error}, 'secret-token')['error']['message'] == 'bad schema [redacted]'


def test_responses_schema_has_skills_without_legacy_automatic_retry_nodes():
    from test_agent_model_layer import CATALOG

    from luna_mujoco.agent.schema import InitialPlan, response_schema
    catalog = deepcopy(CATALOG)
    catalog['inspect']['recovery_retry_safe'] = True
    schema = response_schema(InitialPlan, catalog)
    assert 'Recover' not in schema['$defs']
    all_skills = schema['$defs']['SkillCall']['anyOf']
    assert {item['properties']['skill']['const'] for item in all_skills} == {'pick', 'inspect'}
