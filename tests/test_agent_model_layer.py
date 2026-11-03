import json
from copy import deepcopy

import numpy as np
import pytest

CATALOG = {
    'pick': {'description': 'Pick an identified object', 'arguments': {
        'type': 'object', 'properties': {'target_id': {'type': 'string'}},
        'required': ['target_id'], 'additionalProperties': False}},
    'inspect': {'description': 'Observe current evidence', 'arguments': {
        'type': 'object', 'properties': {}, 'additionalProperties': False}},
}


def task_data():
    return {'instruction': 'Move the red items to the tray', 'supported': True, 'reason': '',
                'scope': 'all', 'target_selector': 'red items', 'target_refs': ['item'], 'quantity': None,
                'source_ref': 'table', 'destination_ref': 'tray', 'work_area_ref': None,
                'selection_order': 'nearest', 'processing': [], 'completion_conditions': ['in tray'],
                'completion_pose': None,
                'constraints': ['do not move other objects'], 'source_inventory_complete': 'confirmed'}


def plan_data():
    def entity(ref, kind, color, parent, membership='excluded'):
        return {'ref': ref, 'kind': kind, 'color': color, 'description': color + ' ' + kind,
                    'bbox': [.1, .1, .5, .5], 'reference_camera_id': '3', 'parent_ref': parent,
                    'membership': membership, 'segmentation_prompt': color + ' ' + kind, 'asset_id': None,
                    'appearance_keywords': [], 'segmentation_aliases': []}
    return {'entities': [entity('table', 'table', 'white', None),
                         entity('tray', 'bin', 'green', 'table'),
                         entity('item', 'object', 'red', 'table', 'required')],
            'task': task_data(), 'graph': {'version': 2, 'id': 'task', 'entry': 'pick_item',
                'subgraphs': [], 'nodes': [
                    {'type': 'skill', 'id': 'pick_item', 'skill': 'pick', 'arguments': {'target_id': 'item'}},
                    {'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''}],
                'edges': [{'id': 'picked', 'source': 'pick_item', 'target': 'done', 'on': 'success'}]}}


def capture():
    return {'observation_id': 'obs-1', 'generation': 1,
            'views': {camera: {'rgb': np.zeros((8, 8, 3), dtype=np.uint8)} for camera in ('1', '2', '3')}}


class Client:
    def __init__(self, outputs):
        self.outputs, self.requests = list(outputs), []

    def post(self, url, **kwargs):
        import httpx
        self.requests.append((url, kwargs['json']))
        output = self.outputs.pop(0)
        body = output if 'answers' in output else {'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(output)}]}]}
        return httpx.Response(200, json=body)


def test_bootstrap_observation_reuse_is_explained_without_inserting_graph_steps():
    from luna_mujoco.agent.responses import ResponsesCompiler

    client = Client([plan_data()])
    compiler = ResponsesCompiler(registry=CATALOG, api_key='test', client=client)
    result = compiler.compile(capture(), instruction=task_data()['instruction'])
    context = json.loads(client.requests[0][1]['input'][0]['content'][0]['text'])
    assert not context['initial_facts']['observed'] and not context['initial_facts']['grasp_view_ready']
    assert context['initial_facts']['inventory_finalized'] is False
    assert 'graph (version=2)' in context['rules']
    assert result.graph.model_dump() == plan_data()['graph']


def test_initial_compiler_receives_public_capability_configuration_without_rewriting_graph():
    from luna_mujoco.agent.responses import ResponsesCompiler
    context = {'capabilities': {'configured_process': {'required_execution_arm': 'A',
        'input_center_world_xy_m': [.4, .2], 'input_xy_tolerance_m': .002}}}
    client = Client([plan_data()])
    compiler = ResponsesCompiler(registry=CATALOG, api_key='test', client=client, scene_context=context)
    plan = compiler.compile(capture(), instruction=task_data()['instruction'])
    payload = json.loads(client.requests[0][1]['input'][0]['content'][0]['text'])
    assert payload['scene_context'] == context
    assert plan.graph.model_dump() == plan_data()['graph']


@pytest.mark.parametrize('completion_pose', [None, 'observation'])
def test_compiler_preserves_optional_requested_completion_pose(completion_pose):
    from luna_mujoco.agent.responses import ResponsesCompiler
    output = plan_data()
    output['task']['completion_pose'] = completion_pose
    if completion_pose:
        output['task']['instruction'] += ' and return to the configured observation pose'
    client = Client([output])
    compiler = ResponsesCompiler(registry=CATALOG, api_key='test', client=client)
    result = compiler.compile(capture(), instruction=output['task']['instruction'])
    assert result.task.completion_pose == completion_pose
    assert result.graph.model_dump() == output['graph']
    context = json.loads(client.requests[0][1]['input'][0]['content'][0]['text'])
    assert 'completion_requirements' in context['rules']


def test_completion_pose_defaults_to_none_and_rejects_unavailable_pose():
    from luna_mujoco.agent.schema import TaskSpec
    legacy = task_data()
    legacy.pop('completion_pose')
    assert TaskSpec.model_validate(legacy).completion_pose is None
    with pytest.raises(ValueError):
        TaskSpec.model_validate(task_data() | {'completion_pose': 'arbitrary-home'})


@pytest.mark.parametrize('fault', ['empty', 'excluded'])
def test_supported_task_cannot_omit_or_contradict_its_target_inventory(fault):
    from luna_mujoco.agent.schema import InitialPlan

    plan = plan_data()
    if fault == 'empty':
        plan['task']['target_refs'] = []
    else:
        plan['entities'][-1]['membership'] = 'excluded'
    with pytest.raises(ValueError, match='target'):
        InitialPlan.model_validate(plan)


def test_one_initial_responses_compiles_generic_inventory_and_keeps_decisions_separate():
    from luna_mujoco.agent.responses import ResponsesCompiler
    client = Client([plan_data()])
    compiler = ResponsesCompiler(registry=CATALOG, api_key='secret', client=client)
    result = compiler.discover(capture(), instruction=task_data()['instruction'])
    assert len(client.requests) == 1
    assert client.requests[0][0].endswith('/responses')
    assert result['task_spec']['target_selector'] == 'red items'
    assert result['task_inventory']['source_object_ids'] == result['task_spec']['target_refs']
    assert result['entities'][result['task_spec']['destination_ref']]['color'] == 'green'
    assert result['policy_graph'] == result['reference']['policy_graph']
    assert 'demo_task' not in result and 'demo_program' not in result
    assert compiler.budget.used == 1
    assert all(entity['identity_source'] == 'openai_responses_proposal' for entity in result['entities'].values())


@pytest.mark.parametrize('quoted_instruction', ['red items have no CAD', 'invented no CAD instruction'])
def test_explicit_unknown_geometry_policy_is_grounded_and_bound_to_permanent_id(quoted_instruction):
    from luna_mujoco.agent.responses import ResponsesBudget, ResponsesCompiler
    from luna_mujoco.visual_inventory import VisualInventoryError
    output = plan_data()
    instruction = output['task']['instruction'] + '; red items have no CAD'
    output['task']['instruction'] = instruction
    output['graph'] = {'version': 2, 'id': 'task', 'entry': 'done', 'subgraphs': [],
        'nodes': [{'type': 'end', 'id': 'done', 'status': 'success', 'reason': ''}], 'edges': []}
    ref = output['task']['target_refs'][0]
    entity = next(item for item in output['entities'] if item['ref'] == ref)
    entity.update(geometry_policy='unknown_obb', geometry_policy_instruction=quoted_instruction)
    client = Client([output])
    compiler = ResponsesCompiler(registry=CATALOG, api_key='test', client=client,
        budget=ResponsesBudget())
    if quoted_instruction not in instruction:
        with pytest.raises(VisualInventoryError):
            compiler.discover(capture(), instruction=instruction)
    else:
        result = compiler.discover(capture(), instruction=instruction)
        identifier = result['task_spec']['target_refs'][0]
        policy = result['entities'][identifier]['geometry_policy']
        assert policy['instance_id'] == identifier and identifier.startswith('instance-')
        assert policy['instruction_quote'] == quoted_instruction
        assert policy['source'] == 'user_instruction_interpreted_by_responses'
        assert 'asset_id' not in result['entities'][identifier]
    assert len(client.requests) == 1


def test_configured_region_cannot_be_registered_as_a_visible_entity():
    from luna_mujoco.agent.responses import ResponsesCompiler

    invalid = plan_data()
    fake = deepcopy(invalid['entities'][-1])
    fake.update(ref='unmarked-region', membership='excluded')
    invalid['entities'].append(fake)
    corrected = plan_data()
    corrected['task']['destination_ref'] = 'unmarked-region'
    client = Client([invalid, corrected])
    compiler = ResponsesCompiler(registry=CATALOG, api_key='test', client=client,
                                 configured_regions={'unmarked-region': {'center_world_xy_m': [.4, .1]}})
    from luna_mujoco.visual_inventory import VisualInventoryError
    with pytest.raises(VisualInventoryError) as error:
        compiler.compile(capture(), instruction=task_data()['instruction'])
    assert len(client.requests) == 1
    assert error.value.evidence['validation_errors'] == 'configured_region_is_not_visual_entity:unmarked-region'
    assert error.value.evidence['rejected_proposal'] == invalid


def test_decisions_reports_unknown_for_ambiguous_evidence_and_rejects_wrong_answer_name():
    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest
    request = JudgmentRequest(name='in_tray', instructions='Is this item visibly in the tray?')
    client = Client([{'answers': [{'type': 'predicate', 'name': 'in_tray', 'probability': .55}]}])
    judge = DecisionsJudgments(api_key='secret', client=client)
    result = judge.judge(capture(), requests=[request], context={'target_id': 'item'})
    assert result['judgments']['in_tray']['status'] == 'unknown'
    assert client.requests[0][0].endswith('/decisions')
    client = Client([{'answers': [{'type': 'predicate', 'name': 'other', 'probability': .99}]}])
    with pytest.raises(ValueError, match='invalid_answer'):
        DecisionsJudgments(api_key='secret', client=client).judge(capture(), requests=[request], context={})


def test_decisions_choices_are_bounded_and_cached_on_the_same_evidence():
    from luna_mujoco.agent.decisions import ChoiceRequest, DecisionsJudgments
    request = ChoiceRequest(name='arm', instructions='Select among feasible arms', choices=('A', 'B', 'unknown'))
    client = Client([{'answers': [{'type': 'choice', 'name': 'arm', 'choice': 'B', 'confidence': .9}]}])
    judge = DecisionsJudgments(api_key='secret', client=client)
    result = judge.judge(capture(), requests=[request], context={'feasible': ['A', 'B']})
    assert result['judgments']['arm'] == {'status': 'known', 'value': 'B', 'confidence': .9}
    assert judge.judge(capture(), requests=[request], context={'feasible': ['A', 'B']}) == result
    assert len(client.requests) == 1


def test_large_decisions_inventory_batches_same_frame_and_retains_each_request():
    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest
    requests = [JudgmentRequest(name=f'target_{index}', instructions='Is this target eligible?')
                for index in range(1027)]
    client = Client([{'answers': [{'type': 'predicate', 'name': request.name, 'probability': .99}
                                 for request in requests[start:start + 200]]}
                     for start in range(0, len(requests), 200)])
    judge = DecisionsJudgments(api_key='secret', client=client)
    result = judge.judge(capture(), requests=requests, context={'source_id': 'table'})
    assert len(result['judgments']) == 1027
    assert len(client.requests) == len(result['evidence']['batches']) == 6
    assert all(len(payload['questions']) <= 200 for _, payload in client.requests)
    assert len({json.dumps(payload['input']) for _, payload in client.requests}) == 1
    assert judge.judge(capture(), requests=requests, context={'source_id': 'table'}) == result
    assert len(client.requests) == 6


def test_action_proposal_does_not_weaken_fact_confidence_or_reuse_its_cached_verdict():
    from luna_mujoco.agent.decisions import ChoiceRequest, DecisionsJudgments
    request = ChoiceRequest(name='scope', instructions='Select scope', choices=('global', 'local', 'unknown'))
    body = {'answers': [{'type': 'choice', 'name': 'scope', 'choice': 'global', 'confidence': .72}]}
    client = Client([deepcopy(body), deepcopy(body)])
    judge = DecisionsJudgments(api_key='secret', client=client)
    fact = judge.judge(capture(), requests=[request], context={})
    assert fact['judgments']['scope'] == {'status': 'unknown', 'value': 'unknown', 'confidence': .72}
    proposal = request.model_copy(update={'selection_mode': 'proposal'})
    action = judge.judge(capture(), requests=[proposal], context={})
    assert action['judgments']['scope'] == {'status': 'proposed', 'value': 'global',
        'confidence': .72, 'selection_mode': 'proposal'}
    assert fact['evidence']['request_hash'] != action['evidence']['request_hash']
    assert judge.judge(capture(), requests=[request], context={}) == fact
    assert judge.judge(capture(), requests=[proposal], context={}) == action
    assert len(client.requests) == 2


def test_processing_capability_does_not_generate_a_fixed_lifecycle():
    from luna_mujoco.agent.responses import ResponsesCompiler
    output = plan_data()
    output['task']['processing'] = ['vla_unpack']
    catalog = {**CATALOG, 'vla_unpack': {**CATALOG['inspect'], 'processing_effect': {'fixture': True}}}
    compiler = ResponsesCompiler(registry=catalog, api_key='secret', client=Client([output]))
    result = compiler.discover(capture(), instruction=task_data()['instruction'])
    inventory = result['task_inventory']
    assert inventory['destination_id'] == result['task_spec']['destination_ref']
    assert 'workflow' not in inventory and 'item_lifecycle' not in inventory
    assert 'demo_task' not in result['reference']


def test_decisions_negative_probability_boundary_is_inclusive():
    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest
    client = Client([{'answers': [{'type': 'predicate', 'name': 'visible', 'probability': .2}]}])
    result = DecisionsJudgments(api_key='secret', client=client).judge(
        capture(), requests=[JudgmentRequest(name='visible', instructions='Is item visible?')], context={})
    assert result['judgments']['visible']['status'] == 'no'


def test_task_keeps_user_quantity_when_visible_inventory_is_incomplete():
    from luna_mujoco.agent.schema import TaskSpec
    task = task_data()
    task.update(scope='count', quantity=5, source_inventory_complete='unknown')
    assert TaskSpec.model_validate(task).quantity == 5


def test_responses_schema_has_closed_per_skill_arguments_and_no_arbitrary_json_definition():
    from luna_mujoco.agent.schema import InitialPlan, response_schema
    schema = response_schema(InitialPlan, CATALOG)
    variants = schema['$defs']['SkillCall']['anyOf']
    assert {variant['properties']['skill']['const'] for variant in variants} == set(CATALOG)
    assert all(variant['properties']['arguments']['additionalProperties'] is False for variant in variants)
    assert 'JsonValue' not in schema['$defs']


def test_support_binding_identifies_a_table_not_a_marker_or_bin():
    from luna_mujoco.agent.responses import ResponsesCompiler
    compiler = ResponsesCompiler(registry=CATALOG, api_key='secret', client=Client([plan_data()]))
    result = compiler.discover(capture(), instruction=task_data()['instruction'])
    table_id = result['task_inventory']['support_id']
    assert result['entities'][table_id]['kind'] == 'table'


def test_compiler_reset_starts_a_new_task_with_new_budget_and_inventory():
    from luna_mujoco.agent.responses import ResponsesCompiler
    client = Client([plan_data(), plan_data()])
    compiler = ResponsesCompiler(registry=CATALOG, api_key='secret', client=client)
    first = compiler.discover(capture(), instruction=task_data()['instruction'])
    compiler.reset()
    assert compiler.budget.used == 0 and compiler.original_task is None
    second = compiler.discover(capture(), instruction=task_data()['instruction'])
    assert compiler.budget.used == 1 and len(client.requests) == 2
    assert first['task_spec']['target_refs'] != second['task_spec']['target_refs']
    prompt = json.loads(client.requests[1][1]['input'][0]['content'][0]['text'])
    assert prompt['initial_facts']['inventory_finalized'] is False
    assert prompt['initial_facts']['target_selected'] is False


def test_decisions_reference_grounding_includes_canonical_entities_and_boxes():
    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest
    client = Client([{'answers': [{'type': 'predicate', 'name': 'visible', 'probability': .9}]}])
    reference = {'views': capture()['views'], 'entities': {'fixed-item': {
        'id': 'fixed-item', 'kind': 'object', 'color': 'red', 'description': 'red cube',
        'bbox': [.1, .1, .5, .5], 'reference_camera_id': '3'}}}
    DecisionsJudgments(api_key='secret', client=client).judge(
        capture(), requests=[JudgmentRequest(name='visible', instructions='Is fixed-item visible?')],
        context={'target_id': 'fixed-item'}, reference=reference)
    content = client.requests[0][1]['input'][0]['content']
    text = '\n'.join(item['text'] for item in content if item['type'] == 'input_text')
    assert 'reference_entities' in text
    assert 'red cube' in text
    assert 'initial_reference_annotations' in text


def test_configured_assets_bind_inventory_without_scene_pose_or_orientation():
    from luna_mujoco.agent.responses import ResponsesCompiler
    from luna_mujoco.cad_assets import cad_asset

    plan = plan_data()
    plan['entities'][1]['asset_id'] = 'blue_box'
    asset = cad_asset('blue_box')
    known = {'blue_box': {'label': asset['label'], 'size_m': asset['size_m'],
                          'actual_world_pose': 'never-send'}}
    client = Client([plan])
    compiler = ResponsesCompiler(registry=CATALOG, known_assets=known, api_key='test', client=client)
    discovery = compiler.discover(capture(), instruction=task_data()['instruction'])
    entity = discovery['entities'][discovery['task_spec']['destination_ref']]
    assert entity['asset_id'] == 'blue_box'
    assert entity['asset_source'] == 'configured_catalog_responses_proposal'
    assert 'geometry_prior' not in entity and 'pose' not in entity
    context = json.loads(client.requests[0][1]['input'][0]['content'][0]['text'])
    assert context['known_assets'] == {'blue_box': {'label': asset['label'], 'size_m': asset['size_m']}}
    assert 'never-send' not in json.dumps(client.requests)
    assert len(client.requests) == 1 and compiler.budget.used == 1


@pytest.mark.parametrize('configured', [False, True])
def test_compiler_rejects_asset_not_in_explicit_catalog(configured):
    from luna_mujoco.agent.responses import ResponsesBudget, ResponsesCompiler
    from luna_mujoco.visual_api import VisualInventoryError

    plan = plan_data()
    plan['entities'][1]['asset_id'] = 'invented' if configured else 'blue_box'
    assets = {'blue_box': {'label': 'Declared container', 'size_m': [.3, .4, .2]}} if configured else {}
    compiler = ResponsesCompiler(registry=CATALOG, known_assets=assets, budget=ResponsesBudget(),
                                 api_key='test', client=Client([plan]))
    with pytest.raises(VisualInventoryError, match='responses_invalid_plan'):
        compiler.compile(capture(), instruction=task_data()['instruction'])


def test_generic_compiler_has_no_implicit_asset_binding():
    from luna_mujoco.agent.responses import ResponsesCompiler

    client = Client([plan_data()])
    discovery = ResponsesCompiler(registry=CATALOG, api_key='test', client=client).discover(
        capture(), instruction=task_data()['instruction'])
    assert all('asset_id' not in entity for entity in discovery['entities'].values())
    context = json.loads(client.requests[0][1]['input'][0]['content'][0]['text'])
    assert context['known_assets'] == {}


def test_configured_container_identity_never_becomes_mapping_geometry():
    from test_demo_mapping import table_evidence
    from test_rgbd_algorithms import observed_scene

    from luna_mujoco.agent.responses import ResponsesCompiler
    from luna_mujoco.algorithms import Algorithms
    from luna_mujoco.cad_assets import cad_asset
    from luna_mujoco.perception_mapping import decode_mapping_payload

    asset = cad_asset('blue_box')
    plan = plan_data()
    plan['entities'][1]['asset_id'] = 'blue_box'
    client = Client([plan])
    compiler = ResponsesCompiler(registry=CATALOG, known_assets={
        'blue_box': {'label': asset['label'], 'size_m': asset['size_m']}}, api_key='test', client=client)
    inventory = compiler.discover(capture(), instruction=task_data()['instruction'])
    destination_id = inventory['task_spec']['destination_ref']
    snapshot = observed_scene()
    snapshot.update(scene_entities=inventory['entities'], demo_mapping=True, demo_tables=table_evidence())
    snapshot['objects'][0].update(id=destination_id, kind='bin', color='green')
    for view in snapshot['views'].values():
        view['instance_masks'] = {destination_id: view['instance_masks']['seen-red']}
    measured_points = snapshot['objects'][0]['points_world_m'].copy()
    calls = []
    algorithms = Algorithms(curobo_request=lambda path, body: (calls.append(body) or {'scene_token': 'mapped'}))
    algorithms.build_scene(snapshot, destination_id)
    without_asset = deepcopy(snapshot)
    without_asset['scene_entities'][destination_id].pop('asset_id')
    algorithms.build_scene(without_asset, destination_id)
    payload, _ = decode_mapping_payload(calls[0])
    baseline, _ = decode_mapping_payload(calls[1])
    assert inventory['entities'][destination_id]['asset_id'] == 'blue_box'
    assert 'demo_bin_priors' not in calls[0] and 'demo_completions' not in calls[0]
    assert 'demo_bin_priors' not in payload and 'demo_completions' not in payload
    np.testing.assert_array_equal(snapshot['objects'][0]['points_world_m'], measured_points)
    for camera, view in payload['views'].items():
        np.testing.assert_array_equal(view['depth_left_m'], baseline['views'][camera]['depth_left_m'])
        np.testing.assert_array_equal(view['instance_masks'][destination_id],
                                      baseline['views'][camera]['instance_masks'][destination_id])
    assert 'geometry_prior' not in inventory['entities'][destination_id]
    assert len(client.requests) == 1
