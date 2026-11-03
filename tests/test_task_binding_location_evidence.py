"""The failed black-A2 binding must name its proposed entity and measured frame."""
import json
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import MethodType, SimpleNamespace

import httpx
import numpy as np
import pytest

from luna_mujoco.agent.decisions import DecisionsJudgments
from luna_mujoco.agent.schema import TaskSpec
from luna_mujoco.runtime.controller import PolicyController
from luna_mujoco.runtime.diagnostics import DiagnosticError


def frozen_case(name='black_a2_task_binding.json'):
    path = Path(__file__).parent / 'fixtures' / name
    return json.loads(path.read_text(encoding='utf-8'))


async def binding_wire(case, probability=.18, selected_id=None):
    original = case['context']
    observation = case['observation']
    payloads = []

    def respond(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        context = json.loads(payload['input'][0]['content'][0]['text'])['context']
        answers = []
        for question in payload['questions']:
            if question['type'] == 'choice':
                proposed = context['proposed_task_binding']['destination_id']
                value = next(row['value'] for row in context['destination_candidates']
                             if row['instance_id'] == (selected_id or proposed))
                answers.append({'type': 'choice', 'name': question['name'],
                                'choice': value, 'confidence': probability})
            else:
                answers.append({'type': 'predicate', 'name': question['name'],
                    'probability': probability if question['name'] == 'destination_requirement_matches' else 1.})
        return httpx.Response(200, json={'answers': answers})


    async def call_algorithm(function, *args, **kwargs):
        return function(*args, **kwargs)

    controller = SimpleNamespace(
        object_tasks={}, task_spec=TaskSpec.model_validate(original['task']),
        active_object_task=None, environment_assumptions={}, observation_epoch=0,
        compiler=SimpleNamespace(configured_regions=deepcopy(original['configured_regions'])),
        task_inventory={'source_object_ids': original['candidate_ids']},
        task_progress=SimpleNamespace(records={identifier: {} for identifier in original['candidate_ids']},
                                      scene_entities=observation['scene_entities']),
        judgments=DecisionsJudgments(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond))),
        state={'target_id': None, 'destination_id': None}, emit=lambda *args: None,
        visual_audit_sink=lambda: (None, Event()), call_algorithm=call_algorithm)
    controller._judge = MethodType(PolicyController._judge, controller)
    capture = {key: observation[key] for key in ('observation_id', 'generation')}
    capture['views'] = {'3': {'rgb': np.zeros((4, 4, 3), dtype=np.uint8)}}
    context = {'observation': observation, 'latest_capture': capture,
               'capture_epochs': {observation['observation_id']: 0},
               'visual_inventory_reference': {'entities': observation['scene_entities']}}
    with pytest.raises(DiagnosticError, match='task_binding_unconfirmed:destination_requirement_matches'):
        await PolicyController._confirm_task_binding(controller, context,
            TaskSpec.model_validate(original['object_task']), original['candidate_ids'])
    assert len(payloads) == 1
    return payloads[0], controller


@pytest.mark.parametrize('probability, expected', [(.18, 'unknown'), (.5, 'unknown')])
async def test_real_binding_wire_locates_proposed_destination_without_overriding_verdict(probability, expected):
    case = frozen_case()
    payload, controller = await binding_wire(case, probability)
    context = json.loads(payload['input'][0]['content'][0]['text'])['context']
    destination = case['context']['object_task']['destination_ref']
    targets = case['context']['candidate_ids']
    proposal = context['proposed_task_binding']
    assert proposal == {
        'status': 'proposed_not_yet_confirmed',
        'original_instruction': case['context']['task']['instruction'],
        'target_ids': targets, 'destination_id': destination,
        'destination_type': 'observed_entity'}
    assert context['target_id'] is None and context['destination_id'] is None
    entities = {row['id']: row for row in context['current_entities']}
    assert entities[destination]['visible_centroid_world'][1] == pytest.approx(.5304488275283166)
    assert entities[destination]['position'] == entities[destination]['visible_centroid_world']
    assert entities[destination]['observation_provenance']['observation_id'] == case['observation']['observation_id']
    assert entities[destination]['entity_type'] == 'observed_entity'
    assert '+Y is robot LEFT' in context['measured_location_rule']
    assert 'image left/right' in context['measured_location_rule']
    assert 'unqualified left/right mean ROBOT/world left/right' in context['measured_location_rule']
    assert 'viewpoint in the instruction overrides this default' in context['measured_location_rule']
    question = next(row for row in payload['questions'] if row['name'] == 'destination_requirement_matches')
    assert destination in {row['instance_id'] for row in context['destination_candidates']}
    assert question['type'] == 'choice'
    assert 'CURRENT visible entity' in question['instructions']
    assert 'destination_identity' not in {row['name'] for row in payload['questions']}
    answer = controller.state['task_binding_evidence']['judgments']['destination_requirement_matches']
    assert answer['status'] == expected and answer['confidence'] == probability
    assert answer['selected_instance_id'] is None


async def test_configured_destination_remains_a_supplied_region_not_a_visual_entity():
    case = frozen_case()
    case['context']['object_task']['destination_ref'] = 'demo-destination'
    case['observation']['objects'].append({'id': 'demo-destination', 'kind': 'marker',
        'position': [.5, .1, .57], 'coverage': {'configured_region': True},
        'placement_reference': {'source': 'configured_xy_measured_support',
            'support_entity_id': case['context']['object_task']['source_ref'],
            'observation_id': case['observation']['observation_id']}})
    payload, _ = await binding_wire(case)
    context = json.loads(payload['input'][0]['content'][0]['text'])['context']
    assert context['proposed_task_binding']['destination_type'] == 'configured_region'
    region = next(row for row in context['current_entities'] if row['id'] == 'demo-destination')
    assert region['entity_type'] == 'configured_region'
    assert 'visible_centroid_world' not in region
    names = {row['name'] for row in payload['questions']}
    assert 'destination_support_visible' in names and 'destination_identity' not in names


async def test_hybrid6_binds_support_without_claiming_a_free_placement():
    case = frozen_case('hybrid6_task_binding.json')
    payload, controller = await binding_wire(case, .69)
    context = json.loads(payload['input'][0]['content'][0]['text'])['context']
    question = next(row for row in payload['questions'] if row['name'] == 'destination_requirement_matches')
    assert 'requested support or container' in question['instructions']
    assert 'destination entity only' in question['instructions']
    assert 'empty landing space' in question['instructions']
    assert 'original side/viewpoint' in question['instructions']
    assert context['proposed_task_binding']['original_instruction'] == case['context']['task']['instruction']
    assert context['object_task'] == case['context']['object_task']
    assert '桌面上不被其他物体占用的区域' in context['task']['instruction']
    answer = controller.state['task_binding_evidence']['judgments']['destination_requirement_matches']
    assert answer['status'] == 'unknown' and answer['confidence'] == .69


async def test_processing_region_is_not_approved_as_an_unspecified_free_table_area():
    case = frozen_case('hybrid6_task_binding.json')
    case['context']['object_task']['destination_ref'] = 'vla-unpack-input'
    case['observation']['objects'].append({'id': 'vla-unpack-input', 'kind': 'marker',
        'placement_reference': {'source': 'configured_xy_measured_support',
            'support_entity_id': 'instance-55f24845d9234605',
            'observation_id': case['observation']['observation_id']}})
    payload, controller = await binding_wire(case, .18)
    question = next(row for row in payload['questions'] if row['name'] == 'destination_requirement_matches')
    assert 'unspecified free table area does not authorize a processing region' in question['instructions']
    assert 'original instruction or required processing' in question['instructions']
    assert controller.state['task_binding_evidence']['judgments']['destination_requirement_matches'] == {
        'status': 'no', 'probability': .18}


async def test_black_a3_destination_question_is_short_and_contains_original_instruction():
    case = frozen_case('black_a3_task_binding.json')
    payload, controller = await binding_wire(case, .47)
    questions = {row['name']: row for row in payload['questions']}
    question = questions['destination_requirement_matches']['instructions']
    assert case['context']['task']['instruction'] in question
    assert len(question.split()) <= 120
    context = json.loads(payload['input'][0]['content'][0]['text'])['context']
    assert case['context']['object_task']['destination_ref'] in {row['instance_id'] for row in context['destination_candidates']}
    assert case['context']['object_task']['target_selector'] in question
    assert 'destination entity only' in question
    assert 'original side/viewpoint' in question
    assert 'checked by placement skills' in question
    for original in case['original_questions']:
        if original['name'] not in {'destination_requirement_matches', 'destination_identity', 'source_identity'}:
            assert questions[original['name']] == original
    source_question = questions['source_identity']['instructions']
    assert 'ORIGINAL supporting surface, containing receptacle or source region' in source_question
    assert 'An input object cannot be its own source' in source_question
    answer = controller.state['task_binding_evidence']['judgments']['destination_requirement_matches']
    assert answer['status'] == 'unknown' and answer['confidence'] == .47


async def test_a10_production_has_one_destination_choice_without_redundant_predicate():
    case = frozen_case('black_a10_task_binding.json')
    payload, _ = await binding_wire(case, .01)
    questions = {row['name']: row for row in payload['questions']}
    assert set(questions) == {'source_identity', 'destination_requirement_matches', 'target_0'}
    assert questions['destination_requirement_matches']['type'] == 'choice'
    assert case['context']['object_task']['target_selector'] in questions['destination_requirement_matches']['instructions']


async def test_confident_different_destination_fails_without_rebinding_the_task():
    case = frozen_case('black_a10_task_binding.json')
    proposed = case['context']['object_task']['destination_ref']
    other = next(row['id'] for row in case['observation']['objects']
                 if row.get('kind') == 'bin' and row['id'] != proposed)
    payload, controller = await binding_wire(case, .99, selected_id=other)
    answer = controller.state['task_binding_evidence']['judgments']['destination_requirement_matches']
    assert answer['status'] == 'no' and answer['selected_instance_id'] == other
    assert controller.task_spec.destination_ref == case['context']['task']['destination_ref']
    assert answer['proposed_instance_id'] == proposed
    assert len(payload['questions']) == 3
