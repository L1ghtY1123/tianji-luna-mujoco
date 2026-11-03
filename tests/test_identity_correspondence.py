import json
from copy import deepcopy

import httpx
import pytest
from test_identity_instruction_factoring import original_instructions

from luna_mujoco.identity_correspondence import (
    correspondence_payload,
    request_identity_correspondences,
    resolve_correspondences,
)
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import VisualInventory


def payload():
    return {'input': [{'content': [
        {'type': 'input_text', 'text': '{}'},
        {'type': 'input_text', 'text': json.dumps({'candidate': 'candidate-2', 'camera': '3'})},
    ]}], 'questions': [
        {'type': 'choice', 'name': 'identity_2', 'instructions': 'Match C2', 'choices': [
            {'value': identifier, 'description': identifier}
            for identifier in ('left-bin', 'right-bin', 'product', 'reject', 'unknown')]},
        {'type': 'choice', 'name': 'visibility_product', 'instructions': 'Visible?',
         'choices': [{'value': 'visible', 'description': 'visible'}]},
    ]}


def response(request, probabilities):
    return {'answers': [
        {'type': 'predicate', 'name': question['name'], 'probability': probabilities[index]}
        if question['type'] == 'predicate' else
        {'type': 'choice', 'name': question['name'], 'choice': 'visible', 'confidence': .99}
        for index, question in enumerate(request['questions'])]}


@pytest.mark.parametrize('changed', [None, 'observation', 'generation', 'reference_camera'])
def test_pairwise_preserves_same_capture_motion_boundary_and_measured_location(changed):
    original = payload()
    context = {'observation_id': 'capture', 'generation': 3,
        'fixed_entities': {'left-bin': {'description': 'blue container'}},
        'same_capture_reference': {'initial_reference_observation_id': 'capture',
            'current_observation_id': 'capture', 'generation': 3, 'verified_camera_ids': ['2', '3']}}
    if changed == 'observation':
        context['same_capture_reference']['initial_reference_observation_id'] = 'old'
    elif changed == 'generation':
        context['same_capture_reference']['generation'] = 2
    original['input'][0]['content'][0]['text'] = json.dumps(context)
    original['input'][0]['content'].append({'type': 'input_text', 'text': json.dumps({
        'image_role': 'initial_reference_annotations',
        'camera': '1' if changed == 'reference_camera' else '2',
        'items': [{'id': 'left-bin', 'box': [0, 0, 1, 1]}]})})
    original['questions'][0]['instructions'] += ' Current measured candidate surface: {"camera": "3", "y": -0.6}'
    request, _ = correspondence_payload(original)
    text = original_instructions(request, request['questions'][0])
    assert ('no intervening motion' in text) is (changed is None)
    assert ('Historical positions' in text) is (changed is not None)
    assert 'Current measured candidate surface: {"camera": "3", "y": -0.6}' in text
    assert 'Historical positions' in original_instructions(request, request['questions'][1])


def test_pairwise_scopes_identical_region_hint_to_its_candidate_reference_and_camera():
    original = payload()
    context = {'observation_id': 'current', 'generation': 3,
        'identical_reference_regions': [{
            'candidate': 'candidate-2', 'reference_entity_id': 'right-bin', 'camera': '3',
            'reference_observation_id': 'initial', 'current_observation_id': 'current',
            'generation': 3, 'union_roi_xyxy_px': [10, 20, 30, 40],
            'rgb_sha256': 'a' * 64, 'depth_m_sha256': 'b' * 64, 'valid_depth_sha256': 'c' * 64}]}
    original['input'][0]['content'][0]['text'] = json.dumps(context)
    request, pairs = correspondence_payload(original)
    instruction = original_instructions(request, request['questions'][1])
    assert 'different acquisitions' in instruction
    assert 'byte-identical RGB, depth and validity' in instruction
    assert 'directly compare their 2D positions' in instruction
    assert 'SAME verified frozen capture' not in instruction
    assert 'Historical positions' not in instruction
    assert 'Historical positions' in original_instructions(request, request['questions'][0])
    assert 'Historical positions' in original_instructions(request, request['questions'][2])
    result, _ = resolve_correspondences(response(request, [.01, .48, .01]), request, original, pairs, .8)
    assert result['answers'][0]['choice'] == 'unknown'


@pytest.mark.parametrize('field,value', [
    ('candidate', 'candidate-7'), ('camera', '1'), ('current_observation_id', 'stale'),
    ('generation', 2), ('reference_observation_id', 'current')])
def test_pairwise_does_not_inherit_another_candidate_or_capture_pixel_claim(field, value):
    original = payload()
    row = {'candidate': 'candidate-2', 'reference_entity_id': 'right-bin', 'camera': '3',
        'reference_observation_id': 'initial', 'current_observation_id': 'current', 'generation': 3,
        'union_roi_xyxy_px': [10, 20, 30, 40]}
    row[field] = value
    original['input'][0]['content'][0]['text'] = json.dumps({
        'observation_id': 'current', 'generation': 3, 'identical_reference_regions': [row]})
    request, _ = correspondence_payload(original)
    assert all('byte-identical' not in original_instructions(request, question)
               for question in request['questions'])


@pytest.mark.parametrize('probabilities,expected', [
    ([0., 0., .99], 'product'), ([.2, 0., .8], 'product'),
    ([.99, .01, .99], 'unknown'), ([.01, .4, .99], 'unknown'),
    ([.01, .01, .79], 'unknown'), ([0., 0., 0.], 'unknown'),
])
def test_identity_needs_one_positive_and_every_alternative_negative(probabilities, expected):
    original = payload()
    frozen = deepcopy(original)
    request, pairs = correspondence_payload(original)
    body = response(request, probabilities)
    resolved, evidence = resolve_correspondences(body, request, original, pairs, .8)
    assert original == frozen
    assert len(request['questions']) == 4
    assert resolved['answers'][0]['choice'] == expected
    assert resolved['answers'][1] == body['answers'][-1]
    assert evidence['original_decisions_answers'] == body['answers']
    assert evidence['identity_correspondence'][0]['probabilities'] == dict(
        zip(('left-bin', 'right-bin', 'product'), probabilities, strict=True))
    assert evidence['identity_correspondence'][0]['metric'] == 'minimum_pairwise_agreement'
    assert resolved['answers'][0]['confidence'] >= .8 if expected == 'product' else True


@pytest.mark.parametrize('error', ['nan', 'missing', 'reordered', 'wrong_type', 'boolean'])
def test_invalid_pair_answers_cannot_establish_identity(error):
    original = payload()
    request, pairs = correspondence_payload(original)
    body = response(request, [0., 0., .99])
    if error == 'missing':
        body['answers'].pop(0)
    elif error == 'reordered':
        body['answers'][0], body['answers'][1] = body['answers'][1], body['answers'][0]
    else:
        body['answers'][0].update({'nan': {'probability': float('nan')},
                                  'wrong_type': {'type': 'choice'},
                                  'boolean': {'probability': False}}[error])
    with pytest.raises(ValueError, match='invalid_response'):
        resolve_correspondences(body, request, original, pairs, .8)


def test_expansion_is_bounded_without_silently_dropping_identity_alternatives():
    original = payload()
    original['questions'][0]['choices'] = [
        {'value': str(index), 'description': str(index)} for index in range(200)]
    with pytest.raises(ValueError, match='input_limit'):
        correspondence_payload(original)


def test_late_candidate_conflict_remains_first_in_failure_diagnostics():
    original = payload()
    question = original['questions'][0]
    original['questions'] = []
    for index in range(20):
        original['questions'].append({**deepcopy(question), 'name': f'identity_{index}'})
        original['input'][0]['content'].append({'type': 'input_text', 'text': json.dumps(
            {'candidate': f'candidate-{index}', 'camera': '3'})})

    def respond(request):
        sent = json.loads(request.content)
        answers = [{'type': 'predicate', 'name': q['name'], 'probability':
                    .99 if q['name'] in {'match_19_0', 'match_19_2'} else 0.}
                   for q in sent['questions']]
        return httpx.Response(200, json={'answers': answers})

    visual = VisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))

    def reject(body, evidence):
        assert body['answers'][19]['choice'] == 'unknown'
        details = visual._identity_failure_evidence(evidence, {'camera_id': '3', 'prompt': 'object',
            'score': .9}, 'candidate-19', body['answers'][19])
        raise VisualInventoryError('visual_identity_unknown', evidence=details)

    with pytest.raises(VisualInventoryError, match='identity_unknown') as caught:
        request_identity_correspondences(visual, original, {'observation_id': 'current'}, 'associate_inventory', reject)
    details = caught.value.evidence
    assert len(details['original_decisions_answers']) == 60
    assert [check['name'] for check in details['diagnostic_checks'][:4]] == [
        'identity_19', 'match_19_0', 'match_19_1', 'match_19_2']
    assert all(check['candidate_id'] == 'candidate-19' for check in details['diagnostic_checks'][:4])
    assert details['diagnostic_checks'][0]['reason'] == 'exactly_one_positive_and_all_alternatives_negative'
