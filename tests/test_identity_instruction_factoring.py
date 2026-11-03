"""Shared presentation must preserve every original identity instruction exactly."""
import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from luna_mujoco.decision_conditions import parse_answers
from luna_mujoco.identity_correspondence import (
    correspondence_payload,
    factor_identity_instructions,
    metadata,
    resolve_correspondences,
)
from luna_mujoco.visual_api import VisualInventoryError


def shared_instructions(request):
    return next(metadata(part)['factored_identity_instructions']
                for part in request['input'][0]['content']
                if 'factored_identity_instructions' in metadata(part))


def original_instructions(request, question):
    """Independent inverse of the public request presentation, used by semantic tests."""
    if not question['name'].startswith('match_'):
        return question['instructions']
    shared = shared_instructions(request)
    number, camera, identifier = re.fullmatch(
        r'The physical subject marked Candidate (\d+) in CURRENT camera (\S+) '
        r'is the same individual physical entity as permanent reference ID (\S+)\.',
        question['instructions'].partition(' Reference subject: ')[0]).groups()
    fragments = shared['shared_instruction_fragments']
    if question['name'] in shared['instruction_overrides_by_question']:
        fragments = shared['instruction_variants'][shared['instruction_overrides_by_question'][question['name']]]
    result = (f'The physical subject marked Candidate {number} in CURRENT camera {camera} '
        f'is the same individual physical entity as reference '
        f"{shared['original_reference_labels_and_descriptions_by_id'][identifier]} (ID {identifier})."
        + fragments['before_candidate_measurement'])
    if fragments['has_candidate_measurement']:
        candidate = 'candidate-' + number
        measurements = {row['candidate']: row for row in metadata(
            request['input'][0]['content'][0]).get('candidate_geometric_location_hints', [])}
        encoded = shared['additional_candidate_measurements'].get(candidate)
        if encoded is None:
            encoded = json.dumps(measurements[candidate])
        result += ' Current measured candidate surface: ' + encoded
    return result + fragments['after_candidate_measurement']


def test_pairwise_factors_rules_once_without_changing_input_images_or_question_contract():
    from test_identity_correspondence import payload

    original = payload()
    measurement = {'candidate': 'candidate-2', 'camera': '3', 'y': -.6}
    original['input'][0]['content'][0]['text'] = json.dumps({
        'candidate_geometric_location_hints': [measurement]})
    original['input'][0]['content'].append({'type': 'input_image', 'image_url': 'original-pixels'})
    original['questions'][0]['instructions'] += (
        ' Current measured candidate surface: ' + json.dumps(measurement) + ' Geometry is not identity.')
    before = deepcopy(original)
    request, pairs = correspondence_payload(original)
    assert original == before
    assert request['input'][0]['content'][:-1] == original['input'][0]['content']
    shared = shared_instructions(request)
    assert shared['additional_candidate_measurements'] == {}
    assert shared['instruction_overrides_by_question'] == {}
    assert request['questions'][-1] == original['questions'][-1]
    assert pairs['identity_2'] == {'match_2_0': 'left-bin', 'match_2_1': 'right-bin', 'match_2_2': 'product'}
    assert json.dumps(request).count('An object and its surrounding container are different entities.') == 1
    assert json.dumps(request).count('Geometry is not identity.') == 1
    for question in request['questions'][:-1]:
        assert len(question['instructions']) < 350
        restored = original_instructions(request, question)
        assert 'Current measured candidate surface: ' + json.dumps(measurement) in restored
        assert restored.endswith(' Geometry is not identity.')


def test_pairwise_without_match_questions_does_not_add_unused_context():
    from test_identity_correspondence import payload

    original = payload()
    original['questions'] = original['questions'][1:]
    request, pairs = correspondence_payload(original)
    assert request == original
    assert pairs == {}


def test_compact_predicates_keep_reference_appearance_and_exclude_containment():
    original = frozen_case('C3')['request']
    request = factor_identity_instructions(original)
    entities = metadata(original['input'][0]['content'][0])['fixed_entities']
    for question in request['questions']:
        if not question['name'].startswith('match_'):
            continue
        identifier = question['instructions'].partition('permanent reference ID ')[2].split('.')[0]
        entity = entities[identifier]
        assert 'Reference subject: ' + entity['color'] + ' ' + entity['kind'] in question['instructions']
        assert 'excluding contents and neighboring objects' in question['instructions']


def frozen_case(name):
    return json.loads((Path(__file__).parent / 'fixtures/identity_factoring_c2_c3.json').read_text('utf-8'))[name]


@pytest.mark.parametrize('name,image_count,question_count', [('C2', 13, 22), ('C3', 12, 21)])
def test_frozen_requests_restore_every_original_byte_and_preserve_images_and_accepted_anchors(
        name, image_count, question_count):
    case = frozen_case(name)
    original = case['request']
    before = deepcopy(original)
    request = factor_identity_instructions(original)
    assert original == before
    assert request['input'][0]['content'][:-1] == original['input'][0]['content']
    assert sum(part['type'] == 'input_image' for part in original['input'][0]['content']) == image_count
    assert len(request['questions']) == question_count
    for current, prior in zip(request['questions'], original['questions'], strict=True):
        assert {**current, 'instructions': original_instructions(request, current)} == prior
    assert shared_instructions(request)['additional_candidate_measurements'] == {}
    assert len(json.dumps(request)) < len(json.dumps(original)) - 10000
    anchors = next(metadata(part)['current_confirmed_identity_anchors']
        for part in request['input'][0]['content'] if 'current_confirmed_identity_anchors' in metadata(part))
    assert anchors
    for anchor in anchors:
        number = anchor['candidate'].removeprefix('candidate-')
        assert not any(q['name'].startswith('match_' + number + '_') for q in request['questions'])
        assert anchor['request_hash'] == case['initial_request_hash']


@pytest.mark.parametrize('name', ['C2', 'C3'])
@pytest.mark.parametrize('answer_source', ['original_answers', 'actual_comparison_answers'])
def test_frozen_actual_answers_have_identical_parser_gates_before_and_after_factoring(name, answer_source):
    case = frozen_case(name)
    original = case['request']
    request = factor_identity_instructions(original)
    initial = {q['name']: q for q in case['initial_questions']}
    pairs = {}
    for question in original['questions']:
        if question['name'].startswith('match_'):
            _, number, index = question['name'].split('_')
            identity = 'identity_' + number
            pairs.setdefault(identity, {})[question['name']] = initial[identity]['choices'][int(index)]['value']
    scoped = {'questions': [q for q in case['initial_questions']
        if not q['name'].startswith('identity_') or q['name'] in pairs]}
    body = {'answers': case[answer_source]}
    expected = resolve_correspondences(body, original, scoped, pairs, .8)
    actual = resolve_correspondences(body, request, scoped, pairs, .8)
    assert actual == expected
    assert actual[1]['original_decisions_answers'] == case[answer_source]
    merged = {a['name']: a for a in case['initial_answers']}
    merged.update({a['name']: a for a in actual[0]['answers']})
    judgments, _ = parse_answers({'answers': [merged[q['name']] for q in case['initial_questions']]},
        case['initial_questions'], .8)
    accepted = {identifier for key, identifier in judgments.items()
                if key.startswith('identity_') and identifier not in ('reject', 'unknown')}
    required = {identifier for identifier, entity in metadata(original['input'][0]['content'][0])[
        'fixed_entities'].items() if entity['kind'] not in ('table', 'marker')}
    missing = required - accepted
    assert missing == ({'instance-95d42de5059b43f0'}
                       if name == 'C3' and answer_source == 'original_answers' else set())
    if answer_source == 'actual_comparison_answers':
        assert judgments['identity_7'] == 'unknown'  # Unresolved auxiliary view stays unresolved.
        if name == 'C3':
            assert judgments['scene_inventory_closure'] == 'unknown'  # Identity coverage is independent.


@pytest.mark.parametrize('conflict', ['context', 'candidate', 'camera', 'measurement', 'reference'])
def test_factoring_rejects_conflicting_reference_or_measurement_bindings(conflict):
    original = frozen_case('C3')['request']
    if conflict == 'context':
        context = metadata(original['input'][0]['content'][0])
        context['candidate_geometric_location_hints'][0]['camera'] = 'different-camera'
        original['input'][0]['content'][0]['text'] = json.dumps(context)
    else:
        text = original['questions'][1]['instructions']
        if conflict == 'candidate':
            text = text.replace('"candidate": "candidate-0"', '"candidate": "candidate-99"')
        elif conflict == 'camera':
            text = text.replace('"camera": "3"', '"camera": "1"')
        elif conflict == 'measurement':
            text = text.replace('"camera": "3"', '"camera":  "3"')
        else:
            original['questions'][4]['instructions'] = original['questions'][4]['instructions'].replace(
                'as reference ', 'as reference altered ')
        original['questions'][1]['instructions'] = text
    with pytest.raises(VisualInventoryError, match='visual_identity_instruction_'):
        factor_identity_instructions(original)


def test_multiline_reference_description_is_preserved_verbatim():
    original = frozen_case('C3')['request']
    original['questions'] = original['questions'][:1]
    original['questions'][0]['instructions'] = original['questions'][0]['instructions'].replace(
        'as reference ', 'as reference original description\nsecond line ')
    request = factor_identity_instructions(original)
    assert original_instructions(request, request['questions'][0]) == original['questions'][0]['instructions']
