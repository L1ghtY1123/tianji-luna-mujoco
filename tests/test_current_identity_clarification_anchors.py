"""C2 wire regression: accepted current subjects stay visible during clarification."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import httpx
import numpy as np
import pytest
from test_visual_inventory import fixture

from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory
from luna_mujoco.identity_correspondence import scope_identity_clarification
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import VisualInventory


def c2_case():
    saved = json.loads((Path(__file__).parent / 'fixtures/unknown_c2_identity_answers.json').read_text('utf-8'))
    capture, template = fixture()
    capture.update(observation_id=saved['observation_id'], generation=saved['generation'])
    capture['views'] = {camera: deepcopy(capture['views']['3']) for camera in ('3', '2', '1')}
    detections = []
    for number, camera in enumerate(('3', '3', '3', '3', '2', '2', '2', '1')):
        view = capture['views'][camera]
        mask = np.zeros((24, 32), bool)
        mask[number * 2:number * 2 + 2, 4:6] = True
        detections.append({**deepcopy(template), 'camera_id': camera, 'view': view,
            'mask': mask, 'silhouette_mask': mask, 'pixels_rc': np.column_stack(np.nonzero(mask)),
            'pixels': view['rgb'][mask], 'points': np.tile([.4, .5 - number * .1, .6], (4, 1))})
    reference = {'entities': saved['entities'], 'views': {}, 'task_spec': saved['task_spec']}
    return saved, capture, detections, reference


def replay_c2(*, use_actual_anchor_comparison=False):
    saved, capture, detections, reference = c2_case()
    requests = []

    def respond(request):
        sent = json.loads(request.content)
        requests.append(sent)
        raw = saved['categorical_answers'] if len(requests) == 1 else saved[
            'actual_anchor_comparison_answers' if use_actual_anchor_comparison else 'pairwise_answers']
        answers = {row['name']: row for row in raw}
        return httpx.Response(200, json={'answers': [answers[q['name']] for q in sent['questions']]})

    visual = DemoVisualInventory(api_key='offline-replay', client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = visual.resolve(capture, detections, entities=saved['entities'], reference=reference,
        instruction=saved['task_spec']['instruction'], purpose='associate_inventory')
    return saved, capture, detections, requests, result


def anchor_metadata(request):
    for part in request['input'][0]['content']:
        if part['type'] != 'input_text':
            continue
        try:
            value = json.loads(part['text'])
        except ValueError:
            continue
        if isinstance(value, dict) and 'current_confirmed_identity_anchors' in value:
            return value
    return {}


def test_c2_clarification_locates_already_accepted_current_subjects_without_changing_images():
    saved, capture, detections, requests, result = replay_c2()
    assert len(requests) == 2
    metadata = anchor_metadata(requests[1])
    anchors = metadata.get('current_confirmed_identity_anchors', [])
    assert [row['candidate'] for row in anchors] == ['candidate-4', 'candidate-5']
    for number, row in zip((4, 5), anchors, strict=True):
        assert row['entity_id'] == saved['categorical_answers'][number]['choice']
        assert row['answer'] == saved['categorical_answers'][number]
        assert row['request_hash'] == result['evidence']['clarifies_request_hash']
        assert row['observation_id'] == capture['observation_id'] and row['generation'] == capture['generation']
        assert row['camera'] == '2'
        assert row['candidate_provenance']['silhouette_mask']['sha256'] == hashlib.sha256(
            detections[number]['silhouette_mask'].tobytes()).hexdigest()
        assert row['candidate_provenance']['view']['rgb']['sha256'] == hashlib.sha256(
            capture['views']['2']['rgb'].tobytes()).hexdigest()
    images = lambda request: [part for message in request['input'] for part in message['content']
                              if part['type'] == 'input_image']
    assert images(requests[0]) == images(requests[1])[:len(images(requests[0]))]
    assert len(images(requests[1])) == len(images(requests[0])) + 6
    assert not any(q['name'].startswith(('match_4_', 'match_5_')) for q in requests[1]['questions'])
    assert 'simultaneously distinct' in metadata['rule']
    assert any(q['name'] == 'match_7_0' for q in requests[1]['questions'])


def test_c2_group_evidence_uses_original_candidate_numbers_and_actual_accepting_requests():
    saved, capture, _, _, result = replay_c2()
    records = [row for group in result['groups'] for entry in group for row in entry.get('identity_evidence', [])]
    by_candidate = {row['candidate']: row for row in records}
    assert set(by_candidate) == {'candidate-4', 'candidate-5', 'candidate-6', 'candidate-7'}
    for number in (4, 5):
        row = by_candidate[f'candidate-{number}']
        assert row['request_hash'] == result['evidence']['clarifies_request_hash']
        assert row['original_decisions_answers'] == [saved['categorical_answers'][number]]
    row = by_candidate['candidate-7']
    assert row['request_hash'] == result['evidence']['request_hash']
    assert row['original_decisions_answers'] == [answer for answer in saved['pairwise_answers']
                                               if answer['name'].startswith('match_7_')]
    assert all(row['source'] == 'openai_decisions' and row['observation_id'] == capture['observation_id']
               and row['generation'] == capture['generation'] for row in records)


def test_actual_c2_anchor_comparison_covers_registered_entities_without_empty_right_box():
    saved, _, _, _, result = replay_c2(use_actual_anchor_comparison=True)
    assert result['missing_geometry_ids'] == []
    assert result['identity_coverage']['complete'] is True
    records = [row for group in result['groups'] for entry in group for row in entry['identity_evidence']]
    assert {row['candidate'] for row in records} == {'candidate-4', 'candidate-5', 'candidate-6'}
    assert {row['entity_id'] for row in records} == {
        identifier for identifier, entity in saved['entities'].items() if entity['kind'] != 'table'}
    assert {row['candidate'] for row in result['unassociated_candidates']} == {
        'candidate-0', 'candidate-1', 'candidate-2', 'candidate-3', 'candidate-7'}
    assert result['evidence']['answers'][7]['choice'] == 'unknown'
    assert result['evidence']['answers'][4] == saved['categorical_answers'][4]
    assert result['evidence']['answers'][5] == saved['categorical_answers'][5]


def test_positive_same_capture_predicate_anchor_keeps_its_original_typed_answer():
    saved, _, _, requests, result = replay_c2()
    payload = deepcopy(requests[0])
    question = next(row for row in payload['questions'] if row['name'] == 'identity_4')
    question.clear()
    question.update(type='predicate', name='identity_4', instructions='Does C4 show the same referenced subject?')
    raw = deepcopy(saved['categorical_answers'])
    raw[4] = {'type': 'predicate', 'name': 'identity_4', 'probability': .86}
    evidence = {'observation_id': result['evidence']['observation_id'],
        'generation': result['evidence']['generation'], 'request_hash': result['evidence']['clarifies_request_hash'],
        'answers': saved['categorical_answers'], 'original_decisions_answers': raw,
        'same_capture_identity_proposals': {'identity_4': saved['categorical_answers'][4]['choice']}}
    scoped, retained = scope_identity_clarification(payload, evidence, .8)
    anchors = anchor_metadata(scoped)['current_confirmed_identity_anchors']
    assert anchors[0]['entity_id'] == saved['categorical_answers'][4]['choice']
    assert anchors[0]['answer'] == retained['identity_4']['answer'] == raw[4]
    assert anchors[0]['question'] == question


def test_same_camera_alias_merge_retains_each_independent_identity_answer():
    from test_visual_inventory import client_with_answers, fixed_entities

    capture, detection = fixture()
    requests = []
    visual = VisualInventory(api_key='offline', client=client_with_answers(
        {'identity_0': 'red', 'identity_1': 'red'}, requests))
    result = visual.resolve(capture, [detection, deepcopy(detection)], instruction='observe', entities=fixed_entities())
    assert len(result['groups']) == len(result['groups'][0]) == 1
    records = result['groups'][0][0]['identity_evidence']
    assert [row['candidate'] for row in records] == ['candidate-0', 'candidate-1']
    assert [row['original_decisions_answers'][0]['name'] for row in records] == ['identity_0', 'identity_1']
    assert all(row['request_hash'] == result['evidence']['request_hash'] for row in records)


@pytest.mark.parametrize('damage', ['generation', 'candidate_observation', 'camera'])
def test_clarification_cannot_publish_stale_or_misbound_accepted_anchor(damage):
    _, _, _, requests, result = replay_c2()
    payload = deepcopy(requests[0])
    evidence = {'observation_id': result['evidence']['observation_id'],
        'generation': result['evidence']['generation'],
        'request_hash': result['evidence']['clarifies_request_hash'],
        'answers': json.loads((Path(__file__).parent / 'fixtures/unknown_c2_identity_answers.json').read_text('utf-8'))[
            'categorical_answers']}
    if damage == 'generation':
        evidence['generation'] -= 1
    else:
        for part in payload['input'][0]['content']:
            if part['type'] != 'input_text':
                continue
            try:
                row = json.loads(part['text'])
            except ValueError:
                continue
            if isinstance(row, dict) and row.get('candidate') == 'candidate-4':
                row.setdefault('candidate_provenance', {})[
                    'observation_id' if damage == 'candidate_observation' else 'camera'] = 'wrong'
                part['text'] = json.dumps(row)
    with pytest.raises(VisualInventoryError, match='clarification_.*(provenance|binding)'):
        scope_identity_clarification(payload, evidence, .8)
