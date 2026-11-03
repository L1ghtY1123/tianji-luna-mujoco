import hashlib
import json
from copy import deepcopy

import httpx
import pytest
from test_demo_partial_identity import inputs

from luna_mujoco.demo_perception import DemoVisualInventory


@pytest.mark.parametrize('case', ['confirmed', 'unknown', 'malformed'])
@pytest.mark.parametrize('historical_reference', [False, True])
def test_product_identity_compares_all_views_and_retains_raw_predicates(case, historical_reference):
    capture, detections, entities, reference = inputs(('3', '2', '1'))
    capture['time'] = 2.
    creation = {'output_entity_id': 'pink', 'owner_id': 'run', 'request_hash': 'verified',
                'generation': 1, 'observation_id': capture['observation_id']}
    entities['pink']['creation_evidence'] = creation
    reference.update(owner_id='run', product_references={'pink': {
        'entity': deepcopy(entities['pink']), 'evidence': deepcopy(creation), 'generation': 1,
        'observation_id': capture['observation_id'], 'views': deepcopy(capture['views']),
        'verification_capture': {'generation': 1, 'observation_id': capture['observation_id'], 'time': 2.,
            'images': {camera: {'rgb_sha256': hashlib.sha256(view['rgb'].tobytes()).hexdigest(),
                               'shape': list(view['rgb'].shape)} for camera, view in capture['views'].items()}}}})
    if historical_reference:
        capture['observation_id'] = 'after-moving-product'
        capture['time'] = 3.
    calls = []
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for question in payload['questions']:
            name = question['name']
            if question['type'] == 'predicate':
                probability = .99 if name == 'match_0_0' and case == 'confirmed' else .6
                answers.append({'name': name, 'type': 'predicate',
                                'probability': None if case == 'malformed' else probability})
            else:
                value = 'none' if name == 'scene_inventory_closure' else 'visible'
                answers.append({'name': name, 'type': 'choice', 'choice': value, 'confidence': .99})
        return httpx.Response(200, json={'answers': answers})
    visual = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    if case != 'confirmed':
        with pytest.raises(ValueError, match='invalid_response' if case == 'malformed' else 'identity_unknown'):
            visual.resolve(capture, detections, instruction='move', entities=entities,
                           reference=reference, purpose='associate_inventory')
        return
    result = visual.resolve(capture, detections, instruction='move', entities=entities,
                            reference=reference, purpose='associate_inventory')
    assert len(calls) == 1
    labels = [part['text'] for part in calls[0]['input'][0]['content'] if part['type'] == 'input_text'
              and part['text'].startswith('CURRENT RGB camera ')]
    assert len(labels) == 3
    assert len([q for q in calls[0]['questions'] if q['type'] == 'predicate']) == 3
    assert [entry['camera_id'] for entry in result['groups'][0]] == ['3']
    assert len(result['evidence']['identity_correspondence']) == 3
    assert result['evidence']['original_decisions_answers'][0]['probability'] == .99
    assert len(result['unassociated_candidates']) == 2
