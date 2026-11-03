"""A confirmed released object does not make its uncertain destination optional."""
import json

import httpx
import pytest
from test_demo_partial_identity import inputs

from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory


@pytest.mark.parametrize('destination_match,missing', [(.99, []), (.6, ['bin'])])
def test_transfer_clarifies_missing_destination_once_and_preserves_target(destination_match, missing):
    capture, detections, entities, reference = inputs()
    entities['bin'] = {'id': 'bin', 'kind': 'bin', 'color': 'blue', 'roles': ['collision']}
    reference['task_spec']['destination_ref'] = 'bin'
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        answers = []
        for question in payload['questions']:
            name = question['name']
            if question['type'] == 'predicate':
                answers.append({'name': name, 'type': 'predicate', 'probability':
                    destination_match if question['instructions'].partition(' Reference subject: ')[0].endswith('permanent reference ID bin.') else .01})
            else:
                value = ('pink' if name == 'identity_0' else 'bin' if name == 'identity_1'
                    else 'at_destination' if name.startswith('progress_')
                    else 'none' if name == 'scene_inventory_closure' else 'visible')
                answers.append({'name': name, 'type': 'choice', 'choice': value,
                    'confidence': .48 if name == 'identity_1' else .99})
        return httpx.Response(200, json={'answers': answers})

    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = client.resolve(capture, detections, instruction='place pink inside bin', entities=entities,
        reference=reference, purpose='verify_transfer',
        progress={'pink': {'release_feedback_id': 'release', 'status': 'released'}})
    assert len(requests) == 2
    assert result['missing_geometry_ids'] == missing
    images = lambda request: [part for message in request['input'] for part in message['content']
                              if part['type'] == 'input_image']
    assert images(requests[0]) == images(requests[1])[:len(images(requests[0]))]
    assert len(images(requests[1])) == len(images(requests[0])) + 1
    assert not any(question['name'].startswith('match_0_') for question in requests[1]['questions'])
    retained = result['evidence']['retained_identity_answers']
    assert retained[0]['answer']['choice'] == 'pink'
    assert retained[0]['request_hash'] == result['evidence']['clarifies_request_hash']
