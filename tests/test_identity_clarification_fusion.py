"""Replay real C1 answer values; clarification must not re-ask accepted identities."""
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


@pytest.mark.parametrize('conflicting_unresolved', [False, True])
def test_archived_c1_keeps_accepted_left_wrist_identity_and_only_clarifies_unresolved(conflicting_unresolved):
    archived = json.loads((Path(__file__).parent / 'fixtures/unknown_c1_identity_answers.json').read_text('utf-8'))
    capture, detection = fixture()
    capture.update(observation_id=archived['observation_id'], generation=archived['generation'])
    capture['views'] = {camera: deepcopy(capture['views']['3']) for camera in ('1', '2', '3')}
    detections = []
    for index, camera in enumerate(('3', '3', '3', '3', '2', '2', '2', '1')):
        item = deepcopy(detection)
        mask = np.zeros((24, 32), bool)
        mask[2 * index:2 * index + 2, 4:6] = True
        item.update(camera_id=camera, mask=mask, silhouette_mask=mask)
        detections.append(item)
    reference = {'entities': archived['entities'], 'views': {}, 'task_spec': archived['task_spec']}
    requests = []

    def respond(request):
        sent = json.loads(request.content)
        requests.append(sent)
        raw = archived['categorical_answers'] if len(requests) == 1 else archived['pairwise_answers']
        by_name = {row['name']: deepcopy(row) for row in raw}
        if conflicting_unresolved and len(requests) == 2:
            by_name['match_0_0']['probability'] = .99
        # Only consume archived answers for questions actually asked by production.
        return httpx.Response(200, json={'answers': [by_name[q['name']] for q in sent['questions']]})

    client = DemoVisualInventory(api_key='offline-archive', client=httpx.Client(transport=httpx.MockTransport(respond)))
    def call():
        return client.resolve(capture, detections, reference=reference, entities=archived['entities'],
                              instruction=archived['task_spec']['instruction'], purpose='associate_inventory')
    if conflicting_unresolved:
        with pytest.raises(VisualInventoryError, match='visual_identity_unknown') as caught:
            call()
        details = caught.value.evidence
        assert details['candidate'] == 'candidate-0'
        assert details['choice'] == 'unknown'
        assert details['initial_identity_failure']['error'] == 'visual_identity_unknown'
    else:
        result = call()
        assert result['missing_geometry_ids'] == []
        assert result['identity_coverage']['complete'] is True
        assert result['evidence']['answers'][4] == archived['categorical_answers'][4]
        assert result['evidence']['answers'][5] == archived['categorical_answers'][5]
        assert result['evidence']['answers'][0]['choice'] == 'instance-5c1bc1928dff452f'
        assert result['evidence']['answers'][7]['choice'] == 'unknown'
        retained = result['evidence']['retained_identity_answers']
        assert [row['answer']['name'] for row in retained] == ['identity_4', 'identity_5']
        assert all(row['request_hash'] == result['evidence']['clarifies_request_hash'] for row in retained)
    assert len(requests) == 2
    names = [q['name'] for q in requests[1]['questions']]
    assert len(names) == 29
    assert not any(name.startswith(('match_4_', 'match_5_')) for name in names)
    assert {f'match_7_{i}' for i in range(4)} <= set(names)


@pytest.mark.parametrize('changed', ['observation_id', 'request_hash'])
def test_accepted_identity_is_not_reused_without_original_capture_and_request_provenance(changed):
    payload = {'input': [{'content': [{'type': 'input_text', 'text': '{"observation_id":"current"}'}]}],
               'questions': []}
    evidence = {'observation_id': 'current', 'request_hash': 'original-hash', 'answers': []}
    evidence[changed] = None
    with pytest.raises(VisualInventoryError, match='clarification_provenance_missing'):
        scope_identity_clarification(payload, evidence, .8)
