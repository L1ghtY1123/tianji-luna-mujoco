"""C4: current pairwise failures must supersede stale categorical diagnostics."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import httpx
import numpy as np
import pytest
from test_visual_inventory import fixture

from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError


def c4_replay(*, invalid_clarification=False, use_mask_comparison=False):
    saved = json.loads((Path(__file__).parent / 'fixtures/unknown_c4_identity_answers.json').read_text('utf-8'))
    capture, template = fixture()
    capture.update(observation_id=saved['observation_id'], generation=saved['generation'])
    capture['views'] = {camera: deepcopy(capture['views']['3']) for camera in ('3', '2', '1')}
    detections = []
    for number, camera in enumerate(('3', '3', '3', '3', '2', '2', '2', '1')):
        mask = np.zeros((24, 32), bool)
        mask[number * 2:number * 2 + 2, 4:6] = True
        detections.append({**deepcopy(template), 'camera_id': camera,
            'view': capture['views'][camera], 'mask': mask, 'silhouette_mask': mask,
            'points': np.array([[.6, .1, .8]] * int(mask.sum())), 'pixels': capture['views'][camera]['rgb'][mask],
            'pixels_rc': np.column_stack(np.nonzero(mask))})
    reference = {'entities': saved['entities'], 'views': {}, 'task_spec': saved['task_spec']}
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 2 and invalid_clarification:
            return httpx.Response(200, json={'answers': []})
        answers = saved['categorical_answers'] if len(requests) == 1 else saved[
            'actual_mask_comparison_answers' if use_mask_comparison else 'pairwise_answers']
        by_name = {row['name']: row for row in answers}
        return httpx.Response(200, json={'answers': [by_name[q['name']] for q in payload['questions']]})

    client = DemoVisualInventory(api_key='offline-c4', client=httpx.Client(transport=httpx.MockTransport(respond)))

    def observe():
        return client.resolve(capture, detections, entities=saved['entities'], reference=reference,
            instruction=saved['task_spec']['instruction'], purpose='associate_inventory')

    return saved, requests, client, observe


def request_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def test_latest_failure_reports_left_container_and_preserves_first_failure_as_history():
    saved, requests, client, observe = c4_replay()
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown') as caught:
        observe()
    current = caught.value.evidence
    assert current['candidate_id'] == 'candidate-3'
    assert current['request_hash'] == request_hash(requests[1])
    assert current['answers'][0]['choice'] == 'instance-9a42b0ff1d544c81'
    assert current['answers'][0]['confidence'] == .97
    assert current['original_decisions_answers'] == saved['pairwise_answers']
    history = current['initial_identity_failure']
    assert history['error'] == 'visual_identity_unknown'
    assert history['evidence']['candidate_id'] == 'candidate-0'
    assert history['evidence']['confidence'] == .41
    assert history['evidence']['request_hash'] == request_hash(requests[0])
    assert 'identity_clarification' not in history['evidence']
    assert 'initial_identity_failure' not in history['evidence']
    assert client.cache[request_hash(requests[0])].evidence == history['evidence']
    assert len(requests) == 2
    images = lambda payload: [p for m in payload['input'] for p in m['content'] if p['type'] == 'input_image']
    assert images(requests[0]) == images(requests[1])[:len(images(requests[0]))]
    assert len(images(requests[1])) == len(images(requests[0])) + 8
    frozen = deepcopy(current)
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown') as repeated:
        observe()
    assert len(requests) == 2  # Same frozen capture reuses both cached verdicts.
    assert repeated.value.evidence == frozen


def test_failure_has_complete_identity_coverage_and_the_actual_conflicting_camera_answers():
    saved, requests, _, observe = c4_replay()
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown') as caught:
        observe()
    evidence = caught.value.evidence
    coverage = evidence['identity_coverage']
    assert coverage['required_entity_ids'] == sorted(
        key for key, row in saved['entities'].items() if row['kind'] != 'table')
    assert coverage['accepted_entity_ids'] == sorted([
        'instance-9a42b0ff1d544c81', 'instance-2253e017f3e04325', 'instance-dcda1cececac449f'])
    assert coverage['complete'] is False
    assert evidence['missing_geometry_ids'] == ['instance-1a561637d9f34fc0']
    assert coverage['request_hash'] == request_hash(requests[1])
    rows = {row['candidate_id']: row for row in evidence['unresolved_identity_candidates']}
    assert set(rows) == {'candidate-3', 'candidate-6', 'candidate-7'}
    conflict = rows['candidate-6']
    assert conflict['status'] == 'conflicting'
    assert conflict['camera_id'] == '2'
    assert conflict['observation_id'] == saved['observation_id'] and conflict['generation'] == saved['generation']
    assert conflict['request_hash'] == request_hash(requests[1])
    assert conflict['positive_entity_ids'] == ['instance-1a561637d9f34fc0', 'instance-2253e017f3e04325']
    assert conflict['answer']['choice'] == 'unknown'
    assert conflict['original_decisions_answers'] == [a for a in saved['pairwise_answers']
                                                     if a['name'].startswith('match_6_')]
    assert rows['candidate-3']['status'] == rows['candidate-7']['status'] == 'unknown'


def test_clarification_protocol_failure_is_not_mislabeled_as_an_old_visual_verdict():
    _, requests, _, observe = c4_replay(invalid_clarification=True)
    with pytest.raises(VisualInventoryError, match='invalid_response') as caught:
        observe()
    assert len(requests) == 2
    assert caught.value.evidence['request_hash'] == request_hash(requests[1])
    assert caught.value.evidence['initial_identity_failure']['error'] == 'visual_identity_unknown'
    assert 'identity_coverage' not in caught.value.evidence  # No valid new verdict was returned.


def test_actual_c4_mask_comparison_covers_registered_entities_without_accepting_ambiguous_views():
    saved, requests, _, observe = c4_replay(use_mask_comparison=True)
    result = observe()
    assert len(requests) == 2
    assert result['missing_geometry_ids'] == []
    assert result['identity_coverage']['complete'] is True
    assert {row['candidate'] for row in result['unassociated_candidates']} == {'candidate-0', 'candidate-3', 'candidate-7'}
    evidence = result['evidence']
    assert evidence['original_decisions_answers'] == saved['actual_mask_comparison_answers']
    accepted = {entry['entity_id'] for group in result['groups'] for entry in group}
    assert accepted == {key for key, row in saved['entities'].items() if row['kind'] != 'table'}
    proof = [row for group in result['groups'] for entry in group for row in entry['identity_evidence']]
    assert {row['candidate'] for row in proof} == {'candidate-1', 'candidate-2', 'candidate-4', 'candidate-5', 'candidate-6'}
    assert all(row['request_hash'] == request_hash(requests[1]) for row in proof)
