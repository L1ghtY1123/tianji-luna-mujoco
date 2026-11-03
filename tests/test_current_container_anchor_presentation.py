"""Current measured container anchors remain visible beside unresolved proposals."""
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from test_demo_container_tracking import case, module
from test_identity_instruction_factoring import original_instructions

from luna_mujoco.demo_identity_presentation import improve_identity_presentation
from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory
from luna_mujoco.visual_inventory import mask_box


def two_container_capture():
    observation, initial, current, detections, entities = case()
    entities['left'] = {'id': 'left', 'kind': 'bin', 'color': 'blue', 'roles': ['collision']}
    observation['visual_reference']['entities'] = deepcopy(entities)
    observation['scene_entities'] = entities
    reference = module().reference_with_container_tracks(observation, initial)
    detections[0]['points'][:, 1] += .5
    other = deepcopy(detections[0])
    other.update(camera_id='2')
    other['points'][:, 1] -= 1.1
    pink = deepcopy(detections[0])
    pink.update(camera_id='3', kind='object', prompt='pink object')
    for camera, view in current['views'].items():
        view['rgb'][:] = int(camera) * 30
    return current, detections + [other, pink], entities, reference


def annotation_parts(payload):
    content = payload['input'][0]['content']
    result = {}
    for index, part in enumerate(content):
        if part['type'] != 'input_text':
            continue
        try:
            metadata = json.loads(part['text'])
        except ValueError:
            continue
        if metadata.get('image_role') == 'current_annotations':
            result[metadata['camera']] = (metadata, content[index + 1])
    return result


@pytest.mark.parametrize('pairwise', [False, True])
@pytest.mark.parametrize('purpose', ['associate_inventory', 'verify_transfer'])
def test_confirmed_anchor_is_labeled_in_current_images_even_without_identity_question(monkeypatch, pairwise, purpose):
    capture, detections, entities, reference = two_container_capture()
    original_points = [row['points'].copy() for row in detections]
    calls = []
    # Exercise both existing presentation routes; the numeric tracker is real.
    monkeypatch.setattr('luna_mujoco.identity_correspondence.has_product_reference', lambda _: pairwise)

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for question in payload['questions']:
            name = question['name']
            if question['type'] == 'predicate':
                answers.append({'name': name, 'type': 'predicate',
                    'probability': .99 if name in {'match_1_2', 'match_2_1'} else .01})
            else:
                value = {'identity_1': 'left', 'identity_2': 'pink', 'scene_inventory_closure': 'none'}.get(name, 'visible')
                answers.append({'name': name, 'type': 'choice', 'choice': value, 'confidence': .99})
        return httpx.Response(200, json={'answers': answers})

    visual = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = visual.resolve(capture, detections, entities=entities, reference=reference,
        instruction='move pink', purpose=purpose)
    assert len(calls) == 1
    sent = calls[0]
    context = json.loads(sent['input'][0]['content'][0]['text'])
    assert context['container_correspondence']['candidate_mapping'] == {'0': 'right'}
    anchors = context['current_confirmed_container_anchors']
    assert len(anchors) == 1
    anchor = anchors[0]
    assert (anchor['entity_id'], anchor['candidate'], anchor['camera']) == ('right', 'candidate-0', '1')
    assert anchor['box'] == mask_box(detections[0]['silhouette_mask'])
    assert anchor['observation_id'] == capture['observation_id'] and anchor['generation'] == capture['generation']
    np.testing.assert_allclose(anchor['measured_bounds_world_m'],
        [original_points[0].min(axis=0), original_points[0].max(axis=0)])
    assert anchor['source'] == 'measured_container_correspondence'
    assert anchor['binding_hash'] == context['container_correspondence']['binding_hash']
    assert anchor['correspondence']['entity_id'] == 'right'
    annotations = annotation_parts(sent)
    assert set(annotations) == {'1', '2', '3'}
    current_anchor, anchor_image = annotations['1']
    assert anchor_image['type'] == 'input_image'  # Compaction must not discard this locating view.
    item = current_anchor['items'][0]
    assert item['id'] == 'right' and item['candidate'] == 'candidate-0'
    assert item['annotation_role'] == 'current_confirmed_container_anchor'
    assert item['display_label'] == 'Current anchor · Reference 3'
    assert annotations['2'][0]['items'][0]['display_label'] == 'Candidate 1'
    names = {question['name'] for question in sent['questions']}
    assert 'identity_0' not in names and not any(name.startswith('match_0_') for name in names)
    assert 'visibility_right' in names
    if pairwise:
        question = next(question for question in sent['questions'] if question['name'] == 'match_1_0')
    else:
        question = next(question for question in sent['questions'] if question['name'] == 'identity_1')
        assert 'right' in {option['value'] for option in question['choices']}
    instruction = original_instructions(sent, question)
    assert 'CURRENT confirmed' in instruction
    assert 'same capture' in instruction
    assert 'between captures' in instruction
    assert 'simultaneously distinct' in context['rules']
    assert [(group[0]['entity_id'], {row['camera_id'] for row in group}) for group in result['groups']] == [
        ('left', {'2'}), ('pink', {'3'}), ('right', {'1'})]
    for row, points in zip(detections, original_points, strict=True):
        np.testing.assert_array_equal(row['points'], points)


def test_current_anchor_presentation_is_stable_without_adding_detail_images():
    from test_demo_identity_presentation import current_payload, metadata

    payload = current_payload()
    for part in payload['input'][0]['content']:
        if part['type'] != 'input_text':
            continue
        try:
            row = json.loads(part['text'])
        except ValueError:
            continue
        if row.get('image_role') == 'current_annotations':
            row['items'].append({'id': 'permanent-pink-1', 'candidate': 'candidate-9',
                'annotation_role': 'current_confirmed_container_anchor', 'box': [.7, .2, .75, .25]})
            part['text'] = json.dumps(row)
    before = deepcopy(payload)
    presented = improve_identity_presentation(payload)
    assert payload == before
    anchors = [item for row in metadata(presented, 'current_annotations') for item in row['items']
               if item.get('annotation_role') == 'current_confirmed_container_anchor']
    assert anchors[0]['display_label'].startswith('Current anchor · Reference ')
    assert len(metadata(presented, 'current_candidate_detail_zoom')) == 2
    assert all('candidate-9' not in [item['id'] for item in row['original_candidates']]
               for row in metadata(presented, 'current_candidate_detail_zoom'))
    assert improve_identity_presentation(presented) == presented
