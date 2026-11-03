"""Current SAM ownership is extra image evidence, never an identity decision."""
import base64
import json
from copy import deepcopy
from io import BytesIO

import httpx
import numpy as np
import pytest
from PIL import Image
from test_visual_inventory import fixture

from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import array_stamp


def subject_case():
    capture, detection = fixture()
    rgb = capture['views']['3']['rgb']
    rgb[6:10, 7:11] = [0, 255, 0]  # A separate nested subject inside the box outline.
    mask = np.zeros(rgb.shape[:2], bool)
    mask[4:16, 3:14] = True
    mask[6:10, 7:11] = False
    provenance = {'candidate': 'candidate-6', 'camera': '3',
        'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'stage': 'segmentation_before_inventory_surface_assignment',
        'mask': array_stamp(detection['mask']), 'silhouette_mask': array_stamp(mask),
        'view': {key: array_stamp(value) for key, value in capture['views']['3'].items()}, 'calibration': {}}
    content = [{'type': 'input_text', 'text': json.dumps({
        'observation_id': capture['observation_id'], 'generation': capture['generation']})},
        {'type': 'input_image', 'image_url': 'original-image-untouched'},
        {'type': 'input_text', 'text': json.dumps({'candidate': 'candidate-6', 'camera': '3',
            'candidate_provenance': provenance})}]
    request = {'input': [{'role': 'user', 'content': content}],
        'questions': [{'name': 'match_6_0', 'type': 'predicate', 'instructions': 'Original identity proposition.'}]}
    subjects = {'candidate-6': {'silhouette_mask': mask.copy(), 'candidate_provenance': deepcopy(provenance)}}
    return capture, request, subjects


def append_views(request, capture, subjects):
    from luna_mujoco.identity_subject_views import append_identity_subject_views

    return append_identity_subject_views(request, capture, subjects)


def test_subject_pixels_exact_and_nested_subject_excluded_without_altering_original_input():
    capture, request, subjects = subject_case()
    original = deepcopy(request)
    result = append_views(request, capture, subjects)
    content = result['input'][0]['content']
    assert request == original
    assert content[:len(original['input'][0]['content'])] == original['input'][0]['content']
    assert result['questions'] == original['questions']
    metadata = json.loads(content[-2]['text'])
    assert metadata['candidate'] == 'candidate-6' and metadata['camera'] == '3'
    assert metadata['source_silhouette_mask_sha256'] == subjects['candidate-6']['candidate_provenance']['silhouette_mask']['sha256']
    assert metadata['original_image_offset_xy_px'] == [0, 32]
    assert content[-1]['image_url'].startswith('data:image/png;base64,')
    pixels = np.asarray(Image.open(BytesIO(base64.b64decode(content[-1]['image_url'].split(',')[1]))))
    rgb, mask = capture['views']['3']['rgb'], subjects['candidate-6']['silhouette_mask']
    assert pixels.shape == (rgb.shape[0] + 32, rgb.shape[1], 3)
    assert np.array_equal(pixels[32:][mask], rgb[mask])
    assert np.all(pixels[32:][~mask] == 128)
    assert np.all(pixels[38:42, 7:11] == 128)  # Green is not part of this container subject.


@pytest.mark.parametrize('changed', ['mask', 'rgb', 'depth', 'observation', 'generation', 'camera', 'candidate', 'stage', 'mask_dtype'])
def test_tampered_or_stale_subject_source_is_rejected(changed):
    capture, request, subjects = subject_case()
    if changed == 'mask':
        subjects['candidate-6']['silhouette_mask'][0, 0] = True
    elif changed == 'rgb':
        capture['views']['3']['rgb'][0, 0] = 1
    elif changed == 'depth':
        capture['views']['3']['depth_m'][0, 0] += .01
    elif changed in ('observation', 'generation'):
        capture['observation_id' if changed == 'observation' else changed] = 'other' if changed == 'observation' else 2
    elif changed == 'mask_dtype':
        subjects['candidate-6']['silhouette_mask'] = subjects['candidate-6']['silhouette_mask'].astype(np.uint8)
    else:
        subjects['candidate-6']['candidate_provenance'][changed] = 'other'
    with pytest.raises(VisualInventoryError, match='visual_identity_subject_binding_invalid'):
        append_views(request, capture, subjects)


def test_missing_subject_and_image_budget_fail_instead_of_silently_omitting_evidence():
    capture, request, subjects = subject_case()
    with pytest.raises(VisualInventoryError, match='visual_identity_subject_binding_invalid'):
        append_views(request, capture, {})
    request['input'][0]['content'].extend([{'type': 'input_image', 'image_url': str(i)} for i in range(127)])
    with pytest.raises(VisualInventoryError, match='visual_inventory_input_limit'):
        append_views(request, capture, subjects)


@pytest.mark.parametrize('purpose', ['associate_inventory', 'verify_transfer'])
def test_only_unresolved_candidates_get_subject_images_with_no_extra_call(purpose):
    capture, detection = fixture()
    other = deepcopy(detection)
    other['mask'] = np.zeros_like(other['mask'])  # Geometry mask is not the image subject.
    other['mask'][4, 3] = True
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for q in payload['questions']:
            if q['type'] == 'predicate':
                answer = {'probability': .99 if q['instructions'].partition(' Reference subject: ')[0].endswith('permanent reference ID blue.') else .01}
            else:
                answer = {'choice': {'identity_0': 'red', 'identity_1': 'unknown',
                    'scene_inventory_closure': 'none'}.get(q['name'], 'visible'), 'confidence': .99}
            answers.append({'name': q['name'], 'type': q['type'], **answer})
        return httpx.Response(200, json={'answers': answers})

    inventory = DemoVisualInventory(api_key='offline', client=httpx.Client(transport=httpx.MockTransport(respond)))
    # Second registered entity keeps missing geometry explicit and triggers bounded clarification.
    entities = {'red': {'kind': 'cube'}, 'blue': {'kind': 'bin'}}
    result = inventory.resolve(capture, [detection, other], instruction='identify subjects',
        entities=entities, purpose=purpose, reference={'inventory_state': {'inventory_finalized': True}})
    assert len(calls) == 2
    images = lambda p: [part for message in p['input'] for part in message['content'] if part['type'] == 'input_image']
    assert images(calls[1])[:len(images(calls[0]))] == images(calls[0])
    assert len(images(calls[1])) == len(images(calls[0])) + 1
    descriptions = [json.loads(p['text']) for p in calls[1]['input'][0]['content']
        if p['type'] == 'input_text' and '"image_role": "current_candidate_sam_subject"' in p['text']]
    assert [row['candidate'] for row in descriptions] == ['candidate-1']
    assert not any(q['name'].startswith('match_0_') for q in calls[1]['questions'])
    pixels = np.asarray(Image.open(BytesIO(base64.b64decode(images(calls[1])[-1]['image_url'].split(',')[1]))))
    assert np.array_equal(pixels[32:][other['silhouette_mask']], capture['views']['3']['rgb'][other['silhouette_mask']])
    assert result['evidence']['retained_identity_answers'][0]['answer']['choice'] == 'red'


@pytest.mark.parametrize('change', ['rgb', 'observation_id', 'generation'])
def test_capture_changed_during_initial_call_cannot_reach_clarification_transport(change):
    capture, detection = fixture()
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if change == 'rgb':
            capture['views']['3']['rgb'][0, 0] = 255
        elif change == 'observation_id':
            capture['observation_id'] = 'a-later-capture'
        else:
            capture['generation'] += 1
        return httpx.Response(200, json={'answers': [{'name': q['name'], 'type': q['type'],
            'choice': 'unknown' if q['name'].startswith('identity_') else 'none'
            if q['name'] == 'scene_inventory_closure' else 'visible', 'confidence': .99}
            for q in payload['questions']]})

    inventory = DemoVisualInventory(api_key='offline', client=httpx.Client(transport=httpx.MockTransport(respond)))
    reason = ('visual_identity_clarification_provenance_missing' if change == 'generation'
              else 'visual_identity_subject_binding_invalid')
    with pytest.raises(VisualInventoryError, match=reason):
        inventory.resolve(capture, [detection], instruction='identify', entities={'red': {'kind': 'cube'}},
                          purpose='associate_inventory')
    assert len(calls) == 1


def test_fully_accepted_initial_identity_does_not_add_images_or_a_model_call():
    capture, detection = fixture()
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'answers': [{'name': q['name'], 'type': q['type'],
            'choice': 'red' if q['name'].startswith('identity_') else 'none'
            if q['name'] == 'scene_inventory_closure' else 'visible', 'confidence': .99}
            for q in payload['questions']]})

    inventory = DemoVisualInventory(api_key='offline', client=httpx.Client(transport=httpx.MockTransport(respond)))
    inventory.resolve(capture, [detection], instruction='identify', entities={'red': {'kind': 'cube'}},
                      purpose='associate_inventory')
    assert len(calls) == 1
    assert not any('current_candidate_sam_subject' in part.get('text', '')
                   for message in calls[0]['input'] for part in message['content'])


@pytest.mark.parametrize('mixed', [False, True])
def test_clarification_subjects_include_remaining_same_capture_predicates_and_omit_accepted(mixed):
    from luna_mujoco.identity_correspondence import correspondence_payload, scope_identity_clarification
    from luna_mujoco.identity_proposals import scope_same_capture_identity

    capture, request, subjects = subject_case()
    content = request['input'][0]['content']
    context = json.loads(content[0]['text'])
    context.update(fixed_entities={'box': {'kind': 'bin'}, 'other': {'kind': 'bin'}},
        same_capture_reference={'initial_reference_observation_id': capture['observation_id'],
            'current_observation_id': capture['observation_id'], 'generation': capture['generation'],
            'verified_camera_ids': ['3']},
        same_capture_box_correspondence_proposals=[{'candidate': 'candidate-' + str(number), 'camera': '3',
            'same_camera_reference_box_overlaps': [{'reference_entity_id': 'box', 'box_iou': .95}]}
            for number in (2, 6)])
    content[0]['text'] = json.dumps(context)
    row = json.loads(content[-1]['text'])
    row['box'] = [.1, .1, .6, .7]
    content[-1]['text'] = json.dumps(row)
    numbers = [2, 6, 8] if mixed else [2, 6]
    for number in numbers:
        name = 'candidate-' + str(number)
        if number != 6:
            copied = deepcopy(row)
            copied['candidate'] = copied['candidate_provenance']['candidate'] = name
            content.append({'type': 'input_text', 'text': json.dumps(copied)})
            subjects[name] = {'silhouette_mask': subjects['candidate-6']['silhouette_mask'].copy(),
                              'candidate_provenance': copied['candidate_provenance']}
    request['questions'] = [{'name': 'identity_' + str(number), 'type': 'choice', 'instructions': 'Which subject?',
        'choices': [{'value': 'box', 'description': 'R1 blue box'}, {'value': 'other', 'description': 'R2 box'},
                    {'value': 'unknown'}, {'value': 'reject'}]} for number in numbers]
    presented, proposals = scope_same_capture_identity(request)
    answers = [{'name': 'identity_2', 'type': 'predicate', 'probability': .99},
               {'name': 'identity_6', 'type': 'predicate', 'probability': .5}]
    if mixed:
        answers.append({'name': 'identity_8', 'type': 'choice', 'choice': 'unknown', 'confidence': .4})
    scoped, retained = scope_identity_clarification(presented, {'original_decisions_answers': answers,
        'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'request_hash': 'original-accepted-request', 'same_capture_identity_proposals': proposals}, .8)
    assert list(retained) == ['identity_2']
    pairwise, _ = correspondence_payload(scoped)
    original = deepcopy(pairwise)
    result = append_views(pairwise, capture, subjects)
    assert result['questions'] == original['questions']
    assert any(q['name'] == 'identity_6' and q['type'] == 'predicate' for q in result['questions'])
    assert not any(q['name'] == 'identity_2' for q in result['questions'])
    descriptions = [json.loads(p['text']) for p in result['input'][0]['content']
        if p['type'] == 'input_text' and '"image_role": "current_candidate_sam_subject"' in p['text']]
    assert [row['candidate'] for row in descriptions] == (['candidate-6', 'candidate-8'] if mixed else ['candidate-6'])
