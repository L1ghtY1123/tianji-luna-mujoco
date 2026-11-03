import json
from copy import deepcopy

import pytest
from test_demo_identity_presentation import metadata, source_payload

from luna_mujoco.demo_identity_presentation import improve_identity_presentation
from luna_mujoco.identity_comparison_images import same_capture_comparison_images
from luna_mujoco.identity_proposals import scope_same_capture_identity


def comparison_payload():
    payload, _, _, _ = source_payload()
    return scope_same_capture_identity(improve_identity_presentation(payload))


def test_question_images_are_scoped_without_mutating_parser_payload_or_other_questions():
    from luna_mujoco.identity_comparison_images import scope_comparison_questions

    payload, proposals = comparison_payload()
    payload = same_capture_comparison_images(payload, proposals)
    original = deepcopy(payload)
    result = scope_comparison_questions(payload)
    assert payload == original
    questions = {q['name']: q for q in result['questions']}
    for question in payload['questions']:
        updated = questions[question['name']]
        if question['name'] in proposals:
            parts = updated['instructions']
            assert parts[0]['type'] == 'input_text'
            assert parts[2]['type'] == 'input_image'
            meta = json.loads(parts[1]['text'])
            assert meta['question'] == question['name']
            assert meta['reference'] in parts[0]['text']
            assert meta['candidate'] in parts[0]['text']
        else:
            assert updated == question
    assert not metadata(result, 'same_capture_identity_comparison')
    assert scope_comparison_questions(result) == result


def test_comparison_preserves_full_evidence_questions_and_never_assigns_identity():
    payload, proposals = comparison_payload()
    original = deepcopy(payload)
    result = same_capture_comparison_images(payload, proposals)
    assert payload == original
    assert result['questions'] == payload['questions']
    content = payload['input'][0]['content']
    assert result['input'][0]['content'][:len(content)] == content
    panels = metadata(result, 'same_capture_identity_comparison')
    assert len(panels) == len(proposals) == 3
    assert {x['question'] for x in panels} == set(proposals)
    assert all(x['camera_id'] == '2' and x['observation_id'] == 'same-capture' for x in panels)
    assert same_capture_comparison_images(result, proposals) == result


def test_pair_panels_are_drawn_from_raw_pixels_not_other_subject_annotations():
    import numpy as np
    from luna_mujoco.demo_identity_presentation import _decode_image
    from luna_mujoco.visual_inventory import image_part

    payload, proposals = comparison_payload()
    before = same_capture_comparison_images(payload, proposals)
    changed = deepcopy(payload)
    content = changed['input'][0]['content']
    for index, part in enumerate(content[:-1]):
        if part['type'] != 'input_text':
            continue
        try:
            meta = json.loads(part['text'])
        except ValueError:
            continue
        if meta.get('image_role') in {'initial_reference_annotations', 'current_annotations'}:
            rgb, _ = _decode_image(content[index + 1])
            content[index + 1] = image_part(np.full_like(rgb, [255, 0, 255]))
    after = same_capture_comparison_images(changed, proposals)
    assert before['input'][0]['content'][len(content):] == after['input'][0]['content'][len(content):]


@pytest.mark.parametrize('damage', ['capture', 'generation', 'camera', 'source_pixels', 'question_type'])
def test_unbound_or_changed_views_never_get_same_capture_panels(damage):
    payload, proposals = comparison_payload()
    content = payload['input'][0]['content']
    context = json.loads(content[0]['text'])
    if damage == 'capture':
        context['same_capture_reference']['initial_reference_observation_id'] = 'older'
    elif damage == 'generation':
        context['same_capture_reference']['generation'] -= 1
    elif damage == 'camera':
        context['same_capture_reference']['verified_camera_ids'] = ['1']
    elif damage == 'source_pixels':
        for part in content[1:]:
            if part['type'] == 'input_text':
                try:
                    record = json.loads(part['text'])
                except ValueError:
                    continue
                if record.get('image_role') == 'current_annotations':
                    record['source_image_sha256'] = 'different'
                    part['text'] = json.dumps(record)
    else:
        for question in payload['questions']:
            question['type'] = 'choice'
    content[0]['text'] = json.dumps(context)
    assert same_capture_comparison_images(payload, proposals) == payload
