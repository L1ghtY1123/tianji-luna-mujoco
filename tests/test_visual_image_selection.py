"""Duplicate pixels keep their current role and correct global image references."""
import json
from copy import deepcopy

import pytest

from luna_mujoco.visual_image_selection import compact_decision_images


def text(value):
    return {'type': 'input_text', 'text': value}


def image(value):
    return {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + value}


def payload(*messages):
    return {'input': [{'role': 'user', 'content': parts} for parts in messages],
            'questions': [{'name': 'identity', 'instructions': 'Keep this judgment unchanged.'}]}


@pytest.mark.parametrize('label', ['CURRENT RGB camera 1', 'CURRENT camera 1 — original RGB'])
def test_current_raw_image_is_retained_in_its_original_position(label):
    original = payload([text('INITIAL REFERENCE RGB camera 1'), image('same'),
                        text(label), image('same')])
    before = deepcopy(original)
    result, counts = compact_decision_images(original)
    parts = result['input'][0]['content']
    assert parts[1]['type'] == 'input_text' and 'image 1 below' in parts[1]['text']
    assert parts[2:] == [text(label), image('same')]
    assert result['questions'] == before['questions'] and original == before
    assert counts == {'original_image_count': 2, 'image_count': 1, 'removed_image_count': 1}
    assert compact_decision_images(result)[0] == result


def test_forward_and_backward_aliases_use_final_global_numbers_across_messages():
    original = payload(
        [text('INITIAL REFERENCE RGB camera 1'), image('camera1'), image('other')],
        [text('CURRENT RGB camera 2'), image('camera2'),
         text('CURRENT RGB camera 1'), image('camera1'), image('other'), image('camera2')])
    result, counts = compact_decision_images(original)
    first, second = [message['content'] for message in result['input']]
    assert 'image 3 below' in first[1]['text']
    assert first[2] == image('other')
    assert second[1] == image('camera2') and second[3] == image('camera1')
    assert 'image 1 above' in second[4]['text']
    assert 'image 2 above' in second[5]['text']
    assert counts['image_count'] == 3


def test_empty_annotation_does_not_claim_the_camera_rgb_is_above():
    metadata = {'image_role': 'initial_reference_annotations', 'camera': '2', 'items': [], 'regions': []}
    result, _ = compact_decision_images(payload([
        text('INITIAL REFERENCE RGB camera 2'), image('same'), text(json.dumps(metadata)), image('empty'),
        text('CURRENT RGB camera 2'), image('same')]))
    message = result['input'][0]['content'][3]['text']
    assert 'same camera' in message
    assert 'above' not in message and 'below' not in message
    assert result['input'][0]['content'][-1] == image('same')


@pytest.mark.parametrize('label', [
    'Judge CURRENT RGB camera 1 using all evidence.',
    json.dumps({'rules': 'Use CURRENT RGB camera 1'}),
    'INITIAL REFERENCE RGB camera 1',
])
def test_incidental_current_text_never_changes_image_role(label):
    result, _ = compact_decision_images(payload([image('same'), text(label), image('same')]))
    parts = result['input'][0]['content']
    assert parts[0] == image('same')
    assert 'image 1 above' in parts[2]['text']
