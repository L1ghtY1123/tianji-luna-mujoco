"""Current measured ID locators may clarify a question, never answer it."""
import json
from copy import deepcopy

import numpy as np
import pytest
from test_agent_model_layer import Client

from luna_mujoco.agent.decisions import DecisionsJudgments
from luna_mujoco.visual_api import VisualInventoryError


def fixture():
    rgb = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    left = np.zeros((12, 16), dtype=bool)
    left[2:7, 1:5] = True
    right = np.fliplr(left).copy()
    view = {'rgb': rgb, 'camera_id': '3', 'frame_id': 'now', 'simulation_generation': 2,
            'calibration_id': 'calibrated', 'instance_masks': {'left': left, 'right': right}}
    observation = {'observation_id': 'now', 'generation': 2, 'views': {'3': view},
                   'objects': [{'id': 'right'}, {'id': 'left'}]}
    capture = deepcopy(observation)
    capture['views']['3'].pop('instance_masks')
    return observation, capture


def locations(observation):
    from luna_mujoco.task_binding_images import current_instance_locations
    return current_instance_locations(observation)


def content(capture, evidence):
    from luna_mujoco.task_binding_images import current_instance_location_content
    return current_instance_location_content(capture, evidence)


def test_all_measured_instances_get_locators_without_assigning_task_roles():
    observation, capture = fixture()
    result = content(capture, locations(observation))
    meta = json.loads(result[0]['text'])
    assert {row['id']: row['image_label'] for row in meta['items']} == {'left': 'ID1', 'right': 'ID2'}
    assert meta['items'][0]['box'] == [1/16, 2/12, 5/16, 7/12]
    assert all(len(row['mask_sha256']) == 64 for row in meta['items'])
    assert 'do not certify task membership' in meta['meaning']
    assert result[1]['type'] == 'input_image'
    np.testing.assert_array_equal(capture['views']['3']['rgb'], observation['views']['3']['rgb'])


@pytest.mark.parametrize('field', ['observation_id', 'generation', 'rgb', 'calibration_id', 'frame_id'])
def test_stale_or_altered_current_image_cannot_receive_measured_id_labels(field):
    observation, capture = fixture()
    evidence = locations(observation)
    if field in {'observation_id', 'generation'}:
        capture[field] = 'different'
    elif field == 'rgb':
        capture['views']['3']['rgb'][0, 0, 0] ^= 1
    else:
        capture['views']['3'][field] = 'different'
    with pytest.raises(VisualInventoryError, match='task_binding_locations_stale'):
        content(capture, evidence)


def test_missing_masks_do_not_fabricate_locations_and_invalid_masks_fail():
    observation, _ = fixture()
    assert locations({'observation_id': 'now', 'generation': 2}) == {}
    observation['views']['3']['instance_masks']['left'] = np.ones((1, 1), dtype=bool)
    with pytest.raises(VisualInventoryError, match='task_binding_mask_invalid'):
        locations(observation)


@pytest.mark.parametrize('probability, expected', [(.1, 'no'), (.5, 'unknown'), (.9, 'yes')])
def test_judgment_wire_preserves_questions_raw_images_and_model_verdict(probability, expected):
    observation, capture = fixture()
    question = {'name': 'destination_requirement_matches', 'instructions': 'Does the instruction refer to left?'}
    answer = {'answers': [{'type': 'predicate', 'name': question['name'], 'probability': probability}]}
    baseline_client = Client([answer])
    DecisionsJudgments(api_key='test', client=baseline_client).judge(capture, requests=[question], context={})
    client = Client([answer])
    result = DecisionsJudgments(api_key='test', client=client).judge(capture, requests=[question],
        context={'current_instance_locations': locations(observation)})
    baseline = baseline_client.requests[0][1]
    payload = client.requests[0][1]
    assert payload['questions'] == baseline['questions']
    assert payload['input'][0]['content'][1:3] == baseline['input'][0]['content'][1:3]
    assert len(payload['input'][0]['content']) == len(baseline['input'][0]['content']) + 2
    assert result['judgments'][question['name']]['status'] == expected
