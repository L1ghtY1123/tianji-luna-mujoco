import json

import httpx
import pytest
from processing_fixtures import before_kwargs, capture_and_action, release_evidence, work_area

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.robot_skills.processing import create_unpack_product
from luna_mujoco.robot_skills.processing_evidence import verify_unpacking


@pytest.mark.parametrize('case', ['complete', 'split', 'all_refused', 'malformed'])
def test_only_one_complete_independent_witness_can_create_a_product(case):
    def respond(request):
        payload = json.loads(request.content)
        camera = json.loads(payload['input'][0]['content'][0]['text'])['camera_id']
        answers = []
        for question in payload['questions']:
            name = question['name']
            if case == 'all_refused' or camera == '1':
                answers.append({'type': 'refusal', 'name': name})
            else:
                probability = (.99 if case == 'complete' and camera == '3'
                    or case == 'split' and (camera == '2') == (name == 'unpack_result') else .1)
                answers.append({'type': 'predicate', 'name': name,
                                'probability': None if case == 'malformed' else probability})
        return httpx.Response(200, json={'answers': answers})

    visual = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    capture, action = capture_and_action()
    action['process_id'] = 'process'
    kwargs = dict(target_id='pink', release_evidence=release_evidence(), action_evidence=action,
        owner_id='run', work_area=work_area(), **before_kwargs())
    if case == 'malformed':
        with pytest.raises(ValueError, match='invalid_response'):
            verify_unpacking(visual, capture, **kwargs)
        return
    result = verify_unpacking(visual, capture, **kwargs)
    assert result['success'] is (case == 'complete')
    assert result['visual_evidence']['camera_evidence'][0]['answers'][0]['type'] == 'refusal'
    if case == 'complete':
        assert result['visual_evidence']['selected_camera_id'] == '3'
        product = create_unpack_product('pink', action, result, capture, owner_id='run')
        assert product['input_entity_id'] == 'pink'
        result['visual_evidence']['selected_camera_id'] = '1'
        with pytest.raises(ValueError, match='product_requires_confirmed_visible_result'):
            create_unpack_product('pink', action, result, capture, owner_id='run')
    else:
        with pytest.raises(ValueError, match='product_requires_confirmed_visible_result'):
            create_unpack_product('pink', action, result, capture, owner_id='run')
