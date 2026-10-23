"""Unpack identity must see the actual previously confirmed table release."""
import json
from copy import deepcopy

import pytest
from test_demo_lifecycle_perception import capture_and_action, client, record, work_area

from luna_mujoco import demo_lifecycle_perception as perception
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import image_part


def reference_inputs():
    current, action = capture_and_action()
    before = deepcopy(current)
    before.update(observation_id='table-before', time=.5, captured_perf_counter_ns=50)
    for view in before['views'].values():
        view['rgb'] = view['rgb'].copy()
        view['rgb'][0, 0] = [231, 13, 101]
    lifecycle = record('unpack_pending')
    lifecycle['table_visual_evidence'].update(observation_id=before['observation_id'], source='openai_decisions')
    metadata = perception.table_release_reference(before, lifecycle['table_visual_evidence'], 'run')
    return current, action, before, lifecycle, metadata


def test_actual_before_and_current_images_have_separate_bound_roles():
    current, action, before, lifecycle, metadata = reference_inputs()
    visual, calls = client()
    result = perception.verify_unpacking(visual, current, target_id='pink', lifecycle=lifecycle,
        action_evidence=action, owner_id='run', work_area=work_area(),
        table_release_capture=before, table_release_reference=metadata, active_target_ids=['pink'])
    content = calls[0]['input'][0]['content']
    images = [part for part in content if part['type'] == 'input_image']
    # Every unique historical/current RGB remains; identical camera pixels share one image.
    expected = {image_part(frame['views'][camera]['rgb'])['image_url']
                for frame in (before, current) for camera in ('2', '1', '3')}
    assert expected <= {image['image_url'] for image in images}
    assert len({image['image_url'] for image in images}) == len(images)
    context = json.loads(content[0]['text'])
    assert 'table_release_reference' not in context
    assert 'one scene' in context['views']
    assert 'BEFORE UNPACKING, AFTER VERIFIED TABLE RELEASE RGB camera 3' in json.dumps(calls[0])
    assert result['visual_evidence']['table_release_reference'] == metadata
    assert 'views' not in json.dumps(metadata)


@pytest.mark.parametrize('mutation', ['missing_capture', 'missing_metadata', 'target', 'observation',
    'generation', 'release', 'binding', 'request', 'owner', 'pixels', 'views', 'future'])
def test_missing_or_changed_release_reference_stops_without_model_call(mutation):
    current, action, before, lifecycle, metadata = reference_inputs()
    if mutation == 'missing_capture':
        before = None
    elif mutation == 'missing_metadata':
        metadata = None
    elif mutation in {'target', 'release', 'binding', 'request', 'owner'}:
        key = {'target': 'target_id', 'release': 'release_feedback_id', 'binding': 'binding_id',
               'request': 'request_hash', 'owner': 'owner_id'}[mutation]
        metadata[key] = 'different'
    elif mutation == 'observation':
        before['observation_id'] = 'different'
    elif mutation == 'generation':
        before['generation'] += 1
    elif mutation == 'pixels':
        before['views']['2']['rgb'][0, 0] = 0
    elif mutation == 'views':
        before['views'].pop('1')
    else:
        before.update(time=3., captured_perf_counter_ns=300)
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='table_release_reference'):
        perception.verify_unpacking(visual, current, target_id='pink', lifecycle=lifecycle,
            action_evidence=action, owner_id='run', work_area=work_area(),
            table_release_capture=before, table_release_reference=metadata, active_target_ids=['pink'])
    assert not calls


def test_before_images_and_valid_history_do_not_promote_uncertain_same_item():
    current, action, before, lifecycle, metadata = reference_inputs()
    visual, _ = client({'unpack_result': ('yes', .77)})
    result = perception.verify_unpacking(visual, current, target_id='pink', lifecycle=lifecycle,
        action_evidence=action, owner_id='run', work_area=work_area(),
        table_release_capture=before, table_release_reference=metadata, active_target_ids=['pink'])
    assert result['status'] == 'unknown' and not result['success']
    assert result['appearance_revision'] == 0 and result['target_id'] == 'pink'
