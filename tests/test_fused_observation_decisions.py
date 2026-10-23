"""One observation produces one joint three-view Decisions request."""
import json

import pytest
from test_demo_lifecycle_perception import client
from test_demo_release_result import release_inputs
from test_drop_region_presentation import measured_drop_inputs, request
from test_lifecycle_persistent_identity import run_case

from luna_mujoco.demo_transfer_verification import verify_release_result
from luna_mujoco.visual_inventory import image_part


def test_place_uses_one_fused_result_with_all_original_views():
    capture, binding, reference = release_inputs()
    reference.setdefault('demo_task', {})['workflow'] = 'unpack_transfer'
    visual, calls = client()
    result = verify_release_result(visual, capture, binding, reference)
    assert len(calls) == 1
    payload = calls[0]
    assert [q['name'] for q in payload['questions']] == [
        'place_result', 'workpiece_still_held', 'observation_sufficient']
    assert 'ONLY: place_result, workpiece_still_held' in payload['questions'][-1]['instructions']
    assert 'each camera separately' not in json.dumps(payload)
    assert 'In camera' not in json.dumps(payload['questions'])
    images = [part for part in payload['input'][0]['content'] if part['type'] == 'input_image']
    assert all(image_part(view['rgb']) in images for view in capture['views'].values())
    assert result['visual_evidence']['view_scope'] == 'fused_current_three_rgb'
    assert result['visual_evidence']['status'] == 'at_destination'


@pytest.mark.parametrize('answer,confidence', [('no', .95), ('unknown', .95), ('yes', .79)])
def test_fused_place_does_not_promote_uncertain_or_failed_result(answer, confidence):
    capture, binding, reference = release_inputs()
    visual, _ = client({'place_result': (answer, confidence)})
    result = verify_release_result(visual, capture, binding, reference)
    assert result['visual_evidence']['status'] == 'unknown'


def test_current_roles_and_original_identity_share_one_request_and_hash():
    result, calls, reference, _ = run_case()
    assert len(calls) == 1
    names = [q['name'] for q in calls[0]['questions']]
    assert len(names) == len(set(names))
    assert 'identity_2' in names and 'wrapped_identity_2' in names
    assert result['wrapped_identity_verification']['request_hash'] == result['evidence']['request_hash']
    images = [part for part in calls[0]['input'][0]['content'] if part['type'] == 'input_image']
    assert all(image_part(view['rgb']) in images for view in reference['views'].values())


def test_drop_asks_one_complete_in_bin_result_without_duplicate_presence_gate():
    result, payload = request(*measured_drop_inputs())
    assert [q['name'] for q in payload['questions']] == ['current_item_added', 'observation_sufficient']
    assert 'ONLY: current_item_added' in payload['questions'][-1]['instructions']
    assert 'all CURRENT views together' in payload['questions'][0]['instructions']
    assert result['visual_evidence']['view_scope'] == 'fused_current_three_rgb'
