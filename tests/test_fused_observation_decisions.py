"""Placement judges all three RGB views in one request; legacy drop is separate."""

import pytest
from processing_fixtures import client
from test_demo_release_result import release_inputs

from luna_mujoco.demo_transfer_verification import verify_release_result
from luna_mujoco.visual_inventory import image_part


def test_place_uses_all_original_views_with_one_request():
    capture, binding, reference = release_inputs()
    for index, view in enumerate(capture['views'].values(), 1):
        view['rgb'] = view['rgb'].copy()
        view['rgb'][:] = index * 50
    visual, calls = client()
    result = verify_release_result(visual, capture, binding, reference)
    assert len(calls) == 1
    for payload in calls:
        assert [q['name'] for q in payload['questions']] == ['place_result', 'workpiece_still_held']
        parts = payload['input'][0]['content']
        images = [part for part in parts if part['type'] == 'input_image']
        assert len(images) == 4
        assert all(image_part(capture['views'][camera]['rgb']) in images for camera in ('1', '2', '3'))
    assert result['visual_evidence']['view_scope'] == 'fused_current_three_rgb'
    assert result['visual_evidence']['status'] == 'at_destination'


@pytest.mark.parametrize('answer,confidence', [('no', .95), ('unknown', .95), ('yes', .79)])
def test_independent_place_does_not_promote_uncertain_or_failed_result(answer, confidence):
    capture, binding, reference = release_inputs()
    visual, _ = client({'place_result': (answer, confidence)})
    result = verify_release_result(visual, capture, binding, reference)
    assert result['visual_evidence']['status'] == ('rejected' if answer == 'no' else 'unknown')


