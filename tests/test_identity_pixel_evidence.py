"""Identical calibrated pixels are evidence for Decisions, never an ID binding."""
import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from test_demo_same_capture import inventory, same_capture_inputs


def inputs():
    capture, detections, entities, reference = same_capture_inputs()
    view = capture['views']['2']
    view.update(camera_id='2', frame_id=capture['observation_id'],
                calibration_id='measured-calibration', intrinsics={'fx': 100., 'fy': 100., 'cx': 16., 'cy': 12.},
                T_world_camera=np.eye(4).tolist())
    initial = deepcopy(capture)
    initial['observation_id'] = 'initial-frame'
    initial['views']['2']['frame_id'] = 'initial-frame'
    reference.update(observation_id='initial-frame', views={'2': {'rgb': initial['views']['2']['rgb'].copy()}})
    return capture, detections, entities, reference, initial


def evidence(capture, detections, reference, initial):
    from luna_mujoco.identity_pixel_evidence import identical_reference_regions
    return identical_reference_regions(capture, detections, reference, initial)


def test_identical_region_records_exact_arrays_and_distinct_acquisition_ids():
    capture, detections, _, reference, initial = inputs()
    capture['views']['2']['rgb'][23, 31, 0] = 1  # Outside both locating boxes.
    rows = evidence(capture, detections, reference, initial)
    assert len(rows) == 1
    row = rows[0]
    assert row['candidate'] == 'candidate-0'
    assert row['reference_entity_id'] == 'pink' and row['camera'] == '2'
    assert row['reference_observation_id'] == 'initial-frame'
    assert row['current_observation_id'] == 'frame-1'
    assert row['union_roi_xyxy_px'] == [3, 4, 14, 16]
    assert row['pixel_count'] == 132
    assert row['rgb_sha256'] == hashlib.sha256(
        np.ascontiguousarray(capture['views']['2']['rgb'][4:16, 3:14]).tobytes()).hexdigest()
    assert row['identity_assignment'] is False


def test_comparison_covers_the_union_of_reference_and_candidate_boxes():
    capture, detections, _, reference, initial = inputs()
    reference['entities']['pink']['bbox'][0] = 0.
    assert evidence(capture, detections, reference, initial)[0]['union_roi_xyxy_px'] == [0, 4, 14, 16]
    capture['views']['2']['rgb'][5, 1, 0] = 1  # In reference-only part of the union.
    assert evidence(capture, detections, reference, initial) == []


@pytest.mark.parametrize('changed', [
    'rgb', 'depth_m', 'valid_depth', 'calibration_id', 'intrinsics', 'T_world_camera',
    'camera', 'reference_camera', 'generation', 'reference_id', 'reference_rgb', 'frame_id',
    'missing_reference_capture', 'missing_depth', 'invalid_depth', 'dtype', 'signed_zero',
])
def test_pixel_claim_is_absent_when_any_required_evidence_differs(changed):
    capture, detections, _, reference, initial = inputs()
    current = capture['views']['2']
    if changed == 'rgb':
        current['rgb'][5, 4, 0] ^= 1
    elif changed == 'depth_m':
        current['depth_m'][5, 4] += 1e-9
    elif changed == 'valid_depth':
        current['valid_depth'][5, 4] = False
    elif changed == 'calibration_id':
        current['calibration_id'] = 'other'
    elif changed == 'intrinsics':
        current['intrinsics']['fx'] += 1e-9
    elif changed == 'T_world_camera':
        current['T_world_camera'][0][3] += 1e-9
    elif changed == 'camera':
        capture['views']['1'] = current
        detections[0]['camera_id'] = '1'
    elif changed == 'reference_camera':
        reference['entities']['pink']['reference_camera_id'] = '1'
    elif changed == 'generation':
        initial['generation'] += 1
    elif changed == 'reference_id':
        initial['observation_id'] = 'unrelated'
    elif changed == 'reference_rgb':
        reference['views']['2']['rgb'][5, 4, 0] ^= 1
    elif changed == 'frame_id':
        current['frame_id'] = 'stale'
    elif changed == 'missing_reference_capture':
        initial = None
    elif changed == 'missing_depth':
        initial['views']['2'].pop('depth_m')
    elif changed == 'invalid_depth':
        current['valid_depth'][:] = False
        initial['views']['2']['valid_depth'][:] = False
    elif changed == 'dtype':
        current['depth_m'] = current['depth_m'].astype(np.float32)
    elif changed == 'signed_zero':
        current['depth_m'][5, 4] = -0.
        initial['views']['2']['depth_m'][5, 4] = 0.
    assert evidence(capture, detections, reference, initial) == []


def test_demo_request_receives_pixel_evidence_without_removing_identity_options():
    capture, detections, entities, reference, initial = inputs()
    calls = []
    result = inventory(calls).resolve(capture, detections, entities=entities, reference=reference,
        reference_capture=initial, instruction='move all', purpose='associate_inventory')
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    assert context['identical_reference_regions'][0]['reference_entity_id'] == 'pink'
    assert 'same_capture_reference' not in context
    question = calls[0]['questions'][0]
    assert question['type'] == 'predicate'
    assert len(calls) == 1
    assert result['evidence']['identity_correspondence'][0]['probabilities']['pink'] == .99
    assert result['evidence']['answers'][0]['choice'] == 'pink'
