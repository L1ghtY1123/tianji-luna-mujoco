"""Container IDs use frozen measured correspondence, never workpiece tracking."""
import importlib
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError


def module():
    spec = importlib.util.find_spec('luna_mujoco.demo_container_tracking')
    assert spec is not None, 'explicit measured container tracking is missing'
    return importlib.import_module(spec.name)


def case():
    mask = np.zeros((48, 64), bool)
    mask[4:44, 4:60] = True
    view = {'rgb': np.zeros((48, 64, 3), np.uint8), 'depth_m': np.ones((48, 64), np.float32),
            'valid_depth': np.ones((48, 64), bool), 'registered_to': 'rgb', 'calibration_id': 'cal-1',
            'intrinsics': {'fx': 100., 'fy': 100., 'cx': 32., 'cy': 24.}, 'T_world_camera': np.eye(4).tolist()}
    capture = {'source': 'rgbd', 'observation_id': 'initial', 'generation': 2, 'time': 1.,
               'views': {camera: deepcopy(view) for camera in ('1', '2', '3')}}
    entities = {'right': {'id': 'right', 'kind': 'bin', 'color': 'blue', 'roles': ['collision']},
                'pink': {'id': 'pink', 'kind': 'object', 'color': 'pink', 'roles': ['collision'],
                         'visual_membership': 'required'}}
    reference = {'observation_id': 'initial', 'entities': deepcopy(entities),
                 'views': {camera: {'rgb': v['rgb']} for camera, v in capture['views'].items()},
                 'demo_task': {'right_bin_id': 'right', 'source_object_ids': ['pink']}}
    observation = deepcopy(capture)
    for camera, v in observation['views'].items():
        v['instance_masks'] = {'right': mask.copy()} if camera == '1' else {}
    observation.update(visual_reference=reference, scene_entities=entities, visual_inventory={
        'missing_geometry_ids': [], 'identity_coverage': {'purpose': 'initialize_inventory',
            'accepted_entity_ids': ['right', 'pink'], 'required_entity_ids': ['right', 'pink']},
        'evidence': {'source': 'openai_decisions', 'observation_id': 'initial', 'request_hash': 'a' * 64,
            'answers': [{'name': 'identity_0', 'type': 'choice', 'choice': 'right', 'confidence': .99}]}},
        demo_inventory_check={'status': 'none', 'confidence': .99})
    current = deepcopy(capture)
    current.update(observation_id='later', time=2.)
    pixels = np.column_stack(np.nonzero(mask))
    detection = {'camera_id': '1', 'kind': 'bin', 'prompt': 'blue box', 'score': .9,
                 'mask': mask.copy(), 'silhouette_mask': mask.copy(), 'pixels_rc': pixels,
                 'points': np.column_stack([pixels[:, 1] / 100., pixels[:, 0] / 100., np.ones(len(pixels))]),
                 'pixels': current['views']['1']['rgb'][mask]}
    return observation, capture, current, [detection], entities


def tracked_case():
    observation, initial, current, detections, entities = case()
    reference = module().reference_with_container_tracks(observation, initial)
    return observation, initial, current, detections, entities, reference


def test_tracks_are_explicit_initial_cleaned_measurements_without_mutating_reference():
    observation, initial, current, detections, entities = case()
    original = deepcopy(observation['visual_reference'])
    reference = module().reference_with_container_tracks(observation, initial)
    assert 'container_identity_tracks' not in observation['visual_reference']
    assert reference['entities'] == original['entities']
    for camera in original['views']:
        np.testing.assert_array_equal(reference['views'][camera]['rgb'], original['views'][camera]['rgb'])
    proof = module().container_correspondence(current, detections, reference, entities)
    assert proof['candidate_mapping'] == {'0': 'right'}
    assert proof['source'] == 'measured_container_correspondence'
    assert 'confidence' not in proof
    assert proof['matches']['0']['common_valid_pixels'] == 2240
    assert proof['matches']['0']['depth_residual_p95_m'] == 0.


@pytest.mark.parametrize('bad', ['generation', 'reference_id', 'reference_entities', 'owner', 'depth',
    'mask_changed', 'calibration', 'intrinsics', 'pose_translation', 'pose_rotation', 'registration',
    'too_few_valid', 'nonfinite', 'workpiece', 'old_capture', 'altered_anchor'])
def test_bad_or_changed_correspondence_falls_back_without_assigning_identity(bad):
    observation, initial, current, detections, entities, reference = tracked_case()
    view = current['views']['1']
    if bad == 'generation': current['generation'] += 1
    elif bad == 'reference_id': reference['observation_id'] = 'another-initial'
    elif bad == 'reference_entities': reference['entities']['right']['color'] = 'red'
    elif bad == 'owner':
        observation['visual_reference']['owner_id'] = 'run-a'
        reference = module().reference_with_container_tracks(observation, initial)
        reference['owner_id'] = 'run-b'
    elif bad == 'depth': view['depth_m'] += .021
    elif bad == 'mask_changed': detections[0]['silhouette_mask'][:20] = False
    elif bad == 'calibration': view['calibration_id'] = 'other'
    elif bad == 'intrinsics': view['intrinsics']['fx'] += 1
    elif bad == 'pose_translation': view['T_world_camera'][0][3] = .0011
    elif bad == 'pose_rotation':
        angle = np.deg2rad(.21)
        view['T_world_camera'] = [[np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    elif bad == 'registration': view['registered_to'] = 'depth'
    elif bad == 'too_few_valid': view['valid_depth'][:42] = False
    elif bad == 'nonfinite': view['depth_m'][:] = np.nan
    elif bad == 'workpiece': detections[0]['kind'] = 'object'
    elif bad == 'old_capture': current['observation_id'] = 'initial'
    elif bad == 'altered_anchor': reference['container_identity_tracks']['containers']['right']['views']['1']['depth_m'] += .001
    proof = module().container_correspondence(current, detections, reference, entities)
    assert proof['candidate_mapping'] == {}


def test_ambiguous_two_initial_boxes_never_choose_one_by_order():
    observation, initial, current, detections, entities = case()
    entities['left'] = deepcopy(entities['right']) | {'id': 'left'}
    observation['visual_reference']['entities'] = deepcopy(entities)
    observation['visual_reference']['demo_task']['source_bin_id'] = 'left'
    observation['views']['1']['instance_masks']['left'] = detections[0]['mask'].copy()
    observation['visual_inventory']['evidence']['answers'].append(
        {'name': 'identity_1', 'type': 'choice', 'choice': 'left', 'confidence': .99})
    observation['visual_inventory']['identity_coverage']['accepted_entity_ids'].append('left')
    reference = module().reference_with_container_tracks(observation, initial)
    assert module().container_correspondence(current, detections, reference, entities)['candidate_mapping'] == {}


def test_unconfirmed_initial_bin_cannot_become_track():
    observation, initial, current, detections, entities = case()
    observation['visual_inventory']['evidence']['answers'][0]['confidence'] = .79
    reference = module().reference_with_container_tracks(observation, initial)
    assert module().container_correspondence(current, detections, reference, entities)['candidate_mapping'] == {}


def test_mixed_current_groups_keep_all_choices_and_do_not_forge_answers():
    _, _, capture, detections, entities, reference = tracked_case()
    pink = deepcopy(detections[0])
    pink.update(camera_id='2', kind='object', prompt='pink object')
    head = deepcopy(detections[0])
    head.update(camera_id='3')
    detections += [pink, head]
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for question in payload['questions']:
            value = {'identity_1': 'pink', 'identity_2': 'right', 'demo_unlisted_pink': 'none'}.get(
                question['name'], 'visible')
            answers.append({'name': question['name'], 'type': 'choice', 'choice': value, 'confidence': .95})
        return httpx.Response(200, json={'answers': answers})
    visual = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = visual.resolve(capture, detections, entities=entities, reference=reference,
                            instruction='move pink', purpose='associate_inventory')
    questions = {q['name']: q for q in calls[0]['questions']}
    assert 'identity_0' not in questions and 'identity_1' in questions and 'identity_2' in questions
    assert 'right' in [c['value'] for c in questions['identity_2']['choices']]
    assert 'visibility_right' in questions
    assert not any(a['name'] == 'identity_0' for a in result['evidence']['answers'])
    assert result['missing_geometry_ids'] == []
    bins = [g for g in result['groups'] if g[0]['entity_id'] == 'right']
    assert len(bins) == 1 and {e['camera_id'] for e in bins[0]} == {'1', '3'}
    coverage = result['identity_coverage']
    assert coverage['source'] == 'mixed_decisions_measured_container'
    assert coverage['measured_container_entity_ids'] == ['right']
    assert coverage['decisions_accepted_entity_ids'] == ['pink', 'right']
    assert result['container_correspondence']['candidate_mapping'] == {'0': 'right'}
    from luna_mujoco.perception import _estimate_object
    obj = _estimate_object(bins[0], {'surface_height_m': .5}, observation_provenance={'observation_id': 'later'})
    assert obj['identity_source'] == 'mixed_decisions_measured_container'
    assert len(obj['identity_evidence']) == 2


def test_actual_2e64_only_confirmed_container_masks_match():
    from pathlib import Path

    path = Path(__file__).parent / 'fixtures/container_correspondence_2e64'
    arrays = np.load(path.with_suffix('.npz'), allow_pickle=False)

    def restore(value):
        if isinstance(value, dict) and 'array_key' in value:
            return arrays[value['array_key']].copy()
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value
    frozen = restore(json.loads(path.with_suffix('.json').read_text(encoding='utf-8')))
    reference = module().reference_with_container_tracks(frozen['observation'], frozen['initial_capture'])
    proof = module().container_correspondence(frozen['current_capture'], frozen['detections'],
                                             reference, frozen['entities'])
    assert proof['candidate_mapping'] == {
        '6': 'instance-d6076b782ef749be', '7': 'instance-d6076b782ef749be',
        '9': 'instance-f1d690494bc64f61', '10': 'instance-f1d690494bc64f61'}
    assert proof['matches']['9']['mask_iou'] > .9994
    assert proof['matches']['6']['mask_iou'] > .975
    assert all(row['depth_residual_p95_m'] < .0001 for row in proof['matches'].values())
    assert all(row['common_valid_pixels'] > 71000 for row in proof['matches'].values())
    arrays.close()


def test_changed_container_falls_back_to_original_question_and_unknown_still_blocks():
    _, _, capture, detections, entities, reference = tracked_case()
    capture['views']['1']['depth_m'] += .021
    pink = deepcopy(detections[0]) | {'camera_id': '2', 'kind': 'object', 'prompt': 'pink object'}
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = [{'name': q['name'], 'type': 'choice', 'choice': {
            'identity_0': 'unknown', 'identity_1': 'pink', 'demo_unlisted_pink': 'none'}.get(q['name'], 'visible'),
            'confidence': .99} for q in payload['questions']]
        return httpx.Response(200, json={'answers': answers})
    visual = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown'):
        visual.resolve(capture, detections + [pink], entities=entities, reference=reference,
                       instruction='move pink', purpose='associate_inventory')
    assert 'identity_0' in {q['name'] for q in calls[0]['questions']}
    assert 'container_correspondence' not in json.loads(calls[0]['input'][0]['content'][0]['text'])


def test_merge_retains_fresh_points_and_removes_pixels_owned_by_current_workpiece():
    _, _, capture, detections, entities, reference = tracked_case()
    tracking = module().container_correspondence(capture, detections, reference, entities)
    workpiece = deepcopy(detections[0]) | {'entity_id': 'pink', 'kind': 'object', 'prompt': 'pink object'}
    workpiece['mask'][:] = False
    workpiece['mask'][10:14, 10:14] = True
    workpiece['silhouette_mask'] = workpiece['mask'].copy()
    pixels = np.column_stack(np.nonzero(workpiece['mask']))
    workpiece.update(pixels_rc=pixels, points=np.ones((16, 3)) * 123., pixels=np.zeros((16, 3), np.uint8))
    fresh = detections[0]['points'].copy() + .001
    detections[0]['points'] = fresh
    result = {'groups': [[workpiece]], 'evidence': {'request_hash': 'current-real-call'}}
    merged = module().merge_container_correspondence(DemoVisualInventory(), detections, entities, tracking, result)
    bin_entry = next(g[0] for g in merged['groups'] if g[0]['entity_id'] == 'right')
    assert len(bin_entry['points']) == len(fresh) - 16
    assert not bin_entry['mask'][10:14, 10:14].any()
    expected = fresh[~workpiece['mask'][detections[0]['pixels_rc'][:, 0], detections[0]['pixels_rc'][:, 1]]]
    np.testing.assert_array_equal(bin_entry['points'], expected)
    assert bin_entry['identity_evidence'][0]['source'] == 'measured_container_correspondence'
    assert merged['missing_geometry_ids'] == []


def test_demo_observation_adds_tracks_after_current_mask_processing(monkeypatch):
    from luna_mujoco.demo_perception import DemoPerception
    from luna_mujoco.perception import Perception

    observation, capture, _, _, _ = case()
    observation.update(objects=[], demo_program={'version': 1, 'steps': []})
    observation['visual_inventory']['demo_inventory_check'] = observation['demo_inventory_check']
    observation['visual_reference']['demo_program'] = observation['demo_program']
    observation['visual_reference']['demo_task']['source_bin_id'] = 'right'
    monkeypatch.setattr(Perception, 'observe', lambda *args, **kwargs: deepcopy(observation))
    engine = DemoPerception()
    engine.demo_tables = {}
    result = engine.observe(capture)
    assert result['visual_reference']['container_identity_tracks']['containers']['right']['views']['1']['mask_source'] == (
        'initial_confirmed_instance_mask_after_robot_and_surface_filter')
    assert 'container_identity_tracks' not in observation['visual_reference']
