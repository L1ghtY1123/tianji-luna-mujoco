import base64
import hashlib
import json
from copy import deepcopy
from io import BytesIO

import httpx
import numpy as np
import pytest
from PIL import Image
from test_visual_inventory import client_with_answers, fixture

from luna_mujoco.observation_labels import make_object_labels, project_placement_regions
from luna_mujoco.visual_inventory import VisualInventory, VisualInventoryError


def scoped_scene():
    capture, detection = fixture()
    capture['views']['3'].update(T_world_camera=np.eye(4),
        intrinsics={'fx': 20., 'fy': 20., 'cx': 16., 'cy': 12.})
    table = {'position': [0., 0., 1.], 'size': [1., 1., .02], 'surface_height_m': 1.,
             'source': 'measured_horizontal_workspace_plane', 'points_world_m': np.zeros((4, 3))}
    return capture, detection, table


def slot_record():
    return {'destination_id': 'box', 'status': 'released', 'region_id': 'region:box', 'slot_id': 2,
            'assigned_slot': {'slot_id': 2, 'center_world_xy_m': [.1, .1],
                'polygon_world_xy_m': [[0., 0.], [.2, 0.], [.2, .2], [0., .2]],
                'floor_z_m': 1., 'layout_revision': 'layout-1'}}


def test_outside_inventory_proposal_is_rejected_from_rgb():
    capture, detection, table = scoped_scene()
    calls = []
    result = VisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'red', 'identity_1': 'reject'}, calls)).resolve(
        capture, [detection, deepcopy(detection)], instruction='move red', table_scope=table,
        entities={'red': {'kind': 'cube'}})
    assert len(result['groups']) == 1
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    assert 'table_scope' not in context


def test_known_lifted_object_keeps_id_outside_table_bounds():
    capture, detection, table = scoped_scene()
    detection['points'][:, :] = [4., 4., 3.]
    result = VisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'red', 'scope_0': 'known_moved', 'progress_red': 'unknown'}, [])).resolve(
            capture, [detection], instruction='move red', table_scope=table,
            entities={'red': {'kind': 'cube', 'color': 'red'}}, progress={'red': {'status': 'held'}})
    assert result['groups'][0][0]['entity_id'] == 'red'


def test_identity_unknown_is_not_silently_removed():
    capture, detection, table = scoped_scene()
    inventory = VisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'unknown'}, []))
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown'):
        inventory.resolve(capture, [detection], instruction='move red', table_scope=table,
                          entities={'red': {'kind': 'cube', 'color': 'red'}})


def test_decisions_images_and_metadata_exclude_mask_interior_and_depth():
    capture, detection = fixture()
    entities = {'red': {'kind': 'cube', 'color': 'red', 'bbox': [.1, .1, .5, .7]}}
    reference = deepcopy(capture)
    reference['entities'] = entities
    calls = []
    for changed in (False, True):
        if changed:
            capture['views']['3']['depth_m'] *= 7
            capture['views']['3']['valid_depth'][:] = False
            detection['points'][:] = [12345., 12345., 12345.]
            # Preserve the outer rectangle while changing the actual segmentation.
            detection['silhouette_mask'][7:12, 6:10] = False
            detection['mask'] = detection['silhouette_mask'].copy()
            capture['views']['3']['instance_masks'] = {'red': detection['mask']}
            reference['views']['3']['instance_masks'] = {'red': ~detection['mask']}
        inventory = VisualInventory(api_key='test', client=client_with_answers({'identity_0': 'red'}, calls))
        inventory.resolve(capture, [detection], instruction='move red', entities=entities, reference=reference)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert sum(part['type'] == 'input_image' for part in calls[0]['input'][0]['content']) == 3


def test_assigned_slot_is_in_question_and_confirmation_evidence():
    capture, detection, table = scoped_scene()
    calls = []
    record = slot_record()
    result = VisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'red', 'scope_0': 'on_table', 'progress_red': 'at_destination'}, calls)).resolve(
            capture, [detection], instruction='move red', table_scope=table,
            entities={'red': {'kind': 'cube', 'color': 'red'}}, progress={'red': record},
            purpose='verify_transfer')
    evidence = result['progress']['red']
    assert evidence['status'] == 'at_destination'
    assert (evidence['region_id'], evidence['slot_id'], evidence['layout_revision']) == (
        'region:box', 2, 'layout-1')
    question = next(item for item in calls[0]['questions'] if item['name'] == 'progress_red')
    assert 'region:box' in question['instructions'] and 'slot 2' in question['instructions']
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    assert context['purpose'] == 'verify_transfer'
    assert 'assigned_slot' not in context['transport_progress']['red']
    assert 'TCP anchor' in question['instructions']


def test_assigned_slot_without_geometry_cannot_claim_destination():
    capture, detection = fixture()
    record = slot_record()
    del record['assigned_slot']
    result = VisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'red', 'progress_red': 'at_destination'}, [])).resolve(
            capture, [detection], instruction='move red',
            entities={'red': {'kind': 'cube'}}, progress={'red': record})
    assert result['progress']['red']['status'] == 'unknown'


def test_current_annotation_contains_only_temporary_ids_and_projected_slot():
    capture, detection, table = scoped_scene()
    original = capture['views']['3']['rgb'].copy()
    capture['views']['3']['instance_masks'] = {'unconfirmed-old-id': detection['mask']}
    calls = []
    record = slot_record()
    VisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'red', 'scope_0': 'on_table', 'progress_red': 'unknown'}, calls)).resolve(
            capture, [detection], instruction='move red', table_scope=table,
            entities={'red': {'kind': 'cube'}}, progress={'red': record})
    parts = calls[0]['input'][0]['content']
    metadata = [json.loads(part['text']) for part in parts
                if part['type'] == 'input_text' and part['text'].startswith('{')]
    annotation = next(item for item in metadata if item.get('image_role') == 'current_annotations')
    assert [item['id'] for item in annotation['items']] == ['candidate-0']
    assert annotation['regions'][0]['slot_id'] == 2
    assert 'table_polygon' not in annotation
    marker_index = next(i for i, part in enumerate(parts)
                        if part['type'] == 'input_text' and '"image_role": "current_annotations"' in part['text'])
    raster = np.asarray(Image.open(BytesIO(base64.b64decode(parts[marker_index + 1]['image_url'].split(',')[1]))))
    assert raster.shape[1] > original.shape[1]  # Text is placed in a margin, away from objects.
    np.testing.assert_array_equal(capture['views']['3']['rgb'], original)


def test_initial_reference_annotation_uses_confirmed_global_ids():
    capture, detection = fixture()
    reference = deepcopy(capture)
    reference['entities'] = {'fixed-red': {'kind': 'cube', 'color': 'red', 'bbox': [.1, .1, .5, .7]}}
    reference['views']['3']['instance_masks'] = {'fixed-red': detection['mask']}
    calls = []
    VisualInventory(api_key='test', client=client_with_answers({'identity_0': 'fixed-red'}, calls)).resolve(
        capture, [detection], instruction='move red', reference=reference,
        entities={'fixed-red': {'kind': 'cube'}})
    metadata = [json.loads(part['text']) for part in calls[0]['input'][0]['content']
                if part['type'] == 'input_text' and part['text'].startswith('{')]
    annotation = next(item for item in metadata if item.get('image_role') == 'initial_reference_annotations')
    assert annotation['items'][0]['id'] == 'fixed-red'


def test_each_assigned_slot_is_projected_when_only_progress_supplies_geometry():
    capture, detection, table = scoped_scene()
    record = slot_record()
    second = deepcopy(record)
    second['slot_id'] = second['assigned_slot']['slot_id'] = 3
    calls = []
    VisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'red', 'scope_0': 'on_table', 'progress_red': 'unknown',
        'progress_other': 'unknown'}, calls)).resolve(
            capture, [detection], instruction='move red', table_scope=table,
            entities={'red': {'kind': 'cube'}, 'other': {'kind': 'cube'}},
            progress={'red': record, 'other': second})
    metadata = [json.loads(part['text']) for part in calls[0]['input'][0]['content']
                if part['type'] == 'input_text' and part['text'].startswith('{')]
    annotation = next(item for item in metadata if item.get('image_role') == 'current_annotations')
    assert {region['slot_id'] for region in annotation['regions']} == {2, 3}


def test_request_audit_is_exact_and_once_per_network_request_not_cache():
    capture, detection = fixture()
    calls, events = [], []
    inventory = VisualInventory(api_key='never-save-me', audit=events.append,
        client=client_with_answers({'identity_0': 'red'}, calls))
    for _ in range(2):
        inventory.resolve(capture, [detection], instruction='move red', entities={'red': {'kind': 'cube'}},
                          purpose='associate_inventory')
    assert len(events) == 1
    event = events[0]
    assert event['request'] == calls[0]
    assert event['request_hash'] == hashlib.sha256(json.dumps(calls[0], sort_keys=True).encode()).hexdigest()
    assert event['observation_id'] == 'frame-1' and event['http_status'] == 200
    assert event['elapsed_s'] >= 0 and event['error'] is None
    assert event['purpose'] == 'associate_inventory'
    assert event['response']['answers'][0]['choice'] == 'red'
    assert 'never-save-me' not in json.dumps(event)


def test_valid_large_answer_batch_is_audited_in_full():
    capture, detection = fixture()
    events, returned = [], []
    def handle(request):
        payload = json.loads(request.content)
        answers = []
        for question in payload['questions']:
            name = question['name']
            if question['type'] == 'predicate':
                answers.append({'type': 'predicate', 'name': name, 'probability': .99})
                continue
            selected = 'red' if name.startswith('identity_') else 'unknown'
            probabilities = [{'value': option['value'], 'probability': float(option['value'] == selected)}
                             for option in question['choices']]
            answers.append({'type': 'choice', 'name': name, 'choice': selected, 'confidence': .99,
                            'probabilities': probabilities})
        returned.extend(answers)
        return httpx.Response(200, json={'model': 'mock', 'answers': answers, 'usage': {'input_tokens': 100}})
    inventory = VisualInventory(api_key='test', audit=events.append,
        client=httpx.Client(transport=httpx.MockTransport(handle)))
    inventory.resolve(capture, [deepcopy(detection) for _ in range(40)], instruction='move red',
        entities={key: {'kind': 'cube'} for key in ['red'] + [f'other-{i}' for i in range(45)]})
    assert len(json.dumps(returned)) > 32768
    assert events[0]['response']['answers'] == returned


def test_display_region_projection_does_not_require_rgb_pixels():
    capture, detection, _ = scoped_scene()
    capture['views']['3']['instance_masks'] = {'box': detection['mask']}
    del capture['views']['3']['rgb']
    record = slot_record()
    region = {'region_id': record['region_id'], 'layout_revision': 'layout-1',
              'layout': {'support_entity_id': 'box', 'floor_z_m': 1., 'cells': [record['assigned_slot']]}}
    projected = project_placement_regions(capture, [region])
    assert projected[0]['center'] == [.5625, 14 / 24]


@pytest.mark.parametrize('failure', ['http', 'transport', 'malformed'])
def test_failed_request_audit_is_bounded_sanitized_and_cached(failure):
    capture, detection = fixture()
    events, requests = [], []
    def handle(request):
        requests.append(request)
        if failure == 'transport':
            raise httpx.ReadTimeout('never-save-me')
        if failure == 'http':
            return httpx.Response(429, json={'error': {'message': 'never-save-me', 'code': 'quota'}})
        return httpx.Response(200, json={'model': 'mock', 'answers': [None],
                                         'extra': 'never-save-me' * 10000})
    inventory = VisualInventory(api_key='never-save-me', audit=events.append,
        client=httpx.Client(transport=httpx.MockTransport(handle)))
    for _ in range(2):
        with pytest.raises(VisualInventoryError):
            inventory.resolve(capture, [detection], instruction='move red', entities={'red': {'kind': 'cube'}})
    assert len(events) == len(requests) == 1
    assert events[0]['error'] and events[0]['observation_id'] == 'frame-1'
    assert 'never-save-me' not in json.dumps(events[0])
    assert len(json.dumps(events[0]['response'])) < 20000


def test_all_entity_roles_display_their_frozen_number():
    capture, detection = fixture()
    capture['objects'] = [{'id': 'box', 'kind': 'bin', 'color': 'blue', 'display_id': 3}]
    capture['views']['3']['instance_masks'] = {'box': detection['mask']}
    labels = make_object_labels(capture, {}, {'box': 'excluded'})
    assert labels['items'][0]['display_id'] == 3
    assert labels['items'][0]['label'].startswith('#3 ')
