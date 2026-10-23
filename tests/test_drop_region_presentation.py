"""Drop detail images locate the measured bin without asserting delivery."""
import base64
import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from test_demo_lifecycle_perception import client, drop_inputs, record
from test_experiments import begin

from luna_mujoco.demo_lifecycle_perception import verify_right_bin_drop
from luna_mujoco.experiments import ExperimentStore
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import image_part


def measured_drop_inputs():
    capture, action, reference = drop_inputs()
    for frame, x in ((reference, .2), (capture, .3)):
        frame['views']['1'].update(rgb=np.full((200, 240, 3), 140, np.uint8),
            T_world_camera=[[1, 0, 0, x], [0, -1, 0, -.6], [0, 0, -1, 1.5], [0, 0, 0, 1]],
            intrinsics={'fx': 180., 'fy': 180., 'cx': 120., 'cy': 100.}, calibration_id='right-camera')
    reference['drop_bin_region'] = {'source': 'published_inventory_container_occupancy_bounds',
        'entity_id': 'right-bin', 'observation_id': 'before', 'generation': 3,
        'minimum_world_m': [.05, -.8, .4], 'maximum_world_m': [.35, -.4, .6],
        'identity_source': 'measured_container_correspondence', 'scene_token': 'native-scene',
        'layer_name': 'semantic_instance::right-bin',
        'bounds_source': 'all_native_surface_cells_before_preview_sampling',
        'cell_count': 180, 'voxel_size_m': .02}
    action['release_tcp'] = {'FORBIDDEN_COMMAND_COORDINATES': [90, 91, 92]}
    return capture, action, reference


def request(capture, action, reference, answers=None):
    visual, calls = client(answers)
    result = verify_right_bin_drop(visual, capture, target_id='pink', lifecycle=record('right_bin_released'),
        action_evidence=action, reference=reference, owner_id='run')
    return result, calls[0]


def zooms(payload):
    return [json.loads(p['text']) for p in payload['input'][0]['content']
            if p['type'] == 'input_text' and '"image_role": "drop_bin_region_zoom"' in p['text']]


def test_paired_measured_bin_zooms_preserve_six_raw_images_and_observable_subject():
    capture, action, reference = measured_drop_inputs()
    before = deepcopy(reference)
    result, payload = request(capture, action, reference)
    images = [p for p in payload['input'][0]['content'] if p['type'] == 'input_image']
    originals = [image_part(frame['views'][camera]['rgb']) for frame in (capture, reference)
                 for camera in ('2', '1', '3')]
    assert images[:6] == originals and len(images) == 10
    details = zooms(payload)
    assert [z['frame_role'] for z in details] == ['BEFORE', 'CURRENT', 'BEFORE', 'CURRENT']
    assert all(z['label'] == 'BIN REGION' and z['source_camera_id'] in {'1', '3'} for z in details)
    assert details[0]['crop_bounds_xyxy_px'] != details[1]['crop_bounds_xyxy_px']
    for z, frame in zip(details, (reference, capture, reference, capture), strict=True):
        assert z['observation_id'] == frame['observation_id']
        assert z['measured_region']['observation_id'] == reference['observation_id']
        assert z['source_image_sha256'] == hashlib.sha256(base64.b64decode(
            image_part(frame['views'][z['source_camera_id']]['rgb'])['image_url'].split(',', 1)[1])).hexdigest()
        assert 'not proof' in z['meaning'] and z['resampling'].startswith('nearest')
        assert z['label_margin_px'] == 28
    assert 'additional black workpiece' in payload['questions'][0]['instructions']
    assert 'FORBIDDEN_COMMAND_COORDINATES' not in json.dumps(payload)
    assert result['visual_evidence']['bin_region_zooms'] == details
    assert reference['drop_bin_region'] == before['drop_bin_region']
    for camera in reference['views']:
        np.testing.assert_array_equal(reference['views'][camera]['rgb'], before['views'][camera]['rgb'])


@pytest.mark.parametrize('name,value', [('current_item_added', ('yes', .77)), ('current_item_added', ('unknown', .99)),
    ('current_item_added', ('no', .99))])
def test_new_presentation_never_promotes_original_uncertain_or_old_contents_answers(name, value):
    result, _ = request(*measured_drop_inputs(), answers={name: value})
    assert result['success'] is False
    answer = next(a for a in result['visual_evidence']['answers'] if a['name'] == name)
    expected = value[1] if value[0] == 'yes' else 1-value[1] if value[0] == 'no' else .5
    assert answer['type'] == 'predicate' and answer['probability'] == expected


@pytest.mark.parametrize('field,value', [('entity_id', 'source-bin'), ('observation_id', 'other-frame'),
    ('generation', 4), ('minimum_world_m', [float('nan'), 0, 0]), ('source', 'commanded_pose'),
    ('source', 'measured_inventory_container_bounds'), ('scene_token', ''),
    ('layer_name', 'semantic_instance::other'), ('bounds_source', 'preview_points'),
    ('cell_count', 0), ('cell_count', True), ('voxel_size_m', float('nan'))])
def test_invalid_region_binding_stops_before_model(field, value):
    capture, action, reference = measured_drop_inputs()
    reference['drop_bin_region'][field] = value
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='drop_bin_region_invalid'):
        verify_right_bin_drop(visual, capture, target_id='pink', lifecycle=record('right_bin_released'),
            action_evidence=action, reference=reference, owner_id='run')
    assert not calls


def test_out_of_view_region_is_optional_not_a_new_visibility_gate():
    capture, action, reference = measured_drop_inputs()
    for frame in (reference, capture):
        for camera in ('1', '3'):
            frame['views'][camera]['T_world_camera'][0][3] = 50.
    result, payload = request(capture, action, reference)
    assert result['success']
    assert not zooms(payload)
    assert sum(p['type'] == 'input_image' for p in payload['input'][0]['content']) == 6


def test_actual_archive_retains_bin_zooms_and_original_image_hashes(tmp_path):
    _, payload = request(*measured_drop_inputs())
    store = ExperimentStore(tmp_path)
    session = begin(store)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = session.visual_decision({'request': payload, 'response': {'answers': []},
        'request_hash': key, 'observation_id': 'after', 'http_status': 200})
    session.finish({'status': 'needs_attention'})
    saved = json.loads(store.file(session.id, path).read_text(encoding='utf8'))
    images = [p['image_url'] for p in saved['request']['input'][0]['content'] if p['type'] == 'input_image']
    assert len(images) == 10 and len(zooms(saved['request'])) == 4
    for image in images:
        assert hashlib.sha256(store.file(session.id, image['path']).read_bytes()).hexdigest() == image['sha256']


def shell_drop_inputs():
    """Use the actual published-bound producer; five upright walls, not a filled bin."""
    from luna_mujoco.collision_mapping import collision_layer_bounds
    from luna_mujoco.demo_unpack_workflow import right_bin_occupied_bounds

    capture, action, reference = measured_drop_inputs()
    region = reference['drop_bin_region']
    cuboids = {
        'x_min': {'pose': [.056, -.6, .5, 1, 0, 0, 0], 'dims': [.012, .4, .2]},
        'x_max': {'pose': [.344, -.6, .5, 1, 0, 0, 0], 'dims': [.012, .4, .2]},
        'y_min': {'pose': [.2, -.794, .5, 1, 0, 0, 0], 'dims': [.3, .012, .2]},
        'y_max': {'pose': [.2, -.406, .5, 1, 0, 0, 0], 'dims': [.3, .012, .2]},
        'floor': {'pose': [.2, -.6, .406, 1, 0, 0, 0], 'dims': [.3, .4, .012]},
    }
    scene = {key: region[key] for key in ('scene_token', 'observation_id', 'generation')}
    scene['mapping_diagnostics'] = {'bin_shell_completions': {'right-bin': {
        'source': 'measured_planes_and_user_bin_dimensions', 'instance_id': 'right-bin',
        'observation_id': 'before', 'generation': 3, 'synthetic_geometry_is_measured': False,
        'cuboids': cuboids}}}
    model = {'cuboid': {'semantic_instance::right-bin::bin_shell::'+face: box
                       for face, box in cuboids.items()}}
    header = collision_layer_bounds([], model, scene)
    header['layers'][region['layer_name']] = {
        'kind': 'voxel', 'enabled': True, 'source': 'native_surface_voxel_cell_bounds',
        **{key: region[key] for key in ('bounds_source', 'cell_count', 'voxel_size_m',
                                      'minimum_world_m', 'maximum_world_m')}}
    scene['collision_layer_bounds'] = header
    reference['drop_bin_region'] = right_bin_occupied_bounds(
        {'id': 'right-bin', 'kind': 'bin'}, scene, reference)
    return capture, action, reference


def test_published_native_shell_union_reaches_drop_images_without_losing_provenance():
    capture, action, reference = shell_drop_inputs()
    before = deepcopy(reference['drop_bin_region'])
    result, payload = request(capture, action, reference)
    assert result['success']
    assert len(zooms(payload)) == 4
    assert sum(p['type'] == 'input_image' for p in payload['input'][0]['content']) == 10
    for detail in zooms(payload):
        assert detail['measured_region'] == {k: v for k, v in before.items()
                                             if k not in {'derived_shell_components', 'layer_names'}}
    assert result['visual_evidence']['bin_region_geometry'] == before
    assert reference['drop_bin_region'] == before


@pytest.mark.parametrize('damage', ['missing_face', 'extra_face', 'wrong_id', 'stale_observation',
    'stale_generation', 'disabled', 'measured_claim', 'unknown_source', 'malformed_bounds',
    'outside_union', 'missing_layer_names', 'duplicate_layer', 'downgraded_native', 'empty_token'])
def test_invalid_shell_sidecar_stops_before_model(damage):
    capture, action, reference = shell_drop_inputs()
    region = reference['drop_bin_region']
    components = region['derived_shell_components']
    name = 'semantic_instance::right-bin::bin_shell::floor'
    layer = components[name]
    if damage == 'missing_face':
        components.pop(name)
    elif damage == 'extra_face':
        components[name+'-extra'] = deepcopy(layer)
    elif damage == 'wrong_id':
        layer['instance_id'] = 'source-bin'
    elif damage == 'stale_observation':
        layer['observation_id'] = 'old'
    elif damage == 'stale_generation':
        layer['generation'] = 2
    elif damage == 'disabled':
        layer['enabled'] = False
    elif damage == 'measured_claim':
        layer['synthetic_geometry_is_measured'] = True
    elif damage == 'unknown_source':
        layer['geometry_source'] = 'raw_mask_aabb'
    elif damage == 'malformed_bounds':
        layer['minimum_world_m'] = [float('nan'), 0, 0]
    elif damage == 'outside_union':
        layer['maximum_world_m'][2] = .9
    elif damage == 'missing_layer_names':
        region.pop('layer_names')
    elif damage == 'duplicate_layer':
        region['layer_names'].append(name)
    elif damage == 'downgraded_native':
        region['bounds_source'] = 'all_native_surface_cells_before_preview_sampling'
    else:
        region['scene_token'] = ''
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='drop_bin_region_invalid'):
        verify_right_bin_drop(visual, capture, target_id='pink', lifecycle=record('right_bin_released'),
            action_evidence=action, reference=reference, owner_id='run')
    assert not calls
