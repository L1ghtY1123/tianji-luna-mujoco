from copy import deepcopy
from xml.etree import ElementTree

import numpy as np

from jev_mujoco.observation_labels import annotated_rgb_svg, make_object_labels


def observation():
    mask = np.zeros((80, 100), bool)
    mask[35:55, 40:60] = True
    return {'observation_id': 'frame1', 'generation': 2, 'objects': [
        {'id': 'target', 'color': 'green', 'kind': 'cube', 'display_id': 1},
        {'id': 'obstacle', 'color': 'red', 'kind': 'cube'},
        {'id': 'bin', 'color': 'blue', 'kind': 'bin'}],
        'views': {camera: {'rgb': np.zeros((80, 100, 3), np.uint8),
            'instance_masks': {key: mask.copy() for key in ['target', 'obstacle', 'bin']}}
            for camera in ['1', '2', '3']}}


def test_all_cameras_include_persistent_ids_names_membership_and_task_status():
    obs = observation()
    result = make_object_labels(obs, {'target': {'status': 'completed', 'display_id': 1}},
                                {'target': 'required', 'obstacle': 'excluded'}, destination_id='bin')
    assert len(result['items']) == 9
    assert {item['camera_id'] for item in result['items']} == {'1', '2', '3'}
    for item in result['items']:
        assert item['id'] in item['label']
        if item['id'] == 'target':
            assert item['name'] == '绿色方块'
            assert item['status'] == 'completed' and '已搬运' in item['label']
        if item['id'] == 'obstacle':
            assert item['role'] == 'obstacle' and '障碍物' in item['label']
            assert item['display_id'] is None
        if item['id'] == 'bin':
            assert item['role'] == 'destination'


def test_no_mask_means_no_overlay_and_annotations_do_not_mutate_rgb():
    obs = observation()
    del obs['views']['1']['instance_masks']['target']
    original = deepcopy(obs['views']['1']['rgb'])
    labels = make_object_labels(obs, {}, {'target': 'unknown'})
    assert not any(item['id'] == 'target' and item['camera_id'] == '1' for item in labels['items'])
    image = annotated_rgb_svg(obs['views']['1']['rgb'], labels, '1')
    ElementTree.fromstring(image)
    assert 'ID target' not in image
    assert '待确认任务归属' in annotated_rgb_svg(obs['views']['3']['rgb'], labels, '3')
    np.testing.assert_array_equal(obs['views']['1']['rgb'], original)


def test_raw_frame_is_explicitly_unprocessed_not_labeled_from_history():
    obs = observation()
    labels = {'observation_id': 'frame1', 'generation': 2, 'analysis_status': 'unprocessed', 'items': []}
    image = annotated_rgb_svg(obs['views']['3']['rgb'], labels, '3')
    assert '未分析' in image and 'data:image/png;base64,' in image


def test_held_and_intermediate_stage_are_distinct_from_completed():
    obs = observation()
    labels = make_object_labels(obs, {'target': {'status': 'selected', 'stage_index': 1}},
        {'target': 'required'}, holding={'instance_id': 'target', 'verified': True})
    assert all(item['status'] == 'held' for item in labels['items'] if item['id'] == 'target')
    labels = make_object_labels(obs, {'target': {'status': 'pending', 'stage_index': 1}}, {'target': 'required'})
    assert all('中转完成' in item['label'] for item in labels['items'] if item['id'] == 'target')


def test_region_projection_uses_current_camera_but_retains_frozen_slot_ids_and_source():
    obs = observation()
    for view in obs['views'].values():
        view.update(T_world_camera=np.eye(4), intrinsics={'fx': 100., 'fy': 100., 'cx': 50., 'cy': 40.})
    layout = {'support_entity_id': 'bin', 'floor_z_m': 1., 'source': 'measured_floor',
              'observation_id': 'initial-region-frame', 'cells': [
                  {'slot_id': 2, 'center_world_xy_m': [.1, .1],
                   'polygon_world_xy_m': [[0., 0.], [.2, 0.], [.2, .2], [0., .2]]}]}
    regions = [{'region_id': 'region:bin', 'layout_revision': 'revision', 'layout': layout}]
    labels = make_object_labels(obs, {}, {}, placement_regions=regions)
    cell = next(item for item in labels['regions'] if item['camera_id'] == '3')
    assert cell['region_id'] == 'region:bin' and cell['slot_id'] == 2
    assert cell['source_observation_id'] == 'initial-region-frame'
    np.testing.assert_allclose(cell['center'], [.6, .625])
    del obs['views']['3']['instance_masks']['bin']
    hidden = make_object_labels(obs, {}, {}, placement_regions=regions)
    assert not any(item['camera_id'] == '3' for item in hidden['regions'])
