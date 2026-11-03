from copy import deepcopy
from xml.etree import ElementTree

import numpy as np

from luna_mujoco.observation_labels import annotated_rgb_svg, make_object_labels


def test_identity_near_box_badges_keep_canvas_and_leave_between_subjects_clear():
    from luna_mujoco.observation_labels import raster_visual_annotations
    rgb = np.full((480, 640, 3), 190, dtype=np.uint8)
    rgb[160:360, :130] = [20, 40, 200]
    rgb[170:360, 530:] = [20, 40, 200]
    items = [
        {'id': 'left', 'display_label': 'Reference 5', 'image_label': 'R5',
         'box': [0., 160/480, 130/640, 360/480]},
        {'id': 'right', 'display_label': 'Reference 3', 'image_label': 'R3',
         'box': [530/640, 170/480, 1., 360/480]},
    ]
    original, original_items = rgb.copy(), deepcopy(items)
    result = raster_visual_annotations(rgb, items, frame_color=(235, 235, 235), label_layout='near_box')
    assert result.shape == rgb.shape
    np.testing.assert_array_equal(rgb, original)
    assert items == original_items
    np.testing.assert_array_equal(result[190:350, 170:500], rgb[190:350, 170:500])
    # Each black badge is adjacent to its own upper edge, not a right-side legend.
    assert np.any(np.all(result[130:160, :35] == [16, 16, 16], axis=2))
    assert np.any(np.all(result[140:170, 530:570] == [16, 16, 16], axis=2))
    np.testing.assert_array_equal(result[220:300, 40:90], rgb[220:300, 40:90])
    default = raster_visual_annotations(rgb, items)
    assert default.shape[1] > rgb.shape[1]
    np.testing.assert_array_equal(default, raster_visual_annotations(rgb, items, label_layout='margin'))


def test_identity_near_box_aliases_retain_every_original_outline():
    from luna_mujoco.observation_labels import group_candidate_boxes, raster_visual_annotations
    rgb = np.full((100, 160, 3), 80, dtype=np.uint8)
    items = [{'id': 'candidate-1', 'image_label': 'C1', 'box': [.2, .3, .4, .7]},
             {'id': 'candidate-2', 'image_label': 'C2', 'box': [.205, .3, .405, .7]}]
    assert group_candidate_boxes(items)[0]['image_label'] == 'C1 / C2'
    result = raster_visual_annotations(rgb, items, label_layout='near_box')
    assert result.shape == rgb.shape
    # Both rectangles remain at their measured bounds; aliasing only shares text.
    assert np.all(result[50, 32] == [235, 235, 235])
    assert np.all(result[50, 64] == [235, 235, 235])


def test_capture_overlay_does_not_paint_later_holding_or_completion_onto_old_rgb():
    obs = observation()
    obs['objects'][0]['identity_source'] = 'openai_decisions'
    first = make_object_labels(obs, {'target': {'status': 'observed'}}, {'target': 'required'})
    later = make_object_labels(obs, {'target': {'status': 'held'}}, {'target': 'required'},
                               holding={'instance_id': 'target', 'verified': True})
    assert first == later
    assert '夹持中' not in annotated_rgb_svg(obs['views']['3']['rgb'], later, '3')


def test_unknown_color_does_not_mean_unknown_identity_or_claim_collision_publication():
    obs = observation()
    obs['objects'][0].update(color='unknown', kind='object', identity_source='openai_decisions')
    obs['scene_entities'] = {'target': {'color': 'pink', 'kind': 'object'}}
    labels = make_object_labels(obs, {}, {'target': 'required'})
    target = next(item for item in labels['items'] if item['id'] == 'target')
    assert target['name'] == '粉色物体'
    assert target['name_source'] == 'inventory_description'
    assert target['identity_source'] == 'openai_decisions'
    assert target['box_source'] == 'associated_instance_mask'
    assert 'unknown' not in target['label']
    assert all('碰撞障碍' not in item['label'] for item in labels['items'])
    assert labels['overlay_scope'] == 'capture_identity_geometry'


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


def test_all_cameras_include_persistent_ids_names_and_roles_without_later_task_status():
    obs = observation()
    result = make_object_labels(obs, {'target': {'status': 'completed', 'display_id': 1}},
                                {'target': 'required', 'obstacle': 'excluded'}, destination_id='bin')
    assert len(result['items']) == 9
    assert {item['camera_id'] for item in result['items']} == {'1', '2', '3'}
    for item in result['items']:
        assert item['id'] in item['label']
        if item['id'] == 'target':
            assert item['name'] == '绿色方块'
            assert item['status'] == 'unverified' and '已搬运' not in item['label']
        if item['id'] == 'obstacle':
            assert item['role'] == 'obstacle' and '碰撞' not in item['label']
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
    assert '身份依据未记录' in annotated_rgb_svg(obs['views']['3']['rgb'], labels, '3')
    np.testing.assert_array_equal(obs['views']['1']['rgb'], original)


def test_raw_frame_is_explicitly_unprocessed_not_labeled_from_history():
    obs = observation()
    labels = {'observation_id': 'frame1', 'generation': 2, 'analysis_status': 'unprocessed', 'items': []}
    image = annotated_rgb_svg(obs['views']['3']['rgb'], labels, '3')
    assert '未分析' in image and 'data:image/png;base64,' in image


def test_later_transfer_ledger_does_not_relabel_the_capture():
    obs = observation()
    labels = make_object_labels(obs, {'target': {'status': 'selected', 'transfer_index': 1}},
        {'target': 'required'}, holding={'instance_id': 'target', 'verified': True})
    original = deepcopy(labels)
    labels = make_object_labels(obs, {'target': {'status': 'observed', 'transfer_index': 0,
        'transfers': [{'destination_id': 'bench', 'transfer_index': 0, 'observation_id': 'earlier'}]}},
        {'target': 'required'})
    targets = [item for item in labels['items'] if item['id'] == 'target']
    assert targets and all('已观测' not in item['label'] for item in targets)
    assert labels == original


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
