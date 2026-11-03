from copy import deepcopy

import numpy as np
import pytest
from test_semantic_transport import payload

from luna_mujoco.perception_mapping import decode_mapping_payload
from luna_mujoco.semantic_builder import _requests
from luna_mujoco.semantic_transport import decode_project_semantic_request, encode_project_semantic_request


def head_payload():
    original = payload()
    original['confirmed_instance_id'] = ''
    original['views'] = {'3': original['views']['3']}
    original['views']['3']['instance_masks'] = {'source': np.ones((2, 3), np.uint8),
                                               'obstacle': np.array([[0, 1, 0], [0, 1, 0]], np.uint8)}
    return original


def test_targetless_head_semantics_roundtrip_and_builder_preserve_all_instances():
    original = head_payload()
    wire = encode_project_semantic_request(original, exclusion_state='observation')
    restored = decode_project_semantic_request(wire, exclusion_state='observation')
    assert restored['confirmed_instance_id'] == ''
    assert tuple(restored['views']) == ('3',)
    for name, mask in original['views']['3']['instance_masks'].items():
        np.testing.assert_array_equal(restored['views']['3']['instance_masks'][name], mask)
    decoded, identity = decode_mapping_payload(wire | {'exclusion_state': 'observation',
        'gripper_joint_positions_rad': {'A': 0., 'B': 0.}})
    assert identity['disabled_obstacle_names'] == ()
    assert identity['target_id'] == ''
    assert len(_requests(decoded)) == 1
    assert decoded['scene_policy'] == 'head_all_instances'


@pytest.mark.parametrize('change', ['state', 'selected', 'extra_camera'])
def test_targetless_transport_cannot_bypass_selected_identity(change):
    wire = encode_project_semantic_request(head_payload(), exclusion_state='observation')
    state = 'observation'
    if change == 'state':
        state = 'selected'
    elif change == 'selected':
        wire['semantic_identity']['confirmed_instance_id'] = 'fabricated-target'
    else:
        wire['semantic_masks']['1'] = deepcopy(wire['semantic_masks']['3'])
    with pytest.raises(ValueError):
        decode_project_semantic_request(wire, exclusion_state=state)


def test_head_to_three_camera_rebuilds_entities_but_fits_table_only_once():
    from test_semantic_builder import builder, request

    value, segmenter, mapper, fits = builder()
    first = request('observation')
    first['confirmed_instance_id'] = ''
    first['views']['3']['instance_masks'] = deepcopy(first['views']['1']['instance_masks'])
    first['views'] = {'3': first['views']['3']}
    first['scene_policy'] = 'head_all_instances'
    initial = value.build(first)
    initial_revision = value.table_cache.diagnostics['table_revision']
    assert len(segmenter.requests) == 1
    assert all(view.background_point_count == 0 for view in mapper.integrated)
    assert set(initial.instance_metrics) == {'source', 'other'}
    for index, capture in enumerate((request('selected'), first, request('selected'))):
        capture = deepcopy(capture)
        capture['observation_set_id'] = f'next-{index}'
        capture.setdefault('scene_policy', 'selected_instances')
        result = value.build(capture)
        assert result.table.model == initial.table.model
        assert value.table_cache.diagnostics['cache_hit']
        assert value.table_cache.diagnostics['table_revision'] == initial_revision
    assert len(fits) == 1


@pytest.mark.parametrize('state,policy', [('selected', 'selected_instances'), ('observation', 'all_instances')])
def test_first_three_camera_scene_initializes_table_from_same_capture_head_only(state, policy):
    from test_semantic_builder import builder, request

    value, segmenter, mapper, fits = builder()
    capture = request(state) | {'scene_policy': policy}
    if state == 'observation':
        capture['confirmed_instance_id'] = ''
    result = value.build(capture)
    assert [item.camera_id for item in segmenter.requests] == ['1', '2', '3']
    assert [item.camera_id for item in mapper.integrated] == ['1', '2', '3']
    assert set(result.instance_metrics) == {'source', 'other'}
    assert len(fits) == 1
    groups, spheres = fits[0]
    assert len(groups) == 1 and len(groups[0]) == 8
    assert groups[0][:, 0].min() == 3  # Head transform, no wrist background.
    assert spheres.shape == (6, 4)
    assert value.table_cache.diagnostics['source_observation_id'] == capture['observation_set_id']
    assert value.table_cache.diagnostics['cache_hit'] is False


def test_three_camera_update_keeps_calibration_change_guard():
    from test_semantic_builder import builder, request

    value, _, _, fits = builder()
    capture = request('observation') | {'scene_policy': 'all_instances', 'confirmed_instance_id': ''}
    value.build(capture)
    changed = deepcopy(capture)
    changed['views']['3']['T_base_left_camera'][0][3] += .1
    with pytest.raises(ValueError, match='initial_head_table_required'):
        value.build(changed)
    assert len(fits) == 1


def test_first_three_camera_scene_does_not_substitute_wrists_for_missing_head_support():
    from test_semantic_builder import builder, request
    from tianji_curobo.mixed_esdf_scene import build_table_completion

    value, _, mapper, _ = builder()
    value.table_cache.table_builder = build_table_completion
    capture = request('observation') | {'scene_policy': 'all_instances', 'confirmed_instance_id': ''}
    capture['views']['3']['valid_left_mask'][:] = 0
    with pytest.raises(ValueError, match='tabletop_not_found: too few finite background points'):
        value.build(capture)
    assert value.table_cache.table is None
    assert mapper.integrated is None


def test_targetless_three_camera_update_keeps_every_instance_and_existing_table():
    original = payload()
    original['confirmed_instance_id'] = ''
    wire = encode_project_semantic_request(original, exclusion_state='observation')
    decoded, identity = decode_mapping_payload(wire | {'exclusion_state': 'observation',
        'gripper_joint_positions_rad': {'A': 0., 'B': 0.}})
    assert tuple(decoded['views']) == ('1', '2', '3')
    assert decoded['scene_policy'] == 'all_instances'
    assert identity['disabled_obstacle_names'] == ()
    for camera, view in original['views'].items():
        for identifier, mask in view['instance_masks'].items():
            np.testing.assert_array_equal(decoded['views'][camera]['instance_masks'][identifier], mask)

    from test_semantic_builder import builder, request
    value, _, _, fits = builder()
    first = request('observation')
    first['confirmed_instance_id'] = ''
    first['views'] = {'3': first['views']['3']}
    initial = value.build(first | {'scene_policy': 'head_all_instances'})
    update = value.build(request('observation') | {'scene_policy': 'all_instances', 'confirmed_instance_id': ''})
    assert update.table.model == initial.table.model
    assert value.table_cache.diagnostics['cache_hit']
    assert len(fits) == 1
