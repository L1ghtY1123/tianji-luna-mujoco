"""A table-confirmed workpiece cannot acquire floor pixels from a broad SAM mask."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.demo_perception import DemoPerception
from luna_mujoco.perception import PerceptionError, _estimate_object


def case(points, support_observation='current'):
    engine = DemoPerception(detector=object(), visual_inventory=object())
    engine.demo_tables = {'primary_table_id': 'table', 'source_observation_id': 'initial', 'tables': {
        'table': {'model': {'pose': [.46, 0., .41, 1., 0., 0., 0.], 'dims': [.60, .80, .30]},
                  'surface_height_m': .56}}}
    pixels = np.column_stack((np.zeros(len(points), int), np.arange(len(points))))
    entry = {'entity_id': 'item', 'kind': 'object', 'camera_id': '1', 'points': np.asarray(points),
             'pixels_rc': pixels, 'pixels': np.zeros((len(points), 3), np.uint8), 'score': .9,
             'mask': np.ones((1, len(points)), bool), 'silhouette_mask': np.ones((1, len(points)), bool)}
    entities = {'item': {'kind': 'object', 'observed_support': {'support_entity_id': 'table', 'observation_id': support_observation, 'source': 'openai_decisions', 'request_hash': 'support-proof'}}}
    return engine, entry, entities


def patch(z, x=.36, y=-.14):
    xx, yy = np.meshgrid(np.arange(x, x + .022, .002), np.arange(y, y + .022, .002))
    return np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))


def test_table_confirmed_black_keeps_current_surface_and_discards_dense_background():
    actual = patch(.62)
    points = np.vstack((actual, patch(0.), patch(.52), patch(.60, x=.9)))
    engine, entry, entities = case(points)
    original = deepcopy(entry)
    groups = engine._filter_geometry_groups([[entry]], {'1': {'frame_id': 'current'}}, entities=entities)
    np.testing.assert_array_equal(groups[0][0]['points'], actual)
    np.testing.assert_array_equal(groups[0][0]['pixels_rc'], original['pixels_rc'][:len(actual)])
    assert groups[0][0]['mask'].sum() == len(actual)
    np.testing.assert_array_equal(groups[0][0]['silhouette_mask'], original['silhouette_mask'])
    evidence = groups[0][0]['geometry_filter_evidence']
    assert evidence['source'] == 'cached_measured_table_and_current_observed_support'
    assert evidence['support_entity_id'] == 'table'
    assert evidence['outside_support_scope_points'] == 3 * len(actual)
    result = _estimate_object(groups[0], {'surface_height_m': .56, 'uncertainty_m': .006})
    assert result['geometry_filter_evidence'][0] == evidence


def test_background_only_identity_cannot_become_black_geometry():
    engine, entry, entities = case(patch(.52))
    groups = engine._filter_geometry_groups([[entry]], {'1': {'frame_id': 'current'}}, entities=entities)
    assert not groups[0][0]['mask'].any()
    with pytest.raises(PerceptionError, match='instance_visible_depth_insufficient'):
        _estimate_object(groups[0], {'surface_height_m': .56})


def test_actual_2e64_floor_proposal_is_removed_and_current_head_black_points_remain():
    path = Path(__file__).parent / 'fixtures/unpacked-background-2e64'
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
    with np.load(path.with_suffix('.npz')) as arrays:
        background, actual = arrays['background'], arrays['head_black']
    assert len(background) == 78044 and len(actual) == 518
    engine, entry, entities = case(np.vstack((actual, background)))
    engine.demo_tables = metadata['demo_tables']
    result = engine._filter_geometry_groups([[entry]], {'1': {'frame_id': 'current'}}, entities=entities)
    retained = actual[actual[:, 2] > .566]
    assert len(retained) == 486  # The other 32 raw head pixels are tabletop returns.
    np.testing.assert_array_equal(result[0][0]['points'], retained)
    assert result[0][0]['geometry_filter_evidence']['outside_support_scope_points'] == len(background)


@pytest.mark.parametrize('support_observation', [None, 'old-frame', 'other-frame'])
def test_table_scope_requires_current_support_evidence(support_observation):
    points = patch(.47, x=.2, y=-.6)
    engine, entry, entities = case(points, support_observation)
    result = engine._filter_geometry_groups([[entry]], {'1': {'frame_id': 'current'}}, entities=entities)
    np.testing.assert_array_equal(result[0][0]['points'], points)
    assert 'geometry_filter_evidence' not in result[0][0]


async def test_verified_processing_records_support_only_for_its_actual_capture(tmp_path):
    from policy_processing_fixture import runner_and_context
    from processing_fixtures import work_area

    runner, context, _ = runner_and_context(tmp_path)
    runner.state['facts']['observation_pose_ready'] = True
    runner._processing_support = lambda ctx: {**work_area(), 'table_id': 'measured-table'}
    result = await runner._step('verify_unpack', context)
    assert result['success']
    support = runner.task_progress.scene_entities[result['output_entity_id']]['observed_support']
    assert support == {'support_entity_id': 'measured-table', 'source': 'openai_decisions',
        'observation_id': context['post_unpack_capture']['observation_id'],
        'request_hash': result['request_hash']}
