"""OBB display keeps capture evidence separate from incremental collision maps."""
from copy import deepcopy

import numpy as np
import pytest
from test_object_obb import box_observation

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.object_obb import observed_object_obb
from luna_mujoco.pointcloud_preview import observed_obb_preview, reconcile_obb_preview


def observation(identifier='seen-red', capture='capture-4', offset=0.):
    value = box_observation()
    value['observation_id'] = capture
    obj = value['objects'][0]
    obj['id'] = identifier
    obj['points_world_m'][:, 0] += offset
    obj['observation_provenance'] = {'observation_id': capture, 'generation': 1}
    obj['obb'] = observed_object_obb(obj)
    return value


def scene(capture, records, generation=1):
    return {'observation_id': capture, 'generation': generation,
            'inventory_map': {'entities': records}}


def test_export_uses_exact_grasp_obb_and_keeps_points_private_and_unchanged():
    source = observation()
    points = source['objects'][0]['points_world_m'].copy()
    rows = observed_obb_preview(source)
    assert rows[0]['obb'] == source['objects'][0]['obb']
    assert rows[0]['instance_id'] == 'seen-red'
    assert rows[0]['status'] == 'current'
    assert 'points_world_m' not in rows[0]
    np.testing.assert_array_equal(source['objects'][0]['points_world_m'], points)
    rows[0]['obb']['half_extents_m'][0] = 100
    assert source['objects'][0]['obb']['half_extents_m'][0] < 1


def test_export_rejects_changed_measurement_or_capture_binding():
    source = observation()
    source['objects'][0]['points_world_m'][0, 0] += .001
    with pytest.raises(ValueError, match='binding_changed'):
        observed_obb_preview(source)
    source = observation()
    source['observation_id'] = 'other'
    with pytest.raises(ValueError, match='capture'):
        observed_obb_preview(source)


def test_local_map_keeps_unselected_obb_original_capture_even_when_seen_again():
    previous = observed_obb_preview(observation())
    current = observed_obb_preview(observation(capture='capture-5', offset=.1))
    current += observed_obb_preview(observation('second', 'capture-5', .2))
    identity = scene('capture-5', {
        'seen-red': {'status': 'retained', 'last_observation_id': 'capture-4'},
        'second': {'status': 'observed', 'last_observation_id': 'capture-5'}})
    before = deepcopy((previous, current, identity))
    rows = reconcile_obb_preview(previous, current, identity)
    assert rows[0]['status'] == 'retained'
    assert rows[0]['obb'] == previous[0]['obb']
    assert rows[0]['observation_id'] == 'capture-4'
    assert rows[1]['status'] == 'current'
    assert (previous, current, identity) == before


def test_global_refresh_replaces_box_without_fusing_old_pose_and_drops_retired_ids():
    previous = observed_obb_preview(observation())
    previous += observed_obb_preview(observation('retired'))
    current = observed_obb_preview(observation(capture='capture-5', offset=.1))
    identity = scene('capture-5', {
        'seen-red': {'status': 'observed', 'last_observation_id': 'capture-5'}})
    rows = reconcile_obb_preview(previous, current, identity)
    assert len(rows) == 1
    assert rows[0]['obb'] == current[0]['obb']
    assert rows[0]['status'] == 'current'


def test_stale_or_missing_export_never_becomes_current_and_generation_does_not_leak():
    previous = observed_obb_preview(observation())
    stale = scene('capture-5', {'seen-red': {'status': 'stale', 'last_observation_id': 'capture-4'}})
    assert reconcile_obb_preview(previous, [], stale)[0]['status'] == 'stale'
    fresh = scene('capture-5', {'seen-red': {'status': 'observed', 'last_observation_id': 'capture-5'}})
    assert reconcile_obb_preview(previous, [], fresh) == []
    stale['generation'] = 2
    assert reconcile_obb_preview(previous, [], stale) == []


def test_build_payload_transports_diagnostics_outside_sdk_geometry_contract():
    from luna_mujoco.perception_mapping import decode_mapping_payload

    calls = []
    source = observation()
    Algorithms(curobo_request=lambda path, body: calls.append(body) or {'ok': True}).build_scene(source, 'seen-red')
    assert calls[0]['object_obb_observations'] == observed_obb_preview(source)
    decoded, identity = decode_mapping_payload(calls[0])
    assert 'object_obb_observations' not in decoded
    plain = deepcopy(calls[0])
    plain.pop('object_obb_observations')
    assert decode_mapping_payload(plain)[1]['scene_token'] == identity['scene_token']


def test_action_update_marks_old_obb_stale_without_predicting_new_measured_pose():
    from luna_mujoco.scene_update import updated_preview

    rows = observed_obb_preview(observation())
    preview = {'object_obbs': {'observed': rows, 'mapped': rows}}
    identity = scene('capture-4', {'seen-red': {'status': 'stale', 'last_observation_id': 'capture-4'}})
    identity.update(scene_token='changed', disabled_obstacle_names=[])
    after = updated_preview(preview, identity, {'operation': 'transfer', 'target_id': 'seen-red'})
    assert after['object_obbs']['mapped'][0]['status'] == 'stale'
    assert after['object_obbs']['observed'][0]['status'] == 'stale'
    assert after['object_obbs']['mapped'][0]['obb'] == rows[0]['obb']
    assert preview['object_obbs']['observed'][0]['status'] == 'current'
