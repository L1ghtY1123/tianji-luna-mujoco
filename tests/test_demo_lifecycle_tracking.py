"""Role proposals use frozen current geometry and valid transitions, never identity scores."""
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.cameras import deproject_view
from luna_mujoco.demo_container_tracking import container_correspondence, reference_with_container_tracks
from luna_mujoco.demo_lifecycle_perception import lifecycle_entities


def frozen_case(*, current_contract=True):
    directory = Path(__file__).parent / 'fixtures'
    path = directory / 'container_correspondence_2e64'
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
        def restore(value):
            if isinstance(value, dict) and 'array_key' in value:
                return arrays[value['array_key']].copy()
            if isinstance(value, dict):
                return {key: restore(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restore(item) for item in value]
            return value
        data = restore(json.loads(path.with_suffix('.json').read_text(encoding='utf8')))
    metadata = json.loads((directory / 'lifecycle_roles_2e64.json').read_text(encoding='utf8'))
    reference = reference_with_container_tracks(data['observation'], data['initial_capture'])
    reference['owner_id'] = metadata['owner_id']
    reference['demo_task']['item_lifecycle'] = metadata['item_lifecycle']
    capture, detections = data['current_capture'], data['detections']
    if current_contract:
        # Synthetic successful six-question replies test current consumers with
        # frozen geometry. The historical two-question fixture is not modified.
        from test_demo_lifecycle_perception import client

        from luna_mujoco.demo_lifecycle_perception import table_release_reference, verify_unpacking

        for identifier, record in reference['demo_task']['item_lifecycle'].items():
            if record['phase'] != 'unpacked_confirmed':
                continue
            action = record['mock_action']
            record['table_visual_evidence'].update(observation_id='synthetic-test-release', source='openai_decisions',
                identity_source='execution_chain', status='at_destination', confidence=.99)
            # This geometry fixture intentionally omits RGB. Use explicit test
            # pixels for the HTTP double, never claim a new historical judgment.
            synthetic_capture = deepcopy(capture)
            for view in synthetic_capture['views'].values():
                view['rgb'] = np.zeros((*np.asarray(view['depth_m']).shape, 3), dtype=np.uint8)
            before = deepcopy(synthetic_capture)
            before.update(observation_id=record['table_visual_evidence']['observation_id'],
                          time=action['completed_time']-.1)
            if action.get('completed_perf_counter_ns') is not None:
                before['captured_perf_counter_ns'] = action['completed_perf_counter_ns']-1
            pending = deepcopy(record)
            pending.update(phase='unpack_pending', appearance='wrapped', appearance_revision=record['appearance_revision']-1)
            visual, _ = client()
            floor = metadata['table_scope']['surface_height_m']
            record['unpacking_evidence'] = verify_unpacking(visual, synthetic_capture, target_id=identifier,
                lifecycle=pending, action_evidence=action, owner_id=reference['owner_id'],
                active_target_ids=[identifier], table_release_capture=before,
                table_release_reference=table_release_reference(before, record['table_visual_evidence'], reference['owner_id']),
                work_area={'center_world_xy_m': [.4, 0.], 'region_polygon_world_xy_m': [[.3, -.1], [.5, -.1], [.5, .1], [.3, .1]],
                           'floor_z_m': floor, 'source': 'configured_xy_measured_support'})
            # This historical same-ID fixture has no retained product RGB reference.
            # Exercise only the original strict same-frame compatibility path.
            record['unpacking_evidence'].pop('verification_capture')
            record['unpacking_evidence']['visual_evidence'].pop('verification_capture')
    for detection in detections:
        detection['points'] = deproject_view(capture['views'][detection['camera_id']], detection['mask'])
        detection['pixels_rc'] = np.column_stack(np.nonzero(detection['mask']))
    entities = lifecycle_entities(reference['entities'], reference['demo_task'])
    tracking = container_correspondence(capture, detections, reference, entities)
    return capture, detections, reference, entities, metadata['table_scope'], tracking


def propose(case, **kwargs):
    module = importlib.import_module('luna_mujoco.demo_lifecycle_tracking')
    return module.lifecycle_candidate_scopes(*case, **kwargs)


def test_historical_same_item_answers_do_not_satisfy_current_visible_result_contract():
    assert propose(frozen_case(current_contract=False)) is None


@pytest.mark.parametrize('change', ['low_result', 'explicit_conflict', 'qualified_forged',
    'identity_source', 'visibility_source', 'active_ids', 'before_binding'])
def test_current_transition_rechecks_raw_views_and_chain_instead_of_trusting_status(change):
    case = frozen_case()
    record = next(row for row in case[2]['demo_task']['item_lifecycle'].values()
                  if row['phase'] == 'unpacked_confirmed')
    visual = record['unpacking_evidence']['visual_evidence']
    if change == 'low_result':
        visual['answers'][0]['probability'] = .79
    elif change == 'explicit_conflict':
        visual['answers'][0]['probability'] = .01
    elif change == 'qualified_forged':
        visual['view_scope'] = 'forged_single_camera'
    elif change in {'identity_source', 'visibility_source'}:
        visual[change] = 'appearance_only'
    elif change == 'active_ids':
        visual['execution_chain']['active_target_ids'].append('another')
    else:
        visual['table_release_reference']['binding_id'] = 'wrong'
    assert propose(case) is None


def test_actual_candidates_keep_indices_and_use_interior_without_changing_raw_geometry():
    case = frozen_case()
    original = deepcopy(case[1])
    result = propose(case)
    scopes = result['candidate_scopes']
    # Historical upright detections 4/5 exceed the current flat-CAD height.
    # Detection 13 still fits; raw historical geometry must not be rewritten.
    assert set(scopes) == {8, 13}
    assert scopes[8]['role'] == 'wrapped_source'
    assert scopes[8]['raw_point_count'] == 1652
    assert scopes[8]['interior_removed_point_count'] == 190
    assert scopes[8]['measured_bounds_world_m'][1][2] == pytest.approx(.51205778, abs=1e-6)
    assert scopes[13]['role'] == 'unpacked_confirmed'
    assert result['bindings'][scopes[13]['entity_id']]['height_prior_m'] == pytest.approx(.044)
    assert scopes[13]['support_removed_point_count'] > 0
    assert result['identity_assignment'] is False
    assert all('confidence' not in row for row in scopes.values())
    assert set(result['bindings']) == set(case[2]['demo_task']['source_object_ids'])
    assert result['bindings'][scopes[8]['entity_id']]['current_description'] == 'a separate pink object inside the left source bin'
    assert len(result['binding_hash']) == 64
    for before, after in zip(original, case[1], strict=True):
        for key in ('points', 'pixels_rc', 'mask', 'silhouette_mask'):
            np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize('change', ['reference_id', 'reference_entities', 'bin_mask', 'bin_depth'])
def test_old_container_result_cannot_authorize_changed_reference_or_capture(change):
    case = frozen_case()
    capture, detections, reference, _, _, tracking = case
    index = int(next(iter(tracking['candidate_mapping'])))
    if change == 'reference_id':
        reference['observation_id'] = 'reset-reference'
    elif change == 'reference_entities':
        reference['entities'][tracking['candidate_mapping'][str(index)]]['description'] = 'changed'
    elif change == 'bin_mask':
        detections[index]['silhouette_mask'][0, 0] ^= True
    else:
        capture['views'][detections[index]['camera_id']]['depth_m'][0, 0] += .01
    assert propose(case) is None


def test_cache_binding_changes_for_same_role_with_changed_current_point_geometry():
    case = frozen_case()
    before = propose(case)
    case[1][13]['points'][:, 0] += .0001
    after = propose(case)
    assert set(before['candidate_scopes']) == set(after['candidate_scopes'])
    assert before['binding_hash'] != after['binding_hash']


@pytest.mark.parametrize('change', ['purpose', 'missing_bin', 'stale_tracking', 'tracking_hash',
    'missing_transition', 'transition_unknown', 'transition_old_frame', 'transition_generation',
    'transition_wrong_target', 'transition_owner', 'transition_revision', 'missing_required', 'duplicate_role',
    'table_invalid'])
def test_invalid_or_ambiguous_binding_falls_back_to_generic(change):
    case = frozen_case()
    _, _, reference, entities, table, tracking = case
    identifier = next(k for k, v in reference['demo_task']['item_lifecycle'].items()
                      if v['phase'] == 'unpacked_confirmed')
    record = reference['demo_task']['item_lifecycle'][identifier]
    if change == 'purpose':
        assert propose(case, purpose='initialize_inventory') is None
        return
    if change == 'missing_bin':
        tracking['candidate_mapping'] = {}
    elif change == 'stale_tracking':
        tracking['observation_id'] = 'old'
    elif change == 'tracking_hash':
        tracking['binding_hash'] = 'changed'
    elif change == 'missing_transition':
        record.pop('unpacking_evidence')
    elif change == 'transition_unknown':
        record['unpacking_evidence']['success'] = False
    elif change == 'transition_old_frame':
        record['unpacking_evidence']['observation_id'] = 'old'
    elif change == 'transition_generation':
        record['unpacking_evidence']['generation'] += 1
    elif change == 'transition_wrong_target':
        record['unpacking_evidence']['target_id'] = 'another'
    elif change == 'transition_owner':
        record['mock_action']['owner_id'] = 'another'
    elif change == 'transition_revision':
        record['appearance_revision'] += 1
    elif change == 'missing_required':
        reference['demo_task']['item_lifecycle'].pop(identifier)
    elif change == 'duplicate_role':
        record.update(phase='wrapped_source', appearance='wrapped', appearance_revision=0)
        entities[identifier].update(lifecycle_phase='wrapped_source', current_appearance='wrapped',
                                    appearance_revision=0)
    else:
        table['surface_height_m'] = float('nan')
    assert propose(case) is None


@pytest.mark.parametrize('change', ['one_outside_point', 'nonfinite', 'empty', 'bad_pixels'])
def test_inconsistent_candidate_is_not_trimmed_to_manufacture_a_role(change):
    case = frozen_case()
    detection = case[1][13]
    if change == 'one_outside_point':
        # Interior row guaranteed retained by the fixed one-pixel erosion.
        from scipy.ndimage import binary_erosion
        keep = binary_erosion(detection['silhouette_mask'], structure=np.ones((3, 3), bool))
        index = np.flatnonzero(keep[tuple(detection['pixels_rc'].T)])[0]
        detection['points'][index] = [5., 5., 5.]
    elif change == 'nonfinite':
        detection['points'][:] = np.nan
    elif change == 'empty':
        detection['points'] = np.empty((0, 3))
        detection['pixels_rc'] = np.empty((0, 2), int)
    else:
        detection['pixels_rc'][0] = [-1, -1]
    assert 13 not in propose(case)['candidate_scopes']


def test_completed_binding_accepts_tilted_current_bin_geometry_without_upright_height():
    case = frozen_case()
    capture, detections, reference, entities, _, _ = case
    identifier = next(k for k, v in reference['demo_task']['item_lifecycle'].items()
                      if v['phase'] == 'unpacked_confirmed')
    record = reference['demo_task']['item_lifecycle'][identifier]
    destination = reference['demo_task']['right_bin_id']
    record['phase'] = 'completed'
    record['drop_action'] = {'source': 'validated_held_attachment_pose_drop', 'target_id': identifier,
        'owner_id': reference['owner_id'], 'generation': capture['generation'], 'completed': True,
        'completed_time': capture['time'] - 1, 'destination_id': destination,
        'action_id': 'release', 'release_feedback_id': 'release', 'holding_verified': True,
        'attachment_verified': True, 'open_confirmed': True}
    names = ('current_item_added',)
    record['drop_visual_evidence'] = {'source': 'openai_decisions', 'status': 'at_destination',
        'target_id': identifier, 'destination_id': destination, 'observation_id': 'after-drop',
        'request_hash': 'drop-request', 'action_id': 'release', 'release_feedback_id': 'release',
        'confidence': .9, 'judgments': dict.fromkeys(names, 'yes'),
        'answers': [{'name': name, 'type': 'choice', 'choice': 'yes', 'confidence': .9} for name in names]}
    from luna_mujoco.demo_lifecycle_perception import drop_result_evidence
    record['drop_visual_evidence'].update(drop_result_evidence(
        {'answers': record['drop_visual_evidence']['answers']}, .8))
    entities[identifier]['lifecycle_phase'] = 'completed'
    detection = detections[4]
    detection['points'][:, :2] = [.20, -.60]
    detection['points'][:, 2] = np.linspace(.43, .55, len(detection['points']))
    result = propose(case)
    assert result['candidate_scopes'][4]['role'] == 'completed'
    assert result['bindings'][identifier]['height_prior_m'] is None
    record['drop_action']['open_confirmed'] = False
    assert propose(case) is None
    record['drop_action']['open_confirmed'] = True
    record['drop_visual_evidence']['confidence'] = .79
    assert propose(case) is None


def test_created_product_uses_current_product_reference_not_initial_pink_id():
    from luna_mujoco.demo_process import create_product, initialize_processes
    case = frozen_case()
    capture, detections, reference, _entities, table, tracking = case
    task = reference['demo_task']
    source = next(key for key, row in task['item_lifecycle'].items() if row['phase'] == 'unpacked_confirmed')
    task['transport_ids'] = list(task['source_object_ids'])
    initialize_processes(task, reference['entities'])
    record = task['item_lifecycle'][source]
    result = deepcopy(record['unpacking_evidence'])
    record.update(phase='unpack_pending', appearance='wrapped', appearance_revision=0)
    product = create_product(task, source, result, capture, owner_id=reference['owner_id'])
    reference['product_references'] = {product['id']: {'observation_id': capture['observation_id'],
        'evidence': deepcopy(product['creation_evidence']), 'views': {key: {} for key in ('1', '2', '3')}}}
    current = lifecycle_entities(reference['entities'], task)
    proposed = propose((capture, detections, reference, current, table, tracking))
    assert proposed is not None and product['id'] in proposed['bindings']
    assert source not in current and source not in proposed['bindings']
    assert product['id'] not in reference['entities']
    assert all(row['entity_id'] == product['id'] for row in proposed['candidate_scopes'].values()
               if row['role'] == 'unpacked_confirmed')
    reference['product_references'][product['id']]['evidence']['request_hash'] = 'forged'
    assert propose((capture, detections, reference, current, table, tracking)) is None


def test_multiple_wrapped_candidates_remain_unassigned_until_strict_identity():
    case = frozen_case()
    capture, detections, reference, entities, table, tracking = case
    task = reference['demo_task']
    original = next(key for key, row in task['item_lifecycle'].items() if row['phase'] == 'wrapped_source')
    task['source_object_ids'].append('additional-pink')
    task['item_lifecycle']['additional-pink'] = deepcopy(task['item_lifecycle'][original])
    entities['additional-pink'] = deepcopy(entities[original])
    reference['entities']['additional-pink'] = deepcopy(reference['entities'][original])
    # Rebuild container hash because the immutable initial catalog was extended in this test fixture.
    import hashlib
    import json

    from luna_mujoco.demo_container_tracking import container_correspondence
    tracks = reference['container_identity_tracks']
    tracks['entities_sha256'] = hashlib.sha256(json.dumps(reference['entities'], sort_keys=True, allow_nan=False).encode()).hexdigest()
    scope = {key: tracks[key] for key in ('reference_observation_id', 'generation', 'initial_request_hash', 'entities_sha256')}
    tracks['scope_id'] = hashlib.sha256(json.dumps(scope, sort_keys=True, allow_nan=False).encode()).hexdigest()
    tracking = container_correspondence(capture, detections, reference, entities)
    result = propose((capture, detections, reference, entities, table, tracking))
    assert result is not None
    wrapped = [row for row in result['candidate_scopes'].values() if row['role'] == 'wrapped_source']
    assert wrapped and all(row['entity_id'] is None and set(row['entity_ids']) == {original, 'additional-pink'} for row in wrapped)


def test_two_completed_products_publish_one_explicit_aggregate_scope():
    from luna_mujoco.demo_process import initialize_processes
    capture, detections, reference, _entities, table, tracking = frozen_case()
    task = reference['demo_task']
    task['transport_ids'] = list(task['source_object_ids'])
    initialize_processes(task, reference['entities'])
    for number, source in enumerate(task['source_object_ids']):
        product = 'product-' + str(number)
        task['item_lifecycle'][source].update(phase='consumed', appearance='wrapped', appearance_revision=0)
        action = {'source': 'validated_held_attachment_pose_drop', 'owner_id': reference['owner_id'],
            'target_id': product, 'generation': capture['generation'], 'completed': True,
            'action_id': 'drop-' + product, 'completed_time': capture['time'] - .1,
            'holding_verified': True, 'attachment_verified': True, 'open_confirmed': True,
            'destination_id': task['right_bin_id'], 'release_feedback_id': 'release-' + product}
        visual = {'source': 'openai_decisions', 'request_hash': 'visual-' + product,
            'observation_id': 'after-' + product, 'confidence': .99, 'status': 'at_destination',
            'target_id': product, 'destination_id': task['right_bin_id'], 'action_id': action['action_id'],
            'release_feedback_id': action['release_feedback_id'],
            'judgments': dict.fromkeys(('current_item_added',), 'yes'),
            'answers': [{'name': name, 'type': 'choice', 'choice': 'yes', 'confidence': .99}
                        for name in ('current_item_added',)]}
        from luna_mujoco.demo_lifecycle_perception import drop_result_evidence
        visual.update(drop_result_evidence({'answers': visual['answers']}, .8))
        task['item_lifecycle'][product] = {'phase': 'completed', 'appearance': 'unpacked', 'appearance_revision': 1,
            'input_entity_id': source, 'drop_action': action, 'drop_visual_evidence': visual,
            'to_table_arm': 'A', 'to_right_bin_arm': 'B'}
        task['product_entities'][product] = {'id': product, 'kind': 'object', 'color': 'black'}
    current = lifecycle_entities(reference['entities'], task)
    result = propose((capture, detections, reference, current, table, tracking))
    collection = 'product-collection:' + task['right_bin_id']
    assert result is not None and set(result['bindings']) == {collection}
    assert result['bindings'][collection]['members'] == ['product-0', 'product-1']
    assert result['bindings'][collection]['identity_assignment'] is False
    assert current[collection]['kind'] == 'collection'
    assert all(scope['entity_id'] not in ('product-0', 'product-1') for scope in result['candidate_scopes'].values())
