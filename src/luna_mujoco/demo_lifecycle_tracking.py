"""Nominate unique lifecycle roles from current geometry; never accept identities."""
import hashlib
import json

import numpy as np
from scipy.ndimage import binary_erosion

from .cad_assets import cad_asset, workpiece_height_prior
from .demo_lifecycle_perception import (
    drop_result_evidence,
    unpack_result_evidence,
    unpack_verification_capture,
)
from .demo_process import completed_product_collection

SUPPORT_BAND_M = .006
HEIGHT_TOLERANCE_M = .015
DESCRIPTIONS = {
    'wrapped_source': 'a separate pink object inside the left source bin',
    'unpacked_confirmed': 'a separate black object on the main working table',
    'completed': 'a separate black object inside the right destination bin',
    'completed_collection': 'one or more visible black workpieces (or their visible parts) inside the right destination bin',
}


def _require(condition):
    if not condition:
        raise ValueError('lifecycle_role_evidence_unavailable')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    array = np.asarray(value)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + np.ascontiguousarray(array).tobytes()).hexdigest()


def _points(value):
    points = np.asarray(value, dtype=float)
    _require(points.ndim == 2 and points.shape[1] == 3 and len(points) and np.isfinite(points).all())
    return points


def _visual_drop(visual):
    _require(visual.get('source') == 'openai_decisions' and bool(visual.get('request_hash'))
             and bool(visual.get('observation_id')))
    result = drop_result_evidence({'answers': visual.get('answers')}, .8)
    _require(result['status'] == 'at_destination'
             and all(visual.get(key) == value for key, value in result.items()))


def _transition(identifier, record, capture, reference):
    role = record['phase']
    input_id = record.get('input_entity_id', identifier)
    if role == 'wrapped_source':
        _require(record['appearance'] == 'wrapped' and record['appearance_revision'] == 0)
        _require(not any(record.get(key) for key in ('mock_action', 'unpacking_evidence', 'product_creation')))
        # This records an unchanged role, never physical identity. Publication
        # separately requires the INITIAL/CURRENT wrapped-identity judgment.
        return {'source': 'initial_inventory_unchanged_wrapped_role'}
    _require(record['appearance'] == 'unpacked' and record['appearance_revision'] > 0)
    action = record['mock_action' if role == 'unpacked_confirmed' else 'drop_action']
    _require(action.get('owner_id') == reference['owner_id'] and bool(reference['owner_id'])
             and action.get('target_id') == (input_id if role == 'unpacked_confirmed' else identifier)
             and action.get('generation') == capture['generation']
             and action.get('completed') is True and bool(action.get('action_id'))
             and type(action.get('completed_time')) in (int, float)
             and np.isfinite(action['completed_time']) and action['completed_time'] <= capture['time'])
    if role == 'unpacked_confirmed':
        evidence = record['unpacking_evidence']
        visual = evidence['visual_evidence']
        views = unpack_result_evidence({'answers': visual.get('answers')}, .8)
        _require(views['judgments'] == {'unpack_result': 'yes'}
                 and all(visual.get(key) == value for key, value in views.items())
                 and visual.get('identity_source') == 'verified_single_active_execution_chain'
                 and visual.get('visibility_source') == 'openai_decisions'
                 and visual.get('execution_chain') == {'target_id': input_id, 'owner_id': reference['owner_id'],
                     'generation': capture['generation'], 'active_target_ids': [input_id],
                     'action_id': action['action_id']})
        table = record['table_visual_evidence']
        release_reference = visual.get('table_release_reference', {})
        _require(action.get('source') == 'mock_vla'
                 and evidence.get('success') is True and evidence.get('status') == 'confirmed'
                 and evidence.get('source') == 'openai_decisions' and visual.get('status') == 'confirmed'
                 and evidence.get('target_id') == visual.get('target_id') == input_id
                 and bool(evidence.get('observation_id'))
                 and evidence.get('observation_id') == visual.get('observation_id')
                 and evidence.get('generation') == capture['generation']
                 and evidence.get('appearance') == 'unpacked'
                 and evidence.get('appearance_revision') == record['appearance_revision']
                 and bool(evidence.get('request_hash'))
                 and evidence.get('request_hash') == visual['request_hash']
                 and evidence.get('action_id') == visual.get('action_id') == action['action_id']
                 and table.get('target_id') == input_id
                 and table.get('request_hash') == visual.get('table_release_request_hash')
                 and table.get('binding_id') == visual.get('table_release_binding_id')
                 and release_reference.get('source') == 'confirmed_table_release_rgb'
                 and release_reference.get('owner_id') == reference['owner_id']
                 and release_reference.get('generation') == capture['generation']
                 and all(release_reference.get(key) == table.get(key) for key in
                         ('target_id', 'observation_id', 'request_hash', 'binding_id', 'release_feedback_id'))
                 and set(release_reference.get('images', {})) == {'1', '2', '3'}
                 and bool(table.get('binding_id')) and bool(table.get('request_hash'))
                 and action.get('table_release_feedback_id') == record['table_release_feedback_id']
                 == table.get('release_feedback_id'))
        verification = evidence.get('verification_capture')
        if verification is None:
            # Historical records have no independently bound result-frame RGB.
            # They retain the original same-capture contract and cannot authorize a later frame.
            _require('verification_capture' not in visual
                     and evidence['observation_id'] == capture['observation_id'])
        else:
            _require(isinstance(verification, dict) and visual.get('verification_capture') == verification
                     and verification.get('observation_id') == evidence['observation_id']
                     and verification.get('generation') == capture['generation'])
            result_time = verification.get('time')
            release_time = release_reference.get('time')
            _require(all(type(value) in (int, float) and np.isfinite(value)
                         for value in (release_time, result_time, capture.get('time')))
                     and release_time < action['completed_time'] <= result_time <= capture['time'])
            result_perf = verification.get('captured_perf_counter_ns')
            action_perf = action.get('completed_perf_counter_ns')
            if action_perf is not None or result_perf is not None:
                _require(type(result_perf) is int and type(capture.get('captured_perf_counter_ns')) is int
                         and result_perf <= capture['captured_perf_counter_ns'])
            if action_perf is not None:
                release_perf = release_reference.get('captured_perf_counter_ns')
                _require(type(action_perf) is int and type(release_perf) is int
                         and release_perf < action_perf < result_perf)
            else:
                _require(action['completed_time'] < result_time)
            if input_id == identifier:
                # Same-ID compatibility still needs its actual result pixels in the same frame.
                _require(evidence['observation_id'] == capture['observation_id']
                         and unpack_verification_capture(capture) == verification)
        if input_id != identifier:
            created = record['product_creation']
            product_reference = reference['product_references'][identifier]
            process = reference['demo_task']['process_records'][created['process_id']]
            _require(created.get('source') == 'confirmed_unpack_process_output'
                     and created.get('input_entity_id') == input_id and created.get('output_entity_id') == identifier
                     and process['input_entity_id'] == input_id and process['output_entity_id'] == identifier
                     and created['owner_id'] == reference['owner_id'] and created['generation'] == capture['generation']
                     and created.get('action_id') == action['action_id']
                     and created['request_hash'] == evidence['request_hash']
                     and created['observation_id'] == evidence['observation_id']
                     and product_reference['observation_id'] == created['observation_id']
                     and product_reference['evidence'] == created
                     and set(product_reference['views']) == {'1', '2', '3'})
            if verification is not None:
                _require(product_reference.get('generation') == capture['generation']
                         and product_reference.get('verification_capture') == verification
                         and unpack_verification_capture({**verification,
                             'views': product_reference['views']}) == verification)
    else:
        visual = record['drop_visual_evidence']
        _visual_drop(visual)
        _require(action.get('source') == 'validated_held_attachment_pose_drop'
                 and all(action.get(key) is True for key in ('holding_verified', 'attachment_verified', 'open_confirmed'))
                 and visual.get('status') == 'at_destination' and visual.get('target_id') == identifier
                 and visual.get('destination_id') == action.get('destination_id')
                 == reference['demo_task']['right_bin_id']
                 and visual.get('action_id') == action['action_id']
                 and visual.get('release_feedback_id') == action.get('release_feedback_id')
                 and bool(action.get('release_feedback_id')))
    return {'source': visual['source'], 'request_hash': visual['request_hash'],
            'observation_id': visual['observation_id'], 'action_id': action['action_id']}


def _container_bounds(capture, detections, reference, entities, tracking):
    tracks = reference['container_identity_tracks']
    scope = {'reference_observation_id': reference['observation_id'], 'generation': capture['generation'],
             'initial_request_hash': tracks['initial_request_hash'], 'entities_sha256': _digest(reference['entities'])}
    unsigned = {key: value for key, value in tracking.items() if key != 'binding_hash'}
    _require(tracking.get('source') == 'measured_container_correspondence'
             and tracking.get('status') == 'matched' and tracking.get('binding_hash') == _digest(unsigned)
             and tracking.get('observation_id') == capture['observation_id']
             and tracking.get('generation') == capture['generation']
             and tracking.get('scope_id') == tracks['scope_id'] == _digest(scope)
             and all(tracks.get(key) == value for key, value in scope.items())
             and tracks.get('source') == 'initial_decisions_registered_rgbd'
             and tracking.get('initial_request_hash') == tracks['initial_request_hash']
             and capture['observation_id'] != reference['observation_id']
             and capture['time'] > tracks['initial_capture_time']
             and (not tracks.get('owner_id') or tracks['owner_id'] == reference['owner_id']))
    task = reference['demo_task']
    ids = (task['source_bin_id'], task['right_bin_id'])
    _require(ids[0] != ids[1] and all(entities[key]['kind'] == 'bin' for key in ids))
    grouped = {identifier: [] for identifier in ids}
    for key, identifier in tracking['candidate_mapping'].items():
        _require(isinstance(key, str) and key.isdecimal() and str(int(key)) == key)
        index = int(key)
        _require(index < len(detections) and identifier in grouped)
        detection, match = detections[index], tracking['matches'][key]
        _require(detection.get('kind') == 'bin' and match['entity_id'] == identifier
                 and match['camera_id'] == detection['camera_id'] and detection['camera_id'] in capture['views']
                 and match['current_mask_sha256'] == _array_digest(detection['silhouette_mask'])
                 and match['current_depth_sha256'] == _array_digest(capture['views'][detection['camera_id']]['depth_m']))
        grouped[identifier].append(_points(detection['points']))
    bounds = {}
    for identifier, groups in grouped.items():
        _require(bool(groups))
        points = np.vstack(groups)
        low, high = points.min(0), points.max(0)
        _require(np.all(high > low))
        bounds[identifier] = (low, high)
    return bounds


def _interior_points(detection):
    points = _points(detection['points'])
    pixels = np.asarray(detection['pixels_rc'])
    silhouette = np.asarray(detection['silhouette_mask'])
    _require(silhouette.ndim == 2 and pixels.shape == (len(points), 2)
             and np.issubdtype(pixels.dtype, np.integer) and np.all(pixels >= 0)
             and np.all(pixels < np.asarray(silhouette.shape)))
    interior = binary_erosion(silhouette.astype(bool), structure=np.ones((3, 3), bool))
    kept = points[interior[tuple(pixels.T)]]
    return points, kept


def lifecycle_candidate_scopes(capture, detections, reference, entities, table_scope,
                               container_tracking, *, purpose='associate_inventory'):
    """Return current role proposals, or None when the unique-role contract is unavailable.

    One-pixel silhouette interiors and support-band exclusion locate candidates
    only. Original clouds/masks are untouched; every remaining point must fit.
    """
    try:
        _require(purpose == 'associate_inventory' and type(capture.get('generation')) is int
                 and bool(capture.get('observation_id')) and np.isfinite(capture['time']))
        task = reference['demo_task']
        required = {key for key, entity in entities.items() if entity.get('kind') not in {'table', 'bin', 'marker'}}
        collection = completed_product_collection(task)
        expected = {key for key in task['source_object_ids'] if task['item_lifecycle'][key]['phase'] != 'consumed'}
        expected |= {key for key in task.get('product_entities', {}) if task['item_lifecycle'][key]['phase'] != 'completed'}
        if collection:
            expected.add(collection['id'])
        _require(bool(required) and required == expected and bool(reference.get('owner_id')))
        bins = _container_bounds(capture, detections, reference, entities, container_tracking)
        position, size = np.asarray(table_scope['position'], float), np.asarray(table_scope['size'], float)
        floor = float(table_scope['surface_height_m'])
        _require(position.shape == size.shape == (3,) and np.isfinite([*position, *size, floor]).all()
                 and np.all(size > 0) and table_scope.get('source') == 'initial_inventory_ransac_table')
        bindings, scopes = {}, {}
        for identifier in sorted(required):
            if collection and identifier == collection['id']:
                transitions = [_transition(key, task['item_lifecycle'][key], capture, reference) for key in collection['members']]
                low, high = bins[task['right_bin_id']]
                scopes['completed_collection'] = ([identifier], low-SUPPORT_BAND_M, high+SUPPORT_BAND_M, float(low[2]))
                bindings[identifier] = {'role': 'completed_collection', 'current_description': DESCRIPTIONS['completed_collection'],
                    'members': collection['members'], 'transition_evidence': transitions, 'identity_assignment': False,
                    'support_entity_id': task['right_bin_id'], 'support_z_m': float(low[2]), 'height_prior_m': None}
                continue
            record, entity = task['item_lifecycle'][identifier], entities[identifier]
            role = record['phase']
            _require(role in DESCRIPTIONS and (role == 'wrapped_source' or role not in scopes)
                     and type(record.get('appearance_revision')) is int
                     and entity.get('current_appearance') == record.get('appearance')
                     and entity.get('lifecycle_phase') == role
                     and entity.get('appearance_revision') == record['appearance_revision']
                     and (reference['entities'].get(identifier) or task.get('product_entities', {}).get(identifier, {})).get('kind')
                     not in {'table', 'bin', 'marker'})
            transition = _transition(identifier, record, capture, reference)
            if role == 'unpacked_confirmed':
                low = np.r_[position[:2] - size[:2] / 2 - SUPPORT_BAND_M, floor - HEIGHT_TOLERANCE_M]
                high = np.r_[position[:2] + size[:2] / 2 + SUPPORT_BAND_M,
                             floor + workpiece_height_prior('black_core') + HEIGHT_TOLERANCE_M]
                support, support_id = floor, task['destination_table_id']
            else:
                support_id = task['source_bin_id' if role == 'wrapped_source' else 'right_bin_id']
                minimum, maximum = bins[support_id]
                low, high = minimum - SUPPORT_BAND_M, maximum + SUPPORT_BAND_M
                support = float(minimum[2])
                if role == 'wrapped_source':
                    low[2] = support - HEIGHT_TOLERANCE_M
                    high[2] = max(maximum[2], support + cad_asset('pink_wrapped')['size_m'][2]) + HEIGHT_TOLERANCE_M
            if role in scopes:
                scopes[role][0].append(identifier)
            else:
                scopes[role] = ([identifier], low, high, support)
            bindings[identifier] = {'role': role, 'current_description': DESCRIPTIONS[role],
                'appearance_revision': record['appearance_revision'], 'transition_evidence': transition,
                'support_entity_id': support_id, 'support_z_m': support,
                'bounds_world_m': [low.tolist(), high.tolist()],
                'height_prior_m': workpiece_height_prior('pink_wrapped' if role == 'wrapped_source' else 'black_core')
                    if role in {'wrapped_source', 'unpacked_confirmed'} else None}
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return None
    proposed = {}
    for index, detection in enumerate(detections):
        if detection.get('kind') == 'bin' or detection.get('camera_id') not in capture['views']:
            continue
        try:
            raw, interior = _interior_points(detection)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        matches = []
        for role, (identifiers, low, high, support) in scopes.items():
            points = interior[np.abs(interior[:, 2] - support) > SUPPORT_BAND_M]
            if len(points) and np.all((points >= low) & (points <= high)):
                matches.append({'entity_id': identifiers[0] if len(identifiers) == 1 else None,
                    'entity_ids': list(identifiers), 'role': role, 'camera_id': detection['camera_id'],
                    'raw_point_count': len(raw), 'interior_removed_point_count': len(raw) - len(interior),
                    'support_removed_point_count': len(interior) - len(points), 'point_count': len(points),
                    'measured_bounds_world_m': [points.min(0).tolist(), points.max(0).tolist()],
                    'raw_points_sha256': _array_digest(detection['points']),
                    'silhouette_sha256': _array_digest(detection['silhouette_mask']),
                    'support_band_m': SUPPORT_BAND_M, 'silhouette_interior_pixels': 1,
                    'identity_assignment': False})
        if len(matches) == 1:
            proposed[index] = matches[0]
    result = {'source': 'current_geometry_unique_lifecycle_role_proposal', 'identity_assignment': False,
        'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'owner_id': reference['owner_id'],
        'reference_observation_id': reference['observation_id'],
        'container_binding_hash': container_tracking['binding_hash'],
        'bindings': bindings, 'candidate_scopes': proposed}
    result['binding_hash'] = _digest(result)
    return result
