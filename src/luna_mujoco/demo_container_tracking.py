"""Track only unmanipulated containers using explicit initial RGBD evidence."""
import hashlib
import json
from copy import deepcopy

import numpy as np

from .visual_inventory import VisualInventory

SOURCE = 'measured_container_correspondence'
THRESHOLDS = {'mask_iou': .85, 'valid_overlap_fraction': .95, 'minimum_pixels': 512,
              'depth_residual_p95_m': .02, 'camera_translation_m': .001,
              'camera_rotation_rad': float(np.deg2rad(.2))}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    array = np.asarray(value)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + np.ascontiguousarray(array).tobytes()).hexdigest()


def _scope(reference, generation, request_hash):
    return {'reference_observation_id': reference['observation_id'], 'generation': generation,
            'initial_request_hash': request_hash, 'entities_sha256': _digest(reference['entities'])}


def _anchors_digest(containers):
    return _digest({identifier: {'kind': record['kind'], 'initial_identity_answers': record['initial_identity_answers'],
        'views': {camera: {key: view[key] for key in ('array_sha256', 'calibration_sha256')}
                  for camera, view in record['views'].items()}} for identifier, record in containers.items()})


def reference_with_container_tracks(observation, capture, minimum_confidence=.8):
    """Add an archiveable sidecar only after the initial current-geometry result."""
    reference = observation.get('visual_reference', {})
    visual = observation.get('visual_inventory', {})
    evidence, coverage = visual.get('evidence', {}), visual.get('identity_coverage', {})
    check = observation.get('demo_inventory_check', {})
    if (reference.get('observation_id') != capture.get('observation_id')
            or observation.get('observation_id') != capture.get('observation_id')
            or observation.get('generation') != capture.get('generation')
            or 'container_identity_tracks' in reference
            or coverage.get('purpose') != 'initialize_inventory'
            or evidence.get('source') != 'openai_decisions'
            or evidence.get('observation_id') != capture.get('observation_id')
            or not evidence.get('request_hash') or visual.get('missing_geometry_ids')
            or check.get('status') != 'none' or check.get('confidence', 0) < minimum_confidence
            or not set(coverage.get('required_entity_ids', [])) <= set(coverage.get('accepted_entity_ids', []))):
        return reference
    task = reference.get('demo_task', {})
    containers = {}
    for identifier in (task.get('source_bin_id'), task.get('right_bin_id')):
        entity = reference.get('entities', {}).get(identifier, {})
        answers = [deepcopy(a) for a in evidence.get('answers', []) if a.get('name', '').startswith('identity_')
                   and a.get('type') == 'choice' and a.get('choice') == identifier
                   and type(a.get('confidence')) in (int, float) and minimum_confidence <= a['confidence'] <= 1.]
        if entity.get('kind') != 'bin' or not answers:
            continue
        views = {}
        for camera, observed in observation.get('views', {}).items():
            if identifier not in observed.get('instance_masks', {}):
                continue
            view = capture['views'][camera]
            mask = np.asarray(observed['instance_masks'][identifier], dtype=bool)
            valid = np.asarray(view['valid_depth'], dtype=bool)
            depth = np.asarray(view['depth_m'])
            if (view.get('registered_to') != 'rgb' or not view.get('calibration_id')
                    or mask.shape != depth.shape or valid.shape != depth.shape
                    or np.count_nonzero(mask & valid & np.isfinite(depth) & (depth > 0)) < THRESHOLDS['minimum_pixels']):
                continue
            frozen = {key: deepcopy(view[key]) for key in
                      ('calibration_id', 'registered_to', 'intrinsics', 'T_world_camera')}
            frozen.update(mask=mask.copy(), depth_m=depth.copy(), valid_depth=valid.copy(),
                          mask_source='initial_confirmed_instance_mask_after_robot_and_surface_filter')
            frozen['array_sha256'] = {key: _array_digest(frozen[key]) for key in ('mask', 'depth_m', 'valid_depth')}
            frozen['calibration_sha256'] = _digest({key: frozen[key] for key in
                ('calibration_id', 'registered_to', 'intrinsics', 'T_world_camera')})
            views[camera] = frozen
        if views:
            containers[identifier] = {'kind': 'bin', 'initial_identity_answers': answers, 'views': views}
    if not containers:
        return reference
    scope = _scope(reference, capture['generation'], evidence['request_hash'])
    track = {**scope, 'scope_id': _digest(scope), 'initial_capture_time': capture['time'],
             'source': 'initial_decisions_registered_rgbd', 'containers': containers,
             'anchors_sha256': _anchors_digest(containers)}
    if reference.get('owner_id'):
        track['owner_id'] = reference['owner_id']
    return {**reference, 'container_identity_tracks': track}


def _compatible_view(current, anchor):
    if (current.get('registered_to') != 'rgb' or current.get('calibration_id') != anchor['calibration_id']
            or current.get('intrinsics') != anchor['intrinsics']):
        return False
    for key in ('mask', 'depth_m', 'valid_depth'):
        if _array_digest(anchor[key]) != anchor['array_sha256'][key]:
            return False
    if _digest({key: anchor[key] for key in ('calibration_id', 'registered_to', 'intrinsics', 'T_world_camera')
                }) != anchor['calibration_sha256']:
        return False
    a, b = np.asarray(anchor['T_world_camera'], float), np.asarray(current['T_world_camera'], float)
    if a.shape != (4, 4) or b.shape != (4, 4) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False
    for matrix in (a, b):
        if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6, rtol=0)):
            return False
    angle = float(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))
    return (np.linalg.norm(a[:3, 3] - b[:3, 3]) <= THRESHOLDS['camera_translation_m']
            and angle <= THRESHOLDS['camera_rotation_rad'])


def container_correspondence(capture, detections, reference, entities):
    """Return independently measured IDs; unavailable evidence leaves model matching intact."""
    result = {'source': SOURCE, 'observation_id': capture.get('observation_id'),
              'generation': capture.get('generation'), 'candidate_mapping': {}, 'matches': {},
              'thresholds': dict(THRESHOLDS), 'status': 'unavailable'}
    tracks = (reference or {}).get('container_identity_tracks')
    if not tracks:
        return result
    try:
        scope = _scope(reference, capture['generation'], tracks['initial_request_hash'])
        if (tracks.get('source') != 'initial_decisions_registered_rgbd'
                or tracks['scope_id'] != _digest(scope)
                or any(tracks.get(key) != value for key, value in scope.items())
                or tracks.get('anchors_sha256') != _anchors_digest(tracks['containers'])
                or capture['observation_id'] == reference['observation_id']
                or not np.isfinite(capture['time']) or not np.isfinite(tracks['initial_capture_time'])
                or capture['time'] <= tracks['initial_capture_time']
                or tracks.get('owner_id') and tracks['owner_id'] != reference.get('owner_id')):
            return result
        result.update(scope_id=tracks['scope_id'], initial_request_hash=tracks['initial_request_hash'])
        allowed = {reference.get('demo_task', {}).get(key) for key in ('source_bin_id', 'right_bin_id')}
        for index, detection in enumerate(detections):
            if detection.get('kind') != 'bin':
                continue
            camera = detection['camera_id']
            current = capture['views'][camera]
            matches = []
            for identifier, container in tracks['containers'].items():
                if (identifier not in allowed or entities.get(identifier, {}).get('kind') != 'bin'
                        or container.get('kind') != 'bin' or camera not in container['views']
                        or not any(a.get('choice') == identifier and a.get('type') == 'choice'
                                   and type(a.get('confidence')) in (int, float) and .8 <= a['confidence'] <= 1.
                                   for a in container.get('initial_identity_answers', []))):
                    continue
                anchor = container['views'][camera]
                if not _compatible_view(current, anchor):
                    continue
                prior, silhouette = np.asarray(anchor['mask'], bool), np.asarray(detection['silhouette_mask'], bool)
                depth, valid = np.asarray(current['depth_m']), np.asarray(current['valid_depth'], bool)
                actual_mask = np.asarray(detection['mask'], bool)
                if any(array.shape != prior.shape for array in (silhouette, depth, valid, actual_mask)):
                    continue
                overlap = prior & silhouette
                union = prior | silhouette
                if not union.any():
                    continue
                iou = float(overlap.sum() / union.sum())
                common = (overlap & actual_mask & valid & anchor['valid_depth'] & np.isfinite(depth)
                          & np.isfinite(anchor['depth_m']) & (depth > 0) & (anchor['depth_m'] > 0))
                count = int(common.sum())
                fraction = float(count / max(int(overlap.sum()), 1))
                if (iou < THRESHOLDS['mask_iou'] or count < THRESHOLDS['minimum_pixels']
                        or fraction < THRESHOLDS['valid_overlap_fraction']):
                    continue
                residual = float(np.quantile(np.abs(depth[common] - anchor['depth_m'][common]), .95))
                if residual > THRESHOLDS['depth_residual_p95_m']:
                    continue
                matches.append({'entity_id': identifier, 'camera_id': camera, 'mask_iou': iou,
                    'common_valid_pixels': count, 'valid_overlap_fraction': fraction, 'depth_residual_p95_m': residual,
                    'initial_mask_sha256': anchor['array_sha256']['mask'],
                    'current_mask_sha256': _array_digest(silhouette), 'current_depth_sha256': _array_digest(depth)})
            if len(matches) == 1:
                key = str(index)
                result['candidate_mapping'][key] = matches[0]['entity_id']
                result['matches'][key] = matches[0]
        result['status'] = 'matched' if result['candidate_mapping'] else 'unmatched'
        result['binding_hash'] = _digest(result)
    except (KeyError, TypeError, ValueError, IndexError):
        result.update(candidate_mapping={}, matches={}, status='unavailable')
    return result


def merge_container_correspondence(inventory, detections, entities, tracking, result):
    """Merge fresh measurements by entity/camera, retaining their separate identity sources."""
    entries = [entry for group in result['groups'] for entry in group]
    origins = {}
    for entry in entries:
        key = (entry['entity_id'], entry['camera_id'])
        origins.setdefault(key, []).extend(entry.get('identity_evidence') or [{'source': 'openai_decisions',
            'observation_id': tracking['observation_id'], 'request_hash': result['evidence']['request_hash'],
            'camera_id': entry['camera_id']}])
    for index, identifier in tracking['candidate_mapping'].items():
        entry = deepcopy(detections[int(index)])
        entry['entity_id'] = identifier
        entries.append(entry)
        origins.setdefault((identifier, entry['camera_id']), []).append({'source': SOURCE,
            'observation_id': tracking['observation_id'], 'binding_hash': tracking['binding_hash'],
            'scope_id': tracking['scope_id'], 'initial_request_hash': tracking['initial_request_hash'],
            'candidate_index': int(index), **tracking['matches'][index]})
    judgments = {'identity_' + str(index): entry['entity_id'] for index, entry in enumerate(entries)}
    merged = VisualInventory._groups(inventory, entries, ['merged-' + str(i) for i in range(len(entries))],
                                     entities, judgments, result['evidence'])
    for group in merged['groups']:
        for entry in group:
            entry['identity_evidence'] = origins[(entry['entity_id'], entry['camera_id'])]
    result['groups'] = merged['groups']
    observed = {g[0]['entity_id'] for g in result['groups'] if any(len(e['points']) for e in g)}
    result['missing_geometry_ids'] = [identifier for identifier, entity in entities.items()
        if entity.get('kind') not in {'table', 'marker'} and identifier not in observed]
    result['container_correspondence'] = tracking
    return result
