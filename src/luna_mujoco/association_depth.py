"""Measured spatial evidence for visual identity; never assigns permanent IDs."""
from copy import deepcopy

import numpy as np

from .cameras import deproject_view

FRAME = 'world (Link_Base): metres, +X robot-forward, +Y robot-left, +Z up'
DEPTH_INSTRUCTIONS = (
    'Use candidate_depth_evidence together with the labelled RGB. All coordinates use the same fixed '
    'robot world frame, not image left/right or distance from each camera. Measurements describe visible '
    'surfaces, not complete object centers; different views need not have identical centers or overlapping '
    'surface bounds. Insufficient depth proves neither a match nor a mismatch. Historical spatial evidence '
    'is dated: account for recorded motion and release destinations, and never require a moved object to '
    'remain at its old position. A release prediction is not a measured landing position. '
    'A reference with the EXACT SAME observation_id and generation is the same frozen capture, '
    'not an earlier object location. Its confirmed spatial identity still applies in other views. '
    'candidate_history_relations separates these same-capture constraints from historical position priors.')


def candidate_depth_evidence(capture, candidate):
    """Summarize already calibrated, workspace-filtered per-mask world points."""
    points = np.asarray(candidate.get('points', []), dtype=float)
    mask = np.asarray(candidate.get('silhouette_mask', []), dtype=bool)
    mask_pixels = int(np.count_nonzero(mask))
    evidence = {'source': 'same_capture_mask_depth_and_calibration', 'frame': FRAME,
        'observation_id': capture.get('observation_id'), 'generation': capture.get('generation'),
        'capture_time_s': capture.get('time'), 'camera_id': candidate['camera_id'],
        'calibration_id': capture.get('views', {}).get(candidate['camera_id'], {}).get('calibration_id'),
        'scope': 'visible_surface_only', 'mask_pixel_count': mask_pixels, 'status': 'unavailable'}
    if points.ndim != 2 or points.shape[1] != 3:
        return evidence
    points = points[np.isfinite(points).all(axis=1)]
    fraction = min(1., len(points) / mask_pixels) if mask_pixels else 0.
    evidence.update(valid_point_count=len(points), valid_depth_fraction=fraction)
    if not len(points):
        return evidence
    bounds = np.quantile(points, [.02, .98], axis=0)
    evidence.update(status='measured' if len(points) >= 30 and fraction >= .5 else 'insufficient',
        visible_surface_center_world_m=np.median(points, axis=0).tolist(),
        visible_bounds_world_m=bounds.tolist(), bounds_quantiles=[.02, .98])
    return evidence


def spatial_relation(left, right):
    """Only reject gross same-time separation; surface agreement is not identity proof."""
    if (left.get('status') != 'measured' or right.get('status') != 'measured'
            or left.get('observation_id') != right.get('observation_id')
            or left.get('generation') != right.get('generation')):
        return {'status': 'insufficient', 'reason': 'reliable_same_capture_depth_required'}
    return _visible_bounds_relation(left, right)


def _visible_bounds_relation(left, right):
    a, b = np.asarray(left['visible_bounds_world_m']), np.asarray(right['visible_bounds_world_m'])
    gap = np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.)
    separation = float(np.linalg.norm(gap))
    # Opposite faces may not overlap. A visible diagonal and a 15 cm floor
    # deliberately reserve room for unobserved object extent; 2 cm covers depth/calibration error.
    allowance = max(.15, float(np.linalg.norm(a[1] - a[0])), float(np.linalg.norm(b[1] - b[0]))) + .02
    return {'status': 'separated' if separation > allowance else 'not_separated',
        'visible_bounds_gap_m': separation, 'separation_allowance_m': allowance,
        'rule': 'visible_bounds_gap_gt_max_0.15m_or_visible_diagonal_plus_0.02m',
        'identity_proven': False}


def reference_depth_evidence(reference):
    """Use dated masks and dated wrist extrinsics; never apply current camera poses."""
    result = {}
    for camera, view in reference.get('views', {}).items():
        if any(key not in view for key in ('mask', 'depth_m', 'valid_depth', 'intrinsics', 'T_world_camera')):
            continue
        mask = np.asarray(view['mask'], dtype=bool)
        scope = view.get('effective_mask', view.get('workspace_mask'))
        if scope is not None:
            mask = mask & np.asarray(scope, dtype=bool)
        candidate = {'camera_id': camera, 'silhouette_mask': mask, 'points': deproject_view(view, mask)}
        result[camera] = candidate_depth_evidence(reference, candidate)
    return result


def historical_spatial_evidence(capture, entity, references):
    receipt = entity.get('last_motion_receipt') or {}
    frames = []
    for label, reference in references.items():
        measured_time, motion_time = reference.get('time'), receipt.get('feedback_time_s')
        comparable = (receipt.get('generation') == reference.get('generation') == capture.get('generation')
                      and isinstance(measured_time, (int, float)) and isinstance(motion_time, (int, float)))
        moved_after = motion_time > measured_time if comparable else None
        same_capture = (bool(capture.get('observation_id'))
                        and reference.get('observation_id') == capture['observation_id']
                        and reference.get('generation') == capture.get('generation'))
        frames.append({'reference_label': label, 'observation_id': reference.get('observation_id'),
            'generation': reference.get('generation'), 'capture_time_s': measured_time,
            'depth_by_camera': reference_depth_evidence(reference),
            'same_capture': same_capture,
            'recorded_motion_after_reference': moved_after,
            'position_interpretation': ('same_frozen_capture_spatial_constraint' if same_capture else
                'pre_motion_position_do_not_require_current_match' if moved_after else
                'dated_position_prior_only_not_identity_constraint')})
    motion = {key: deepcopy(receipt[key]) for key in ('kind', 'arm', 'generation', 'feedback_time_s', 'source')
              if key in receipt}
    carried = (receipt.get('evidence') or {}).get('carried_geometry_at_open') or {}
    if carried:
        motion['release_prediction'] = {key: deepcopy(carried[key]) for key in (
            'source', 'destination_id', 'object_reference_world_m', 'T_world_tcp', 'scope') if key in carried}
        motion['release_prediction']['measurement_status'] = 'action_prediction_not_current_measurement'
    return {'dated_references': frames, 'last_measured_position': deepcopy(entity.get('last_position_measurement')),
        'last_motion_receipt': motion or None, 'current_region': entity.get('current_region'),
        'placement_evidence': {key: deepcopy(entity[key]) for key in ('visual_release', 'container_entry')
                               if entity.get(key)},
        'interpretation': 'Appearance, dated measurements and executed motion jointly support identity. '
            'Same frozen capture spatial contradictions exclude identity candidates. Different captures use '
            'position only as a prior; missing motion history does not prove that an object stayed still.'}


def candidate_history_relation(current, history):
    comparisons = []
    for frame in history['dated_references']:
        for camera, previous in frame['depth_by_camera'].items():
            if current.get('status') != 'measured' or previous.get('status') != 'measured':
                continue
            if current.get('generation') != previous.get('generation'):
                continue
            relation = _visible_bounds_relation(current, previous)
            comparisons.append({'reference_label': frame['reference_label'], 'reference_camera_id': camera,
                'observation_id': frame['observation_id'], 'same_capture': frame['same_capture'],
                'recorded_motion_after_reference': frame['recorded_motion_after_reference'], **relation,
                'interpretation': frame['position_interpretation']})
    return {'same_capture_conflict': any(row['same_capture'] and row['status'] == 'separated'
                                        for row in comparisons), 'comparisons': comparisons}


def association_depth_state(capture, candidates, entities=None, reference=None):
    """Small JSON evidence shared by selection, ownership and temporal questions."""
    result = {'spatial_frame': FRAME, 'candidate_depth_evidence': {
        label: row.get('depth_evidence') or candidate_depth_evidence(capture, row)
        for label, row in candidates.items()}}
    if entities:
        reference = reference or {}
        owners = reference.get('archive_owners', {})
        refs = reference.get('identity_references', {})
        result['historical_spatial_evidence'] = {identifier: historical_spatial_evidence(capture, entity, {
            label: row for label, row in refs.items()
            if label == identifier or owners.get(label, {}).get('entity_id') == identifier})
            for identifier, entity in entities.items()}
        result['candidate_history_relations'] = {label: {
            identifier: candidate_history_relation(current, history)
            for identifier, history in result['historical_spatial_evidence'].items()}
            for label, current in result['candidate_depth_evidence'].items()}
    return result
