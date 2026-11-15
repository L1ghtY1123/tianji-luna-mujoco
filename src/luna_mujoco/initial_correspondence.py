"""Calibrated evidence for unresolved same-capture camera correspondence."""
import numpy as np


def _directional_projection(subject, candidates, capture):
    """Project measured surface points into other views without assigning identities."""
    points = np.asarray(subject['points'], dtype=float)
    # Bound diagnostic payload cost while retaining the measured point distribution.
    points = points[::max(1, int(np.ceil(len(points) / 4096)))]
    comparisons = []
    for candidate in candidates:
        camera = candidate['camera_id']
        if camera == subject['camera_id']:
            continue
        view = capture['views'][camera]
        transform = np.asarray(view['T_world_camera'], dtype=float)
        local = (points - transform[:3, 3]) @ transform[:3, :3]
        local = local[np.isfinite(local).all(axis=1) & (local[:, 2] > 0)]
        intrinsics = view['intrinsics']
        columns = np.rint(local[:, 0] * intrinsics['fx'] / local[:, 2] + intrinsics['cx']).astype(int)
        rows = np.rint(local[:, 1] * intrinsics['fy'] / local[:, 2] + intrinsics['cy']).astype(int)
        height, width = view['depth_m'].shape
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        rows, columns, z = rows[inside], columns[inside], local[inside, 2]
        observed_depth = view['depth_m'][rows, columns]
        valid = view['valid_depth'][rows, columns] & np.isfinite(observed_depth) & (observed_depth > 0)
        rows, columns, z = rows[valid], columns[valid], z[valid]
        hits = np.asarray(candidate['silhouette_mask'], dtype=bool)[rows, columns]
        residuals = z - view['depth_m'][rows, columns]
        # Metric sensor/calibration error budget, independent of model confidence.
        tolerance = float(view.get('correspondence_depth_tolerance_m', .01))
        compatible = np.abs(residuals) <= tolerance
        supported = compatible & hits
        status = 'no_evidence'
        if int(supported.sum()) >= 30:
            status = 'support'
        elif int(compatible.sum()) >= 30 and not np.any(supported):
            status = 'conflict'
        comparisons.append({
            'candidate_id': candidate['candidate_id'], 'camera_id': camera,
            'projected_valid_samples': len(z), 'samples_inside_candidate_mask': int(hits.sum()),
            'inside_fraction': float(hits.mean()) if len(hits) else None,
            'median_projected_minus_observed_depth_m': float(np.median(residuals[hits])) if np.any(hits) else None,
            'depth_tolerance_m': tolerance, 'minimum_reliable_samples': 30,
            'depth_compatible_samples': int(compatible.sum()),
            'depth_compatible_mask_samples': int(supported.sum()),
            'occluded_samples': int((residuals > tolerance).sum()),
            'free_space_inconsistent_samples': int((residuals < -tolerance).sum()),
            'status': status,
            'visible_surface_centroid_world_m': candidate['geometry']['centroid_world_m'],
        })
    return {
        'source': 'same_capture_depth_intrinsics_extrinsics', 'frame': 'world',
        'subject_candidate_id': subject['candidate_id'],
        'visible_surface_centroid_world_m': subject['geometry']['centroid_world_m'],
        'comparisons': comparisons,
        'interpretation': 'Fractions measure projected sampled surface points inside another mask. '
            'They are not silhouette IoU. Positive depth residual can indicate occlusion. '
            'Different views may see different surfaces, so absent overlap does not prove different identity. '
            'Use this evidence together with the three RGBs. No identity has been assigned by geometry.',
    }


def projection_evidence(subject, candidates, capture, *, confirmed_subjects=()):
    """Check both projection directions through every confirmed same-object view."""
    candidates = [row for row in candidates if row['camera_id'] != subject['camera_id']]
    anchors = {row['candidate_id']: row for row in [subject, *confirmed_subjects]}
    evidence = _directional_projection(subject, candidates, capture)
    evidence['confirmed_subject_ids'] = list(anchors)
    for candidate, comparison in zip(candidates, evidence['comparisons'], strict=True):
        directions = []
        for anchor in anchors.values():
            if anchor['camera_id'] == candidate['camera_id']:
                continue
            for source, destination in ((anchor, candidate), (candidate, anchor)):
                projected = _directional_projection(source, [destination], capture)['comparisons'][0]
                directions.append({**projected, 'source_candidate_id': source['candidate_id'],
                                   'source_camera_id': source['camera_id']})
        statuses = {row['status'] for row in directions}
        comparison['status'] = ('conflict' if 'conflict' in statuses else
                                'support' if 'support' in statuses else 'no_evidence')
        comparison['directional_evidence'] = directions
    evidence['interpretation'] = (
        'Each comparison checks both projection directions against the confirmed same-object views. '
        'A reliable conflict in any direction excludes that candidate. Otherwise any supported direction '
        'provides support. No common visible surface leaves no_evidence. Counts and fractions describe '
        'sampled measured surfaces, not silhouette IoU. The primary counts describe the source-to-destination '
        'projection. directional_evidence contains the other directions and their camera/label bindings. '
        'Use the labelled RGB views to judge identity. Geometry does not assign an identity.')
    return evidence
