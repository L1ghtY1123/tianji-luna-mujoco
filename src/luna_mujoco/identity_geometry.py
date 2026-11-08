"""Necessary geometric exclusions for simultaneous masks of a known rigid instance."""
import hashlib

import numpy as np

from .cameras import deproject_view


def projected_reference_surfaces(capture, detections, references):
    """Historical measured surfaces projected into current masks; never assign identity."""
    rows = []
    for identifier, reference in references.items():
        if reference.get('generation') != capture.get('generation'):
            continue
        for camera, view in capture['views'].items():
            targets = [(index, item) for index, item in enumerate(detections) if item['camera_id'] == camera]
            if not targets:
                continue
            points = [deproject_view(view, view['mask']) for view in reference.get('views', {}).values()]
            if not points or not any(len(part) for part in points):
                continue
            points = np.concatenate(points)
            transform = np.asarray(view['T_world_camera'])
            optical = (points - transform[:3, 3]) @ transform[:3, :3]
            optical = optical[np.isfinite(optical).all(axis=1) & (optical[:, 2] > 0)]
            intrinsics = view['intrinsics']
            pixels = np.rint(optical[:, :2] / optical[:, 2:3]
                * [intrinsics['fx'], intrinsics['fy']] + [intrinsics['cx'], intrinsics['cy']])
            height, width = view['depth_m'].shape
            inside = (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
            pixels, optical = pixels[inside].astype(np.int64), optical[inside]
            order = np.argsort(optical[:, 2])
            _, first = np.unique(pixels[order, 1] * width + pixels[order, 0], return_index=True)
            chosen = order[first]
            pixels, depths = pixels[chosen], optical[chosen, 2]
            for index, target in targets:
                mask = np.asarray(target['silhouette_mask'], dtype=bool)
                overlap = mask[pixels[:, 1], pixels[:, 0]]
                measured = view['depth_m'][pixels[:, 1], pixels[:, 0]]
                valid = (view['valid_depth'][pixels[:, 1], pixels[:, 0]] & np.isfinite(measured)
                    & (measured > 0) & overlap)
                residual = depths[valid] - measured[valid]
                intersection = int(overlap.sum())
                union = len(pixels) + int(mask.sum()) - intersection
                rows.append({'entity_id': identifier, 'target_candidate': 'candidate-' + str(index),
                    'target_camera': camera, 'reference_observation_id': reference['observation_id'],
                    'reference_time_s': reference.get('time'), 'observation_id': capture['observation_id'],
                    'generation': capture['generation'], 'projected_pixels': len(pixels),
                    'pixels_inside_target_mask': intersection, 'mask_iou': intersection / max(1, union),
                    'target_mask_covered_fraction': intersection / max(1, int(mask.sum())),
                    'overlap_valid_depth_pixels': int(valid.sum()),
                    'projected_minus_current_depth_m_p10_p50_p90': (
                        np.quantile(residual, [.1, .5, .9]).tolist() if len(residual) else None),
                    'source': 'historical_masked_rgbd_projection', 'identity_assignment': False})
    return rows


def projected_surface_pairs(capture, detections):
    """Project current measured surfaces into each mask's camera, without assigning IDs."""
    rows = []
    for source_index, source in enumerate(detections):
        points = np.asarray(source.get('points', []), dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,):
            continue
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        for camera, view in capture['views'].items():
            transform, intrinsics = np.asarray(view['T_world_camera']), view['intrinsics']
            optical = (points - transform[:3, 3]) @ transform[:3, :3]
            optical = optical[optical[:, 2] > 0]
            pixels = optical[:, :2] / optical[:, 2:3]
            pixels = pixels * [intrinsics['fx'], intrinsics['fy']] + [intrinsics['cx'], intrinsics['cy']]
            height, width = view['depth_m'].shape
            pixels = np.rint(pixels)
            inside = (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
            pixels, optical = pixels[inside].astype(np.int64), optical[inside]
            # One nearest measured surface sample per projected pixel.
            order = np.argsort(optical[:, 2])
            flat = pixels[order, 1] * width + pixels[order, 0]
            _, first = np.unique(flat, return_index=True)
            chosen = order[first]
            pixels, depths = pixels[chosen], optical[chosen, 2]
            for target_index, target in enumerate(detections):
                if target['camera_id'] != camera or source_index == target_index:
                    continue
                mask = np.asarray(target['silhouette_mask'], dtype=bool)
                overlap = mask[pixels[:, 1], pixels[:, 0]]
                measured = view['depth_m'][pixels[:, 1], pixels[:, 0]]
                valid = (view['valid_depth'][pixels[:, 1], pixels[:, 0]] & np.isfinite(measured)
                         & (measured > 0) & overlap)
                residual = depths[valid] - measured[valid]
                rows.append({
                    'source_candidate': 'candidate-' + str(source_index),
                    'target_candidate': 'candidate-' + str(target_index),
                    'source_camera': source['camera_id'], 'target_camera': camera,
                    'observation_id': capture['observation_id'], 'generation': capture['generation'],
                    'source_surface_points': len(points), 'projected_pixels': len(pixels),
                    'pixels_inside_target_mask': int(overlap.sum()),
                    'projected_samples_in_mask_fraction': float(overlap.mean()) if len(pixels) else None,
                    'target_mask_covered_fraction': float(overlap.sum() / max(1, mask.sum())),
                    'overlap_valid_depth_pixels': int(valid.sum()),
                    'projected_minus_current_depth_m_p10_p50_p90': (
                        np.quantile(residual, [.1, .5, .9]).tolist() if len(residual) else None)})
    return rows


def _bounds(points):
    points = np.asarray(points, dtype='<f8')
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 32 or not np.isfinite(points).all():
        return None
    return points.min(0), points.max(0), hashlib.sha256(points.tobytes()).hexdigest()


def cad_identity_exclusions(capture, detections, anchors, models):
    """Separated observed surfaces cannot belong to one smaller rigid CAD.

    Only a current measured correspondence supplies an anchor. Historical CAD
    contributes shape size, never a historical world position or a new identity.
    """
    bounds = [_bounds(item.get('points', [])) for item in detections]
    rows = []
    for anchor in anchors:
        identifier = anchor['entity_id']
        model = models.get(identifier, {})
        if (anchor.get('observation_id') != capture['observation_id']
                or anchor.get('generation') != capture['generation']
                or anchor.get('source') != 'measured_container_correspondence'
                or not anchor.get('binding_hash')
                or model.get('generation') != capture['generation']
                or model.get('instance_id') != identifier
                or model.get('source') != 'validated_cad_model_binding'
                or not model.get('validated_observation_id')
                or not model.get('source_sha256') or not model.get('mesh_sha256')):
            continue
        maximum = model['maximum_diameter_m'] + 2 * model['surface_tolerance_m']
        if not np.isfinite(maximum) or maximum <= 0:
            continue
        index = int(anchor['candidate'].removeprefix('candidate-'))
        if not 0 <= index < len(detections) or detections[index]['camera_id'] != anchor['camera']:
            continue
        fixed = bounds[index]
        if fixed is None:
            continue
        for number, candidate in enumerate(bounds):
            if candidate is None or number == index:
                continue
            gap = np.maximum(np.maximum(candidate[0] - fixed[1], fixed[0] - candidate[1]), 0.)
            separation = float(np.linalg.norm(gap))
            if separation <= maximum:
                continue
            rows.append({'candidate': 'candidate-' + str(number),
                'camera': detections[number]['camera_id'], 'excluded_instance_id': identifier,
                'assignment': 'excluded_not_reassigned', 'source': 'simultaneous_surface_distance_bound',
                'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'minimum_surface_separation_m': separation,
                'maximum_same_instance_separation_m': float(maximum),
                'candidate_points_sha256': candidate[2], 'anchor_points_sha256': fixed[2],
                'anchor_candidate': anchor['candidate'], 'anchor_binding_hash': anchor['binding_hash'],
                'asset_id': model['asset_id'], 'source_sha256': model['source_sha256'],
                'mesh_sha256': model['mesh_sha256'],
                'model_validated_observation_id': model['validated_observation_id']})
    return rows
