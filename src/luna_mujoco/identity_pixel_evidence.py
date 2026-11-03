"""Compare calibrated reference pixels without deciding physical identity."""
import hashlib
import json

import numpy as np

from .visual_inventory import mask_box

IDENTICAL_REGION_RULE = (
    'These are different acquisitions. Only the listed same-camera union regions have byte-identical '
    'raw RGB, depth and validity with identical calibration, intrinsics and camera pose. Directly compare '
    'the reference and candidate 2D positions there. This is local pixel evidence, not proof of continuous '
    'history or physical identity; boxes may contain different subjects. Decide identity from the images, '
    'and retain unknown when ambiguous. Other regions and cameras receive no such claim.')


def _same_bytes(first, second):
    return (first.shape == second.shape and first.dtype == second.dtype
            and np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(second).tobytes())


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _matching_views(camera, capture, reference, measurement):
    current = capture.get('views', {}).get(camera, {})
    initial = measurement.get('views', {}).get(camera, {})
    reference_rgb = reference.get('views', {}).get(camera, {}).get('rgb')
    required = ('rgb', 'depth_m', 'valid_depth', 'intrinsics', 'T_world_camera', 'calibration_id')
    if reference_rgb is None or any(key not in view for view in (initial, current) for key in required):
        return None
    if (initial.get('camera_id') != camera or current.get('camera_id') != camera
            or initial.get('frame_id') != measurement['observation_id']
            or current.get('frame_id') != capture['observation_id']
            or not initial['calibration_id'] or initial['calibration_id'] != current['calibration_id']
            or not initial['intrinsics'] or initial['intrinsics'] != current['intrinsics']):
        return None
    poses = [np.asarray(view['T_world_camera']) for view in (initial, current)]
    if (any(pose.shape != (4, 4) or not np.isfinite(pose).all() for pose in poses)
            or not _same_bytes(*poses)):
        return None
    rgb = np.asarray(initial['rgb'])
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not _same_bytes(rgb, np.asarray(reference_rgb)):
        return None
    for key, shape in (('rgb', rgb.shape), ('depth_m', rgb.shape[:2]), ('valid_depth', rgb.shape[:2])):
        before, after = np.asarray(initial[key]), np.asarray(current[key])
        if before.shape != shape or after.shape != shape or before.dtype != after.dtype:
            return None
    return initial, current


def identical_reference_regions(capture, detections, reference, reference_capture):
    """Return only exact, calibrated, same-camera evidence; never an ID assignment."""
    if (not reference or not reference_capture or not capture.get('observation_id')
            or not reference.get('observation_id')
            or reference['observation_id'] == capture['observation_id']
            or reference_capture.get('observation_id') != reference['observation_id']
            or type(capture.get('generation')) is not int
            or reference_capture.get('generation') != capture['generation']):
        return []
    pairs = {camera: _matching_views(camera, capture, reference, reference_capture)
             for camera in capture.get('views', {})}
    rows = []
    for index, detection in enumerate(detections):
        camera = detection['camera_id']
        if pairs.get(camera) is None:
            continue
        initial, current = pairs[camera]
        height, width = np.asarray(current['rgb']).shape[:2]
        silhouette = np.asarray(detection['silhouette_mask'])
        if silhouette.shape != (height, width) or not np.any(silhouette):
            continue
        candidate_box = np.asarray(mask_box(silhouette))
        for identifier, entity in reference.get('entities', {}).items():
            if entity.get('reference_camera_id') != camera or entity.get('kind') in {'table', 'marker'}:
                continue
            box = np.asarray(entity.get('bbox', []), dtype=float)
            if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1)
                    or np.any(box[2:] <= box[:2])):
                continue
            # Overlap only selects regions worth describing; it never selects an identity.
            if np.any(np.minimum(box[2:], candidate_box[2:]) <= np.maximum(box[:2], candidate_box[:2])):
                continue
            lower = np.floor(np.minimum(box[:2], candidate_box[:2]) * [width, height]).astype(int)
            upper = np.ceil(np.maximum(box[2:], candidate_box[2:]) * [width, height]).astype(int)
            region = np.s_[lower[1]:upper[1], lower[0]:upper[0]]
            arrays = {key: np.asarray(current[key])[region] for key in ('rgb', 'depth_m', 'valid_depth')}
            if any(not _same_bytes(np.asarray(initial[key])[region], array) for key, array in arrays.items()):
                continue
            valid = arrays['valid_depth'].astype(bool)
            if not np.any(valid) or not np.all(np.isfinite(arrays['depth_m'][valid]) & (arrays['depth_m'][valid] > 0)):
                continue
            rows.append({
                'candidate': 'candidate-' + str(index), 'reference_entity_id': identifier, 'camera': camera,
                'reference_observation_id': reference['observation_id'],
                'current_observation_id': capture['observation_id'], 'generation': capture['generation'],
                'reference_box': box.tolist(), 'current_box': candidate_box.tolist(),
                'union_roi_xyxy_px': [*lower.tolist(), *upper.tolist()],
                'pixel_count': int(np.prod(upper - lower)), 'valid_depth_pixel_count': int(np.count_nonzero(valid)),
                'calibration_id': current['calibration_id'],
                'intrinsics_sha256': hashlib.sha256(json.dumps(current['intrinsics'], sort_keys=True).encode()).hexdigest(),
                'T_world_camera_sha256': _hash(np.asarray(current['T_world_camera'])),
                **{key + '_sha256': _hash(array) for key, array in arrays.items()},
                'array_layouts': {key: {'shape': list(array.shape), 'dtype': str(array.dtype)}
                                  for key, array in arrays.items()},
                'hash_encoding': 'C-contiguous original array bytes; exact shape and dtype equality required',
                'identity_assignment': False})
    return rows
