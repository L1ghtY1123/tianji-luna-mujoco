"""Measured positions and table evidence, independent of task state."""
from copy import deepcopy

import numpy as np

SOURCE_BIN_MIN_XY_FRACTION = .5


def current_table_measurements(observation):
    """Summarize same-capture measured planes, without asserting visual identity."""
    tables = observation.get('demo_tables', {})
    capture_id = observation.get('observation_id')
    if not capture_id or tables.get('source_observation_id') != capture_id:
        return {}
    result = {}
    for identifier, measured in tables.get('tables', {}).items():
        entity = observation.get('scene_entities', {}).get(identifier, {})
        height, count = measured.get('surface_height_m'), measured.get('inlier_count')
        if (entity.get('kind') != 'table' or measured.get('source') != 'initial_objects_ransac_table'
                or type(height) not in (int, float) or not np.isfinite(height)
                or type(count) is not int or count <= 0):
            continue
        cameras = {entity.get('reference_camera_id')}
        fusion = measured.get('fit', {}).get('multicamera_fusion') or {}
        cameras.update(camera for camera, points in fusion.get('accepted_point_count_by_camera', {}).items()
                       if type(points) is int and points > 0)
        result[identifier] = {
            'id': identifier, 'kind': 'table',
            **{key: deepcopy(entity[key]) for key in ('description', 'color') if key in entity},
            'observation_id': capture_id, 'measurement_source': measured['source'],
            'measurement_scope': 'measured_horizontal_surface_not_visual_confirmation',
            'surface_height_m': float(height), 'inlier_count': count,
            'camera_ids': sorted(camera for camera in cameras if camera in observation.get('views', {})),
        }
    return result


def measured_position(obj):
    value = obj.get('visible_centroid_world', obj.get('position'))
    if value is None:
        return None
    position = np.asarray(value, dtype=float)
    return position if position.shape == (3,) and np.isfinite(position).all() else None
