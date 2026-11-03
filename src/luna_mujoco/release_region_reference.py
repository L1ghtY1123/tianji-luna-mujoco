"""Project declared destination geometry for approximate RGB verification only."""
import numpy as np

from .observation_labels import project_placement_regions


def _clip_region_polygon(polygon, halfplanes):
    """Clip the display outline by the already established language constraints."""
    polygon = np.asarray(polygon, dtype=float)
    for plane in halfplanes:
        output = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            first, second = np.dot(start, plane[:2]) + plane[2], np.dot(end, plane[:2]) + plane[2]
            if (first <= 0) != (second <= 0):
                output.append(start + first / (first - second) * (end - start))
            if second <= 0:
                output.append(end)
        polygon = np.asarray(output, dtype=float).reshape(-1, 2)
        if len(polygon) < 3:
            break
    return polygon


def release_region_reference(capture, binding):
    zone = binding.get('placement_zone', {})
    intent = zone.get('intent', {})
    narrowed = (zone.get('geometry_source') == 'configured_xy_measured_support'
        or intent.get('zone', 'whole') != 'whole' or intent.get('relative', 'none') != 'none')
    if not narrowed:
        return None
    polygon = zone.get('polygon_world_xy_m')
    reference = {'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
        'region_source_observation_id': zone.get('observation_id'),
        'source': 'declared_region_projected_using_current_camera_calibration',
        'not_a_landing_measurement': True, 'status': 'unavailable', 'projections': [],
        'meaning': 'The outlines mark the expected region on its support plane, not measured object '
            'boundaries or a real painted marker. Judge approximate proximity of the released object '
            'base/support contact using all raw and annotated CURRENT views; allow perspective and object '
            'height. No exact centroid, containment of every pixel or millimetre tolerance is required. '
            'Projection alone does not prove visibility, lack of occlusion or that the object landed there.'}
    if polygon is None or zone.get('floor_z_m') is None:
        return reference
    polygon = _clip_region_polygon(polygon, zone.get('halfplanes_world_xy', []))
    if len(polygon) < 3:
        return reference
    layout = {'support_entity_id': zone.get('support_entity_id'), 'floor_z_m': zone['floor_z_m'],
        'source': reference['source'], 'observation_id': zone.get('observation_id'),
        'cells': [{'slot_id': 'target-region', 'center_world_xy_m': np.mean(polygon, axis=0).tolist(),
                   'polygon_world_xy_m': polygon}]}
    projections = project_placement_regions(capture, [{'region_id': 'release-target', 'layout': layout}],
        require_visible_support=False)
    for row in projections:
        points = np.asarray(row['polygon'])
        if np.isfinite(points).all() and np.all(points.max(axis=0) >= 0) and np.all(points.min(axis=0) <= 1):
            reference['projections'].append(row)
    reference['status'] = 'projected' if reference['projections'] else 'outside_current_views'
    return reference
