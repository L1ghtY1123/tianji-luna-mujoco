"""Derive placement intent from an identified, RGBD-measured table support."""
import numpy as np
from scipy.spatial.transform import Rotation


def measured_table_region(observation, table_id):
    """Return a separate region; never replace the table or its collision model."""
    entity = observation.get('scene_entities', {}).get(table_id, {})
    cache = observation.get('demo_tables', {})
    measured = cache.get('tables', {}).get(table_id, {})
    if (entity.get('kind') != 'table' or not cache.get('source_observation_id')
            or measured.get('source') != 'initial_inventory_ransac_table'
            or measured.get('inlier_count', 0) < 100):
        raise ValueError('measured_table_region_requires_identified_rgbd_support:' + table_id)
    model = measured.get('model', {})
    pose = np.asarray(model.get('pose', []), dtype=float)
    size = np.asarray(model.get('dims', []), dtype=float)
    floor = float(measured.get('surface_height_m', float('nan')))
    if (pose.shape != (7,) or size.shape != (3,) or not np.isfinite(pose).all()
            or not np.isfinite(size).all() or np.any(size <= 0) or not np.isfinite(floor)
            or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=.001)):
        raise ValueError('measured_table_region_invalid_geometry:' + table_id)
    rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    if (rotation[2, 2] < np.cos(np.deg2rad(5))
            or abs(pose[2] + size[2] / 2 - floor) > .006):
        raise ValueError('measured_table_region_requires_horizontal_support:' + table_id)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size[:2] / 2
    polygon = corners @ rotation[:2, :2].T + pose[:2]
    identifier = 'table-region:' + table_id
    position = [*pose[:2].tolist(), floor]
    provenance = {'source': measured['source'], 'observation_id': cache['source_observation_id'],
                  'support_entity_id': table_id, 'point_count': measured['inlier_count']}
    return {'id': identifier, 'kind': 'marker', 'color': 'unmarked',
        'roles': ['target_region'], 'visual_membership': 'excluded',
        'source': 'measured_table_region', 'parent_entity_id': table_id,
        'position_source': 'rgbd_table_support', 'position': position,
        'visible_centroid_world': position.copy(), 'measurement_sources': provenance,
        'placement_reference': {'floor_z_m': floor, 'position_world_m': position.copy(),
            'region_polygon_world_xy_m': polygon.tolist(), 'region_id': 'region:' + identifier,
            'support_entity_id': table_id, 'source': 'measured_table_region',
            'observation_id': observation['observation_id'], 'generation': observation.get('generation'),
            'measurement_sources': provenance.copy()}}
