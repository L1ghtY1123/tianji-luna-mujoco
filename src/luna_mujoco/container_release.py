"""Container release geometry. Semantic mode is supplied by the graph binding."""
from copy import deepcopy

import numpy as np


def release_mode(request):
    mode = (request or {}).get('release_mode')
    if mode not in {'surface_setdown', 'container_release'}:
        raise ValueError('placement_release_mode_required')
    return mode


def configure_release_region(destination, settings):
    """Bind configured XY to the measured container support, with explicit bounds."""
    from scipy.spatial import ConvexHull

    if not settings['container_release_fixed_xy']:
        return
    reference = destination['placement_reference']
    if 'measured_container_polygon_world_xy_m' not in reference:
        reference['measured_container_polygon_world_xy_m'] = deepcopy(
            reference['region_polygon_world_xy_m'])
    center = np.array([settings['container_release_x_m'], settings['container_release_y_m']])
    half = settings['container_release_region_size_m'] / 2
    polygon = center + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    hull = ConvexHull(np.asarray(reference['measured_container_polygon_world_xy_m']))
    tolerance = float(settings['ik_position_threshold_m'])
    if np.any(center @ hull.equations[:, :2].T + hull.equations[:, 2] + tolerance > 1e-9):
        raise ValueError('configured_release_center_outside_measured_container')
    reference.update(fixed_xy=True, position_world_m=[*center.tolist(), reference['floor_z_m']],
        region_polygon_world_xy_m=polygon.tolist(),
        configured_release_region={'center_world_xy_m': center.tolist(),
            'side_m': settings['container_release_region_size_m'], 'source': 'operator_configuration',
            'purpose': 'robot_release_position_only', 'center_clearance_required_m': tolerance})


def container_release_reference(destination, scene, observation, settings):
    from .robot_skills.container_geometry import container_occupied_bounds

    bounds = container_occupied_bounds(destination, scene, observation)
    return {'release_mode': 'container_release',
        'rim_z_m': float(bounds['maximum_world_m'][2]),
        'container_release_bottom_clearance_m': settings['container_release_bottom_clearance_m'],
        'release_position_tolerance_m': settings['ik_position_threshold_m'],
        'rim_provenance': deepcopy(bounds), 'scene_token': scene['scene_token']}
