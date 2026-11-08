"""Container release geometry. Semantic mode is supplied by the graph binding."""
from copy import deepcopy

import numpy as np

from .geometry import rigid_matrix
from .object_obb import held_geometry_points


def release_mode(request):
    mode = (request or {}).get('release_mode')
    if mode not in {'surface_setdown', 'container_release'}:
        raise ValueError('placement_release_mode_required')
    return mode


def container_release_reference(destination, scene, observation, settings):
    from .robot_skills.container_geometry import container_occupied_bounds

    bounds = container_occupied_bounds(destination, scene, observation)
    return {'release_mode': 'container_release',
        'rim_z_m': float(bounds['maximum_world_m'][2]),
        'container_release_gap_m': settings['container_release_gap_m'],
        'release_position_tolerance_m': settings['ik_position_threshold_m'],
        'rim_provenance': deepcopy(bounds), 'scene_token': scene['scene_token']}


def container_goal(goal, holding, reference):
    """Retain bound XY and rotation; place the complete object's bottom above the rim."""
    result = deepcopy(goal)
    pose = rigid_matrix(result['T_world_tcp']).copy()
    bottom = float((held_geometry_points(holding) @ pose[:3, :3].T)[:, 2].min())
    pose[2, 3] = reference['rim_z_m'] + reference['container_release_gap_m'] - bottom
    old_z = result['T_world_tcp'][2][3]
    target = result['release_reference']
    target.update(deepcopy(reference))
    for field in ('attachment_floor_gap_m', 'gripper_floor_gap_m', 'release_height_m',
                  'requested_release_height_m'):
        target[field] += float(pose[2, 3] - old_z)
    target.update(height_source='complete_object_bottom_above_published_container_rim',
        height_limiting_geometry='complete_object', analytic_tcp_z_m=float(pose[2, 3]),
        nominal_object_rim_gap_m=reference['container_release_gap_m'])
    result['T_world_tcp'] = pose.tolist()
    result['object_reference_world_m'] = (
        pose[:3, :3] @ np.asarray(target['tool_reference_point_m']) + pose[:3, 3]).tolist()
    return result


def container_height_bounds(pose, points, floor_z, rim_z, gap, tolerance):
    """One commanded height; the existing IK position tolerance bounds achieved height."""
    from .release_height_search import height_search_bounds, object_floor_gap

    if (not np.isfinite([floor_z, rim_z, gap, tolerance]).all()
            or rim_z <= floor_z or gap <= 0 or tolerance <= 0
            or abs(object_floor_gap(points, pose, rim_z) - gap) > 1e-9):
        raise ValueError('container_release_height_binding_changed')
    result = height_search_bounds(pose, points, floor_z, rim_z - floor_z + gap + tolerance)
    result.update(release_mode='container_release', rim_z_m=float(rim_z),
        container_release_gap_m=float(gap), release_position_tolerance_m=float(tolerance),
        minimum_object_floor_gap_m=float(rim_z - floor_z + max(0., gap - tolerance)),
        height_offsets_m=[0.], maximum_height_increase_m=0.)
    return result


def rebuild_container_bounds(bounds, pose, points, *, gap=None, tolerance=None):
    return container_height_bounds(pose, points, bounds['floor_z_m'], bounds['rim_z_m'],
        bounds['container_release_gap_m'] if gap is None else gap,
        bounds['release_position_tolerance_m'] if tolerance is None else tolerance)
