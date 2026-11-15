"""Fixed-square release goals using the held object bounds and measured support."""

from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull

from .carrying import held_geometry_points
from .geometry import rigid_matrix


def sphere_array(value):
    spheres = np.asarray(value, dtype=float)
    if (spheres.ndim != 2 or spheres.shape[1] != 4 or not len(spheres)
            or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)):
        raise ValueError("release_requires_finite_positive_spheres")
    return spheres


def held_object_height(holding):
    """Measure vertical CAD/TSDF extent, including the measured source support."""
    reference = holding['grasp_support_reference']
    grasp = rigid_matrix(reference['T_world_tcp_at_grasp'])
    points = held_geometry_points(holding) @ grasp[:3, :3].T + grasp[:3, 3]
    return float(points[:, 2].max() - min(points[:, 2].min(), reference['support_height_m']))


def release_height_reference(grasp_reference, surface_z_m, bottom_clearance_m, *, object_height_m):
    """Use support + object height + margin, independently of attachment balls."""
    try:
        grasp = rigid_matrix(grasp_reference['T_world_tcp_at_grasp'])
        source_bottom = float(grasp_reference['support_height_m'])
        surface = float(surface_z_m)
        gap = float(bottom_clearance_m)
        height = float(object_height_m)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('release_requires_measured_source_support_and_grasp_tcp') from error
    if not np.isfinite([source_bottom, surface, gap, height]).all() or gap < 0 or height <= 0:
        raise ValueError('invalid_object_bottom_release_height')
    offset = float(grasp[2, 3]) - source_bottom
    # Preserve clearance when the front TCP is above the measured object top.
    height_basis = max(height, offset)
    tcp_z = surface + height_basis + gap
    return {'source_object_bottom_z_m': source_bottom, 'tcp_to_object_bottom_z_m': offset,
        'object_height_m': height, 'release_height_basis_m': height_basis,
        'tcp_above_object_height_m': height_basis - height,
        'object_bottom_world_z_m': tcp_z - offset, 'tcp_world_z_m': tcp_z,
        'configured_bottom_clearance_m': gap, 'height_source': 'measured_object_height_and_support',
        'source_support_measurement': deepcopy(grasp_reference)}


def fixed_release_goal(destination, holding, placement_request, attachment_spheres, gripper_spheres,
                       *, clearance_m, observation_id, bottom_clearance_m):
    """Use configured XY and object-bottom clearance, preserving the actual grasp rotation."""
    request = placement_request
    layout = request['layout']
    if request['object_id'] != holding['instance_id']:
        raise ValueError('placement_object_not_bound')
    support = destination['placement_reference']
    if support.get('observation_id', observation_id) != observation_id:
        raise ValueError('destination_support_reference_stale')
    center = np.asarray(support.get('position_world_m', [*layout['center_world_xy_m'], 0.]), float)[:2]
    polygon = np.asarray(layout['region_polygon_world_xy_m'], float)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError('configured_placement_square_required')
    polygon = polygon[ConvexHull(polygon).vertices]
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    if (not np.allclose(lengths, lengths[0], atol=1e-6) or lengths[0] <= 0
            or not np.allclose((edges * np.roll(edges, 1, axis=0)).sum(axis=1), 0., atol=1e-6)
            or not np.allclose(center, polygon.mean(axis=0), atol=1e-6)):
        raise ValueError('configured_region_center_or_square_invalid')
    grasp = rigid_matrix(holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    pose = grasp.copy()
    tool_reference = np.zeros(3)
    floor = float(support['floor_z_m'])
    top = float(request.get('release_surface_z_m', floor))
    height = release_height_reference(holding['grasp_support_reference'], top, bottom_clearance_m,
                                      object_height_m=held_object_height(holding))
    if not np.isfinite(floor):
        raise ValueError('invalid_fixed_release_height')
    pose[:2, 3] = center
    pose[2, 3] = height['tcp_world_z_m']
    region_center = pose[:3, 3].copy()
    mode = request['release_mode']
    reference = {'position_constraint': 'configured_xy_and_object_bottom_clearance',
        'tool_reference_point_m': tool_reference.tolist(),
        'target_reference_world_xy_m': center.tolist(), 'position_world_m': region_center.tolist(),
        'region_center_world_m': region_center.tolist(), 'region_side_m': float(lengths[0]),
        'floor_z_m': floor, 'release_surface_z_m': top, **height,
        'required_clearance_m': float(clearance_m), 'height_search_performed': False,
        'release_mode': mode, 'object_id': holding['instance_id'],
        'region_id': request.get('region_id'), 'layout_revision': request.get('layout_revision'),
        'region_polygon_world_xy_m': polygon.tolist(), 'T_world_tcp_at_grasp': grasp.tolist(),
        'attachment_spheres_tool_m': sphere_array(attachment_spheres).tolist(),
        'gripper_spheres_tool_m': sphere_array(gripper_spheres).tolist(),
        'assumptions': ['source_object_bottom_on_measured_support', 'grasp_world_rotation_preserved',
                        'object_rigid_relative_to_grasp_tcp']}
    if mode == 'container_release':
        reference.update(rim_z_m=top,
            rim_provenance=deepcopy(request['rim_provenance']), scene_token=request['scene_token'])
    return {'T_world_tcp': pose.tolist(), 'destination_id': destination['id'], 'candidate_index': 0,
        'yaw_delta_world_rad': 0., 'frame': 'world', 'source': height['height_source'],
        'object_reference_world_m': region_center.tolist(), 'release_reference': reference}


def release_endpoint_evidence(goal, holding, actual_tcp):
    """Check the actual release pose against the fixed object-center and bottom-height goal."""
    reference = goal['release_reference']
    actual, expected = rigid_matrix(actual_tcp), rigid_matrix(goal['T_world_tcp'])
    height = release_height_reference(holding['grasp_support_reference'], reference['release_surface_z_m'],
        reference['configured_bottom_clearance_m'], object_height_m=held_object_height(holding))
    if any(not np.isclose(height[key], reference[key], atol=1e-9, rtol=0)
           for key in ('source_object_bottom_z_m', 'tcp_to_object_bottom_z_m', 'object_height_m', 'tcp_world_z_m')):
        raise ValueError('release_bottom_reference_changed')
    tool_reference = np.zeros(3)
    if not np.allclose(reference['tool_reference_point_m'], tool_reference, atol=1e-9):
        raise ValueError('release_tool_reference_changed')
    actual_reference = actual[:3, :3] @ tool_reference + actual[:3, 3]
    expected_reference = expected[:3, :3] @ tool_reference + expected[:3, 3]
    distance = float(np.linalg.norm(actual_reference - expected_reference))
    angle = float(np.arccos(np.clip((np.trace(actual[:3, :3].T @ expected[:3, :3]) - 1) / 2, -1, 1)))
    bottom = float(actual[2, 3]) - height['tcp_to_object_bottom_z_m']
    gap = bottom - reference['release_surface_z_m']
    hull = ConvexHull(np.asarray(reference['region_polygon_world_xy_m']))
    inside = bool(np.all(actual_reference[:2] @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9))
    return {'source': 'configured_front_tcp_and_actual_tcp',
        'release_mode': reference['release_mode'], 'tool_reference_point_m': tool_reference.tolist(),
        'at_release_goal': bool(inside and distance <= .005 and angle <= .05),
        'actual_reference_world_m': actual_reference.tolist(),
        'actual_object_reference_world_xy_m': actual_reference[:2].tolist(),
        'position_error_m': distance, 'rotation_error_rad': angle,
        'reference_xy_error_m': float(np.linalg.norm(actual_reference[:2] - expected_reference[:2])),
        'object_floor_gap_m': bottom - reference['floor_z_m'],
        'bottom_height_source': height['height_source'],
        'object_bottom_to_release_surface_m': gap,
        'object_bottom_to_rim_gap_m': gap if reference['release_mode'] == 'container_release' else None,
        'region_center_world_m': reference['region_center_world_m'],
        'object_geometry_inside_region': inside, 'actual_tcp_world': actual.tolist(),
        'height_policy': height['height_source'],
        'required_clearance_m': reference['required_clearance_m']}
