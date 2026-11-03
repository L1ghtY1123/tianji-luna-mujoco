"""Observed upright OBB approximations shared by grasping and carrying.

GaP reference: graph-robots/open-robot-skills, tools/geometry/_impl.py::compute_obb
at 752dfbd96968d4d08379f8111765efbf6457800b (Apache-2.0).
Use its minimum-width XY orientation. A convex hull supplies candidate edges;
no noisy rasterization or percentile trimming is used for a collision envelope.
Every retained measurement is enclosed. The box is an approximation, not a
measurement of unseen surfaces, a container cavity, or a world-map replacement.
"""
import hashlib
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .geometry import rigid_matrix


def points_digest(points):
    return hashlib.sha256(np.asarray(points, dtype='<f8').tobytes()).hexdigest()


def validate_obb(obb, points, *, frame, instance_id):
    if (not isinstance(obb, dict) or obb.get('frame') != frame
            or obb.get('instance_id') != instance_id
            or obb.get('source') != 'observed_points_obb_approximation'
            or obb.get('points_sha256') != points_digest(points)):
        raise ValueError('obb_measurement_binding_changed')
    transform = rigid_matrix(obb['T_frame_obb'])
    half = np.asarray(obb['half_extents_m'], float)
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError('obb_requires_positive_half_extents')
    local = (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]
    if np.any(np.abs(local) > half + 1e-8):
        raise ValueError('obb_does_not_cover_observed_points')
    return transform, half


def observed_object_obb(target):
    points = np.asarray(target['points_world_m'], float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError('obb_requires_measured_points')
    if target.get('obb') is not None:
        validate_obb(target['obb'], points, frame='world', instance_id=target['id'])
        return deepcopy(target['obb'])
    xy = np.unique(points[:, :2], axis=0)
    try:
        boundary = xy[ConvexHull(xy).vertices]
    except QhullError:
        boundary = xy
    best_width, angle = np.inf, 0.
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0)):
        edge = end - start
        if np.linalg.norm(edge) < 1e-9:
            continue
        candidate = np.arctan2(edge[1], edge[0]) % np.pi
        transverse = xy @ np.array([-np.sin(candidate), np.cos(candidate)])
        width = float(np.ptp(transverse))
        if width < best_width - 1e-12:
            best_width, angle = width, candidate
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    local = points @ rotation
    lower, upper = local.min(0), local.max(0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation @ ((lower + upper) / 2)
    return {'frame': 'world', 'instance_id': target['id'],
        'source': 'observed_points_obb_approximation', 'fit_method': 'gap_minimum_width_xy_full_bounds',
        'T_frame_obb': transform.tolist(), 'half_extents_m': np.maximum((upper - lower) / 2, .0005).tolist(),
        'points_sha256': points_digest(points), 'measured_point_count': len(points),
        'observation_provenance': deepcopy(target.get('observation_provenance', {})),
        'unobserved_surfaces_are_measured': False}


def obb_surface_points(obb, count=4096):
    """Follow the official GraspGen mesh demo's public trimesh sampling API."""
    import trimesh

    mesh = trimesh.creation.box(extents=2 * np.asarray(obb['half_extents_m']),
                                transform=rigid_matrix(obb['T_frame_obb']))
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
    return points


def obb_in_tool(obb, tcp, points_tool):
    transform = np.linalg.inv(rigid_matrix(tcp)) @ rigid_matrix(obb['T_frame_obb'])
    # Float32 FK poses can pass rigid validation while retaining tiny scale/shear.
    # Keep an orthonormal OBB and enclose the whole transformed box, not only the
    # measured points. Otherwise transpose-based projection can reject its faces
    # and sphere radii can underestimate the transformed box volume.
    left, _, right = np.linalg.svd(transform[:3, :3])
    rotation = left @ right
    half = np.asarray(obb['half_extents_m'], float)
    half = np.maximum(half, np.abs(rotation.T @ transform[:3, :3]) @ half)
    transform[:3, :3] = rotation
    result = deepcopy(obb)
    result.update(frame='gripper_front',
        T_frame_obb=transform.tolist(), half_extents_m=half.tolist(),
        points_sha256=points_digest(points_tool))
    return result


def held_geometry_points(holding):
    """CAD support vertices or OBB corners; measured points stay separate."""
    if holding.get('cad_tool') is not None:
        from .carrying import cad_geometry_points

        return cad_geometry_points(holding)
    points = np.asarray(holding['points_tool_m'], float)
    obb = holding.get('obb_tool')
    if obb is None:  # Archived measured-surface bindings remain readable.
        return points
    transform, half = validate_obb(obb, points, frame='gripper_front', instance_id=holding['instance_id'])
    corners = np.array(list(product(*[[-h, h] for h in half])))
    return corners @ transform[:3, :3].T + transform[:3, 3]


def obb_covering_spheres(obb, *, maximum_spheres, minimum_size_m, padding_m):
    """Cover the entire box volume by subdividing cells and enclosing each cell."""
    half = np.asarray(obb['half_extents_m'])
    transform = rigid_matrix(obb['T_frame_obb'])
    cells = [(-half, half)]
    while len(cells) < maximum_spheres:
        index = max(range(len(cells)), key=lambda i: float(np.max(cells[i][1] - cells[i][0])))
        lower, upper = cells[index]
        axis = int(np.argmax(upper - lower))
        if upper[axis] - lower[axis] <= minimum_size_m:
            break
        middle = (lower[axis] + upper[axis]) / 2
        left_upper, right_lower = upper.copy(), lower.copy()
        left_upper[axis], right_lower[axis] = middle, middle
        cells[index:index + 1] = [(lower, left_upper), (right_lower, upper)]
    spheres = []
    for lower, upper in cells:
        center = transform[:3, :3] @ ((lower + upper) / 2) + transform[:3, 3]
        radius = float(np.linalg.norm((upper - lower) / 2) + padding_m)
        spheres.append([*center.tolist(), max(.002, radius)])
    return spheres
