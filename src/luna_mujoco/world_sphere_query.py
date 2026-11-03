"""Bounded, read-only world collision queries for pre-grasp endpoint screening."""
import hashlib

import numpy as np

from .release_clearance import CLEARANCE_COST_EPSILON

MAX_QUERY_SPHERES = 262144


def prepare_query(payload, scene):
    if payload.get('carrying_binding') is not None:
        raise ValueError('world sphere screening cannot attach an object')
    if payload.get('arm') not in ('A', 'B'):
        raise ValueError('world sphere screening requires arm A or B')
    for name in ('scene_token', 'observation_id', 'generation'):
        if scene.get(name) is None or payload.get(name) != scene[name]:
            raise ValueError(f'world sphere screening has stale {name}')
    if type(payload['generation']) is not int:
        raise ValueError('generation must be an integer')
    for name in ('start_joints_deg', 'other_joints_deg'):
        joints = np.asarray(payload.get(name), dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f'{name} requires seven finite joints')
    feedback = payload.get('gripper_joint_positions_rad')
    if (not isinstance(feedback, dict) or set(feedback) != {'A', 'B'}
            or not np.isfinite(np.asarray(list(feedback.values()), dtype=float)).all()):
        raise ValueError('world sphere screening requires finite A and B gripper feedback')
    raw = np.asarray(payload.get('spheres_world_m'))
    if raw.dtype.kind not in 'fiu':
        raise ValueError('spheres must contain numeric coordinates and radii')
    spheres = np.asarray(raw, dtype='<f8')
    if (spheres.ndim != 3 or spheres.shape[-1] != 4
            or not spheres.shape[0] or not spheres.shape[1]
            or spheres.shape[0] * spheres.shape[1] > MAX_QUERY_SPHERES
            or not np.isfinite(spheres).all() or np.any(spheres[..., 3] <= 0)):
        raise ValueError('invalid or unbounded world sphere query')
    return spheres


def query_result(spheres, costs, scene_token, clearance):
    costs = np.asarray(costs, dtype=float)
    if costs.shape != spheres.shape[:2] or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError('invalid world sphere collision costs')
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError('invalid world sphere clearance')
    maximum = costs.max(axis=1)
    return {'ok': True, 'scene_token': scene_token, 'attached': False,
        'scope': 'world_spheres_only', 'trajectory_checked': False,
        'pose_collision_free': (maximum <= CLEARANCE_COST_EPSILON).tolist(),
        'maximum_world_collision_cost': maximum.tolist(),
        'required_clearance_m': float(clearance),
        'query_sha256': hashlib.sha256(np.asarray(spheres, dtype='<f8').tobytes()).hexdigest()}
