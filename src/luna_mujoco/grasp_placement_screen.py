"""Finite placement endpoint screening using only the fixed part of the gripper."""
import hashlib
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .geometry import front_tcp_offset, rigid_matrix
from .placement_candidates import subdivision_cell_centers
from .release_clearance import CLEARANCE_COST_EPSILON

MAX_QUERY_SPHERES = 262144
MAX_XY_SAMPLES = 28
HEIGHT_STEP_M = .005
MAX_OBJECT_FLOOR_GAP_M = .05


@lru_cache(maxsize=2)
def fixed_gripper_spheres(arm):
    """Authored TCP and base spheres in front coordinates, independent of closure."""
    if arm not in ('A', 'B'):
        raise ValueError('grasp_placement_requires_selected_arm')
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / 'resources/curobo/marvin_gui_dual.yml').read_text(encoding='utf-8'))
    kin = config['robot_cfg']['kinematics']
    prefix, tool = ('gripL_', 'TCP_Link_L') if arm == 'A' else ('gripR_', 'TCP_Link_R')
    mount = ET.parse(root / 'assets/marvin_urdf/marvin_with_grippers.urdf').getroot().find(
        f"joint[@name='{prefix}mount']")
    if (mount is None or mount.get('type') != 'fixed' or mount.find('parent').get('link') != tool
            or mount.find('child').get('link') != prefix + 'Base'):
        raise ValueError('grasp_placement_requires_fixed_gripper_mount')
    origin = mount.find('origin')
    translation = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    spheres = []
    for link in (tool, prefix + 'Base'):
        if link not in kin['collision_link_names']:
            raise ValueError('grasp_placement_fixed_geometry_not_enabled')
        buffer = kin['collision_sphere_buffer'].get(link, 0.)
        for sphere in kin['collision_spheres'][link]:
            center = np.asarray(sphere['center'], dtype=float)
            if link != tool:
                center = rotation @ center + translation
            center = center - front_tcp_offset(arm)
            spheres.append([*center, float(sphere['radius']) + buffer])
    values = np.asarray(spheres, dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 3] <= 0):
        raise ValueError('grasp_placement_requires_positive_fixed_spheres')
    values.setflags(write=False)
    return values


def placement_endpoint_poses(tcp, object_points, reference, settings, *, uncertainty_m):
    """Retain the grasp rotation and move the entire current model over the region."""
    transform = rigid_matrix(tcp)
    points = np.asarray(object_points, dtype=float)
    floor = float(reference['floor_z_m'])
    maximum = min(MAX_OBJECT_FLOOR_GAP_M, float(settings['place_max_release_gap_m']))
    clearance = float(settings['trajectory_collision_clearance_m'])
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
            or not np.isfinite(points).all() or not np.isfinite([floor, maximum, clearance, uncertainty_m]).all()
            or maximum <= 0 or min(clearance, uncertainty_m) < 0):
        raise ValueError('grasp_placement_requires_finite_model_and_region')
    # The whole grasp attitude is preserved, so these world-oriented offsets
    # remain constant. No synthetic carrying binding or closure is needed.
    relative = points - transform[:3, 3]
    polygon = reference['region_polygon_world_xy_m']
    xy = np.vstack([subdivision_cell_centers(polygon, offsets_world_xy_m=relative[:, :2], cell_count=count)
                    for count in (4, 8, 16)])
    hull = ConvexHull(polygon)
    extent = np.max(relative[:, :2] @ hull.equations[:, :2].T, axis=0)
    distances = xy @ hull.equations[:, :2].T + hull.equations[:, 2] + extent
    xy = xy[np.all(distances <= -clearance - uncertainty_m + 1e-9, axis=1)]
    available = len(xy)
    gaps = np.arange(HEIGHT_STEP_M, maximum + 1e-10, HEIGHT_STEP_M)
    if maximum >= HEIGHT_STEP_M and (not len(gaps) or maximum - gaps[-1] > 1e-10):
        gaps = np.append(gaps, maximum)
    positions = np.empty((len(xy) * len(gaps), 3))
    positions[:, :2] = np.repeat(xy, len(gaps), axis=0)
    positions[:, 2] = floor - relative[:, 2].min() + np.tile(gaps, len(xy))
    poses = np.broadcast_to(transform, (len(positions), 4, 4)).copy()
    poses[:, :3, 3] = positions
    return poses, {'available_xy_sample_count': available, 'xy_sample_count': len(xy),
        'sampled_pose_count': len(poses), 'object_floor_gaps_m': gaps.tolist(),
        'floor_z_m': floor, 'maximum_object_floor_gap_m': maximum, 'xy_spacing_m': None,
        'xy_subdivision_levels': [4, 8, 16],
        'footprint_clearance_m': clearance + uncertainty_m,
        'whole_destination_infeasible': False}


def _query_result(response, world, scene_token, clearance):
    digest = hashlib.sha256(np.asarray(world, dtype='<f8').tobytes()).hexdigest()
    if (not isinstance(response, dict) or response.get('ok') is not True
            or response.get('scene_token') != scene_token or response.get('attached') is not False
            or response.get('query_sha256') != digest
            or type(response.get('required_clearance_m')) not in (int, float)
            or response['required_clearance_m'] != clearance):
        raise ValueError('grasp_placement_query_binding_changed')
    free = response.get('pose_collision_free')
    if not isinstance(free, list) or len(free) != len(world) or any(type(value) is not bool for value in free):
        raise ValueError('grasp_placement_query_invalid_pose_results')
    values = response.get('maximum_world_collision_cost')
    if not isinstance(values, list) or any(type(value) not in (int, float) for value in values):
        raise ValueError('grasp_placement_query_invalid_collision_costs')
    costs = np.asarray(values, dtype=float)
    if (costs.shape != (len(world),) or not np.isfinite(costs).all() or np.any(costs < 0)
            or not np.array_equal(free, costs <= CLEARANCE_COST_EPSILON)):
        raise ValueError('grasp_placement_query_invalid_collision_costs')
    return np.asarray(free), costs, digest


def screen_grasp_placement(candidates, object_points, reference, spheres, query, *, scene_token,
                           settings, uncertainty_m):
    """Reject only within this finite endpoint budget; query receives world spheres."""
    spheres = np.asarray(spheres, dtype=float)
    if (spheres.ndim != 2 or spheres.shape[1:] != (4,) or not len(spheres)
            or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)
            or len(spheres) > MAX_QUERY_SPHERES):
        raise ValueError('grasp_placement_requires_positive_fixed_spheres')
    capacity = MAX_QUERY_SPHERES // len(spheres)
    clearance = float(settings['trajectory_collision_clearance_m'])
    rows, endpoint_groups, pending, pending_count, queries = [], [], [], 0, []

    def flush():
        nonlocal pending_count
        if not pending:
            return
        world = np.concatenate([entry[2] for entry in pending])
        free, costs, digest = _query_result(query(world), world, scene_token, clearance)
        queries.append({'query_sha256': digest, 'pose_count': len(world), 'sphere_count': len(world) * len(spheres)})
        offset = 0
        for row, poses, group in pending:
            result = free[offset:offset + len(group)]
            group_costs = costs[offset:offset + len(group)]
            row['checked_pose_count'] += len(group)
            best = float(group_costs.min())
            previous = row['minimum_maximum_world_collision_cost']
            row['minimum_maximum_world_collision_cost'] = best if previous is None else min(previous, best)
            if np.any(result) and not row['accepted']:
                index = int(np.flatnonzero(result)[0])
                row.update(accepted=True, reason='fixed_gripper_clear_at_sampled_endpoint',
                    selected_endpoint=poses[index].tolist())
            offset += len(group)
        pending.clear()
        pending_count = 0

    for candidate in candidates:
        poses, sampling = placement_endpoint_poses(candidate['T_world_tcp'], object_points, reference, settings,
            uncertainty_m=uncertainty_m)
        row = {'candidate_index': candidate['candidate_index'], **sampling, 'accepted': False,
            'reason': ('fixed_gripper_collides_at_all_sampled_endpoints' if len(poses)
                else 'no_sampled_endpoint_inside_current_region'),
            'selected_endpoint': None, 'minimum_maximum_world_collision_cost': None, 'checked_pose_count': 0,
            'T_world_tcp_at_grasp': np.asarray(candidate['T_world_tcp']).tolist()}
        rows.append(row)
        endpoint_groups.append(poses)
    # Batch a small prefix across grasps first. An accepted grasp needs one
    # collision-free witness, not the rest of its sampled endpoints. Rejected
    # grasps still exhaust the identical finite domain and all binding checks.
    for stop in (1, 8, max((len(poses) for poses in endpoint_groups), default=0)):
        for row, poses in zip(rows, endpoint_groups, strict=True):
            start = row['checked_pose_count']
            limit = min(stop, len(poses))
            while start < limit and not row['accepted']:
                count = min(capacity - pending_count, limit - start)
                group = poses[start:start + count]
                world = np.broadcast_to(spheres, (count, *spheres.shape)).copy()
                world[:, :, :3] = spheres[:, :3] @ group[0, :3, :3].T + group[:, None, :3, 3]
                pending.append((row, group, world))
                pending_count += count
                start += count
                if pending_count == capacity:
                    flush()
        flush()
    kept = [candidate for candidate, row in zip(candidates, rows, strict=True) if row['accepted']]
    return kept, {'applied': True, 'scope': 'fixed_gripper_at_finite_placement_endpoints',
        'scene_token': scene_token, 'attached': False, 'input_count': len(candidates),
        'accepted_count': len(kept), 'rejected_count': len(candidates) - len(kept),
        'checked_links': 'selected_arm_tcp_and_fixed_gripper_base', 'fixed_sphere_count': len(spheres),
        'fixed_spheres_sha256': hashlib.sha256(spheres.astype('<f8').tobytes()).hexdigest(),
        'object_geometry_sha256': hashlib.sha256(np.asarray(object_points, dtype='<f8').tobytes()).hexdigest(),
        'required_clearance_m': clearance, 'distance_reporting': 'activation_cost_only',
        'candidate_evidence': rows, 'queries': queries, 'trajectory_checked': False,
        'query_schedule': 'batched_prefix_1_8_then_remaining_until_first_clear_endpoint',
        'attachment_checked': False, 'moving_fingers_checked': False,
        'whole_destination_infeasible': False, 'placement_authorized': False}
