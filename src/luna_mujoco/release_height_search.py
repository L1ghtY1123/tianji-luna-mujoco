"""Configured release samples and held-IK evidence, without trajectory planning."""
import hashlib
import json
import time
from copy import deepcopy

import numpy as np

from .geometry import rigid_matrix
from .release_clearance import CLEARANCE_COST_EPSILON

HEIGHT_STEP_M = .001
IK_BATCH_SIZE = 100
# Same numerical feedback-consistency contract as runtime.placement_retry.
# This is not an IK, collision, servo or release acceptance tolerance.
FEEDBACK_CONSISTENCY_ATOL_RAD = 1e-6


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def geometry_points(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError('release_requires_complete_finite_object_geometry')
    return points


def object_floor_gap(points, pose, floor_z):
    transform = rigid_matrix(pose)
    return float(np.min((geometry_points(points) @ transform[:3, :3].T + transform[:3, 3])[:, 2]) - floor_z)


def height_search_bounds(pose, points, floor_z, maximum_gap):
    from .planning_settings import PlanningSettings

    transform = rigid_matrix(pose)
    points = geometry_points(points)
    maximum_gap = PlanningSettings(place_max_release_gap_m=maximum_gap).place_max_release_gap_m
    if not np.isfinite(floor_z):
        raise ValueError('release_object_floor_gap_bound_invalid')
    gap = object_floor_gap(points, transform, floor_z)
    increase = float(maximum_gap - gap)
    if increase < -1e-9:
        error = ValueError('placement_release_too_high')
        error.code = 'placement_release_too_high'
        error.details = {'object_floor_gap_m': gap, 'maximum_object_floor_gap_m': maximum_gap}
        raise error
    increase = max(0., increase)
    offsets = (np.arange(int(np.floor(increase / HEIGHT_STEP_M + 1e-9)) + 1) * HEIGHT_STEP_M).tolist()
    if increase - offsets[-1] > 1e-9:
        offsets.append(increase)
    return {'version': 1, 'initial_T_world_tcp': transform.tolist(), 'floor_z_m': float(floor_z),
        'maximum_object_floor_gap_m': float(maximum_gap), 'initial_object_floor_gap_m': gap,
        'maximum_height_increase_m': increase, 'resolution_m': HEIGHT_STEP_M,
        'height_offsets_m': offsets, 'object_geometry_point_count': len(points),
        'object_geometry_sha256': hashlib.sha256(points.astype('<f8').tobytes()).hexdigest(),
        'geometry_source': 'complete_bound_cad_support_vertices_or_obb_corners_or_archived_measurements'}


def height_positions(bounds):
    initial = rigid_matrix(bounds['initial_T_world_tcp'])[:3, 3]
    return [[float(initial[0]), float(initial[1]), float(initial[2] + offset)]
            for offset in bounds['height_offsets_m']]


def height_input_digest(payload):
    """Bind robot feedback, geometry provenance and settings, excluding derived goals."""
    return digest({key: payload[key] for key in ('arm', 'scene_token', 'start_joints_deg',
        'other_joints_deg', 'gripper_joint_positions_rad', 'carrying_binding', 'planning_settings')})


def feedback_consistency(previous, current):
    """Preserve both snapshots; ignore only the established numerical identity noise."""
    report = {'consistent': True, 'consistency_atol_rad': FEEDBACK_CONSISTENCY_ATOL_RAD,
              'reference_feedback': deepcopy(previous), 'current_feedback': deepcopy(current), 'arms': {}}
    if not isinstance(previous, dict) or not isinstance(current, dict) or set(previous) != {'A', 'B'} or set(current) != {'A', 'B'}:
        report['consistent'] = False
        return report
    for arm in ('A', 'B'):
        left, right = previous[arm], current[arm]
        a, b = np.asarray(left['joints_rad'], float), np.asarray(right['joints_rad'], float)
        fingers = np.asarray([left['gripper_position_rad'], right['gripper_position_rad']], float)
        if a.shape != (7,) or b.shape != (7,) or not np.isfinite(np.concatenate((a, b, fingers))).all():
            report['consistent'] = False
            return report
        joint_delta = float(np.max(np.abs(a - b)))
        gripper_delta = float(fingers[1] - fingers[0])
        report['arms'][arm] = {'maximum_joint_delta_rad': joint_delta, 'gripper_delta_rad': gripper_delta}
        report['consistent'] &= max(joint_delta, abs(gripper_delta)) <= FEEDBACK_CONSISTENCY_ATOL_RAD
    return report


def validate_search(search, points, payload):
    bounds = search['bounds']
    pose = rigid_matrix(bounds['initial_T_world_tcp'])
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_quat(np.asarray(payload['goal_quaternion_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    settings = payload['planning_settings']
    expected = height_search_bounds(pose, points, bounds['floor_z_m'], settings['place_max_release_gap_m'])
    if (bounds != expected or search['arm'] != payload['arm']
            or search['input_sha256'] != height_input_digest(payload)
            or not np.allclose(pose[:3, 3], payload['goal_position_m'], atol=1e-9, rtol=0)
            or not np.allclose(pose[:3, :3], rotation, atol=1e-9, rtol=0)
            or any(search[key] != settings[key] for key in
                   ('ik_position_threshold_m', 'ik_rotation_threshold_rad'))):
        raise ValueError('release_height_search_binding_changed')
    positions = height_positions(bounds)
    rows = search['workspace']
    if len(rows) != len(positions) or any(row.get('position_world_m') != position
            or type(row.get('reachable')) is not bool for row, position in zip(rows, positions, strict=True)):
        raise ValueError('release_height_workspace_binding_changed')


class HeldReleaseHeightError(ValueError):
    code = 'release_height_no_held_ik_goal'

    def __init__(self, details):
        super().__init__(self.code)
        self.details = details


def _ik_rejection(item, pose, points, search, scene_token):
    """Check public success and achieved FK; never interpret failed IK as collision."""
    if type(item.get('success')) is not bool:
        raise ValueError('release_height_ik_result_invalid')
    if not item['success']:
        return 'held_ik_not_solved', None
    achieved = item.get('achieved_tool_pose', {})
    joints = np.asarray(item.get('joints_rad'), float)
    if (joints.shape != (7,) or not np.isfinite(joints).all()
            or achieved.get('source') != 'curobo_public_compute_kinematics'
            or achieved.get('frame') != 'world' or achieved.get('arm') != search['arm']
            or achieved.get('tool_frame') != 'gripper_front_' + search['arm']
            or achieved.get('scene_token') != scene_token
            or achieved.get('joints_rad') != item['joints_rad']
            or achieved.get('joint_names') != item.get('joint_names')):
        raise ValueError('release_height_achieved_fk_binding_changed')
    actual = rigid_matrix(achieved['T_world_tcp'])
    position_error = float(np.linalg.norm(actual[:3, 3] - pose[:3, 3]))
    rotation_error = float(np.arccos(np.clip((np.trace(actual[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)))
    metrics = [item['position_error_m'], item['rotation_error_rad'], position_error, rotation_error]
    if not np.isfinite(metrics).all() or min(metrics) < 0:
        raise ValueError('release_height_ik_metrics_invalid')
    if (max(item['position_error_m'], position_error) > search['ik_position_threshold_m'] + 1e-9
            or max(item['rotation_error_rad'], rotation_error) > search['ik_rotation_threshold_rad'] + 1e-9):
        return 'held_ik_pose_tolerance_exceeded', None
    gap = object_floor_gap(points, actual, search['bounds']['floor_z_m'])
    if gap > search['bounds']['maximum_object_floor_gap_m'] + 1e-9:
        return 'achieved_object_floor_gap_exceeded', gap
    return None, gap


def select_held_height(world, search, points, solve_ik):
    """Each world/workspace-qualified sample enters one public held-IK batch at most."""
    from .demo_placement import ReleaseHeightError

    proof = deepcopy(world)
    proof.update(search=deepcopy(search), scope='held_ik_release_endpoint', trajectory_checked=False,
                 ik_batches=[], held_ik_sample_count=0)
    bounds = search['bounds']
    base = rigid_matrix(bounds['initial_T_world_tcp'])
    samples = proof['tested_heights']
    eligible = []
    first_clear = None
    for index, (row, workspace) in enumerate(zip(samples, search['workspace'], strict=True)):
        row['workspace'] = deepcopy(workspace)
        row['object_floor_gap_m'] = bounds['initial_object_floor_gap_m'] + bounds['height_offsets_m'][index]
        row['held_ik'] = None
        if row['conflicting_sphere_indices']:
            row['rejection'] = 'tool_world_collision'
        else:
            if first_clear is None:
                first_clear = {'sample_index': index, 'z_m': row['z_m'],
                    'height_increase_m': bounds['height_offsets_m'][index],
                    'world_collision_costs': deepcopy(row['world_collision_costs'])}
            if not workspace['reachable']:
                row['rejection'] = 'workspace_rejected'
            else:
                row['rejection'] = 'held_ik_not_evaluated_above_selected'
                eligible.append(index)
    proof['first_world_clear_height'] = first_clear
    proof['world_clear_sample_count'] = sum(not row['conflicting_sphere_indices'] for row in samples)
    if first_clear is None:
        raise ReleaseHeightError(proof)
    selected = None
    # The lowest clear height commonly works. Do not evaluate the rest of
    # the interval before trying it; expand only after actual IK rejection.
    start = 0
    while start < len(eligible):
        indices = eligible[start:start + (1 if start == 0 else IK_BATCH_SIZE)]
        start += len(indices)
        poses = []
        for index in indices:
            pose = base.copy()
            pose[2, 3] = samples[index]['z_m']
            poses.append(pose.tolist())
        began = time.monotonic()
        response = solve_ik(poses)
        wall_s = time.monotonic() - began
        if (response.get('attached') is not True or response.get('scene_token') != proof['scene_token']
                or response.get('collision_evidence', {}).get('optimizer_attachment') is not True
                or len(response.get('items', [])) != len(indices)
                or any(response.get('effective_config', {}).get(key) != search[key]
                       for key in ('ik_position_threshold_m', 'ik_rotation_threshold_rad'))):
            raise ValueError('release_height_held_ik_evidence_invalid')
        proof['ik_batches'].append({'sample_indices': indices, 'sample_count': len(indices),
            'batch_capacity': response.get('batch_capacity', IK_BATCH_SIZE), 'wall_s': wall_s,
            'padding_slots_ignored': response.get('batch_capacity', IK_BATCH_SIZE) - len(indices),
            'solve_time_s': response.get('solve_time_s'), 'effective_config': deepcopy(response['effective_config']),
            'collision_evidence': deepcopy(response['collision_evidence'])})
        proof['held_ik_sample_count'] += len(indices)
        for local_index, (index, pose, item) in enumerate(zip(indices, poses, response['items'], strict=True)):
            if item.get('candidate_index') != local_index:
                raise ValueError('release_height_ik_sample_order_changed')
            row = samples[index]
            row['held_ik'] = deepcopy(item)
            rejection, gap = _ik_rejection(item, np.asarray(pose), points, search, proof['scene_token'])
            row['rejection'] = rejection
            row['achieved_object_floor_gap_m'] = gap
            if rejection is None and selected is None:
                selected = index
        if selected is not None:
            break
    if selected is None:
        raise HeldReleaseHeightError(proof)
    row = samples[selected]
    proof.update(status='first_world_and_held_ik_valid_sample', selected_sample_index=selected,
        selected_goal_position_m=[*base[:2, 3].tolist(), row['z_m']],
        height_increase_m=bounds['height_offsets_m'][selected],
        selected_object_floor_gap_m=row['object_floor_gap_m'],
        achieved_object_floor_gap_m=row['achieved_object_floor_gap_m'],
        final_world_collision_costs=row['world_collision_costs'])
    return proof


def valid_held_height_resolution(reference, expected, minimum_z, points, scene_token):
    """Replay the acceptance predicates against the original geometry and bound samples."""
    try:
        proof = reference['height_resolution']
        search = reference['height_search']
        bounds = search['bounds']
        initial = rigid_matrix(bounds['initial_T_world_tcp'])
        if (proof['scope'] != 'held_ik_release_endpoint' or proof['search'] != search
                or proof['status'] != 'first_world_and_held_ik_valid_sample'
                or proof['scene_token'] != scene_token
                or proof['maximum_height_increase_m'] != bounds['maximum_height_increase_m']
                or proof['resolution_m'] != bounds['resolution_m']
                or bounds != height_search_bounds(initial, points, reference['floor_z_m'],
                                                   bounds['maximum_object_floor_gap_m'])
                or not np.allclose(initial[:3, 3], [*expected[:2, 3], minimum_z], atol=1e-9, rtol=0)
                or not np.allclose(initial[:3, :3], expected[:3, :3], atol=1e-9, rtol=0)):
            return False
        positions = height_positions(bounds)
        samples = proof['tested_heights']
        selected = proof['selected_sample_index']
        if (type(selected) is not int or not 0 <= selected < len(positions)
                or len(samples) != len(positions) or len(search['workspace']) != len(positions)
                or not np.allclose(expected[:3, 3], positions[selected], atol=1e-9, rtol=0)
                or proof['height_increase_m'] != bounds['height_offsets_m'][selected]
                or search['workspace'][selected]['reachable'] is not True
                or samples[selected]['conflicting_sphere_indices']
                or samples[selected]['rejection'] is not None):
            return False
        first_clear = None
        for index, (row, position, workspace) in enumerate(zip(samples, positions, search['workspace'], strict=True)):
            costs = np.asarray(row['world_collision_costs'], float)
            if (row['z_m'] != position[2] or costs.shape != (proof['attachment_sphere_count'] + proof['gripper_sphere_count'],)
                    or not np.isfinite(costs).all() or np.any(costs < 0)
                    or row['conflicting_sphere_indices'] != np.flatnonzero(costs > CLEARANCE_COST_EPSILON).tolist()
                    or row['maximum_world_collision_cost'] != float(costs.max())
                    or workspace['position_world_m'] != position or row['workspace'] != workspace
                    or type(workspace['reachable']) is not bool
                    or abs(row['object_floor_gap_m'] - bounds['initial_object_floor_gap_m']
                           - bounds['height_offsets_m'][index]) > 1e-9):
                return False
            if not row['conflicting_sphere_indices'] and first_clear is None:
                first_clear = index
            if index <= selected and not row['conflicting_sphere_indices'] and workspace['reachable']:
                pose = initial.copy()
                pose[:3, 3] = position
                rejection, gap = _ik_rejection(row['held_ik'], pose, points, search, scene_token)
                if (row['rejection'] != rejection or (rejection is None) != (index == selected)
                        or row.get('achieved_object_floor_gap_m') != gap):
                    return False
        first = proof['first_world_clear_height']
        return bool(first_clear is not None and first['sample_index'] == first_clear
            and first['z_m'] == positions[first_clear][2]
            and first['height_increase_m'] == bounds['height_offsets_m'][first_clear]
            and first['world_collision_costs'] == samples[first_clear]['world_collision_costs']
            and proof['final_world_collision_costs'] == samples[selected]['world_collision_costs']
            and proof['selected_goal_position_m'] == positions[selected]
            and proof['selected_object_floor_gap_m'] == samples[selected]['object_floor_gap_m']
            and proof['selected_object_floor_gap_m'] <= bounds['maximum_object_floor_gap_m'] + 1e-9
            and proof['achieved_object_floor_gap_m'] == samples[selected]['achieved_object_floor_gap_m'])
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError):
        return False
