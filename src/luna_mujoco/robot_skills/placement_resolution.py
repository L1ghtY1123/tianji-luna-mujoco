"""Bounded numerical candidates and provenance, independent of task policy."""
import hashlib
import json
from copy import deepcopy

import numpy as np

from ..carrying import held_collision_binding

RETRYABLE_PLACEMENT_ERRORS = {'CUROBO_TRAJOPT_FAILED', 'CUROBO_TRAJECTORY_COLLISION',
    'CUROBO_PLANNING_FAILED', 'CUROBO_NO_VALID_TRAJECTORY', 'CUROBO_GRASP_FAILED',
    'execution_collision', 'release_height_no_valid_goal', 'release_height_no_held_ik_goal', 'placement_configuration_conflict',
    'place_goal_workspace_rejected', 'placement_release_above_container', 'placement_release_too_high'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
        default=lambda item: item.tolist()).encode()).hexdigest()


def resolution_input(observation, target_id, arm, scene, holding, destination_id, request, settings):
    return {'target_id': target_id, 'destination_id': destination_id, 'arm': arm,
        'observation_id': observation['observation_id'], 'generation': observation['generation'],
        'scene_token': scene['scene_token'], 'placement_request': request, 'planning_settings': settings,
        'robot_feedback': {name: {key: feedback[key] for key in ('joints_rad', 'gripper_position_rad')}
                           for name, feedback in observation['arms'].items()},
        'occupancy_bounds': scene.get('collision_layer_bounds'),
        'carrying_binding': held_collision_binding(holding, scene, arm, target_id),
        'holding': {key: holding.get(key) for key in ('instance_id', 'arm', 'owner_id', 'frame',
            'verified', 'attached', 'points_tool_m', 'uncertainty_m', 'measurement_provenance',
            'grasp_support_reference', 'attachment_evidence')},
        'destination': next(item for item in observation['objects'] if item['id'] == destination_id)}


def validate_resolution(resolution, inputs, *, require_same_feedback=True):
    """Validate immutable pose bindings; stale feedback never authorizes a cached path."""

    unsigned = {key: value for key, value in resolution.items() if key != 'resolution_id'}
    check = feedback_consistency(resolution.get('robot_feedback'), inputs['robot_feedback'])
    bound_inputs = {**inputs, 'robot_feedback': resolution.get('robot_feedback')}
    if (set(check['arms']) != {'A', 'B'} or (require_same_feedback and not check['consistent'])
            or resolution.get('input_digest') != digest(bound_inputs)
            or resolution.get('resolution_id') != digest(unsigned) or not resolution.get('candidates')):
        raise ValueError('placement_resolution_changed')
    for candidate in resolution['candidates']:
        reference = candidate.get('goal', {}).get('release_reference', {})
        if reference.get('height_source') != 'measured_object_height_and_support':
            raise ValueError('placement_resolution_requires_fixed_height')
    return check



FEEDBACK_CONSISTENCY_ATOL_RAD = 1e-6

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
