"""Compare failed placement inputs, independently of result IDs and camera pixels."""
from copy import deepcopy

import numpy as np

# Equality for retry eligibility only; never a collision or execution tolerance.
CONSISTENCY_ATOL = 1e-6
_OFFSETS = ('place_approach_offset_m', 'place_retreat_offset_m')
_MOTION = ('approach_axis', 'approach_offset_m', 'approach_in_tool_frame',
           'lift_axis', 'lift_offset_m', 'lift_in_tool_frame',
           'trajectory_collision_clearance_m', 'optimizer_collision_activation_distance_m')
_HOLDING = ('arm', 'owner_id', 'instance_id', 'frame', 'state', 'verified', 'attached',
            'points_tool_m', 'uncertainty_m', 'minimum_partition_size_m', 'voxel_size_m',
            'margin_m', 'sphere_budget', 'obb_tool', 'cad_tool')


def _same_geometry(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_geometry(left[key], right[key]) for key in left)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        return a.shape == b.shape and bool(np.allclose(a, b, atol=CONSISTENCY_ATOL, rtol=0))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_geometry(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, (int, float)):
        return bool(np.isfinite(left) and np.isfinite(right) and abs(left - right) <= CONSISTENCY_ATOL)
    return left == right


def placement_attempt(context, state, observation, requirements, *, node_id=None):
    """Snapshot the planner's effective inputs; provenance is retained separately."""
    resolution = context.get('placement_resolution')
    if not resolution or not resolution.get('candidates'):
        return None
    active = state.get('active_policy_node') or {}
    candidates = []
    for candidate in resolution['candidates']:
        settings, payload = candidate['settings'], candidate['payload']
        candidates.append({'settings': {key: value for key, value in settings.items() if key not in _OFFSETS},
            'geometry': {'T_world_tcp': candidate['goal']['T_world_tcp'],
                'offsets': {key: settings[key] for key in _OFFSETS if key in settings},
                'motion': {key: payload[key] for key in _MOTION if key in payload},
                'binding': candidate['binding'], 'carrying_binding': payload.get('carrying_binding')}})
    scene = context['scene']
    return deepcopy({'scope': {**{key: state.get(key) for key in ('target_id', 'destination_id', 'arm')},
            'graph_id': active.get('graph_id'), 'subgraph_id': active.get('subgraph_id'),
            'node_id': node_id or active.get('node_id')},
        'prerequisites': {key: state['facts'].get(key) for key in requirements},
        'scene': {key: scene.get(key) for key in ('scene_token', 'generation', 'collision_layer_bounds',
                    'exclusion_state', 'excluded_object_ids', 'disabled_obstacle_names', 'table_collision_enabled')},
        'holding': {key: context['holding'][key] for key in _HOLDING if key in context['holding']},
        'robot': {arm: {key: row[key] for key in ('joints_rad', 'gripper_position_rad')}
                  for arm, row in observation['arms'].items()},
        'settings': context['planning_settings'], 'candidates': candidates,
        'provenance': {'resolution_id': resolution.get('resolution_id'),
            'input_digest': resolution.get('input_digest'),
            'candidate_ids': [candidate['candidate_id'] for candidate in resolution['candidates']]}})


def _same_attempt(previous, current):
    for key in ('scope', 'prerequisites', 'scene', 'settings'):
        if previous[key] != current[key]:
            return False
    if not all(_same_geometry(previous[key], current[key]) for key in ('holding', 'robot')):
        return False
    for candidate in current['candidates']:
        if not any(prior['settings'] == candidate['settings']
                   and _same_geometry(prior['geometry'], candidate['geometry'])
                   for prior in previous['candidates']):
            return False
    return True  # Reordering, duplicating or dropping exhausted goals supplies no new goal.


def record_placement_failure(context, attempt, details):
    if attempt is not None:
        context.setdefault('failed_placement_attempts', []).append({
            'inputs': deepcopy(attempt), 'candidate_failures': deepcopy(details.get('candidate_failures', []))})


def placement_retry_block(context, attempt):
    """Use the existing rejected-proposal contract without choosing a recovery."""
    if attempt is None:
        return None
    for failed in context.get('failed_placement_attempts', []):
        previous = failed['inputs']
        if _same_attempt(previous, attempt):
            return {'action': 'retry', 'skill': None, 'arguments': None,
                'reason': 'identical_evidence_and_operation_no_progress',
                'placement_evidence': {'reason': 'identical_placement_inputs_no_progress',
                    'previous': previous['provenance'], 'current': attempt['provenance'],
                    'candidate_failures': failed['candidate_failures'], 'consistency_atol': CONSISTENCY_ATOL,
                    'unchanged_inputs': ['candidate_goals_and_constraints', 'published_scene',
                        'held_geometry', 'both_arm_and_gripper_positions', 'planning_settings', 'prerequisites'],
                    'goal_poses_world': [candidate['geometry']['T_world_tcp']
                                         for candidate in attempt['candidates']],
                    'scope': attempt['scope'], 'scene_token': attempt['scene']['scene_token'],
                    'whole_destination_infeasible': False}}
    return None
