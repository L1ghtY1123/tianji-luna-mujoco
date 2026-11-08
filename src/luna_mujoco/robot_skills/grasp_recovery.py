"""Explicit empty-grasp recovery; never infer an object's identity or location."""
from copy import deepcopy

import numpy as np

EMPTY_GRASP_ERRORS = {'grasp_not_held_after_lift', 'grasp_closure_limit_reached_without_holding'}


def empty_grasp_feedback(runner, context, failure):
    step = failure.get('partial_execution', {})
    code = str(failure.get('error_code') or failure.get('reason', '')).partition(':')[0]
    if (step.get('skill') != 'execute_grasp' or not step.get('motor_commands')
            or code not in EMPTY_GRASP_ERRORS):
        return None
    arm, target = failure.get('arm'), failure.get('target_id')
    if arm not in ('A', 'B') or not target or arm != runner.state.get('arm') or target != runner.state.get('target_id'):
        return None
    snapshot = runner.simulation.robot_snapshot()
    if snapshot['generation'] != runner.state['generation'] or failure.get('generation') != snapshot['generation']:
        return None
    feedback = snapshot['arms'][arm]
    limits = np.asarray(feedback.get('gripper_limits_rad', []), dtype=float)
    command = feedback.get('gripper_command_position_rad')
    # A bounded closing command reaching its target does not prove an empty hand.
    if (limits.shape != (2,) or not np.isfinite(limits).all() or type(command) not in (int, float)
            or not np.isfinite(command) or abs(command - limits[1]) > 1e-6):
        return None
    result = runner.position_verdict(feedback, context)
    if result.get('status') != 'empty':
        return None
    if any((value or {}).get('attached') for value in (context.get('holding'), runner.holding_state)):
        return None
    return {'target_id': target, 'arm': arm, 'generation': snapshot['generation'],
            'failure_invocation_id': step.get('invocation_id'), 'feedback_time': snapshot['time'],
            'source': 'full_close_mechanical_feedback', 'feedback': result}


async def recover_failed_grasp(runner, context):
    """Reconcile an empty failed action; all subsequent motion stays in the graph."""
    failure = runner.state.get('partial_execution_failure') or {}
    proof = empty_grasp_feedback(runner, context, failure)
    if proof is None or not proof['failure_invocation_id']:
        raise ValueError('grasp_recovery_empty_hand_unconfirmed')
    # No actuator command, capture, identity judgment or map publication here.
    # Clear only the provisional carrying model; an attached object is rejected above.
    context.pop('holding', None)
    runner.holding_state = None
    runner.spatial_memory.invalidate([proof['target_id']])
    for key in ('plan', 'plan_binding', 'plan_observation_id', 'grasps', 'grasps_observation_id',
                'grasp_verification', 'tcp_before_lift', 'points_before_lift'):
        context.pop(key, None)
    runner.state['facts'].update(recovery_needed=False, grasp_held=False, attached=False,
        lifted=False, grasp_completed=False, plan_ready=False, grasps_ready=False, verified=False,
        observed=False, scene_ready=False, observation_motion_scene_ready=False, observation_map_ready=False)
    return {'success': True, 'empty_grasp_evidence': deepcopy(proof),
            'failed_action_receipt': deepcopy(failure), 'physical_action_executed': False,
            'next_step_owner': 'responses_graph', 'old_trajectory_reused': False}
