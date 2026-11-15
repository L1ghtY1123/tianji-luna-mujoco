"""Explicit empty-grasp recovery; never infer an object's identity or location."""
from copy import deepcopy


def empty_grasp_evidence(runner, context):
    result = context.get('grasp_verification', {})
    empty = result.get('holding_judgment', {})
    judgment = empty.get('judgments', {}).get('object_held', {})
    if (result.get('success') is not False or result.get('status') != 'no'
            or result.get('source') != 'current_rgb_decisions'
            or result.get('grasp_state') != 'empty'
            or judgment.get('status') != 'no'
            or any(result.get(key) != runner.state.get(key) for key in ('target_id', 'arm', 'generation'))
            or result.get('action_epoch') != runner.observation_epoch
            or any((value or {}).get('attached') for value in (context.get('holding'), runner.holding_state))):
        return None
    return {'target_id': result['target_id'], 'arm': result['arm'], 'generation': result['generation'],
        'failure_invocation_id': result['invocation_id'], 'source': 'current_rgb_decisions',
        'observation_id': result['observation_id'], 'visual_evidence': deepcopy(empty['evidence'])}


async def recover_failed_grasp(runner, context):
    """Reconcile an empty failed action; all subsequent motion stays in the graph."""
    failure = runner.state.get('partial_execution_failure') or {}
    proof = empty_grasp_evidence(runner, context)
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
