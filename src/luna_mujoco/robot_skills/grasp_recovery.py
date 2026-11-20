"""Explicit empty-grasp recovery; never infer an object's identity or location."""
from copy import deepcopy


def empty_grasp_evidence(runner, context):
    result = context.get('grasp_verification', {})
    capture = context.get('grasp_verification_sample') or {}
    judgment = result.get('holding_judgment') or {}
    evidence = judgment.get('evidence') or {}
    execution = context.get('grasp_execution') or {}
    owner = {'owner_id': runner.run_id, 'target_id': runner.state.get('target_id'),
             'arm': runner.state.get('arm'), 'generation': runner.state.get('generation'),
             'motion_epoch': runner.motion_epoch, 'action_epoch': runner.observation_epoch}
    source = evidence.get('source')
    visual = result.get('visual_evidence') or {}
    if (result.get('grasp_state') != 'empty' or result.get('success') is not False
            or result.get('status') != 'no' or not result.get('invocation_id')
            or judgment.get('judgments', {}).get('object_held', {}).get('status') != 'no'
            or source not in {'openai_decisions', 'measured_gripper_position'}
            or source == 'openai_decisions' and not evidence.get('request_hash')
            or not capture.get('observation_id') or not owner['target_id'] or owner['arm'] not in ('A', 'B')
            or any(row.get('observation_id') != capture['observation_id'] for row in (result, evidence, visual))
            or any(row.get('generation') != owner['generation'] for row in (result, capture, visual))
            or evidence.get('generation', owner['generation']) != owner['generation']
            or any(result.get(k) != owner[k] for k in ('arm', 'target_id', 'action_epoch'))
            or execution.get('completed') is not True or any(execution.get(k) != v for k, v in owner.items())
            or capture.get('time', 0) < execution.get('completed_feedback_time_s', float('inf'))
            or any((value or {}).get('attached') for value in (context.get('holding'), runner.holding_state))):
        return None
    return {**owner, 'failure_invocation_id': result['invocation_id'], 'source': source,
            'observation_id': capture['observation_id'], 'request_hash': evidence.get('request_hash')}


async def recover_failed_grasp(runner, context):
    """Reconcile an empty failed action; all subsequent motion stays in the graph."""
    failure = runner.state.get('partial_execution_failure') or {}
    proof = empty_grasp_evidence(runner, context)
    if proof is None:
        raise ValueError('grasp_recovery_empty_hand_unconfirmed')
    key = (proof['target_id'], runner.task_progress.transfer_index(proof['target_id']))
    retries = context.setdefault('empty_grasp_retries', set())
    if key in retries:
        raise ValueError('empty_grasp_retry_exhausted')
    retries.add(key)
    # No actuator command, capture, identity judgment or map publication here.
    # Clear only the provisional carrying model; an attached object is rejected above.
    context['last_empty_grasp'] = (proof, context['grasp_verification_sample'])
    context.pop('holding', None)
    runner.holding_state = None
    runner.spatial_memory.invalidate([proof['target_id']])
    for key in ('plan', 'plan_binding', 'plan_observation_id', 'grasps', 'grasps_observation_id',
                'grasp_verification', 'grasp_verification_sample', 'grasp_execution', 'tcp_before_lift', 'points_before_lift'):
        context.pop(key, None)
    runner.state['facts'].update(recovery_needed=False, grasp_held=False, attached=False,
        lifted=False, grasp_completed=False, plan_ready=False, grasps_ready=False, verified=False,
        observed=False, scene_ready=False, observation_motion_scene_ready=False, observation_map_ready=False)
    return {'success': True, 'empty_grasp_evidence': deepcopy(proof),
            'failed_action_receipt': deepcopy(failure), 'physical_action_executed': False,
            'next_step_owner': 'responses_graph', 'old_trajectory_reused': False}
