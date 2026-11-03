"""Return to measured task-start joints; no task ordering or visual judgments."""
import numpy as np


def observation_motion_scene_ready(context):
    """Expose the existing motion-map checks as a planning prerequisite."""
    scene = context.get('scene')
    if not scene or context.get('scene_refresh_required'):
        return False
    transfer = context.get('pending_transfer')
    if transfer is not None and transfer.get('verified') is not True:
        return False
    if transfer is not None or context.get('processing_action'):
        return True
    return (scene.get('exclusion_state') == 'observation'
            and scene.get('observation_id') is not None
            and scene.get('observation_id') == context.get('observation', {}).get('observation_id'))


def prepare_empty_hand_motion_scene(context):
    if context.get('scene_refresh_required'):
        raise ValueError('observation_motion_requires_current_map')
    transfer = context.get('pending_transfer')
    if transfer is not None and transfer.get('verified') is not True:
        raise ValueError('return_requires_verified_release')
    if transfer is not None or context.get('processing_action'):
        context['use_retained_scene_for_return'] = True
    else:
        context.pop('use_retained_scene_for_return', None)


def initial_joint_reference(snapshot):
    joints = {arm: np.asarray(snapshot['arms'][arm]['joints_rad'], dtype=float) for arm in ('A', 'B')}
    if any(value.shape != (7,) or not np.isfinite(value).all() for value in joints.values()):
        raise ValueError('initial_pose_requires_finite_dual_arm_feedback')
    return {'generation': snapshot['generation'], 'feedback_time': snapshot.get('time'),
            'joints_rad': {arm: value.tolist() for arm, value in joints.items()},
            'source': 'task_start_joint_feedback'}


def initial_pose_verdict(reference, snapshot):
    errors = {}
    for arm in ('A', 'B'):
        goal = np.asarray(reference['joints_rad'][arm], dtype=float)
        actual = np.asarray(snapshot.get('arms', {}).get(arm, {}).get('joints_rad', []), dtype=float)
        errors[arm] = (float(np.rad2deg(np.max(np.abs(actual - goal))))
            if goal.shape == actual.shape == (7,) and np.isfinite([goal, actual]).all() else None)
    return {'success': snapshot.get('generation') == reference['generation']
                and all(error is not None and error <= 1. for error in errors.values()),
            'maximum_joint_error_deg': errors, 'tolerance_deg': 1.,
            'expected_generation': reference['generation'], 'actual_generation': snapshot.get('generation'),
            'source': 'actual_dual_arm_joint_feedback'}


async def return_to_initial_pose(runner, context):
    if context.get('holding') or getattr(runner, 'holding_state', None) is not None:
        raise ValueError('initial_pose_return_requires_empty_hands')
    transfer = context.get('pending_transfer')
    if transfer is not None and transfer.get('verified') is not True:
        raise ValueError('return_requires_verified_release')
    reference = context['initial_joint_reference']
    snapshot = runner.simulation.robot_snapshot()
    if snapshot.get('generation') != reference['generation']:
        raise ValueError('initial_pose_reference_stale')
    proof = initial_pose_verdict(reference, snapshot)
    if proof['success']:
        return {'success': True, 'already_at_initial_pose': True, 'pose_verification': proof}
    # Share the checked dual-arm joint planner, never replay an earlier path.
    prepare_empty_hand_motion_scene(context)
    result = await runner.move_to_observe(context, joint_goals=reference['joints_rad'])
    proof = initial_pose_verdict(reference, runner.simulation.robot_snapshot())
    if not proof['success']:
        raise ValueError('initial_pose_unconfirmed')
    return {**result, 'success': True, 'already_at_initial_pose': False, 'pose_verification': proof}
