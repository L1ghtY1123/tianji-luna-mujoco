"""Terminal recovery preserves the object and checks every motion from actual feedback."""
import asyncio
from copy import deepcopy

import numpy as np

from ..core.decisions import JudgmentRequest
from ..observation_motion import observation_poses


def gripper_is_open(feedback, settings):
    return (feedback.get('gripper_closed_command') is False
            and abs(feedback['gripper_closed_fraction']) <= settings['holding_position_tolerance_fraction'])


def return_mode(runner, context):
    feedback = runner.simulation.robot_snapshot()
    holding = context.get('holding') or runner.holding_state
    if (runner.is_observation_pose(context, feedback)
            and (holding or all(gripper_is_open(arm, context['planning_settings'])
                                for arm in feedback['arms'].values()))):
        return 'at_pose'
    if holding:
        if (holding.get('verified') is True and holding.get('attached') is True
                and holding.get('attachment_evidence', {}).get('scene_token') == (runner.spatial_memory.scene or {}).get('scene_token')
                and not runner.spatial_memory.refresh_required):
            return 'held'
        return 'refresh'
    if context.get('failure_empty_confirmed') is True or not runner.state.get('facts', {}).get('grasp_evidence_pending'):
        if runner.state.get('partial_execution_failure') and not context.get('failure_empty_confirmed'):
            return 'refresh'
        return 'refresh' if runner.spatial_memory.refresh_required or not runner.spatial_memory.scene else 'empty'
    return 'refresh'


async def prepare_failure_return(runner, context):
    if context.get('failure_return_attempted'):
        raise ValueError('terminal_return_already_attempted')
    context['failure_return_attempted'] = True
    stopped = await asyncio.to_thread(runner.simulation.stop_motion)
    runner.motion_epoch = stopped['motion_epoch']
    runner.observation_epoch += 1
    context['terminal_motion_stopped'] = True
    runner._database_committed(runner.object_database.changes_since(runner.spatial_memory.database_revision))
    mode = return_mode(runner, context)
    context['failure_return_mode'] = mode
    result = {'status': 'already_at_observation' if mode == 'at_pose' else 'preparing',
              'mode': mode, 'motion_stopped': stopped, 'task_succeeded': False,
              'original_failure': deepcopy(context.get('terminal_original_failure'))}
    runner.state['failure_return'] = result
    return {'success': True, **result}


async def observe_after_failure(runner, context):
    from .grasp_verification import HOLDING_POSITION_RULE, gripper_evidence_context

    if not context.get('terminal_motion_stopped'):
        raise ValueError('terminal_motion_not_stopped')
    capture = await runner.capture_observation(context, recovery=True)
    runner.record_observation(capture, 'terminal_failure')
    target = runner.state.get('target_id')
    reference = context.get('identity_reference', {})
    evidence = await runner.call_algorithm(runner.algorithms.prepare_gripper_evidence,
        capture, ['A', 'B'], {target: reference.get('entities', {}).get(target, {})} if target else {})
    result = await runner._judge(context, [JudgmentRequest(name='failure_gripper_holds', instructions=(
        '至少一个机器人夹爪当前仍夹持着物体。逐个夹爪判断，有一个持物则本命题为真。'
        + HOLDING_POSITION_RULE))],
        include_task=False, reference={}, extra={'verification_scope': 'grasp_result',
            'target_id': target, 'selected_arm': runner.state.get('arm'),
            **gripper_evidence_context(capture, evidence)})
    answer = result['judgments']['failure_gripper_holds']['status']
    context['failure_gripper_evidence'] = result
    result['holding_presence'] = {'status': answer, 'rule': 'measured_position_then_closed_gripper_vision'}
    if answer != 'no':
        mode = return_mode(runner, context)
        if answer == 'yes' and mode == 'held':
            context['failure_return_mode'] = 'held'
            return {'success': True, 'mode': 'held', 'visual_evidence': result}
        return {'success': False, 'status': 'unknown', 'reason': 'terminal_holding_state_unconfirmed', 'evidence': result}
    detached = await runner.call_algorithm(runner.algorithms.detach_carrying)
    if detached.get('attached') is not False:
        raise ValueError('terminal_detachment_unconfirmed_after_empty_gripper_evidence')
    context.pop('holding', None)
    runner.holding_state = None
    context['failure_empty_confirmed'] = True
    runner.state['facts'].update(grasp_held=False, attached=False, recovery_needed=False)
    # This node explicitly owns observation; no movement or opening command is issued.
    observed = await runner.observe(context, checkpoint='initial', capture=capture, recovery=True)
    context['failure_return_mode'] = 'empty'
    return {'success': True, 'observation_id': observed['observation_id'], 'empty_grippers': result}


async def build_failure_scene(runner, context):
    if context.get('failure_return_mode') == 'held':
        return {'success': True, 'source': 'current_verified_attachment_and_scene'}
    if context.get('failure_empty_confirmed') is not True:
        raise ValueError('terminal_empty_grippers_unconfirmed')
    from ..memory_contracts import CollisionUpdate

    strategy = await runner._mapping_strategy(context, {'exclusion_state': 'observation'})
    runner.spatial_memory.request_update(CollisionUpdate.model_validate(strategy))
    return await runner.build_scene(context, 'observation')


async def return_after_failure(runner, context):
    await runner.verify_scene(context)
    if not context.get('terminal_motion_stopped') or runner.spatial_memory.refresh_required:
        raise ValueError('terminal_return_requires_current_collision_scene')
    scene = runner.spatial_memory.scene
    if not scene:
        raise ValueError('terminal_return_requires_current_collision_scene')
    held = context.get('holding') or runner.holding_state
    if held and not (held.get('verified') is True and held.get('attached') is True):
        raise ValueError('terminal_holding_state_unconfirmed')
    if not held and context.get('failure_return_mode') != 'empty':
        raise ValueError('terminal_empty_grippers_unconfirmed')
    from ..carrying import held_collision_binding

    binding = (held_collision_binding(held, scene, held['arm'], held['instance_id']) if held else None)
    observation = runner.measured_state(context)
    if not runner.is_observation_pose(context, observation):
        goals = observation_poses(context['planning_settings'])
        plan = await runner.call_algorithm(runner.algorithms.plan_dual_observation_motion, observation, scene, goals,
            planning_settings=context['planning_settings'], terminal_return=True, carrying_binding=binding)
        feedback = runner.simulation.robot_snapshot()
        paths = {arm: np.asarray(plan['paths'][arm], dtype=float) for arm in ('A', 'B')}
        if len(paths['A']) != len(paths['B']):
            raise ValueError('terminal_return_trajectory_lengths_differ')
        for arm in ('A', 'B'):
            if np.max(np.abs(paths[arm][0] - feedback['arms'][arm]['joints_rad'])) > np.deg2rad(1.):
                raise ValueError('terminal_return_start_changed')
        paths = {arm: np.vstack((feedback['arms'][arm]['joints_rad'], paths[arm])).tolist() for arm in ('A', 'B')}
        await runner.call_algorithm(runner.algorithms.validate_dual_observation, runner.measured_state(context),
            scene, paths, planning_settings=context['planning_settings'], terminal_return=True, carrying_binding=binding)
        await runner.verify_scene(context)
        await runner.execution_stage(context, 'terminal_observation_return', runner.motor,
            runner.simulation.execute_dual_trajectory, paths, plan['dt'], execution_settings=context['planning_settings'])
    opening = {}
    if not held and runner.is_observation_pose(context, runner.simulation.robot_snapshot()):
        for arm in ('A', 'B'):
            measured = runner.measured_state(context)
            feedback = measured['arms'][arm]
            if gripper_is_open(feedback, context['planning_settings']):
                continue
            check = await runner.call_algorithm(runner.algorithms.validate_execution, measured, arm, scene,
                'open_grasp', {'positions_rad': [feedback['joints_rad']]},
                planning_settings=context['planning_settings'], gripper_end_position_rad=feedback['gripper_limits_rad'][0])
            await runner.verify_scene(context)
            command = await runner.execution_stage(context, 'terminal_empty_gripper_open', runner.motor,
                runner.simulation.set_gripper, arm, False)
            opening[arm] = {'collision_check': check, 'command': command}
    actual = runner.simulation.robot_snapshot()
    empty_grippers_open = not held and all(gripper_is_open(arm, context['planning_settings'])
                                          for arm in actual['arms'].values())
    success = runner.is_observation_pose(context, actual) and (held is not None or empty_grippers_open)
    result = {'success': success, 'status': 'returned' if success else 'return_incomplete',
              'source': 'actual_joint_feedback', 'holding_preserved': held is not None, 'task_succeeded': False,
              'empty_grippers_open': empty_grippers_open, 'gripper_opening': opening,
              'actual_feedback': actual, 'original_failure': deepcopy(context.get('terminal_original_failure'))}
    result['motion_stopped'] = await asyncio.to_thread(runner.simulation.stop_motion)
    runner.state['failure_return'] = result
    runner.spatial_memory.invalidate([])
    return result
