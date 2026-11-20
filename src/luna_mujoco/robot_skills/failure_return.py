"""Terminal recovery preserves the object and checks every motion from actual feedback."""
import asyncio
from copy import deepcopy

from .grasp_recovery import empty_grasp_evidence
from .grasp_verification import GRIPPER_ENDPOINT_TOLERANCE_FRACTION, holding_from_feedback
from .observation import stationary_feedback_rejection


from ..observation_motion import observation_poses


def gripper_is_open(feedback):
    fraction = feedback.get('gripper_closed_fraction')
    return type(fraction) in (int, float) and 0 <= fraction <= GRIPPER_ENDPOINT_TOLERANCE_FRACTION


def return_mode(runner, context):
    feedback = runner.simulation.joint_feedback(runner.state['generation'])
    holding = context.get('holding') or runner.holding_state
    if (not feedback['active_commands'] and runner.is_observation_pose(context, feedback)
            and ((holding and holding.get('verified') and holding.get('attached'))
                 or not holding and all(gripper_is_open(feedback['arms'][arm]) for arm in ('A', 'B')))):
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


def stopped_empty_hands(previous, current, stopped):
    if not previous:
        return None
    proof, capture = previous
    # The acknowledged stop is the sole command allowed between captures.
    if (stopped.get('ok') is not True or stopped.get('robot_identity_changed') is not False
            or stopped.get('previous_generation') != proof['generation']
            or stopped.get('generation') != current.get('generation')
            or capture.get('generation') != proof['generation']
            or type(capture.get('command_revision')) is not int
            or current.get('command_revision') != capture['command_revision'] + 1
            or current.get('time', 0.) <= capture['time']
            or any(not capture.get(k) or capture[k] != current.get(k) for k in ('session_id', 'bridge_instance_id'))):
        return None
    if not capture.get('manifest_revision') or capture['manifest_revision'] != current.get('manifest_revision'):
        return None
    peer = 'B' if proof['arm'] == 'A' else 'A'
    if (stationary_feedback_rejection(capture, current)
            or holding_from_feedback(current, peer)['status'] != 'no'):
        return None
    return {'judgments': {'failure_gripper_holds': {'status': 'no'}},
            'evidence': {'source': 'verified_empty_grasp_and_stationary_stop', 'grasp': proof,
                         'observation_id': current['observation_id'], 'peer_arm_open': peer}}


async def prepare_failure_return(runner, context):
    if context.get('failure_return_attempted'):
        raise ValueError('terminal_return_already_attempted')
    context['failure_return_attempted'] = True
    proof = empty_grasp_evidence(runner, context)
    prior = (proof, context['grasp_verification_sample']) if proof else context.get('last_empty_grasp')
    if prior and all(prior[0].get(k) == v for k, v in (
            ('owner_id', runner.run_id), ('arm', runner.state.get('arm')), ('target_id', runner.state.get('target_id')),
            ('action_epoch', runner.observation_epoch), ('motion_epoch', runner.motion_epoch),
            ('generation', runner.state['generation']))):
        context['terminal_empty_grasp'] = prior
    stopped = await asyncio.to_thread(runner.simulation.stop_motion, confirm_world=True)
    if (stopped.get('ok') is not True or stopped['robot_identity_changed']
            or stopped['previous_generation'] != runner.state['generation']):
        raise ValueError('terminal_stop_identity_or_completion_unconfirmed')
    runner.motion_epoch = stopped['motion_epoch']
    # Only this acknowledged stop may advance the run into a recovery
    # generation. Old captures and collision maps remain invalid.
    runner.state['generation'] = stopped['generation']
    scene = runner.spatial_memory.scene
    world = {**stopped['world'], 'run_id': runner.run_id}
    if (scene and stopped['world_continuity_confirmed'] and scene.get('mapping_world') == world
            and scene['generation'] == stopped['previous_generation']):
        context['stopped_map_parent'] = {'scene_token': scene['scene_token'], 'world': world,
            'previous_generation': stopped['previous_generation'], 'generation': stopped['generation']}
    partial = (runner.state.get('partial_execution_failure') or {}).get('partial_execution', {})
    target = runner.state.get('target_id')
    uncertain = partial.get('skill') in {'execute_grasp', 'execute_place'} or runner.state['facts'].get('grasp_evidence_pending')
    runner.spatial_memory.invalidate([target] if uncertain and target else [])
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
    from .grasp_verification import WRIST_CAMERAS, object_held_question
    from ..visual_api import VisualDecisionError

    if not context.get('terminal_motion_stopped'):
        raise ValueError('terminal_motion_not_stopped')
    stopped = runner.state['failure_return']['motion_stopped']
    state = runner.simulation.robot_snapshot(include_tcp=False, expected_generation=runner.state['generation'])
    state['observation_id'] = 'stop-feedback-' + state['feedback_id']
    result = stopped_empty_hands(context.get('terminal_empty_grasp'), state, stopped)
    if result is None:
        answers = {}
        for arm in ('A', 'B'):
            capture = await runner.sample_gripper(context, arm, stopped['stopped_time_s'], 'terminal_gripper')
            measured = holding_from_feedback(capture, arm)
            answer = {'status': measured['status'], 'source': measured['source'],
                      'feedback_id': capture['feedback_id'], 'position': measured['reason']}
            if measured['requires_visual'] and capture.get('views'):
                wrist = WRIST_CAMERAS[arm]
                try:
                    decision = await runner._judge(context, [object_held_question(arm)],
                        include_task=False, reference={}, verification_capture=capture,
                        extra={'verification_scope': 'grasp_result', 'selected_arm': arm,
                               'primary_wrist_camera_id': wrist, 'visual_camera_ids': [wrist], 'visual_modalities': ['rgb']})
                except VisualDecisionError as error:
                    answer.update(status='unknown', error=str(error))
                else:
                    answer.update(status=decision['judgments']['object_held']['status'], evidence=decision['evidence'])
            elif capture.get('wrist_error'):
                answer['wrist_error'] = capture['wrist_error']
            answers[arm] = answer
            if answer['status'] == 'yes':
                break
        statuses = {a['status'] for a in answers.values()}
        presence = 'yes' if 'yes' in statuses else 'no' if statuses == {'no'} else 'unknown'
        result = {'judgments': {'failure_gripper_holds': {'status': presence}}, 'evidence': {'arms': answers}}
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
    observed = await runner.observe(context, checkpoint='initial', recovery=True)
    context['failure_return_mode'] = 'empty'
    return {'success': True, 'observation_id': observed['observation_id'], 'empty_grippers': result}


async def build_failure_scene(runner, context):
    if context.get('failure_return_mode') == 'held':
        return {'success': True, 'source': 'current_verified_attachment_and_scene'}
    if context.get('failure_empty_confirmed') is not True:
        raise ValueError('terminal_empty_grippers_unconfirmed')

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
        await runner.dispatch_observation_path(context, scene, plan, 'terminal_observation_return',
                                               terminal_return=True, carrying_binding=binding)
    opening = {}
    if not held and runner.is_observation_pose(context, runner.simulation.joint_feedback(runner.state['generation'])):
        for arm in ('A', 'B'):
            measured = runner.measured_state(context)
            feedback = measured['arms'][arm]
            if gripper_is_open(feedback):
                continue
            check = await runner.call_algorithm(runner.algorithms.validate_execution, measured, arm, scene,
                'open_grasp', {'positions_rad': [feedback['joints_rad']]},
                planning_settings=context['planning_settings'], gripper_end_position_rad=feedback['gripper_limits_rad'][0])
            await runner.verify_scene(context)
            command = await runner.execution_stage(context, 'terminal_empty_gripper_open', runner.motor,
                runner.simulation.set_gripper, arm, False)
            opening[arm] = {'collision_check': check, 'command': command}
    actual = runner.simulation.joint_feedback(runner.state['generation'])
    empty_grippers_open = not held and all(gripper_is_open(actual['arms'][arm]) for arm in ('A', 'B'))
    success = not actual['active_commands'] and runner.is_observation_pose(context, actual) and (held is not None or empty_grippers_open)
    result = {'success': success, 'status': 'returned' if success else 'return_incomplete',
              'source': 'actual_joint_feedback', 'holding_preserved': held is not None, 'task_succeeded': False,
              'empty_grippers_open': empty_grippers_open, 'gripper_opening': opening,
              'actual_feedback': actual, 'original_failure': deepcopy(context.get('terminal_original_failure'))}
    result['motion_stopped'] = await asyncio.to_thread(runner.simulation.stop_motion)
    runner.state['failure_return'] = result
    runner.spatial_memory.invalidate([])
    return result
