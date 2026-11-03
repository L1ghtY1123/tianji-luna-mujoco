"""A motion rejected before actuation must remain recoverable after map repair."""
import json
from pathlib import Path

from luna_mujoco.agent.recovery import RecoveryPlanner
from luna_mujoco.runtime.controller import BuildSceneArguments, ObservationMotionArguments
from luna_mujoco.skills import argument_contract

ROOT = Path(__file__).resolve().parents[1]


def contract(name):
    result = json.loads((ROOT / 'skill_catalog' / name / 'contract.json').read_text())
    schema = ObservationMotionArguments if name == 'move_to_observe' else BuildSceneArguments
    result['arguments'] = schema.model_json_schema()
    return result


async def test_map_repair_makes_explicit_retry_of_observation_motion_available():
    calls = []

    async def judge(requests, context):
        calls.append(requests)
        assert requests[0].name == 'recovery_continuation'
        assert 'retry_failed_node' in requests[0].choices
        return {'judgments': {requests[0].name: {'status': 'proposed', 'value': 'retry_failed_node'}}}

    planner = RecoveryPlanner(judge, {'move_to_observe': contract('move_to_observe')})
    result = await planner.propose(
        {'type': 'skill', 'id': 'return', 'skill': 'move_to_observe', 'arguments': {'viewpoint': 'configured'}},
        {'status': 'failure', 'reason': 'observation_motion_requires_current_map',
         'partial_execution': {'skill': 'move_to_observe', 'invocation_id': 'no-motor'}},
        [{'node': {'skill': 'build_scene'}, 'status': 'success', 'result': {'status': 'success'}}], {},
        facts={'scene_available': True, 'grasp_held': False, 'observation_motion_scene_ready': True},
        inventory=[], context={})
    assert result['action'] == 'retry'
    assert len(calls) == 1


def test_scene_modes_expose_their_actual_observation_map_effect():
    value = contract('build_scene')
    assert argument_contract(value, {'exclusion_state': 'observation'})['planning_effects']['observation_map_ready'] is True
    assert argument_contract(value, {'exclusion_state': 'selected'})['planning_effects']['observation_map_ready'] is False


def test_motion_scene_requirement_preserves_authorized_retained_scene_without_accepting_stale_maps():
    from luna_mujoco.robot_skills.return_pose import observation_motion_scene_ready

    context = {'scene': {'exclusion_state': 'selected', 'observation_id': 'frame'},
               'observation': {'observation_id': 'frame'}}
    assert not observation_motion_scene_ready(context)
    context['scene']['exclusion_state'] = 'observation'
    assert observation_motion_scene_ready(context)
    context['scene']['observation_id'] = 'old'
    assert not observation_motion_scene_ready(context)
    context['pending_transfer'] = {'verified': True}
    assert observation_motion_scene_ready(context)
    context['pending_transfer']['verified'] = False
    assert not observation_motion_scene_ready(context)
    context['pending_transfer']['verified'] = True
    context['scene_refresh_required'] = True
    assert not observation_motion_scene_ready(context)


async def test_selected_map_without_retained_authorization_does_not_offer_a_motion_retry():
    async def judge(requests, context):
        question = next(q for q in requests if q.name == 'recovery_capability')
        assert 'retry_failed_node' not in question.choices
        assert context['retry_precondition_conflicts']['observation_motion_scene_ready']['actual'] is False
        return {'judgments': {question.name: {'status': 'proposed', 'value': 'stop'}}}

    result = await RecoveryPlanner(judge, {'move_to_observe': contract('move_to_observe')}).propose(
        {'type': 'skill', 'id': 'return', 'skill': 'move_to_observe', 'arguments': {'viewpoint': 'configured'}},
        {'status': 'failure', 'reason': 'observation_motion_requires_current_map'}, [], {},
        facts={'scene_available': True, 'grasp_held': False, 'observation_motion_scene_ready': False},
        inventory=[], context={})
    assert result['action'] == 'stop'
