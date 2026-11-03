import pytest

from luna_mujoco.agent.visibility_recovery import (
    VISIBILITY_FIELDS, visibility_allows_motion, visibility_assessment, visibility_requests,
)


def diagnostic_answers(cause):
    return {name: {'status': 'unknown' if cause == 'unknown' else 'yes' if value == cause else 'no'}
            for name, value in VISIBILITY_FIELDS.items()}


@pytest.mark.parametrize('value,allowed', [
    ('not_visible', True), ('cropped', True), ('occluded', True),
    ('sufficient', False), ('nonvisual_failure', False), ('identity_ambiguous', False), ('unknown', False)])
def test_viewpoint_requires_confirmed_visual_limitation(value, allowed):
    answer = {'status': 'known', 'value': value}
    assert visibility_allows_motion(answer) is allowed
    assert not visibility_allows_motion({**answer, 'status': 'unknown'})
    assert not visibility_allows_motion({**answer, 'status': 'proposed'})

def test_visibility_is_evidence_not_a_motion_proposal():
    requests = visibility_requests()
    assert {request.name for request in requests} == set(VISIBILITY_FIELDS)
    assert all(not hasattr(request, 'choices') and 'ALL three' in request.instructions
               and 'CAD' in request.instructions for request in requests)


def test_cropping_and_occlusion_can_coexist_without_a_conflicting_choice():
    answers = diagnostic_answers('cropped')
    answers['recovery_occluded'] = {'status': 'yes', 'probability': .9}
    assessment = visibility_assessment(answers)
    assert assessment['causes'] == ['cropped', 'occluded']
    assert visibility_allows_motion(assessment)
    assert not visibility_allows_motion(visibility_assessment(diagnostic_answers('unknown')))


@pytest.mark.parametrize('value,action', [('occluded', 'skill'), ('identity_ambiguous', 'rejected'),
                                        ('nonvisual_failure', 'rejected'), ('unknown', 'rejected')])
async def test_recovery_selects_and_diagnoses_in_one_request_without_waiving_motion_reason(value, action):
    from luna_mujoco.agent.recovery import RecoveryPlanner

    calls = []
    async def judge(requests, context):
        calls.append(requests)
        return {'judgments': {
            **diagnostic_answers(value),
            'recovery_capability': {'status': 'proposed', 'value': 'move_to_observe'}}}
    planner = RecoveryPlanner(judge, {'move_to_observe': {
        'arguments': {'type': 'object', 'properties': {}}}})
    result = await planner.propose({'id': 'failed', 'type': 'skill', 'skill': 'identify_target'},
        {'status': 'unknown'}, [], {}, facts={}, inventory=[], context={})
    assert result['action'] == action
    assert len(calls) == 1
    assert {q.name for q in calls[0]} == {*VISIBILITY_FIELDS, 'recovery_capability'}


async def test_stale_visibility_cannot_unlock_motion(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision
    from luna_mujoco.runtime.controller import PolicyController

    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.observation_epoch = 5
    context = {'latest_capture': {'observation_id': 'now'}, 'visibility_recovery': {
        'observation_id': 'old', 'epoch': 4, 'answer': {'status': 'known', 'value': 'occluded'}}}
    calls = []
    async def judge(context, requests, **kwargs):
        calls.append(requests)
        return {'judgments': diagnostic_answers('nonvisual_failure')}
    runner._judge = judge
    with pytest.raises(ValueError, match='viewpoint_requires_visual_limitation'):
        await runner._require_visibility_recovery(context)
    assert len(calls) == 1


@pytest.mark.parametrize('visibility,action', [('sufficient', 'rejected'), ('unknown', 'rejected'),
                                             ('occluded', 'skill')])
async def test_release_failure_diagnoses_visibility_before_geometry_even_without_legal_motion(visibility, action):
    from luna_mujoco.agent.recovery import RecoveryPlanner

    async def judge(requests, context):
        assert {q.name for q in requests} == {*VISIBILITY_FIELDS, 'recovery_capability'}
        assert 'release_verification_scope' in context
        return {'judgments': {
            **diagnostic_answers(visibility),
            'recovery_capability': {'status': 'proposed', 'value': 'observe_scene'}}}
    planner = RecoveryPlanner(judge, {'observe_scene': {'arguments': {'type': 'object', 'properties': {}}}})
    result = await planner.propose({'id': 'failed', 'type': 'skill', 'skill': 'verify_place'},
        {'status': 'unknown'}, [], {}, facts={}, inventory=[], context={})
    assert result['action'] == action


async def test_unknown_visibility_is_reused_for_same_failure_and_capture(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision
    from luna_mujoco.agent.decisions import ChoiceRequest
    from luna_mujoco.runtime.controller import PolicyController

    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.observation_epoch = 5
    context = {'latest_capture': {'observation_id': 'now'}}
    calls = []
    async def judge(context, requests, **kwargs):
        names = {q.name for q in requests}
        calls.append(names)
        return {'judgments': {name: {'status': 'unknown'} for name in names}}
    runner._judge = judge
    requests = [*visibility_requests(), ChoiceRequest(name='capability', instructions='Choose.', choices=('stop', 'unknown'))]
    extra = {'failed_node': {'id': 'verify', 'skill': 'verify_place'}, 'failed_result': {'reason': 'visual_unknown'}}
    await runner._request_recovery_judgments(context, requests, extra)
    repeated = await runner._request_recovery_judgments(context, requests, {**extra, 'rejected_proposals': ['observe']})
    assert calls == [{*VISIBILITY_FIELDS, 'capability'}, {'capability'}]
    assert repeated['judgments']['recovery_occluded']['status'] == 'unknown'
    context['latest_capture']['observation_id'] = 'new'
    await runner._request_recovery_judgments(context, requests, extra)
    assert calls[-1] == {*VISIBILITY_FIELDS, 'capability'}


async def test_rejected_capability_without_arguments_is_not_offered_again():
    from luna_mujoco.agent.recovery import RecoveryPlanner

    async def judge(requests, context):
        choice = next(q for q in requests if q.name == 'recovery_capability')
        assert 'observe_scene' not in choice.choices
        return {'judgments': {**diagnostic_answers('sufficient'),
            'recovery_capability': {'status': 'proposed', 'value': 'stop'}}}
    planner = RecoveryPlanner(judge, {'observe_scene': {'arguments': {'type': 'object', 'properties': {}}}})
    result = await planner.propose({'id': 'failed', 'type': 'skill', 'skill': 'verify_place'},
        {'status': 'unknown'}, [], {}, facts={}, inventory=[], context={
            'rejected_proposals': [{'skill': 'observe_scene', 'reason': 'rgb_release_does_not_require_geometry'}]})
    assert result['action'] == 'stop'


@pytest.mark.parametrize('required,reason,expected', [
    (True, 'processing_requires_post_delivery_observation', 'skill'),
    (False, 'processing_requires_post_delivery_observation', 'rejected'),
    (True, 'identity_ambiguous', 'rejected')])
async def test_user_required_return_is_separate_from_visual_viewpoint_recovery(required, reason, expected):
    from luna_mujoco.agent.recovery import RecoveryPlanner

    async def judge(requests, context):
        return {'judgments': {**diagnostic_answers('sufficient'),
            'recovery_capability': {'status': 'proposed', 'value': 'move_to_observe'}}}
    planner = RecoveryPlanner(judge, {'move_to_observe': {
        'arguments': {'type': 'object', 'properties': {}}}})
    result = await planner.propose({'id': 'failed', 'type': 'skill', 'skill': 'prepare_unpack'},
        {'status': 'failure', 'reason': reason}, [], {}, facts={}, inventory=[], context={
            'completion_requirements': {'after_each_object': {'robot_pose': 'observation'}} if required else {}})
    assert result['action'] == expected


@pytest.mark.parametrize('skill,viewpoint,checks,pending', [
    ('observe_scene', 'configured', 0, True),
    ('move_to_observe', 'configured', 0, False),
    ('move_to_observe', 'wrist_tilt', 1, True)])
async def test_runtime_consumes_required_return_only_for_configured_movement(
        tmp_path, monkeypatch, skill, viewpoint, checks, pending):
    from unittest.mock import AsyncMock
    from test_completion_evidence import post_delivery_runner

    runner, context, _ = post_delivery_runner(tmp_path)
    context.update(pending_required_return=True, pending_visibility_motion=True)
    monkeypatch.setattr(runner, 'check_generation', lambda: None)
    monkeypatch.setattr(runner, 'check_log', lambda: None)
    guard = AsyncMock()
    monkeypatch.setattr(runner, '_require_visibility_recovery', guard)

    class ReachedSkill(Exception):
        pass

    def begin(name):
        raise ReachedSkill

    monkeypatch.setattr(runner, 'begin_step', begin)
    with pytest.raises(ReachedSkill):
        await runner._step(skill, context, {'viewpoint': viewpoint})
    assert guard.await_count == checks
    assert context.get('pending_required_return', False) is pending
    if skill == 'move_to_observe':
        assert not context.get('pending_visibility_motion')
