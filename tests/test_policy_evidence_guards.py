from types import SimpleNamespace

import pytest
from test_agent_model_layer import task_data
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController

from luna_mujoco.agent.decisions import JudgmentRequest
from luna_mujoco.agent.schema import TaskSpec


def controller(tmp_path):
    runner = TestPolicyController(DemoRobot(), DemoVision(1), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.task_spec = TaskSpec.model_validate(task_data() | {'destination_ref': 'b', 'work_area_ref': 'a'})
    runner.task_inventory = {'selected_ids': ['item'], 'source_object_ids': ['item']}
    runner.state.update(target_id='item', destination_id='a', arm='A', generation=1)
    return runner


async def test_recovery_decision_captures_current_rgb_without_publishing_inventory_or_map(tmp_path):
    runner = controller(tmp_path)
    context = {'planning_settings': TEST_SETTINGS.model_dump(), 'visual_inventory_reference': {},
               'observation': {'observation_id': 'measured-before'}, 'scene': {'scene_token': 'existing'}}
    before = await runner.capture_observation(context)
    await runner.motor(runner.simulation.set_gripper, 'A', False)
    requests = [JudgmentRequest(name='safe', instructions='Can recovery proceed?')]
    await runner._judge(context, requests, extra={'verification_scope': 'graph_recovery'})
    fresh = context['latest_capture']['observation_id']
    assert fresh != before['observation_id']
    assert context['capture_epochs'][fresh] == runner.observation_epoch
    assert context['observation']['observation_id'] == 'measured-before'
    assert context['scene']['scene_token'] == 'existing'
    assert not runner.state['facts'].get('observed', False)
    await runner._judge(context, requests, extra={'verification_scope': 'graph_recovery_continuation'})
    assert context['latest_capture']['observation_id'] == fresh


@pytest.mark.parametrize('destination', ['a', 'b'])
async def test_destination_rebinding_revokes_old_plan_before_motor(tmp_path, destination):
    runner = controller(tmp_path)
    runner.task_progress = SimpleNamespace(records={'item': {}})
    runner.state['facts'].update(place_plan_ready=True, plan_ready=True,
                                grasps_ready=True, arm_selected=True)
    context = {'observation': {'objects': [{'id': destination, 'kind': 'bin'}]},
               'place_plan': {'destination_id': 'a'},
               'grasps': {'plan_id': 'old'},
               'place_plan_binding': {'target_id': 'item', 'destination_id': 'a', 'arm': 'A'}}
    await runner.execute('identify_destination', context, {'destination_id': destination, 'destination_kind': 'container'})
    assert not runner.state['facts']['place_plan_ready']
    assert not runner.state['facts']['plan_ready']
    assert not runner.state['facts']['grasps_ready']
    assert not runner.state['facts']['arm_selected']
    assert 'grasps' not in context
    assert 'place_plan' not in context
    with pytest.raises(ValueError, match='place_plan_binding_changed'):
        await runner.execute('execute_place', context)
    assert runner.simulation.moves == []


async def test_graph_can_choose_measured_intermediate_destination_without_changing_goal(tmp_path):
    runner = controller(tmp_path)
    runner.task_spec = runner.task_spec.model_copy(update={'constraints': ()})
    original = runner.task_spec
    runner.task_progress = SimpleNamespace(records={'item': {}})
    result = await runner.execute('identify_destination', {'observation': {'objects': [
        {'id':'intermediate-bin', 'kind':'bin'}]}}, {'destination_id':'intermediate-bin', 'destination_kind':'container'})
    assert result['success']
    assert runner.task_spec == original and runner.task_spec.destination_ref == 'b'
    assert runner.task_progress.records['item']['destination_ref'] == 'intermediate-bin'
    with pytest.raises(ValueError):
        await runner.execute('identify_destination', {'observation': {'objects': []}},
                             {'destination_id': 'nonexistent-bin', 'destination_kind': 'container'})


@pytest.mark.parametrize('verdict, status, reason', [
    ('no', 'failure', 'intermediate_destination_rejected'),
    ('unknown', 'unknown', 'intermediate_destination_constraint_unconfirmed'),
])
async def test_intermediate_destination_must_respect_user_constraints(tmp_path, verdict, status, reason):
    from unittest.mock import AsyncMock

    runner = controller(tmp_path)
    runner.task_progress = SimpleNamespace(records={'item': {}})
    runner._judge = AsyncMock(return_value={'judgments': {'intermediate_destination_allowed':
        {'status': verdict, 'probability': .02 if verdict == 'no' else .5}}, 'evidence':{}})
    result = await runner.execute('identify_destination', {'observation': {'objects': [
        {'id':'intermediate-bin', 'kind':'bin'}]}}, {'destination_id':'intermediate-bin', 'destination_kind':'container'})
    assert result['success'] is False
    assert result['status'] == status
    assert result['reason'] == reason
    assert runner.state['destination_id'] == 'a'
    runner._judge.assert_awaited_once()


async def test_visual_branch_rejects_pre_action_capture_then_accepts_explicit_new_capture(tmp_path):
    runner = controller(tmp_path)
    context = {'planning_settings': TEST_SETTINGS.model_dump(), 'visual_inventory_reference': {}}
    before = await runner.capture_observation(context)
    context['latest_capture'] = before
    await runner.motor(runner.simulation.set_gripper, 'A', False)
    requests = [JudgmentRequest(name='safe', instructions='Is the current task condition visible?')]
    with pytest.raises(ValueError, match='fresh_observation_required_before_judgment'):
        await runner._judge(context, requests)
    with pytest.raises(ValueError, match='fresh_observation_required_after_action'):
        await runner.capture_observation(context, capture=before)
    context['latest_capture'] = await runner.capture_observation(context)
    result = await runner._judge(context, requests)
    assert result['judgments']['safe']['status'] == 'yes'
