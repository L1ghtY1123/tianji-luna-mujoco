import json
from copy import deepcopy

import httpx
import pytest

from luna_mujoco.agent.completion import completion_obligations
from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest
from luna_mujoco.task_progress import TaskProgress


def post_delivery_runner(tmp_path):
    from types import SimpleNamespace

    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.runtime.controller import PolicyController
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.environment_assumptions = {'completion_requirements': {'after_each_object': {
        'robot_pose': 'observation', 'observe_inventory': True}}}
    feedback = {'generation': 1, 'arms': {arm: {'joints_rad': [0.] * 7} for arm in ('A', 'B')}}
    runner.simulation = SimpleNamespace(robot_snapshot=lambda: deepcopy(feedback))
    records = {'pink': {'id': 'pink'}, 'green': {'id': 'green'}}
    runner.task_inventory = {'selected_ids': ['pink', 'green']}
    runner.task_progress = SimpleNamespace(completed_input_ids=lambda: {'pink', 'green'},
        records=records, ledger_revision=0, inventory_state=dict,
        snapshot=lambda: {'objects': deepcopy(list(records.values()))})
    runner.state['facts'].update(observed=True, observation_pose_ready=True)
    runner.observation_epoch = 7
    runner.emit = lambda *args: None
    capture = {**deepcopy(feedback), 'observation_id': 'after-pink'}
    context = {'planning_settings': TEST_SETTINGS.model_dump(), 'latest_capture': capture,
        'observation': {'observation_id': 'after-pink'}, 'capture_epochs': {'after-pink': 7}}
    return runner, context, feedback


def test_each_object_observation_receipt_reaches_final_decisions_summary(tmp_path):
    runner, context, _ = post_delivery_runner(tmp_path)
    runner._record_post_delivery_observation(context, 'pink')
    context['latest_capture']['observation_id'] = 'after-green'
    context['observation']['observation_id'] = 'after-green'
    runner.observation_epoch = 9
    context['capture_epochs']['after-green'] = 9
    runner._record_post_delivery_observation(context, 'green')
    rows = completion_obligations(runner.task_progress.snapshot())['objects']
    assert [row['post_delivery_observation']['observation_id'] for row in rows] == ['after-pink', 'after-green']
    assert [row['post_delivery_observation']['action_epoch'] for row in rows] == [7, 9]
    assert runner.task_progress.ledger_revision == 2
    assert all(set(row['post_delivery_observation']['actual_joints_rad']) == {'A', 'B'} for row in rows)


def test_later_return_motion_does_not_erase_completed_observation(tmp_path):
    runner, context, feedback = post_delivery_runner(tmp_path)
    runner._record_post_delivery_observation(context, 'green')
    receipt = deepcopy(runner.task_progress.records['green']['post_delivery_observation'])
    runner.observation_epoch += 1
    runner.state['facts'].update(observed=False, observation_pose_ready=False)
    feedback['arms']['A']['joints_rad'][0] = .1
    runner._record_post_delivery_observation(context, 'green')
    assert runner.task_progress.records['green']['post_delivery_observation'] == receipt
    assert runner.task_progress.ledger_revision == 1


def test_receipt_cannot_prove_a_different_delivery_or_simulation_generation(tmp_path):
    runner, context, feedback = post_delivery_runner(tmp_path)
    runner.task_progress.records['green']['confirmed_observation_id'] = 'delivery-one'
    runner._record_post_delivery_observation(context, 'green')
    runner.state['facts']['observed'] = False
    runner.task_progress.records['green']['confirmed_observation_id'] = 'delivery-two'
    with pytest.raises(ValueError, match='object_post_delivery_observation_missing'):
        runner._record_post_delivery_observation(context, 'green')
    runner.task_progress.records['green']['confirmed_observation_id'] = 'delivery-one'
    feedback['generation'] += 1
    with pytest.raises(ValueError, match='object_post_delivery_observation_missing'):
        runner._record_post_delivery_observation(context, 'green')


async def test_receipt_is_created_when_inventory_observation_finishes(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    runner, context, _ = post_delivery_runner(tmp_path)
    runner.state['facts']['observed'] = False
    runner.state['current_step'] = {'invocation_id': 'observe-test'}
    monkeypatch.setattr(runner, 'check_generation', lambda: None)
    monkeypatch.setattr(runner, 'check_log', lambda: None)
    monkeypatch.setattr(runner, 'begin_step', lambda name: None)
    monkeypatch.setattr(runner.registry, 'execute', AsyncMock(return_value=deepcopy(context['observation'])))
    await runner._step('observe_scene', context)
    assert runner.task_progress.records['green']['post_delivery_observation']['action_epoch'] == 7


@pytest.mark.parametrize('status', ['completed', 'observed'])
@pytest.mark.parametrize('evidence', ['confirmed', 'awaiting_confirmation', 'stale_transfer'])
async def test_intermediate_processing_delivery_gets_its_own_observation_receipt(
        tmp_path, monkeypatch, status, evidence):
    from unittest.mock import AsyncMock
    runner, context, _ = post_delivery_runner(tmp_path)
    runner.task_progress.completed_input_ids = lambda: set()
    runner.task_progress.records['pink'].update(status=status, confirmed_observation_id='release',
        current_region='work', transfer_index=0, transport_status='confirmed',
        transfers=[{'transfer_index': 0, 'destination_id': 'work', 'observation_id': 'release'}])
    if evidence == 'awaiting_confirmation':
        runner.task_progress.records['pink']['transport_status'] = 'awaiting_visual_confirmation'
    elif evidence == 'stale_transfer':
        runner.task_progress.records['pink']['transfers'][0]['observation_id'] = 'old-release'
    runner.state['facts']['observed'] = False
    runner.state['current_step'] = {'invocation_id': 'observe-intermediate'}
    monkeypatch.setattr(runner, 'check_generation', lambda: None)
    monkeypatch.setattr(runner, 'check_log', lambda: None)
    monkeypatch.setattr(runner, 'begin_step', lambda name: None)
    monkeypatch.setattr(runner.registry, 'execute', AsyncMock(return_value=deepcopy(context['observation'])))
    await runner._step('observe_scene', context)
    transfer = runner.task_progress.records['pink']['transfers'][0]
    if evidence != 'confirmed':
        assert 'post_delivery_observation' not in transfer
        return
    assert transfer['post_delivery_observation']['delivery']['confirmed_observation_id'] == 'release'
    assert transfer['post_delivery_observation']['observation_id'] == 'after-pink'


@pytest.mark.parametrize('missing', ['pose', 'epoch', 'observed', 'capture', 'delivery'])
def test_post_delivery_receipt_cannot_be_inferred_from_delivery_or_stale_images(tmp_path, missing):
    runner, context, feedback = post_delivery_runner(tmp_path)
    if missing == 'pose': feedback['arms']['B']['joints_rad'][0] = .1
    if missing == 'epoch': context['capture_epochs']['after-pink'] = 6
    if missing == 'observed': runner.state['facts']['observed'] = False
    if missing == 'capture': context['latest_capture']['observation_id'] = 'unprocessed'
    if missing == 'delivery': runner.task_progress.completed_input_ids = set
    with pytest.raises(ValueError):
        runner._record_post_delivery_observation(context, 'pink')
    assert 'post_delivery_observation' not in runner.task_progress.records['pink']


async def test_completion_keeps_each_model_condition_in_failure_diagnostics(tmp_path):
    from test_unpack_final_return import controller

    from luna_mujoco.runtime.diagnostics import DiagnosticError

    runner, context, _ = controller(tmp_path)
    conditions = ('Requested outputs are in the destination', 'Both arms returned')
    runner.task_spec = runner.task_spec.model_copy(update={'completion_conditions': conditions})
    runner._judge.return_value = {
        'judgments': {'goal_satisfied': {'status': 'unknown', 'probability': .49},
                      'completion_condition_0': {'status': 'no', 'probability': .01},
                      'completion_condition_1': {'status': 'yes', 'probability': .99}},
        'evidence': {}}
    with pytest.raises(DiagnosticError) as caught:
        await runner._complete(context)
    requests = runner._judge.call_args.args[1]
    assert all('For original input IDs ["pink"]' in request.instructions for request in requests)
    assert all('A consumed input need not remain visible' in request.instructions for request in requests)
    assert all('Missing or unverified historical receipts remain unproven' in request.instructions for request in requests)
    for index, condition in enumerate(conditions):
        assert any(request.name == f'completion_condition_{index}' and condition in request.instructions
                   for request in requests)
    assert any(check['name'] == 'completion_condition_0' and check['actual'] == .01
               for check in caught.value.details['diagnostic_checks'])


@pytest.mark.parametrize('later_status', ['unknown', 'elsewhere'])
def test_later_observation_cannot_erase_verified_delivery_or_hide_new_uncertainty(later_status):
    ledger = TaskProgress('move item into bin')
    ledger.scene_entities['bin'] = {'kind': 'bin'}
    ledger.select({'id': 'item'}, 'bin')
    ledger.record_release('item', feedback_id='release')
    verified = {'status': 'at_destination', 'destination_id': 'bin', 'observation_id': 'delivered',
                'source': 'openai_decisions', 'request_hash': 'proof', 'confidence': .98}
    assert ledger.confirm_visual_transfer('item', verified)
    later = {'status': later_status, 'destination_id': 'bin', 'observation_id': 'later', 'source': 'openai_decisions'}
    ledger.record_visual_progress('item', later)
    record = ledger.records['item']
    assert record['destination_visual_evidence'] == verified
    assert record['destination_tracking_evidence'] == later
    compact = completion_obligations(ledger.snapshot())['objects'][0]
    assert compact['destination_visual_evidence'] == verified
    assert compact['destination_tracking_evidence'] == later


def test_current_transfer_does_not_inherit_previous_confirmed_release():
    ledger = TaskProgress('move item')
    ledger.select({'id': 'item'}, 'new-bin')
    record = ledger.records['item']
    record['visual_release'] = {'confirmed_observation_id': 'prior', 'destination_id': 'old-bin',
                                'transfer_index': -1}
    later = {'status': 'unknown', 'destination_id': 'new-bin', 'observation_id': 'now'}
    ledger.record_visual_progress('item', later)
    assert record['destination_visual_evidence'] == later
    assert 'destination_tracking_evidence' not in record


def test_completion_summary_keeps_uncertainties_and_receipts_without_geometry_history():
    original = {'selected_input_ids': ['a'], 'completed_input_ids': [], 'deferred': [{'id': 'a'}],
        'unresolved_membership': [{'id': 'b'}], 'objects': [{'id': 'a', 'status': 'observed',
            'initial_source': {'large_historical_geometry': [1] * 1000},
            'destination_visual_evidence': {'status': 'unknown', 'request_hash': 'no-proof', 'confidence': .5}}]}
    frozen = deepcopy(original)
    result = completion_obligations(original)
    assert result['completed_input_ids'] == []
    assert result['deferred'] == original['deferred']
    assert result['unresolved_membership'] == original['unresolved_membership']
    assert result['objects'][0]['destination_visual_evidence'] == original['objects'][0]['destination_visual_evidence']
    assert 'initial_source' not in result['objects'][0]
    assert original == frozen


def test_completion_history_rules_are_scoped_and_do_not_modify_predicate_results():
    from test_agent_model_layer import capture
    calls = []
    def reply(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'answers': [
            {'type': 'predicate', 'name': 'source_complete', 'probability': .01}]})
    judge = DecisionsJudgments(api_key='test', client=httpx.Client(transport=httpx.MockTransport(reply)))
    for scope in ('ordinary', 'task_completion', 'policy_condition'):
        result = judge.judge(capture(), requests=[JudgmentRequest(name='source_complete', instructions='Check source')],
                             context={'verification_scope': scope})
        assert result['judgments']['source_complete']['status'] == 'no'
    ordinary, final, condition = [json.loads(call['input'][0]['content'][0]['text']) for call in calls]
    assert 'Retired entities are consumed inputs' not in ordinary['rules']
    assert 'Retired entities are consumed inputs' in final['rules']
    assert 'task.source_ref' in final['rules']
    assert 'must still remain uncertain' in final['rules']
    assert condition['rules'] == final['rules']
