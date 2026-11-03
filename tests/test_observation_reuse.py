"""Reuse the captured evidence, never a different question's visual verdict."""
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController


@pytest.mark.parametrize('change,expected', [
    ('none', ['1', '2', '3']), ('pixels', ['1', '2']),
    ('id', []), ('generation', []), ('missing_generation', [])])
def test_semantic_judgments_receive_exact_same_capture_provenance_without_reusing_verdict(change, expected):
    import json

    from test_agent_model_layer import Client, capture

    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest

    current = capture()
    reference = deepcopy(current)
    if change == 'pixels': reference['views']['3']['rgb'][0, 0, 0] += 1
    elif change == 'id': reference['observation_id'] = 'different-frame'
    elif change == 'generation': reference['generation'] += 1
    elif change == 'missing_generation': reference.pop('generation')
    client = Client([{'answers': [{'type': 'predicate', 'name': 'visible', 'probability': .62}]}])
    result = DecisionsJudgments(api_key='test', client=client).judge(
        current, requests=[JudgmentRequest(name='visible', instructions='Is the tabletop visible?')],
        context={}, reference=reference)
    assert len(client.requests) == 1
    # Pixel equality neither answers the question nor changes its acceptance threshold.
    assert result['judgments']['visible'] == {'status': 'unknown', 'probability': .62}
    content = client.requests[0][1]['input'][0]['content']
    proofs = [json.loads(part['text'])['same_capture_reference'] for part in content
              if part.get('type') == 'input_text' and part['text'].startswith('{')
              and 'same_capture_reference' in json.loads(part['text'])]
    if expected:
        assert len(proofs) == 1
        assert proofs[0]['verified_camera_ids'] == expected
        assert proofs[0]['observation_id'] == current['observation_id']
        assert proofs[0]['identity_or_visibility_proof'] is False
    else:
        assert proofs == []


class VersionedRobot(DemoRobot):
    def __init__(self):
        super().__init__()
        self.manifest_revision = 1
        self.command_revision = 0

    def robot_snapshot(self):
        result = super().robot_snapshot() | {
            'manifest_revision': self.manifest_revision,
            'command_revision': self.command_revision, 'active_commands': []}
        for arm in result['arms'].values():
            arm['gripper_command_position_rad'] = arm['gripper_position_rad']
        return result

    def capture_rgbd(self):
        result = deepcopy(super().capture_rgbd()) | {
            'manifest_revision': self.manifest_revision,
            'command_revision': self.command_revision, 'active_commands': []}
        for arm in result['arms'].values():
            arm['gripper_command_position_rad'] = arm['gripper_position_rad']
        return result


async def prepared(tmp_path):
    runner = TestPolicyController(VersionedRobot(), DemoVision(1), None, tmp_path,
                                  planning_settings=TEST_SETTINGS)
    runner.state['generation'] = 1
    context = {'planning_settings': TEST_SETTINGS.model_dump()}
    frame = await runner.capture_observation(context)
    context.update(latest_capture=frame, post_place_capture=frame)
    return runner, context, frame


@pytest.mark.parametrize('change', ['none', 'before_release', 'motion', 'epoch'])
async def test_release_recheck_reuses_only_unchanged_post_release_capture(tmp_path, change):
    from luna_mujoco.robot_skills.observation import reusable_release_capture

    runner, context, frame = await prepared(tmp_path)
    frame['captured_perf_counter_ns'] = 200
    binding = {'release_verified_perf_counter_ns': 100, 'release_feedback_time': frame['time'] - 1}
    if change == 'before_release':
        binding['release_verified_perf_counter_ns'] = 201
    elif change == 'motion':
        runner.simulation.command_revision += 1
    elif change == 'epoch':
        runner.observation_epoch += 1
    result = reusable_release_capture(context, runner.simulation.robot_snapshot(),
        epoch=runner.observation_epoch, binding=binding)
    assert (result is frame) == (change == 'none')


@pytest.mark.parametrize('failed', [False, True])
async def test_identical_analysis_is_not_repeated_but_new_capture_can_be_analyzed(tmp_path, monkeypatch, failed):
    from types import SimpleNamespace

    from luna_mujoco.runner import Runner

    runner, context, frame = await prepared(tmp_path)
    runner.task_spec = SimpleNamespace()
    runner.task_inventory = {}
    runner.compiler.configured_regions = {}
    async def analyze(controller, ctx, *, capture, **kwargs):
        if failed:
            raise ValueError('visual_identity_unknown')
        return {'observation_id': capture['observation_id'], 'generation': capture['generation'], 'objects': []}
    analysis = AsyncMock(side_effect=analyze)
    monkeypatch.setattr(Runner, 'observe', analysis)
    if failed:
        with pytest.raises(ValueError, match='visual_identity_unknown'):
            await runner.observe(context, capture=frame)
        with pytest.raises(ValueError, match='observation_unchanged_since_failed_analysis'):
            await runner.observe(context, capture=frame)
    else:
        first = await runner.observe(context, capture=frame)
        assert await runner.observe(context, capture=frame) is first
    assert analysis.await_count == 1
    assert await runner.capture_scene_observation(context, {'capture_mode': 'fresh'}) is frame
    runner.observation_epoch += 1
    current = await runner.capture_scene_observation(context, {})
    assert current['observation_id'] != frame['observation_id']
    if failed:
        with pytest.raises(ValueError, match='visual_identity_unknown'):
            await runner.observe(context, capture=current)
    else:
        await runner.observe(context, capture=current)
    assert analysis.await_count == 2


async def test_post_place_and_post_drop_frame_reused_without_another_capture(tmp_path):
    runner, context, frame = await prepared(tmp_path)
    reused = await runner.capture_scene_observation(context, {})
    assert reused is frame
    assert context['observation_count'] == 1
    event = runner.events[-1]
    assert event['kind'] == 'observation_reused'
    assert event['data']['observation_id'] == frame['observation_id']
    assert reused['captured_perf_counter_ns'] == frame['captured_perf_counter_ns']
    context.pop('post_place_capture')  # verify_drop supplies latest_capture only.
    assert await runner.capture_scene_observation(context, {}) is frame
    assert context['observation_count'] == 1


@pytest.mark.parametrize('change', ['epoch', 'manifest', 'command', 'expired', 'future',
                                  'joints', 'gripper', 'active', 'missing_version',
                                  'missing_epoch', 'missing_camera', 'physics_error', 'force_fresh'])
async def test_changed_or_unverifiable_evidence_requires_new_capture(tmp_path, change):
    runner, context, frame = await prepared(tmp_path)
    arguments = {}
    if change == 'epoch':
        runner.observation_epoch += 1
    elif change == 'manifest':
        runner.simulation.manifest_revision += 1
    elif change == 'command':
        runner.simulation.command_revision += 1  # Includes away-and-back motions.
    elif change == 'expired':
        frame['captured_monotonic_s'] -= 11
    elif change == 'future':
        frame['captured_monotonic_s'] += 1
    elif change == 'joints':
        frame['arms']['B']['joints_rad'][0] += .01
    elif change == 'gripper':
        frame['arms']['A']['gripper_position_rad'] += .01
    elif change == 'active':
        frame['active_commands'] = ['A']
    elif change == 'missing_version':
        frame.pop('command_revision')
    elif change == 'missing_epoch':
        context['capture_epochs'].clear()
    elif change == 'missing_camera':
        frame['views'].pop('1')
    elif change == 'physics_error':
        original = runner.simulation.robot_snapshot
        runner.simulation.robot_snapshot = lambda: original() | {'physics_error': 'test error'}
    else:
        arguments['capture_mode'] = 'fresh'
    result = await runner.capture_scene_observation(context, arguments)
    if change in {'expired', 'force_fresh'}:
        assert result is frame
        assert context['observation_count'] == 1
        return
    assert result['observation_id'] != frame['observation_id']
    assert context['observation_count'] == 2


async def test_new_analysis_does_not_keep_old_product_capture_constraint(tmp_path):
    runner, context, frame = await prepared(tmp_path)
    context.update(unpacked_grasp_capture=frame, unpacked_grasp_target_id='product',
                   visual_inventory_reference={'product_references': {'product': {'proof': 'retained'}}})
    reference = deepcopy(context['visual_inventory_reference'])
    runner.simulation.command_revision += 1
    await runner.capture_scene_observation(context, {'capture_mode': 'fresh'})
    assert 'unpacked_grasp_capture' not in context
    assert 'unpacked_grasp_target_id' not in context
    assert context['visual_inventory_reference'] == reference


async def test_observe_dispatch_analyzes_reused_pixels_without_using_placement_verdict(tmp_path):
    runner, context, frame = await prepared(tmp_path)
    context['post_place_result'] = {'success': True, 'visual_evidence': {'status': 'at_destination'}}
    observation = {**frame, 'objects': [], 'visual_inventory': {'identity': 'separate analysis'}}
    runner.observe = AsyncMock(return_value=observation)
    runner._check_source_view = lambda result: None
    runner._require_visible_inventory = lambda result: None
    result = await runner.execute('observe_scene', context)
    assert result is observation
    assert context['observation_count'] == 1
    runner.observe.assert_awaited_once_with(context, checkpoint='initial', capture=frame)


async def test_model_catalog_exposes_explicit_fresh_capture_option(tmp_path):
    runner, _, _ = await prepared(tmp_path)
    schema = runner.registry.describe()['observe_scene']['arguments']['properties']
    assert schema['capture_mode']['enum'] == ['reuse_current', 'fresh']
    assert schema['capture_mode']['default'] == 'reuse_current'
    with pytest.raises(ValueError):
        runner.registry.validate_arguments('observe_scene', {'capture_mode': 'ignore_staleness'})


async def test_fresh_request_reuses_unchanged_compilation_frame(tmp_path):
    runner, context, frame = await prepared(tmp_path)
    context['initial_capture'] = frame
    context['visual_inventory_reference'] = {'observation_id': frame['observation_id'], 'entities': {'a': {'id': 'a'}}}
    context.pop('analyzed_capture_ids', None)
    reference = deepcopy(context['visual_inventory_reference'])
    current = await runner.capture_scene_observation(context, {'capture_mode': 'fresh'})
    assert current is frame
    assert context['initial_capture'] is frame
    assert context['visual_inventory_reference'] == reference
    assert context['observation_acquisition']['reused'] is True


async def test_completion_reuses_current_return_frame_but_still_judges_goal(tmp_path):
    from test_unpack_final_return import controller

    runner, context, frame = await prepared(tmp_path)
    template, _, _ = controller(tmp_path)
    runner.task_inventory, runner.task_spec = template.task_inventory, template.task_spec
    runner.task_progress = template.task_progress
    runner._judge = AsyncMock(return_value={'judgments': {'goal_satisfied': {'status': 'yes'}}})
    await runner._complete(context)
    assert context['latest_capture'] is frame
    assert context['observation_count'] == 1
    runner._judge.assert_awaited_once()
    assert runner.state['status'] == 'succeeded'


async def test_first_analysis_stays_bound_to_initial_compilation_frame_after_model_latency(tmp_path):
    runner, context, frame = await prepared(tmp_path)
    frame['captured_monotonic_s'] -= 90
    context.update(initial_capture=frame, visual_inventory_reference={'observation_id': frame['observation_id']})
    assert await runner.capture_scene_observation(context, {}) is frame
    assert context['observation_count'] == 1
    context['analyzed_capture_ids'] = {frame['observation_id']}
    assert await runner.capture_scene_observation(context, {}) is frame
    assert context['observation_count'] == 1


async def test_rechecking_unknown_completion_requires_new_pixels(tmp_path):
    from test_unpack_final_return import controller

    runner, context, frame = await prepared(tmp_path)
    template, _, _ = controller(tmp_path)
    runner.task_inventory, runner.task_spec = template.task_inventory, template.task_spec
    runner.task_progress = template.task_progress
    runner._judge = AsyncMock(side_effect=[
        {'judgments': {'goal_satisfied': {'status': 'unknown'},
                       'completion_condition_0': {'status': 'unknown'}}},
        {'judgments': {'goal_satisfied': {'status': 'yes'}}}])
    with pytest.raises(ValueError, match='task_completion_unconfirmed'):
        await runner._complete(context)
    with pytest.raises(ValueError, match='task_completion_unchanged_evidence'):
        await runner._complete(context)
    assert runner._judge.await_count == 1
    runner.simulation.command_revision += 1
    await runner._complete(context)
    assert context['latest_capture']['observation_id'] != frame['observation_id']
    assert context['observation_count'] == 2
    assert runner._judge.await_count == 2


async def test_initial_compilation_frame_invalidated_by_action_is_not_silently_replaced(tmp_path):
    runner, context, frame = await prepared(tmp_path)
    context.update(initial_capture=frame, visual_inventory_reference={'observation_id': frame['observation_id']})
    runner.simulation.command_revision += 1
    with pytest.raises(ValueError, match='initial_reference_capture_invalidated'):
        await runner.capture_scene_observation(context, {})
    assert context['observation_count'] == 1


@pytest.mark.parametrize('answer', ['known', 'unknown', 'current'])
async def test_support_evidence_refresh_uses_current_decisions_without_relabeling_history(tmp_path, answer):
    from test_unpack_final_return import controller

    runner, context, frame = await prepared(tmp_path)
    template, _, _ = controller(tmp_path)
    runner.task_spec, runner.task_progress = template.task_spec, template.task_progress
    runner.task_inventory = template.task_inventory
    old = {'support_entity_id': 'table', 'observation_id': 'prior',
           'source': 'openai_decisions', 'request_hash': 'prior-proof'}
    if answer == 'current': old['observation_id'] = frame['observation_id']
    runner.task_progress.scene_entities = {
        'pink': {'id': 'pink', 'kind': 'object', 'observed_support': deepcopy(old)},
        'table': {'id': 'table', 'kind': 'table'}, 'bin': {'id': 'bin', 'kind': 'bin'}}
    runner.task_progress.records['pink']['observed_support'] = deepcopy(old)
    context['visual_inventory_reference'] = {}
    from test_support_context import example
    context['observation'] = example()[1] | {'observation_id': frame['observation_id'],
        'generation': frame.get('generation'), 'inventory_state': runner.task_progress.inventory_state()}
    context['observation']['objects'][0]['id'] = 'pink'
    runner._judge = AsyncMock(return_value={'judgments': {
        'support_0': {'status': 'no', 'probability': .05},
        'support_1': {'status': 'unknown' if answer == 'unknown' else 'yes',
                      'probability': .5 if answer == 'unknown' else .95}},
        'evidence': {'request_hash': 'current-proof', 'observation_id': frame['observation_id']}})
    if answer == 'unknown':
        with pytest.raises(ValueError, match='observed_support_unknown'):
            await runner._refresh_observed_support(context, 'pink')
        assert runner.task_progress.scene_entities['pink']['observed_support'] == old
    else:
        await runner._refresh_observed_support(context, 'pink')
        current = runner.task_progress.scene_entities['pink']['observed_support']
        if answer == 'known':
            assert current['observation_id'] == frame['observation_id']
            assert current['support_entity_id'] == 'bin'
            assert current['request_hash'] == 'current-proof'
            assert runner.task_progress.records['pink']['observed_support'] == current
            assert context['observation']['inventory_state']['entities']['pink']['observed_support'] == current
        else:
            assert current == old
    assert runner._judge.await_count == (0 if answer == 'current' else 1)
    assert old['request_hash'] == 'prior-proof'
    assert context['observation_count'] == 1


async def test_scene_observation_does_not_require_prior_target_support(tmp_path, monkeypatch):
    from luna_mujoco.runner import Runner
    from luna_mujoco.runtime.controller import PolicyController

    runner, context, frame = await prepared(tmp_path)
    runner.task_spec = object()
    runner.task_inventory = {}
    runner.compiler.configured_regions = {}
    runner._refresh_observed_support = AsyncMock(side_effect=ValueError('observed_support_unknown'))
    monkeypatch.setattr(Runner, 'observe', AsyncMock(return_value={**frame, 'objects': []}))
    result = await PolicyController.observe(runner, context, capture=frame)
    assert result['observation_id'] == frame['observation_id']
    assert runner._refresh_observed_support.await_count == 0


def test_router_uses_nonempty_question_and_typed_edge_outcomes():
    from jsonschema import Draft202012Validator, ValidationError
    from luna_mujoco.agent.schema import InitialPlan, response_schema

    schema = response_schema(InitialPlan, {'observe_scene': {'requires': {}, 'provides': []}})
    question = Draft202012Validator(schema['$defs']['RouterNode']['properties']['condition'])
    question.validate('物体是否已释放并位于目标容器中？')
    with pytest.raises(ValidationError):
        question.validate('')
    outcome = Draft202012Validator(schema['$defs']['Edge']['properties']['on'])
    for value in ('yes', 'no', 'unknown', 'success', 'failure'):
        outcome.validate(value)
    with pytest.raises(ValidationError):
        outcome.validate('probably')


def test_real_simulator_capture_versions_follow_all_actuator_entrypoints():
    from luna_mujoco.simulation import Simulation

    robot = Simulation()
    robot.start()
    try:
        robot.settle()
        initial = robot.robot_snapshot()
        frame = robot.capture_rgbd()
        assert frame['command_revision'] == initial['command_revision'] == 0
        assert frame['manifest_revision'] == initial['manifest_revision']
        assert frame['active_commands'] == []
        assert not {'objects', 'geometry_poses'} & frame.keys()
        arm = initial['arms']['A']['joints_rad']
        robot.execute_trajectory('A', [arm, arm], .02, initial['generation'])
        assert robot.robot_snapshot()['command_revision'] == 1
        current = robot.robot_snapshot()
        paths = {name: [data['joints_rad'], data['joints_rad']] for name, data in current['arms'].items()}
        robot.execute_dual_trajectory(paths, .02, initial['generation'])
        assert robot.robot_snapshot()['command_revision'] == 2
        robot.set_gripper('A', False, initial['generation'])
        assert robot.robot_snapshot()['command_revision'] == 3
        assert frame['command_revision'] == 0
        after = robot.capture_rgbd()
        assert after['command_revision'] == 3
        assert after['observation_id'] != frame['observation_id']
    finally:
        robot.close()


@pytest.mark.parametrize('corruption', [None, 'rgb', 'depth_m', 'time', 'timestamp', 'missing_digest'])
def test_release_identity_can_be_analyzed_on_exact_confirmed_capture(corruption):
    import numpy as np
    from test_demo_completed_tracking import resolve

    from luna_mujoco.demo_completed_tracking import confirmed_capture_record
    from luna_mujoco.visual_api import VisualInventoryError

    def mutate(capture, detections, entities, reference, progress):
        capture.update(observation_id='post-place', time=2., captured_perf_counter_ns=123)
        for view in capture['views'].values():
            view['depth_m'] = np.ones(view['rgb'].shape[:2], dtype=np.float32)
        prior = confirmed_capture_record(capture)
        progress['pink']['completed_identity_track']['capture'] = prior
        if corruption in {'rgb', 'depth_m'}:
            capture['views']['3'][corruption][0, 0] += 1
        elif corruption == 'time':
            capture['time'] += 1
        elif corruption == 'timestamp':
            capture['captured_perf_counter_ns'] += 1
        elif corruption == 'missing_digest':
            prior.pop('sensor_fingerprint')

    if corruption is not None:
        with pytest.raises(VisualInventoryError, match='completed_tracking'):
            resolve(mutate=mutate)
        return
    result, calls, _ = resolve(mutate=mutate)
    assert {g[0]['entity_id'] for g in result['groups']} == {'pink', 'remaining'}
    assert len(calls) == 2  # Correspondence and remaining identities still independently judged.
    assert result['completed_tracking']['observation_id'] == 'post-place'
