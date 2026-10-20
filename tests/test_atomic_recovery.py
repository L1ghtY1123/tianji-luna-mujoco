import asyncio

import pytest
from test_route_runner import RouteAlgorithms, RouteChoices, RouteRobot

from jev_mujoco.algorithms import AlgorithmError
from jev_mujoco.runner import Runner


@pytest.mark.parametrize('requested_arm', ['auto', 'A'])
async def test_explicit_plan_retry_preserves_stage_arm_and_existing_scene(tmp_path, requested_arm):
    class FirstFailure(RouteAlgorithms):
        planned_arms = None

        def __init__(self):
            super().__init__()
            self.planned_arms = []

        def plan_grasp(self, observation, target, arm, *args, **kwargs):
            self.planned_arms.append(arm)
            if len(self.planned_arms) == 1:
                raise AlgorithmError('no first path', code='planning_no_valid_candidate')
            return super().plan_grasp(observation, target, arm, *args, **kwargs)

    class Retry(RouteChoices):
        async def bind_request(self, instruction):
            return {**await super().bind_request(instruction), 'arm': requested_arm}

        async def choose_recovery(self, state):
            return 'retry_plan'

    algorithms = FirstFailure()
    runner = Runner(RouteRobot(), algorithms, lambda record: Retry('box_table'), tmp_path)
    await runner.start('move the selected object to the table')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert algorithms.planned_arms == ['A', 'A']
    assert algorithms.maps == ['observation', 'selected']
    assert algorithms.checkpoints == ['initial', 'fine', 'post_place']
    assert len([event for event in runner.events
                if event['kind'] == 'skill_complete' and event['title'] == 'recover_task']) == 1


@pytest.mark.parametrize('already_attached', [False, True])
async def test_place_attachment_requires_current_holding_feedback(tmp_path, already_attached):
    from test_current_grasp_attachment import fixture

    runner, context, current, prepared, _, _ = fixture(tmp_path)
    await runner.verify_grasp(context)
    if already_attached:
        await runner.prepare_attachment(context)
    before = len(prepared)
    current['arms']['A']['gripper_closed_command'] = False
    with pytest.raises(ValueError, match='attachment_requires_current_holding_feedback'):
        await runner.prepare_attachment(context)
    assert len(prepared) == before


@pytest.mark.parametrize('failure_stage', ['initial_map', 'fine_observation', 'instance_map'])
async def test_explicit_reobserve_recovers_scene_preparation_without_losing_motion_world(tmp_path, failure_stage):
    class Transient(RouteAlgorithms):
        failed = False
        live_scene_token = None

        def observe(self, *args, **kwargs):
            if failure_stage == 'fine_observation' and kwargs.get('checkpoint') == 'fine' and not self.failed:
                self.failed = True
                raise RuntimeError('transient_mapping_error')
            return super().observe(*args, **kwargs)

        def build_scene(self, observation, target_id, **kwargs):
            stage = kwargs['exclusion_state']
            fail_now = (failure_stage == 'initial_map' and stage == 'observation'
                        or failure_stage == 'instance_map' and stage == 'selected')
            if fail_now and not self.failed:
                self.failed = True
                self.live_scene_token = None  # Real service invalidates the world on failure.
                raise RuntimeError('transient_mapping_error')
            result = super().build_scene(observation, target_id, **kwargs)
            self.live_scene_token = result['scene_token']
            return result

        def plan_dual_observation_motion(self, observation, scene, goals, **kwargs):
            assert scene['scene_token'] == self.live_scene_token
            assert scene['exclusion_state'] == 'observation' or scene.get('return_after_place')
            return super().plan_dual_observation_motion(observation, scene, goals, **kwargs)

    class Reobserve(RouteChoices):
        recovery_states = None

        def __init__(self):
            super().__init__('box_table')
            self.recovery_states = []

        async def choose_recovery(self, state):
            self.recovery_states.append(state)
            return 'reobserve'

    algorithms, choices = Transient(), Reobserve()
    runner = Runner(RouteRobot(), algorithms, lambda record: choices, tmp_path)
    await runner.start('move the selected object to the table')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert len(choices.recovery_states) == 1
    state = choices.recovery_states[0]
    assert not state['facts']['scene_ready'] and not state['facts'].get('plan_ready')
    assert state['recovery_context']['scene_available'] == (failure_stage != 'initial_map')
    expected_maps = (['observation', 'selected'] if failure_stage == 'initial_map'
                     else ['observation', 'observation', 'selected'])
    assert algorithms.maps == expected_maps
    assert algorithms.prepare_count == 1


async def test_recovery_snapshot_exposes_bound_arm_and_frozen_inputs(tmp_path):
    from test_planning_recovery import BothArms, Selecting, make_runner

    runner = make_runner(tmp_path, BothArms(), Selecting(stop_after_failure=True))
    await runner.start('grasp the selected object')
    await asyncio.wait_for(runner.worker, 5)
    context = {'scene': {'scene_token': 'frozen'}, 'grasps': {'candidates': [1]}}
    runner.state['facts']['scene_ready'] = True
    evidence = runner.snapshot_for_decision(context)['recovery_context']
    assert evidence['stage_arm_fixed'] and evidence['bound_arm'] == 'A'
    assert evidence['grasp_candidates_available'] and evidence['scene_current']
    assert evidence['retry_plan_scope'] == 'same_bound_arm'


async def test_empty_grasp_retry_rebuilds_target_inclusive_motion_world(tmp_path):
    class EmptyFirst(RouteRobot):
        closures = 0

        def set_gripper(self, arm, closed, generation):
            if closed:
                self.closures += 1
            return super().set_gripper(arm, closed, generation)

        def robot_snapshot(self):
            feedback = super().robot_snapshot()
            if self.closed and self.closures == 1:
                feedback['arms']['A'].update(gripper_position_rad=.8,
                    gripper_position_samples=[[t, .8] for t in (0., .05, .1, .15, .2, .25)])
            return feedback

    class StrictWorld(RouteAlgorithms):
        def plan_dual_observation_motion(self, observation, scene, goals, **kwargs):
            assert scene['exclusion_state'] == 'observation' or scene.get('return_after_place')
            return super().plan_dual_observation_motion(observation, scene, goals, **kwargs)

    class RetryGrasp(RouteChoices):
        async def choose_recovery(self, state):
            return 'retry_grasp'

    algorithms, robot = StrictWorld(), EmptyFirst()
    runner = Runner(robot, algorithms, lambda record: RetryGrasp('box_table'), tmp_path)
    await runner.start('move the selected object to the table')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert robot.closures == 2
    assert algorithms.maps == ['observation', 'selected', 'observation', 'selected']


@pytest.mark.parametrize('bootstrap_failure', [False, True])
async def test_collection_defer_prepares_target_inclusive_world_for_next_target(tmp_path, bootstrap_failure):
    from test_collection_runner import CollectionChoices, CollectionVision
    from test_rgbd_workflow import SensorRobot

    class FailedSource(CollectionVision):
        failed_bootstrap = False
        planned_targets = None

        def __init__(self):
            super().__init__()
            self.planned_targets = []

        def plan_grasp(self, observation, target, *args, **kwargs):
            self.planned_targets.append(target)
            if target == 'source':
                raise AlgorithmError('source has no path', code='planning_no_valid_candidate')
            return super().plan_grasp(observation, target, *args, **kwargs)

        def build_scene(self, observation, target_id, **kwargs):
            if (bootstrap_failure and self.planned_targets and kwargs['exclusion_state'] == 'observation'
                    and not self.failed_bootstrap):
                self.failed_bootstrap = True
                raise RuntimeError('transient_recovery_bootstrap_error')
            return super().build_scene(observation, target_id, **kwargs)

        def plan_dual_observation_motion(self, observation, scene, goals, **kwargs):
            assert scene['exclusion_state'] == 'observation' or scene.get('return_after_place')
            return super().plan_dual_observation_motion(observation, scene, goals, **kwargs)

    class Defer(CollectionChoices):
        async def choose_recovery(self, state):
            return ('reobserve' if state['failure']['code'] == 'transient_recovery_bootstrap_error'
                    else 'defer')

        async def choose_target(self, state, objects):
            unavailable = {item['id'] for key in ('completed', 'deferred')
                           for item in state['task_progress'][key]}
            return next(identifier for identifier in self.selected if identifier not in unavailable)

        async def assess_task(self, state):
            answer = await super().assess_task(state)
            progress = state['task_progress']
            if len(progress['completed']) + len(progress['deferred']) == len(self.selected):
                answer['judgment'] = 'complete'
            return answer

    algorithms = FailedSource()
    runner = Runner(SensorRobot(), algorithms, lambda record: Defer(('source', 'second')), tmp_path)
    await runner.start('move both selected objects, defer an infeasible object')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'partial_success', runner.state.get('error')
    assert runner.task_progress.completed_ids == ['second']
    assert [item['id'] for item in runner.task_progress.snapshot()['deferred']] == ['source']
    failures = [event['data']['code'] for event in runner.events if event['kind'] == 'task_failure']
    assert algorithms.planned_targets == ['source', 'second']
    assert failures == (['planning_no_valid_candidate', 'transient_recovery_bootstrap_error']
                        if bootstrap_failure else ['planning_no_valid_candidate'])


async def test_recovery_question_allows_explicit_same_arm_retry_with_requested_arm():
    from jev_mujoco.decisions import Decisions

    received = []
    engine = Decisions(None, lambda *args: None)

    async def ask(kind, state, questions, **kwargs):
        received.append((state, questions))
        return {'recovery': 'retry_plan'}

    engine.ask = ask
    state = {'requested_arm': 'A', 'execution_arm': 'A',
             'recovery_context': {'bound_arm': 'A', 'scene_current': True,
                                  'grasp_candidates_available': True, 'retry_plan_scope': 'same_bound_arm'}}
    assert await engine.choose_recovery(state) == 'retry_plan'
    supplied, questions = received[0]
    assert supplied['requested_arm'] == 'A' and supplied['recovery_context'] == state['recovery_context']
    instructions = questions['recovery']['instructions']
    criterion = questions['recovery']['criteria']['retry_plan']
    assert '旧流程' not in instructions
    assert '用户未指定臂' not in criterion and '未尝试的臂' not in criterion
    assert '同一阶段绑定臂' in criterion and '不得换臂' in criterion
    assert 'attachment已验证' not in questions['recovery']['criteria']['retry_place']
