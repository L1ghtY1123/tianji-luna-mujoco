import asyncio

import pytest
from test_route_runner import RouteAlgorithms, RouteChoices, RouteRobot

from jev_mujoco.runner import Runner
from jev_mujoco.skills import Registry


class AtomicChoices(RouteChoices):
    async def choose_skill(self, state, registry):
        f = state['facts']
        if f.get('recovery_needed'):
            return 'recover_task'
        for fact, skill in [('observed', 'observe_scene'), ('task_assessed', 'assess_task'),
                            ('target_selected', 'identify_target'), ('destination_ready', 'identify_destination')]:
            if not f.get(fact):
                return skill
        if not f.get('arm_selected'):
            return 'choose_arm' if f.get('arms_evaluated') else 'evaluate_reachability'
        for fact, skill in [('scene_ready', 'build_scene'), ('grasps_ready', 'generate_grasps'),
                            ('plan_ready', 'plan_grasp'), ('lifted', 'execute_grasp'),
                            ('verified', 'verify_grasp'), ('place_plan_ready', 'plan_place'),
                            ('released', 'execute_place')]:
            if not f.get(fact):
                return skill
        return 'verify_place'


class CountedRobot(RouteRobot):
    def __init__(self):
        super().__init__()
        self.captures = 0

    def capture_rgbd(self):
        self.captures += 1
        return super().capture_rgbd()


@pytest.mark.parametrize('next_status,refresh', [('observed', True), ('pending', True),
                                               ('completed', False), ('deferred', False)])
async def test_release_immediately_refreshes_when_another_required_object_remains(tmp_path, next_status, refresh):
    from test_release_verification import release_case

    runner, context = release_case(tmp_path)
    runner.run_id = 'multi-release'
    runner.state['current_step'] = {'skill': 'verify_place'}
    runner.task_progress.inventory_finalized = True
    runner.task_progress.membership.update(target='required', other='required')
    runner.task_progress.records['other'] = {'id': 'other', 'status': next_status}
    calls = []
    confirm = runner.observe

    async def observe(context, *, checkpoint, purpose):
        calls.append(('observe', checkpoint, purpose))
        return await confirm(context, checkpoint=checkpoint, purpose=purpose)

    async def build(context, exclusion_state):
        calls.append(('build', exclusion_state))
        return {}

    runner.observe, runner.build_scene = observe, build
    await runner.execute('verify_place', context)
    assert calls == [('observe', 'post_place', 'scene')] + ([('build', 'observation')] if refresh else [])


@pytest.mark.parametrize('route,rounds', [('box_table', 1), ('table_box', 1), ('box_table_box', 2)])
async def test_route_observes_initial_before_each_grasp_and_each_release(tmp_path, route, rounds):
    robot, algorithms = CountedRobot(), RouteAlgorithms()
    runner = Runner(robot, algorithms, lambda record: AtomicChoices(route), tmp_path)
    await runner.start('搬运同一个目标')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('failure')
    assert robot.captures == 2 * rounds + 1
    assert algorithms.maps == ['observation', 'selected'] * rounds
    assert algorithms.returns == [False] * rounds
    stages = [e for e in runner.events if e['kind'] == 'execution_stage_complete']
    assert sum(e['data']['stage'] == 'prepare_attachment' for e in stages) == rounds
    assert not any(e['kind'] == 'skill_complete' and e['title'] in
                   {'observe_geometry', 'move_to_observe', 'observe_result'} for e in runner.events)
    for record in runner.task_progress.records.values():
        for step in record.get('completed_stages', []):
            assert step['observation_id'].startswith('frame-')


async def test_next_leg_scene_is_measured_before_jev_selects_next_step(tmp_path):
    class RecordedAlgorithms(RouteAlgorithms):
        def __init__(self):
            super().__init__()
            self.cameras = []

        def observe(self, *args, **kwargs):
            self.cameras.append(kwargs['source_hint'].get('camera_ids', ['1', '2', '3']))
            return super().observe(*args, **kwargs)

    algorithms = RecordedAlgorithms()
    runner = Runner(CountedRobot(), algorithms, lambda record: AtomicChoices('box_table_box'), tmp_path)
    await runner.start('搬运同一个目标')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('failure')
    assert algorithms.cameras == [['3']] + [['1', '2', '3']] * 4
    complete = next(i for i, e in enumerate(runner.events)
                    if e['kind'] == 'skill_complete' and e['title'] == 'verify_place')
    before = runner.events[:complete]
    assert any(e['kind'] == 'execution_stage_complete' and e['data']['stage'] == 'post_place_collision_scene'
               for e in before)


async def test_scene_update_failure_cannot_count_one_release_as_two_route_stages(tmp_path):
    from test_release_verification import release_case

    runner, state = release_case(tmp_path)
    runner.run_id = 'release-retry'
    runner.state['current_step'] = {'skill': 'verify_place'}
    runner.task_progress.route = 'box_table_box'
    runner.task_progress.inventory_finalized = True
    runner.task_progress.membership['target'] = 'required'
    state['scene'] = {'exclusion_state': 'selected'}

    async def failed_update(*args, **kwargs):
        raise ValueError('scene_update_failed')

    runner.build_scene = failed_update
    for _ in range(2):
        with pytest.raises(ValueError, match='scene_update_failed'):
            await runner.execute('verify_place', state)
        runner.simulation.state['time'] += 1
    record = runner.task_progress.records['target']
    assert len(record['completed_stages']) == 1
    assert record['stage_index'] == 1 and record['status'] == 'pending'
    assert runner.task_progress.completed_ids == []


def test_retired_skills_cannot_be_selected_by_jev():
    registry = Registry()
    for name in ['observe', 'observe_geometry', 'observe_result', 'move_to_observe',
                 'inspect_visual_result', 'check_scene_changes', 'update_scene',
                 'evaluate_arms', 'plan_grasp_and_place']:
        with pytest.raises(ValueError, match='unknown_skill'):
            registry.validate_selection(name, {})


async def test_failed_fresh_scene_never_leaves_previous_world_executable(tmp_path):
    class Failure(RouteAlgorithms):
        def build_scene(self, observation, target_id, **kwargs):
            if kwargs['exclusion_state'] == 'selected':
                raise RuntimeError('instance_mapping_failed')
            return super().build_scene(observation, target_id, **kwargs)

    runner = Runner(CountedRobot(), Failure(), lambda record: AtomicChoices('box_table'), tmp_path)
    await runner.start('搬运目标')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['error'] == 'instance_mapping_failed'
    assert not runner.state['facts']['scene_ready']
    assert not runner.state['facts'].get('plan_ready')
    assert not any(e['kind'] == 'skill_start' and e['title'] == 'execute_grasp' for e in runner.events)
