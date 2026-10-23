
import pytest
from test_route_runner import RouteChoices, RouteRobot


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
