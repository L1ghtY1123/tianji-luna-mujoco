import copy

import pytest
from test_runner import Decisions, Sim, wait
from test_visible_workflow import VisibleAlgorithms

from jev_mujoco.runner import Runner
from jev_mujoco.skills import SkillSelectionError


class RequestDecisions(Decisions):
    async def bind_request(self, instruction):
        return {'task': 'grasp', 'kind': 'any', 'color': 'black', 'position': 'any',
                'arm': 'A', 'scope': 'single', 'visual_semantics': 'optional',
                'destination_kind': 'any', 'destination_color': 'any', 'destination_position': 'any'}


@pytest.mark.parametrize('choice', ['retry_grasp', 'stop'])
async def test_empty_grasp_returns_to_jev_and_obeys_its_choice(tmp_path, choice):
    class EmptyFirst(Sim):
        def __init__(self):
            super().__init__()
            self.closures = 0

        def set_gripper(self, arm, closed, generation):
            result = super().set_gripper(arm, closed, generation)
            if closed:
                self.closures += 1
                if self.closures == 1:
                    self.state['arms'][arm]['gripper_position_rad'] = .8
            return result

    class Choices(RequestDecisions):
        def __init__(self):
            self.failures = []

        async def choose_skill(self, state, registry):
            if state['facts'].get('recovery_needed'):
                return 'recover_task'
            return await super().choose_skill(state, registry)

        async def choose_recovery(self, state):
            self.failures.append(copy.deepcopy(state['failure']))
            return choice

        async def choose_arm(self, *args):
            raise AssertionError('explicit arm must not call JEV arm selection')

    simulation, decisions = EmptyFirst(), Choices()
    runner = Runner(simulation, VisibleAlgorithms(), lambda record: decisions, tmp_path)
    await runner.start('用左臂抓取黑色物体')
    await wait(runner)
    assert len(decisions.failures) == 1
    assert decisions.failures[0]['stage'] == 'verify_grasp'
    assert simulation.closures == (2 if choice == 'retry_grasp' else 1)
    assert runner.state['status'] == ('succeeded' if choice == 'retry_grasp' else 'failed'), runner.state


async def test_recovery_cannot_discard_an_object_still_held(tmp_path):
    from jev_mujoco.task_progress import TaskProgress

    runner = Runner(Sim(), VisibleAlgorithms(), None, tmp_path)
    runner.state.update(arm='A', target_id='black', generation=1)
    runner.task_progress = TaskProgress('全部放进去', 'collection', None)
    runner.task_progress.select({'id': 'black'}, 'bin')
    runner.task_progress.fail('black', 'plan_place', 'no_path', {})
    context = {'holding': {'attached': True, 'verified': True, 'arm': 'A'}, 'planning_settings': {}}
    with pytest.raises(ValueError, match='requires_confirmed_empty_hand'):
        await runner.recover_task(context, 'defer')
    assert runner.task_progress.snapshot()['deferred'] == []


async def test_jev_selected_stop_preserves_unknown_identity_status(tmp_path):
    from jev_mujoco.task_progress import TaskProgress

    runner = Runner(Sim(), VisibleAlgorithms(), None, tmp_path)
    runner.task_progress = TaskProgress('抓取红色物体', 'single', 'A')
    evidence = {'error': 'instance association ambiguous', 'details': {
        'status': 'unknown', 'observation_id': 'new-frame', 'previous_ids': ['red'],
        'candidates': [{'id': 'red-a'}, {'id': 'red-b'}]}}
    runner.record_failure({}, 'observe', 'instance association ambiguous', evidence)
    assert runner.task_progress.failure['evidence']['status'] == 'unknown'
    assert runner.state['status'] != 'needs_attention'  # Recording evidence cannot select a recovery.
    result = await runner.recover_task({}, 'stop')
    assert result['status'] == 'needs_attention'


@pytest.mark.parametrize('stage,refresh,keep_scene', [
    ('plan_place', False, True), ('plan_place', True, False), ('execute_place', False, False),
])
async def test_explicit_place_retry_keeps_only_an_unchanged_valid_planning_scene(tmp_path, stage, refresh, keep_scene):
    from jev_mujoco.task_progress import TaskProgress

    runner = Runner(Sim(), VisibleAlgorithms(), None, tmp_path)
    runner.state.update(arm='A', target_id='black', generation=1)
    runner.state['facts'].update(scene_ready=True, observed=True, recovery_needed=True)
    runner.task_progress = TaskProgress('放入盒内', 'single', 'A')
    runner.task_progress.fail('black', stage, 'CUROBO_CUDA_OUT_OF_MEMORY', {})
    context = {'holding': {'attached': True, 'verified': True, 'arm': 'A'},
        'scene': {'scene_token': 'same', 'observation_id': 'frame'},
        'scene_observation': {'observation_id': 'frame'}, 'observation': {'observation_id': 'frame'},
        'scene_refresh_required': refresh}
    if keep_scene:
        result = await runner.recover_task(context, 'retry_place')
        assert runner.state['facts']['scene_ready']
        assert runner.state['facts']['observed']
        assert result['scene_reused']
    else:
        failure = copy.deepcopy(runner.task_progress.failure)
        facts = copy.deepcopy(runner.state['facts'])
        with pytest.raises(SkillSelectionError, match='retry_place_requires_saved_grasp_scene'):
            await runner.recover_task(context, 'retry_place')
        assert runner.task_progress.failure == failure
        assert runner.state['facts'] == facts
    assert context['scene']['scene_token'] == 'same'
    assert runner.simulation.frame == 0
