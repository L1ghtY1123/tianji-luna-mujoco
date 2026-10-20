import copy

import pytest
from test_planning_recovery import BothArms, Selecting, make_runner
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms
from test_runner import wait

from jev_mujoco.runner import Runner


async def test_reobserve_cannot_bypass_target_action_budget(tmp_path):
    class Reobserve(Selecting):
        async def choose_arm(self, state, evaluations):
            return 'A'

        async def choose_recovery(self, state):
            return 'stop' if state['failure']['code'] == 'recovery_retry_budget_exhausted' else 'reobserve'

    algorithms = BothArms(failed_arms=('A', 'B'))
    runner = make_runner(tmp_path, algorithms, Reobserve())
    await runner.start('抓取黑色物体')
    await wait(runner)
    assert algorithms.planned_arms == ['A', 'A', 'A']
    assert runner.state['error'] == 'recovery_retry_budget_exhausted'


async def test_held_reobservation_does_not_rebuild_scene_or_resume_place(tmp_path):
    class ChangedScene(VisionAlgorithms):
        place_calls = 0

        def __init__(self):
            super().__init__()
            self.maps = []
            self.latest_ids = []

        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            if self.place_calls:
                obstacle = copy.deepcopy(observation['objects'][0])
                obstacle.update(id='new-obstacle', color='green')
                observation['objects'].append(obstacle)
            self.latest_ids = [obj['id'] for obj in observation['objects']]
            return observation

        def build_scene(self, observation, target_id, **kwargs):
            self.maps.append(kwargs['exclusion_state'])
            return super().build_scene(observation, target_id, **kwargs)

        def plan_place(self, observation, *args, **kwargs):
            self.place_calls += 1
            raise ValueError('place_plan_failed')

    class Reobserve(Choices):
        async def choose_skill(self, state, registry):
            if (state.get('skill_rejection')
                    or (state.get('failure') or {}).get('code') == 'grasp_scene_requires_empty_hands'):
                return 'stop'
            if 'new-obstacle' in [obj['id'] for obj in (state.get('observation') or {}).get('objects', [])]:
                return 'build_scene'
            return await super().choose_skill(state, registry)

        async def choose_recovery(self, state):
            return 'reobserve'

    algorithms = ChangedScene()
    runner = Runner(SensorRobot(), algorithms, lambda record: Reobserve(), tmp_path)
    await runner.start('将红色方块放到蓝色盒子里')
    await wait(runner)
    assert runner.state['status'] in {'failed', 'needs_attention'}
    assert algorithms.place_calls == 1
    assert algorithms.maps == ['observation', 'selected']
    assert 'new-obstacle' in algorithms.latest_ids
    assert runner.holding_state['verified']
    assert not runner.state['facts']['scene_ready']


async def test_observation_budget_blocks_sensor_work_without_selecting_recovery(tmp_path):
    algorithms = VisionAlgorithms()
    runner = Runner(SensorRobot(), algorithms, None, tmp_path)
    runner.state.update(generation=1, task='grasp', instruction='抓取红色物体')
    context = {'recovery_capture_count': 8}
    before = runner.state['status']
    with pytest.raises(ValueError, match='observation_budget_exhausted'):
        await runner.observe(context, recovery=True)
    assert algorithms.checkpoints == []
    assert runner.state['status'] == before
