import copy

import pytest
from test_runner import Algorithms as FakeAlgorithms
from test_runner import Sim

from jev_mujoco.planning_settings import PlanningSettings
from jev_mujoco.runner import Runner


class ObservationAlgorithms(FakeAlgorithms):
    def __init__(self, *, fail_b=False):
        self.map_calls = []
        self.fail_b = fail_b
        self.moves = []
        self.goals = {}

    def observe(self, capture, **kwargs):
        result = super().observe(capture, **kwargs)
        result.setdefault('views', copy.deepcopy(capture['views']))
        cameras = ['3'] + (['2', '1'] if self.moves else [])
        result['objects'][0]['coverage']['camera_ids'] = cameras
        return result

    def build_scene(self, observation, target, *, head_only=False, **kwargs):
        self.map_calls.append((head_only, kwargs['exclusion_state'], observation['observation_id']))
        return super().build_scene(observation, target, **kwargs)

    def plan_dual_observation_motion(self, observation, scene, goals, **kwargs):
        if self.fail_b:
            raise RuntimeError('dual_observation_no_path')
        self.moves.append('AB')
        self.goals = goals
        return {'paths': {arm: [observation['arms'][arm]['joints_rad'], goals[arm]]
                          for arm in ('A', 'B')}, 'dt': .02}

    def validate_dual_observation(self, observation, scene, paths, **kwargs):
        assert len(paths['A']) == len(paths['B'])
        return {'collision_free': True, 'scene_token': scene['scene_token']}


class DualSim(Sim):
    def execute_dual_trajectory(self, paths, dt, generation):
        for arm in ('A', 'B'):
            self.state['arms'][arm]['joints_rad'] = paths[arm][-1]
        return {'synchronized': True}


def context(tmp_path, algorithms):
    sim = DualSim()
    runner = Runner(sim, algorithms, lambda record: None, tmp_path)
    settings = PlanningSettings().model_dump()
    observation = algorithms.observe(sim.capture_rgbd())
    runner.run_id = 'observe-test'
    runner.state.update(status='running', task='grasp', generation=1, target_id='black',
                        instruction='抓取黑色方块', current_step={'skill': 'move_to_observe'},
                        attributes={'kind': 'cube', 'color': 'black'}, planning_settings=settings)
    return runner, {'observation': observation, 'planning_settings': settings,
                    'visual_plan': {'initial': 'initialize_inventory', 'fine': 'initialize_inventory'}}


async def test_observation_uses_one_joint_problem_and_one_atomic_execution(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    settings = PlanningSettings.model_validate({**state['planning_settings'],
        'observation': {'A': [.1] * 7, 'B': [-.1] * 7}})
    state['planning_settings'] = settings.model_dump()
    await runner.build_scene(state, 'observation')
    result = await runner.move_to_observe(state)
    assert result['semantic_processing'] is False
    assert algorithms.moves == ['AB']
    assert result['movements']['synchronized'] is True
    assert algorithms.goals == {'A': [.1] * 7, 'B': [-.1] * 7}
    assert [row[:2] for row in algorithms.map_calls] == [(False, 'observation')]
    assert runner.state['arm'] is None
    assert state['scene']['exclusion_state'] == 'observation'
    assert state['scene_refresh_required']
    assert not runner.state['facts']['scene_ready']
    with pytest.raises(ValueError, match='recovery_requires_fresh_collision_scene'):
        await runner.verify_scene(state)


async def test_failed_joint_plan_moves_neither_arm(tmp_path):
    algorithms = ObservationAlgorithms(fail_b=True)
    runner, state = context(tmp_path, algorithms)
    await runner.build_scene(state, 'observation')
    before = copy.deepcopy(runner.simulation.robot_snapshot())
    with pytest.raises(RuntimeError, match='dual_observation_no_path'):
        await runner.move_to_observe(state)
    after = runner.simulation.robot_snapshot()
    assert after['arms']['A']['joints_rad'] == before['arms']['A']['joints_rad']
    assert after['arms']['B']['joints_rad'] == before['arms']['B']['joints_rad']
    assert not runner.state['facts'].get('views_ready')
    assert runner.state['arm'] is None


async def test_dirty_all_obstacle_map_cannot_authorize_observation_motion(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    await runner.build_scene(state, 'observation')
    state['scene_refresh_required'] = True
    with pytest.raises(ValueError, match='observation_motion_requires_current_map'):
        await runner.move_to_observe(state)
    assert algorithms.moves == []
