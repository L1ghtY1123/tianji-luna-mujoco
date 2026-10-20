import asyncio
from copy import deepcopy

import numpy as np
import pytest
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from jev_mujoco.runner import Runner
from jev_mujoco.task_progress import TaskProgress


async def test_assessment_binds_task_membership_without_filtering_observed_obstacles(tmp_path):
    runner = Runner(RouteRobot(), RouteAlgorithms(), lambda record: None, tmp_path)
    runner.task_progress = TaskProgress('把盒子中物体放到桌面黄色区域', 'collection', route='box_table')
    observation = {'observation_id': 'head', 'objects': [
        {'id': 'inside', 'kind': 'cube', 'position': [.4, 0., .65], 'measured_relationships': [
            {'relation': 'above_observed_bin_floor', 'object_id': 'source-box', 'visible_xy_fraction': 1.}]},
        {'id': 'table-object', 'kind': 'cube', 'position': [.5, .2, .65]}],
        'views': {'3': {'instance_masks': {'inside': np.ones((2, 2), dtype=bool),
                                         'table-object': np.ones((2, 2), dtype=bool)}}}}
    runner.task_progress.observe(observation['objects'], 'head')
    runner.task_progress.assign_display_ids(observation)
    runner.state.update(target_id=None, arm=None, route='box_table',
                        object_labels={'items': [{'id': obj['id']} for obj in observation['objects']]})
    result = await runner.execute('assess_task', {'observation': observation}, {
        'judgment': 'continue', 'membership': {'inside': 'required', 'table-object': 'excluded'}})
    assert [obj['id'] for obj in result['task_progress']['objects']] == ['inside']
    assert [obj['id'] for obj in observation['objects']] == ['inside', 'table-object']
    assert [obj['id'] for obj in runner.state['object_labels']['items']] == ['inside', 'table-object']
    assert runner.state['object_labels']['items'][1]['role'] == 'obstacle'
    assert observation['objects'][1]['display_id'] == 2
    assert runner.task_progress.scene_entities['table-object']['display_id'] == 2


async def test_new_observation_allows_jev_to_reassess_unknown_membership(tmp_path):
    class ReassessChoices(RouteChoices):
        def __init__(self):
            super().__init__('box_table')
            self.assessments = []
            self.observation_count = 0

        async def choose_skill(self, state, registry):
            if len(self.assessments) == 2:
                return 'stop'
            if state['facts'].get('recovery_needed'):
                return 'recover_task'
            if not state['facts'].get('observed'):
                return 'observe_scene'
            return 'assess_task'

        async def choose_recovery(self, state):
            return 'reobserve'

        async def assess_task(self, state):
            self.assessments.append(state)
            return {'judgment': 'unknown' if len(self.assessments) == 1 else 'continue', 'membership': {
                'source': 'unknown' if len(self.assessments) == 1 else 'required'}}

    choices = ReassessChoices()
    algorithms = RouteAlgorithms()
    algorithms.membership_status = 'unknown'
    runner = Runner(RouteRobot(), algorithms, lambda record: choices, tmp_path)
    await runner.start('确认源物体后搬运')
    await asyncio.wait_for(runner.worker, 5)
    assert len(choices.assessments) == 2
    assert runner.task_progress.snapshot()['objects'][0]['id'] == 'source'
    assert not any(event['kind'] == 'skill_rejected' for event in runner.events)


class RouteRobot(SensorRobot):
    def __init__(self):
        super().__init__()
        self.peer_joints = [0.] * 7

    def robot_snapshot(self):
        state = super().robot_snapshot()
        state['arms']['B']['joints_rad'] = self.peer_joints[:]
        return state

    def execute_dual_trajectory(self, paths, dt, generation):
        self.joints, self.peer_joints = paths['A'][-1], paths['B'][-1]
        return {'completed': True}


class RouteAlgorithms(VisionAlgorithms):
    def __init__(self):
        super().__init__()
        self.maps, self.returns = [], []

    def observe(self, *args, **kwargs):
        observation = super().observe(*args, **kwargs)
        observation['objects'][0]['measured_relationships'] = [
            {'relation': 'above_observed_bin_floor', 'object_id': 'source-box', 'visible_xy_fraction': 1.}]
        marker = deepcopy(observation['objects'][1])
        marker.update(id='marker-yellow', kind='marker', color='yellow')
        marker['placement_reference']['region_polygon_world_xy_m'] = [[.52, -.26], [.68, -.26], [.68, -.14], [.52, -.14]]
        observation['objects'].append(marker)
        observation['visual_inventory']['membership'][marker['id']] = 'excluded'
        for view in observation['views'].values():
            view['depth_m'] = np.full((4, 4), view['depth_m'][0, 0])
            view['valid_depth'] = np.ones((4, 4), dtype=bool)
            view['instance_masks'] = {key: np.ones((4, 4), dtype=bool) for key in view['instance_masks']}
            view['instance_masks']['marker-yellow'] = view['instance_masks']['destination'].copy()
        return observation

    def build_scene(self, observation, target_id, **kwargs):
        self.maps.append(kwargs['exclusion_state'])
        return super().build_scene(observation, target_id, **kwargs)

    def evaluate_reachability(self, *args, **kwargs):
        return {'A': {'reachable': True, 'feasibility_scope': 'center_position_ik_only'}}

    def evaluate_arms(self, observation, target, grasps, scene, **kwargs):
        assert kwargs['destination_id'] in {'marker-yellow', 'destination'}
        assert kwargs['requested_arm'] == 'A'
        return super().evaluate_arms(observation, target, grasps, scene, **kwargs)

    def plan_dual_observation_motion(self, observation, scene, goals, **kwargs):
        self.returns.append(bool(scene.get('return_after_place')))
        return {'paths': {arm: [observation['arms'][arm]['joints_rad'], goals[arm]] for arm in ('A', 'B')}, 'dt': .02}

    def validate_dual_observation(self, *args, **kwargs):
        return {'collision_free': True}

    def plan_grasp(self, observation, *args, **kwargs):
        plan = super().plan_grasp(observation, *args, **kwargs)
        plan['phases']['approach']['positions_rad'][0] = observation['arms']['A']['joints_rad']
        return plan


class RouteChoices(Choices):
    def __init__(self, route):
        super().__init__()
        self.route = route

    async def bind_request(self, instruction):
        return {**await super().bind_request(instruction), 'route': self.route}

    async def assess_task(self, state):
        return {'judgment': 'complete' if state['task_progress']['completed'] else 'continue',
                'membership': {key: 'required' for key in state['task_progress']['membership_pending_ids']}}

    async def choose_destination(self, state, objects):
        return ('marker-yellow' if self.route == 'box_table' or
                self.route == 'box_table_box' and state['task_progress']['stage_index'] == 0 else 'destination')

    async def choose_skill(self, state, registry):
        f = state['facts']
        if f.get('recovery_needed'):
            return 'recover_task'
        if not f.get('observed'):
            return 'observe_scene'
        if not f.get('task_assessed'):
            return 'assess_task'
        if not f.get('target_selected'):
            return 'identify_target'
        if not f.get('destination_ready'):
            return 'identify_destination'
        if not f.get('arm_selected'):
            return 'choose_arm' if f.get('arms_evaluated') else 'evaluate_reachability'
        if not f.get('scene_ready'):
            return 'build_scene'
        if not f.get('grasps_ready'):
            return 'generate_grasps'
        if not f.get('plan_ready'):
            return 'plan_grasp'
        if not f.get('lifted'):
            return 'execute_grasp'
        if not f.get('verified'):
            return 'verify_grasp'
        if not f.get('place_plan_ready'):
            return 'plan_place'
        if not f.get('released'):
            return 'execute_place'
        return 'verify_place'


@pytest.mark.parametrize('route,rounds', [('box_table', 1), ('table_box', 1), ('box_table_box', 2)])
async def test_route_runs_through_real_runner_with_fresh_map_each_round(tmp_path, route, rounds):
    algorithms, choices = RouteAlgorithms(), RouteChoices(route)
    runner = Runner(RouteRobot(), algorithms, lambda record: choices, tmp_path)
    await runner.start('搬运同一物体')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', (runner.state.get('error'), runner.state.get('failure'))
    assert algorithms.maps == ['observation', 'selected'] * rounds
    assert algorithms.returns == [False] * rounds
    assert runner.task_progress.completed_ids == ['source']
    record = runner.task_progress.records['source']
    assert len(record['completed_stages']) == rounds
    assert algorithms.prepare_count == algorithms.clear_count == rounds
    skills = [e['title'] for e in runner.events if e['kind'] == 'skill_complete']
    assert skills[-1] == 'assess_task'
    assert not set(skills) & {'update_scene', 'check_scene_changes'}


async def test_jev_argument_decisions_precede_skill_execution(tmp_path):
    class RecordedChoices(RouteChoices):
        async def choose_target(self, state, objects):
            runner.emit('jev', '选择目标', {'duration_s': .01})
            return await super().choose_target(state, objects)

        async def choose_destination(self, state, objects):
            runner.emit('jev', '选择目的地', {'duration_s': .01})
            return await super().choose_destination(state, objects)

    runner = Runner(RouteRobot(), RouteAlgorithms(), lambda record: RecordedChoices('box_table'), tmp_path)
    await runner.start('把盒子中的物体放到黄色区域')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded'
    active = None
    decisions = 0
    for event in runner.events:
        if event['kind'] == 'skill_start':
            assert active is None
            active = event['title']
        elif event['kind'] == 'skill_complete':
            assert active == event['title']
            active = None
        elif event['kind'] == 'jev':
            decisions += 1
            assert active is None, f'JEV interrupted {active}'
            assert event.get('parent_invocation_id') is None
    assert decisions == 2
    assert active is None


async def test_failed_parameter_selection_does_not_start_identify_or_grasp_scene(tmp_path):
    class FailingChoices(RouteChoices):
        async def choose_target(self, state, objects):
            raise ValueError('decision_service_unavailable')

        async def choose_skill(self, state, registry):
            return 'stop' if state['facts'].get('recovery_needed') else await super().choose_skill(state, registry)

    algorithms = RouteAlgorithms()
    runner = Runner(RouteRobot(), algorithms, lambda record: FailingChoices('box_table'), tmp_path)
    await runner.start('把盒子中的物体放到黄色区域')
    await asyncio.wait_for(runner.worker, 5)
    assert algorithms.maps == []
    assert not any(e['kind'] == 'skill_start' and e['title'] == 'build_scene' for e in runner.events)
    error = next(e for e in runner.events if e['kind'] == 'skill_error')
    assert error['data']['phase'] == 'argument_selection'
    assert error['data']['for_skill'] == 'identify_target'
    assert error['data']['executed'] is False
    assert error.get('invocation_id') is None


async def test_box_task_finishes_after_member_without_moving_table_obstacles(tmp_path):
    class WithTableObjects(RouteAlgorithms):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            observation['objects'][0]['color'] = 'green'
            for color in ('red', 'black'):
                obstacle = deepcopy(observation['objects'][0])
                obstacle.update(id=color + '-on-table', color=color, measured_relationships=[])
                observation['objects'].append(obstacle)
                observation['visual_inventory']['membership'][obstacle['id']] = 'unknown'
            return observation

        def build_scene(self, observation, target_id, **kwargs):
            assert {'red-on-table', 'black-on-table'} <= {obj['id'] for obj in observation['objects']}
            assert target_id in (None, 'source')
            return super().build_scene(observation, target_id, **kwargs)

    class ScopedChoices(RouteChoices):
        async def assess_task(self, state):
            answer = await super().assess_task(state)
            answer['membership'] = {key: 'required' if key == 'source' else 'excluded'
                                    for key in state['task_progress']['membership_pending_ids']}
            return answer

    algorithms = WithTableObjects()
    runner = Runner(RouteRobot(), algorithms, lambda record: ScopedChoices('box_table'), tmp_path)
    await runner.start('把盒子中的物体拿到桌面黄色区域')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('failure')
    assert runner.task_progress.completed_ids == ['source']
    assert runner.task_progress.membership == {
        'source': 'required', 'red-on-table': 'excluded', 'black-on-table': 'excluded'}
    skills = [e['title'] for e in runner.events if e['kind'] == 'skill_complete']
    assert skills.count('execute_grasp') == skills.count('execute_place') == 1
    assert skills[-1] == 'assess_task'


class MultiRouteAlgorithms(RouteAlgorithms):
    def __init__(self, identical):
        super().__init__()
        self.selected, self.identical = 'source', identical

    def observe(self, *args, **kwargs):
        result = super().observe(*args, **kwargs)
        source = result['objects'][0]
        source['id'] = self.selected
        other = deepcopy(source)
        other.update(id='second' if self.selected == 'source' else 'source',
                     color='red' if self.identical else 'green', position=[.45, -.1, .65])
        other['points_world_m'] = np.asarray(other['points_world_m']) + [0., .3, 0.]
        result['objects'].append(other)
        result['visual_inventory']['membership'][other['id']] = 'required'
        for view in result['views'].values():
            mask = view['instance_masks'].pop('source')
            view['instance_masks'][self.selected] = mask
            view['instance_masks'][other['id']] = mask.copy()
        return result

    def build_scene(self, observation, target_id, **kwargs):
        if target_id:
            self.selected = target_id
        return super().build_scene(observation, target_id, **kwargs)


class MultiRouteChoices(RouteChoices):
    async def bind_request(self, instruction):
        return {**await super().bind_request(instruction), 'scope': 'collection'}

    async def assess_task(self, state):
        return {'judgment': 'complete' if len(state['task_progress']['completed']) == 2 else 'continue',
                'membership': {key: 'required' for key in state['task_progress']['membership_pending_ids']}}

    async def choose_target(self, state, objects):
        completed = {item['id'] for item in state['task_progress']['completed']}
        return next(identifier for identifier in ('source', 'second') if identifier not in completed)


@pytest.mark.parametrize('identical', [False, True])
@pytest.mark.parametrize('route,rounds', [('box_table', 2), ('table_box', 2), ('box_table_box', 4)])
async def test_multiple_objects_keep_id_and_stage_arm_across_fresh_maps(tmp_path, identical, route, rounds):
    algorithms = MultiRouteAlgorithms(identical)
    runner = Runner(RouteRobot(), algorithms, lambda record: MultiRouteChoices(route), tmp_path)
    await runner.start('逐个搬运两个物体')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', (runner.state.get('error'), runner.state.get('failure'))
    assert runner.task_progress.completed_ids == ['source', 'second']
    assert algorithms.maps == ['observation', 'selected'] * rounds
    assert algorithms.returns == [False] * rounds
    assert algorithms.prepare_count == algorithms.clear_count == rounds
    assert all(item['arm'] == 'A' for record in runner.task_progress.records.values()
               for item in record.get('completed_stages', []))
