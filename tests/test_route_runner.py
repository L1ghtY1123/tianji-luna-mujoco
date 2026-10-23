from copy import deepcopy

import numpy as np
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from luna_mujoco.runner import Runner
from luna_mujoco.task_progress import TaskProgress


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
