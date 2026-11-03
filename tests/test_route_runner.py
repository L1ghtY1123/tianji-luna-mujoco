from copy import deepcopy

import numpy as np
from rgbd_workflow_fixture import SensorRobot, VisionAlgorithms

from luna_mujoco.runner import Runner


async def test_removed_assessment_cannot_choose_task_membership(tmp_path):
    import pytest
    runner = Runner(RouteRobot(), RouteAlgorithms(), None, tmp_path)
    context = {'observation': {'objects': [{'id': 'inside'}, {'id': 'table-object'}]}}
    before = deepcopy(context)
    with pytest.raises(ValueError, match='unknown_skill:assess_task'):
        await runner.execute('assess_task', context, {'judgment': 'complete'})
    assert context == before
    assert runner.state['status'] == 'idle'


class RouteRobot(SensorRobot):
    def __init__(self):
        super().__init__()
        self.peer_joints = [0.] * 7

    def robot_snapshot(self):
        state = super().robot_snapshot()
        state['arms']['B']['joints_rad'] = self.peer_joints[:]
        return state

    def execute_dual_trajectory(self, paths, dt, generation, *, execution_settings=None):
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
