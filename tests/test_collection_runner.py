import copy

import numpy as np
from test_rgbd_workflow import Choices, VisionAlgorithms


class CollectionVision(VisionAlgorithms):
    def __init__(self):
        super().__init__(membership_status='unknown')
        self.selected = 'source'
        self.scenes = []

    def observe(self, capture, **kwargs):
        observation = super().observe(capture, **kwargs)
        source = observation['objects'][0]
        source['id'] = self.selected
        other = copy.deepcopy(source)
        other.update(id='second' if self.selected == 'source' else 'source', color='green')
        other['position'] = [.45, -.1, .65]
        other['points_world_m'] = np.asarray(other['points_world_m']) + [0., .3, 0.]
        observation['objects'].append(other)
        observation['visual_inventory']['membership'] = {
            obj['id']: 'excluded' if obj['kind'] == 'bin' else 'unknown' for obj in observation['objects']}
        for view in observation['views'].values():
            mask = view['instance_masks'].pop('source')
            view['instance_masks'][self.selected] = mask
            view['instance_masks'][other['id']] = mask.copy()
        return observation

    def build_scene(self, observation, target_id, **kwargs):
        if target_id is not None:
            self.selected = target_id
        result = super().build_scene(observation, target_id, **kwargs)
        self.scenes.append(copy.deepcopy(result))
        return result

    def plan_grasp(self, snapshot, *args, **kwargs):
        result = super().plan_grasp(snapshot, *args, **kwargs)
        result['phases']['approach']['positions_rad'][0] = snapshot['arms']['A']['joints_rad']
        return result


class CollectionChoices(Choices):
    def __init__(self, selected):
        super().__init__()
        self.selected = selected
        self.assessments = []

    async def bind_request(self, instruction):
        return {**await super().bind_request(instruction), 'scope': 'collection'}

    async def assess_task(self, state):
        self.assessments.append(copy.deepcopy(state))
        completed = {item['id'] for item in state['task_progress']['completed']}
        return {'judgment': 'complete' if completed == set(self.selected) else 'continue',
                'membership': {key: 'required' if key in self.selected else 'excluded'
                               for key in state['task_progress']['membership_pending_ids']}}

    async def choose_target(self, state, objects):
        completed = {item['id'] for item in state['task_progress']['completed']}
        return next(identifier for identifier in self.selected if identifier not in completed)

    async def choose_skill(self, state, registry):
        return await super().choose_skill(state, registry)

    async def choose_scene_update(self, state):
        return 'transfer' if state['facts'].get('transfer_pending') else 'select'
