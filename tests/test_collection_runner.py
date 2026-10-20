import asyncio
import copy

import numpy as np
import pytest
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from jev_mujoco.runner import Runner


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


@pytest.mark.parametrize('selected', [('source',), ('source', 'second')])
async def test_jev_selects_subset_and_rebuilds_before_each_grasp(tmp_path, selected):
    choices = CollectionChoices(selected)
    algorithms = CollectionVision()
    runner = Runner(SensorRobot(), algorithms, lambda record: choices, tmp_path)
    await runner.start('仅把指定物体放进盒子；未指定物体不要动')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert [item['id'] for item in runner.state['task_progress']['completed']] == list(selected)
    assert {item['id'] for item in runner.state['task_progress']['objects']} == set(selected)
    assert choices.assessments[0]['task_progress']['objects'] == []
    assert set(choices.assessments[0]['task_progress']['membership_pending_ids']) == {'source', 'second'}
    assert [scene['target_id'] for scene in algorithms.scenes] == [item for target in selected for item in (None, target)]
    assert len(choices.assessments) == len(selected) + 1
    assert all(not state['facts'].get('scene_ready') and not state['facts'].get('plan_ready')
               for state in choices.assessments)
    assert runner.holding_state is None
    # A coarse single-color query must never hide another requested group.
    assert all(hint['kind'] == hint['color'] == 'any' for hint in algorithms.source_hints)


@pytest.mark.parametrize('status', ['elsewhere', 'unknown'])
async def test_release_without_visual_confirmation_does_not_complete_transfer(tmp_path, status):
    class UnconfirmedVision(CollectionVision):
        def observe(self, *args, **kwargs):
            result = super().observe(*args, **kwargs)
            if kwargs.get('checkpoint') == 'post_place':
                result['visual_inventory']['progress']['source']['status'] = status
            return result

    choices = CollectionChoices(('source',))
    algorithms = UnconfirmedVision()
    runner = Runner(SensorRobot(), algorithms, lambda record: choices, tmp_path)
    await runner.start('只把一个物体放进盒子')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'needs_attention', runner.state.get('error')
    assert runner.task_progress.completed_ids == []
    assert runner.task_progress.records['source']['destination_visual_evidence']['status'] == status
