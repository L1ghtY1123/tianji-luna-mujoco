"""A new RGBD frame can hide the held object without erasing its identity."""
import asyncio
import copy

from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from jev_mujoco.runner import Runner


async def test_standalone_place_keeps_held_source_when_latest_frame_is_occluded(tmp_path):
    class Judgment(Choices):
        async def choose_skill(self, state, registry):
            if self.task == 'place':
                if state['facts'].get('destination_selected'):
                    return 'stop'
                return 'identify_destination'
            return await super().choose_skill(state, registry)

    judgment = Judgment('grasp')
    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: judgment, tmp_path)
    await runner.start('grasp red cube')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state['error']
    holding = runner.holding_state
    latest = copy.deepcopy(holding['observation'])
    latest['observation_id'] = 'occluded-held-frame'
    latest['objects'] = [obj for obj in latest['objects'] if obj['id'] != 'source']
    for view in latest['views'].values():
        view['instance_masks'].pop('source', None)
    runner.check_holding(latest, holding)
    assert runner.holding_state['verified']
    assert 'source' in runner.holding_state['scene_entities']
    assert any(obj['id'] == 'source' for obj in holding['scene_observation']['objects'])

    judgment.task = 'place'
    await runner.start('place held object in blue bin')
    await asyncio.wait_for(runner.worker, 5)

    assert runner.state['error'] == 'jev_selected_stop'
    assert runner.task_progress.current_id == 'source'
    assert runner.state['facts']['destination_selected']
    assert runner.holding_state['observation']['observation_id'] == 'occluded-held-frame'
    assert all(obj['id'] != 'source' for obj in runner.holding_state['observation']['objects'])


async def test_fresh_feedback_reverification_preserves_held_identity_for_later_place(tmp_path):
    class Robot(SensorRobot):
        uncertain = False

        def robot_snapshot(self):
            feedback = super().robot_snapshot()
            if self.uncertain:
                feedback['arms']['A']['gripper_position_samples'] = []
            return feedback

    class Vision(VisionAlgorithms):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            if kwargs.get('checkpoint') == 'post_grasp':
                observation['objects'] = [obj for obj in observation['objects'] if obj['id'] != 'source']
                for view in observation['views'].values():
                    view['instance_masks'].pop('source', None)
            return observation

    class Judgment(Choices):
        injected_unknown = False

        async def choose_skill(self, state, registry):
            if state.get('verification', {}).get('status') == 'unknown':
                runner.simulation.uncertain = False
                return 'verify_grasp'
            skill = await super().choose_skill(state, registry)
            if skill == 'verify_grasp' and not self.injected_unknown:
                self.injected_unknown = True
                runner.simulation.uncertain = True
            return skill

    judgment = Judgment('grasp')
    runner = Runner(Robot(), Vision(), lambda record: judgment, tmp_path)
    await runner.start('grasp red cube')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state['error']
    assert [obj['id'] for obj in runner.holding_state['observation']['objects']] == ['source', 'destination']
    assert runner.task_progress.attempts == {}, 'unknown evidence must not consume grasp retry attempts'
    judgment.task = 'place'
    await runner.start('place held object in blue bin')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state['error']
    assert runner.algorithms.checkpoints == ['initial', 'fine', 'post_place']
