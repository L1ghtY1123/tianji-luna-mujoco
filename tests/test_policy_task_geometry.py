"""Independent task runs must not inherit another inventory's GPU evidence."""
import asyncio
import json
from copy import deepcopy

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot
from test_policy_integration import FixedInventoryVision, ModelHTTP, transfer_plan

from luna_mujoco.agent.decisions import DecisionsJudgments
from luna_mujoco.agent.responses import ResponsesCompiler
from luna_mujoco.runtime.controller import PolicyController


def controller_for(tmp_path, vision):
    plan = transfer_plan()
    http = ModelHTTP(plan)
    http.outputs.append(deepcopy(http.outputs[0]))
    compiler = ResponsesCompiler(registry={}, configured_regions={'demo-destination': {}},
                                 api_key='test', client=http)
    robot = DemoRobot()
    controller = PolicyController(robot, vision, None, tmp_path, planning_settings=TEST_SETTINGS,
        compiler=compiler, judgments=DecisionsJudgments(api_key='test', client=http))
    compiler.registry = controller.registry
    return controller, http, robot


async def test_same_generation_new_tasks_clear_map_once_before_observation(tmp_path):
    class Vision(FixedInventoryVision):
        clears = 0
        first_capture_after_clear = None

        def clear_carrying(self):
            self.clears += 1
            self.first_capture_after_clear = None
            return {'ok': True, 'attached': False}

        def observe(self, capture, **kwargs):
            assert self.clears >= 1
            if self.first_capture_after_clear is None:
                self.first_capture_after_clear = capture['observation_id']
            return super().observe(capture, **kwargs)

    vision = Vision(count=1)
    controller, _, _ = controller_for(tmp_path, vision)
    for run in range(2):
        await controller.start('Move the red items to the tray')
        await asyncio.wait_for(controller.worker, 10)
        assert controller.state['status'] == 'succeeded', controller.state.get('error')
        assert vision.clears == run + 1
        assert vision.first_capture_after_clear is not None


async def test_bootstrap_uses_actual_pose_without_forcing_a_return(tmp_path):
    controller, http, robot = controller_for(tmp_path, FixedInventoryVision(count=1))
    robot.joints[0] = .3
    graph = http.outputs[0]['graph']
    graph['nodes'].append({'type': 'end', 'id': 'stop_after_capture', 'status': 'failure',
                          'reason': 'test stops after raw-pose observation'})
    graph['edges'][0]['target'] = 'stop_after_capture'
    graph['edges'].append({'id': 'unexecuted_branch', 'source': 'initial_observe',
                           'target': 'confirm_inventory', 'on': 'unknown'})
    await controller.start('Move the red items to the tray')
    await asyncio.wait_for(controller.worker, 10)
    prompt = json.loads(http.requests[0][1]['input'][0]['content'][0]['text'])
    assert prompt['initial_facts']['observation_pose_ready'] is False
    assert controller.state['facts']['observation_pose_ready'] is False
    assert any(event['kind'] == 'skill_complete' and event['title'] == 'observe_scene'
               for event in controller.events)
    assert robot.moves == []


@pytest.mark.parametrize('result', [{'ok': False, 'attached': False}, {'ok': True},
                                   {'ok': True, 'attached': True}])
async def test_unconfirmed_cleanup_stops_before_model_observation_or_motor(tmp_path, result):
    class Vision(FixedInventoryVision):
        def clear_carrying(self):
            return result

    vision = Vision(count=1)
    controller, http, robot = controller_for(tmp_path, vision)
    await controller.start('Move the red items to the tray')
    await asyncio.wait_for(controller.worker, 10)
    assert controller.state['error_code'] == 'task_geometry_reset_unconfirmed'
    assert http.requests == []
    assert vision.checkpoints == []
    assert robot.moves == []


@pytest.mark.parametrize('state', ['provisional', 'held', 'release_pending'])
@pytest.mark.parametrize('attached', [False, True])
async def test_new_task_never_clears_any_unresolved_holding_state(tmp_path, state, attached):
    controller, http, _ = controller_for(tmp_path, FixedInventoryVision(count=1))
    holding = {'state': state, 'attached': attached}
    controller.holding_state = deepcopy(holding)
    with pytest.raises(ValueError, match='holding_state_requires_explicit_recovery'):
        await controller.start('Move the red items to the tray')
    assert controller.holding_state == holding
    assert http.requests == []
