from types import SimpleNamespace

import pytest
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from jev_mujoco.perception import PerceptionError
from jev_mujoco.runner import Runner
from jev_mujoco.task_progress import TaskProgress


async def test_missing_visual_identity_result_cannot_enter_inventory(tmp_path):
    class MissingIdentity(VisionAlgorithms):
        def observe(self, *args, **kwargs):
            result = super().observe(*args, **kwargs)
            result.pop('visual_inventory', None)
            return result

        def associate(self, *args, **kwargs):
            raise AssertionError('Old identity matcher must never be used')

    algorithms = MissingIdentity()
    algorithms.perception = SimpleNamespace(visual_inventory=SimpleNamespace(audit=None))
    runner = Runner(SensorRobot(), algorithms, lambda record: Choices(), tmp_path)
    runner.state.update(task='grasp_place', generation=1, instruction='move red')
    runner.task_progress = TaskProgress('move red')
    with pytest.raises(PerceptionError, match='visual_inventory_required'):
        await runner.observe({'visual_plan': {'initial': 'initialize_inventory'}}, purpose='scene')
    assert not runner.task_progress.records


async def test_injected_algorithm_cannot_disable_outer_visual_selection(tmp_path):
    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: Choices(), tmp_path)
    runner.state.update(task='grasp_place', generation=1, instruction='move red')
    runner.task_progress = TaskProgress('move red')
    calls = []

    class Selection:
        async def choose_visual_plan(self, state, skill, requests):
            calls.append((skill, requests))
            return requests

    arguments = await runner.select_arguments('observe_scene', {}, Selection())
    assert calls == [('observe_scene', {'initial': 'initialize_inventory'})]
    assert arguments['visual_plan'] == {'initial': 'initialize_inventory'}
