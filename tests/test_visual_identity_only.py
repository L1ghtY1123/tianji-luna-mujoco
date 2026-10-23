from types import SimpleNamespace

import pytest
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from luna_mujoco.perception import PerceptionError
from luna_mujoco.runner import Runner
from luna_mujoco.task_progress import TaskProgress


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
