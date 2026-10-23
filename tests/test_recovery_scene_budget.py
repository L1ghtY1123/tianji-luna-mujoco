
import pytest
from test_rgbd_workflow import SensorRobot, VisionAlgorithms

from luna_mujoco.runner import Runner


async def test_observation_budget_blocks_sensor_work_without_selecting_recovery(tmp_path):
    algorithms = VisionAlgorithms()
    runner = Runner(SensorRobot(), algorithms, None, tmp_path)
    runner.state.update(generation=1, task='grasp', instruction='抓取红色物体')
    context = {'recovery_capture_count': 8}
    before = runner.state['status']
    with pytest.raises(ValueError, match='observation_budget_exhausted'):
        await runner.observe(context, recovery=True)
    assert algorithms.checkpoints == []
    assert runner.state['status'] == before
