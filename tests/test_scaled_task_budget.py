"""Whole-task captures survive per-object reset and have independent recovery quota."""

import pytest
from test_rgbd_workflow import SensorRobot, VisionAlgorithms

from luna_mujoco.runner import Runner
from luna_mujoco.task_progress import TaskProgress


@pytest.mark.parametrize('count,route', [(8, 'table_box'), (4, 'box_table_box')])
async def test_large_task_capture_replay_keeps_counters_across_transfers(tmp_path, count, route):
    runner = Runner(SensorRobot(), VisionAlgorithms(), None, tmp_path)
    runner.state.update(generation=1, task='grasp_place')
    progress = TaskProgress('搬运全部物体', 'collection', route=route)
    progress.inventory_finalized = True
    progress.membership = {f'part-{i}': 'required' for i in range(count)}
    runner.task_progress = progress
    context = {}
    for index in range(16):
        capture = await runner.capture_observation(context)
        await runner.capture_observation(context, capture=capture)  # Analysis must not charge twice.
        if index % 2:
            runner.reset_transfer(context)
    budget = runner.task_budgets(context)
    assert budget['normal'] == {'used': 16, 'limit': 16}
    assert budget['recovery'] == {'used': 0, 'limit': 8}
    before = runner.simulation.time
    with pytest.raises(ValueError, match='normal_observation_budget_exhausted'):
        await runner.capture_observation(context)
    assert runner.simulation.time == before
    for _ in range(8):
        await runner.capture_observation(context, recovery=True)
    with pytest.raises(ValueError, match='recovery_observation_budget_exhausted'):
        await runner.capture_observation(context, recovery=True)
    assert runner.task_budgets(context)['decision_limit'] == 400


def test_decision_budget_scales_for_large_frozen_inventory(tmp_path):
    runner = Runner(SensorRobot(), VisionAlgorithms(), None, tmp_path)
    runner.task_progress = TaskProgress('搬运物体', 'collection', route='box_table_box')
    runner.task_progress.inventory_finalized = True
    runner.task_progress.membership = {f'p{i}': 'required' for i in range(8)}
    assert runner.task_budgets({})['decision_limit'] == 720
