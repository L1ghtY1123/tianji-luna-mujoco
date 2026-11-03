import copy

import pytest
from planning_recovery_fixture import Selecting, make_runner
from test_runner import Algorithms

from luna_mujoco.task_progress import TaskProgress


async def test_exhausted_reobserve_budget_rejects_without_masking_root_failure(tmp_path):
    runner = make_runner(tmp_path, Algorithms(), Selecting())
    runner.task_progress = TaskProgress('抓取物体')
    runner.task_progress.fail('target', 'choose_arm', 'no_reachable_arm', {})
    before = copy.deepcopy(runner.task_progress.failure)
    with pytest.raises(ValueError, match='unknown_skill:recover_task'):
        await runner.execute('recover_task', {'recovery_capture_count': 8}, {'action': 'reobserve'})
    assert runner.task_progress.failure == before
    assert runner.simulation.frame == 0
