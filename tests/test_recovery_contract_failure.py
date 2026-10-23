import copy

import pytest
from test_planning_recovery import Selecting, make_runner
from test_runner import Algorithms

from luna_mujoco.skills import SkillSelectionError
from luna_mujoco.task_progress import TaskProgress


async def test_exhausted_reobserve_budget_rejects_without_masking_root_failure(tmp_path):
    runner = make_runner(tmp_path, Algorithms(), Selecting())
    runner.task_progress = TaskProgress('抓取物体')
    runner.task_progress.fail('target', 'choose_arm', 'no_reachable_arm', {})
    before = copy.deepcopy(runner.task_progress.failure)
    with pytest.raises(SkillSelectionError, match='observation_budget_exhausted'):
        await runner.recover_task({'recovery_capture_count': 8}, 'reobserve')
    assert runner.task_progress.failure == before
    assert runner.simulation.frame == 0
