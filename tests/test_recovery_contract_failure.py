import copy

import pytest
from test_planning_recovery import Selecting, make_runner
from test_runner import Algorithms, wait

from jev_mujoco.skills import SkillSelectionError
from jev_mujoco.task_progress import TaskProgress


async def test_rejected_recovery_preserves_original_failure_and_returns_evidence(tmp_path):
    class Choice(Selecting):
        async def choose_recovery(self, state):
            return 'stop' if state.get('skill_rejection') else 'retry_plan'

    class NoEvaluation(Algorithms):
        def evaluate_arms(self, *args, **kwargs):
            raise ValueError('initial_ik_service_failed')

    choice = Choice()
    runner = make_runner(tmp_path, NoEvaluation(), choice)
    await runner.start('抓取黑色物体')
    await wait(runner)
    rejected = [s for s in choice.states if s.get('skill_rejection')]
    assert rejected
    assert rejected[0]['failure']['code'] == 'initial_ik_service_failed'
    assert rejected[0]['failure']['attempt_count'] == 3
    assert rejected[0]['skill_rejection']['executed'] is False
    assert runner.state['error'] == 'initial_ik_service_failed'
    assert len([e for e in runner.events if e['kind'] == 'task_failure']) == 3
    assert runner.state['conclusion']['root_cause']['code'] == 'initial_ik_service_failed'
    assert runner.state['conclusion']['rejected_calls']
    assert copy.deepcopy(runner.state['conclusion']) == runner.events[-1]['data']['conclusion']


async def test_exhausted_reobserve_budget_rejects_without_masking_root_failure(tmp_path):
    runner = make_runner(tmp_path, Algorithms(), Selecting())
    runner.task_progress = TaskProgress('抓取物体')
    runner.task_progress.fail('target', 'choose_arm', 'no_reachable_arm', {})
    before = copy.deepcopy(runner.task_progress.failure)
    with pytest.raises(SkillSelectionError, match='observation_budget_exhausted'):
        await runner.recover_task({'recovery_capture_count': 8}, 'reobserve')
    assert runner.task_progress.failure == before
    assert runner.simulation.frame == 0
