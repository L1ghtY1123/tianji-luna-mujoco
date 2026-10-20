"""IK and trajectory planning consume one monotonic operation budget."""

import asyncio
from types import SimpleNamespace

import pytest
from test_runner import Sim

from jev_mujoco.runner import Runner


@pytest.mark.parametrize('ik_elapsed', [4., 11.])
async def test_ik_consumes_grasp_deadline_before_trajectory_dispatch(tmp_path, monkeypatch, ik_elapsed):
    clock = [100.]
    monkeypatch.setattr('jev_mujoco.runner.time.monotonic', lambda: clock[0])
    algorithms = SimpleNamespace(evaluate_arms=object(), plan_grasp=object())
    runner = Runner(Sim(), algorithms, None, tmp_path)
    runner.state.update(task='grasp', arm='A', target_id='source', generation=1, planning_attempts=[])
    context = {'grasps': {}, 'scene': {}, 'planning_settings': {'plan_timeout_s': 10.}}
    calls = []

    async def call(function, *args, **kwargs):
        calls.append((function, kwargs.get('timeout_s')))
        if function is algorithms.evaluate_arms:
            clock[0] += ik_elapsed
            return {}
        return {'planned': True}

    runner.call_algorithm = call
    if ik_elapsed > 10.:
        with pytest.raises(RuntimeError, match='plan_grasp_timeout'):
            await runner.plan_grasp({}, 'source', 'A', context)
        assert len(calls) == 1
    else:
        assert await runner.plan_grasp({}, 'source', 'A', context) == {'planned': True}
        assert calls[1][1] == 6.
    assert calls[0][1] == 10.


async def test_blocked_ik_times_out_and_late_reply_never_starts_trajectory(tmp_path):
    from test_runner import Algorithms, Decisions
    from test_runner_failures import WaitingPlanner

    class WaitingIK(WaitingPlanner):
        def evaluate_arms(self, *args, cancelled=None, **kwargs):
            self.cancelled = cancelled
            self.entered.set()
            self.release.wait(3)
            return Algorithms.evaluate_arms(self, *args)

        def plan_grasp(self, *args, **kwargs):
            pytest.fail('timed out IK must not dispatch trajectory planning')

    algorithms = WaitingIK()
    runner = Runner(Sim(), algorithms, lambda record: Decisions(), tmp_path)
    runner.plan_timeout_s = .05
    await runner.start('抓取黑色物体')
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        await asyncio.wait_for(asyncio.shield(runner.worker), 2)
        assert 'plan_grasp_timeout' in runner.state['error']
        assert algorithms.cancelled.is_set() and runner.algorithm_pending
        assert not runner.state['facts'].get('plan_ready')
    finally:
        algorithms.release.set()
        await asyncio.gather(runner.planning_worker, return_exceptions=True)
    assert not runner.state['facts'].get('plan_ready')
