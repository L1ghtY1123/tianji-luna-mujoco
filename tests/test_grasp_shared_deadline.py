"""Planning uses SDK attempt budgets, without a project wall-clock deadline."""

from types import SimpleNamespace

import pytest
from test_runner import Sim

from luna_mujoco.runner import Runner


@pytest.mark.parametrize('ik_elapsed', [4., 1000.])
async def test_slow_ik_does_not_create_an_unrequested_planning_deadline(tmp_path, monkeypatch, ik_elapsed):
    clock = [100.]
    monkeypatch.setattr('luna_mujoco.runner.time.monotonic', lambda: clock[0])
    algorithms = SimpleNamespace(evaluate_arms=object(), plan_grasp=object())
    runner = Runner(Sim(), algorithms, None, tmp_path)
    runner.state.update(task='grasp', arm='A', target_id='source', generation=1, planning_attempts=[])
    context = {'grasps': {}, 'scene': {}, 'planning_settings': {}}
    calls = []

    async def call(function, *args, **kwargs):
        calls.append((function, kwargs.get('timeout_s')))
        if function is algorithms.evaluate_arms:
            clock[0] += ik_elapsed
            return {}
        return {'planned': True}

    runner.call_algorithm = call
    assert await runner.plan_grasp({}, 'source', 'A', context) == {'planned': True}
    assert len(calls) == 2
    assert all(timeout is None for _, timeout in calls)
