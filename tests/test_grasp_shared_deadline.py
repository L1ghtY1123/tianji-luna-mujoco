"""IK and trajectory planning consume one monotonic operation budget."""

from types import SimpleNamespace

import pytest
from test_runner import Sim

from luna_mujoco.runner import Runner


@pytest.mark.parametrize('ik_elapsed', [4., 11.])
async def test_ik_consumes_grasp_deadline_before_trajectory_dispatch(tmp_path, monkeypatch, ik_elapsed):
    clock = [100.]
    monkeypatch.setattr('luna_mujoco.runner.time.monotonic', lambda: clock[0])
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
