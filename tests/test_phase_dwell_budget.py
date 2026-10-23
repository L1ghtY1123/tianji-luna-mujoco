from types import SimpleNamespace

import pytest

from luna_mujoco.runner import Runner


@pytest.mark.asyncio
@pytest.mark.parametrize('check_time,remaining,overrun', [(.3, .7, 0.), (1.4, 0., .4)])
async def test_validation_uses_existing_dwell_budget(monkeypatch, check_time, remaining, overrun):
    clock = [10.]
    waits = []
    monkeypatch.setattr('luna_mujoco.runner.time.monotonic', lambda: clock[0])

    async def validate(*args, **kwargs):
        clock[0] += check_time
        return {'collision_free': True}

    async def motor(fn, duration):
        waits.append(duration)
        clock[0] += duration
        return {'completed': True}

    runner = SimpleNamespace(validate_motion=validate, motor=motor,
                             simulation=SimpleNamespace(dwell=object()))
    result = await Runner.validate_during_dwell(runner, {}, 'retreat', {}, duration_s=1.)
    assert waits == pytest.approx([remaining])
    assert result['overrun_s'] == pytest.approx(overrun)
    assert result['duration_s'] == pytest.approx(max(1., check_time))


@pytest.mark.asyncio
async def test_failed_validation_never_becomes_successful_dwell():
    async def fail(*args, **kwargs):
        raise ValueError('collision')
    runner = SimpleNamespace(validate_motion=fail)
    with pytest.raises(ValueError, match='collision'):
        await Runner.validate_during_dwell(runner, {}, 'place', {}, duration_s=1.)
