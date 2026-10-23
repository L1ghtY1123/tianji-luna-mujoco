from types import SimpleNamespace

import pytest
from test_runner import Sim

from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.runner import Runner


@pytest.mark.asyncio
@pytest.mark.parametrize("task,skill,timeout_key", [
    ("grasp", "plan_grasp", "plan_timeout_s"),
    ("grasp_place", "plan_grasp", "plan_timeout_s"),
    ("grasp_place", "plan_place", "place_timeout_s"),
    ("place", "plan_place", "place_timeout_s"),
])
@pytest.mark.parametrize("grasp_timeout,place_timeout", [(180., 10.), (10., 180.)])
async def test_each_planning_phase_uses_its_own_frozen_deadline(
        tmp_path, task, skill, timeout_key, grasp_timeout, place_timeout):
    algorithms = SimpleNamespace(plan_grasp=object(), plan_place=object())
    runner = Runner(Sim(), algorithms, None, tmp_path)
    runner.state.update(task=task, arm="A", target_id="source", destination_id="bin",
                        generation=1, planning_attempts=[], current_step={'skill': skill})
    settings = PlanningSettings(plan_timeout_s=grasp_timeout, place_timeout_s=place_timeout).model_dump()
    context = {"observation": {"observation_id": "current", "objects": []},
               "scene": {"observation_id": "current"}, "evaluations": {},
               "planning_settings": settings, "holding": {"attached": True, 'verified': True,
                   'attachment_evidence': {'attached': True}}}
    calls = []

    async def capture_algorithm_call(function, *args, **kwargs):
        calls.append((function, kwargs))
        return {"planned": skill}

    runner.call_algorithm = capture_algorithm_call
    result = await runner.execute(skill, context, None)
    assert result == {"planned": skill}
    assert len(calls) == 1
    assert calls[0][0] is getattr(algorithms, skill)
    assert calls[0][1]["timeout_s"] == pytest.approx(settings[timeout_key], abs=.01)
    assert calls[0][1]["planning_settings"] == settings
    if skill != "plan_place":
        assert runner.state["planning_attempts"][0]["effective_timeout_s"] == grasp_timeout
