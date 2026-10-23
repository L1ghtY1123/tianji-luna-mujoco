from types import SimpleNamespace

import pytest
from test_runner import Sim

from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.runner import Runner


@pytest.mark.asyncio
@pytest.mark.parametrize("task,skill", [
    ("grasp", "plan_grasp"),
    ("grasp_place", "plan_grasp"),
    ("grasp_place", "plan_place"),
    ("place", "plan_place"),
])
async def test_each_planning_phase_uses_sdk_budgets_without_wrapper_deadlines(
        tmp_path, task, skill):
    algorithms = SimpleNamespace(plan_grasp=object(), plan_place=object())
    runner = Runner(Sim(), algorithms, None, tmp_path)
    runner.state.update(task=task, arm="A", target_id="source", destination_id="bin",
                        generation=1, planning_attempts=[], current_step={'skill': skill})
    settings = PlanningSettings().model_dump()
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
    assert calls[0][1].get("timeout_s") is None
    assert calls[0][1]["planning_settings"] == settings
    if skill != "plan_place":
        assert "effective_timeout_s" not in runner.state["planning_attempts"][0]
