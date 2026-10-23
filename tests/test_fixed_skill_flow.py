
import numpy as np
from test_runner import Sim
from test_visible_workflow import VisibleAlgorithms

from luna_mujoco.runner import Runner


async def test_empty_hand_scene_rebuild_invalidates_prior_planning_products(tmp_path, monkeypatch):
    runner = Runner(Sim(), VisibleAlgorithms(), None, tmp_path)
    runner.state.update(target_id="source", arm="A", planning_attempts=[{"arm": "A"}])
    runner.state["facts"] = {"plan_ready": True, "place_plan_ready": True, "grasps_ready": True,
                             "arms_evaluated": True, "arm_selected": True}
    observation = {"objects": [], "observation_id": "new", "arms": {"A": {"tcp": np.eye(4).tolist()}}}
    context = {"observation": observation, "scene": {"scene_token": "old"}, "planning_settings": {},
               **{name: {} for name in ("grasps", "evaluations", "plan", "place_plan")}}
    async def build(*args, **kwargs):
        return {"scene_token": "new"}

    monkeypatch.setattr(runner, "call_algorithm", build)
    await runner.build_scene(context, "selected")
    assert not runner.state["facts"]["plan_ready"]
    assert not runner.state["facts"]["place_plan_ready"]
    assert not runner.state["facts"]["arms_evaluated"]
    assert not runner.state["facts"]["grasps_ready"]
    assert not runner.state["facts"]["arm_selected"]
    assert runner.state["planning_attempts"] == []
    assert not {"grasps", "evaluations", "plan", "place_plan"} & context.keys()
