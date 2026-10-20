import copy

import numpy as np
from test_runner import Decisions, Sim, wait
from test_visible_workflow import VisibleAlgorithms

from jev_mujoco.runner import Runner


async def test_grasp_preserves_pregrasp_cloud_and_shared_scene(tmp_path):
    class Algorithms(VisibleAlgorithms):
        def __init__(self):
            super().__init__()
            self.maps = []
            self.initial_points = None

        def observe(self, capture, **kwargs):
            result = super().observe(capture, **kwargs)
            if self.initial_points is None:
                self.initial_points = copy.deepcopy(result['objects'][0]['points_world_m'])
            else:
                result['objects'][0]['points_world_m'] = (
                    np.asarray(result['objects'][0]['points_world_m']) + [.002, 0, 0])
            return result

        def build_scene(self, *args, **kwargs):
            result = super().build_scene(*args, **kwargs)
            self.maps.append(copy.deepcopy(result))
            return result

    algorithms = Algorithms()
    runner = Runner(Sim(), algorithms, lambda record: Decisions(), tmp_path)
    await runner.start('抓取黑色物体')
    await wait(runner)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert len(algorithms.maps) == 2
    np.testing.assert_allclose(runner.holding_state['points_tool_m'],
                               np.asarray(algorithms.initial_points) + [.002, 0., 0.])
    assert runner.holding_state['scene']['scene_token'] == algorithms.maps[-1]['scene_token']
    assert len(algorithms.observations) == 2


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
