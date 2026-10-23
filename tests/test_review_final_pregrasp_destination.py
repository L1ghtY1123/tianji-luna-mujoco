"""A hypothetical open-hand destination failure cannot invalidate real grasp IK."""

import numpy as np

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.planning_settings import PlanningSettings


def test_combined_evaluation_keeps_grasp_candidates_when_open_hand_destination_ik_fails():
    from test_rgbd_algorithms import observed_scene, with_bin

    observation = with_bin(observed_scene())
    scene = {"target_id": "seen-red", "generation": 1,
             "observation_id": "capture-4", "scene_token": "map-1"}
    settings = PlanningSettings().model_dump()
    calls = []

    def request(path, body):
        assert path == "/tools/solve_ik_batch"
        assert body["planning_settings"] == settings
        assert body["trajectory_collision_clearance_m"] == settings["trajectory_collision_clearance_m"]
        assert "carrying_binding" not in body
        calls.append(body)
        # The observed source is at y=.1; the measured bin reference is y=-.2.
        # This models the current fixed-open destination checker rejecting all
        # goals, while ordinary source IK rejects only the middle grasp.
        goals = np.asarray(body["goal_matrices_world"])
        at_destination = goals[:, 1, 3] < 0.
        return {"scene_token": "map-1", "batch_size": len(goals), "items": [
            {"candidate_index": i, "success": bool(not at_destination[i] and i != 1)}
            for i in range(len(goals))]}

    grasps = {"candidates": [{"index": i + 10, "score": .9 - i * .1,
                              "T_object_grasp": np.eye(4).tolist()} for i in range(3)]}
    result = Algorithms(curobo_request=request).evaluate_arms(
        observation, "seen-red", grasps, scene, destination_id="blue-bin", planning_settings=settings)

    for arm in ("A", "B"):
        assert [candidate["candidate_index"] for candidate in result[arm]["valid_candidates"]] == [10, 12]
        assert result[arm]["destination_id"] == "blue-bin"
        assert result[arm]["planning_settings"] == settings
    assert {body["arm"] for body in calls} == {"A", "B"}
