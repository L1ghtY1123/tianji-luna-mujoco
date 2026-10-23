"""Every candidate stage owns the same-round submitted clouds, without inference."""
import json
from copy import deepcopy

import numpy as np
import pytest
from test_demo_planning import GraspClient, grasp_placement
from test_experiments import begin
from test_grasp_height_completion import completed_scene
from test_planning_batch import batch_response
from test_rgbd_algorithms import observed_scene

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.experiments import ExperimentStore


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("sampling_round,expected_calls,expected_indices", [
    (1, 3, [6, 7, 8]), (3, 10, list(range(20, 30))),
])
def test_later_stages_keep_exact_same_round_inputs_and_provenance(
        tmp_path, completed, sampling_round, expected_calls, expected_indices):
    snapshot = completed_scene() if completed else observed_scene()
    client = GraspClient()
    grasps = DemoAlgorithms(graspgen_client=client).generate_grasps(
        snapshot, "seen-red", arm="A", sampling_round=sampling_round,
        placement_request=grasp_placement())
    grasps["candidate_replay"]["graspgen_input_sampling"] = {
        "source": "test_recorded_input_sampling",
        "actual_point_count": len(client.calls[0]["object_points_camera_m"]),
    }
    original = deepcopy(grasps["candidate_replay"])
    scene = {"target_id": "seen-red", "generation": 1,
             "observation_id": "capture-4", "scene_token": "map-1"}
    calls = []

    def request(path, body):
        calls.append((path, deepcopy(body)))
        if path == "/tools/solve_ik_batch":
            return {"scene_token": body["scene_token"], "batch_size": len(body["goal_matrices_world"]),
                    "items": [{"candidate_index": index, "success": True}
                              for index in range(len(body["goal_matrices_world"]))]}
        assert path == "/tools/plan_grasp_batch"
        return batch_response(body)

    algorithms = Algorithms(curobo_request=request)
    algorithms.validate_execution = lambda *args, **kwargs: {"collision_free": True}
    evaluations = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene, requested_arm="A")
    after_ik = evaluations["A"]["batches"][0]["candidate_replay"]
    # Replacing the source artifact cannot relabel or alter already frozen IK evidence.
    grasps["candidate_replay"]["graspgen_request"]["object_points_camera_m"][:] = 999.
    grasps["candidate_replay"]["graspgen_input_source"] = "later-unrelated-round"
    events = []
    algorithms.plan_grasp(snapshot, "seen-red", "A", evaluations, scene,
                          progress=lambda kind, title, data: events.append(data))
    after_plan = next(event["candidate_replay"] for event in events if "candidate_replay" in event)
    for replay in (after_ik, after_plan):
        assert replay["sampling_round"] == sampling_round
        assert replay["target_id"] == "seen-red"
        assert replay["observation_id"] == original["observation_id"]
        assert replay["candidate_indices"] == expected_indices
        assert replay["T_world_cloud"] == original["T_world_cloud"]
        assert replay["graspgen_input_source"] == original["graspgen_input_source"]
        assert replay["graspgen_input_sampling"] == original["graspgen_input_sampling"]
        assert replay["completion_provenance"] == original["completion_provenance"]
        assert replay["graspgen_input_cloud"] == original["graspgen_input_cloud"]
        assert replay["measured_cloud_reference"] == original["measured_cloud_reference"]
        for key in ("object_points_camera_m", "scene_points_camera_m"):
            np.testing.assert_array_equal(replay["graspgen_request"][key], original["graspgen_request"][key])
            assert replay["graspgen_request"][key].dtype == np.float32
        np.testing.assert_array_equal(replay["measured_points_world_m"], original["measured_points_world_m"])
        if completed:
            np.testing.assert_array_equal(replay["completed_geometry_points_world_m"],
                                          original["completed_geometry_points_world_m"])
            assert len(replay["graspgen_request"]["object_points_camera_m"]) > len(replay["measured_points_world_m"])
        else:
            assert "completed_geometry_points_world_m" not in replay
    assert len(client.calls) == expected_calls
    assert [path for path, _ in calls] == ["/tools/solve_ik_batch", "/tools/plan_grasp_batch"]
    assert all("graspgen_request" not in body for _, body in calls)
    # Each downstream event can render its own NPZ arrays without the before-IK artifact.
    store = ExperimentStore(tmp_path)
    session = begin(store)
    for index, replay in enumerate((after_ik, after_plan), 1):
        session.event({"id": index, "kind": "planning_replay", "title": replay["stage"],
                       "t_s": float(index), "data": {}}, {"candidate_replay": replay})
    session.finish({"status": "succeeded"})
    for event in store.events(session.id):
        archived = json.loads(store.file(session.id, event["artifact"]).read_text(encoding="utf-8"))["candidate_replay"]
        assert archived["sampling_round"] == sampling_round
        for key in ("object_points_camera_m", "scene_points_camera_m"):
            reference = archived["graspgen_request"][key]
            assert reference["type"] == "ndarray"
            with np.load(store.file(session.id, reference["path"]), allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays[reference["key"]], original["graspgen_request"][key])


def test_missing_historical_actual_input_is_not_reconstructed_from_measurement():
    from test_planning_batch import evaluated

    algorithms, snapshot, scene, evaluations, _ = evaluated(1)
    after_ik = evaluations["A"]["batches"][0]["candidate_replay"]
    algorithms.request = lambda path, body: batch_response(body)
    events = []
    algorithms.plan_grasp(snapshot, "red", "A", evaluations, scene,
                          progress=lambda kind, title, data: events.append(data))
    after_plan = next(event["candidate_replay"] for event in events if "candidate_replay" in event)
    for replay in (after_ik, after_plan):
        assert "graspgen_request" not in replay
        assert "graspgen_input_source" not in replay
        assert "measured_points_world_m" in replay
