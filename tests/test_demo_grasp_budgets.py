"""Adaptive pools reduce service work without changing the frozen grasp input."""
import numpy as np
import pytest
from test_demo_planning import grasp_placement
from test_rgbd_algorithms import observed_scene

from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.demo_sampling import grasp_input_identity, round_evidence


class DownwardClient:
    def __init__(self):
        self.calls = []

    def generate(self, payload):
        self.calls.append(payload)
        return {"ok": True, "model": "fixed-model", "candidates": [
            {"index": index, "score": 1. - index / 100,
             "T_camera_grasp": np.diag([-1., 1., -1., 1.]).tolist()}
            for index in range(100)]}


def test_increasing_pools_keep_identical_inference_inputs_and_bound_service_calls():
    client = DownwardClient()
    algorithms = DemoAlgorithms(graspgen_client=client)
    snapshot = observed_scene()
    identity = None
    for sampling_round, expected_count, expected_calls in [(1, 256, 3), (2, 512, 6), (3, 1000, 10)]:
        before = len(client.calls)
        result = algorithms.generate_grasps(snapshot, "seen-red", arm="A",
            placement_request=grasp_placement(), sampling_round=sampling_round)
        assert len(client.calls) - before == expected_calls
        assert result["requested_candidate_count"] == expected_count
        assert result["candidate_count"] == expected_count
        assert result["raw_candidate_count"] == expected_calls * 100
        assert len(result["candidate_replay"]["candidates_before_ik"]) == expected_count
        assert result["candidate_replay"]["requested_candidate_count"] == expected_count
        assert result["candidate_replay"]["raw_candidate_count"] == expected_calls * 100
        assert [candidate["index"] for candidate in result["candidates"]] == list(range(expected_count))
        evidence = round_evidence(result, {"valid_candidates": [], "batches": []}, sampling_round)
        assert evidence["requested_candidate_count"] == expected_count
        assert evidence["generated_candidate_count"] == expected_count
        assert evidence["raw_candidate_count"] == expected_calls * 100
        current_identity = grasp_input_identity(result)
        if identity is None:
            identity = current_identity
        assert current_identity == identity
    assert len(client.calls) == 19
    for payload in client.calls:
        assert set(payload) == {"scene_points_camera_m", "object_points_camera_m", "gripper_config_id"}
        assert payload["gripper_config_id"] == "robotiq_2f_140"
        for field in ("scene_points_camera_m", "object_points_camera_m"):
            np.testing.assert_array_equal(payload[field], client.calls[0][field])


@pytest.mark.parametrize("sampling_round", [0, 4, -1, True, 1., "1", None])
def test_invalid_round_stops_before_external_sampling(sampling_round):
    client = DownwardClient()
    with pytest.raises(ValueError, match="invalid_grasp_sampling_round"):
        DemoAlgorithms(graspgen_client=client).generate_grasps(observed_scene(), "seen-red", arm="A",
            placement_request=grasp_placement(), sampling_round=sampling_round)
    assert not client.calls
