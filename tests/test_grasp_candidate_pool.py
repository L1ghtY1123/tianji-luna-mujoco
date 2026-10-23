"""Candidate budgets bound sampling while retaining the service's fixed protocol."""

import threading

import numpy as np
import pytest
from test_rgbd_algorithms import observed_scene

from luna_mujoco.algorithms import AlgorithmError, Algorithms


class GraspClient:
    def __init__(self, *, failure=None):
        self.calls = []
        self.failure = failure

    def generate(self, payload):
        self.calls.append(payload)
        batch = len(self.calls)
        if batch == 3 and self.failure == "service":
            return {"ok": False, "error": "inference failed"}
        if batch == 3 and self.failure == "empty":
            return {"ok": True, "model": "fixed-model", "candidates": []}
        return {"ok": True, "model": "changed" if batch == 3 and self.failure == "model" else "fixed-model",
                "candidates": [{"index": index, "score": index / 100,
                                "T_camera_grasp": np.eye(4).tolist()}
                               for index in range(100)]}


@pytest.mark.parametrize("count,expected_calls", [(None, 2), (1, 1), (100, 1), (101, 2), (1000, 10)])
def test_fixed_service_samples_sort_and_slice_requested_budget(count, expected_calls):
    client = GraspClient()
    kwargs = {} if count is None else {"candidate_count": count}
    result = Algorithms(graspgen_client=client).generate_grasps(observed_scene(), "seen-red", **kwargs)
    requested = 200 if count is None else count
    assert len(client.calls) == expected_calls
    assert all(set(payload) == {"scene_points_camera_m", "object_points_camera_m", "gripper_config_id"}
               for payload in client.calls)
    assert result["requested_candidate_count"] == result["candidate_count"] == requested
    assert len(result['all_generated_candidates']) == expected_calls * 100
    assert result['all_generated_candidates'][:requested] == result['candidates']
    assert [candidate["index"] for candidate in result["candidates"]] == list(range(requested))
    keys = [(-candidate["score"], candidate["source_batch"], candidate["source_index"])
            for candidate in result["candidates"]]
    assert keys == sorted(keys)
    assert [batch["candidate_count"] for batch in result["batches"]] == [100] * expected_calls
    np.testing.assert_array_equal(client.calls[0]["object_points_camera_m"],
                                  client.calls[-1]["object_points_camera_m"])


@pytest.mark.parametrize("count", [0, -1, 1001, True, 100., "100", None])
def test_invalid_candidate_budget_rejected_before_external_inference(count):
    client = GraspClient()
    with pytest.raises(ValueError, match="candidate_count"):
        Algorithms(graspgen_client=client).generate_grasps(
            observed_scene(), "seen-red", candidate_count=count)
    assert not client.calls


@pytest.mark.parametrize("failure", ["model", "service", "empty"])
def test_larger_pool_fails_fast_without_retry_or_partial_success(failure):
    client = GraspClient(failure=failure)
    with pytest.raises(AlgorithmError, match="GraspGen"):
        Algorithms(graspgen_client=client).generate_grasps(
            observed_scene(), "seen-red", candidate_count=1000)
    assert len(client.calls) == 3


@pytest.mark.parametrize("cancel_before", [True, False])
def test_cancelled_generation_does_not_launch_remaining_inference_batches(cancel_before):
    cancelled = threading.Event()
    if cancel_before:
        cancelled.set()

    class CancellingClient(GraspClient):
        def generate(self, payload):
            result = super().generate(payload)
            cancelled.set()
            return result

    client = CancellingClient()
    with pytest.raises(AlgorithmError) as caught:
        Algorithms(graspgen_client=client).generate_grasps(
            observed_scene(), "seen-red", candidate_count=1000, cancelled=cancelled)
    assert caught.value.code == "planning_cancelled"
    assert len(client.calls) == (0 if cancel_before else 1)
