import hashlib
import json
import threading

import httpx
import pytest
from test_planning_batch import batch_response, evaluated

from luna_mujoco.algorithms import AlgorithmError, Algorithms


def planning_inputs():
    _, snapshot, scene, evaluations, _ = evaluated()
    for feedback in snapshot["arms"].values():
        feedback["gripper_position_rad"] = 0.
    return snapshot, "red", "A", evaluations, scene


def use_http_transport(monkeypatch, handler):
    client_class = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_class(transport=transport, **kwargs))
    return Algorithms(graspgen_client=object())


@pytest.mark.parametrize("status", [200, 422])
def test_service_error_keeps_code_details_and_message(monkeypatch, status):
    details = {"phase": "grasp", "status": "IK_FAIL", "arm": "A"}
    algorithms = use_http_transport(monkeypatch, lambda request: httpx.Response(status, json={
        "ok": False, "error": {"code": "CUROBO_GRASP_FAILED", "message": "grasp planning failed",
                                "details": details}}))
    with pytest.raises(AlgorithmError) as caught:
        algorithms._request("/tools/plan_grasp", {"arm": "A"})
    assert caught.value.code == "CUROBO_GRASP_FAILED"
    assert caught.value.details == details
    assert "cuRobo /tools/plan_grasp" in str(caught.value)
    assert "grasp planning failed" in str(caught.value)


@pytest.mark.parametrize("exception,code", [
    (httpx.ReadTimeout, "algorithm_request_timeout"),
    (httpx.ConnectError, "algorithm_service_unavailable"),
])
def test_transport_failures_have_actionable_error_codes(monkeypatch, exception, code):
    def handler(request):
        raise exception("test transport failure", request=request)
    algorithms = use_http_transport(monkeypatch, handler)
    with pytest.raises(AlgorithmError) as caught:
        algorithms._request("/tools/plan_grasp", {})
    assert caught.value.code == code
    assert "/tools/plan_grasp" in str(caught.value)


@pytest.mark.parametrize("content", [b"not json", b"[]", b"null"])
def test_invalid_response_is_a_domain_error(monkeypatch, content):
    algorithms = use_http_transport(monkeypatch, lambda request: httpx.Response(200, content=content))
    with pytest.raises(AlgorithmError) as caught:
        algorithms._request("/tools/plan_grasp", {})
    assert caught.value.code == "algorithm_invalid_response"


def test_unstructured_service_error_still_has_default_code(monkeypatch):
    algorithms = use_http_transport(monkeypatch, lambda request: httpx.Response(503, json={
        "ok": False, "error": "planner unavailable"}))
    with pytest.raises(AlgorithmError) as caught:
        algorithms._request("/tools/plan_grasp", {})
    assert caught.value.code == "algorithm_error"
    assert "planner unavailable" in str(caught.value)


def test_candidate_failures_reported_as_batch_results_without_serial_retry():
    events = []
    calls = []
    def request(path, body):
        calls.append((path, body))
        if path == "/tools/plan_grasp_batch":
            return batch_response(body, successful=[1])
        assert path == "/carrying/validate"
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        return {"scene_token": body["scene_token"], "binding_sha256": digest,
                "collision_free": True, "attached": False}
    algorithms = Algorithms(curobo_request=request)
    result = algorithms.plan_grasp(*planning_inputs(), progress=lambda *event: events.append(event))
    planning_calls = [body for path, body in calls if path == "/tools/plan_grasp_batch"]
    validation_calls = [body for path, body in calls if path == "/carrying/validate"]
    assert len(planning_calls) == 1
    assert len(planning_calls[0]["goal_matrices_world"]) == 2
    assert [body["phase"] for body in validation_calls] == ["approach_grasp", "grasp"]
    assert all(body["gripper_geometry_mode"] == "fixed_open" for body in validation_calls)
    assert result["candidate_index"] == 100
    assert list(result["phases"]) == ["approach", "grasp", "lift"]
    assert result["excluded_object_ids"] == ["red"]
    assert result["phases"]["grasp"]["dt"] == .025
    assert len(result["phases"]["grasp"]["positions_rad"]) == 4
    assert result["prior_failures"][0]["candidate_index"] == 101
    assert result["prior_failures"][0]["status"] == "IK_FAIL"
    first = events[0][2]
    assert first == {"arm": "A", "planning_stage": "grasp", "batch_index": 1, "batch_count": 1,
                     "candidate_indices": [101, 100], "duration_s": 0., "success_count": 0}
    completed = next(event[2] for event in events if event[0] == "planning_batch_complete")
    assert completed["duration_s"] >= 0
    assert completed["failures"][0]["candidate_index"] == 101


@pytest.mark.parametrize("code", [
    "CUROBO_BACKEND_FAILED", "CUROBO_INVALID_RESULT", "algorithm_request_timeout",
    "algorithm_service_unavailable", "algorithm_error",
])
def test_service_failure_stops_after_one_request(code):
    calls = []
    failure = AlgorithmError("planner service failed", code=code)
    def request(path, body):
        calls.append(path)
        raise failure
    algorithms = Algorithms(curobo_request=request)
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(*planning_inputs())
    assert calls == ["/tools/plan_grasp_batch"]
    assert caught.value is failure


@pytest.mark.parametrize("details", [None, {"arm": "A"}, {"arm": "A", "phase": "unknown"}])
def test_wrapped_optimizer_exception_is_not_retried_as_an_infeasible_candidate(details):
    calls = []
    failure = AlgorithmError("cuRobo grasp trajectory optimization failed",
                             code="CUROBO_GRASP_FAILED", details=details)
    def request(path, body):
        calls.append(path)
        raise failure
    algorithms = Algorithms(curobo_request=request)
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(*planning_inputs())
    assert calls == ["/tools/plan_grasp_batch"]
    assert caught.value is failure


@pytest.mark.parametrize("result_kind", ["success", "candidate_failure"])
def test_cancellation_during_request_discards_result_and_stops_retries(result_kind):
    cancelled = threading.Event()
    calls = []
    events = []
    def request(path, body):
        calls.append(path)
        cancelled.set()
        if result_kind == "candidate_failure":
            raise AlgorithmError("candidate infeasible", code="CUROBO_GRASP_FAILED")
        return batch_response(body)
    algorithms = Algorithms(curobo_request=request)
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(*planning_inputs(), cancelled=cancelled,
                              progress=lambda *event: events.append(event))
    assert caught.value.code == "planning_cancelled"
    assert calls == ["/tools/plan_grasp_batch"]
    assert [event[0] for event in events] == ["planning_batch_start"]


def test_cancelled_plan_does_not_submit_any_request():
    cancelled = threading.Event()
    cancelled.set()
    calls = []
    algorithms = Algorithms(curobo_request=lambda *args: calls.append(args))
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(*planning_inputs(), cancelled=cancelled)
    assert caught.value.code == "planning_cancelled"
    assert calls == []


@pytest.mark.parametrize("cancel_on,expected_requests", [
    ("planning_batch_start", 0), ("planning_batch_complete", 1),
])
def test_cancel_from_progress_prevents_the_next_request(cancel_on, expected_requests):
    cancelled = threading.Event()
    calls = []
    def request(path, body):
        calls.append(path)
        return batch_response(body, successful=[])
    def progress(kind, title, data):
        if kind == cancel_on:
            cancelled.set()
    algorithms = Algorithms(curobo_request=request)
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(*planning_inputs(), cancelled=cancelled, progress=progress)
    assert caught.value.code == "planning_cancelled"
    assert len(calls) == expected_requests


def test_all_infeasible_candidates_report_terminal_failure_with_evidence():
    calls = []
    def request(path, body):
        calls.append(path)
        return batch_response(body, successful=[])
    algorithms = Algorithms(curobo_request=request)
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(*planning_inputs())
    assert len(calls) == 1
    assert caught.value.code == "planning_no_valid_candidate"
    assert caught.value.details["arm"] == "A"
    assert [entry["candidate_index"] for entry in caught.value.details["failures"]] == [101, 100]
