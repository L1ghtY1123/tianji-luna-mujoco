import base64
import copy
import json
from io import BytesIO

import httpx
import numpy as np
import pytest
from PIL import Image

from jev_mujoco import visual_assistance


def capture(identifier="scene"):
    views = {}
    for camera in ("1", "2", "3"):
        mask = np.zeros((80, 120), bool)
        mask[20:40, 15:35] = True
        bin_mask = np.zeros_like(mask)
        bin_mask[10:65, 10:100] = True
        views[camera] = {"rgb": np.zeros((80, 120, 3), np.uint8),
                         "instance_masks": {"cube-1": mask, "bin-1": bin_mask}}
    return {"source": "rgbd", "observation_id": identifier, "generation": 1, "views": views}


OBJECTS = [{"id": "cube-1", "kind": "cube", "color": "red", "position": [9, 8, 7]},
           {"id": "bin-1", "kind": "bin", "color": "blue"}]


async def test_runner_rejects_repeat_visual_input_but_accepts_new_masks(tmp_path):
    from test_rgbd_workflow import SensorRobot, VisionAlgorithms

    from jev_mujoco.runner import Runner
    from jev_mujoco.skills import SkillSelectionError

    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: None, tmp_path)
    calls = []
    async def inspect(context):
        calls.append(context['observation']['observation_id'])
        return {'status': 'unknown', 'observation_id': calls[-1]}
    runner.inspect_scene = inspect
    context = {'observation': capture(), 'initial': {}}
    await runner.execute('inspect_scene', context)
    with pytest.raises(SkillSelectionError, match='visual_input_already_analyzed'):
        await runner.execute('inspect_scene', context)
    context['observation']['views']['3']['instance_masks']['cube-1'][0, 0] = True
    await runner.execute('inspect_scene', context)
    assert len(calls) == 2


def scene_reply():
    return {"objects": [
        {"id": "cube-1", "label": "red cube", "kind": "cube", "color": "red",
         "visibility": "visible", "ambiguous": False, "on_table": False,
         "inside_id": "bin-1", "on_marker_id": None, "evidence": "Cube visible in the bin."},
        {"id": "bin-1", "label": "blue bin", "kind": "bin", "color": "blue",
         "visibility": "visible", "ambiguous": False, "on_table": True,
         "inside_id": None, "on_marker_id": None, "evidence": "Bin on table."}],
        "relationships": [{"subject_id": "cube-1", "relation": "inside", "object_id": "bin-1"}],
        "evidence": "Both supplied objects are visible."}


def client_with_reply(reply=None, status="completed"):
    requests = []

    def handle(request):
        requests.append(request)
        body = {"status": status, "output": [{"content": [{"type": "output_text", "text":
                json.dumps(reply if reply is not None else scene_reply())}]}],
                "usage": {"input_tokens": 21, "output_tokens": 12}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps({"type": "response." + status, "response": body}) + "\n\n")

    service = visual_assistance.VisualAssistance(enabled=True, model="configured-model", api_key="secret",
                                                base_url="https://configured.example/v1",
                                                client=httpx.Client(transport=httpx.MockTransport(handle)))
    return service, requests


@pytest.mark.parametrize("arrival_s,response_status,timeout_s,expected", [
    (65., "completed", None, "observed"),
    (65., "incomplete", None, "unknown"),
    (91., "completed", None, "unknown"),
    (65., "completed", 60., "unknown"),
    (95., "completed", 100., "observed"),
])
def test_configurable_deadline_accepts_only_timely_completed_scene(monkeypatch, arrival_s, response_status,
                                                                  timeout_s, expected):
    clock, requests = [0.], []
    monkeypatch.setattr(visual_assistance.time, "monotonic", lambda: clock[0])

    class DelayedStream(httpx.SyncByteStream):
        def __iter__(self):
            clock[0] = arrival_s
            body = {"status": response_status, "output": [{"content": [{"type": "output_text",
                    "text": json.dumps(scene_reply())}]}]}
            yield ("data: " + json.dumps({"type": "response." + response_status, "response": body}) + "\n\n").encode()

    def handle(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=DelayedStream())

    options = {} if timeout_s is None else {"timeout_s": timeout_s}
    service = visual_assistance.VisualAssistance(enabled=True, model="configured", api_key="private-test",
        client=httpx.Client(transport=httpx.MockTransport(handle)), **options)
    budget = visual_assistance.VisualCallBudget(limit=1)
    result = service.inspect_scene(capture(), OBJECTS, "Describe the scene", call_budget=budget)
    assert result["status"] == expected
    assert requests[0].extensions["timeout"]["read"] == (90. if timeout_s is None else timeout_s)
    if arrival_s > (90. if timeout_s is None else timeout_s):
        assert result["reason"] == "TimeoutError"
    assert service.inspect_scene(capture(), OBJECTS, "Describe the scene", call_budget=budget) == result
    assert service.inspect_scene(capture("after"), OBJECTS, "Describe the scene",
                                 call_budget=budget)["reason"] == "visual_call_budget_exhausted"
    assert budget.used == 1 and len(requests) == 1
    assert "private-test" not in json.dumps(result)


@pytest.mark.parametrize("invalid", [0, -1, 181, float("nan"), float("inf"), True, "private-invalid"])
def test_visual_timeout_validation_is_finite_bounded_and_redacted(invalid):
    with pytest.raises(ValueError, match="visual_timeout_s_must_be_finite_between_1_and_180"):
        visual_assistance.VisualAssistance(timeout_s=invalid)


def test_visual_timeout_environment_preserves_provider_configuration(tmp_path, monkeypatch):
    path = tmp_path / "provider.json"
    path.write_text(json.dumps({"api_key": "private-test", "base_url": "https://existing.invalid/v1", "model": "existing"}))
    monkeypatch.setenv("OPENAI_CONFIG_FILE", str(path))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "JEV_VISION_MODEL", "JEV_VISUAL_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    service = visual_assistance.VisualAssistance.from_environment()
    assert service.timeout_s == 90.
    monkeypatch.setenv("JEV_VISUAL_TIMEOUT_S", "120.5")
    service = visual_assistance.VisualAssistance.from_environment()
    assert service.timeout_s == 120.5
    assert service.base_url == "https://existing.invalid/v1" and service.model == "existing"
    assert service.api_key == "private-test"
    assert json.loads(path.read_text())["api_key"] == "private-test"
    monkeypatch.setenv("JEV_VISUAL_TIMEOUT_S", "nan")
    with pytest.raises(ValueError, match="visual_timeout_s_must_be_finite_between_1_and_180"):
        visual_assistance.VisualAssistance.from_environment()


def test_scene_bundles_three_grounded_views_without_metric_or_private_inputs():
    service, requests = client_with_reply()
    observed = capture()
    original = observed["views"]["3"]["rgb"].copy()
    result = service.inspect_scene(observed, OBJECTS, "Move red cubes out of the blue bin")
    assert result["status"] == "observed"
    assert result["objects"][0]["inside_id"] == "bin-1"
    assert result["relationships"][0]["subject_id"] == "cube-1"
    assert result["generation"] == 1 and result["observation_id"] == "scene"
    assert len(requests) == 1 and str(requests[0].url) == "https://configured.example/v1/responses"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "configured-model" and payload["store"] is False
    content = payload["input"][0]["content"]
    text = " ".join(item["text"] for item in content if item["type"] == "input_text")
    assert "cube-1" in text and "bin-1" in text and "overlay_number" in text
    assert "position" not in text and "[9, 8, 7]" not in text
    images = [item for item in content if item["type"] == "input_image"]
    assert len(images) == 3
    image = Image.open(BytesIO(base64.b64decode(images[0]["image_url"].split(",")[1])))
    assert np.asarray(image).max() > 0  # The real RGB payload contains the grounding overlay.
    np.testing.assert_array_equal(observed["views"]["3"]["rgb"], original)
    assert "secret" not in json.dumps(result) and "position" not in json.dumps(result)


@pytest.mark.parametrize("change", ["invented_id", "invented_relation", "invented_membership", "pose", "action", "duplicate"])
def test_scene_rejects_ungrounded_ids_or_control_outputs(change):
    reply = scene_reply()
    if change == "invented_id":
        reply["objects"][0]["id"] = "invented"
    elif change == "invented_relation":
        reply["relationships"][0]["object_id"] = "invented"
    elif change == "invented_membership":
        reply["objects"][0]["inside_id"] = "invented"
    elif change == "pose":
        reply["objects"][0]["position_world_m"] = [0.2, 0.1, 0.8]
    elif change == "action":
        reply["next_action"] = "regrasp"
    else:
        reply["objects"].append(copy.deepcopy(reply["objects"][0]))
    service, _ = client_with_reply(reply)
    result = service.inspect_scene(capture(), OBJECTS, "Move the cubes")
    assert result["status"] == "unknown"
    assert result["objects"] == [] and result["relationships"] == []


@pytest.mark.parametrize("visibility,ambiguous", [("occluded", False), ("unknown", False), ("visible", True)])
def test_unseen_or_ambiguous_object_does_not_become_visible_membership(visibility, ambiguous):
    reply = scene_reply()
    reply["objects"][0].update(visibility=visibility, ambiguous=ambiguous)
    service, _ = client_with_reply(reply)
    result = service.inspect_scene(capture(), OBJECTS, "Move the cubes")
    cube = result["objects"][0]
    assert cube["inside_id"] is None and cube["on_table"] is None
    assert cube["label"] == "" and not result["relationships"]


def test_missing_mask_and_missing_model_entry_are_explicit_unknowns():
    observed = capture()
    for view in observed["views"].values():
        view["instance_masks"].pop("cube-1")
    reply = scene_reply()
    reply["objects"] = [reply["objects"][1]]
    reply["relationships"] = []
    service, _ = client_with_reply(reply)
    result = service.inspect_scene(observed, OBJECTS, "Move the cubes")
    assert result["objects"][0]["id"] == "cube-1"
    assert result["objects"][0]["visibility"] == "unknown"
    assert result["objects"][0]["ambiguous"] is True


def test_no_masks_does_not_spend_a_call_or_invent_grounding():
    observed = capture()
    for view in observed["views"].values():
        view["instance_masks"] = {}
    service, requests = client_with_reply()
    result = service.inspect_scene(observed, OBJECTS, "Move the cubes")
    assert result["status"] == "unknown" and result["reason"] == "no_grounded_objects"
    assert not requests


def test_scene_cache_invalidates_on_pixels_masks_generation_and_instruction():
    service, requests = client_with_reply()
    observed = capture()
    first = service.inspect_scene(observed, OBJECTS, "Move the cubes")
    assert service.inspect_scene(copy.deepcopy(observed), OBJECTS, "Move the cubes") == first
    assert len(requests) == 1
    observed["views"]["1"]["rgb"][0, 0] = 255
    service.inspect_scene(observed, OBJECTS, "Move the cubes")
    observed["views"]["2"]["instance_masks"]["cube-1"][50, 50] = True
    service.inspect_scene(observed, OBJECTS, "Move the cubes")
    observed["generation"] = 2
    service.inspect_scene(observed, OBJECTS, "Move the cubes")
    service.inspect_scene(observed, OBJECTS, "Move all cubes")
    assert len(requests) == 5


def test_shared_budget_covers_scene_observations_and_cache_is_free():
    service, requests = client_with_reply()
    budget = visual_assistance.VisualCallBudget(limit=1)
    first = service.inspect_scene(capture(), OBJECTS, "Move the cubes", call_budget=budget)
    assert first["status"] == "observed"
    assert service.inspect_scene(capture(), OBJECTS, "Move the cubes", call_budget=budget) == first
    result = service.inspect_scene(capture("after"), OBJECTS, "Move the cubes", call_budget=budget)
    assert result["reason"] == "visual_call_budget_exhausted" and budget.used == 1
    assert len(requests) == 1


def test_failed_scene_call_consumes_budget_is_cached_and_is_not_retried():
    service, requests = client_with_reply(status="incomplete")
    budget = visual_assistance.VisualCallBudget(limit=1)
    result = service.inspect_scene(capture(), OBJECTS, "Move the cubes", call_budget=budget)
    assert result["status"] == "unknown"
    assert service.inspect_scene(capture(), OBJECTS, "Move the cubes", call_budget=budget) == result
    assert budget.used == 1 and len(requests) == 1


@pytest.mark.parametrize("code,expected_code", [
    ("gateway_queue_full", "gateway_queue_full"),
    ("a" * 64, "a" * 64),
    ("a" * 65, None),
    ("private provider message", None),
    ("https://private.example/?secret=private", None),
    ("sk-private-token", None),
    ("gateway_queue_full\nprivate", None),
    (None, None),
    ({"private": "message"}, None),
])
def test_terminal_provider_failure_exposes_only_bounded_safe_code(code, expected_code):
    requests = []

    def handle(request):
        requests.append(request)
        body = {"status": "failed", "error": {"code": code, "message": "private provider message"},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(scene_reply())}]}]}
        return httpx.Response(200, json=body, headers={"private-header": "private credential"})

    service = visual_assistance.VisualAssistance(enabled=True, model="configured", api_key="private-api-key",
        client=httpx.Client(transport=httpx.MockTransport(handle)))
    budget = visual_assistance.VisualCallBudget(limit=1)
    result = service.inspect_scene(capture(), OBJECTS, "Describe scene", call_budget=budget)
    assert result["status"] == "unknown" and result["success"] is False
    assert result["reason"] == "visual_response_failed"
    assert result.get("provider_error_code") == expected_code
    assert not result["objects"] and not result["relationships"]
    assert "private" not in json.dumps(result)
    assert service.inspect_scene(capture(), OBJECTS, "Describe scene", call_budget=budget) == result
    assert budget.used == 1 and len(requests) == 1


@pytest.mark.parametrize("body_status", ["failed", "completed", None])
def test_failed_sse_event_cannot_be_overridden_by_completed_body_or_scene(body_status):
    def handle(request):
        body = {"error": {"code": "gateway_queue_full", "message": "private provider message"},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(
                    scene_reply() | {"evidence": "private false claim"})}]}]}
        if body_status is not None:
            body["status"] = body_status
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps({"type": "response.failed", "response": body}) + "\n\n")

    service = visual_assistance.VisualAssistance(enabled=True, model="configured", api_key="test",
        client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = service.inspect_scene(capture(), OBJECTS, "Describe scene")
    assert result["status"] == "unknown" and result["success"] is False
    assert result["reason"] == "visual_response_failed" and result["provider_error_code"] == "gateway_queue_full"
    assert "private" not in json.dumps(result)


def test_large_scene_instruction_is_rejected_without_silently_truncating_task():
    service, requests = client_with_reply()
    result = service.inspect_scene(capture(), OBJECTS, "x" * 50000)
    assert result["status"] == "unknown" and result["reason"] == "visual_input_budget_exceeded"
    assert not requests


def test_reference_images_are_bounded_and_reference_change_invalidates_cache():
    service, requests = client_with_reply()
    reference = capture("before")
    service.inspect_scene(capture(), OBJECTS, "Move the cubes", reference=reference)
    reference["views"]["3"]["rgb"][0, 0] = 255
    result = service.inspect_scene(capture(), OBJECTS, "Move the cubes", reference=reference)
    assert len(requests) == 2 and result["image_count"] == 4


def test_scene_cache_cannot_reuse_changed_pixels_or_reference_generation():
    service, requests = client_with_reply()
    observed, reference = capture(), capture("before")
    assert service.inspect_scene(observed, OBJECTS, "Describe scene", reference=reference)["status"] == "observed"
    observed["views"]["1"]["rgb"][0, 0] = 255
    service.inspect_scene(observed, OBJECTS, "Describe scene", reference=reference)
    reference["generation"] = 2
    service.inspect_scene(observed, OBJECTS, "Describe scene", reference=reference)
    assert len(requests) == 3


def test_bad_scene_capture_is_unknown_and_does_not_consume_budget():
    service, requests = client_with_reply()
    observed = capture()
    observed["views"].pop("2")
    budget = visual_assistance.VisualCallBudget(limit=1)
    result = service.inspect_scene(observed, OBJECTS, "Describe scene", call_budget=budget)
    assert result["status"] == "unknown" and budget.used == 0 and not requests


def test_unmasked_model_claim_is_rejected_even_for_a_supplied_object_id():
    service, _ = client_with_reply()
    observed = capture()
    for view in observed["views"].values():
        view["instance_masks"].pop("cube-1")
    result = service.inspect_scene(observed, OBJECTS, "Move the cubes")
    assert result["status"] == "unknown" and not result["objects"]


def test_scene_marker_membership_is_tied_to_a_supplied_mask():
    observed = capture()
    for view in observed["views"].values():
        view["instance_masks"]["marker-1"] = view["instance_masks"].pop("bin-1")
    objects = [OBJECTS[0], {"id": "marker-1", "kind": "marker"}]
    reply = scene_reply()
    reply["objects"][0].update(inside_id=None, on_marker_id="marker-1")
    reply["objects"][1].update(id="marker-1", label="yellow marked region", kind="marker", color="yellow")
    reply["relationships"] = [{"subject_id": "cube-1", "relation": "on_marker", "object_id": "marker-1"}]
    service, _ = client_with_reply(reply)
    result = service.inspect_scene(observed, objects, "Move onto the marked region")
    assert result["objects"][0]["on_marker_id"] == "marker-1"


def test_default_budget_bounds_changed_observations_without_explicit_task_budget():
    service, requests = client_with_reply()
    for index in range(8):
        assert service.inspect_scene(capture(str(index)), OBJECTS, "Move cubes")["status"] == "observed"
    result = service.inspect_scene(capture("new"), OBJECTS, "Move cubes")
    assert result["reason"] == "visual_call_budget_exhausted" and len(requests) == 8


def test_scene_limits_large_rgb_payload_dimensions():
    observed = capture()
    for view in observed["views"].values():
        view["rgb"] = np.zeros((900, 1600, 3), np.uint8)
        view["instance_masks"] = {identifier: np.tile(mask, (12, 14))[:900, :1600]
                                  for identifier, mask in view["instance_masks"].items()}
    service, requests = client_with_reply()
    assert service.inspect_scene(observed, OBJECTS, "Move cubes")["status"] == "observed"
    for item in json.loads(requests[0].content)["input"][0]["content"]:
        if item["type"] == "input_image":
            image = Image.open(BytesIO(base64.b64decode(item["image_url"].split(",")[1])))
            assert image.width <= 640 and image.height <= 640


def test_cached_scene_does_not_accept_caller_mutation():
    service, requests = client_with_reply()
    first = service.inspect_scene(capture(), OBJECTS, "Move cubes")
    first["objects"][0]["inside_id"] = "forged-id"
    result = service.inspect_scene(capture(), OBJECTS, "Move cubes")
    assert result["objects"][0]["inside_id"] == "bin-1" and len(requests) == 1


def test_scene_preserves_unicode_instruction_without_expanding_text_cost():
    service, requests = client_with_reply()
    instruction = "搬" * 7000
    result = service.inspect_scene(capture(), OBJECTS, instruction)
    assert result["status"] == "observed"
    text = json.loads(requests[0].content)["input"][0]["content"][0]["text"]
    assert json.loads(text)["instruction"] == instruction
    assert len(text) < 12000


def test_http_gateway_error_exposes_only_status_and_bounded_code():
    def handle(request):
        return httpx.Response(502, json={'error': {'code': 'bad_gateway', 'message': 'private details'}})
    service = visual_assistance.VisualAssistance(enabled=True, model='configured', api_key='private-key',
        client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = service.inspect_scene(capture(), OBJECTS, 'Compare scene')
    assert result['status'] == 'unknown'
    assert result['http_status'] == 502 and result['provider_error_code'] == 'bad_gateway'
    assert 'private' not in json.dumps(result)
