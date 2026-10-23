"""Demo boundaries preserve measured views, one RGB inventory and fixed task IDs."""
import json

import httpx
import pytest
from test_perception import capture_fixture
from test_visual_inventory import client_with_answers, fixture

from luna_mujoco import visual_discovery
from luna_mujoco.visual_api import VisualInventoryError


def discovery_output(count=None):
    def entity(ref, kind, color, bbox, parent=None, camera="2"):
        return {"ref": ref, "kind": kind, "color": color, "description": ref, "bbox": bbox,
                "membership": "required" if color == "pink" else "excluded",
                "segmentation_prompt": color + " " + kind, "parent_ref": parent,
                "reference_camera_id": camera}
    return {"entities": [
        entity("large", "table", "gray", [0, 0, 1, 1], camera="3"),
        entity("small", "table", "gray", [0, 0, 1, 1]),
        entity("blue", "bin", "blue", [0, 0, 1, 1], "small"),
        entity("right", "cube", "pink", [.6, .10, .8, .3], "blue"),
        entity("left", "sphere", "pink", [.1, .11, .3, .31], "blue"),
        entity("bottom", "cylinder", "pink", [.1, .6, .3, .8], "blue"),
    ], "task": {"supported": True, "reason": "", "scope": "all" if count is None else "count",
        "requested_count": count, "source_bin_ref": "blue", "destination_table_ref": "large",
        "source_inventory_complete": "confirmed"},
        "program": {"version": 1, "steps": [{"type": "stop", "reason": "test generated plan"}]}}


def discovery_client(body, requests):
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "model": "gpt-6-luna", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(body)}]}]})
    return httpx.Client(transport=httpx.MockTransport(respond))


def demo_discovery(body, requests):
    assert hasattr(visual_discovery, "DemoVisualDiscovery"), "three-camera demo discovery is missing"
    return visual_discovery.DemoVisualDiscovery(api_key="test", client=discovery_client(body, requests))


def test_demo_discovery_uses_three_rgb_and_freezes_camera2_row_order():
    capture, _ = capture_fixture()
    requests = []
    discovery = demo_discovery(discovery_output(2), requests)
    first = discovery.discover(capture, instruction="move two pink objects")
    second = discovery.discover(capture, instruction="move two pink objects")
    task = first["demo_task"]
    names = [first["entities"][key]["description"] for key in task["source_object_ids"]]
    assert names == ["left", "right", "bottom"]
    assert task["transport_ids"] == task["source_object_ids"][:2]
    assert task["destination_table_id"] == "table"
    assert len({key for key, value in first["entities"].items() if value["kind"] == "table"}) == 2
    assert first["entities"] == second["entities"]
    assert len(requests) == 1
    assert set(first["reference"]["views"]) == {"1", "2", "3"}
    assert first["demo_program"] == discovery_output()["program"]
    content = requests[0]["input"][0]["content"]
    assert sum(part["type"] == "input_image" for part in content) == 3
    assert "SIM_SECRET" not in json.dumps(requests)
    assert "skill_catalog" in content[0]["text"]
    context = json.loads(content[0]["text"])
    assert context["allowed_program_arguments"]["plan_place"] == {}
    assert context["allowed_program_arguments"]["choose_arm"]["arm"] == ["auto", "A", "B"]


def test_discovery_completeness_covers_visible_subjects_and_supported_means_expressible_task():
    capture, _ = capture_fixture()
    requests = []
    demo_discovery(discovery_output(), requests).discover(capture, instruction="move all pink objects")
    prompt = json.loads(requests[0]["input"][0]["content"][0]["text"])["instructions"]
    assert "every independently distinguishable pink source object in the primary camera 2 view is enumerated" in prompt
    assert "an auxiliary clipped view alone does not create a new entity" in prompt
    assert "does not require assigning every auxiliary fragment to an individual source ID" in prompt
    assert "credible evidence of an additional independent object or a count contradiction" in prompt
    assert "Do not omit a genuinely possible additional source object" in prompt
    assert "Occluded bare floor or box-rim pixels alone do not imply unknown" in prompt
    assert "a camera-2 source object's identity, count or source membership is uncertain" in prompt
    assert "does not certify that hidden contents are absent" in prompt
    assert "supported describes task/skill expressibility only" in prompt
    assert "supported=true and source_inventory_complete=unknown" in prompt
    assert "Do not change unknown to confirmed" in prompt
    assert "If actual source contents are occluded" not in prompt
    assert "including visible fragments that cannot be assigned to an individual subject" not in prompt
    description = visual_discovery.demo_discovery_schema()['$defs']['DemoDiscoveredTask']['properties'][
        'source_inventory_complete']['description']
    assert 'primary camera 2' in description and 'auxiliary clipped fragment' in description
    assert 'extra independent object or count contradiction' in description


def test_auxiliary_entity_returned_unknown_is_not_silently_removed_by_parser():
    capture, _ = capture_fixture()
    body = discovery_output()
    body['task']['source_inventory_complete'] = 'unknown'
    body['entities'].append({**body['entities'][3], 'ref': 'possible-extra',
        'reference_camera_id': '3', 'membership': 'unknown'})
    with pytest.raises(VisualInventoryError, match='source_inventory_unknown'):
        demo_discovery(body, []).discover(capture, instruction='move all pink objects')


@pytest.mark.parametrize("mutation,reason", [
    ("too_many", "requested_count_exceeds_confirmed_inventory"),
    ("wrong_camera", "demo_source_left_reference_required"),
    ("unknown", "source_inventory_unknown"),
    ("unsupported", "demo_task_unsupported"),
])
def test_demo_discovery_rejects_unsupported_or_unconfirmed_transport(mutation, reason):
    capture, _ = capture_fixture()
    body = discovery_output()
    if mutation == "too_many":
        body["task"].update(scope="count", requested_count=4)
    elif mutation == "wrong_camera":
        body["entities"][3]["reference_camera_id"] = "1"
    elif mutation == "unknown":
        body["entities"][3]["membership"] = "unknown"
    else:
        body["task"].update(supported=False, reason="different route")
    with pytest.raises(VisualInventoryError, match=reason):
        demo_discovery(body, []).discover(capture, instruction="move objects")


def test_raw_bootstrap_is_not_a_demo_runtime_entry():
    from luna_mujoco.demo_perception import DemoPerception
    from luna_mujoco.perception import PerceptionError

    with pytest.raises(PerceptionError, match='demo_initial_observation_required'):
        DemoPerception().bootstrap(capture_fixture()[0])


@pytest.mark.parametrize("status", ["none", "unknown"])
def test_demo_decisions_checks_unlisted_pink_and_ignores_configured_marker_identity(status):
    from luna_mujoco.demo_perception import DemoVisualInventory

    capture, detection = fixture()
    entities = {"pink": {"id": "pink", "kind": "cube", "color": "pink", "visual_membership": "required"},
        "demo-destination": {"id": "demo-destination", "kind": "marker", "source": "configured_demo_region"}}
    calls = []
    inventory = DemoVisualInventory(api_key="test", client=client_with_answers(
        {"identity_0": "pink", "visibility_pink": "visible", "demo_unlisted_pink": status}, calls))
    result = inventory.resolve(capture, [detection], instruction="move all", entities=entities,
        reference={"demo_task": {"source_bin_id": "blue", "source_object_ids": ["pink"]}})
    assert result["demo_inventory_check"]["status"] == status
    assert "visibility_demo-destination" not in [question["name"] for question in calls[0]["questions"]]
    assert result["groups"][0][0]["entity_id"] == "pink"


def test_demo_progress_question_uses_calibrated_finger_front():
    from luna_mujoco.demo_perception import DemoVisualInventory

    capture, _ = fixture()
    calls = []
    inventory = DemoVisualInventory(api_key="test", client=client_with_answers({
        "visibility_pink": "visible", "progress_pink": "unknown", "demo_unlisted_pink": "none"}, calls))
    inventory.resolve(capture, [], instruction="move all",
        entities={"pink": {"kind": "cube", "color": "pink"}},
        progress={"pink": {"destination_id": "demo-destination", "region_id": "region:demo-destination", "slot_id": 1}})
    question = next(item for item in calls[0]["questions"] if item["name"] == "progress_pink")
    assert "calibrated finger front" in question["instructions"]
    assert "grasped-object reference" not in question["instructions"]
    assert "Slot XY specifies the TCP anchor" not in question["instructions"]


@pytest.mark.parametrize("completeness", ["confirmed", "unknown"])
def test_real_demo_pipeline_discovers_once_associates_first_capture_and_unions_partial_cameras(completeness):
    from luna_mujoco.demo_perception import DemoPerception, DemoVisualInventory

    capture, masks = capture_fixture()
    body = discovery_output()
    body["task"]["source_inventory_complete"] = completeness
    body["entities"] = body["entities"][:5]
    for entity in body["entities"][3:]:
        entity.update(kind="cube", segmentation_prompt="pink object")
    for camera, view in capture["views"].items():
        view["rgb"][0, 0] = [int(camera), 0, 0]
        for mask in masks:
            view["rgb"][mask] = [240, 80, 180]
    order = []
    class Detector:
        def segment(self, rgb, text, timestamp_us):
            order.append("sam3")
            camera = str(rgb[0, 0, 0])
            if text != "pink object" or camera == "1":
                return []
            return [{"mask": masks[0 if camera == "3" else 1], "score": .98}]
    discovery_calls = []
    discovery = demo_discovery(body, discovery_calls)
    decision_calls = []
    def decide(request):
        payload = json.loads(request.content)
        decision_calls.append(payload)
        context = json.loads(payload["input"][0]["content"][0]["text"])
        ids = {entity["description"]: key for key, entity in context["fixed_entities"].items()}
        answers = []
        for question in payload["questions"]:
            name = question["name"]
            value = ids["left"] if name == "identity_0" else ids["right"] if name == "identity_1" else (
                "none" if name == "demo_unlisted_pink" else "visible")
            answers.append({"name": name, "type": "choice", "choice": value, "confidence": .99})
        order.append("decisions")
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers})
    engine = DemoPerception(Detector(), visual_discovery=discovery,
        visual_inventory=DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(decide))),
        robot_surface_classifier=lambda *args: {"status": "measured", "masks": {}})
    first = engine.observe(capture, instruction="move all pink objects",
        source_hint={"visual_decision": {"source": "responses_program", "purpose": "initialize_inventory"}})
    task = first["demo_task"]
    assert task["source_inventory_complete"] == completeness
    assert first["visual_reference"]["demo_task"]["source_inventory_complete"] == completeness
    sources = {obj["id"]: obj for obj in first["objects"]}
    assert sources[task["transport_ids"][0]]["coverage"]["camera_ids"] == ["3"]
    assert sources[task["transport_ids"][1]]["coverage"]["camera_ids"] == ["2"]
    assert first["views"]["1"]["instance_masks"] == {}
    assert first["demo_inventory_check"]["status"] == "none" and order[-1] == "decisions"
    assert len(discovery_calls) == len(decision_calls) == 1
    capture["observation_id"] = "second-capture"
    for view in capture["views"].values():
        view["frame_id"] = capture["observation_id"]
    later = engine.observe(capture, instruction="move all pink objects", source_hint={
        "visual_decision": {"source": "responses_program", "purpose": "associate_inventory"},
        "scene_entities": first["scene_entities"], "visual_reference": first["visual_reference"], "table": first["table"]})
    assert len(discovery_calls) == 1 and len(decision_calls) == 2
    assert later["demo_task"] == task and later["demo_program"] == first["demo_program"]
