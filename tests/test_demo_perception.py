"""Controller-provided manifests preserve measured views and fixed task IDs."""
import json
from copy import deepcopy

from graph_contract_fixtures import call, linear_graph

import httpx
import pytest
from test_perception import capture_fixture
from test_visual_inventory import client_with_answers, fixture

from luna_mujoco.agent.responses import ResponsesCompiler


def discovery_output(count=None):
    def entity(ref, kind, color, bbox, parent=None, camera="2"):
        return {"ref": ref, "kind": kind, "color": color, "description": ref, "bbox": bbox,
                "membership": "required" if color == "pink" else "excluded",
                "segmentation_prompt": "pink object" if color == "pink" else color + " " + kind,
                "parent_ref": parent, "reference_camera_id": camera}
    return {"entities": [
        entity("large", "table", "gray", [0, 0, 1, 1], camera="3"),
        entity("small", "table", "gray", [0, 0, 1, 1]),
        entity("blue", "bin", "blue", [0, 0, 1, 1], "small"),
        entity("right", "object", "pink", [.6, .10, .8, .3], "blue"),
        entity("left", "object", "pink", [.1, .11, .3, .31], "blue"),
        entity("bottom", "object", "pink", [.1, .6, .3, .8], "blue")],
        "task": {"instruction": "fixture", "supported": True, "reason": "",
            "scope": "all" if count is None else "count", "quantity": count,
            "target_selector": "pink source objects", "target_refs": ["left", "right", "bottom"],
            "source_ref": "blue", "destination_ref": "large", "work_area_ref": None,
            "selection_order": "nearest", "processing": [], "completion_conditions": ["placed"],
            "constraints": [], "source_inventory_complete": "confirmed"},
        # These fixtures exercise inventory proposals, not physical task completion.
        "graph": linear_graph([call("observe_inventory", "observe_scene")])}


def discovery_client(body, requests):
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        output = deepcopy(body)
        output['task']['instruction'] = json.loads(payload['input'][0]['content'][0]['text'])['instruction']
        return httpx.Response(200, json={"status": "completed", "model": "gpt-6-luna", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(output)}]}]})
    return httpx.Client(transport=httpx.MockTransport(respond))


def demo_discovery(body, requests):
    catalog = {'observe_scene': {'arguments': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}
    catalog['vla_unpack'] = {'arguments': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                             'processing_effect': {'fixture': True}}
    return ResponsesCompiler(registry=catalog, api_key="test", client=discovery_client(body, requests))


def test_raw_bootstrap_is_not_a_demo_runtime_entry():
    from luna_mujoco.demo_perception import DemoPerception
    from luna_mujoco.perception import PerceptionError

    with pytest.raises(PerceptionError, match='demo_initial_observation_required'):
        DemoPerception().bootstrap(capture_fixture()[0])



@pytest.mark.parametrize("status", ["none", "unknown"])
def test_demo_decisions_checks_unlisted_pink_and_ignores_configured_marker_identity(status):
    from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory

    capture, detection = fixture()
    entities = {"pink": {"id": "pink", "kind": "cube", "color": "pink", "visual_membership": "required"},
        "demo-destination": {"id": "demo-destination", "kind": "marker", "source": "configured_demo_region"}}
    calls = []
    inventory = DemoVisualInventory(api_key="test", client=client_with_answers(
        {"identity_0": "pink", "visibility_pink": "visible", "scene_inventory_closure": status}, calls))
    result = inventory.resolve(capture, [detection], instruction="move all", entities=entities,
        reference={"task_spec": {"source_ref": "blue", "target_refs": ["pink"]}})
    assert result["demo_inventory_check"]["status"] == status
    assert "visibility_demo-destination" not in [question["name"] for question in calls[0]["questions"]]
    assert result["groups"][0][0]["entity_id"] == "pink"



def test_demo_progress_question_uses_calibrated_finger_front():
    from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory

    capture, _ = fixture()
    calls = []
    inventory = DemoVisualInventory(api_key="test", client=client_with_answers({
        "visibility_pink": "visible", "progress_pink": "unknown", "scene_inventory_closure": "none"}, calls))
    inventory.resolve(capture, [], instruction="move all",
        entities={"pink": {"kind": "cube", "color": "pink"}},
        progress={"pink": {"destination_id": "demo-destination", "region_id": "region:demo-destination", "slot_id": 1}})
    question = next(item for item in calls[0]["questions"] if item["name"] == "progress_pink")
    assert "calibrated finger front" in question["instructions"]
    assert "grasped-object reference" not in question["instructions"]
    assert "Slot XY specifies the TCP anchor" not in question["instructions"]



@pytest.mark.parametrize("completeness", ["confirmed", "unknown"])
def test_real_demo_pipeline_discovers_once_associates_first_capture_and_unions_partial_cameras(completeness):
    from luna_mujoco.demo_perception import DemoPerception
    from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory

    capture, masks = capture_fixture()
    body = discovery_output()
    body["task"]["source_inventory_complete"] = completeness
    body["entities"] = body["entities"][:5]
    body["task"]["target_refs"] = ["left", "right"]
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
                "none" if name == "scene_inventory_closure" else "visible")
            answers.append({"name": name, "type": "choice", "choice": value, "confidence": .99})
        order.append("decisions")
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers})
    proposal = discovery.discover(capture, instruction="move all pink objects")
    engine = DemoPerception(Detector(),
        visual_inventory=DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(decide))),
        robot_surface_classifier=lambda *args: {"status": "measured", "masks": {}})
    first = engine.observe(capture, instruction="move all pink objects",
        source_hint={"scene_entities": proposal["entities"], "visual_reference": proposal["reference"],
                     "visual_decision": {"source": "responses_program", "purpose": "initialize_inventory"}})
    task = first["task_spec"]
    assert task["source_inventory_complete"] == completeness
    assert first["visual_reference"]["task_spec"]["source_inventory_complete"] == completeness
    sources = {obj["id"]: obj for obj in first["objects"]}
    assert sources[task["target_refs"][0]]["coverage"]["camera_ids"] == ["3"]
    assert sources[task["target_refs"][1]]["coverage"]["camera_ids"] == ["2"]
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
    assert later["task_spec"] == task and later["policy_graph"] == first["policy_graph"]
