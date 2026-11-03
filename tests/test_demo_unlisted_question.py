"""Visible unlisted-object evidence is distinct from hidden-source completeness."""
import json

import httpx
import pytest
from test_visual_inventory import fixture

from luna_mujoco.demo_perception import DemoVisualInventory


def inventory_call(*, reference=None, answer="none", confidence=.99):
    capture, _ = fixture()
    entities = {
        "source-bin": {"kind": "bin", "color": "blue", "roles": ["container", "collision"]},
        "source-pink": {"kind": "cylinder", "color": "pink", "initial_parent_id": "source-bin",
                        "visual_membership": "excluded", "roles": ["collision"]},
        "other-pink": {"kind": "cylinder", "color": "pink", "initial_parent_id": "table",
                       "visual_membership": "excluded", "roles": ["collision"]},
    }
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = [{"name": question["name"], "type": "choice",
                    "choice": answer if question["name"] == "scene_inventory_closure" else "visible",
                    "confidence": confidence if question["name"] == "scene_inventory_closure" else .99}
                   for question in payload["questions"]]
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers})

    inventory = DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = inventory.resolve(capture, [], instruction="move all", entities=entities, reference=reference,
                               purpose="verify_transfer")
    question = next(item for item in calls[0]["questions"] if item["name"] == "scene_inventory_closure")
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    return result, question, context


def test_question_uses_declared_task_candidate_ids_and_only_visible_evidence():
    task = {"source_ref": "frozen-bin", "target_refs": ["frozen-pink-1", "frozen-pink-2"],
            "target_selector": "requested task objects"}
    _, question, context = inventory_call(reference={"task_spec": task})
    assert context["source_inventory"] == {
        "source_id": "frozen-bin", "source_object_ids": task["target_refs"]}
    assert all(identifier in question["instructions"] for identifier in task["target_refs"])
    assert "CURRENT full RGB" in question["instructions"]
    assert "does not certify hidden space" in question["instructions"]
    assert "task_spec.target_selector" in question["instructions"]
    assert "Do not infer hidden contents or empty space" in context["rules"]


def test_without_task_context_color_and_parentage_do_not_invent_task_candidates():
    _, question, context = inventory_call()
    assert context["source_inventory"] == {"source_id": None, "source_object_ids": []}
    guide = json.loads(question["instructions"].split(" Visible-source reference guide: ", 1)[1])
    assert guide["references"] == []
    assert guide["listed_reference_count"] == 0


@pytest.mark.parametrize("answer,confidence,expected", [
    ("none", .79, "unknown"),
    ("none", .8, "none"),
    ("unlisted", .79, "unknown"),
    ("unknown", .99, "unknown"),
])
def test_visible_scope_change_keeps_confidence_and_unknown_parsing(answer, confidence, expected):
    result, _, _ = inventory_call(answer=answer, confidence=confidence)
    assert result["demo_inventory_check"]["status"] == expected
    assert result["demo_inventory_check"]["confidence"] == confidence
    assert result["demo_inventory_check"]["observation_id"] == "frame-1"
    assert result["evidence"]["answers"][-1]["choice"] == answer
