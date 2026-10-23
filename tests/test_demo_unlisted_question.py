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
                    "choice": answer if question["name"] == "demo_unlisted_pink" else "visible",
                    "confidence": confidence if question["name"] == "demo_unlisted_pink" else .99}
                   for question in payload["questions"]]
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers})

    inventory = DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = inventory.resolve(capture, [], instruction="move all", entities=entities, reference=reference,
                               purpose="verify_transfer")
    question = next(item for item in calls[0]["questions"] if item["name"] == "demo_unlisted_pink")
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    return result, question, context


def test_question_names_permanent_task_source_ids_and_assesses_only_visible_subjects():
    task = {"source_bin_id": "frozen-bin", "source_object_ids": ["frozen-pink-1", "frozen-pink-2"]}
    _, question, context = inventory_call(reference={"demo_task": task})
    assert context["source_inventory"] == task
    assert all(identifier in question["instructions"]
               for identifier in ["frozen-bin", "frozen-pink-1", "frozen-pink-2"])
    assert "VISIBLE" in question["instructions"]
    assert "Occluded interior alone does not mean extra or unknown" in question["instructions"]
    assert "does not claim the source bin is empty or hidden contents are absent" in question["instructions"]
    assert "none requires a clear source-bin view" not in question["instructions"]
    assert "every visible pink task subject matches a permanent source ID" in question["instructions"]


def test_without_demo_task_source_context_uses_container_parentage_not_all_pink_entities():
    _, question, context = inventory_call()
    assert context["source_inventory"] == {"source_bin_id": "source-bin", "source_object_ids": ["source-pink"]}
    assert "source-bin" in question["instructions"] and "source-pink" in question["instructions"]
    assert "other-pink" not in question["instructions"]


@pytest.mark.parametrize("answer,confidence,expected", [
    ("none", .79, "unknown"),
    ("none", .8, "none"),
    ("extra", .79, "unknown"),
    ("unknown", .99, "unknown"),
])
def test_visible_scope_change_keeps_confidence_and_unknown_parsing(answer, confidence, expected):
    result, _, _ = inventory_call(answer=answer, confidence=confidence)
    assert result["demo_inventory_check"]["status"] == expected
    assert result["demo_inventory_check"]["confidence"] == confidence
    assert result["demo_inventory_check"]["observation_id"] == "frame-1"
    assert result["evidence"]["answers"][-1]["choice"] == answer
