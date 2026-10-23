"""Regressions from the frozen 30dd post-place request, without model calls."""
import json
from copy import deepcopy
from pathlib import Path

from test_demo_identity_presentation import source_payload
from test_demo_partial_identity import inputs
from test_visual_inventory import client_with_answers

from luna_mujoco.demo_identity_presentation import improve_identity_presentation
from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.visual_inventory import choice

FIXTURE = Path(__file__).parent / "fixtures/demo_postplace_30dd_request.json"
TARGET = "instance-a03f57d4b5fe4633"


def test_choice_questions_do_not_share_mutable_options():
    values = [{"value": "pink", "description": "pink object"}]
    first = choice("one", "first", values)
    second = choice("two", "second", values)
    first["choices"][0]["description"] = "changed"
    assert second["choices"][0]["description"] == values[0]["description"] == "pink object"


def test_actual_30dd_options_have_one_label_and_presentation_is_idempotent():
    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload = frozen["request"]
    before = deepcopy(payload)
    options = payload["questions"][10]["choices"]
    assert next(item["description"] for item in options if item["value"] == TARGET).count("R4 · ") == 13
    result = improve_identity_presentation(payload)
    for question in result["questions"]:
        for item in question["choices"]:
            if item["value"] == TARGET:
                assert item["description"] == "R4 · Leftmost visible pink oval-shaped object in the source box"
    assert improve_identity_presentation(result) == result
    assert payload == before
    assert [q["name"] for q in result["questions"]] == [q["name"] for q in payload["questions"]]
    assert [[c["value"] for c in q["choices"]] for q in result["questions"]] == [
        [c["value"] for c in q["choices"]] for q in payload["questions"]]
    result["questions"][0]["choices"][0]["description"] = "changed"
    assert result["questions"][1]["choices"][0]["description"] != "changed"


def test_reapplying_presentation_keeps_identical_images_zooms_and_instructions():
    payload, _, _, _ = source_payload()
    payload["questions"].append(choice("demo_unlisted_pink", "Check visible subjects.", ["none", "extra", "unknown"]))
    # Reproduce the runtime alias, unlike JSON transport which separates dictionaries.
    for question in payload["questions"][1:3]:
        question["choices"] = payload["questions"][0]["choices"]
    first = improve_identity_presentation(payload)
    assert improve_identity_presentation(first) == first
    assert len([p for p in first["input"][0]["content"] if p["type"] == "input_image"]) == 6
    first["questions"][0]["choices"][0]["description"] = "changed"
    assert first["questions"][1]["choices"][0]["description"] != "changed"


def test_actual_recorded_release_context_is_in_request_without_claiming_visual_success():
    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    capture, detections, entities, reference = inputs(("2",))
    entities = {TARGET: dict(entities["pink"], id=TARGET)}
    reference["demo_task"]["source_object_ids"] = [TARGET]
    reference["entities"] = entities
    calls = []
    answers = {"identity_0": TARGET, "visibility_" + TARGET: "visible",
               "progress_" + TARGET: "unknown", "demo_unlisted_pink": "none"}
    progress = {TARGET: frozen["controller_progress"][TARGET]}
    original = deepcopy(progress)
    result = DemoVisualInventory(api_key="test", client=client_with_answers(answers, calls)).resolve(
        capture, detections, instruction="move all", entities=entities,
        reference=reference, progress=progress, purpose="verify_transfer")
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    release = context["release_verification_context"]
    assert release["targets"][0]["entity_id"] == TARGET
    assert release["targets"][0]["release_feedback_id"] == progress[TARGET]["release_feedback_id"]
    assert release["targets"][0]["region_id"] == "region:demo-destination"
    assert release["targets"][0]["slot_id"] == 1
    assert release["targets"][0]["assigned_slot"] == progress[TARGET]["assigned_slot"]
    assert "not proof" in release["rule"]
    assert "unknown" in release["rule"]
    assert context["transport_progress"][TARGET]["release_feedback_id"] == progress[TARGET]["release_feedback_id"]
    assert context["transport_progress"][TARGET]["status"] == "selected"
    assert result["progress"][TARGET]["status"] == "unknown"
    assert progress == original
    assert len(calls) == 1


def test_release_context_is_not_inferred_for_unreleased_or_completed_targets():
    from luna_mujoco.demo_identity_presentation import release_verification_context
    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    progress = frozen["controller_progress"]
    context = release_verification_context(progress, "verify_transfer")
    assert [item["entity_id"] for item in context["targets"]] == [TARGET]
    assert release_verification_context(progress, "initialize_inventory") is None
    progress[TARGET]["status"] = "completed"
    assert release_verification_context(progress, "verify_transfer") is None
    progress[TARGET].pop("release_feedback_id")
    progress[TARGET]["status"] = "selected"
    assert release_verification_context(progress, "verify_transfer") is None
