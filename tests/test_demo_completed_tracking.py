"""Completed identities need fresh correspondence, not an appearance re-identification."""
import hashlib
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from test_demo_partial_identity import inputs

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError


def tracking_inputs():
    capture, detections, entities, reference = inputs(("3", "2"))
    capture.update(generation=1, observation_id="current", time=3.)
    for view in capture["views"].values():
        view.update(T_world_camera=np.eye(4),
                    intrinsics={"fx": 10., "fy": 10., "cx": 16., "cy": 12.})
    entities["remaining"] = deepcopy(entities["pink"]) | {"id": "remaining"}
    reference.update(owner_id="run", demo_task={"source_object_ids": ["pink", "remaining"]})
    slot = {"slot_id": 1, "center_world_xy_m": [.4, 0.],
            "polygon_world_xy_m": [[.3, -.1], [.5, -.1], [.5, .1], [.3, .1]],
            "floor_z_m": .56, "layout_revision": "layout"}
    binding = {"owner_id": "run", "generation": 1, "target_id": "pink", "instance_id": "pink",
               "arm": "A", "scene_token": "old-place-scene", "observation_id": "original",
               "destination_id": "demo-destination", "region_id": "region:demo-destination",
               "slot_id": 1, "layout_revision": "layout", "assigned_slot": slot, "stage_index": 0,
               "release_feedback_id": "feedback-1-1.0", "release_feedback_time": 1.,
               "source": "validated_held_attachment_place_release", "identity_source": "execution_chain"}
    binding["binding_id"] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    visual = {"source": "openai_decisions", "identity_source": "execution_chain",
              "status": "at_destination", "confidence": .94, "observation_id": "post-place",
              "request_hash": "decisions-post-place", "completion_semantics": "released_in_region_and_separate",
              **{key: binding[key] for key in ("binding_id", "target_id", "destination_id", "region_id",
                                              "slot_id", "layout_revision", "release_feedback_id")}}
    progress = {"pink": {"status": "completed", "destination_id": "demo-destination",
                         "stage_index": 0, "release_feedback_id": binding["release_feedback_id"],
                         "region_id": binding["region_id"], "slot_id": 1, "assigned_slot": deepcopy(slot),
                         "completed_identity_track": {"binding": binding, "visual_evidence": visual,
                             "capture": {"generation": 1, "observation_id": "post-place", "time": 2.}}},
                "remaining": {"status": "observed"}}
    for detection, point in zip(detections, ([.4, 0., .63], [.2, .5, .50]), strict=True):
        detection["points"] = np.tile(point, (len(detection["points"]), 1))
    return capture, detections, entities, reference, progress


def resolve(values=None, *, mutate=None):
    capture, detections, entities, reference, progress = tracking_inputs()
    if mutate:
        mutate(capture, detections, entities, reference, progress)
    calls = []
    values = values or {"tracked_0": ("yes", .99), "identity_1": ("remaining", .95)}

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for question in payload["questions"]:
            name = question["name"]
            fallback = ("yes" if name.startswith("tracked_") else
                        "none" if name == "demo_unlisted_pink" else
                        "elsewhere" if name.startswith("progress_") else "visible")
            value, confidence = values.get(name, (fallback, .99))
            answers.append({"name": name, "type": "choice", "choice": value, "confidence": confidence})
        return httpx.Response(200, json={"answers": answers, "model": "gpt-6-luna"})

    inventory = DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = inventory.resolve(capture, detections, instruction="move all", reference=reference,
                               entities=entities, progress=progress, purpose="associate_inventory")
    return result, calls, (capture, detections, entities, reference, progress)


def test_scoped_completed_candidate_keeps_required_identity_and_current_geometry():
    result, calls, (_, detections, _, _, _) = resolve()
    assert len(calls) == 2
    tracked = calls[0]["questions"]
    assert [q["name"] for q in tracked] == ["tracked_0"]
    assert [c["value"] for c in tracked[0]["choices"]] == ["yes", "no", "unknown"]
    questions = {q["name"]: q for q in calls[1]["questions"]}
    assert "identity_0" not in questions
    assert [c["value"] for c in questions["identity_1"]["choices"]] == [
        "remaining", "reject", "unknown"]
    assert "visibility_pink" not in questions
    assert "progress_pink" not in questions
    assert "progress_remaining" in questions
    assert result["identity_coverage"]["required_entity_ids"] == ["pink", "remaining"]
    assert result["identity_coverage"]["remaining_source_ids"] == ["remaining"]
    assert result["missing_geometry_ids"] == []
    assert "pink" not in result["progress"]
    group = next(group for group in result["groups"] if group[0]["entity_id"] == "pink")
    np.testing.assert_array_equal(group[0]["points"], detections[0]["points"])
    tracking = result["completed_tracking"]
    assert tracking["response"]["answers"][0]["choice"] == "yes"
    assert tracking["candidate_mapping"]["0"] == "pink"
    assert result["identity_coverage"]["request_hashes"]["completed_correspondence"] == tracking["request_hash"]
    assert result["identity_coverage"]["request_hashes"]["remaining_inventory"] == result["evidence"]["request_hash"]


@pytest.mark.parametrize("value,confidence", [("yes", .79), ("unknown", .99), ("no", .99)])
def test_geometry_scope_never_promotes_unknown(value, confidence):
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        resolve({"tracked_0": (value, confidence), "identity_1": ("remaining", .95)})


def test_scoped_answer_cannot_choose_a_different_owner():
    with pytest.raises(VisualInventoryError, match="visual_decisions_invalid_response"):
        resolve({"tracked_0": ("remaining", .99), "identity_1": ("remaining", .95)})


@pytest.mark.parametrize("field,value", [
    ("generation", 2), ("owner_id", "other-run"), ("slot_id", 2),
    ("release_feedback_id", "other-release"), ("target_id", "remaining"),
])
def test_changed_completed_binding_fails_before_decisions(field, value):
    def mutate(capture, detections, entities, reference, progress):
        progress["pink"]["completed_identity_track"]["binding"][field] = value
    with pytest.raises(VisualInventoryError, match="completed_tracking"):
        resolve(mutate=mutate)


@pytest.mark.parametrize("case", [
    "changed_live_slot", "wrong_visual_slot", "low_visual_confidence", "missing_request_hash",
    "current_generation", "reused_frame", "older_frame", "same_acquisition", "other_active_run",
])
def test_inconsistent_completion_track_never_narrows_identity(case):
    def mutate(capture, detections, entities, reference, progress):
        track = progress["pink"]["completed_identity_track"]
        if case == "changed_live_slot":
            progress["pink"]["assigned_slot"]["center_world_xy_m"][0] += .02
        elif case == "wrong_visual_slot":
            track["visual_evidence"]["slot_id"] = 2
        elif case == "low_visual_confidence":
            track["visual_evidence"]["confidence"] = .79
        elif case == "missing_request_hash":
            del track["visual_evidence"]["request_hash"]
        elif case == "current_generation":
            capture["generation"] = 2
        elif case == "reused_frame":
            capture["observation_id"] = "post-place"
        elif case == "older_frame":
            capture["time"] = 1.9
        elif case == "same_acquisition":
            track["capture"]["captured_perf_counter_ns"] = capture["captured_perf_counter_ns"] = 123
        else:
            reference["owner_id"] = "different-run"
    with pytest.raises(VisualInventoryError, match="completed_tracking"):
        resolve(mutate=mutate)


@pytest.mark.parametrize("point", [
    [.51, 0., .63], [.4, .11, .63], [.4, 0., .70], [.4, 0., .53], [float("nan"), 0., .63],
])
def test_one_point_outside_slot_or_height_prevents_locating_scope(point):
    def mutate(capture, detections, entities, reference, progress):
        detections[0]["points"][0] = point
    # Exercise the local scope independently: a nonfinite cloud would correctly
    # fail later geometry processing, even though it must never narrow identity.
    from luna_mujoco.demo_completed_tracking import completed_candidate_scopes, completed_identity_tracks
    capture, detections, entities, reference, progress = tracking_inputs()
    mutate(capture, detections, entities, reference, progress)
    tracks = completed_identity_tracks(capture, reference, progress, entities)
    assert completed_candidate_scopes(capture, detections, tracks) == {}


def test_overlapping_completed_slots_are_rejected_instead_of_picking_first_owner():
    def mutate(capture, detections, entities, reference, progress):
        other = deepcopy(progress["pink"])
        binding = other["completed_identity_track"]["binding"]
        binding.update(target_id="remaining", instance_id="remaining", slot_id=2)
        binding["assigned_slot"]["slot_id"] = 2
        binding["binding_id"] = hashlib.sha256(json.dumps(
            {key: value for key, value in binding.items() if key != "binding_id"}, sort_keys=True).encode()).hexdigest()
        other.update(slot_id=2, assigned_slot=deepcopy(binding["assigned_slot"]))
        other["completed_identity_track"]["visual_evidence"].update(
            target_id="remaining", slot_id=2, binding_id=binding["binding_id"])
        progress["remaining"] = other
    with pytest.raises(VisualInventoryError, match="completed_slots_overlap"):
        resolve(mutate=mutate)


def test_negative_completed_correspondence_cannot_drop_the_known_collision_entity():
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown") as caught:
        resolve({"tracked_0": ("no", .99), "identity_1": ("remaining", .95)})
    assert caught.value.evidence["missing_required_ids"] == ["pink"]


def test_same_slot_aliases_remain_actual_measured_pixel_union():
    def mutate(capture, detections, entities, reference, progress):
        detections.append(deepcopy(detections[0]))
    result, calls, (_, detections, _, _, _) = resolve({
        "tracked_0": ("yes", .9), "identity_1": ("remaining", .95), "tracked_2": ("yes", .95)}, mutate=mutate)
    group = next(group for group in result["groups"] if group[0]["entity_id"] == "pink")
    assert len(group) == 1
    assert len(group[0]["pixels_rc"]) == len(detections[0]["pixels_rc"])
    assert [q["name"] for q in calls[0]["questions"]] == ["tracked_0", "tracked_2"]
    assert len(result["completed_tracking"]["candidate_scopes"]) == 2


def test_current_only_request_and_remaining_payload_have_distinct_bound_evidence():
    result, calls, _ = resolve()
    first, second = calls
    content = first["input"][0]["content"]
    assert sum(part["type"] == "input_image" for part in content) == 2  # Identical raw views share bytes; the scoped head remains.
    texts = [part["text"] for part in content if part["type"] == "input_text"]
    assert not any("INITIAL" in text for text in texts)
    assert not any("canonical permanent ID" in question["instructions"] for question in first["questions"])
    assert not any("identity_" in question["name"] for question in first["questions"])
    context = json.loads(second["input"][0]["content"][0]["text"])
    assert "pink" not in context["fixed_entities"]
    assert context["fixed_reference_labels"]["remaining"] == "R2"  # Preserve the original full-inventory label.
    current = context["completed_source_correspondences"]
    assert current[0]["entity_id"] == "pink"
    assert current[0]["status"] == "confirmed"
    assert current[0]["current_candidates"] == [{"candidate": "C0", "camera": "3",
        "box": [0.09375, 1/6, 0.4375, 2/3]}]
    assert current[0]["request_hash"] == result["completed_tracking"]["request_hash"]
    question = next(question for question in second["questions"] if question["name"] == "demo_unlisted_pink")
    guide = json.loads(question["instructions"].split("Visible-source reference guide: ")[1])
    assert guide["references"][0]["reference_kind"] == "same_capture_completed_correspondence"
    assert guide["references"][0]["current_candidates"][0]["candidate"] == "C0"
    assert "reference_boxes" not in guide["references"][0]


@pytest.mark.parametrize("field,value", [
    ("observation_id", "different-frame"), ("generation", 2), ("request_hash", "different-hash"),
])
def test_two_request_join_rejects_changed_capture_or_hash(field, value):
    from luna_mujoco.demo_completed_tracking import merge_completed_correspondence
    result, _, (capture, detections, entities, _, _) = resolve()
    tracking = deepcopy(result["completed_tracking"])
    tracking[field] = value
    with pytest.raises(VisualInventoryError, match="completed_tracking_request_binding_mismatch"):
        merge_completed_correspondence(DemoVisualInventory(), capture, detections, entities, tracking, result)


def test_unknown_unlisted_check_never_completes_dual_request_merge():
    with pytest.raises(VisualInventoryError, match="source_inventory_unknown"):
        resolve({"tracked_0": ("yes", .99), "identity_1": ("remaining", .95),
                 "demo_unlisted_pink": ("none", .79)})


def test_missing_current_completed_geometry_stops_before_any_model_call():
    def mutate(capture, detections, entities, reference, progress):
        detections[0]["points"][:, 2] = .8
    with pytest.raises(VisualInventoryError, match="completed_tracking_current_geometry_missing"):
        resolve(mutate=mutate)


def test_invalid_binary_response_keeps_scope_evidence_and_does_not_call_remaining_request():
    capture, detections, entities, reference, progress = tracking_inputs()
    calls = []

    def invalid(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": [{"name": "tracked_0", "type": "choice",
                                                      "choice": "yes", "confidence": True}]})

    inventory = DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(invalid)))
    with pytest.raises(VisualInventoryError, match="visual_decisions_invalid_response") as caught:
        inventory.resolve(capture, detections, instruction="move all", reference=reference, entities=entities,
                          progress=progress, purpose="associate_inventory")
    assert len(calls) == 1
    assert caught.value.evidence["completed_tracking"]["candidate_mapping"] == {}
    assert caught.value.evidence["completed_tracking"]["candidate_scopes"][0]["candidate"] == "candidate-0"


def test_invalid_extra_or_uncertain_noncompleted_id_still_stops():
    with pytest.raises(VisualInventoryError, match="demo_unlisted_pink_object"):
        resolve({"tracked_0": ("yes", .99), "identity_1": ("remaining", .95),
                 "demo_unlisted_pink": ("extra", .9)})
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        resolve({"tracked_0": ("yes", .99), "identity_1": ("remaining", .79)})


async def test_runtime_preserves_per_object_chain_across_transfer_reset(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.demo_runner import DemoRunner

    class InspectTracking(DemoVision):
        def observe(self, capture, **kwargs):
            hint = kwargs.get("source_hint", {})
            if hint.get("visual_progress", {}).get("pink-1", {}).get("status") == "completed":
                record = hint["visual_progress"]["pink-1"]
                assert "completed_identity_track" in record
                track = record["completed_identity_track"]
                assert track["binding"]["target_id"] == "pink-1"
                assert track["binding"]["owner_id"] == hint["visual_reference"]["owner_id"]
                assert track["visual_evidence"]["binding_id"] == track["binding"]["binding_id"]
            return super().observe(capture, **kwargs)

    runner = DemoRunner(DemoRobot(), InspectTracking(2), None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    first = runner.task_progress.records["pink-1"]
    assert first["destination_visual_evidence"]["identity_source"] == "execution_chain"
    assert first["completed_identity_track"]["binding"]["slot_id"] == 1


async def test_unconfirmed_release_creates_no_completed_track(tmp_path):
    from test_demo_runner import run_demo
    runner, _, _ = await run_demo(tmp_path, count=1, progress="unknown")
    assert runner.state["status"] == "needs_attention"
    assert "completed_identity_track" not in runner.task_progress.records["pink-1"]
