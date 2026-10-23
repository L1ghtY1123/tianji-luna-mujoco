"""Same-frame hints require matching capture provenance and exact per-camera pixels."""
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from test_visual_inventory import fixture

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import VisualInventory, mask_box


def same_capture_inputs():
    capture, detection = fixture()
    capture["views"] = {"2": capture["views"]["3"]}
    detection["camera_id"] = "2"
    entities = {"pink": {"id": "pink", "kind": "cylinder", "color": "pink",
                         "bbox": mask_box(detection["silhouette_mask"]), "reference_camera_id": "2",
                         "visual_membership": "required"}}
    reference = {"observation_id": capture["observation_id"], "entities": deepcopy(entities),
                 "views": deepcopy(capture["views"]), "demo_task": {"source_object_ids": ["pink"]}}
    return capture, [detection], entities, reference


def inventory(calls, *, value="pink", confidence=.99, legacy=False):
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = [{"name": question["name"], "type": "choice",
                    "choice": value if question["name"].startswith("identity_") else
                              "none" if question["name"] == "demo_unlisted_pink" else "visible",
                    "confidence": confidence if question["name"].startswith("identity_") else .99}
                   for question in payload["questions"]]
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers})
    cls = VisualInventory if legacy else DemoVisualInventory
    return cls(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))


def run(calls, capture, detections, entities, reference, **options):
    return inventory(calls, **options).resolve(capture, detections, entities=entities, reference=reference,
                                               instruction="move all", purpose="initialize_inventory")


def test_exact_same_capture_adds_locating_hints_without_changing_identity_options():
    capture, detections, entities, reference = same_capture_inputs()
    calls = []
    result = run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    provenance = context["same_capture_reference"]
    assert provenance["initial_reference_observation_id"] == provenance["current_observation_id"] == "frame-1"
    assert provenance["verified_camera_ids"] == ["2"]
    assert len(provenance["raw_rgb_sha256_by_camera"]["2"]) == 64
    proposal = context["same_capture_box_correspondence_proposals"][0]
    assert proposal["camera"] == "2" and proposal["candidate"] == "candidate-0"
    assert proposal["same_camera_reference_box_overlaps"] == [
        {"reference_entity_id": "pink", "reference_box": entities["pink"]["bbox"], "box_iou": 1.}]
    assert "not established identity" in context["box_correspondence_rule"]
    question = calls[0]["questions"][0]
    assert "identical frozen images" in question["instructions"]
    assert [item["value"] for item in question["choices"]] == ["pink", "reject", "unknown"]
    assert result["groups"][0][0]["entity_id"] == "pink"
    location = context["candidate_geometric_location_hints"][0]
    np.testing.assert_allclose(location["measured_centroid_world_m"], detections[0]["points"].mean(axis=0))
    assert "Robot world +Y" in context["geometric_location_rule"]
    assert "identity" not in location


@pytest.mark.parametrize("case", ["different_id", "different_pixels", "missing_reference",
                                 "missing_reference_id", "missing_reference_view"])
def test_same_capture_claim_is_omitted_without_both_provenance_and_pixel_equality(case):
    capture, detections, entities, reference = same_capture_inputs()
    if case == "different_id":
        reference["observation_id"] = "older-frame"
    elif case == "different_pixels":
        reference["views"]["2"]["rgb"][0, 0, 0] ^= 1
    elif case == "missing_reference":
        reference = None
    elif case == "missing_reference_id":
        reference.pop("observation_id")
    else:
        reference["views"] = {}
    calls = []
    run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    assert "same_capture_reference" not in context
    assert "same_capture_box_correspondence_proposals" not in context
    assert "identical frozen images" not in calls[0]["questions"][0]["instructions"]


def test_hints_are_scoped_to_verified_cameras_and_matching_reference_box_camera():
    capture, detections, entities, reference = same_capture_inputs()
    capture["views"]["1"] = deepcopy(capture["views"]["2"])
    reference["views"]["1"] = deepcopy(capture["views"]["1"])
    reference["views"]["1"]["rgb"][0, 0, 0] ^= 1
    detections.append({**deepcopy(detections[0]), "camera_id": "1"})
    reference["entities"]["other-camera"] = {
        **deepcopy(entities["pink"]), "id": "other-camera", "reference_camera_id": "1"}
    calls = []
    run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    assert context["same_capture_reference"]["verified_camera_ids"] == ["2"]
    proposals = context["same_capture_box_correspondence_proposals"]
    assert len(proposals) == 1
    assert [item["reference_entity_id"] for item in proposals[0]["same_camera_reference_box_overlaps"]] == ["pink"]
    assert "identical frozen images" not in calls[0]["questions"][1]["instructions"]


def test_reference_box_in_another_camera_does_not_produce_a_correspondence_proposal():
    capture, detections, entities, reference = same_capture_inputs()
    reference["entities"]["pink"]["reference_camera_id"] = "1"
    calls = []
    run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    assert context["same_capture_box_correspondence_proposals"] == []
    assert "identical frozen images" not in calls[0]["questions"][0]["instructions"]


@pytest.mark.parametrize("value,confidence", [("unknown", .99), ("pink", .79)])
def test_ambiguous_full_overlap_never_overrides_unknown_or_low_confidence(value, confidence):
    capture, detections, entities, reference = same_capture_inputs()
    entities["twin"] = {**deepcopy(entities["pink"]), "id": "twin"}
    reference["entities"]["twin"] = deepcopy(entities["twin"])
    calls = []
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        run(calls, capture, detections, entities, reference, value=value, confidence=confidence)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    overlaps = context["same_capture_box_correspondence_proposals"][0]["same_camera_reference_box_overlaps"]
    assert [item["box_iou"] for item in overlaps] == [1., 1.]


def test_legacy_same_capture_does_not_receive_demo_grounding():
    capture, detections, entities, reference = same_capture_inputs()
    calls = []
    run(calls, capture, detections, entities, reference, legacy=True)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    assert "same_capture_reference" not in context
    assert "same_capture_box_correspondence_proposals" not in context
