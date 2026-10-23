"""Supplementary uncertain proposals never erase a fully identified view union."""
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from test_perception import capture_fixture
from test_visual_inventory import fixture

from luna_mujoco.demo_perception import DemoPerception, DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import VisualInventory


def clients(identities, *, extra="unknown"):
    def respond(request):
        payload = json.loads(request.content)
        answers = []
        for question in payload["questions"]:
            name = question["name"]
            if name.startswith("identity_"):
                value, confidence = identities[int(name.split("_")[1])]
            else:
                value = extra if name == "demo_unlisted_pink" else (
                    "at_destination" if name.startswith("progress_") else "visible")
                confidence = .99
            answers.append({"name": name, "type": "choice", "choice": value, "confidence": confidence})
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers})
    return httpx.Client(transport=httpx.MockTransport(respond))


def inputs(cameras=("2", "1")):
    capture, detection = fixture()
    capture["views"] = {camera: deepcopy(capture["views"]["3"]) for camera in ("1", "2", "3")}
    detections = [{**deepcopy(detection), "camera_id": camera, "prompt": "pink cup",
                   "requested_color": "pink"} for camera in cameras]
    entities = {"pink": {"id": "pink", "kind": "object", "color": "pink", "roles": ["collision"],
                         "visual_membership": "required"}}
    reference = {"demo_task": {"source_object_ids": ["pink"]}, "views": {}, "entities": entities}
    return capture, detections, entities, reference


def test_primary_identity_survives_uncertain_supplement_without_lowering_threshold():
    capture, detections, entities, reference = inputs()
    before = deepcopy(capture)
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)]))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
        reference=reference, purpose="initialize_inventory")
    assert [entry["camera_id"] for entry in result["groups"][0]] == ["2"]
    assert result["missing_geometry_ids"] == []
    assert result["demo_inventory_check"]["status"] == "unknown"
    assert result["unassociated_candidates"] == [{
        "candidate": "candidate-1", "camera_id": "1", "status": "unknown",
        "choice": "pink", "confidence": .64, "minimum_confidence": .8,
        "geometry_policy": "discard_from_inventory_map", "reason": "supplementary_identity_uncertain"}]
    np.testing.assert_array_equal(capture["views"]["1"]["depth_m"], before["views"]["1"]["depth_m"])
    assert result["evidence"]["answers"][1]["confidence"] == .64


@pytest.mark.parametrize("case", ["only_unknown", "missing_bin", "unknown_primary", "no_left_source"])
def test_incomplete_or_primary_uncertainty_remains_a_failure(case):
    cameras = ("2", "1")
    identities = [("pink", .99), ("pink", .64)]
    if case == "only_unknown":
        cameras, identities = ("1",), [("pink", .64)]
    elif case == "unknown_primary":
        cameras, identities = ("1", "2"), [("pink", .99), ("pink", .64)]
    elif case == "no_left_source":
        cameras = ("3", "1")
    capture, detections, entities, reference = inputs(cameras)
    if case == "missing_bin":
        entities["bin"] = {"id": "bin", "kind": "bin", "color": "blue", "roles": ["collision"]}
    inventory = DemoVisualInventory(api_key="test", client=clients(identities))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        inventory.resolve(capture, detections, instruction="move all", entities=entities,
            reference=reference, purpose="initialize_inventory")


def test_unknown_primary_error_keeps_its_own_answer_after_skipping_supplement():
    capture, detections, entities, reference = inputs(("1", "2", "2"))
    inventory = DemoVisualInventory(api_key="test",
        client=clients([("pink", .64), ("pink", .99), ("pink", .61)]))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown") as caught:
        inventory.resolve(capture, detections, instruction="move all", entities=entities,
            reference=reference, purpose="associate_inventory")
    assert caught.value.evidence["candidate"] == "candidate-2"
    assert caught.value.evidence["camera_id"] == "2"
    assert caught.value.evidence["confidence"] == .61


def test_visible_right_bin_without_confident_identity_still_blocks_complete_union():
    # Frozen run 3c5879: left sources are trusted; the right bin is visible but
    # its head identity is .79 and the right wrist misidentifies it at .63.
    capture, detections, entities, reference = inputs(("3", "2", "2", "1"))
    entities.update({
        "source-bin": {"id": "source-bin", "kind": "bin", "color": "blue", "roles": ["collision"]},
        "right-bin": {"id": "right-bin", "kind": "bin", "color": "blue", "roles": ["collision"]},
    })
    inventory = DemoVisualInventory(api_key="test", client=clients([
        ("right-bin", .79), ("pink", .94), ("source-bin", .97), ("source-bin", .63)],
        extra="none"))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown") as caught:
        inventory.resolve(capture, detections, instruction="move all", entities=entities,
            reference=reference, purpose="initialize_inventory")
    assert caught.value.evidence["candidate"] == "candidate-0"
    assert caught.value.evidence["minimum_confidence"] == .8
    assert any(answer["name"] == "visibility_right-bin" and answer["choice"] == "visible"
               and answer["confidence"] == .99 for answer in caught.value.evidence["answers"])


def test_redundant_primary_proposal_is_discarded_only_with_complete_visible_inventory():
    capture, detections, entities, reference = inputs(("2", "2"))
    inventory = DemoVisualInventory(api_key="test",
        client=clients([("pink", .99), ("pink", .61)], extra="none"))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
        reference=reference, purpose="initialize_inventory")
    assert len(result["groups"]) == 1
    assert result["unassociated_candidates"][0]["camera_id"] == "2"
    assert result["unassociated_candidates"][0]["confidence"] == .61
    assert result["demo_inventory_check"]["status"] == "none"


def test_post_place_uses_camera_union_and_keeps_unknown_source_check_separate():
    capture, detections, entities, reference = inputs(("3", "2"))
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)]))
    result = inventory.resolve(capture, detections, instruction="move one", entities=entities,
        reference=reference, purpose="verify_transfer", progress={"pink": {"destination_id": "table"}})
    assert result["progress"]["pink"]["status"] == "at_destination"
    assert result["demo_inventory_check"]["status"] == "unknown"
    assert result["groups"][0][0]["camera_id"] == "3"


def test_association_after_transfer_does_not_require_completed_objects_in_left_camera():
    capture, detections, entities, reference = inputs(("3", "1"))
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)]))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
        reference=reference, purpose="associate_inventory",
        progress={"pink": {"status": "completed", "destination_id": "table"}})
    assert result["groups"][0][0]["camera_id"] == "3"


def test_legacy_inventory_still_rejects_any_uncertain_candidate():
    capture, detections, entities, reference = inputs()
    inventory = VisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)]))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        inventory.resolve(capture, detections, instruction="move all", entities=entities, reference=reference)


def test_perception_discards_uncertain_secondary_depth_from_inventory_map():
    capture, masks = capture_fixture()
    for camera, view in capture["views"].items():
        view["rgb"][0, 0] = [int(camera), 0, 0]
    entities = {"pink": {"id": "pink", "kind": "cube", "color": "pink", "roles": ["collision"],
        "visual_membership": "required", "segmentation_query": {"kind": "cube", "color": "pink", "prompt": "pink cup"}}}
    entities["table"] = {"kind": "table", "bbox": [0, 0, 1, 1], "reference_camera_id": "3"}
    reference = {"observation_id": capture["observation_id"], "entities": entities, "views": {},
        "demo_task": {"source_bin_id": "bin", "source_object_ids": ["pink"], "destination_table_id": "table"},
        "demo_program": {"version": 1, "steps": [{"type": "stop", "reason": "fixture"}]}}
    class Detector:
        def segment(self, rgb, text, timestamp_us):
            return [] if rgb[0, 0, 0] == 3 else [{"mask": masks[0], "score": .98}]
    engine = DemoPerception(Detector(),
        visual_inventory=DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)])),
        robot_surface_classifier=lambda *args: {"status": "measured", "masks": {}})
    result = engine.observe(capture, instruction="move all", source_hint={"scene_entities": entities,
        "visual_reference": reference, "visual_decision": {"source": "responses_program", "purpose": "associate_inventory"}})
    assert result["objects"][0]["coverage"]["camera_ids"] == ["2"]
    assert result["views"]["1"]["instance_masks"] == {}
    assert not np.any(result["views"]["1"]["depth_m"])
    assert not result["views"]["1"]["valid_depth"].any()
    assert result["visual_inventory"]["unassociated_candidates"][0]["camera_id"] == "1"
