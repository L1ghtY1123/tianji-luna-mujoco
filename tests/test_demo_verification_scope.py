"""Transfer evidence scopes identity to pending releases and rejects SAM false positives."""
import json
from copy import deepcopy

import numpy as np
import pytest
from test_demo_partial_identity import clients, inputs
from test_perception import capture_fixture
from test_visual_inventory import client_with_answers, fixture

from luna_mujoco.demo_perception import DemoPerception, DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError


def verification_inputs():
    capture, detections, entities, reference = inputs(("2", "3"))
    detections[1].update(prompt="blue box", requested_color="blue")
    entities["bin"] = {"id": "bin", "kind": "bin", "color": "blue", "roles": ["collision", "container"]}
    progress = {"pink": {"status": "selected", "destination_id": "table", "release_feedback_id": "release-1"}}
    return capture, detections, entities, reference, progress


def test_verified_target_union_does_not_require_uncertain_source_bin_geometry():
    capture, detections, entities, reference, progress = verification_inputs()
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("bin", .67)], extra="none"))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
                               reference=reference, progress=progress, purpose="verify_transfer")
    assert result["progress"]["pink"]["status"] == "at_destination"
    assert result["missing_geometry_ids"] == ["bin"]
    assert result["unassociated_candidates"][0]["camera_id"] == "3"
    assert result["unassociated_candidates"][0]["geometry_policy"] == "discard_from_inventory_map"


@pytest.mark.parametrize("purpose", ["initialize_inventory", "associate_inventory"])
def test_nonverification_still_requires_bin_identity(purpose):
    capture, detections, entities, reference, progress = verification_inputs()
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("bin", .67)]))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        inventory.resolve(capture, detections, instruction="move all", entities=entities,
                          reference=reference, progress=progress, purpose=purpose)


@pytest.mark.parametrize("case", ["uncertain_target", "second_pending_release", "no_release_context"])
def test_transfer_does_not_infer_or_omit_pending_verification_targets(case):
    capture, detections, entities, reference, progress = verification_inputs()
    identities = [("pink", .99), ("bin", .67)]
    if case == "uncertain_target":
        identities = [("pink", .67), ("bin", .99)]
    elif case == "second_pending_release":
        entities["second"] = {"kind": "object", "color": "pink", "roles": ["collision"]}
        progress["second"] = {"status": "selected", "release_feedback_id": "release-2"}
    else:
        progress["pink"].pop("release_feedback_id")
    inventory = DemoVisualInventory(api_key="test", client=clients(identities))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        inventory.resolve(capture, detections, instruction="move all", entities=entities,
                          reference=reference, progress=progress, purpose="verify_transfer")


def test_missing_pending_target_stops_even_without_an_uncertain_candidate():
    capture, detections, entities, reference, progress = verification_inputs()
    inventory = DemoVisualInventory(api_key="test", client=clients([("bin", .99)]))
    with pytest.raises(VisualInventoryError, match="visual_verification_target_unconfirmed"):
        inventory.resolve(capture, [detections[1]], instruction="move all", entities=entities,
                          reference=reference, progress=progress, purpose="verify_transfer")


def test_completed_and_unreleased_targets_do_not_expand_current_verification_scope():
    capture, detections, entities, reference, progress = verification_inputs()
    for identifier in ["completed", "unreleased"]:
        entities[identifier] = {"kind": "object", "color": "pink", "roles": ["collision"]}
    progress["completed"] = {"status": "completed", "release_feedback_id": "old-release"}
    progress["unreleased"] = {"status": "pending"}
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("bin", .67)]))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
                               reference=reference, progress=progress, purpose="verify_transfer")
    assert set(result["missing_geometry_ids"]) == {"bin", "completed", "unreleased"}


def test_verification_perception_discards_untrusted_bin_and_keeps_mapping_unresolved():
    capture, masks = capture_fixture()
    for camera, view in capture["views"].items():
        view["rgb"][0, 0] = [int(camera), 0, 0]
    entities = {
        "pink": {"kind": "cube", "color": "pink", "roles": ["transport", "collision"],
                 "segmentation_query": {"kind": "cube", "color": "pink", "prompt": "pink cup"}},
        "bin": {"kind": "bin", "color": "blue", "roles": ["container", "collision"],
                "segmentation_query": {"kind": "bin", "color": "blue", "prompt": "blue box"}},
    }
    entities["table"] = {"kind": "table", "bbox": [0, 0, 1, 1], "reference_camera_id": "3"}
    reference = {"observation_id": capture["observation_id"], "entities": entities, "views": {},
                 "demo_task": {"source_bin_id": "bin", "source_object_ids": ["pink"], "destination_table_id": "table"},
                 "demo_program": {"version": 1, "steps": [{"type": "stop", "reason": "fixture"}]}}

    class Detector:
        def segment(self, rgb, text, timestamp_us):
            selected = (rgb[0, 0, 0] == 2 and text == "pink cup"
                        or rgb[0, 0, 0] == 3 and text == "blue box")
            return [{"mask": masks[0], "score": .98}] if selected else []

    engine = DemoPerception(Detector(), robot_surface_classifier=lambda *args: {"status": "measured", "masks": {}},
                           visual_inventory=DemoVisualInventory(
                               api_key="test", client=clients([("bin", .67), ("pink", .99)], extra="none")))
    result = engine.observe(capture, instruction="move all", source_hint={
        "scene_entities": entities, "visual_reference": reference,
        "visual_progress": {"pink": {"status": "selected", "release_feedback_id": "release-1"}},
        "visual_decision": {"source": "responses_program", "purpose": "verify_transfer"}})
    assert [item["id"] for item in result["objects"]] == ["pink"]
    assert result["mapping_unresolved"][0]["previous_id"] == "bin"
    assert result["views"]["3"]["instance_masks"] == {}
    assert not np.any(result["views"]["3"]["depth_m"])


def test_rejected_pink_proposal_is_unassigned_and_not_evidence_of_an_extra_workpiece():
    capture, detections, entities, reference = inputs(("2", "1"))
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("reject", .91)], extra="none"))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
                               reference=reference, purpose="verify_transfer",
                               progress={"pink": {"release_feedback_id": "release-1"}})
    assert len(result["groups"]) == 1
    assert [entry["camera_id"] for entry in result["groups"][0]] == ["2"]
    assert result["demo_inventory_check"]["status"] == "none"
    assert result["evidence"]["answers"][1]["choice"] == "reject"


def test_explicit_confident_extra_object_stops_with_the_actual_answer_evidence():
    capture, detections, entities, reference = inputs(("2",))
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99)], extra="extra"))
    with pytest.raises(VisualInventoryError, match="demo_unlisted_pink_object") as caught:
        inventory.resolve(capture, detections, instruction="move one", entities=entities,
                          reference=reference, purpose="verify_transfer")
    assert caught.value.evidence["answers"][-1]["name"] == "demo_unlisted_pink"
    assert caught.value.evidence["answers"][-1]["choice"] == "extra"
    assert caught.value.evidence["answers"][-1]["confidence"] == .99


def test_projected_region_duplicates_are_removed_but_distinct_slots_and_geometry_remain():
    capture, _ = fixture()
    capture["views"]["3"].update(T_world_camera=np.eye(4),
                                intrinsics={"fx": 10., "fy": 10., "cx": 16., "cy": 12.})
    first = {"slot_id": 1, "center_world_xy_m": [0., 0.],
             "polygon_world_xy_m": [[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]]}
    second = {**deepcopy(first), "slot_id": 2}
    distinct_geometry = {**deepcopy(first), "center_world_xy_m": [.1, 0.]}
    region = {"region_id": "region:table", "layout_revision": "revision",
              "layout": {"floor_z_m": 1., "cells": [first, second, distinct_geometry]}}
    assigned = {**first, "floor_z_m": 1., "layout_revision": "revision"}
    calls = []
    inventory = DemoVisualInventory(api_key="test", client=client_with_answers(
        {"visibility_pink": "visible", "progress_pink": "at_destination", "demo_unlisted_pink": "none"}, calls))
    inventory.resolve(capture, [], instruction="move all", entities={"pink": {"kind": "object", "color": "pink"}},
                      placement_regions=[region], progress={"pink": {"destination_id": "table",
                          "region_id": "region:table", "slot_id": 1, "assigned_slot": assigned}})
    annotation = next(json.loads(part["text"]) for part in calls[0]["input"][0]["content"]
                      if part["type"] == "input_text" and '"image_role": "current_annotations"' in part["text"])
    assert [item["slot_id"] for item in annotation["regions"]] == [1, 2, 1]
    question = next(item for item in calls[0]["questions"] if item["name"] == "progress_pink")
    assert "support-plane" in question["instructions"]
    assert "base/contact" in question["instructions"]
    assert "not its entire silhouette or SAM3 box" in question["instructions"]
    assert "millimetre TCP accuracy" not in question["instructions"]
