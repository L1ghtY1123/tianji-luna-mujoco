"""Only complete, same-capture Decisions evidence resolves Responses uncertainty."""
from copy import deepcopy

import pytest
from test_demo_count_inventory import UnknownContentsVision
from test_demo_partial_identity import clients, inputs
from test_demo_runner import TEST_SETTINGS, DemoRobot

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.demo_runner import DemoRunner


@pytest.mark.parametrize("supplement", [False, True])
def test_identity_coverage_contains_only_trusted_choices_and_original_request(supplement):
    capture, detections, entities, reference = inputs(("2", "3") if supplement else ("2",))
    identities = [("pink", .99)] + ([("pink", .61)] if supplement else [])
    inventory = DemoVisualInventory(api_key="test", client=clients(identities, extra="none"))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
        reference=reference, purpose="initialize_inventory")
    coverage = result["identity_coverage"]
    assert coverage == {
        "source": "openai_decisions", "observation_id": capture["observation_id"],
        "request_hash": result["evidence"]["request_hash"], "purpose": "initialize_inventory",
        "minimum_confidence": .8, "required_entity_ids": ["pink"], "accepted_entity_ids": ["pink"],
        "accepted_left_entity_ids": ["pink"], "source_object_ids": ["pink"], "remaining_source_ids": ["pink"]}
    assert result["demo_inventory_check"]["request_hash"] == coverage["request_hash"]
    if supplement:
        assert result["unassociated_candidates"][0]["status"] == "unknown"


@pytest.mark.parametrize("completeness", ["unknown", "confirmed"])
@pytest.mark.parametrize("mutation", [
    "missing_certificate", "missing_identity", "missing_left", "wrong_required", "wrong_sources",
    "wrong_remaining", "stale_capture", "stale_request", "wrong_source", "low_threshold",
    "later_purpose", "stale_reference", "stale_responses", "missing_responses_hash",
    "stale_none_request", "missing_bin_identity",
])
async def test_all_binding_requires_same_capture_full_identity_evidence(tmp_path, completeness, mutation):
    class MutatedBind(DemoRunner):
        def _bind_inventory(self, context):
            observation = deepcopy(context["observation"])
            visual = observation["visual_inventory"]
            coverage = visual["identity_coverage"]
            if mutation == "missing_certificate":
                visual.pop("identity_coverage")
            elif mutation == "missing_identity":
                coverage["accepted_entity_ids"] = ["pink-1", "pink-2"]
            elif mutation == "missing_left":
                coverage["accepted_left_entity_ids"] = ["pink-1", "pink-2"]
            elif mutation == "wrong_required":
                coverage["required_entity_ids"] = ["pink-1"]
            elif mutation == "wrong_sources":
                coverage["source_object_ids"] = ["pink-1"]
            elif mutation == "wrong_remaining":
                coverage["remaining_source_ids"] = ["pink-1"]
            elif mutation == "stale_capture":
                coverage["observation_id"] = "old-frame"
            elif mutation == "stale_request":
                coverage["request_hash"] = "other-decisions"
            elif mutation == "wrong_source":
                coverage["source"] = "openai_responses"
            elif mutation == "low_threshold":
                coverage["minimum_confidence"] = .79
            elif mutation == "later_purpose":
                coverage["purpose"] = "verify_transfer"
            elif mutation == "stale_reference":
                observation["visual_reference"]["observation_id"] = "old-reference"
            elif mutation == "stale_responses":
                observation["visual_discovery"]["observation_id"] = "old-response"
            elif mutation == "missing_responses_hash":
                observation["visual_discovery"].pop("request_hash")
            elif mutation == "stale_none_request":
                observation["demo_inventory_check"]["request_hash"] = "other-decisions"
            elif mutation == "missing_bin_identity":
                observation["scene_entities"]["right-bin"] = {"kind": "bin"}
                coverage["required_entity_ids"].append("right-bin")
            return super()._bind_inventory(context | {"observation": observation})

    algorithms = UnknownContentsVision(requested=None, completeness=completeness)
    runner = MutatedBind(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"] == "source_inventory_unknown"
    assert runner.state["failed_step"]["skill"] == "bind_inventory"
    assert algorithms.target_order == []
    assert algorithms.map_order == []
    assert "source_inventory_resolution" not in runner.state


@pytest.mark.parametrize("completeness", ["unknown", "confirmed"])
async def test_resolution_is_separate_from_original_responses_and_archived_at_binding(tmp_path, completeness):
    algorithms = UnknownContentsVision(requested=None, completeness=completeness)
    runner = DemoRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    resolution = runner.state["source_inventory_resolution"]
    assert resolution["responses_status"] == completeness
    assert runner.demo_task["source_inventory_complete"] == completeness
    event = next(item for item in runner.events
                 if item["kind"] == "skill_complete" and item["title"] == "bind_inventory")
    assert event["data"]["result"]["source_inventory_resolution"] == resolution
    assert resolution["responses_request_hash"].startswith("responses-")
    assert resolution["decisions_request_hash"].startswith("decisions-")
    assert algorithms.discoveries == 1
