"""Only complete, same-capture Decisions evidence resolves Responses uncertainty."""
from copy import deepcopy

import pytest
from test_demo_count_inventory import UnknownContentsVision
from test_demo_partial_identity import clients, inputs
from test_demo_runner import TEST_SETTINGS, DemoRobot
from test_demo_runner import TestPolicyController as RobotOperations

from luna_mujoco.demo_perception import DemoVisualInventory


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
        "source_object_ids": ["pink"], "remaining_source_ids": ["pink"], "complete": True}
    assert result["demo_inventory_check"]["request_hash"] == coverage["request_hash"]
    if supplement:
        assert result["unassociated_candidates"][0]["status"] == "unknown"


@pytest.mark.parametrize("completeness", ["unknown", "confirmed"])
@pytest.mark.parametrize("mutation", [
    "missing_certificate", "missing_identity", "stale_capture", "stale_request",
    "wrong_source", "low_threshold", "stale_none_request",
])
async def test_binding_requires_current_identity_certificate(tmp_path, completeness, mutation):
    class MutatedBind(RobotOperations):
        def _bind_inventory(self, context):
            observation = deepcopy(context["observation"])
            visual = observation["visual_inventory"]
            coverage = visual["identity_coverage"]
            if mutation == "missing_certificate":
                visual.pop("identity_coverage")
            elif mutation == "missing_identity":
                coverage["accepted_entity_ids"] = ["pink-1", "pink-2"]
            elif mutation == "stale_capture":
                coverage["observation_id"] = "old-frame"
            elif mutation == "stale_request":
                coverage["request_hash"] = "other-decisions"
            elif mutation == "wrong_source":
                coverage["source"] = "openai_responses"
            elif mutation == "low_threshold":
                coverage["minimum_confidence"] = .79
            elif mutation == "stale_none_request":
                observation["demo_inventory_check"]["request_hash"] = "other-decisions"
            return super()._bind_inventory(context | {"observation": observation})

    algorithms = UnknownContentsVision(requested=None, completeness=completeness)
    runner = MutatedBind(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"].endswith(":source_inventory_unknown")
    assert runner.state["failed_step"]["skill"] == "confirm_inventory"
    assert algorithms.target_order == []
    assert algorithms.map_order == []


@pytest.mark.parametrize("completeness", ["unknown", "confirmed"])
async def test_decisions_resolution_preserves_original_responses_claim(tmp_path, completeness):
    algorithms = UnknownContentsVision(requested=None, completeness=completeness)
    runner = RobotOperations(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    resolution = runner.state["source_inventory_resolution"]
    assert resolution["responses_status"] == completeness
    assert runner.task_spec.source_inventory_complete == completeness
    assert resolution["evidence"]["source"] == "openai_decisions"
    assert resolution["evidence"]["request_hash"] == "task-judgment"
    assert any(item['kind'] == 'task_binding' for item in runner.events)
    assert algorithms.discoveries == 1
