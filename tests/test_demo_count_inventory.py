"""Visible inventory evidence gates both all and count tasks without inventing hidden contents."""
from copy import deepcopy

import pytest
from test_demo_perception import demo_discovery, discovery_output
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision
from test_perception import capture_fixture

from luna_mujoco.demo_runner import DemoRunner
from luna_mujoco.visual_api import VisualInventoryError


def test_count_discovery_preserves_unknown_completeness_and_all_source_entities():
    capture, _ = capture_fixture()
    body = discovery_output(2)
    body["task"]["source_inventory_complete"] = "unknown"
    original = deepcopy(body)
    result = demo_discovery(body, []).discover(capture, instruction="move two pink objects")
    assert result["demo_task"]["source_inventory_complete"] == "unknown"
    assert len(result["demo_task"]["transport_ids"]) == 2
    assert len(result["demo_task"]["source_object_ids"]) == 3
    unselected = set(result["demo_task"]["source_object_ids"]) - set(result["demo_task"]["transport_ids"])
    assert all("collision" in result["entities"][identifier]["roles"] for identifier in unselected)
    assert body == original


@pytest.mark.parametrize("case,reason", [
    ("insufficient_count", "requested_count_exceeds_confirmed_inventory"),
    ("ambiguous_member", "source_inventory_unknown"),
])
def test_count_exception_does_not_repair_other_discovery_unknowns(case, reason):
    capture, _ = capture_fixture()
    body = discovery_output(4 if case == "insufficient_count" else 2)
    body["task"]["source_inventory_complete"] = "unknown"
    if case == "ambiguous_member":
        body["entities"][3]["membership"] = "unknown"
    with pytest.raises(VisualInventoryError, match=reason):
        demo_discovery(body, []).discover(capture, instruction="move pink objects")


class UnknownContentsVision(DemoVision):
    def __init__(self, *, requested=2, bad_check=None, bad_at=0, completeness="unknown"):
        super().__init__(3, requested)
        self.bad_check, self.bad_at = bad_check, bad_at
        self.completeness = completeness
        self.source_checks = 0

    def observe(self, capture, **kwargs):
        result = super().observe(capture, **kwargs)
        result["demo_task"]["source_inventory_complete"] = self.completeness
        result["visual_reference"]["demo_task"]["source_inventory_complete"] = self.completeness
        result["demo_inventory_check"].update(source="openai_decisions", confidence=.99)
        if kwargs.get("checkpoint") == "initial":
            if self.source_checks == self.bad_at and self.bad_check is not None:
                result["demo_inventory_check"].update(self.bad_check)
            self.source_checks += 1
        return result


async def run_unknown_count(tmp_path, **options):
    algorithms = UnknownContentsVision(**options)
    runner = DemoRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move two pink objects")
    await runner.worker
    return runner, algorithms


async def test_count_moves_only_frozen_visible_subset_without_promoting_completeness(tmp_path):
    runner, algorithms = await run_unknown_count(tmp_path)
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    assert runner.state["demo_task"]["source_inventory_complete"] == "unknown"
    assert runner.task_progress.completed_ids == ["pink-1", "pink-2"]
    assert algorithms.target_order == ["pink-1", "pink-2"]
    assert "pink-3" in runner.task_progress.scene_entities
    assert "collision" in runner.task_progress.scene_entities["pink-3"]["roles"]
    assert runner.task_progress.placement_request("demo-destination")["layout"]["target_count"] == 2
    assert algorithms.discoveries == 1


@pytest.mark.parametrize("bad_check", [
    {"status": "unknown"}, {"status": "extra"}, {"confidence": .79},
    {"confidence": True}, {"confidence": None}, {"observation_id": "older-capture"},
    {"source": "untrusted"},
])
async def test_count_requires_fresh_confident_visible_unlisted_none_before_motion(tmp_path, bad_check):
    runner, algorithms = await run_unknown_count(tmp_path, bad_check=bad_check)
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"] in {"source_inventory_unknown", "unlisted_source_object"}
    assert algorithms.target_order == []
    assert algorithms.map_order == []


async def test_count_rechecks_visible_scope_before_next_target(tmp_path):
    runner, algorithms = await run_unknown_count(tmp_path, bad_check={"status": "unknown"}, bad_at=1)
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"] == "source_inventory_unknown"
    assert algorithms.target_order == ["pink-1"]
    assert runner.task_progress.completed_ids == ["pink-1"]


async def test_all_mode_resolves_unknown_initial_completeness_with_independent_evidence(tmp_path):
    runner, algorithms = await run_unknown_count(tmp_path, requested=None)
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    assert runner.state["demo_task"]["source_inventory_complete"] == "unknown"
    resolution = runner.state["source_inventory_resolution"]
    assert resolution["status"] == "confirmed"
    assert resolution["responses_status"] == "unknown"
    assert resolution["scope"] == "visible_source_inventory"
    assert resolution["left_source_ids"] == ["pink-1", "pink-2", "pink-3"]
    assert algorithms.discoveries == 1
    assert algorithms.target_order == ["pink-1", "pink-2", "pink-3"]


@pytest.mark.parametrize("bad_at", [0, 1])
@pytest.mark.parametrize("completeness", ["confirmed", "unknown"])
@pytest.mark.parametrize("bad_check", [
    {"status": "unknown"}, {"status": "extra"}, {"status": None},
    {"confidence": .79}, {"confidence": True}, {"confidence": None},
    {"observation_id": "older-capture"}, {"source": "untrusted"},
])
async def test_all_requires_fresh_confident_visible_none_at_first_and_between_observations(
        tmp_path, bad_at, bad_check, completeness):
    runner, algorithms = await run_unknown_count(tmp_path, requested=None, completeness=completeness,
                                                 bad_check=bad_check, bad_at=bad_at)
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"] in {"source_inventory_unknown", "unlisted_source_object"}
    assert algorithms.target_order == ([] if bad_at == 0 else ["pink-1"])
    assert runner.task_progress.completed_ids == ([] if bad_at == 0 else ["pink-1"])
    maps = [entry for entry in algorithms.map_order if entry[0] == "map"]
    assert len(maps) == bad_at


@pytest.mark.parametrize("requested", [None, 2])
async def test_binding_rechecks_visible_evidence_even_after_observation_passes(tmp_path, requested):
    class StaleAtBind(DemoRunner):
        def _bind_inventory(self, context):
            observation = deepcopy(context["observation"])
            observation["demo_inventory_check"]["observation_id"] = "previous-frame"
            return super()._bind_inventory(context | {"observation": observation})

    algorithms = UnknownContentsVision(requested=requested, completeness="confirmed")
    runner = StaleAtBind(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move pink objects")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"] == "source_inventory_unknown"
    assert runner.state["failed_step"]["skill"] == "bind_inventory"
    assert algorithms.target_order == []
    assert algorithms.map_order == []


async def test_all_completion_still_rejects_unknown_fresh_final_visible_check(tmp_path):
    class UnknownFinalVision(DemoVision):
        def verify_release(self, *args, **kwargs):
            result = super().verify_release(*args, **kwargs)
            result["demo_inventory_check"]["status"] = "unknown"
            return result

    algorithms = UnknownFinalVision(1)
    runner = DemoRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.task_progress.completed_ids == ["pink-1"]
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"] == "source_inventory_unknown"
