"""Visible inventory evidence gates both all and count tasks without inventing hidden contents."""
from copy import deepcopy

import pytest
from test_demo_perception import demo_discovery, discovery_output
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision
from test_demo_runner import TestPolicyController as RobotOperations
from test_perception import capture_fixture


def test_count_discovery_preserves_unknown_completeness_and_all_source_entities():
    capture, _ = capture_fixture()
    body = discovery_output(2)
    body["task"]["source_inventory_complete"] = "unknown"
    original = deepcopy(body)
    result = demo_discovery(body, []).discover(capture, instruction="move two pink objects")
    assert result["task_spec"]["source_inventory_complete"] == "unknown"
    assert len(result["task_inventory"]["selected_ids"]) == 2
    assert len(result["task_inventory"]["source_object_ids"]) == 3
    unselected = set(result["task_inventory"]["source_object_ids"]) - set(result["task_inventory"]["selected_ids"])
    assert all("collision" in result["entities"][identifier]["roles"] for identifier in unselected)
    assert body == original


@pytest.mark.parametrize("case", ["insufficient_count", "ambiguous_member"])
def test_count_proposal_preserves_constraints_and_ambiguity_for_decisions(case):
    capture, _ = capture_fixture()
    body = discovery_output(4 if case == "insufficient_count" else 2)
    body["task"]["source_inventory_complete"] = "unknown"
    if case == "ambiguous_member":
        body["entities"][3]["membership"] = "unknown"
    result = demo_discovery(body, []).discover(capture, instruction="move pink objects")
    assert result['task_spec']['quantity'] == body['task']['quantity']
    assert result['task_spec']['source_inventory_complete'] == 'unknown'
    assert len(result['task_inventory']['source_object_ids']) == 3
    if case == 'ambiguous_member':
        assert any(entity['visual_membership'] == 'unknown' for entity in result['entities'].values())


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
            if self.source_checks >= self.bad_at and self.bad_check is not None:
                result["demo_inventory_check"].update(self.bad_check)
            self.source_checks += 1
        return result


async def run_unknown_count(tmp_path, **options):
    algorithms = UnknownContentsVision(**options)
    runner = RobotOperations(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move two pink objects")
    await runner.worker
    return runner, algorithms


async def test_count_moves_only_frozen_visible_subset_without_promoting_completeness(tmp_path):
    runner, algorithms = await run_unknown_count(tmp_path)
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    assert runner.state["task_spec"]["source_inventory_complete"] == "unknown"
    assert runner.task_progress.completed_ids == ["pink-1", "pink-2"]
    assert algorithms.target_order == ["pink-1", "pink-2"]
    assert "pink-3" in runner.task_progress.scene_entities
    assert "collision" in runner.task_progress.scene_entities["pink-3"]["roles"]
    assert runner.task_progress.placement_request("demo-destination")["layout"]["method"] == 'shared_work_area'
    assert algorithms.discoveries == 1


@pytest.mark.parametrize("bad_check", [
    {"status": "extra"}, {"status": "unlisted"}, {"confidence": .79},
    {"confidence": True}, {"confidence": None}, {"observation_id": "older-capture"},
    {"source": "untrusted"}, {"request_hash": "stale-request"}, {"request_hash": None},
])
async def test_count_rejects_invalid_or_unlisted_inventory_evidence_before_motion(tmp_path, bad_check):
    runner, algorithms = await run_unknown_count(tmp_path, bad_check=bad_check)
    assert runner.state["status"] == "needs_attention"
    assert failure_reasons(runner) & {"source_inventory_unknown", "unlisted_source_object"}
    assert algorithms.target_order == []
    assert algorithms.map_order == []


@pytest.mark.parametrize("requested", [None, 2])
@pytest.mark.parametrize("bad_at", [0, 1])
@pytest.mark.parametrize("confidence", [.79, .99])
async def test_unknown_closure_preserves_identity_but_cannot_establish_initial_nearest_order(
        tmp_path, requested, bad_at, confidence):
    runner, algorithms = await run_unknown_count(tmp_path, requested=requested,
        bad_check={"status": "unknown", "confidence": confidence}, bad_at=bad_at)
    if bad_at == 0:
        assert runner.state["status"] == "needs_attention"
        assert runner.state["error"].endswith(":inventory_order_unconfirmed")
        assert runner.state["visible_inventory_resolution"]["status"] == "unknown"
        assert runner.state["visible_inventory_resolution"]["confidence"] == confidence
        assert algorithms.target_order == []
        assert algorithms.map_order == []
    else:
        # The initial visible IDs and their nearest order were already frozen.
        # Later enumeration uncertainty does not revoke those identities.
        selected = ["pink-1", "pink-2"] if requested == 2 else ["pink-1", "pink-2", "pink-3"]
        assert runner.state["status"] == "succeeded", runner.state.get("error")
        assert runner.task_progress.completed_ids == selected
        assert algorithms.target_order == selected
        assert runner.state["task_spec"]["source_inventory_complete"] == "unknown"
        if requested == 2:
            assert runner.state["visible_inventory_resolution"]["status"] == "unknown"
            assert runner.state["visible_inventory_resolution"]["confidence"] == confidence


async def test_all_mode_resolves_unknown_initial_completeness_with_independent_evidence(tmp_path):
    runner, algorithms = await run_unknown_count(tmp_path, requested=None)
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    assert runner.state["task_spec"]["source_inventory_complete"] == "unknown"
    resolution = runner.state["source_inventory_resolution"]
    assert resolution["status"] == "confirmed"
    assert resolution["responses_status"] == "unknown"
    assert resolution["scope"] == "task_source_scope"
    assert resolution["accepted_ids"] == ["pink-1", "pink-2", "pink-3"]
    assert algorithms.discoveries == 1
    assert algorithms.target_order == ["pink-1", "pink-2", "pink-3"]


@pytest.mark.parametrize("bad_at", [0, 1])
@pytest.mark.parametrize("completeness", ["confirmed", "unknown"])
@pytest.mark.parametrize("bad_check", [
    {"status": "extra"}, {"status": "unlisted"}, {"status": None},
    {"confidence": .79}, {"confidence": True}, {"confidence": None},
    {"observation_id": "older-capture"}, {"source": "untrusted"},
    {"request_hash": "stale-request"}, {"request_hash": None},
])
async def test_all_rejects_invalid_or_unlisted_evidence_at_first_and_between_observations(
        tmp_path, bad_at, bad_check, completeness):
    runner, algorithms = await run_unknown_count(tmp_path, requested=None, completeness=completeness,
                                                 bad_check=bad_check, bad_at=bad_at)
    assert runner.state["status"] == "needs_attention"
    assert failure_reasons(runner) & {"source_inventory_unknown", "unlisted_source_object"}
    assert algorithms.target_order == ([] if bad_at == 0 else ["pink-1"])
    assert runner.task_progress.completed_ids == ([] if bad_at == 0 else ["pink-1"])
    maps = [entry for entry in algorithms.map_order if entry[0] == "map"]
    assert len(maps) == bad_at


@pytest.mark.parametrize("requested", [None, 2])
async def test_binding_rechecks_visible_evidence_even_after_observation_passes(tmp_path, requested):
    class StaleAtBind(RobotOperations):
        def _bind_inventory(self, context):
            observation = deepcopy(context["observation"])
            observation["demo_inventory_check"]["observation_id"] = "previous-frame"
            return super()._bind_inventory(context | {"observation": observation})

    algorithms = UnknownContentsVision(requested=requested, completeness="confirmed")
    runner = StaleAtBind(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move pink objects")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"].endswith("source_inventory_unknown")
    assert runner.state["failed_step"]["skill"] == "confirm_inventory"
    assert algorithms.target_order == []
    assert algorithms.map_order == []


async def test_all_completion_still_rejects_unknown_fresh_final_visible_check(tmp_path):
    class UnknownFinalVision(DemoVision):
        final_source_status = "unknown"
        def verify_release(self, *args, **kwargs):
            result = super().verify_release(*args, **kwargs)
            result["demo_inventory_check"]["status"] = "unknown"
            return result

    algorithms = UnknownFinalVision(1)
    runner = RobotOperations(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.task_progress.completed_ids == ["pink-1"]
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"].endswith(":task_completion_unconfirmed")


@pytest.mark.parametrize("missing", ["demo_inventory_check", "identity_coverage", "accepted_entity_ids"])
@pytest.mark.parametrize("bad_at", [0, 1])
async def test_missing_identity_or_inventory_evidence_still_stops_selected_targets(tmp_path, missing, bad_at):
    class MissingEvidenceVision(UnknownContentsVision):
        def observe(self, capture, **kwargs):
            index = self.source_checks
            result = super().observe(capture, **kwargs)
            if kwargs.get("checkpoint") == "initial" and index == bad_at:
                if missing == "demo_inventory_check":
                    result.pop(missing)
                elif missing == "identity_coverage":
                    result["visual_inventory"].pop(missing)
                else:
                    result["visual_inventory"]["identity_coverage"].pop(missing)
            return result
    algorithms = MissingEvidenceVision()
    runner = RobotOperations(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move two pink objects")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert "source_inventory_unknown" in failure_reasons(runner)
    assert algorithms.target_order == ([] if bad_at == 0 else ["pink-1"])


async def test_all_with_frozen_ids_and_unknown_intermediate_closure_still_needs_current_final_yes(tmp_path):
    algorithms = UnknownContentsVision(requested=None,
        bad_check={"status": "unknown", "confidence": .79}, bad_at=1)
    algorithms.final_source_status = "unknown"
    runner = RobotOperations(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    final_checks = []
    captures = []
    capture_observation = runner.capture_observation
    async def record_capture(context, **kwargs):
        capture = await capture_observation(context, **kwargs)
        captures.append(capture["observation_id"])
        return capture
    runner.capture_observation = record_capture
    judge = runner.judgments.judge
    def record_judgment(capture, **kwargs):
        result = judge(capture, **kwargs)
        if "source_complete" in result["judgments"]:
            final_checks.append((capture["observation_id"], result["judgments"]["source_complete"]))
        return result
    runner.judgments.judge = record_judgment
    await runner.start("move all pink objects")
    await runner.worker
    assert algorithms.target_order == ["pink-1", "pink-2", "pink-3"]
    assert runner.task_progress.completed_ids == ["pink-1", "pink-2", "pink-3"]
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"].endswith(":task_completion_unconfirmed")
    assert final_checks[-1][1]["status"] == "unknown"
    assert final_checks[-1][0] == captures[-1]
    assert final_checks[-1][0] != final_checks[0][0]
    assert any(event['kind'] == 'observation_reused'
               and event['data']['purpose'] == 'task_completion'
               and event['data']['observation_id'] == captures[-1] for event in runner.events)


def failure_reasons(runner):
    return {runner.state['error'].split(':')[-1]} | {
        event['data'].get('reason') for event in runner.events if event['kind'] == 'policy_node'}
