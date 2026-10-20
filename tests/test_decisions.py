import copy
import json
import math
from types import SimpleNamespace

import pytest

from jev_mujoco import decisions
from jev_mujoco.decisions import DecisionError, Decisions
from jev_mujoco.skills import Registry


def install_agent(monkeypatch, output, confidence, captured=None, wire=None):
    class Agent:
        def __init__(self, *args, **kwargs):
            self.output_type = kwargs["output_type"]

        async def run(self, state):
            if wire is not None:
                wire.append(state)
            if captured is not None:
                captured.append({"state": json.loads(state), "schema": self.output_type.model_json_schema()})
            return SimpleNamespace(output=SimpleNamespace(model_dump=lambda: output),
                                   response=SimpleNamespace(provider_details={"confidence": confidence}),
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=0))
    monkeypatch.setattr(decisions, "Agent", Agent)


async def test_two_valid_skill_orders_do_not_become_a_false_failure(monkeypatch):
    install_agent(monkeypatch, {"skill": "generate_grasps"}, {"skill": .38})
    engine = Decisions(None, lambda *args: None)
    assert await engine.choose_skill({"facts": {"observed": True, "target_selected": True}}, Registry()) == "generate_grasps"


@pytest.mark.parametrize("facts", [
    {},
    {"observed": True, "target_selected": True},
    {name: True for name in ("observed", "target_selected", "scene_ready", "grasps_ready",
                            "arms_evaluated", "arm_selected", "plan_ready", "lifted", "verified")},
])
async def test_skill_choice_always_offers_full_catalog_without_precomputed_eligibility(monkeypatch, facts):
    registry = Registry()
    captured, records = [], []
    install_agent(monkeypatch, {"skill": "execute_grasp"}, {"skill": .4}, captured)
    state = {"instruction": "Pick the blue cylinder", "facts": facts,
             "skill_contracts": {"old_hint": {"eligible": True, "unmet": []}}}
    original = copy.deepcopy(state)
    engine = Decisions(None, lambda kind, title, data: records.append(data))
    assert await engine.choose_skill(state, registry) == "execute_grasp"
    assert set(records[0]["questions"]["skill"]["criteria"]) == (set(registry.contracts) - {"observe"}) | {"stop"}
    supplied = captured[0]["state"]
    assert supplied["facts"] == facts
    assert set(supplied["skill_catalog"]) == (set(registry.contracts) - {"observe"})
    assert "skill_documents" not in supplied and "skill_contracts" not in supplied
    for name, contract in registry.contracts.items():
        if name == 'observe':
            continue
        assert supplied["skill_catalog"][name]["requires"] == contract.requires
    assert state == original
    assert decisions.compact_decision_state(supplied, skill_selection=True) == supplied


async def test_new_registered_skill_is_offered_without_changing_decision_code(monkeypatch, tmp_path):
    for name, requires in (("observe_scene", {"observed": False}), ("inspect_contact", {"observed": True})):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "contract.json").write_text(json.dumps({
            "id": name, "description": name, "requires": requires, "provides": [name + "_done"]}),
            encoding="utf-8")
        (folder / "SKILL.md").write_text("Instructions for " + name, encoding="utf-8")
    records = []
    install_agent(monkeypatch, {"skill": "inspect_contact"}, {"skill": .4})
    engine = Decisions(None, lambda kind, title, data: records.append(data))
    assert await engine.choose_skill({"facts": {}}, Registry(tmp_path)) == "inspect_contact"
    assert set(records[0]["questions"]["skill"]["criteria"]) == {"observe_scene", "inspect_contact", "stop"}


async def test_uncertain_target_identity_still_fails(monkeypatch):
    install_agent(monkeypatch, {"target": "black_cube"}, {"target": .38})
    engine = Decisions(None, lambda *args: None)
    with pytest.raises(DecisionError, match="low_confidence:target"):
        await engine.choose_target({}, [{"id": "black_cube"}])


async def test_low_confidence_reports_the_rejected_answer_and_actual_threshold(monkeypatch):
    install_agent(monkeypatch, {'quantity': 'unbounded'}, {'quantity': .48})
    records = []
    engine = Decisions(None, lambda kind, title, data: records.append(data), minimum_confidence=.6)
    with pytest.raises(DecisionError, match='low_confidence:quantity') as caught:
        await engine.ask('理解完整任务', {'instruction': '搬绿色物体'}, {'quantity': {
            'instructions': '原文明确指定的搬运数量', 'criteria': {'unbounded': '未指定', '1': '一个'}}})
    assert caught.value.code == 'low_confidence:quantity'
    assert caught.value.details == {'field': 'quantity', 'selected': 'unbounded',
                                    'confidence': .48, 'minimum_confidence': .6}
    assert records[0]['output'] == {'quantity': 'unbounded'}


async def test_missing_confidence_remains_rejected_and_reported(monkeypatch):
    install_agent(monkeypatch, {'quantity': '1'}, {})
    engine = Decisions(None, lambda *args: None)
    with pytest.raises(DecisionError, match='low_confidence:quantity') as caught:
        await engine.ask('数量', {}, {'quantity': {
            'instructions': '数量', 'criteria': {'unbounded': '未指定', '1': '一个'}}})
    assert caught.value.details['confidence'] is None


@pytest.mark.parametrize("position_confidence", [.58, .6])
async def test_unspecified_position_keeps_identity_confidence_gate(monkeypatch, position_confidence):
    # Recorded failure: correct cylinder/blue/any choices with position confidence .58.
    attributes = {"kind": "cylinder", "color": "blue", "position": "any"}
    records = []
    install_agent(monkeypatch, attributes, {"kind": 1., "color": 1., "position": position_confidence})
    engine = Decisions(None, lambda kind, title, data: records.append(data))
    if position_confidence < .6:
        with pytest.raises(DecisionError, match="low_confidence:position"):
            await engine.bind_attributes("Pick the blue cylinder")
    else:
        assert await engine.bind_attributes("Pick the blue cylinder") == attributes
    assert records[0]["output"] == attributes
    assert records[0]["confidence_policy"] == "identity_threshold"


async def test_explicit_unsupported_position_is_not_treated_as_unspecified(monkeypatch):
    install_agent(monkeypatch, {"kind": "cylinder", "color": "blue", "position": "unsupported"},
                  {"kind": 1., "color": 1., "position": 1.})
    engine = Decisions(None, lambda *args: None)
    with pytest.raises(DecisionError, match="unsupported_object_attributes"):
        await engine.bind_attributes("Pick the blue cylinder in the middle")


def arm_evaluations(count=100):
    return {arm: {
        "arm": arm, "tested_count": count, "start_joints_deg": [0.] * 7,
        "planning_settings": {"unused_solver_option": "x" * 1000},
        "batches": [{"candidate_indices": list(range(100)), "effective_config": {"data": "x" * 1000}}],
        "valid_candidates": [{
            "candidate_index": index, "score": .8 + index / 1000,
            "T_world_tcp": [[.1234567890123456] * 4 for _ in range(4)],
            "ik": {"success": True, "joints_rad": [index / 1000] + [0.] * 6,
                   "joint_names": [f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)],
                   "position_error_m": 6.664001972467304e-8,
                   "rotation_error_rad": 3.2784054582180033e-8},
        } for index in range(count)],
    } for arm in ("A", "B")}


async def test_choose_arm_projects_every_candidate_once_without_mutating_algorithm_evidence(monkeypatch):
    from jev_mujoco.runner import public_summary

    evaluations = arm_evaluations()
    recent = {"kind": "skill_complete", "title": "evaluate_arms",
              "data": {"result": public_summary(evaluations), "duration_s": 12.5}}
    state = {"instruction": "抓取黄色方块", "facts": {"arms_evaluated": True}, "recent_results": [recent]}
    original = copy.deepcopy((state, evaluations))
    captured = []
    install_agent(monkeypatch, {"arm": "B"}, {"arm": .4}, captured)
    assert await Decisions(None, lambda *args: None).choose_arm(state, evaluations) == "B"
    assert len(captured) == 1
    supplied = captured[0]["state"]
    for arm in ("A", "B"):
        evidence = supplied["arm_evaluations"][arm]
        assert evidence["valid_candidate_count"] == 100
        assert evidence["tested_count"] == 100
        assert evidence["max_score"] == pytest.approx(.899)
        assert evidence["min_joint_distance_rad"] == 0
        assert [row[0] for row in evidence["candidates"]] == list(range(100))
        assert evidence["candidates"][99] == [99, .099]
        assert supplied["grasp_scores_by_candidate_index"]["99"] == .899
        assert evidence["ik_error_ranges"]["position_error_m"] == [6.664001972467304e-8] * 2
        assert evidence["ik_error_ranges"]["rotation_error_rad"] == [3.2784054582180033e-8] * 2
    assert supplied["facts"] == state["facts"]
    assert supplied["recent_results"][0]["data"] == {
        "result_reference": "arm_evaluations", "duration_s": 12.5}
    assert "T_world_tcp" not in json.dumps(captured)
    assert len(json.dumps(captured, ensure_ascii=False)) < 32000
    assert (state, evaluations) == original


async def test_choose_arm_sends_all_200_candidates_per_arm_with_compact_json(monkeypatch):
    evaluations = arm_evaluations(200)
    original = copy.deepcopy(evaluations)
    captured, wire = [], []
    install_agent(monkeypatch, {"arm": "A"}, {"arm": .4}, captured, wire)
    state = {"instruction": "抓取 红色方块"}
    assert await Decisions(None, lambda *args: None).choose_arm(state, evaluations) == "A"
    assert len(wire) == 1
    supplied = captured[0]["state"]
    assert supplied["instruction"] == state["instruction"]
    for arm in ("A", "B"):
        evidence = supplied["arm_evaluations"][arm]
        assert evidence["tested_count"] == evidence["valid_candidate_count"] == 200
        assert evidence["candidates"] == [[index, index / 1000] for index in range(200)]
        assert supplied["grasp_scores_by_candidate_index"] == {
            str(index): round(.8 + index / 1000, 6) for index in range(200)}
    assert '": ' not in wire[0]
    assert ", " not in wire[0]
    assert len(wire[0]) < len(json.dumps(supplied, ensure_ascii=False))
    assert evaluations == original


async def test_choose_arm_joint_distance_uses_named_joints_and_still_offers_unreachable_arm(monkeypatch):
    evaluations = arm_evaluations(1)
    evaluations["A"]["start_joints_deg"] = [180.] + [0.] * 6
    ik = evaluations["A"]["valid_candidates"][0]["ik"]
    ik["joint_names"].reverse()
    ik["joints_rad"] = [0.] * 6 + [math.pi]
    evaluations["B"]["valid_candidates"] = []
    recorded = []
    install_agent(monkeypatch, {"arm": "A"}, {"arm": .4})
    await Decisions(None, lambda *args: recorded.append(args[2])).choose_arm({}, evaluations)
    assert set(recorded[0]["questions"]["arm"]["criteria"]) == {"A", "B", "none"}
    assert recorded[0]["state"]["arm_evaluations"]["A"]["candidates"][0][1] == 0
    assert recorded[0]["state"]["arm_evaluations"]["B"]["valid_candidate_count"] == 0


async def test_zero_candidate_arms_are_decided_by_jev_with_both_options_present(monkeypatch):
    evaluations = arm_evaluations(0)
    records = []
    install_agent(monkeypatch, {"arm": "none"}, {"arm": 1.})
    engine = Decisions(None, lambda kind, title, data: records.append(data))
    assert await engine.choose_arm({}, evaluations) == "none"
    assert set(records[0]["questions"]["arm"]["criteria"]) == {"A", "B", "none"}
    assert all(info["valid_candidate_count"] == 0 for info in records[0]["state"]["arm_evaluations"].values())


async def test_attempted_arm_remains_an_option_and_attempt_outcome_is_sent_to_jev(monkeypatch):
    state = {"planning_attempts": [{"arm": "B", "status": "no_solution",
                                  "code": "planning_no_valid_candidate", "duration_s": 10.,
                                  "details": {"failed_phase": "lift"}}]}
    original = copy.deepcopy(state)
    records = []
    install_agent(monkeypatch, {"arm": "A"}, {"arm": .4})
    engine = Decisions(None, lambda kind, title, data: records.append(data))
    assert await engine.choose_arm(state, arm_evaluations(1)) == "A"
    assert set(records[0]["questions"]["arm"]["criteria"]) == {"A", "B", "none"}
    assert records[0]["state"]["planning_attempts"] == original["planning_attempts"]
    assert state == original


@pytest.mark.parametrize("tool_frame", [True, False])
def test_candidate_world_offsets_use_frozen_axes_signs_and_each_arms_tcp_rotation(tool_frame):
    from jev_mujoco.planning_settings import PlanningSettings

    evaluations = arm_evaluations(1)
    evaluations["A"]["valid_candidates"][0]["T_world_tcp"] = [
        [0., 0., 1., .5], [0., 1., 0., .2], [-1., 0., 0., .8], [0., 0., 0., 1.]]
    evaluations["B"]["valid_candidates"][0]["T_world_tcp"] = [
        [1., 0., 0., .5], [0., -1., 0., -.2], [0., 0., -1., .8], [0., 0., 0., 1.]]
    settings = PlanningSettings(approach_axis="x", approach_offset_m=-.2,
                                approach_in_tool_frame=tool_frame,
                                lift_axis="z", lift_offset_m=.1,
                                lift_in_tool_frame=tool_frame).model_dump()
    original = copy.deepcopy((evaluations, settings))
    supplied = decisions.compact_decision_state({"arm_evaluations": evaluations,
                                                "planning_settings": settings})
    expected = ({"A": [[0., 0., .2], [.1, 0., 0.]], "B": [[-.2, 0., 0.], [0., 0., -.1]]}
                if tool_frame else {"A": [[-.2, 0., 0.], [0., 0., .1]],
                                    "B": [[-.2, 0., 0.], [0., 0., .1]]})
    for arm in ("A", "B"):
        info = supplied["arm_evaluations"][arm]
        assert info["candidate_columns"][-2:] == ["approach_world_delta_index", "lift_world_delta_index"]
        assert [supplied["world_delta_vectors_m"][i] for i in info["candidates"][0][-2:]] == expected[arm]
    assert supplied["planning_settings"] == settings
    assert (evaluations, settings) == original
    assert decisions.compact_decision_state(supplied) == supplied


async def test_skill_context_projects_complete_evidence_with_attempts_settings_and_table(monkeypatch):
    from jev_mujoco.planning_settings import PlanningSettings
    from jev_mujoco.runner import public_summary

    evaluations = arm_evaluations(5)
    state = {"facts": {"arms_evaluated": True, "arm_selected": False},
             "planning_settings": PlanningSettings(lift_in_tool_frame=True).model_dump(),
             "planning_attempts": [{"arm": "B", "status": "no_solution", "duration_s": 3.}],
             "simulation": {"table": {"position": [0., 0., .6], "size": [1., 1., .05]}},
             "arm_evaluations": evaluations, "recent_results": [{"title": "evaluate_arms", "data": {
                 "result": public_summary(evaluations), "duration_s": 2.}}]}
    original = copy.deepcopy(state)
    captured = []
    install_agent(monkeypatch, {"skill": "choose_arm"}, {"skill": .4}, captured)
    assert await Decisions(None, lambda *args: None).choose_skill(state, Registry()) == "choose_arm"
    supplied = captured[0]["state"]
    assert supplied["planning_settings"] == public_summary(original["planning_settings"])
    assert supplied["planning_attempts"] == original["planning_attempts"]
    assert supplied["simulation"] == original["simulation"]
    assert supplied["arm_evaluations"]["A"]["candidate_indices"] == list(range(5))
    assert supplied["recent_results"][0]["data"] == {"result_reference": "arm_evaluations", "duration_s": 2.}
    assert "T_world_tcp" not in json.dumps(supplied)
    assert state == original


async def test_skill_context_compacts_evaluation_trace_without_treating_sample_as_all_candidates(monkeypatch):
    from jev_mujoco.runner import public_summary

    evaluations = arm_evaluations()
    state = {"facts": {"arms_evaluated": True}, "recent_results": [{
        "title": "evaluate_arms", "data": {"result": public_summary(evaluations), "duration_s": 12.5}}]}
    captured = []
    install_agent(monkeypatch, {"skill": "choose_arm"}, {"skill": .4}, captured)
    registry = Registry()
    await Decisions(None, lambda *args: None).choose_skill(state, registry)
    supplied = captured[0]["state"]
    assert supplied["facts"] == state["facts"]
    assert supplied["skill_catalog"] == registry.describe()
    evidence = supplied["recent_results"][0]["data"]["result"]["A"]
    assert evidence["valid_candidate_count"] == 100
    assert evidence["candidate_sample_indices"] == [0, 1, 2]
    assert "candidates" not in evidence
    assert "max_score" not in evidence
    assert evidence["range_scope"] == "sample_only"
    assert "T_world_tcp" not in json.dumps(captured)


def test_captured_decision_state_can_be_replayed_without_projecting_evidence_twice():
    from jev_mujoco.runner import public_summary

    state = {"facts": {"arms_evaluated": True}, "recent_results": [{
        "title": "evaluate_arms", "data": {"result": public_summary(arm_evaluations()), "duration_s": 1.}}]}
    projected = decisions.compact_decision_state(state)
    original = copy.deepcopy(projected)
    assert decisions.compact_decision_state(projected) == original
    assert projected == original


def test_recent_only_arm_evidence_keeps_vector_table_and_replayable_skill_summary():
    from jev_mujoco.planning_settings import PlanningSettings
    from jev_mujoco.runner import public_summary

    state = {"planning_settings": PlanningSettings().model_dump(), "recent_results": [{
        "title": "evaluate_arms", "data": {"result": public_summary(arm_evaluations()), "duration_s": 1.}}]}
    projected = decisions.compact_decision_state(state)
    info = projected["recent_results"][0]["data"]["result"]["A"]
    assert all(isinstance(projected["world_delta_vectors_m"][i], list)
               for row in info["candidate_sample"] for i in row[-2:])
    summary = decisions.compact_decision_state(state, skill_selection=True)
    assert decisions.compact_decision_state(summary, skill_selection=True) == summary


async def test_runner_sends_full_candidate_evidence_to_arm_decision_before_trace_sampling(monkeypatch, tmp_path):
    from jev_mujoco.runner import Runner

    simulation = SimpleNamespace(robot_snapshot=lambda: {"generation": 1, "time": 0, "arms": {}})
    runner = Runner(simulation, None, None, tmp_path)
    evaluations = arm_evaluations()
    captured = []
    install_agent(monkeypatch, {"arm": "A"}, {"arm": .4}, captured)
    engine = Decisions(None, lambda *args: None)
    observation = {"source": "rgbd", "observation_id": "rgbd-1", "generation": 1,
                   "time": 0, "objects": [], "arms": {}}
    context = {"evaluations": evaluations, "observation": observation}
    arguments = await runner.select_arguments("choose_arm", context, engine)
    assert await runner.execute("choose_arm", context, arguments) == {"arm": "A"}
    assert len(captured[0]["state"]["arm_evaluations"]["A"]["candidates"]) == 100
    assert len(captured[0]["state"]["arm_evaluations"]["B"]["candidates"]) == 100


async def test_skill_state_summarizes_all_400_candidates_without_filtering_options(monkeypatch):
    from jev_mujoco.planning_settings import PlanningSettings

    evaluations = arm_evaluations(200)
    state = {"facts": {"arms_evaluated": True}, "planning_settings": PlanningSettings().model_dump(),
             "arm_evaluations": evaluations, "planning_attempts": []}
    original = copy.deepcopy(state)
    records = []
    install_agent(monkeypatch, {"skill": "choose_arm"}, {"skill": .4})
    registry = Registry()
    assert await Decisions(None, lambda *args: records.append(args[2])).choose_skill(state, registry) == "choose_arm"
    supplied = records[0]["state"]
    assert set(records[0]["questions"]["skill"]["criteria"]) == (set(registry.contracts) - {"observe"}) | {"stop"}
    assert supplied["skill_catalog"] == registry.describe()
    for arm in ("A", "B"):
        evidence = supplied["arm_evaluations"][arm]
        assert evidence["candidate_indices"] == list(range(200))
        assert evidence["candidate_ranges"]["grasp_score"] == [.8, .999]
        assert evidence["candidate_ranges"]["joint_distance_rad"] == [0., .199]
        assert evidence["range_scope"] == "all_candidates"
        assert "candidates" not in evidence
    assert state == original


def test_planning_failure_projection_retains_all_candidates_without_duplicate_error_text():
    failures = [{"candidate_index": i, "score": .8 + i / 1000, "status": "failed", "error": "no_solution",
                 "phase_success": {"approach": True, "grasp": True, "lift": False}} for i in range(200)]
    attempt = {"arm": "A", "status": "failed", "code": "planning_no_valid_candidate",
               "error": f"no valid three-phase grasp plan for arm A: {failures}",
               "details": {"arm": "A", "failures": failures}, "duration_s": 10.}
    state = {"planning_attempts": [attempt], "facts": {"arm_selected": False}}
    original = copy.deepcopy(state)
    supplied = decisions.compact_decision_state(state)
    projected = supplied["planning_attempts"][0]
    assert projected["error_reference"] == "details.failure_groups"
    assert "error" not in projected
    assert projected["details"]["failure_count"] == 200
    group = projected["details"]["failure_groups"][0]
    assert group["candidate_columns"] == ["candidate_index", "score"]
    assert group["candidates"] == [[i, round(.8 + i / 1000, 6)] for i in range(200)]
    assert group["phase_success"] == failures[0]["phase_success"]
    assert group["status"] == "failed" and group["error"] == "no_solution"
    assert state == original
    assert decisions.compact_decision_state(supplied) == supplied


@pytest.mark.parametrize("status, error_type, expected_code", [
    (400, "max_tokens_exceeded", "jev_input_too_long"),
    (429, "rate_limit_exceeded", None),
    (400, "invalid_request", None),
])
async def test_provider_input_limit_has_specific_diagnostic_without_reclassifying_other_errors(
        monkeypatch, status, error_type, expected_code):
    from pydantic_ai.exceptions import ModelHTTPError

    error = ModelHTTPError(status, "jev-1.13.0", {"detail": {"error_type": error_type}})

    class FailingAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, state):
            raise error

    monkeypatch.setattr(decisions, "Agent", FailingAgent)
    records = []
    engine = Decisions(None, lambda *args: records.append(args[2]))
    with pytest.raises(Exception) as raised:
        await engine.bind_task("Pick the blue cylinder")
    assert getattr(raised.value, "code", None) == expected_code
    if expected_code:
        assert raised.value.details == {"status_code": 400, "model_name": "jev-1.13.0",
                                      "provider_error_type": "max_tokens_exceeded"}
        assert records[0]["provider_error"] == raised.value.details
    else:
        assert raised.value is error


def test_arm_state_shares_displayed_scores_and_failure_scores_without_dropping_candidates():
    evaluations = arm_evaluations(200)
    state = {"arm_evaluations": evaluations, "planning_attempts": []}
    for arm, info in evaluations.items():
        failures = [{"candidate_index": c["candidate_index"], "score": c["score"],
                     "status": "failed", "phase_success": {"lift": False}}
                    for c in info["valid_candidates"]]
        state["planning_attempts"].append({"arm": arm, "status": "failed",
            "code": "planning_no_valid_candidate", "details": {"failures": failures}})
    original = copy.deepcopy(state)
    supplied = decisions.compact_decision_state(state)
    scores = supplied["grasp_scores_by_candidate_index"]
    assert scores == {str(i): round(.8 + i / 1000, 6) for i in range(200)}
    for arm in ("A", "B"):
        info = supplied["arm_evaluations"][arm]
        assert info["candidate_columns"] == ["candidate_index", "joint_distance_rad"]
        assert info["candidates"] == [[i, i / 1000] for i in range(200)]
    for attempt in supplied["planning_attempts"]:
        group = attempt["details"]["failure_groups"][0]
        assert group["candidate_indices_reference"] == f"arm_evaluations.{attempt['arm']}.candidates"
        assert group["score_reference"] == "grasp_scores_by_candidate_index"
        assert "candidates" not in group
    assert decisions.compact_decision_state(supplied) == supplied
    summary = decisions.compact_decision_state(supplied, skill_selection=True)
    assert summary["arm_evaluations"]["A"]["candidate_ranges"]["grasp_score"] == [.8, .999]
    assert summary["planning_attempts"][0]["details"]["failure_groups"][0]["candidate_indices_reference"] == (
        "arm_evaluations.A.candidate_indices")
    assert decisions.compact_decision_state(summary, skill_selection=True) == summary
    assert state == original


def test_inconsistent_candidate_scores_remain_arm_specific_instead_of_being_overwritten():
    evaluations = arm_evaluations(1)
    evaluations["B"]["valid_candidates"][0]["score"] = .1234567890123456
    supplied = decisions.compact_decision_state({"arm_evaluations": evaluations})
    assert "grasp_scores_by_candidate_index" not in supplied
    for arm in ("A", "B"):
        info = supplied["arm_evaluations"][arm]
        score_index = info["candidate_columns"].index("grasp_score")
        assert info["candidates"][0][score_index] == round(evaluations[arm]["valid_candidates"][0]["score"], 6)


def test_failure_references_preserve_subsets_and_do_not_replace_different_scores():
    evaluations = arm_evaluations(3)
    failures = [
        {"candidate_index": 0, "score": .8, "status": "failed", "phase": "lift"},
        {"candidate_index": 1, "score": .801, "status": "failed", "phase": "approach"},
        {"candidate_index": 2, "score": .1234567890123456, "status": "failed", "phase": "approach"},
    ]
    state = {"arm_evaluations": evaluations, "planning_attempts": [
        {"arm": "A", "status": "failed", "code": "planning_no_valid_candidate", "details": {"failures": failures}},
        {"arm": "B", "status": "running", "details": None},
    ]}
    supplied = decisions.compact_decision_state(state)
    groups = supplied["planning_attempts"][0]["details"]["failure_groups"]
    assert groups[0] == {"status": "failed", "phase": "lift", "candidate_indices": [0],
                         "score_reference": "grasp_scores_by_candidate_index"}
    assert groups[1] == {"status": "failed", "phase": "approach",
                         "candidate_columns": ["candidate_index", "score"],
                         "candidates": [[1, .801], [2, .123457]]}
    assert decisions.compact_decision_state(supplied) == supplied


def test_decision_evidence_uses_declared_precision_without_changing_execution_values():
    from jev_mujoco.planning_settings import PlanningSettings

    evaluations = arm_evaluations(1)
    for arm in ("A", "B"):
        candidate = evaluations[arm]["valid_candidates"][0]
        candidate["score"] = .8123456789012345
        candidate["ik"]["joints_rad"] = [.1234567890123456] + [0.] * 6
    settings = PlanningSettings().model_dump()
    state = {"planning_settings": settings, "arm_evaluations": evaluations}
    original = copy.deepcopy(state)
    supplied = decisions.compact_decision_state(state)
    assert supplied["candidate_evidence_precision"] == {
        "grasp_score_decimal_places": 6, "joint_distance_rad_decimal_places": 6,
        "world_delta_m_decimal_places": 6,
        "scope": "JEV selection evidence only; original algorithm and execution data are unchanged.",
    }
    assert supplied["grasp_scores_by_candidate_index"]["0"] == .812346
    assert supplied["arm_evaluations"]["A"]["candidates"][0][1] == .123457
    for vector in supplied["world_delta_vectors_m"]:
        assert all(value == round(value, 6) for value in vector)
    assert state == original
    assert supplied["planning_settings"] == settings
    assert decisions.compact_decision_state(supplied) == supplied


def test_disk_only_full_evaluation_is_projected_before_replaying_as_recent_context():
    from jev_mujoco.runner import public_summary

    evaluations = arm_evaluations(12)
    state = {"recent_results": [{"kind": "skill_complete", "title": "evaluate_arms", "data": {
        "result": public_summary(evaluations), "full_result": evaluations, "duration_s": 3.}}]}
    original = copy.deepcopy(state)
    projected = decisions.compact_decision_state(state)
    data = projected["recent_results"][0]["data"]
    assert "full_result" not in data
    assert [row[0] for row in data["result"]["A"]["candidates"]] == list(range(12))
    assert data["result"]["A"]["range_scope"] == "all_candidates"
    assert "T_world_tcp" not in json.dumps(projected)
    assert decisions.compact_decision_state(projected) == projected
    assert state == original


async def test_input_budget_rejects_before_provider_call_without_truncating_options(monkeypatch):
    sent, records = [], []
    install_agent(monkeypatch, {'skill': 'stop'}, {'skill': 1.}, wire=sent)
    engine = Decisions(None, lambda *args: records.append(args[-1]))
    with pytest.raises(DecisionError, match='budget'):
        await engine.ask('large', {'instruction': 'x'*60000}, {'skill': {
            'instructions': 'choose', 'criteria': {'stop': 'stop'}}})
    assert sent == []
    assert records[-1]['estimated_input_tokens'] > 24000


async def test_input_budget_accepts_compact_payload_without_discarding_evidence(monkeypatch):
    sent, records = [], []
    install_agent(monkeypatch, {'skill': 'stop'}, {'skill': 1.}, wire=sent)
    engine = Decisions(None, lambda *args: records.append(args[-1]))
    state = {'instruction': 'Use all measured positions', 'positions_world_m': [[.1, .2, .3]] * 3200}
    questions = {'skill': {'instructions': 'choose', 'criteria': {'stop': 'stop'}}}
    original = copy.deepcopy(state)

    assert await engine.ask('compact', state, questions) == {'skill': 'stop'}

    assert len(sent) == 1 and json.loads(sent[0]) == original
    assert state == original and records[-1]['questions'] == questions
    assert records[-1]['estimated_input_tokens'] <= records[-1]['input_soft_budget_tokens'] == 24000


async def test_binding_skill_consumes_selected_arguments_without_model(tmp_path):
    from test_runner import Algorithms, Sim

    from jev_mujoco.runner import Runner
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    runner.state.update(generation=1)
    observation = runner.algorithms.observe(runner.simulation.capture_rgbd())
    result = await runner.execute('identify_target', {'observation': observation}, {'target_id': 'black'})
    assert result['target_id'] == runner.state['target_id'] == 'black'


def test_context_compaction_preserves_grouped_destination_and_visual_relationship_evidence():
    instruction = '两个红色放蓝盒，绿色放黄色标记，黑色不动'
    semantics = {'id': 'red', 'inside_id': 'source-bin', 'label': 'red cube'}
    state = {'instruction': instruction,
             'task_progress': {'instruction': instruction, 'current': {'id': 'red', 'color': 'red'},
                               'completed': [{'id': 'red-old', 'color': 'red'}],
                               'deferred': [{'id': 'green', 'reason': 'unreachable'}]},
             'observation': {'objects': [{'id': 'red', 'visual_semantics': semantics}]},
             'visual_scene': {'objects': [semantics], 'relationships': [{'subject_id': 'red', 'relation': 'inside',
                                                                       'object_id': 'source-bin'}]}}
    original = copy.deepcopy(state)
    compact = decisions.compact_decision_state(state, skill_selection=True)
    assert compact['instruction'] == instruction
    assert compact['task_progress']['current']['color'] == 'red'
    assert compact['task_progress']['completed'] == state['task_progress']['completed']
    assert compact['task_progress']['deferred'] == state['task_progress']['deferred']
    assert compact['observation']['objects'][0]['visual_semantics'] == semantics
    assert 'objects' not in compact['visual_scene']
    assert compact['visual_scene']['relationships'] == state['visual_scene']['relationships']
    assert decisions.compact_decision_state(compact, skill_selection=True) == compact
    assert state == original


def test_successful_place_planning_context_keeps_decision_evidence_without_dense_geometry():
    result = {'arm': 'A', 'target_id': 'red', 'destination_id': 'bin', 'scene_token': 'current',
              'phases': {'approach': {'positions_rad': [[.1]*7]*200},
                         'place': {'positions_rad': [[.2]*7]*200}},
              'retreat_requires_replan': False, 'candidate_count': 40,
              'command_checks': {'full_geometry': [[.3]*4]*300},
              'candidate_failures': [{'release_clearance': {'full_geometry': [[.4]*4]*100}}]*30}
    state = {'instruction': '把红色放入盒里', 'facts': {'place_plan_ready': True},
             'recent_results': [{'kind': 'skill_complete', 'title': 'plan_place',
                                 'data': {'result': result, 'duration_s': 12.}}]}
    original = copy.deepcopy(state)
    projected = decisions.compact_decision_state(state, skill_selection=True)
    evidence = projected['recent_results'][0]['data']['result']
    assert evidence['target_id'] == 'red'
    assert evidence['destination_id'] == 'bin'
    assert evidence['retreat_requires_replan'] is False
    assert evidence['phase_names'] == ['approach', 'place']
    assert len(json.dumps(projected)) < 1500
    assert decisions.compact_decision_state(projected, skill_selection=True) == projected
    assert state == original
