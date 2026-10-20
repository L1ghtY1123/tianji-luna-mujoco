import copy
import json
from types import SimpleNamespace

import pytest
from test_decisions import arm_evaluations

from jev_mujoco import decisions
from jev_mujoco.decisions import DecisionError, Decisions
from jev_mujoco.skills import Registry


def install_answer(monkeypatch, answer, confidence, requests):
    class Agent:
        def __init__(self, *args, **kwargs):
            self.output_type = kwargs["output_type"]

        async def run(self, state):
            requests.append({"state": json.loads(state), "schema": self.output_type.model_json_schema()})
            return SimpleNamespace(
                output=SimpleNamespace(model_dump=lambda: answer),
                response=SimpleNamespace(provider_details={"confidence": confidence}),
                usage=SimpleNamespace(input_tokens=10, output_tokens=0))

    monkeypatch.setattr(decisions, "Agent", Agent)


async def test_source_attributes_bind_the_source_role_without_changing_the_composite_task(monkeypatch):
    requests, records = [], []
    install_answer(monkeypatch, {"kind": "cube", "color": "red", "position": "left"},
                   {"kind": 1., "color": 1., "position": 1.}, requests)
    result = await Decisions(None, lambda *args: records.append(args[2])).bind_attributes(
        "抓取左边的红色方块到右边的蓝色盒子里", task="grasp_place")
    assert result == {"kind": "cube", "color": "red", "position": "left"}
    assert requests[0]["state"] == {
        "instruction": "抓取左边的红色方块到右边的蓝色盒子里",
        "task": "grasp_place", "selection_role": "source"}
    assert records[0]["confidence_policy"] == "identity_threshold"


@pytest.mark.parametrize("task", ["grasp", "grasp_place", "place"])
async def test_task_choice_keeps_all_task_types(monkeypatch, task):
    requests, records = [], []
    install_answer(monkeypatch, {"task": task}, {"task": 1.}, requests)
    assert await Decisions(None, lambda *args: records.append(args[2])).bind_task("用户指令") == task
    assert set(records[0]["questions"]["task"]["criteria"]) == {
        "grasp", "grasp_place", "place", "unsupported"}


async def test_destination_attributes_have_an_independent_container_role(monkeypatch):
    requests, records = [], []
    install_answer(monkeypatch, {"kind": "bin", "color": "blue", "position": "right"},
                   {"kind": 1., "color": 1., "position": 1.}, requests)
    result = await Decisions(None, lambda *args: records.append(args[2])).bind_destination_attributes(
        "抓取左边的红色方块到右边的蓝色盒子里")
    assert result == {"kind": "bin", "color": "blue", "position": "right"}
    assert requests[0]["state"]["selection_role"] == "destination"
    assert set(records[0]["questions"]["kind"]["criteria"]) == {"bin", "any", "unsupported"}
    assert records[0]["confidence_policy"] == "identity_threshold"


async def test_destination_unsupported_attributes_fail_instead_of_widening_the_match(monkeypatch):
    install_answer(monkeypatch, {"kind": "unsupported", "color": "blue", "position": "any"},
                   {"kind": 1., "color": 1., "position": 1.}, [])
    with pytest.raises(DecisionError, match="unsupported_destination_attributes"):
        await Decisions(None, lambda *args: None).bind_destination_attributes("放到蓝色托盘上")


def observed_instances():
    return [{"id": "red_cube", "kind": "cube", "color": "red", "position": [.5, .1, .62]},
            {"id": "solid_blue_box", "kind": "box", "color": "blue", "position": [.6, 0., .62]},
            {"id": "bin_blue_left", "kind": "bin", "color": "blue", "position": [.6, .2, .65]},
            {"id": "bin_blue_right", "kind": "bin", "color": "blue", "position": [.6, -.2, .65]}]


@pytest.mark.parametrize("answer", ["bin_blue_right", "ambiguous", "missing"])
async def test_destination_choices_include_only_detected_bins_with_explicit_identity_failures(monkeypatch, answer):
    requests, records = [], []
    install_answer(monkeypatch, {"destination": answer}, {"destination": 1.}, requests)
    state = {"destination_attributes": {"kind": "bin", "color": "blue", "position": "any"}}
    assert await Decisions(None, lambda *args: records.append(args[2])).choose_destination(
        state, observed_instances()) == answer
    assert set(records[0]["questions"]["destination"]["criteria"]) == {
        "bin_blue_left", "bin_blue_right", "missing", "ambiguous"}
    assert [obj["id"] for obj in requests[0]["state"]["observed_destinations"]] == [
        "bin_blue_left", "bin_blue_right"]


async def test_non_container_model_answer_is_rejected(monkeypatch):
    install_answer(monkeypatch, {"destination": "solid_blue_box"}, {"destination": 1.}, [])
    with pytest.raises(DecisionError, match="invalid_choice:destination"):
        await Decisions(None, lambda *args: None).choose_destination({}, observed_instances())


async def test_destination_identity_keeps_the_confidence_gate(monkeypatch):
    install_answer(monkeypatch, {"destination": "bin_blue_left"}, {"destination": .59}, [])
    with pytest.raises(DecisionError, match="low_confidence:destination"):
        await Decisions(None, lambda *args: None).choose_destination({}, observed_instances())


async def test_destination_has_no_invented_option_when_no_bin_is_observed(monkeypatch):
    requests, records = [], []
    install_answer(monkeypatch, {"destination": "missing"}, {"destination": 1.}, requests)
    assert await Decisions(None, lambda *args: records.append(args[2])).choose_destination(
        {}, observed_instances()[:2]) == "missing"
    assert set(records[0]["questions"]["destination"]["criteria"]) == {"missing", "ambiguous"}
    assert requests[0]["state"]["observed_destinations"] == []


@pytest.mark.parametrize("method, answer_key, identifier", [
    ("choose_target", "target", "red_cube"),
    ("choose_destination", "destination", "bin_blue_left"),
])
async def test_instance_decisions_send_public_estimates_without_raw_or_privileged_fields(
        monkeypatch, method, answer_key, identifier):
    instances = observed_instances()
    for obj in instances:
        obj.update(points_world_m=[[999., 999., 999.]], rgb_feature=[999.], mask="private pixels",
                   true_pose="privileged sentinel", mujoco_body_id=999)
    original = copy.deepcopy(instances)
    requests, records = [], []
    install_answer(monkeypatch, {answer_key: identifier}, {answer_key: 1.}, requests)
    engine = Decisions(None, lambda *args: records.append(args[2]))
    assert await getattr(engine, method)({"facts": {"observed": True}}, instances) == identifier
    assert "999" not in json.dumps(requests, ensure_ascii=False)
    assert "private pixels" not in json.dumps(requests)
    assert "privileged sentinel" not in json.dumps(records)
    assert instances == original
    field = 'observed_objects' if method == 'choose_target' else 'observed_destinations'
    assert next(obj for obj in requests[0]['state'][field] if obj['id'] == identifier) == next(
        obj for obj in observed_instances() if obj['id'] == identifier)
    assert field in records[0]['questions'][answer_key]['criteria'][identifier]


async def test_place_skill_choice_retains_every_skill_including_grasp(monkeypatch):
    requests, records = [], []
    install_answer(monkeypatch, {"skill": "plan_place"}, {"skill": .2}, requests)
    registry = Registry()
    assert await Decisions(None, lambda *args: records.append(args[2])).choose_skill(
        {"task": "place", "facts": {"grasp_held": True}}, registry) == "plan_place"
    assert set(records[0]["questions"]["skill"]["criteria"]) == (set(registry.contracts) - {"observe"}) | {"stop"}
    assert requests[0]["state"]["skill_catalog"] == registry.describe()


@pytest.mark.parametrize("skill_selection", [False, True])
def test_combined_selection_exposes_deferred_place_without_invented_ik_evidence(skill_selection):
    evaluations = arm_evaluations(2)
    for info in evaluations.values():
        info.update(feasibility_scope="grasp_ik_only", destination_id="blue_bin",
                    place_planning_stage="after_grasp_attachment")
    original = copy.deepcopy(evaluations)
    supplied = decisions.compact_decision_state({"arm_evaluations": evaluations},
                                               skill_selection=skill_selection)
    for info in supplied["arm_evaluations"].values():
        assert info["valid_candidate_count"] == 2
        assert info["feasibility_scope"] == "grasp_ik_only"
        assert info["destination_id"] == "blue_bin"
        assert info["place_planning_stage"] == "after_grasp_attachment"
        assert info["complete_path_status"] == "unknown"
        assert not any(key.startswith("place_") for key in info["ik_error_ranges"])
    assert evaluations == original
    assert decisions.compact_decision_state(supplied, skill_selection=skill_selection) == supplied


@pytest.mark.parametrize("skill_selection", [False, True])
def test_paired_ik_projection_keeps_same_candidate_place_evidence_without_claiming_a_complete_path(
        skill_selection):
    evaluations = arm_evaluations(2)
    for info in evaluations.values():
        info.update(feasibility_scope="paired_ik_only", destination_id="blue_bin")
        for index, candidate in enumerate(info["valid_candidates"]):
            candidate["destination_ik"] = {
                "success": True, "joints_rad": [.3 + index * .1] + [0.] * 6,
                "joint_names": candidate["ik"]["joint_names"],
                "position_error_m": (1 + 2 * index) * .00001,
                "rotation_error_rad": (1 + 2 * index) * .001}
            candidate["T_world_place_tcp"] = [[.123456789] * 4 for _ in range(4)]
    original = copy.deepcopy(evaluations)
    supplied = decisions.compact_decision_state({"arm_evaluations": evaluations},
                                               skill_selection=skill_selection)
    for arm in ("A", "B"):
        info = supplied["arm_evaluations"][arm]
        assert info["feasibility_scope"] == "paired_ik_only"
        assert info["destination_id"] == "blue_bin"
        assert info["complete_path_status"] == "unknown"
        assert info["ik_error_ranges"]["place_position_error_m"] == pytest.approx([.00001, .00003])
        assert info["ik_error_ranges"]["place_rotation_error_rad"] == pytest.approx([.001, .003])
        if skill_selection:
            assert info["candidate_indices"] == [0, 1]
            assert info["candidate_ranges"]["place_joint_distance_from_grasp_rad"] == [.3, .399]
        else:
            column = info["candidate_columns"].index("place_joint_distance_from_grasp_rad")
            assert [row[column] for row in info["candidates"]] == [.3, .399]
    assert "T_world_place_tcp" not in json.dumps(supplied)
    assert evaluations == original
    assert decisions.compact_decision_state(supplied, skill_selection=skill_selection) == supplied
