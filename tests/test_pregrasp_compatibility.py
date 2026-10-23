"""Frozen source clouds expose self-envelope incompatibility before execution."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.demo_gripper_quality import gripper_width_model, predict_gripper_compatibility
from luna_mujoco.geometry import TOOL_PROFILE, grasp_targets

FIXTURES = Path(__file__).parent / "fixtures"


def frozen(case):
    record = json.loads((FIXTURES / "pregrasp_compatibility.json").read_text(encoding="utf-8"))[case]
    record["settings"]["attachment"]["sphere_budget"] = 24
    record["settings"]["trajectory_collision_clearance_m"] = .01
    with np.load(FIXTURES / "pregrasp_compatibility.npz", allow_pickle=False) as arrays:
        target = record["target"]
        for key in ("points_world_m", "completed_geometry_points_world_m"):
            target[key] = arrays[f"{case}_{key}"].copy()
    record["snapshot"]["objects"] = [target]
    return record


@pytest.mark.parametrize("case,expected", [("3ce", [931]), ("5be", [508, 826, 991, 992, 993]),
                                          ("ffb", [971, 973, 974, 975, 976, 978])])
def test_frozen_ik_success_retains_only_point_envelope_compatible_goals(case, expected):
    record = frozen(case)
    algorithms = DemoAlgorithms()
    originals = deepcopy(record["candidates"])
    accepted, filters = [], []
    for candidate in record["candidates"]:
        result, evidence = algorithms._grasp_ik_candidate(
            {**candidate, "T_world_tcp": candidate["nominal_T_world_tcp"]},
            record["snapshot"], record["target"]["id"], "A", record["scene"], record["settings"])
        filters.append(evidence)
        if result is not None:
            accepted.append(result["candidate_index"])
            assert result["source_index"] == candidate["source_index"]
            assert result["score"] == candidate["score"]
            assert result["T_world_tcp"] == candidate["T_world_tcp"]
    assert accepted == expected
    assert record["candidates"] == originals
    for evidence in filters:
        if evidence["reason"] == "achieved_tool_points_toward_robot":
            assert case == "ffb" and evidence["candidate_index"] == 972
            assert "predicted_gripper_compatibility" not in evidence
            continue
        prediction = evidence["predicted_gripper_compatibility"]
        assert prediction["point_count"] == 1955
        assert prediction["padding_m"] == pytest.approx(.006)
        assert prediction["is_measurement"] is False
        assert prediction["cloud_translated"] is False
        assert prediction["continuous_sweep_checked"] is False
        assert prediction["accepted"] == evidence["accepted"]


def prediction(record, candidate, *, points=None, matrix=None, arm="A", padding=.006):
    return predict_gripper_compatibility(
        record["target"]["completed_geometry_points_world_m"] if points is None else points,
        candidate["T_world_tcp"] if matrix is None else matrix,
        arm=arm, uncertainty_m=.005, margin_m=padding - .005)


def test_own_model_both_tool_profiles_keep_z_and_opposite_x_without_double_conversion():
    record = frozen("5be")
    candidate = record["candidates"][1]
    source = np.array(candidate["T_world_tcp"])
    local = (record["target"]["completed_geometry_points_world_m"] - source[:3, 3]) @ source[:3, :3]
    goals = [grasp_targets(np.eye(4)[None], arm)[0] for arm in ("A", "B")]
    np.testing.assert_allclose(goals[0][:3, 0], -goals[1][:3, 0])
    np.testing.assert_allclose(goals[0][:3, 2], goals[1][:3, 2])
    results = []
    for arm, goal in zip(("A", "B"), goals):
        world = local @ goal[:3, :3].T + goal[:3, 3]
        results.append(prediction(record, candidate, points=world, matrix=goal, arm=arm))
    assert results[0]["q_width_rad"] == pytest.approx(results[1]["q_width_rad"], abs=1e-8)
    # TCP includes distinct left/right camera brackets; do not impose mirror equality.
    assert results[0]["link"] == "TCP_Link_L"
    assert results[1]["link"] == "TCP_Link_R"
    assert all(result["minimum_padded_noncontact_gap_m"] > .1 for result in results)
    assert all(result["accepted"] and abs(result["width_residual_m"]) <= 1e-8 for result in results)


def test_every_point_and_original_padding_are_retained_across_chunks():
    record = frozen("5be")
    candidate = record["candidates"][1]
    original = record["target"]["completed_geometry_points_world_m"].copy()
    base = prediction(record, candidate)
    larger_padding = prediction(record, candidate, padding=.007)
    assert larger_padding["q_width_rad"] == base["q_width_rad"]
    assert larger_padding["minimum_padded_noncontact_gap_m"] == pytest.approx(base["minimum_padded_noncontact_gap_m"] - .001)
    np.testing.assert_array_equal(record["target"]["completed_geometry_points_world_m"], original)
    model = gripper_width_model("A")
    rows = model.spheres[model.tool]
    last_local = rows[np.argmax(rows[:, 3]), :3]
    pose = np.array(candidate["T_world_tcp"])
    last_world = pose[:3, :3] @ last_local + pose[:3, 3]
    full = np.vstack((np.tile(original, (2, 1)), last_world))
    result = prediction(record, candidate, points=full)
    assert not result["accepted"]
    assert result["point_count"] == 3911 and result["point_index"] == 3910
    assert result["link"] == model.tool
    assert result["points_world_sha256"] != base["points_world_sha256"]
    assert set(result["contact_exemptions"]) == {"gripL_Base"} | {
        f"gripL_Link{n}_{side}" for n in range(1, 5) for side in ("L", "R")}
    assert result["contact_policy"] == "owner_hand_assembly_v1"
    assert "TCP_Link_L" in result["checked_links"]


@pytest.mark.parametrize("width", [.001, .2])
def test_outside_whole_cloud_width_does_not_replace_noncontact_collision_check(width):
    model = gripper_width_model("A")
    points = np.array([model.normal * -width / 2, model.normal * width / 2])
    result = predict_gripper_compatibility(points, np.eye(4), arm="A", uncertainty_m=.005, margin_m=.001)
    assert result['width_prediction_status'] == 'whole_cloud_width_outside_range'
    assert result['accepted'] == (result['minimum_padded_noncontact_gap_m'] >= 0)
    if width == .001:
        assert result['reason'] == 'predicted_noncontact_geometry_conflict'
    assert result["q_width_rad"] is None


@pytest.mark.parametrize("fault", ["nan_point", "missing_uncertainty", "bad_completion", "bad_model"])
def test_malformed_prediction_input_never_defaults_to_acceptance(monkeypatch, fault):
    record = frozen("5be")
    if fault == "nan_point":
        record["target"]["completed_geometry_points_world_m"][-1, 0] = np.nan
    elif fault == "missing_uncertainty":
        record["target"].pop("uncertainty_m")
    elif fault == "bad_completion":
        record["target"]["geometry_completion"]["observation_id"] = "old"
    else:
        def bad_model(arm):
            raise ValueError("authored pad gap is not strictly decreasing")
        monkeypatch.setattr("luna_mujoco.demo_gripper_quality.gripper_width_model", bad_model)
    candidate = record["candidates"][1]
    with pytest.raises(AlgorithmError) as caught:
        DemoAlgorithms()._grasp_ik_candidate({**candidate, "T_world_tcp": candidate["nominal_T_world_tcp"]},
            record["snapshot"], record["target"]["id"], "A", record["scene"], record["settings"])
    assert caught.value.code in {"grasp_gripper_prediction_invalid", "grasp_height_completion_provenance_invalid"}


def test_final_fk_cannot_cross_prediction_boundary_within_original_pose_tolerance():
    record = frozen("5be")
    algorithms = DemoAlgorithms()
    original = record["candidates"][1]
    # A controlled synthetic boundary witness extends this test fixture only.
    # It tests retained TCP collision independently of historical hand overlap.
    model = gripper_width_model("A")
    rows = model.spheres[model.tool]
    sphere = rows[np.argmax(rows[:, 2] + rows[:, 3])]
    local_witness = sphere[:3] + [0., 0., sphere[3] + .0064]
    pose = np.array(original["T_world_tcp"])
    append_test_points(record, [pose[:3, :3] @ local_witness + pose[:3, 3]])
    accepted, _ = algorithms._grasp_ik_candidate({**original, "T_world_tcp": original["nominal_T_world_tcp"]},
        record["snapshot"], record["target"]["id"], "A", record["scene"], record["settings"])
    source = np.array(accepted["T_world_tcp"])
    evidence = accepted["fk_goal_validation"]["predicted_gripper_compatibility"]
    model = gripper_width_model("A")
    spheres = model.noncontact_spheres(evidence["q_width_rad"])
    index = model.labels.index((evidence["link"], evidence["sphere_index"]))
    point = record["target"]["completed_geometry_points_world_m"][evidence["point_index"]]
    delta = (point - source[:3, 3]) @ source[:3, :3] - spheres[index, :3]
    final = source.copy()
    final[:3, 3] += source[:3, :3] @ (delta / np.linalg.norm(delta)) * .001
    fk = deepcopy(original["source_ik_fk"])
    phases = {"grasp": {"positions_rad": [fk["joints_rad"]]}}
    # Same nominal tolerance/direction passes, but predicted noncontact gap fails.
    check = algorithms._grasp_pose_validation(np.array(original["nominal_T_world_tcp"]), final,
        record["snapshot"], record["target"]["id"], "A", record["settings"])
    assert check["accepted"]
    fk["T_world_tcp"] = final.tolist()
    with pytest.raises(AlgorithmError) as caught:
        algorithms._grasp_plan_evidence(accepted, {"grasp_endpoint_fk": fk}, phases,
            record["snapshot"], record["target"]["id"], "A", record["scene"], record["settings"])
    assert caught.value.code == "planning_grasp_endpoint_rejected"
    assert caught.value.details["predicted_gripper_compatibility"]["minimum_padded_noncontact_gap_m"] < 0


def append_test_points(record, points):
    target = record["target"]
    target["completed_geometry_points_world_m"] = np.vstack((target["completed_geometry_points_world_m"], points))
    target["geometry_completion"]["synthetic_point_count"] += len(points)


def evaluated_frozen(indices, *, tcp_conflict=False):
    from test_rgbd_algorithms import observed_scene
    record = frozen("5be")
    snapshot = observed_scene() | record["snapshot"]
    candidates = [c for c in record["candidates"] if c["candidate_index"] in indices]
    if tcp_conflict:
        # Deliberate negative geometry: the actual fixture remains immutable on disk.
        model = gripper_width_model("A")
        rows = model.spheres[model.tool]
        center = rows[np.argmax(rows[:, 3]), :3]
        points = [np.array(c["T_world_tcp"])[:3, :3] @ center + np.array(c["T_world_tcp"])[:3, 3]
                  for c in candidates]
        append_test_points(record, points)
    grasps = {"sampling_round": 2, "arm": "A", "tool_profile_revision": TOOL_PROFILE["revision"],
              "T_world_cloud": np.eye(4).tolist(), "candidates": [{**c, "index": c["candidate_index"],
                "T_world_tcp": c["nominal_T_world_tcp"], "T_object_grasp": c["nominal_T_world_tcp"]} for c in candidates]}
    def request(*args):
        return {"scene_token": record["scene"]["scene_token"], "batch_size": len(candidates),
                "items": [{**c["ik"], "candidate_index": i} for i, c in enumerate(candidates)]}
    algorithm = DemoAlgorithms(curobo_request=request)
    result = algorithm.evaluate_arms(snapshot, record["target"]["id"], grasps, record["scene"],
        requested_arm="A", planning_settings=record["settings"])["A"]
    return record, result


def test_after_ik_evidence_keeps_raw_success_order_round_and_eligible_zero():
    record, evaluation = evaluated_frozen([508, 991], tcp_conflict=True)
    assert evaluation["valid_candidates"] == []
    batch = evaluation["batches"][0]
    assert batch["success_count"] == 2 and batch["eligible_count"] == 0
    assert [r["candidate_index"] for r in batch["candidate_filters"]] == [508, 991]
    replay = batch["candidate_replay"]
    assert replay["sampling_round"] == 2 and replay["candidate_indices"] == [508, 991]
    assert replay["candidate_filters"] == batch["candidate_filters"]
    assert all(r["success"] for r in replay["response"]["items"])
    for row in replay["candidate_filters"]:
        prediction = row["predicted_gripper_compatibility"]
        assert prediction["completion_provenance"] == record["target"]["geometry_completion"]
        assert prediction["measured_point_count"] == 1361
        assert len(prediction["model_fingerprint"]) == 64


def test_ordinary_algorithms_does_not_apply_demo_prediction():
    record = frozen("5be")
    candidate = record["candidates"][0]
    result, evidence = Algorithms()._grasp_ik_candidate(candidate, record["snapshot"],
        record["target"]["id"], "A", record["scene"], record["settings"])
    assert result is candidate and evidence is None


async def test_quality_empty_with_raw_ik_success_uses_existing_shared_pool_budget(tmp_path, monkeypatch):
    from test_demo_sampling import sampling_case
    _, real = evaluated_frozen([508], tcp_conflict=True)
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [False, True])
    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.evaluate_arms and args[2]["sampling_round"] == 1:
            batch = deepcopy(real["batches"][0])
            batch["candidate_indices"] = [17]
            result["A"]["batches"] = [batch]
        return result
    runner.call_algorithm = call
    result = await runner.plan_grasp(snapshot, "source", "A", context)
    assert result["sampling_round"] == 2
    evidence = result["sampling_evidence"]
    assert evidence["total_requested_candidate_count"] == 768
    assert evidence["rounds"][0]["batch_success_counts"] == [1]
    assert evidence["rounds"][0]["valid_count"] == 0
    assert [row[2]["timeout_s"] for row in calls] == [180., 175., 170., 165.]


async def test_final_quality_error_never_authorizes_another_pool(tmp_path, monkeypatch):
    from test_demo_sampling import sampling_case
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [True])
    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.plan_grasp:
            raise AlgorithmError("planned endpoint violates predicted compatibility",
                code="planning_grasp_endpoint_rejected", details={"reason": "predicted_noncontact_geometry_conflict"})
        return result
    runner.call_algorithm = call
    with pytest.raises(AlgorithmError):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert [row[0] for row in calls] == ["evaluate_arms", "plan_grasp"]


def test_malformed_own_sphere_model_fails_before_geometry_use(tmp_path, monkeypatch):
    import yaml

    from luna_mujoco import demo_gripper_quality as quality
    config = yaml.safe_load(quality.CONFIG.read_bytes())
    config["robot_cfg"]["kinematics"]["collision_spheres"]["TCP_Link_L"] = []
    path = tmp_path / "bad-model.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(quality, "CONFIG", path)
    with pytest.raises(ValueError, match="self-collision spheres"):
        quality.gripper_width_model("A")
