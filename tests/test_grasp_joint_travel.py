"""Select minimum joint travel from the first batch's successful planner trajectories."""

import hashlib
import json
from copy import deepcopy
from types import MethodType

import numpy as np
import pytest
from test_planning_batch import batch_response, evaluated

from luna_mujoco.algorithms import AlgorithmError, Algorithms


def set_path(item, segments, *, joint=0):
    for name, values in zip(("approach", "grasp", "lift"), segments):
        phase = item["phases"][name]
        positions = np.zeros((len(values), 7))
        # The service fixture uses reverse joint order.
        positions[:, 6 - joint] = values
        phase["positions_rad"] = positions.tolist()
        phase["velocities_rad_s"] = np.zeros_like(positions).tolist()
        phase["accelerations_rad_s2"] = np.zeros_like(positions).tolist()


@pytest.mark.parametrize("destination_id", [None, "blue-bin"])
def test_lower_score_wins_on_complete_joint_travel(destination_id):
    algorithms, state, scene, evaluations, _ = evaluated(2, scores=[.9, .1])
    evaluations["A"]["destination_id"] = destination_id

    def request(path, body):
        result = batch_response(body)
        set_path(result["items"][0], [[0., .4], [.4, .5], [.5, .6]])
        set_path(result["items"][1], [[0., .1], [.1, .2], [.2, .3]])
        return result

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene, destination_id=destination_id)
    assert result["candidate_index"] == 101
    assert result["joint_travel_rad"] == pytest.approx(.3)
    assert result["selection"]["all_ik_candidates_evaluated"] is True
    assert result["selection"]["raw_complete_candidate_count"] == 2


def test_successful_batch_does_not_plan_later_candidates_for_a_smaller_cost():
    algorithms, state, scene, evaluations, _ = evaluated(17, scores=[.9] * 16 + [.1])
    batch_count = 0

    def request(path, body):
        nonlocal batch_count
        batch_count += 1
        result = batch_response(body)
        for item in result["items"]:
            end = .6 if batch_count == 1 else .3
            set_path(item, [[0., end], [end, end], [end, end]])
        return result

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert batch_count == 1
    assert result["candidate_index"] == 100
    assert result["joint_travel_rad"] == pytest.approx(.6)
    assert result["planned_candidate_count"] == 16
    assert result["unplanned_candidate_indices"] == [116]
    assert result["selection"]["all_ik_candidates_evaluated"] is False


def test_travel_counts_measured_start_all_joints_and_phase_connections():
    algorithms, state, scene, evaluations, _ = evaluated(1)
    state["arms"]["A"]["joints_rad"] = [.005] * 7

    def request(path, body):
        result = batch_response(body)
        for name, values in zip(("approach", "grasp", "lift"), ([0., .1], [.11, .2], [.19, .3])):
            phase = result["items"][0]["phases"][name]
            phase["positions_rad"] = [[value] * 7 for value in values]
        return result

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    # Per joint: .005 start + .1 approach + .01 seam + .09 grasp + .01 seam + .11 lift.
    assert result["joint_travel_rad"] == pytest.approx(2.275)
    assert result["selection"]["start_joints_rad"] == [.005] * 7


def test_joint_reversals_are_counted_instead_of_final_displacement():
    algorithms, state, scene, evaluations, _ = evaluated(2, scores=[.9, .1])

    def request(path, body):
        result = batch_response(body)
        set_path(result["items"][0], [[0., .4, 0.], [0., 0.], [0., 0.]])
        set_path(result["items"][1], [[0., .1], [.1, .2], [.2, .3]])
        return result

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert result["candidate_index"] == 101
    assert result["joint_travel_rad"] == pytest.approx(.3)


def test_joint_travel_does_not_wrap_across_joint_limits():
    algorithms, state, scene, evaluations, _ = evaluated(2, scores=[.9, .1])
    state["arms"]["A"]["joints_rad"][0] = 2.9
    evaluations["A"]["start_joints_deg"][0] = float(np.rad2deg(2.9))

    def request(path, body):
        result = batch_response(body)
        set_path(result["items"][0], [[2.9, -2.9], [-2.9, -2.9], [-2.9, -2.9]])
        set_path(result["items"][1], [[2.9, 2.], [2., 2.], [2., 2.]])
        return result

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert result["candidate_index"] == 101
    assert result["joint_travel_rad"] == pytest.approx(.9)


def test_rank_returned_paths_before_interpolating_or_validating_candidates(monkeypatch):
    from luna_mujoco import algorithms as algorithms_module

    algorithms, state, scene, evaluations, _ = evaluated(16, scores=[.9] * 16)
    prepare = algorithms_module.build_execution_phases
    prepared, checked = [], []

    def prepare_once(*args):
        prepared.append(True)
        return prepare(*args)

    monkeypatch.setattr(algorithms_module, 'build_execution_phases', prepare_once)
    algorithms.validate_execution = lambda *args, **kwargs: checked.append(True) or {'collision_free': True}

    def request(path, body):
        result = batch_response(body)
        for index, item in enumerate(result['items']):
            end = (16 - index) * .1
            set_path(item, [[0., end], [end, end], [end, end]])
        return result

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert result["candidate_index"] == 115
    assert result["joint_travel_rad"] == pytest.approx(.1)
    assert len(prepared) == 1
    assert len(checked) == 2


def test_unneeded_later_batch_is_never_requested():
    algorithms, state, scene, evaluations, _ = evaluated(17)
    count = 0

    def request(path, body):
        nonlocal count
        count += 1
        if count == 2:
            raise AlgorithmError("service unavailable", code="algorithm_service_unavailable")
        return batch_response(body)

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert result['candidate_index'] == 116
    assert count == 1


@pytest.mark.parametrize("bad_phase", ["grasp", "lift"])
def test_cheaper_candidate_with_invalid_phase_connection_cannot_mask_valid_plan(bad_phase):
    algorithms, state, scene, evaluations, _ = evaluated(2, scores=[.9, .1])

    def request(path, body):
        result = batch_response(body)
        segments = [[0., .01], [.01, .02], [.02, .03]]
        segments[1 if bad_phase == "grasp" else 2] = [.1, .11]
        set_path(result["items"][0], segments)
        set_path(result["items"][1], [[0., .1], [.1, .2], [.2, .3]])
        return result

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert result["candidate_index"] == 101
    assert result["prior_failures"][0]["stage"] == "command_validation"
    assert result["prior_failures"][0]["code"] == "planning_phase_start_changed"


def test_cheaper_command_collision_cannot_mask_costlier_valid_plan():
    algorithms, state, scene, evaluations, _ = evaluated(2, scores=[.9, .1])

    def request(path, body):
        result = batch_response(body)
        set_path(result["items"][0], [[0., .01], [.01, .02], [.02, .03]])
        set_path(result["items"][1], [[0., .1], [.1, .2], [.2, .3]])
        return result

    def validate(observation, arm, scene, phase, command, **kwargs):
        if command["positions_rad"][-1][0] < .1:
            raise AlgorithmError("command collides", code="execution_collision")
        return {"collision_free": True}

    algorithms.request = request
    algorithms.validate_execution = validate
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert result["candidate_index"] == 101
    assert result["prior_failures"][0]["code"] == "execution_collision"


def test_command_validation_service_error_aborts_search():
    algorithms, state, scene, evaluations, _ = evaluated(2, scores=[.9, .1])
    algorithms.request = lambda path, body: batch_response(body)

    def validate(*args, **kwargs):
        raise AlgorithmError("service unavailable", code="algorithm_service_unavailable")

    algorithms.validate_execution = validate
    with pytest.raises(AlgorithmError, match="service unavailable"):
        algorithms.plan_grasp(state, "red", "A", evaluations, scene)


def test_next_batch_is_used_only_when_current_commands_are_all_rejected():
    algorithms, state, scene, evaluations, _ = evaluated(17)
    batches = []

    def request(path, body):
        batches.append(body)
        result = batch_response(body)
        end = .01 if len(batches) == 1 else .3
        for item in result['items']:
            set_path(item, [[0., end], [end, end], [end, end]])
        return result

    def validate(observation, arm, scene, phase, command, **kwargs):
        if command['positions_rad'][-1][0] < .1:
            raise AlgorithmError('command collides', code='execution_collision')
        return {'collision_free': True}

    algorithms.request = request
    algorithms.validate_execution = validate
    result = algorithms.plan_grasp(state, 'red', 'A', evaluations, scene)
    assert len(batches) == 2
    assert result['candidate_index'] == 100
    assert len(result['prior_failures']) == 16


def test_command_checks_bind_measured_connector_and_predicted_phase_start_without_lift_check():
    algorithms, state, scene, evaluations, _ = evaluated(1)
    algorithms.validate_execution = MethodType(Algorithms.validate_execution, algorithms)
    for feedback in state["arms"].values():
        feedback["gripper_position_rad"] = 0.
    state["arms"]["A"]["joints_rad"] = [.005] * 7
    original = deepcopy(state)
    checks = []

    def request(path, body):
        if path == "/tools/plan_grasp_batch":
            return batch_response(body)
        assert path == "/carrying/validate"
        checks.append(deepcopy(body))
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        return {"scene_token": body["scene_token"], "binding_sha256": digest,
                "collision_free": True, "attached": False}

    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert [check["phase"] for check in checks] == ["approach_grasp", "grasp"]
    assert all(check["gripper_geometry_mode"] == "fixed_open" for check in checks)
    assert checks[0]["current_joints_rad"] == [.005] * 7
    assert checks[0]["positions_rad"][:2] == [[.005] * 7, [0.] * 7]
    assert checks[1]["current_joints_rad"] == checks[0]["positions_rad"][-1]
    assert checks[1]["contact_goal_pose_world"] == result["T_world_tcp"]
    assert result["selection"]["command_validated_candidate_count"] == 1
    assert result["selection"]["lift_command_collision_checked"] is False
    assert state["arms"] == original["arms"]
