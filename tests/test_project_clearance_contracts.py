"""The project requires 10 mm through every copied contract boundary."""
from dataclasses import replace

import pytest

from luna_mujoco import planning_contracts as project


def test_project_contracts_keep_ten_mm_through_replace_and_vendor_type_boundary():
    import numpy as np
    from tianji_curobo import contracts as vendor

    common = {"arm": "A", "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
                  "trajectory_collision_clearance_m": .01}
    cases = (
        ("SolveIkBatchRequest", {"goal_matrices_world": [np.eye(4).tolist()], "scene_token": "map"}),
        ("PlanGraspBatchRequest", {"goal_matrices_world": [np.eye(4).tolist()], "scene_token": "map", "freq_hz": 50}),
        ("PlanPlaceRequest", {"goal_position_m": [.3, .2, .8], "goal_quaternion_wxyz": [1., 0., 0., 0.],
                                  "freq_hz": 50, "max_attempts": 5}),
    )
    for name, values in cases:
        kind = getattr(project, name)
        request = kind.from_mapping(common | values)
        assert isinstance(request, getattr(vendor, name))
        assert replace(request).trajectory_collision_clearance_m == .01
        with pytest.raises(ValueError):
            kind.from_mapping(common | values | {"trajectory_collision_clearance_m": .00999})
        with pytest.raises(ValueError):
            kind.from_mapping(common | values | {"start_joints_deg": [0.] * 6})
        assert getattr(vendor, name).from_mapping(common | values).trajectory_collision_clearance_m == .01


def test_dual_contracts_preserve_validated_clearance_in_serialization():
    from tianji_curobo.dual_contracts import DualPlanRequest as VendorDual

    from luna_mujoco.planning_contracts import DualPlanRequest, DualValidationRequest

    joints = {"A": [0.] * 7, "B": [0.] * 7}
    request = DualPlanRequest.from_mapping({"start_joints_deg": joints, "goal_joints_deg": joints,
        "optimizer_collision_activation_distance_m": .03})
    assert isinstance(request, VendorDual)
    assert request.to_mapping()["trajectory_collision_clearance_m"] == .01
    assert request.to_mapping()["optimizer_collision_activation_distance_m"] == .03
    checked = DualValidationRequest.from_mapping({"points_deg": {a: [v, v] for a, v in joints.items()},
        "trajectory_collision_clearance_m": .01})
    assert checked.trajectory_collision_clearance_m == .01
    with pytest.raises(ValueError):
        DualPlanRequest.from_mapping({"start_joints_deg": joints, "goal_joints_deg": joints,
            "trajectory_collision_clearance_m": .01, "optimizer_collision_activation_distance_m": .004})


def test_dual_native_internal_validation_receives_actual_ten_mm(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from luna_mujoco import dual_planning as module
    from luna_mujoco.planning_contracts import DualPlanRequest

    joints = {"A": [0.] * 7, "B": [0.] * 7}
    request = DualPlanRequest.from_mapping({"start_joints_deg": joints, "goal_joints_deg": joints,
        "trajectory_collision_clearance_m": .01})
    names = list(module.JOINT_NAMES["A"] + module.JOINT_NAMES["B"])
    positions = np.zeros((2, 14))
    result = SimpleNamespace(get_interpolated_plan=lambda: object())
    monkeypatch.setattr(module, "_success_matrix", lambda value: [True])
    monkeypatch.setattr(module, "_trajectory_values", lambda *args: (positions, names))
    monkeypatch.setattr(module, "_hermite_command_positions", lambda *args: positions)
    monkeypatch.setattr(module, "_planner_interpolation_dt", lambda value: .02)
    monkeypatch.setattr(module, "_planning_time", lambda value: .1)
    validations = []
    backend = SimpleNamespace(joint_names=names, _state=lambda value: value,
        _ordered_dual_joints=lambda value: value["A"] + value["B"],
        motion_planner=SimpleNamespace(plan_cspace=lambda *args, **kwargs: result),
        validate_dual_native=lambda value: validations.append(value) or 2)
    response = module.ValidatedDualBackend.plan_dual_native(backend, request)
    assert response["ok"] is True
    assert len(validations) == 1
    assert validations[0].trajectory_collision_clearance_m == .01
