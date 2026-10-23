from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco import backend_builder
from luna_mujoco.backend_builder import CuroboToolError
from luna_mujoco.release_clearance import release_clearance_evidence


def request(**changes):
    from tianji_curobo.contracts import PlanPlaceRequest

    return PlanPlaceRequest.create(**({"arm": "A", "start_joints_deg": [0.] * 7,
        "other_joints_deg": [0.] * 7, "goal_position_m": [.5, .2, .7],
        "goal_quaternion_wxyz": [1., 0., 0., 0.], "freq_hz": 50, "max_attempts": 1,
        "trajectory_collision_clearance_m": .01} | changes))


def test_release_uses_actual_tool_spheres_goal_rotation_and_hard_world_cost_without_padding_twice():
    angle = np.pi / 4
    goal = request(goal_quaternion_wxyz=[np.cos(angle), 0., 0., np.sin(angle)])
    spheres = np.array([[.02, 0., -.05, .013], [0., .01, .03, .007]])
    original = spheres.copy()
    queries = []
    def query(world):
        queries.append(world.copy())
        return np.array([[[.003, 0.]]])
    evidence = release_clearance_evidence(spheres, goal, query, scene_token="scene-current")
    np.testing.assert_allclose(queries[0][0, 0], [[.5, .22, .65, .013], [.49, .2, .73, .007]])
    np.testing.assert_array_equal(spheres, original)
    assert evidence["conflict"] is True
    assert evidence["conflicting_sphere_indices"] == [0]
    assert evidence["conflicting_spheres_world_m"][0][3] == .013
    assert evidence["required_clearance_m"] == .01
    assert evidence["scene_token"] == "scene-current"
    assert evidence["min_clearance_m"] is None  # Activation costs are not signed distances.
    assert evidence["full_object_coverage"] is False


def test_zero_world_cost_is_only_no_detected_conflict_not_a_reachability_or_coverage_proof():
    proof = release_clearance_evidence([[0., 0., -.01, .01]], request(), lambda spheres: np.zeros((1, 1, 1)),
                                      scene_token="scene-current")
    assert proof["conflict"] is False
    assert proof["status"] == "no_conflict_detected"
    assert "collision_free" not in proof and "reachable" not in proof
    assert proof["scope"] == "attached_surface_spheres_at_requested_release_goal"


def test_nominal_release_floor_conflict_accounts_for_uploaded_radius_and_hard_clearance():
    goal = request()
    measured_floor_z = .63
    def floor_cost(world):
        return np.maximum(0., goal.trajectory_collision_clearance_m -
                          (world[..., 2] - world[..., 3] - measured_floor_z))
    # These callbacks model a measured planar ESDF solely to exercise the boundary;
    # production queries the existing world checker and never invents an infinite floor.
    small = release_clearance_evidence([[0., 0., -.05, .003]], goal, floor_cost, scene_token="scene-current")
    covered = release_clearance_evidence([[0., 0., -.05, .013]], goal, floor_cost, scene_token="scene-current")
    assert small["conflict"] is False
    assert covered["conflict"] is True
    assert covered["world_collision_costs"][0] == pytest.approx(.003)
    assert goal.goal_position_m == (.5, .2, .7)


@pytest.mark.parametrize("cost", [np.array([[[np.nan]]]), np.array([[[np.inf]]]), np.array([[[-.1]]]),
    np.array([[0.]]), np.zeros((1, 2, 1)), np.zeros((1, 1, 2))])
def test_invalid_collision_evidence_is_never_classified_as_clearance_conflict(cost):
    with pytest.raises(ValueError, match="world_cost"):
        release_clearance_evidence([[0., 0., 0., .01]], request(), lambda spheres: cost, scene_token="scene-current")


class Tensor:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float32)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.values


def backend(cost):
    value = backend_builder.ValidatedSingleBackend.__new__(backend_builder.ValidatedSingleBackend)
    value.arm = "A"
    value._torch = SimpleNamespace(as_tensor=lambda data, **kwargs: Tensor(data))
    value._device, value._dtype = "cpu", "float32"
    value.carrying_link = "jev_carried_object_A"
    value.carrying_scene_token = "scene-current"
    value.carrying_spheres_tool = np.array([[0., 0., -.05, .013]], dtype=np.float32)
    value._motion_clearance_checker = SimpleNamespace(get_collision_distance=lambda spheres: Tensor(cost))
    value.motion_planner = SimpleNamespace(last_grasp_result="old", last_pose_result="old")
    return value


def test_positive_release_conflict_stops_before_native_planning_with_auditable_geometry(monkeypatch):
    calls = []
    monkeypatch.setattr(backend_builder.ValidatedSingleBackend, "_plan_pre_release_native", lambda *args: calls.append(args))
    value = backend([[[.001]]])
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value.code == "release_clearance_conflict"
    assert caught.value.details["conflicting_sphere_indices"] == [0]
    assert caught.value.details["frame"] == "world"
    assert calls == []
    assert value.motion_planner.last_grasp_result is None


def test_zero_cost_preserves_native_path_failure_and_does_not_change_release_goal(monkeypatch):
    goal = request()
    failure = CuroboToolError("path unavailable", code="CUROBO_GRASP_FAILED", stage="trajopt", status=422)
    calls = []
    def failed_native(self, forwarded):
        calls.append(forwarded)
        raise failure
    monkeypatch.setattr(backend_builder.ValidatedSingleBackend, "_plan_pre_release_native", failed_native)
    with pytest.raises(CuroboToolError) as caught:
        backend([[[0.]]]).plan_place_native(goal)
    assert caught.value is failure and calls == [goal]
    assert goal.goal_position_m == (.5, .2, .7)


def test_zero_cost_keeps_native_planning_derivatives_and_all_final_phase_checks(monkeypatch):
    calls = []
    phases = SimpleNamespace(**{name: SimpleNamespace(positions_rad=[[0.] * 7, [.1] * 7])
                               for name in ("approach", "place")})
    def planned(self, goal):
        calls.append("native")
        return phases
    monkeypatch.setattr(backend_builder.ValidatedSingleBackend, "_plan_pre_release_native", planned)
    value = backend([[[0.]]])
    value.validate_trajectory_native = lambda phase: calls.append("hard_phase_check")
    assert value.plan_place_native(request()) is phases
    assert calls == ["native", "hard_phase_check", "hard_phase_check"]


def test_missing_attachment_cannot_be_used_for_endpoint_proof(monkeypatch):
    calls = []
    monkeypatch.setattr(backend_builder.ValidatedSingleBackend, "_plan_pre_release_native", lambda *args: calls.append(args))
    value = backend([[[0.]]])
    value.carrying_spheres_tool = None
    with pytest.raises(CuroboToolError) as caught:
        value.plan_place_native(request())
    assert caught.value.code != "release_clearance_conflict"
    assert calls == []


def test_attachment_cache_retains_exact_uploaded_sphere_values_without_aliasing(monkeypatch):
    value = backend([[[0.]]])
    value._state = lambda joints: object()
    uploads = []
    value.motion_planner.attachment_manager = SimpleNamespace(update=lambda spheres, *args, **kwargs: uploads.append(spheres))
    monkeypatch.setattr(backend_builder, "carrying_spheres", lambda binding: [[.0123456789123, 0., 0., .013456789123]])
    value.attach_measured_carrying(SimpleNamespace(arm="A", scene_token="fresh"), [0.] * 7)
    np.testing.assert_array_equal(value.carrying_spheres_tool, uploads[0].values)
    uploads[0].values[:] = 99
    assert value.carrying_spheres_tool.max() < 1
    assert value.carrying_scene_token == "fresh"
