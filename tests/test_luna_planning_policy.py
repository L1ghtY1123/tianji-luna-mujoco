"""Ten-millimetre clearance and region-based release acceptance."""
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation
from test_demo_planning import PlaceService, placement_inputs

from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.placement import release_endpoint_evidence
from luna_mujoco.planning_settings import PlanningSettings


def test_ten_mm_shared_clearance_preserves_other_distance_semantics():
    settings = PlanningSettings()
    assert settings.trajectory_collision_clearance_m == .01
    assert settings.place_release_height_m == .01
    assert settings.ik_position_threshold_m == .005
    assert settings.optimizer_collision_activation_distance_m == .03
    assert settings.attachment.minimum_partition_size_m == .01
    assert "voxel_size_m" not in settings.attachment.model_dump()
    assert PlanningSettings(attachment={"voxel_size_m": .007}).attachment.minimum_partition_size_m == .007
    with pytest.raises(ValidationError):
        PlanningSettings(trajectory_collision_clearance_m=.00999)


def test_region_accepts_pose_error_without_a_two_mm_gate(monkeypatch):
    observation, scene, holding, placement = placement_inputs()
    algorithms = DemoAlgorithms(curobo_request=PlaceService())
    monkeypatch.setattr(algorithms, "gripper_release_spheres", lambda *args: [[0., 0., .2, .02]])
    plan = algorithms.plan_place(observation, "seen-red", "A", scene, holding,
                                 "demo-destination", placement_request=placement)
    goal = np.asarray(plan["goal"]["T_world_tcp"])
    assert goal[2, 3] == pytest.approx(.8301)
    reference = plan["goal"]["release_reference"]
    assert reference["planning_reserve_m"] == .0001
    assert reference["required_clearance_m"] == .01
    assert reference["height_resolution"]["endpoint_margin_m"] == 0.
    actual = goal.copy()
    actual[:3, 3] += [.015, 0., .02]
    actual[:3, :3] = Rotation.from_euler("z", 5., degrees=True).as_matrix() @ actual[:3, :3]
    evidence = release_endpoint_evidence(plan["goal"], holding, actual)
    assert evidence["at_release_goal"] is True
    assert evidence["position_error_m"] > .002
    assert evidence["rotation_error_rad"] > np.deg2rad(2.)
    actual[0, 3] += .2
    assert release_endpoint_evidence(plan["goal"], holding, actual)["at_release_goal"] is False


def test_attachment_completion_preserves_measured_reference_and_labels_synthetic_points():
    from luna_mujoco.carrying import CarryingBinding, carrying_geometry
    from luna_mujoco.placement import carrying_binding

    measured = np.array([[x, y, .70] for x in np.linspace(.3, .34, 4)
                         for y in np.linspace(.1, .12, 4)])
    completed = np.vstack((measured, [[.3, .1, .6], [.34, .12, .6]]))
    obj = {"id": "object", "points_world_m": measured,
           "completed_geometry_points_world_m": completed,
           "geometry_completion": {"source": "measured_surface_and_user_height_prior",
               "synthetic_points_are_measured": False, "measured_point_count": 16,
               "synthetic_point_count": 2, "height_prior_m": .10}}
    observation = {"observation_id": "rgbd-1", "generation": 1, "objects": [obj],
                   "arms": {"A": {"tcp": np.eye(4).tolist()}}}
    payload = carrying_binding(observation, "object", "A", "map", owner_id="owner")
    assert len(payload["points_tool_m"]) == 18
    provenance = payload["measurement_provenance"]
    assert provenance["source"] == "registered_rgbd_surface_with_explicit_height_completion"
    np.testing.assert_allclose(provenance["measured_reference_tool_m"], measured.mean(axis=0))
    parsed = CarryingBinding.from_mapping(payload, scene={"scene_token": "map",
        "observation_id": "rgbd-1", "generation": 1, "target_id": "object"}, arm="A")
    geometry = carrying_geometry(parsed)
    assert sum(row["measured"] for row in geometry["source_counts_by_sphere"]) == 16
    assert sum(row["height_completion"] for row in geometry["source_counts_by_sphere"]) == 2
    spheres = np.array(geometry["spheres"])
    gaps = np.linalg.norm(completed[:, None] - spheres[None, :, :3], axis=-1) - spheres[None, :, 3]
    assert np.max(gaps.min(axis=1)) <= -.006 + 1e-10
    obj["completed_geometry_points_world_m"][0, 0] += .01
    with pytest.raises(ValueError, match="completion"):
        carrying_binding(observation, "object", "A", "map", owner_id="owner")
