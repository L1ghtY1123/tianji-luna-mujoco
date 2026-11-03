"""Five-millimetre clearance and region-based release acceptance."""
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation
from test_demo_planning import PlaceService, placement_inputs

from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.demo_placement import release_endpoint_evidence
from luna_mujoco.planning_settings import PlanningSettings


def test_five_mm_shared_clearance_preserves_other_distance_semantics():
    settings = PlanningSettings()
    assert settings.trajectory_collision_clearance_m == .005
    assert settings.place_release_height_m == 0.
    assert settings.ik_position_threshold_m == .005
    assert settings.optimizer_collision_activation_distance_m == .03
    assert settings.attachment.minimum_partition_size_m == .01
    assert "voxel_size_m" not in settings.attachment.model_dump()
    assert PlanningSettings(attachment={"voxel_size_m": .007}).attachment.minimum_partition_size_m == .007
    with pytest.raises(ValidationError):
        PlanningSettings(trajectory_collision_clearance_m=-.001)


def test_region_accepts_pose_error_without_a_two_mm_gate(monkeypatch):
    observation, scene, holding, placement = placement_inputs()
    algorithms = DemoAlgorithms(curobo_request=PlaceService())
    monkeypatch.setattr(algorithms, "gripper_release_spheres", lambda *args: [[0., 0., .2, .02]])
    plan = algorithms._plan_place_at_reference(observation, "seen-red", "A", scene, holding,
                                 "demo-destination", placement_request=placement)
    goal = np.asarray(plan["goal"]["T_world_tcp"])
    assert goal[2, 3] == pytest.approx(.8251)
    reference = plan["goal"]["release_reference"]
    assert reference["planning_reserve_m"] == .0001
    assert reference["required_clearance_m"] == .005
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


def test_attachment_rejects_completion_and_preserves_all_actual_measured_points():
    from luna_mujoco.carrying import CarryingBinding, carrying_geometry
    from luna_mujoco.placement import carrying_binding, visible_reference_tool_m

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
    with pytest.raises(ValueError, match="synthetic_scene_geometry_unsupported"):
        carrying_binding(observation, "object", "A", "map", owner_id="owner")
    obj.pop("completed_geometry_points_world_m")
    obj.pop("geometry_completion")
    payload = carrying_binding(observation, "object", "A", "map", owner_id="owner")
    assert len(payload["points_tool_m"]) == len(measured)
    provenance = payload["measurement_provenance"]
    assert provenance["source"] == "registered_rgbd_visible_surface"
    np.testing.assert_allclose(visible_reference_tool_m(payload), measured.mean(axis=0))
    parsed = CarryingBinding.from_mapping(payload, scene={"scene_token": "map",
        "observation_id": "rgbd-1", "generation": 1, "target_id": "object"}, arm="A")
    geometry = carrying_geometry(parsed)
    assert geometry['fit_method'] == 'obb_volume_partition_cover'
    assert geometry['obb_tool']['measured_point_count'] == len(measured)
    np.testing.assert_array_equal(parsed.points_tool_m, measured)
    spheres = np.array(geometry["spheres"])
    gaps = np.linalg.norm(measured[:, None] - spheres[None, :, :3], axis=-1) - spheres[None, :, 3]
    assert np.max(gaps.min(axis=1)) <= -.006 + 1e-10
