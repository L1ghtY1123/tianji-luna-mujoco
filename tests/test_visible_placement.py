import copy

import numpy as np
import pytest

from luna_mujoco.placement import carrying_binding, public_observation, release_goal


def measured_inputs():
    tcp = np.eye(4)
    tcp[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    tcp[:3, 3] = [.48, .08, .80]
    points_tool = np.array([[.02 + index * .001, -.03, -.15] for index in range(16)])
    points_world = points_tool @ tcp[:3, :3].T + tcp[:3, 3]
    target = {"id": "source", "kind": "cube", "color": "red",
              "visible_centroid_world": points_world.mean(axis=0).tolist(),
              "points_world_m": points_world.tolist()}
    destination = {"id": "destination", "kind": "bin", "placement_reference": {
        "position_world_m": [.7, -.2, .615], "floor_z_m": .615,
        "source": "rgbd_visible_floor", "measurement_sources": [{"camera_id": "head", "point_count": 20}]}}
    holding = {"arm": "A", "points_tool_m": points_tool.tolist(), "grasp_support_reference": {
        "T_world_tcp_at_grasp": tcp.tolist(), "table_height_m": .6,
        "source": "rgbd_table_and_joint_feedback"}}
    return target, destination, holding


def test_release_keeps_grasp_attitude_and_fixed_gripping_reference_xy():
    target, destination, holding = measured_inputs()
    before = copy.deepcopy((target, destination, holding))
    result = release_goal(target, destination, holding)
    tcp = np.asarray(result["T_world_tcp"])
    grasp = np.asarray(holding["grasp_support_reference"]["T_world_tcp_at_grasp"])
    np.testing.assert_array_equal(tcp[:3, :3], grasp[:3, :3])
    np.testing.assert_allclose(tcp[:2, 3], [.7, -.2], atol=1e-12)
    assert tcp[2, 3] == pytest.approx(.8 + .615 - .6 + .015)
    assert "T_world_object" not in result
    assert result["release_reference"]["release_height_m"] == .015
    assert before == (target, destination, holding)


def test_release_uses_pregrasp_visible_centroid_without_shape_or_holding_cloud():
    target, destination, holding = measured_inputs()
    holding.pop("points_tool_m")
    result = release_goal(target, destination, holding, release_height_m=.02, offset_world_m=(.01, -.02, 0.))
    tcp = np.asarray(result["T_world_tcp"])
    np.testing.assert_allclose(tcp[:2, 3], [.71, -.22])
    assert tcp[2, 3] == pytest.approx(.835)


@pytest.mark.parametrize("missing", ["placement_reference", "floor_z_m", "grasp_support_reference", "table_height_m"])
def test_release_missing_measured_floor_or_support_fails(missing):
    target, destination, holding = measured_inputs()
    if missing == "placement_reference":
        destination.pop(missing)
    elif missing == "floor_z_m":
        destination["placement_reference"].pop(missing)
    elif missing == "grasp_support_reference":
        holding.pop(missing)
    else:
        holding["grasp_support_reference"].pop(missing)
    with pytest.raises(ValueError, match="reference"):
        release_goal(target, destination, holding)


def test_carrying_binding_only_transforms_measured_points_and_never_reads_shape_geometry():
    target, _, holding = measured_inputs()
    target.update({"size": "poison", "T_world_object": "poison", "cavity": "poison"})
    observation = {"objects": [target], "observation_id": "rgbd-1", "generation": 1,
                   "arms": {"A": {"tcp": holding["grasp_support_reference"]["T_world_tcp_at_grasp"]}}}
    result = carrying_binding(observation, "source", "A", "scene-1", owner_id="run-1")
    np.testing.assert_allclose(result["points_tool_m"], holding["points_tool_m"], atol=1e-12)
    assert "estimated_bounds_tool_m" not in result
    assert "shape_prior_provenance" not in result
    assert result["coverage"] is None
    assert result["measurement_provenance"]["point_count"] == len(target["points_world_m"])


def test_public_observation_excludes_poison_geometry_including_nested_reference():
    target, destination, _ = measured_inputs()
    poison = {"size": [999] * 3, "quaternion": [1, 0, 0, 0], "T_world_object": "poison",
              "cavity": "poison", "shape_prior_provenance": "poison", "simulator_object_id": "secret"}
    target.update(poison)
    target["coverage"] = {"observed_surface_only": True, "camera_ids": ["head"], "point_count": 16, **poison}
    destination.update(poison)
    destination["placement_reference"].update(poison)
    destination["placement_reference"]["measurement_sources"] = {
        "camera_ids": ["head"], "point_count": 20, "reference_pixel_rc": [20, 30], "point_indices": [1, 2], **poison}
    observation = {"source": "rgbd", "objects": [target, destination], "table": {
        "surface_height_m": .6, "cavity": "poison"}, "views": "pixels"}
    result = public_observation(observation)
    for obj in result["objects"]:
        assert not set(poison).intersection(obj)
        assert "points_world_m" not in obj
    assert not set(poison).intersection(result["objects"][1]["placement_reference"])
    assert result["objects"][0]["coverage"] == {"observed_surface_only": True, "camera_ids": ["head"], "point_count": 16}
    assert result["objects"][1]["placement_reference"]["measurement_sources"] == {"camera_ids": ["head"], "point_count": 20}
    assert result["objects"][0]["visible_centroid_world"] == target["visible_centroid_world"]
    assert result["table"] == {"surface_height_m": .6}
