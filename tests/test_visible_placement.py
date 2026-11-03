
import numpy as np

from luna_mujoco.placement import carrying_binding, public_observation


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
    holding = {"frame": "gripper_front", "arm": "A", "points_tool_m": points_tool.tolist(), "grasp_support_reference": {
        "T_world_tcp_at_grasp": tcp.tolist(), "table_height_m": .6,
        "source": "rgbd_table_and_joint_feedback"}}
    return target, destination, holding


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
