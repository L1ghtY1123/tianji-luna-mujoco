"""Synthetic registered camera fixture validates the real public GPU Mapper."""

import json
import time

import numpy as np


def main():
    from tianji_curobo.curobo_service import materialize_urdf
    from tianji_curobo.semantic_esdf_transport import encode_semantic_esdf_request

    from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity

    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    depth = np.full((96, 96), .6, dtype=np.float32)
    source_mask = np.zeros(depth.shape, dtype=np.uint8)
    source_mask[42:54, 56:68] = 1
    obstacle_mask = np.zeros(depth.shape, dtype=np.uint8)
    obstacle_mask[30:42, 25:37] = 1
    depth[source_mask != 0] = .54
    depth[obstacle_mask != 0] = .53
    transform = [[1, 0, 0, .55], [0, -1, 0, 0], [0, 0, -1, 1.3], [0, 0, 0, 1]]
    intrinsics = [[120, 0, 47.5], [0, 120, 47.5], [0, 0, 1]]
    views = {camera: {"camera_id": camera, "stereo_sequence": 1,
        "depth_left_m": depth, "valid_left_mask": np.ones(depth.shape, dtype=np.uint8),
        "left_intrinsics": intrinsics, "T_base_left_camera": transform,
        "robot_joints_at_capture": {"A": [0, 0, 0, -42.5, 0, 0, 0], "B": [0, 0, 0, -42.5, 0, 0, 0]},
        "rgb_size": [96, 96], "rgb_intrinsics": intrinsics, "left_to_rgb": np.eye(4).tolist(),
        "instance_masks": {"selected": source_mask, "obstacle": obstacle_mask}} for camera in ("1", "2", "3")}
    payload = encode_semantic_esdf_request({"observation_set_id": "synthetic-registered", "sam3_generation": 1,
        "confirmed_instance_id": "selected", "selection_generation": 1, "views": views}) | {
            "gripper_joint_positions_rad": {"A": 0., "B": 0.}}
    started = time.perf_counter()
    result = run_operation(runtime, "/perception/build", payload | {"planning_settings": {}})
    assert result["source"] == "registered_rgbd"
    assert result["excluded_object_ids"] == ["selected"]
    assert "obstacle" in result["collision_objects"]
    assert result["layer_count"] == 3 and result["table_collision_enabled"]
    grids = {grid.name: grid for grid in runtime.geometry_grids}
    assert "semantic_instance::selected" in grids and "semantic_instance::obstacle" in grids
    selected_centers = np.asarray(runtime.geometry_model["voxel"]["semantic_instance::selected"]["centers_base_m"])
    assert np.allclose(np.median(selected_centers[:, 2]), .76, atol=.025), selected_centers
    first = result["scene_token"]
    capture_joints = views["1"]["robot_joints_at_capture"]
    open_spheres = runtime.perception_mapper.segmenter_boundary.robot_spheres(capture_joints)
    held = run_operation(runtime, "/perception/build", payload | {"planning_settings": {}, "exclusion_state": "held",
        "gripper_joint_positions_rad": {"A": .55, "B": 0.}})
    held_mapper = runtime.perception_mapper
    held_spheres = held_mapper.segmenter_boundary.robot_spheres(capture_joints)
    mask_geometry_motion = np.linalg.norm(held_spheres[:, :3] - open_spheres[:, :3], axis=1).max()
    assert np.isfinite(held_spheres).all() and mask_geometry_motion > .005
    assert held["gripper_joint_positions_rad"] == {"A": .55, "B": 0.}
    assert runtime.perception_gripper_feedback == {"A": .55, "B": 0.}
    released = run_operation(runtime, "/perception/build", payload | {"planning_settings": {}, "exclusion_state": "released"})
    assert released["scene_token"] != first and released["excluded_object_ids"] == []
    assert set(released["collision_objects"]) == {"selected", "obstacle"}
    assert runtime.geometry_scene["disabled_obstacle_names"] == ()
    assert runtime.perception_mapper is held_mapper
    released_spheres = runtime.perception_mapper.segmenter_boundary.robot_spheres(capture_joints)
    assert np.allclose(released_spheres, open_spheres, atol=1e-6)
    print(json.dumps({"sdk": sdk_identity()[0], "mapper": "public_curobo_perception_Mapper",
        "source": result["source"], "layer_count": result["layer_count"],
        "selected_surface_height_m": float(np.median(selected_centers[:, 2])),
        "non_target_layer_enabled": True, "released_source_layer_enabled": True,
        "held_mapper_driver_position_rad": .55, "released_mapper_driver_position_rad": 0.,
        "mapper_reconfigured_for_actual_gripper": True,
        "measured_gripper_mask_sphere_motion_m": float(mask_geometry_motion),
        "wall_time_s": time.perf_counter() - started, "timings_ms": result["timings_ms"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
