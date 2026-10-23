"""Offline parity for articulated perception; never calls the running service."""

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np


def main():
    import torch
    import yaml
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.perception import RobotSegmenter
    from curobo.types import CameraObservation, JointState, Pose
    from tianji_curobo.curobo_robot_config import ARM_CONFIG_SPECS, create_robot_segmentation_config
    from tianji_curobo.curobo_service import materialize_urdf
    from tianji_curobo.robot_segmenter import (
        FfsDepthObservationRequest,
        SemanticProjectionRequest,
        create_robot_segmenter_adapter,
    )

    from luna_mujoco.carrying import configure_gripper_feedback
    from luna_mujoco.perception_mapping import create_feedback_segmenter
    from luna_mujoco.semantic_depth import partition_semantic_depth
    from luna_mujoco.semantic_transport import decode_project_semantic_request

    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    root = Path("/opt/jev")
    config_path = root / "resources/curobo/marvin_gui_dual.yml"
    original = yaml.safe_load(config_path.read_bytes())
    sphere_count = sum(map(len, original["robot_cfg"]["kinematics"]["collision_spheres"].values()))
    urdf = materialize_urdf(Path(os.environ["LUNA_ROBOT_URDF"]),
        marvin_meshes=Path(os.environ["LUNA_MARVIN_MESHES"]),
        rag_meshes=Path(os.environ["LUNA_RAG_MESHES"]))
    payload = json.loads(args.capture.read_text())["body"]
    fields = {key: value for key, value in payload.items() if key not in (
        "planning_settings", "exclusion_state", "gripper_joint_positions_rad",
        "selected_surface_world_m", "pointcloud_filter", "checkpoint", "held_instance_id",
        "camera_capture_metadata", "retained_bin_ids", "completed_transfer",
        "full_observation_complete", "mapping_max_sphere_radius_m")}
    decoded = decode_project_semantic_request(fields, exclusion_state=payload.get("exclusion_state", "selected"))
    torch.cuda.synchronize()
    started = time.perf_counter()
    boundary = create_feedback_segmenter(robot_config_path=config_path, urdf_path=urdf,
        asset_root_path=urdf.parent, feedback={"A": 0., "B": 0.}).sdk_boundary
    torch.cuda.synchronize()
    construction_s = time.perf_counter() - started
    kinematics_id = id(boundary.kinematics)
    assert len(boundary.kinematics.joint_names) == 16
    baseline_helpers = create_robot_segmenter_adapter(robot_config_path=config_path,
        urdf_path=urdf, asset_root_path=urdf.parent).sdk_boundary
    feedbacks = ({"A": 0., "B": 0.}, {"A": .55, "B": .2}, {"A": .1, "B": .65}, {"A": 0., "B": 0.})
    results, camera_ids = [], None
    semantic_views = []
    for feedback in feedbacks:
        started = time.perf_counter()
        boundary.update_feedback(feedback)
        feedback_ms = (time.perf_counter() - started) * 1000
        configured = configure_gripper_feedback(deepcopy(original), feedback, urdf)
        locked_config = create_robot_segmentation_config(configured, urdf_path=urdf, asset_root_path=urdf.parent)
        torch.cuda.synchronize()
        started = time.perf_counter()
        locked = Kinematics(KinematicsCfg.from_data_dict(locked_config, device_cfg=boundary.device_cfg))
        torch.cuda.synchronize()
        locked_construction_s = time.perf_counter() - started
        max_center_error, max_radius_error, removed = 0., 0., 0
        for view in decoded["views"].values():
            request = FfsDepthObservationRequest.from_mapping(view)
            positions = np.deg2rad([value for arm in ("A", "B") for value in request.joint_positions_deg[arm]])
            names = [name for arm in ("A", "B") for name in ARM_CONFIG_SPECS[arm]["joint_names"]]
            state = JointState.from_position(boundary.device_cfg.to_device(positions), joint_names=names)
            expected = locked.compute_kinematics(locked.get_active_js(state)).robot_spheres.detach().cpu().numpy().reshape(-1, 4)
            actual = boundary.robot_spheres(request.joint_positions_deg)
            assert actual.shape == expected.shape == (sphere_count, 4)
            max_center_error = max(max_center_error, float(np.max(np.abs(actual[:, :3] - expected[:, :3]))))
            max_radius_error = max(max_radius_error, float(np.max(np.abs(actual[:, 3] - expected[:, 3]))))
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
            raw = np.asarray(request.depth_left_m, dtype=np.float32).copy()
            raw[~((np.asarray(request.valid_left_mask) != 0) & np.isfinite(raw) & (raw > 0))] = 0
            source = CameraObservation(depth_image=boundary.device_cfg.to_device(raw[None]),
                intrinsics=boundary.device_cfg.to_device(request.left_intrinsics),
                pose=Pose.from_matrix(boundary.device_cfg.to_device(request.T_base_left_camera)), depth_to_meter=1.)
            baseline = RobotSegmenter(locked, distance_threshold=.05, use_cuda_graph=True, ops_dtype=torch.float32)
            baseline.update_camera_projection(source)
            mask, expected_depth = baseline.get_robot_mask(source, state)
            prepared = boundary.prepare_ffs_observation(request)
            np.testing.assert_array_equal(prepared.filtered_depth_m.cpu().numpy(), expected_depth.cpu().numpy())
            assert prepared.removed_point_count == int(mask.sum().item())
            removed += prepared.removed_point_count
            if len(results) == 0:
                semantic_views.append(partition_semantic_depth(prepared,
                    SemanticProjectionRequest.from_mapping(view, allow_empty=True)))
        ids = {camera: id(value[2]) for camera, value in boundary.robot_segmenters.items()}
        if camera_ids is not None:
            assert ids == camera_ids
        camera_ids = ids
        assert id(boundary.kinematics) == kinematics_id
        results.append({"feedback": feedback, "feedback_update_ms": feedback_ms,
            "locked_construction_s": locked_construction_s, "max_center_error_m": max_center_error,
            "max_radius_error_m": max_radius_error, "removed_pixels": removed, "mask_equal": True})
        print(json.dumps({"stage": "feedback_complete", **results[-1]}), flush=True)
    table = {"pose": [.5, 0., .5, 1., 0., 0., 0.], "dims": [.8, 1., .8]}
    for view in semantic_views:
        actual = boundary.crop_semantic_observation(view, table)
        expected = baseline_helpers.crop_semantic_observation(view, table)
        np.testing.assert_array_equal(actual.camera_observation.depth_image.cpu(), expected.camera_observation.depth_image.cpu())
        np.testing.assert_array_equal(actual.background_points_base_m, expected.background_points_base_m)
        for name in actual.instance_points_base_m:
            np.testing.assert_array_equal(actual.instance_points_base_m[name], expected.instance_points_base_m[name])
            for a, e in zip(actual.instance_depth_samples[name], expected.instance_depth_samples[name], strict=True):
                np.testing.assert_array_equal(a, e)
    instances = set().union(*(view.instance_points_base_m for view in semantic_views))
    for name in instances:
        actual = boundary.stack_instance_observations(semantic_views, name)
        expected = baseline_helpers.stack_instance_observations(semantic_views, name)
        np.testing.assert_array_equal(actual.depth_image.cpu(), expected.depth_image.cpu())
        np.testing.assert_array_equal(actual.intrinsics.cpu(), expected.intrinsics.cpu())
        np.testing.assert_allclose(actual.pose.get_matrix().cpu(), expected.pose.get_matrix().cpu(), atol=1e-6)
    print(json.dumps({"capture": str(args.capture), "construction_s": construction_s,
        "feedback_cases": results, "camera_ids_preserved": True, "kinematics_preserved": True,
        "crop_and_stack_equal": True, "sphere_count": sphere_count}, indent=2), flush=True)


if __name__ == "__main__":
    main()
