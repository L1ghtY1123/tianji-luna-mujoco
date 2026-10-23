"""Public cuRobo robot masking with articulated, measured gripper drivers."""

from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType

import numpy as np


def articulated_segmentation_config(robot_config, *, urdf_path, asset_root_path):
    from tianji_curobo.curobo_robot_config import create_robot_segmentation_config

    config = create_robot_segmentation_config(robot_config, urdf_path=urdf_path,
                                               asset_root_path=asset_root_path)
    kin = config["robot_cfg"]["kinematics"]
    for name in ("gripL_finger_joint", "gripR_finger_joint"):
        kin["lock_joints"].pop(name)
    # The authored cspace already names both drivers. The frozen helper expands
    # the arm-only clipping vector to 14; these two FK inputs use raw URDF limits.
    kin["cspace"]["position_limit_clip"] += [0., 0.]
    return config


class RawRobotDepthBoundary:
    """Own one robot model and one public segmentation graph per camera shape."""

    def __init__(self, configured, *, urdf_path, asset_root_path, feedback):
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg

        self.robot_config = deepcopy(configured)
        self.urdf_path = urdf_path
        self.device_cfg = DeviceCfg()
        self.update_feedback(feedback)
        config = articulated_segmentation_config(configured, urdf_path=urdf_path, asset_root_path=asset_root_path)
        self.kinematics = Kinematics(KinematicsCfg.from_data_dict(config, device_cfg=self.device_cfg))
        self.robot_segmenters = {}

    @property
    def sdk_boundary(self):
        return self

    @property
    def numpy(self):
        return np

    def update_feedback(self, feedback):
        from .carrying import configure_gripper_feedback

        if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
            raise ValueError("robot masking requires measured A and B gripper feedback")
        # Keep the existing finite/range validation and soft-limit normalization.
        configured = configure_gripper_feedback(self.robot_config, feedback, self.urdf_path)
        locks = configured["robot_cfg"]["kinematics"]["lock_joints"]
        self.gripper_feedback = {"A": locks["gripL_finger_joint"], "B": locks["gripR_finger_joint"]}

    def joint_state(self, joint_positions_deg):
        from curobo.types import JointState
        from tianji_curobo.curobo_robot_config import ARM_CONFIG_SPECS

        positions, names = [], []
        for arm in ("A", "B"):
            measured = np.asarray(joint_positions_deg[arm], dtype=float)
            if measured.shape != (7,) or not np.isfinite(measured).all():
                raise ValueError("robot masking requires seven finite measured joints for each arm")
            positions.extend(np.deg2rad(measured))
            names.extend(ARM_CONFIG_SPECS[arm]["joint_names"])
        positions.extend([self.gripper_feedback["A"], self.gripper_feedback["B"]])
        names.extend(["gripL_finger_joint", "gripR_finger_joint"])
        return JointState.from_position(self.device_cfg.to_device(positions), joint_names=names)

    def robot_spheres(self, joint_positions_deg):
        state = self.kinematics.get_active_js(self.joint_state(joint_positions_deg))
        spheres = self.kinematics.compute_kinematics(state).robot_spheres
        result = spheres.detach().cpu().numpy().reshape(-1, 4).astype(np.float64, copy=True)
        if not len(result) or not np.isfinite(result).all() or np.any(result[:, 3] <= 0):
            raise ValueError("robot masking returned invalid collision spheres")
        return result

    def prepare_ffs_observation(self, request):
        import torch
        from curobo.perception import RobotSegmenter
        from curobo.types import CameraObservation, Pose
        from tianji_curobo.robot_segmenter import PreparedFfsDepth

        depth = np.asarray(request.depth_left_m, dtype=np.float32).copy()
        valid = (np.asarray(request.valid_left_mask) != 0) & np.isfinite(depth) & (depth > 0)
        depth[~valid] = 0
        intrinsics = self.device_cfg.to_device(request.left_intrinsics)
        matrix = self.device_cfg.to_device(request.T_base_left_camera)
        pose = Pose.from_matrix(matrix)
        source = CameraObservation(depth_image=self.device_cfg.to_device(depth[None]),
                                   intrinsics=intrinsics, pose=pose, depth_to_meter=1.)
        shape, calibration, segmenter = self.robot_segmenters.get(request.camera_id, (None, None, None))
        if shape != depth.shape:
            segmenter = RobotSegmenter(self.kinematics, distance_threshold=.05,
                                       use_cuda_graph=True, ops_dtype=torch.float32)
        # Rays depend only on image size, intrinsics and the fixed metre scale.
        # Depth, pose and joints still come from this capture on every call.
        if shape != depth.shape or not np.array_equal(calibration, request.left_intrinsics):
            segmenter.update_camera_projection(source)
            self.robot_segmenters[request.camera_id] = (
                depth.shape, np.array(request.left_intrinsics, copy=True), segmenter)
        state = self.joint_state(request.joint_positions_deg)
        robot_mask, free_depth = segmenter.get_robot_mask(source, state)
        observation = CameraObservation(depth_image=free_depth, intrinsics=intrinsics.unsqueeze(0),
                                        pose=pose, depth_to_meter=1.)
        return PreparedFfsDepth(camera_id=request.camera_id, stereo_sequence=request.stereo_sequence,
            camera_observation=observation, filtered_depth_m=free_depth,
            base_pointcloud_m=observation.get_pointcloud(project_to_pose=True),
            intrinsics_tensor=intrinsics, pose_matrix_tensor=matrix,
            removed_point_count=int(robot_mask.sum().item()))

    def prepare_filtered_observation(self, request):
        import base64
        import io

        import torch
        from curobo.types import CameraObservation, Pose
        from PIL import Image
        from tianji_curobo.robot_segmenter import PreparedRobotDepth

        with Image.open(io.BytesIO(base64.b64decode(request.depth_png_base64, validate=True))) as image:
            depth = np.array(image)
            if image.format != "PNG" or depth.ndim != 2 or depth.dtype != np.uint16:
                raise ValueError("depth image must be a uint16 PNG")
        height, width = depth.shape
        calibration_width, calibration_height = request.depth_calibration_resolution
        intrinsics = np.array(request.depth_intrinsics, dtype=float)
        intrinsics[0, [0, 2]] *= width / calibration_width
        intrinsics[1, [1, 2]] *= height / calibration_height
        tensor = self.device_cfg.to_device(depth.astype(np.float32)[None] * .001)
        observation = CameraObservation(depth_image=tensor,
            rgb_image=torch.zeros((1, height, width, 3), device=self.device_cfg.device, dtype=torch.uint8),
            intrinsics=self.device_cfg.to_device(intrinsics[None]),
            pose=Pose.from_matrix(self.device_cfg.to_device(request.camera_pose_base_m)), depth_to_meter=1.)
        return PreparedRobotDepth(camera_observation=observation, filtered_depth_m=tensor,
            intrinsics=tuple(map(tuple, intrinsics)), removed_point_count=0)

    def crop_semantic_observation(self, prepared, table_model):
        from curobo.types import CameraObservation
        from tianji_curobo.mixed_esdf_scene import table_footprint_mask

        source = prepared.camera_observation
        depth = source.depth_image.detach().cpu().numpy().copy()
        points = source.get_pointcloud(project_to_pose=True).detach().cpu().numpy().reshape(-1, 3)
        depth.reshape(-1)[~table_footprint_mask(points, table_model)] = 0
        background = prepared.background_points_base_m
        background = np.ascontiguousarray(background[table_footprint_mask(background, table_model)])
        background.setflags(write=False)
        instances, samples = {}, {}
        for name, points in prepared.instance_points_base_m.items():
            keep = table_footprint_mask(points, table_model)
            instances[name] = np.ascontiguousarray(points[keep])
            instances[name].setflags(write=False)
            if name in (prepared.instance_depth_samples or {}):
                indices, values = prepared.instance_depth_samples[name]
                samples[name] = (indices[keep], values[keep])
        observation = CameraObservation(depth_image=source.depth_image.new_tensor(depth),
            intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.)
        return replace(prepared, camera_observation=observation,
            background_points_base_m=background, background_point_count=len(background),
            instance_points_base_m=MappingProxyType(instances), instance_depth_samples=MappingProxyType(samples))

    def separate_support_observation(self, view, plane):
        from curobo.types import CameraObservation
        from tianji_curobo.support_contact import support_mask

        source = view.camera_observation
        depth = source.depth_image.detach().cpu().numpy().copy()
        pixels = np.flatnonzero(depth.reshape(-1) > 0)
        support = support_mask(view.background_points_base_m, plane)
        if len(pixels) != len(support):
            raise ValueError("support classification is not aligned with background depth pixels")
        depth.reshape(-1)[pixels[support]] = 0
        background = np.ascontiguousarray(view.background_points_base_m[~support])
        background.setflags(write=False)
        observation = CameraObservation(depth_image=source.depth_image.new_tensor(depth),
            intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.)
        return replace(view, camera_observation=observation,
                       background_points_base_m=background, background_point_count=len(background))

    def stack_instance_observations(self, observations, instance_id):
        import torch
        from curobo.types import CameraObservation

        views = []
        for item in observations:
            source = item.camera_observation
            depth = torch.zeros_like(source.depth_image)
            samples = (item.instance_depth_samples or {}).get(instance_id)
            if samples is not None:
                indices, values = samples
                depth.reshape(-1)[torch.tensor(indices, device=depth.device)] = torch.tensor(
                    values, device=depth.device, dtype=depth.dtype)
            observation = CameraObservation(depth_image=depth, intrinsics=source.intrinsics,
                                            pose=source.pose, depth_to_meter=1.)
            views.append(replace(item, camera_observation=observation))
        return self.stack_prepared_observations(views)

    def stack_prepared_observations(self, observations):
        import torch
        from curobo.types import CameraObservation, Pose

        views = tuple(observations)
        if tuple(item.camera_id for item in views) not in (("3",), ("1", "3"), ("2", "3"), ("1", "2", "3")):
            raise ValueError("mapping requires an ordered camera subset containing head camera 3")
        shapes = {tuple(item.camera_observation.depth_image.shape[-2:]) for item in views}
        if len(shapes) != 1:
            raise ValueError("cuRobo multiview mapping requires equal left-depth shapes")
        height, width = next(iter(shapes))
        return CameraObservation(
            depth_image=torch.cat([item.camera_observation.depth_image for item in views], dim=0),
            rgb_image=torch.zeros((len(views), height, width, 3), device=self.device_cfg.device, dtype=torch.uint8),
            intrinsics=torch.stack([item.camera_observation.intrinsics.reshape(3, 3) for item in views]),
            pose=Pose.from_matrix(torch.stack([item.pose_matrix_tensor.reshape(4, 4) for item in views])),
            depth_to_meter=1.)
