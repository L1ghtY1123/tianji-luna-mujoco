"""Partition measured depth into semantic layers without implicit point filters.

The project applies its configured outlier filter before robot segmentation.
This adapter preserves every remaining valid return, including small isolated
surfaces, and keeps the native PreparedSemanticDepth/CameraObservation contract.
"""

from types import MappingProxyType

import numpy as np


def _numpy(value):
    return value.detach().cpu().numpy()


def _readonly(values, dtype):
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def partition_semantic_depth(prepared, projection, *, registered_mask_limits=None,
                             camera_observation_type=None):
    """Register RGB masks to the supplied depth returns and preserve pixel samples.

    ``prepared`` is the public PreparedFfsDepth produced after robot removal;
    ``projection`` is its validated SemanticProjectionRequest. No selected ID is
    special here: all named instances are retained, and their union is excluded
    only from the background layer. Optional depth-space boolean limits further
    restrict individual registered masks; their other valid returns stay in the
    background. Native scene assembly owns layer exclusion and global emptiness.
    """
    from tianji_curobo.robot_segmenter import PreparedSemanticDepth

    if camera_observation_type is None:
        from curobo.types import CameraObservation
        camera_observation_type = CameraObservation

    source = prepared.camera_observation
    image_shape = prepared.filtered_depth_m.shape[-2:]
    depth = _numpy(prepared.filtered_depth_m).reshape(image_shape)
    left_points = _numpy(source.get_pointcloud(project_to_pose=False)).reshape(*image_shape, 3)
    world_points = _numpy(prepared.base_pointcloud_m).reshape(*image_shape, 3)
    valid = (np.isfinite(depth) & (depth > 0)
             & np.isfinite(left_points).all(axis=-1) & np.isfinite(world_points).all(axis=-1))
    count = int(valid.sum())

    transform = np.asarray(projection.left_to_rgb, dtype=float)
    rgb_points = left_points @ transform[:3, :3].T + transform[:3, 3]
    intrinsics = np.asarray(projection.rgb_intrinsics, dtype=float)
    projectable = valid & np.isfinite(rgb_points).all(axis=-1) & (rgb_points[..., 2] > 0)
    columns, rows = np.zeros(image_shape, dtype=np.int64), np.zeros(image_shape, dtype=np.int64)
    pixels = rgb_points[projectable]
    columns[projectable] = np.rint(intrinsics[0, 0]*pixels[:, 0]/pixels[:, 2] + intrinsics[0, 2]).astype(np.int64)
    rows[projectable] = np.rint(intrinsics[1, 1]*pixels[:, 1]/pixels[:, 2] + intrinsics[1, 2]).astype(np.int64)
    width, height = projection.rgb_size
    inside = projectable & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    semantic_union = np.zeros(image_shape, dtype=bool)
    instance_points, instance_samples = {}, {}
    limits = registered_mask_limits or {}
    for identifier, rgb_mask in projection.instance_masks.items():
        selected = np.zeros(image_shape, dtype=bool)
        selected[inside] = np.asarray(rgb_mask)[rows[inside], columns[inside]] != 0
        if identifier in limits:
            limit = np.asarray(limits[identifier])
            if limit.shape != image_shape or limit.dtype != np.bool_:
                raise ValueError("registered mask limit must be a boolean image matching depth shape")
            selected &= limit
        indices = np.flatnonzero(selected)
        instance_points[identifier] = _readonly(world_points[selected], np.float32)
        instance_samples[identifier] = (_readonly(indices, np.int64),
                                        _readonly(depth.reshape(-1)[indices], np.float32))
        semantic_union |= selected

    background = valid & ~semantic_union
    background_depth = np.where(background, depth, 0).reshape(prepared.filtered_depth_m.shape)
    observation = camera_observation_type(
        depth_image=prepared.filtered_depth_m.new_tensor(background_depth),
        intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.)
    return PreparedSemanticDepth(
        camera_id=prepared.camera_id, stereo_sequence=prepared.stereo_sequence,
        camera_observation=observation, intrinsics_tensor=prepared.intrinsics_tensor,
        pose_matrix_tensor=prepared.pose_matrix_tensor,
        instance_points_base_m=MappingProxyType(instance_points),
        background_points_base_m=_readonly(world_points[background], np.float32),
        input_point_count=count, filtered_point_count=count, background_point_count=int(background.sum()),
        removed_point_count=prepared.removed_point_count,
        instance_depth_samples=MappingProxyType(instance_samples))
