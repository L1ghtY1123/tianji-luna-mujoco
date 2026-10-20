"""Color existing world-space preview centers from frozen, visibility-tested RGB.

This display-only projection never changes depth, TSDF or collision distances.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


FALLBACK_RGB = (111, 135, 155)


def _pixels(points, intrinsics, width, height):
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 0)
    u = np.zeros(len(points), dtype=np.int64)
    v = np.zeros(len(points), dtype=np.int64)
    indices = np.flatnonzero(valid)
    u[indices] = np.rint(intrinsics[0, 0] * points[indices, 0] / z[indices] + intrinsics[0, 2]).astype(np.int64)
    v[indices] = np.rint(intrinsics[1, 1] * points[indices, 1] / z[indices] + intrinsics[1, 2]).astype(np.int64)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


class SurfaceColorizer:
    def __init__(self, views, *, voxel_size_m):
        self.views = []
        self.tolerance_m = 2 * voxel_size_m
        for camera_id in sorted(views):
            view = views[camera_id]
            encoded = view.get("rgb_jpeg_base64")
            if encoded is None:
                continue  # Older frozen maps have no RGB observation.
            if not isinstance(encoded, str) or len(encoded) > 45_000_000:
                raise ValueError("frozen RGB exceeds the image size limit")
            try:
                raw = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if list(image.size) != list(view["rgb_size"]) or image.width * image.height > 4_194_304:
                        raise ValueError("frozen RGB size differs from calibration")
                    rgb = np.asarray(image.convert("RGB"))
            except (OSError, ValueError) as error:
                raise ValueError("invalid frozen RGB image") from error
            depth = np.asarray(view["depth_left_m"])
            valid_depth = np.asarray(view["valid_left_mask"], dtype=bool) & np.isfinite(depth) & (depth > 0)
            left_k = np.asarray(view["left_intrinsics"])
            rgb_k = np.asarray(view["rgb_intrinsics"])
            transform = np.asarray(view["left_to_rgb"])
            pose = np.asarray(view["T_base_left_camera"])
            # A z-buffer prevents a left-visible point from sampling RGB behind
            # a nearer surface seen by the offset RGB camera.
            y, x = np.nonzero(valid_depth)
            z = depth[y, x]
            left_points = np.column_stack(((x-left_k[0, 2])*z/left_k[0, 0],
                                          (y-left_k[1, 2])*z/left_k[1, 1], z))
            rgb_points = left_points @ transform[:3, :3].T + transform[:3, 3]
            height, width = rgb.shape[:2]
            u, v, visible = _pixels(rgb_points, rgb_k, width, height)
            rgb_depth = np.full((height, width), np.inf, dtype=np.float32)
            np.minimum.at(rgb_depth, (v[visible], u[visible]), rgb_points[visible, 2])
            masks = {key: np.asarray(mask, dtype=bool) for key, mask in view["instance_masks"].items()}
            background = (~np.logical_or.reduce(list(masks.values()))
                          if masks else np.ones((height, width), dtype=bool))
            self.views.append((rgb, depth, valid_depth, left_k, rgb_k, transform, pose,
                               rgb_depth, masks, background))

    def colors(self, centers_base_m, instance_id=None):
        points = np.asarray(centers_base_m, dtype=np.float64).reshape(-1, 3)
        colors = np.tile(np.asarray(FALLBACK_RGB, dtype=np.uint8), (len(points), 1))
        best_error = np.full(len(points), np.inf)
        for rgb, depth, valid_depth, left_k, rgb_k, transform, pose, rgb_depth, masks, background in self.views:
            mask = background if instance_id is None else masks.get(instance_id)
            if mask is None:
                continue
            left_points = (points - pose[:3, 3]) @ pose[:3, :3]
            u, v, valid = _pixels(left_points, left_k, depth.shape[1], depth.shape[0])
            indices = np.flatnonzero(valid)
            errors = np.full(len(points), np.inf)
            errors[indices] = np.abs(depth[v[indices], u[indices]] - left_points[indices, 2])
            valid[indices] &= valid_depth[v[indices], u[indices]]
            valid &= errors <= self.tolerance_m
            rgb_points = left_points @ transform[:3, :3].T + transform[:3, 3]
            u, v, in_rgb = _pixels(rgb_points, rgb_k, rgb.shape[1], rgb.shape[0])
            indices = np.flatnonzero(valid & in_rgb & (errors < best_error))
            visible = (rgb_points[indices, 2] <= rgb_depth[v[indices], u[indices]] + self.tolerance_m)
            visible &= mask[v[indices], u[indices]]
            indices = indices[visible]
            colors[indices] = rgb[v[indices], u[indices]]
            best_error[indices] = errors[indices]
        return colors
