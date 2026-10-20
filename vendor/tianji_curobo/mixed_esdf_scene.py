"""Build one cuRobo scene from an ESDF grid and named semantic geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import queue
import threading
from typing import Any

from .scene_config import SCENE_VOXEL_SIZE_M


FRONT_WORKSPACE_MINIMUM_M = (0.0, -0.6, 0.0)
FRONT_WORKSPACE_MAXIMUM_M = (1.1, 0.6, 1.6)
MAXIMUM_INSTANCE_POINTS = 20_000
MAXIMUM_MESH_VERTICES = 4_096
MAXIMUM_MESH_FACES = 8_192
CONVEX_HULL_TIMEOUT_S = 5.0
INSTANCE_VOXEL_SIZE_M = SCENE_VOXEL_SIZE_M


@dataclass(frozen=True)
class MappingBounds:
    center_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]


@dataclass(frozen=True)
class InstanceGeometry:
    name: str
    instance_id: str
    kind: str
    model: Mapping[str, Any]
    input_point_count: int
    filtered_point_count: int


@dataclass(frozen=True)
class TableCompletion:
    model: Mapping[str, Any]
    height_m: float
    normal_xyz: tuple[float, float, float]
    inlier_count: int
    robot_clearance_m: float
    support_plane: Mapping[str, Any] | None = None


def _numpy():
    import numpy as np

    return np


def table_footprint_mask(points, table_model):
    """Keep the table's XY column at or above the cuboid bottom."""

    np = _numpy()
    points = np.asarray(points)
    center = np.asarray(table_model["pose"][:2], dtype=np.float64)
    half_extent = np.asarray(table_model["dims"][:2], dtype=np.float64) * 0.5
    return (
        np.isfinite(points).all(axis=-1)
        & (points[..., :2] >= center - half_extent).all(axis=-1)
        & (points[..., :2] <= center + half_extent).all(axis=-1)
        & (points[..., 2] >= float(table_model["pose"][2]) - float(table_model["dims"][2]) * 0.5)
    )


def fixed_mapping_bounds() -> MappingBounds:
    """Return the fixed cuRobo volume covering only the robot-front workcell."""

    minimum = FRONT_WORKSPACE_MINIMUM_M
    maximum = FRONT_WORKSPACE_MAXIMUM_M
    center = tuple(
        0.5 * (minimum[index] + maximum[index]) for index in range(3)
    )
    extent = tuple(maximum[index] - minimum[index] for index in range(3))
    return MappingBounds(
        center_m=center,
        extent_m=extent,
        minimum_m=minimum,
        maximum_m=maximum,
    )


def _bounded_convex_hull(cloud, timeout_s: float):
    output: queue.Queue[object] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            output.put(cloud.compute_convex_hull(), block=False)
        except BaseException as error:
            output.put(error, block=False)

    worker = threading.Thread(
        target=run,
        name="semantic-instance-convex-hull",
        daemon=True,
    )
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        return None
    result = output.get_nowait()
    if isinstance(result, BaseException):
        return None
    return result


def _nearest_table_side(bounds_xy) -> str:
    x_min, x_max, y_min, y_max = bounds_xy
    candidates = {
        "x_min": abs(x_min),
        "x_max": abs(x_max),
        "y_min": abs(y_min),
        "y_max": abs(y_max),
    }
    return min(candidates, key=candidates.get)


def build_table_completion(
    point_groups,
    robot_spheres,
    *,
    open3d_runtime=None,
    robot_clearance_m: float = 0.02,
) -> TableCompletion:
    """Fit one horizontal Open3D plane and crop its robot-facing completion."""

    np = _numpy()
    groups = [np.asarray(value, dtype=np.float64).reshape(-1, 3) for value in point_groups]
    points = np.concatenate(groups, axis=0) if groups else np.empty((0, 3))
    points = points[np.isfinite(points).all(axis=1)]
    minimum = np.asarray(FRONT_WORKSPACE_MINIMUM_M)
    maximum = np.asarray(FRONT_WORKSPACE_MAXIMUM_M)
    points = points[np.all((points >= minimum) & (points <= maximum), axis=1)]
    if len(points) < 3:
        raise ValueError("tabletop_not_found: too few finite background points")
    if open3d_runtime is None:
        import open3d as open3d_runtime
    cloud = open3d_runtime.geometry.PointCloud()
    cloud.points = open3d_runtime.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(INSTANCE_VOXEL_SIZE_M)
    model, inliers = cloud.segment_plane(
        distance_threshold=0.015,
        ransac_n=3,
        num_iterations=2_000,
        probability=0.999,
    )
    normal = np.asarray(model[:3], dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 0.0:
        raise ValueError("tabletop_not_found: Open3D returned a degenerate plane")
    normal /= magnitude
    offset = float(model[3]) / magnitude
    if normal[2] < 0.0:
        normal = -normal
        offset = -offset
    if normal[2] < math.cos(math.radians(15.0)):
        raise ValueError("tabletop_not_found: plane is not horizontal")
    plane_points = np.asarray(cloud.points, dtype=np.float64)[
        np.asarray(inliers, dtype=np.int64)
    ]
    if len(plane_points) < 3:
        raise ValueError("tabletop_not_found: Open3D returned too few inliers")
    height = -offset / normal[2]
    x_min, y_min = plane_points[:, :2].min(axis=0)
    x_max, y_max = plane_points[:, :2].max(axis=0)
    expanded = [
        max(minimum[0], x_min - 0.05), min(maximum[0], x_max + 0.05),
        max(minimum[1], y_min - 0.05), min(maximum[1], y_max + 0.05),
    ]
    floor = height - 0.30
    top = height + 0.005
    clearance = float(robot_clearance_m)
    spheres = np.asarray(robot_spheres, dtype=np.float64).reshape(-1, 4)
    if (
        not np.isfinite(spheres).all()
        or np.any(spheres[:, 3] < 0.0)
        or not math.isfinite(clearance)
        or clearance < 0.02
    ):
        raise ValueError("full robot collision spheres or clearance are invalid")
    side = _nearest_table_side((x_min, x_max, y_min, y_max))
    # Crop the synthetic downward box. Only classified support rays may later
    # move into this bounded representation; residual observations stay in ESDF.
    for sx, sy, sz, radius in spheres:
        z_overlaps = sz + radius + clearance >= floor and sz - radius - clearance <= top
        if not z_overlaps:
            continue
        if side in {"x_min", "x_max"}:
            cross_overlaps = (
                sy + radius + clearance >= expanded[2]
                and sy - radius - clearance <= expanded[3]
            )
            x_overlaps = sx + radius + clearance >= expanded[0] and sx - radius - clearance <= expanded[1]
            if not cross_overlaps or not x_overlaps:
                continue
            if side == "x_min":
                expanded[0] = max(expanded[0], sx + radius + clearance)
            elif side == "x_max":
                expanded[1] = min(expanded[1], sx - radius - clearance)
        else:
            cross_overlaps = (
                sx + radius + clearance >= expanded[0]
                and sx - radius - clearance <= expanded[1]
            )
            y_overlaps = sy + radius + clearance >= expanded[2] and sy - radius - clearance <= expanded[3]
            if not cross_overlaps or not y_overlaps:
                continue
            if side == "y_min":
                expanded[2] = max(expanded[2], sy + radius + clearance)
            elif side == "y_max":
                expanded[3] = min(expanded[3], sy - radius - clearance)
    if expanded[0] >= expanded[1] or expanded[2] >= expanded[3]:
        raise ValueError("tabletop_not_found: robot-side crop removed the table")
    center = [
        0.5 * (expanded[0] + expanded[1]),
        0.5 * (expanded[2] + expanded[3]),
        0.5 * (floor + top),
    ]
    dims = [expanded[1] - expanded[0], expanded[3] - expanded[2], top - floor]
    table_model = {
            "pose": [*center, 1.0, 0.0, 0.0, 0.0],
            "dims": dims,
        }
    from .support_contact import build_support_plane
    support_plane = build_support_plane(points, normal, offset, table_model)
    return TableCompletion(
        model=table_model,
        height_m=float(height),
        normal_xyz=tuple(float(value) for value in normal),
        inlier_count=int(len(plane_points)),
        robot_clearance_m=clearance,
        support_plane=support_plane,
    )


def _rotation_to_quaternion_wxyz(rotation) -> list[float]:
    np = _numpy()
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return [float(value) for value in quaternion]


def _cuboid_model(cloud, points) -> dict[str, Any]:
    np = _numpy()
    try:
        box = cloud.get_oriented_bounding_box()
        center = np.asarray(box.center, dtype=np.float64)
        extent = np.asarray(box.extent, dtype=np.float64)
        rotation = np.asarray(box.R, dtype=np.float64)
    except (AttributeError, RuntimeError, ValueError):
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        extent = points.max(axis=0) - points.min(axis=0)
        rotation = np.eye(3)
    extent = np.maximum(extent, INSTANCE_VOXEL_SIZE_M)
    return {
        "pose": [*center.tolist(), *_rotation_to_quaternion_wxyz(rotation)],
        "dims": extent.tolist(),
    }


def build_instance_geometry(
    instance_id: str,
    points_by_camera: Mapping[str, Any],
    *,
    open3d_runtime=None,
    timeout_s: float = CONVEX_HULL_TIMEOUT_S,
) -> InstanceGeometry:
    """Create a bounded convex Mesh, falling back to one Open3D OBB."""

    np = _numpy()
    name = f"semantic_instance::{instance_id}"
    groups = []
    for camera_id in ("1", "2", "3"):
        if camera_id not in points_by_camera:
            continue
        points = np.asarray(points_by_camera[camera_id], dtype=np.float64).reshape(-1, 3)
        groups.append(points[np.isfinite(points).all(axis=1)])
    if not groups:
        raise ValueError(f"semantic instance {instance_id} has no visible surface")
    points = np.concatenate(groups, axis=0)
    input_count = int(len(points))
    if input_count == 0:
        raise ValueError(f"semantic instance {instance_id} has no finite surface")
    if len(points) > MAXIMUM_INSTANCE_POINTS:
        stride = math.ceil(len(points) / MAXIMUM_INSTANCE_POINTS)
        points = points[::stride][:MAXIMUM_INSTANCE_POINTS]
    if open3d_runtime is None:
        import open3d as open3d_runtime
    cloud = open3d_runtime.geometry.PointCloud()
    cloud.points = open3d_runtime.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(INSTANCE_VOXEL_SIZE_M)
    points = np.asarray(cloud.points, dtype=np.float64).reshape(-1, 3)
    if len(points) > MAXIMUM_INSTANCE_POINTS:
        points = points[:MAXIMUM_INSTANCE_POINTS]
        cloud.points = open3d_runtime.utility.Vector3dVector(points)
    hull_result = (
        _bounded_convex_hull(cloud, float(timeout_s))
        if len(points) >= 4 and np.linalg.matrix_rank(points - points.mean(axis=0)) >= 3
        else None
    )
    if hull_result is not None:
        hull = hull_result[0] if isinstance(hull_result, tuple) else hull_result
        vertices = np.asarray(hull.vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(hull.triangles, dtype=np.int64).reshape(-1, 3)
        if (
            4 <= len(vertices) <= MAXIMUM_MESH_VERTICES
            and 4 <= len(faces) <= MAXIMUM_MESH_FACES
            and np.isfinite(vertices).all()
        ):
            return InstanceGeometry(
                name=name,
                instance_id=instance_id,
                kind="mesh",
                model={
                    "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "vertices": vertices.tolist(),
                    "faces": faces.tolist(),
                },
                input_point_count=input_count,
                filtered_point_count=int(len(points)),
            )
    return InstanceGeometry(
        name=name,
        instance_id=instance_id,
        kind="cuboid",
        model=_cuboid_model(cloud, points),
        input_point_count=input_count,
        filtered_point_count=int(len(points)),
    )


def update_mixed_esdf_scene(motion_planner, scene):
    """Replace captured geometry, including new vertices under existing names."""

    if scene.mesh:
        motion_planner.scene_collision_checker.data.meshes.clear(clear_warp_cache=True)
    motion_planner.update_world(scene)
    if len(getattr(scene, "voxel", None) or []) > 1:
        # a35a708 SceneData.add_obstacle(VoxelGrid) replaces the voxel batch.
        # After the general world update, use its public typed batch loader so
        # every named layer is resident in the already-allocated GPU buffers.
        motion_planner.scene_collision_checker.data.voxels.load_batch(scene.voxel, env_idx=0)


def create_mixed_esdf_scene(scene_type, voxel_grid, scene_model: Mapping):
    if voxel_grid is None:
        raise ValueError("mixed ESDF scene requires a voxel grid")
    if not isinstance(scene_model, Mapping):
        raise TypeError("semantic scene model must be a mapping")
    # GPU voxel fields come from Mapper, not the JSON preview metadata.
    primitives = scene_type.create({key: value for key, value in scene_model.items() if key != "voxel"})
    return scene_type(
        cuboid=list(primitives.cuboid),
        mesh=list(primitives.mesh),
        voxel=list(voxel_grid) if isinstance(voxel_grid, (list, tuple)) else [voxel_grid],
    )


__all__ = [
    "FRONT_WORKSPACE_MAXIMUM_M",
    "FRONT_WORKSPACE_MINIMUM_M",
    "InstanceGeometry",
    "MappingBounds",
    "TableCompletion",
    "build_instance_geometry",
    "build_table_completion",
    "fixed_mapping_bounds",
    "create_mixed_esdf_scene",
    "update_mixed_esdf_scene",
]
