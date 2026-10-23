"""Known simulation geometry, object-local surface cache and explicit tool frames."""

import hashlib
import json
from functools import lru_cache

import numpy as np

TOOL_PROFILE = {
    "revision": "marvin-rag-source-default-185mm-v1",
    "A": {"orientation_deg": 180, "tcp_ee_z_m": .185},
    "B": {"orientation_deg": 0, "tcp_ee_z_m": .185},
    "grasp_ee_z_m": .195,
    "gripper_config_id": "robotiq_2f_140",
}


def front_tcp_offset(arm):
    """Calibrated finger-front center in the SDK's TCP_Link mount frame (metres)."""
    if arm not in ('A', 'B'):
        raise ValueError('arm must be A or B')
    return np.array([0., 0., TOOL_PROFILE[arm]['tcp_ee_z_m']])


def front_tcp_pose(mount_pose, arm):
    result = rigid_matrix(mount_pose).copy()
    result[:3, 3] += result[:3, :3] @ front_tcp_offset(arm)
    return result


def rigid_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError("expected a finite rigid 4x4 transform")
    return matrix


def scene_object_rotation(item):
    """Authored scene pitch about local Y, followed by yaw about world Z."""
    yaw, pitch = np.radians([item['yaw_deg'], item.get('pitch_deg', 0.)])
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0., cp]])


def primitive_sdf(points, kind, size):
    """Exact signed distance, metres; negative inside known primitive geometry."""
    points = np.asarray(points, dtype=float)
    half = np.asarray(size, dtype=float) / 2
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError("size must have three positive finite extents")
    if kind in {"cube", "box"}:
        delta = np.abs(points) - half
        return np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(np.max(delta, axis=-1), 0)
    if kind == "sphere":
        if not np.allclose(half, half[0]):
            raise ValueError("sphere extents must match")
        return np.linalg.norm(points, axis=-1) - half[0]
    if kind == "cylinder":
        if not np.isclose(half[0], half[1]):
            raise ValueError("cylinder X/Y extents must match")
        delta = np.stack([np.linalg.norm(points[..., :2], axis=-1) - half[0],
                          np.abs(points[..., 2]) - half[2]], axis=-1)
        return np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(np.max(delta, axis=-1), 0)
    raise ValueError(f"unsupported geometry kind: {kind}")


@lru_cache(maxsize=128)
def _surface_cloud(kind, size, count=4096):
    """Sample once at scene-geometry boundary, never in GraspGen inference."""
    half = np.asarray(size) / 2
    rng = np.random.default_rng(42)
    if kind in {"cube", "box"}:
        points = rng.uniform(-half, half, (count, 3))
        areas = np.array([half[1]*half[2], half[0]*half[2], half[0]*half[1]])
        axes = rng.choice(3, count, p=areas / areas.sum())
        points[np.arange(count), axes] = half[axes] * rng.choice([-1, 1], count)
    elif kind == "sphere":
        points = rng.normal(size=(count, 3))
        points *= half[0] / np.linalg.norm(points, axis=1, keepdims=True)
    elif kind == "cylinder":
        angle = rng.uniform(0, 2*np.pi, count)
        radius = np.full(count, half[0])
        z = rng.uniform(-half[2], half[2], count)
        cap = rng.random(count) < half[0] / (half[0] + 2*half[2])
        radius[cap] *= np.sqrt(rng.random(np.count_nonzero(cap)))
        z[cap] = half[2] * rng.choice([-1, 1], np.count_nonzero(cap))
        points = np.column_stack([radius*np.cos(angle), radius*np.sin(angle), z])
    else:
        raise ValueError(f"unsupported geometry kind: {kind}")
    primitive_sdf(points[:1], kind, size)
    cloud = points.astype(np.float32)
    cloud.setflags(write=False)
    return cloud


class GeometryCache:
    def object_cloud(self, obj):
        return _surface_cloud(obj["kind"], tuple(obj["size"]))

    def reference(self, obj):
        cloud = self.object_cloud(obj)
        return {"id": hashlib.sha256(cloud.tobytes()).hexdigest(), "frame": f"object:{obj['id']}",
                "point_count": len(cloud), "source": "simulation_surface_geometry"}

    def clouds_in_object_frame(self, snapshot, target):
        object_points = self.object_cloud(target)
        target_inverse = np.linalg.inv(rigid_matrix(target["T_world_object"]))
        groups = []
        for obj in snapshot["objects"]:
            local = self.object_cloud(obj)
            transform = target_inverse @ rigid_matrix(obj["T_world_object"])
            groups.append(local @ transform[:3, :3].T + transform[:3, 3])
        table = snapshot["table"]
        table_points = _surface_cloud("box", tuple(table["size"])) + np.asarray(table["position"])
        groups.append(table_points @ target_inverse[:3, :3].T + target_inverse[:3, 3])
        return np.concatenate(groups).astype(np.float32), object_points


def grasp_targets(world_grasps, arm):
    """world_grasp @ grasp_ee @ inverse(tcp_ee), with arm-local 180 flip."""
    result = np.asarray(world_grasps, dtype=float).copy()
    if arm not in {"A", "B"}:
        raise ValueError("arm must be A or B")
    if TOOL_PROFILE[arm]["orientation_deg"] == 180:
        result[:, :3, :2] *= -1
    grasp_ee = np.eye(4)
    grasp_ee[2, 3] = TOOL_PROFILE["grasp_ee_z_m"]
    tcp_ee = np.eye(4)
    tcp_ee[2, 3] = TOOL_PROFILE[arm]["tcp_ee_z_m"]
    return result @ grasp_ee @ np.linalg.inv(tcp_ee)


def geometry_payload(snapshot, target_id):
    objects = snapshot["objects"]
    if target_id not in {obj["id"] for obj in objects}:
        raise ValueError("target does not exist in the scene")
    if len({obj["id"] for obj in objects}) != len(objects):
        raise ValueError("object identifiers must be unique")
    payload = {"generation": snapshot["generation"], "target_id": target_id,
               "objects": [{key: obj[key] for key in ("id", "kind", "size", "position", "quaternion")}
                           for obj in objects], "table": snapshot["table"], "voxel_size_m": .02}
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    payload["scene_token"] = "geometry:" + hashlib.sha256(encoded).hexdigest()
    return payload
