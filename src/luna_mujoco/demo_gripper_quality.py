"""Static rigid-width grasp prediction, separate from actual carrying checks."""

import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from .carrying import carrying_touch_links
from .geometry import rigid_matrix
from .kinematics import robot_model

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf"
CONFIG = ROOT / "resources/curobo/marvin_gui_dual.yml"
MAX_Q_STEP_RAD = .01
WIDTH_RESIDUAL_M = 1e-8


def _origin(node):
    origin = node.find("origin")
    result = np.eye(4)
    if origin is not None:
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
            raise ValueError("invalid gripper mesh origin")
        result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        result[:3, 3] = xyz
    return result


def gripper_width_model(arm):
    """Cache only immutable own-model geometry, identified by content hashes."""
    if arm not in ("A", "B"):
        raise ValueError("gripper width prediction requires arm A or B")
    robot = ET.fromstring(URDF.read_bytes())
    prefix = "gripL_" if arm == "A" else "gripR_"
    assets = [URDF, CONFIG]
    for side in ("L", "R"):
        node = robot.find(f"link[@name='{prefix}Link3_{side}']/collision/geometry/mesh")
        if node is None:
            raise ValueError("opposing Link3 collision meshes are required")
        assets.append((URDF.parent / node.get("filename")).resolve())
    digest = hashlib.sha256()
    for path in assets:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return _cached_model(arm, digest.hexdigest())


@lru_cache(maxsize=2)
def _cached_model(arm, fingerprint):
    return GripperWidthModel(arm, fingerprint)


class GripperWidthModel:
    """Authored parallel-pad planes and all noncontact gripper spheres in TCP."""

    def __init__(self, arm, fingerprint):
        self.arm, self.fingerprint = arm, fingerprint
        self.prefix = "gripL_" if arm == "A" else "gripR_"
        self.tool = "TCP_Link_L" if arm == "A" else "TCP_Link_R"
        self.pads = tuple(self.prefix + "Link3_" + side for side in ("L", "R"))
        self.touch_links = carrying_touch_links(arm)
        self.driver = self.prefix + "finger_joint"
        robot = ET.parse(URDF).getroot()
        self.joints = {joint.get("name"): joint for joint in robot.findall("joint")}
        kin = yaml.safe_load(CONFIG.read_bytes())["robot_cfg"]["kinematics"]
        self.locks = dict(kin["lock_joints"])
        self.model = robot_model()
        limit = self.joints[self.driver].find("limit")
        self.lower, self.upper = float(limit.get("lower")), float(limit.get("upper"))
        if not np.isfinite([self.lower, self.upper]).all() or not self.lower < self.upper:
            raise ValueError("invalid authored gripper range")
        if not {*self.touch_links, self.tool}.issubset(kin["collision_link_names"]):
            raise ValueError("noncontact gripper model is incomplete")
        links = [name for name in kin["collision_link_names"]
                 if (name.startswith(self.prefix) or name == self.tool) and name not in self.touch_links]
        self.spheres, self.labels = {}, []
        for link in links:
            rows = np.array([[*s["center"], s["radius"]] for s in kin["collision_spheres"][link]], dtype=float)
            if rows.ndim != 2 or rows.shape[1] != 4 or not len(rows) or not np.isfinite(rows).all() or np.any(rows[:, 3] <= 0):
                raise ValueError("invalid own gripper self-collision spheres")
            buffer = kin.get("collision_sphere_buffer", 0.)
            rows[:, 3] += float(buffer.get(link, 0.) if isinstance(buffer, dict) else buffer)
            rows[:, 3] += float(kin.get("self_collision_buffer", {}).get(link, 0.))
            if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all() or np.any(rows[:, 3] <= 0):
                raise ValueError("invalid own gripper self-collision spheres")
            self.spheres[link] = rows
            self.labels.extend((link, i) for i in range(len(rows)))
        opened = self.transforms(self.lower)
        meshes = {}
        for link in self.pads:
            node = robot.find(f"link[@name='{link}']/collision")
            mesh_node = node.find("geometry/mesh")
            mesh = trimesh.load_mesh(URDF.parent / mesh_node.get("filename"), process=True)
            scale = np.fromstring(mesh_node.get("scale", "1 1 1"), sep=" ")
            if not isinstance(mesh, trimesh.Trimesh) or scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("invalid opposing pad mesh")
            mesh.apply_scale(scale)
            mesh.apply_transform(_origin(node))
            meshes[link] = mesh
        centers = [opened[link][:3, :3] @ meshes[link].centroid + opened[link][:3, 3] for link in self.pads]
        axis = centers[1] - centers[0]
        if np.linalg.norm(axis) < 1e-9:
            raise ValueError("opposing pad direction is undefined")
        axis /= np.linalg.norm(axis)
        self.planes = {}
        for link, direction in zip(self.pads, (axis, -axis)):
            mesh, transform = meshes[link], opened[link]
            faces = np.flatnonzero((mesh.face_normals @ transform[:3, :3].T) @ direction > .99)
            groups = defaultdict(list)
            for i in faces:
                normal = mesh.face_normals[i]
                groups[(*np.round(normal, 4), round(float(normal @ mesh.triangles_center[i]), 5))].append(i)
            if not groups:
                raise ValueError("no opposing planar Link3 pad surface")
            ids = max(groups.values(), key=lambda group: float(mesh.area_faces[group].sum()))
            areas = mesh.area_faces[ids]
            normal = np.average(mesh.face_normals[ids], axis=0, weights=areas)
            normal /= np.linalg.norm(normal)
            center = np.average(mesh.triangles_center[ids], axis=0, weights=areas)
            if np.max(np.abs((mesh.triangles[ids].reshape(-1, 3) - center) @ normal)) > 1e-5:
                raise ValueError("opposing pad surface is not planar")
            self.planes[link] = (center, normal, float(areas.sum()))
        self.normal = opened[self.pads[0]][:3, :3] @ self.planes[self.pads[0]][1]
        self.normal /= np.linalg.norm(self.normal)
        self.angles = np.linspace(self.lower, self.upper, int(np.ceil((self.upper - self.lower) / MAX_Q_STEP_RAD)) + 1)
        self.gaps = np.array([self.jaw_gap(q) for q in self.angles])
        if not np.isfinite(self.gaps).all() or np.any(np.diff(self.gaps) >= 0) or self.gaps[0] <= 0:
            raise ValueError("authored pad gap is not strictly decreasing")

    def transforms(self, q):
        data = mujoco.MjData(self.model)
        values = {**self.locks, self.driver: float(q)}

        def value(name, visiting=()):
            if name in visiting:
                raise ValueError("cyclic gripper mimic model")
            mimic = self.joints[name].find("mimic")
            if mimic is not None:
                return value(mimic.get("joint"), (*visiting, name)) * float(mimic.get("multiplier", 1)) + float(mimic.get("offset", 0))
            return values.get(name, 0.)

        for name, joint in self.joints.items():
            if joint.get("type") != "fixed" and (name in values or joint.find("mimic") is not None):
                data.qpos[int(self.model.joint(name).qposadr[0])] = value(name)
        mujoco.mj_fwdPosition(self.model, data)
        tool = self.model.body(self.tool).id
        rotation = data.xmat[tool].reshape(3, 3)
        result = {}
        for link in (*self.pads, *self.spheres):
            body = self.model.body(link).id
            transform = np.eye(4)
            transform[:3, :3] = rotation.T @ data.xmat[body].reshape(3, 3)
            transform[:3, 3] = rotation.T @ (data.xpos[body] - data.xpos[tool])
            result[link] = transform
        return result

    def jaw_gap(self, q):
        poses, offsets = self.transforms(q), []
        for index, link in enumerate(self.pads):
            center, normal, _ = self.planes[link]
            pose = poses[link]
            if (pose[:3, :3] @ normal) @ self.normal * (1 if index == 0 else -1) < 1 - 1e-5:
                raise ValueError("authored pad normals are not parallel throughout closure")
            offsets.append(float((pose[:3, :3] @ center + pose[:3, 3]) @ self.normal))
        return offsets[1] - offsets[0]

    def width_angle(self, width):
        if width > self.gaps[0] or width < self.gaps[-1] or width <= 0:
            return None
        high = min(max(1, int(np.searchsorted(-self.gaps, -width))), len(self.angles) - 1)
        low_q, high_q = self.angles[high - 1:high + 1]
        bracket = [float(low_q), float(high_q)]
        # A fixed 24-step bisection bounds work; only the final q is collision checked.
        for _ in range(24):
            q = float((low_q + high_q) / 2)
            residual = self.jaw_gap(q) - width
            if abs(residual) <= WIDTH_RESIDUAL_M:
                return q, residual, bracket
            if residual > 0:
                low_q = q
            else:
                high_q = q
        raise ValueError("predicted pad width root did not converge")

    def noncontact_spheres(self, q):
        poses = self.transforms(q)
        return np.concatenate([np.column_stack((rows[:, :3] @ poses[link][:3, :3].T + poses[link][:3, 3], rows[:, 3]))
                               for link, rows in self.spheres.items()])


def predict_gripper_compatibility(points_world_m, T_world_tcp, *, arm, uncertainty_m, margin_m):
    """No cloud translation, point deletion, fitted envelope, GT or closed feedback."""
    points = np.asarray(points_world_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("gripper width prediction requires finite full Nx3 geometry")
    if not np.isfinite([uncertainty_m, margin_m]).all() or min(uncertainty_m, margin_m) < 0:
        raise ValueError("gripper width prediction requires original nonnegative padding")
    pose = rigid_matrix(T_world_tcp)
    model = gripper_width_model(arm)
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    projection = local @ model.normal
    width = float(np.ptp(projection))
    padding = float(uncertainty_m + margin_m)
    evidence = {"source": "predicted_centered_jaw_width_on_frozen_cloud", "is_measurement": False,
        "cloud_translated": False, "continuous_sweep_checked": False, "arm": arm, "tool_frame": model.tool,
        "model_fingerprint": model.fingerprint,
        "model_source": "own_urdf_collision_stl_and_self_effective_spheres",
        "point_count": len(points),
        "points_world_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "uncertainty_m": float(uncertainty_m), "margin_m": float(margin_m), "padding_m": padding,
        "cloud_width_m": width, "closing_normal_tool": model.normal.tolist(),
        "cloud_axis_min_max_m": [float(projection.min()), float(projection.max())],
        "contact_policy": "owner_hand_assembly_v1",
        "contact_exemptions": list(model.touch_links), "checked_links": list(model.spheres),
        "q_width_rad": None, "minimum_padded_noncontact_gap_m": None}
    match = model.width_angle(width)
    if match is None:
        # A CAD part can be gripped at a narrower local section. Its whole-cloud
        # extent is not proof that the contact section cannot enter the jaws.
        # The retained noncontact geometry is the fixed TCP/camera assembly, so
        # its full-cloud collision check does not need an invented closure q.
        if set(model.spheres) != {model.tool}:
            raise ValueError("partial grasp requires closure-independent noncontact geometry")
        q, residual, bracket = model.lower, None, None
        evidence['width_prediction_status'] = 'whole_cloud_width_outside_range'
    else:
        q, residual, bracket = match
        evidence['width_prediction_status'] = 'whole_cloud_width_estimate_only'
    spheres = model.noncontact_spheres(q)
    minimum, point_index, sphere_index = float("inf"), None, None
    for start in range(0, len(local), 2048):
        gaps = np.linalg.norm(local[start:start + 2048, None] - spheres[None, :, :3], axis=2) - spheres[None, :, 3] - padding
        pi, si = np.unravel_index(np.argmin(gaps), gaps.shape)
        if gaps[pi, si] < minimum:
            minimum, point_index, sphere_index = float(gaps[pi, si]), start + int(pi), int(si)
    link, link_index = model.labels[sphere_index]
    return {**evidence, "q_width_rad": q if match is not None else None,
        "noncontact_check_q_rad": q, "width_residual_m": residual, "q_bracket_rad": bracket,
        "minimum_padded_noncontact_gap_m": minimum, "point_index": point_index,
        "link": link, "sphere_index": link_index, "accepted": minimum >= 0,
        "reason": "accepted" if minimum >= 0 else "predicted_noncontact_geometry_conflict"}
