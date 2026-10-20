"""Registered RGBD sensor boundary using the public MuJoCo 3.10 API.

Only this adapter reads rendered physics poses. Downstream perception receives
pixels, RGB optical calibration, and a frozen copy of robot feedback.
"""

import json
import threading
import uuid
from copy import copy, deepcopy
from pathlib import Path

import mujoco
import numpy as np

CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "resources" / "cameras" / "calibration.json"
CAMERA_NAMES = {camera_id: "rgbd_" + camera_id for camera_id in ("1", "2", "3")}
# MuJoCo camera: X right, Y up, -Z forward. RGB optical: X right, Y down, +Z forward.
OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def load_calibration(path=CALIBRATION_PATH):
    with Path(path).open(encoding="utf-8") as source:
        calibration = json.load(source)
    if set(calibration["cameras"]) != {"1", "2", "3"}:
        raise ValueError("RGBD calibration must contain cameras 1, 2 and 3")
    for camera in calibration["cameras"].values():
        intrinsics = camera["intrinsics"]
        if any(not np.isfinite(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")):
            raise ValueError("RGB intrinsics must be finite")
        if min(intrinsics["fx"], intrinsics["fy"], intrinsics["width"], intrinsics["height"]) <= 0:
            raise ValueError("RGB focal lengths and dimensions must be positive")
        transform = np.asarray(camera["T_parent_rgb_m"], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("RGB optical pose must be a finite 4 by 4 matrix")
    return calibration


def install_cameras(spec, calibration=None):
    """Install cameras before compilation; no new bodies/joints or physics edits.

    Wrist extrinsics attach to the URDF flange bodies whose poses already include
    arm mounting and FK. The historical head matrix is already world/base_link.
    """
    calibration = load_calibration() if calibration is None else calibration
    for camera_id, definition in calibration["cameras"].items():
        parent_name = definition["parent_body"]
        parent = spec.worldbody if parent_name == "world" else spec.body(parent_name)
        if parent is None:
            raise ValueError(f"missing calibrated camera parent {parent_name}")
        transform = np.asarray(definition["T_parent_rgb_m"], dtype=float)
        rotation = transform[:3, :3] @ OPTICAL_TO_MUJOCO
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        intrinsics = definition["intrinsics"]
        width, height = intrinsics["width"], intrinsics["height"]
        # MuJoCo samples viewport pixels at index +0.5. Preserve the supplied
        # OpenCV convention where integer indices are pixel centers.
        parent.add_camera(name=CAMERA_NAMES[camera_id], pos=transform[:3, 3], quat=quaternion,
                          resolution=[width, height], sensor_size=[width, height],
                          focal_pixel=[intrinsics["fx"], intrinsics["fy"]],
                          principal_pixel=[width / 2 - intrinsics["cx"] - .5,
                                           height / 2 - intrinsics["cy"] - .5],
                          ipd=0)
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, width)
        spec.visual.global_.offheight = max(spec.visual.global_.offheight, height)


def clone_render_model(model):
    """Public model copy for a GL worker; never shares mutable rendering options."""
    return copy(model)


def camera_pose(model, data, camera_id):
    index = model.camera(CAMERA_NAMES[str(camera_id)]).id
    transform = np.eye(4)
    transform[:3, :3] = data.cam_xmat[index].reshape(3, 3) @ OPTICAL_TO_MUJOCO
    transform[:3, 3] = data.cam_xpos[index]
    return transform


def _immutable(array):
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


class RgbdRenderer:
    """One thread owns all Renderer/GL calls and their close lifecycle.

    The caller briefly copies physics data under its lock. This adapter makes
    another local copy once and renders all three RGB/depth pairs from that frame.
    """

    def __init__(self, model, calibration=None):
        self.model = model
        self.calibration = deepcopy(load_calibration() if calibration is None else calibration)
        self._data = mujoco.MjData(model)
        self._renderers = {}
        self._owner_thread = None
        self._closed = False
        self._options = mujoco.MjvOption()
        self._options.geomgroup[3] = 0  # Robot visual meshes, without duplicate collision hulls.
        for camera_id in self.calibration["cameras"]:
            model.camera(CAMERA_NAMES[camera_id])

    def _check_thread(self):
        current = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise RuntimeError("RGBD rendering and close must use the owning GL thread")

    def capture(self, data, generation, arms, *, rgb_only=False):
        self._check_thread()
        if self._closed:
            raise RuntimeError("RGBD renderer is closed")
        mujoco.mj_copyData(self._data, self.model, data)
        mujoco.mj_forward(self.model, self._data)
        capture_time = float(self._data.time)
        prefix = "preview" if rgb_only else "rgbd"
        observation_id = f"{prefix}-{generation}-{uuid.uuid4().hex}"
        telemetry = deepcopy(arms)
        joints = {arm: np.degrees(feedback["joints_rad"]).tolist() for arm, feedback in telemetry.items()}
        views = {}
        for camera_id, definition in self.calibration["cameras"].items():
            intrinsics = definition["intrinsics"]
            dimensions = (intrinsics["height"], intrinsics["width"])
            if dimensions not in self._renderers:
                self._renderers[dimensions] = mujoco.Renderer(self.model, height=dimensions[0],
                                                             width=dimensions[1])
            renderer = self._renderers[dimensions]
            renderer.disable_depth_rendering()
            renderer.update_scene(self._data, camera=CAMERA_NAMES[camera_id], scene_option=self._options)
            rgb = renderer.render().copy()
            if rgb_only:
                views[camera_id] = {"rgb": _immutable(rgb), "frame_id": observation_id}
                continue
            renderer.enable_depth_rendering()
            try:
                depth = renderer.render().copy()
            finally:
                renderer.disable_depth_rendering()
            far_m = self.model.vis.map.zfar * self.model.stat.extent
            valid = np.isfinite(depth) & (depth > 0) & (depth < far_m * .9999)
            depth[~valid] = 0
            views[camera_id] = {
                "camera_id": camera_id, "frame_id": observation_id, "simulation_generation": generation,
                "capture_time": capture_time, "calibration_id": self.calibration["calibration_id"],
                "rgb": _immutable(rgb), "depth_m": _immutable(depth), "valid_depth": _immutable(valid),
                "intrinsics": deepcopy(intrinsics),
                "T_world_camera": camera_pose(self.model, self._data, camera_id).tolist(),
                "robot_joints_deg": deepcopy(joints), "registered_to": "rgb",
            }
        if rgb_only:
            return {"preview_frame_id": observation_id, "generation": generation, "time": capture_time,
                    "views": views, "source": "rgb_preview"}
        return {"observation_id": observation_id, "generation": generation, "time": capture_time,
                "arms": telemetry, "views": views, "source": "rgbd"}

    def close(self):
        self._check_thread()
        if self._closed:
            return
        self._closed = True
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


def letterbox_view(view, width=640, height=480):
    """Pad registered RGBD and masks without scaling or inventing depth samples."""
    source_height, source_width = view["depth_m"].shape
    if source_width > width or source_height > height:
        raise ValueError("registered view is larger than the mapper resolution")
    top, left = (height - source_height) // 2, (width - source_width) // 2
    result = dict(view)
    result["intrinsics"] = {**view["intrinsics"], "cx": view["intrinsics"]["cx"] + left,
                            "cy": view["intrinsics"]["cy"] + top, "width": width, "height": height}
    for key in ("rgb", "depth_m", "valid_depth"):
        source = np.asarray(view[key])
        shape = (height, width) + source.shape[2:]
        padded = np.zeros(shape, dtype=source.dtype)
        padded[top:top + source_height, left:left + source_width] = source
        result[key] = _immutable(padded)
    result["instance_masks"] = {}
    for instance_id, mask in view.get("instance_masks", {}).items():
        padded = np.zeros((height, width), dtype=bool)
        padded[top:top + source_height, left:left + source_width] = mask
        result["instance_masks"][instance_id] = _immutable(padded)
    return result


def deproject_view(view, mask=None):
    """Visible measured points in world metres; never generates hidden surfaces."""
    depth = np.asarray(view["depth_m"])
    valid = np.asarray(view["valid_depth"], dtype=bool) & np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    intrinsics = view["intrinsics"]
    points = np.column_stack(((columns - intrinsics["cx"]) * z / intrinsics["fx"],
                              (rows - intrinsics["cy"]) * z / intrinsics["fy"], z))
    transform = np.asarray(view["T_world_camera"], dtype=float)
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
