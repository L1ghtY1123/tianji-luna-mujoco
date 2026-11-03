"""Measured camera identity, native MuJoCo projection, and optical pose regressions."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from luna_mujoco.cameras import camera_pose, install_cameras, load_calibration

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = [("1", "right_wrist"), ("2", "left_wrist"), ("3", "head")]


def measured_camera(role):
    path = ROOT / "resources/cameras/user-extrinsics" / f"camera_{role}_extrinsic.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("camera_id,role", CAMERAS)
def test_active_rgb_intrinsics_belong_to_the_measured_extrinsic_sensor(camera_id, role):
    path, measured = measured_camera(role)
    native = measured["samples"][0]["camera_calibration"]
    assert all(sample["camera_calibration"] == native for sample in measured["samples"])
    assert native["serial_number"] == measured["camera_serial_number"]
    definition = load_calibration()["cameras"][camera_id]
    expected = {**native["rgb_intrinsics"], "width": native["rgb_resolution"][0],
                "height": native["rgb_resolution"][1]}
    assert definition["intrinsics"] == expected
    assert definition["device_identity"]["serial_number"] == native["serial_number"]
    provenance = definition["intrinsic_provenance"]
    assert provenance["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["native_calibration"] == native
    assert definition["distortion"] == [0.] * 5  # The simulator emits ideal pinhole pixels.
    assert definition["projection_model"] == "ideal_pinhole"


@pytest.mark.parametrize("camera_id,role", CAMERAS)
def test_compiled_camera_frustum_projects_using_the_measured_rgb_matrix(camera_id, role):
    _, measured = measured_camera(role)
    native = measured["samples"][0]["camera_calibration"]
    intrinsic = native["rgb_intrinsics"]
    width, height = native["rgb_resolution"]
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="TCP_Link_L"/><body name="TCP_Link_R"/>
      </worldbody></mujoco>''')
    install_cameras(spec)
    model = spec.compile()
    index = model.camera("rgbd_" + camera_id).id
    np.testing.assert_array_equal(model.cam_resolution[index], [width, height])
    np.testing.assert_allclose(model.cam_intrinsic[index, :2],
                               [intrinsic["fx"], intrinsic["fy"]], rtol=0, atol=2e-5)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = index
    vertical, horizontal, clip = (np.zeros(2, np.float32) for _ in range(3))
    mujoco.mjv_cameraFrustum(vertical, horizontal, clip, model, camera)
    # Recover OpenCV pixel coordinates from public native frustum edges.
    for x, y, z in [(0., 0., 1.), (.2, -.15, .8), (-.3, .2, 1.2)]:
        pixel = [width * (x * clip[0] / z + horizontal[0]) / horizontal.sum() - .5,
                 height * (y * clip[0] / z + vertical[0]) / vertical.sum() - .5]
        expected = [intrinsic["fx"] * x / z + intrinsic["cx"],
                    intrinsic["fy"] * y / z + intrinsic["cy"]]
        np.testing.assert_allclose(pixel, expected, rtol=0, atol=6e-5)


@pytest.mark.parametrize("camera_id,role", CAMERAS)
def test_measured_optical_extrinsics_follow_flange_once_and_head_stays_world(camera_id, role):
    _, measured = measured_camera(role)
    optical = np.asarray(measured["extrinsic"], float)
    optical[:3, 3] *= .001
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="mount" pos="1 2 3" euler="0 0 90">
        <body name="TCP_Link_L" pos=".1 .2 .3"/>
        <body name="TCP_Link_R" pos="-.1 -.2 -.3"/>
      </body></worldbody></mujoco>''')
    install_cameras(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    parent = np.eye(4)
    if role != "head":
        parent[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        parent[:3, 3] = [0.8, 2.1, 3.3] if role == "left_wrist" else [1.2, 1.9, 2.7]
    np.testing.assert_allclose(camera_pose(model, data, camera_id), parent @ optical, atol=1e-8)
