"""Independent native projection and rendered-depth calibration regressions."""

from copy import deepcopy

import mujoco
import numpy as np

from jev_mujoco.cameras import RgbdRenderer, deproject_view, install_cameras


def calibration():
    intrinsic = {"fx": 60., "fy": 55., "cx": 33.25, "cy": 47.125, "width": 96, "height": 80}
    return {"calibration_id": "pixel-center-regression", "cameras": {
        str(i): {"parent_body": "world", "T_parent_rgb_m": np.eye(4).tolist(), "intrinsics": dict(intrinsic)}
        for i in (1, 2, 3)}}


def test_native_frustum_pixel_center_matches_supplied_opencv_intrinsics():
    source = calibration()
    original = deepcopy(source)
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    install_cameras(spec, source)
    model = spec.compile()
    intrinsic = source["cameras"]["1"]["intrinsics"]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = model.camera("rgbd_1").id
    vertical, horizontal, clip = (np.zeros(2, np.float32) for _ in range(3))
    mujoco.mjv_cameraFrustum(vertical, horizontal, clip, model, camera)
    # Native frustum maps to viewport edge coordinates; array centers are +0.5.
    cx = intrinsic["width"] * horizontal[0] / horizontal.sum() - .5
    cy = intrinsic["height"] * vertical[0] / vertical.sum() - .5
    np.testing.assert_allclose([cx, cy], [intrinsic["cx"], intrinsic["cy"]], atol=2e-5, rtol=0)
    assert source == original


def test_rendered_slanted_plane_deprojects_at_integer_pixel_centers():
    source = calibration()
    normal = np.array([.2, -.1, -1.])
    quaternion = np.empty(4)
    mujoco.mju_quatZ2Vec(quaternion, normal / np.linalg.norm(normal))
    quat_text = " ".join(str(value) for value in quaternion)
    spec = mujoco.MjSpec.from_string(f'''<mujoco><visual><quality offsamples="0"/></visual>
      <worldbody><geom type="plane" pos="0 0 1" size="5 5 .1" quat="{quat_text}"/>
      </worldbody></mujoco>''')
    install_cameras(spec, source)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = RgbdRenderer(model, source)
    try:
        capture = renderer.capture(data, 1, {"A": {"joints_rad": [0.] * 7}, "B": {"joints_rad": [0.] * 7}})
    finally:
        renderer.close()
    for view in capture["views"].values():
        assert view["valid_depth"].all()
        points = deproject_view(view)
        # Geometry is an analytic offline calibration plane z = 1 + .2*x - .1*y.
        residual = points[:, 2] - 1 - .2 * points[:, 0] + .1 * points[:, 1]
        assert np.max(np.abs(residual)) < 2e-5
        assert view["intrinsics"] == source["cameras"][view["camera_id"]]["intrinsics"]
