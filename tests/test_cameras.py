"""Calibrated public MuJoCo rendering, with independent projection expectations."""

import importlib
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest


def cameras_module():
    assert importlib.util.find_spec("luna_mujoco.cameras") is not None, "RGBD adapter is missing"
    return importlib.import_module("luna_mujoco.cameras")


def small_calibration():
    intrinsics = {"fx": 60.0, "fy": 55.0, "cx": 33.0, "cy": 47.0, "width": 96, "height": 80}
    return {"calibration_id": "projection-test", "cameras": {
        str(index): {"parent_body": "world", "T_parent_rgb_m": np.eye(4).tolist(),
                     "intrinsics": dict(intrinsics)} for index in (1, 2, 3)}}


def test_user_extrinsics_keep_rgb_intrinsics_and_convert_translation_once():
    calibration = cameras_module().load_calibration()
    right = calibration["cameras"]["1"]
    assert right["intrinsics"]["cx"] == 317.9415584415584
    np.testing.assert_allclose(np.asarray(right["T_parent_rgb_m"])[:3, 3],
                               [.00211961167771889, .09119845517638979, .03725148394276182])
    assert right["T_parent_rgb_m"][0][0] == -.9999617995060496
    assert right["source_sha256"] == "e8cd47a5d902ade17d045fbaad7e008a7a173fb4695208b7f8d0bae07604ce8d"
    head = calibration["cameras"]["3"]
    assert head["parent_body"] == "world"
    assert head["T_parent_rgb_m"][2][3] == pytest.approx(1.199981040227405)


def test_wrist_uses_captured_fk_and_head_is_already_in_world():
    cameras = cameras_module()
    calibration = cameras.load_calibration()
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="mount" pos="1 2 3" euler="0 0 90">
       <body name="TCP_Link_R" pos=".1 .2 .3"><joint name="wrist" axis="0 0 1"/>
        <geom type="sphere" size=".01" mass=".1"/>
       </body>
       <body name="TCP_Link_L" pos=".4 .5 .6"/>
      </body></worldbody></mujoco>''')
    cameras.install_cameras(spec, calibration)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[0] = np.pi / 2
    mujoco.mj_forward(model, data)
    pose = cameras.camera_pose(model, data, "1")
    flange = np.eye(4)
    flange[:3, :3] = data.body("TCP_Link_R").xmat.reshape(3, 3)
    flange[:3, 3] = [0.8, 2.1, 3.3]
    np.testing.assert_allclose(pose, flange @ np.asarray(calibration["cameras"]["1"]["T_parent_rgb_m"]),
                               atol=1e-7)
    np.testing.assert_allclose(cameras.camera_pose(model, data, "3"),
                               calibration["cameras"]["3"]["T_parent_rgb_m"], atol=1e-7)


def test_real_renderer_off_center_rgb_and_metric_depth_are_aligned_and_immutable():
    cameras = cameras_module()
    calibration = small_calibration()
    spec = mujoco.MjSpec.from_string('''<mujoco><visual><headlight ambient="1 1 1" diffuse="0 0 0"/>
      </visual><worldbody><geom type="sphere" size=".04" pos=".1 .08 1" rgba="1 0 0 1"/>
      </worldbody></mujoco>''')
    cameras.install_cameras(spec, calibration)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.time = 12.5
    mujoco.mj_forward(model, data)
    renderer = cameras.RgbdRenderer(cameras.clone_render_model(model), calibration)
    arms = {"A": {"joints_rad": [0.0] * 7}, "B": {"joints_rad": [0.1] * 7}}
    try:
        capture = renderer.capture(data, 7, arms)
        assert capture["generation"] == 7 and capture["time"] == 12.5
        assert set(capture["views"]) == {"1", "2", "3"}
        for view in capture["views"].values():
            red = (view["rgb"][:, :, 0] > 200) & (view["rgb"][:, :, 1] < 20)
            rows, columns = np.nonzero(red)
            assert columns.mean() == pytest.approx(39.0, abs=1.0)
            assert rows.mean() == pytest.approx(51.4, abs=1.0)
            assert view["valid_depth"][red].all()
            assert view["depth_m"][red].min() == pytest.approx(0.96, abs=.015)
            assert not view["valid_depth"][0, 0]
            assert view["frame_id"] == capture["observation_id"]
            assert view["capture_time"] == capture["time"]
            assert view["simulation_generation"] == capture["generation"]
            assert not view["rgb"].flags.writeable and not view["depth_m"].flags.writeable
        repeated = renderer.capture(data, 7, arms)
        for camera_id in ("1", "2", "3"):
            np.testing.assert_array_equal(repeated["views"][camera_id]["rgb"], capture["views"][camera_id]["rgb"])
            np.testing.assert_array_equal(repeated["views"][camera_id]["depth_m"], capture["views"][camera_id]["depth_m"])
        arms["A"]["joints_rad"][0] = 9.0
        assert capture["arms"]["A"]["joints_rad"][0] == 0.0
        with ThreadPoolExecutor() as executor, pytest.raises(RuntimeError, match="thread"):
            executor.submit(renderer.capture, data, 7, arms).result()
    finally:
        renderer.close()


def test_head_letterbox_preserves_invalid_padding_and_changes_principal_point():
    cameras = cameras_module()
    view = {"rgb": np.full((360, 640, 3), 20, np.uint8), "depth_m": np.ones((360, 640), np.float32),
            "valid_depth": np.ones((360, 640), bool), "instance_masks": {"detector-1": np.ones((360, 640), bool)},
            "intrinsics": {"fx": 291.8918918918919, "fy": 291.8918918918919,
                           "cx": 320.0405405405405, "cy": 194.6351351351351, "width": 640, "height": 360},
            "T_world_camera": np.eye(4).tolist()}
    boxed = cameras.letterbox_view(view)
    assert boxed["depth_m"].shape == (480, 640)
    assert boxed["valid_depth"].sum() == 360 * 640
    assert not boxed["instance_masks"]["detector-1"][:60].any()
    assert boxed["intrinsics"]["cy"] == pytest.approx(254.6351351351351)
    assert boxed["intrinsics"]["fy"] == view["intrinsics"]["fy"]
    assert len(cameras.deproject_view(boxed)) == 360 * 640


def test_deprojection_uses_registered_rgb_intrinsics_and_valid_depth_only():
    cameras = cameras_module()
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    view = {"depth_m": np.array([[2., 0., np.nan]], np.float32),
            "valid_depth": np.array([[True, False, True]]),
            "intrinsics": {"fx": 4., "fy": 5., "cx": 1., "cy": 0., "width": 3, "height": 1},
            "T_world_camera": transform.tolist()}
    np.testing.assert_allclose(cameras.deproject_view(view), [[.5, 2., 5.]])


def test_hidden_object_motion_cannot_change_three_camera_depth_or_deprojected_cloud():
    """Real local renderer: move hidden geometry, then expose it without touching sensor code."""
    cameras = cameras_module()
    calibration = small_calibration()
    for camera, x in [('1', -.15), ('2', .15), ('3', 0.)]:
        calibration['cameras'][camera]['T_parent_rgb_m'][0][3] = x
    spec = mujoco.MjSpec.from_string('''<mujoco><statistic extent="4" center="0 0 1"/>
      <visual><quality offsamples="0"/></visual><worldbody>
      <body name="wall" mocap="true" pos="0 0 1"><geom type="box" size=".4 .4 .04"/></body>
      <body name="hidden" mocap="true" pos="0 0 1.6"><geom type="box" size=".08 .08 .08"/></body>
      </worldbody></mujoco>''')
    cameras.install_cameras(spec, calibration)
    model = spec.compile()
    data = mujoco.MjData(model)
    renderer = cameras.RgbdRenderer(cameras.clone_render_model(model), calibration)
    captures = []
    try:
        for object_x, wall_x in [(-.06, 0.), (.06, 0.), (.06, 1.5)]:
            data.mocap_pos[model.body('hidden').mocapid[0], 0] = object_x
            data.mocap_pos[model.body('wall').mocapid[0], 0] = wall_x
            mujoco.mj_forward(model, data)
            captures.append(renderer.capture(data, 1, {}))
    finally:
        renderer.close()
    for camera in ('1', '2', '3'):
        first, moved, revealed = [capture['views'][camera] for capture in captures]
        np.testing.assert_array_equal(first['depth_m'], moved['depth_m'])
        np.testing.assert_array_equal(cameras.deproject_view(first), cameras.deproject_view(moved))
        assert cameras.deproject_view(first)[:, 2].max() < 1.1
        assert np.count_nonzero(cameras.deproject_view(revealed)[:, 2] > 1.4) > 0
