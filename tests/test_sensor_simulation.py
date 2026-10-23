import numpy as np

from luna_mujoco.simulation import Simulation


def test_robot_feedback_exposes_only_real_robot_measurements():
    sim = Simulation()
    try:
        state = sim.robot_snapshot()
        assert "objects" not in state and "table" not in state and "geometry_poses" not in state
        assert set(state["arms"]) == {"A", "B"}
        for feedback in state["arms"].values():
            assert len(feedback["joints_rad"]) == 7
            assert np.asarray(feedback["tcp"]).shape == (4, 4)
            assert np.isfinite(feedback["gripper_position_rad"])
    finally:
        sim.close()


def test_three_rgbd_capture_remains_pixel_only_and_physics_continues():
    sim = Simulation()
    sim.start()
    try:
        capture = sim.capture_rgbd()
        assert set(capture["views"]) == {"1", "2", "3"}
        assert "objects" not in capture and "table" not in capture
        for view in capture["views"].values():
            assert view["frame_id"] == capture["observation_id"]
            assert view["capture_time"] == capture["time"]
            assert view["simulation_generation"] == capture["generation"]
            assert view["rgb"].shape[:2] == view["depth_m"].shape
        assert sim.robot_snapshot()["time"] > capture["time"]
    finally:
        sim.close()


def test_camera_lighting_keeps_red_object_bright_and_saturated_in_preview_and_task_rgb():
    sim = Simulation()
    sim.start()
    try:
        sim.settle()
        for capture in (sim.preview_rgb(), sim.capture_rgbd()):
            rgb = capture["views"]["3"]["rgb"].astype(float)
            red = (rgb[:, :, 0] > 3 * rgb[:, :, 1]) & (rgb[:, :, 0] > 3 * rgb[:, :, 2]) & (rgb[:, :, 0] > 30)
            assert red.sum() > 100
            median = np.median(rgb[red], axis=0)
            assert median[0] > 150, median
            assert median[1] < 50 and median[2] < 50, median
    finally:
        sim.close()
