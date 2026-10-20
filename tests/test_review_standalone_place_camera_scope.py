import copy

import numpy as np
import pytest
from test_rgbd_workflow import SensorRobot, VisionAlgorithms
from test_standalone_place_freshness_review import held_runner

from jev_mujoco.runner import Runner


def current_wrist_observation(runner, cameras=("1", "2"), change=None):
    original = runner.algorithms.observe

    def observe(capture, **kwargs):
        result = original(capture, **kwargs)
        destination = result["objects"][1]
        destination["coverage"] = {"camera_ids": list(cameras)}
        for camera, view in result["views"].items():
            if camera not in cameras:
                view["instance_masks"].pop("destination", None)
        if change == "missing":
            result["objects"] = result["objects"][:1]
        elif change == "hypothesis_only":
            for view in result["views"].values():
                view["instance_masks"].pop("destination", None)
        elif change == "invalid_depth":
            for camera in cameras:
                result["views"][camera]["valid_depth"][:] = False
        elif change == "stale_observation":
            result["observation_id"] = "previous-capture"
        elif change == "stale_point_sources":
            destination["point_sources"] = {"observation_id": "previous-capture"}
        elif change == "stale_view":
            for camera in cameras:
                result["views"][camera]["frame_id"] = "previous-capture"
        elif change == "uncalibrated":
            for camera in cameras:
                result["views"][camera]["intrinsics"]["fx"] = 0.
        elif change == "lost_holding":
            result["objects"][0]["points_world_m"] += [.1, 0., 0.]
        return result

    runner.algorithms.observe = observe


def standalone_context(tmp_path, **kwargs):
    runner, previous = held_runner(tmp_path)
    runner.state.update(generation=1, status="running", instruction="放到蓝色盒子里",
        attributes={"kind": "cube", "color": "red"}, destination_attributes={"kind": "bin", "color": "blue"})
    context = {"observation": copy.deepcopy(previous), "holding": copy.deepcopy(runner.holding_state),
               "visual_plan": {"standalone_place": "initialize_inventory"}}
    current_wrist_observation(runner, **kwargs)
    return runner, context


@pytest.mark.parametrize("cameras", [("1",), ("2",), ("1", "2")])
async def test_standalone_place_accepts_current_wrist_destination_and_rechecks_holding(tmp_path, cameras):
    runner, context = standalone_context(tmp_path, cameras=cameras)
    previous = context["observation"]["observation_id"]
    points = np.asarray(context["holding"]["points_tool_m"]).copy()
    observed = await runner.observe(context, checkpoint="standalone_place")
    assert observed["observation_id"] != previous
    assert runner.state["facts"]["views_ready"] is True
    assert context["view_evidence"]["destination_camera_ids"] == list(cameras)
    assert context["holding"]["last_position_evidence"]["status"] == "held"
    assert context["holding"]["observation"]["observation_id"] == observed["observation_id"]
    np.testing.assert_array_equal(context["holding"]["points_tool_m"], points)
    assert runner.simulation.stage == 2 and runner.simulation.closed
    assert runner.algorithms.checkpoints == ["post_grasp", "standalone_place"]


async def test_initial_wrist_only_destination_still_rejected_before_motion(tmp_path):
    runner = Runner(SensorRobot(), VisionAlgorithms(), None, tmp_path)
    runner.state.update(generation=1, status="running", task="grasp_place", instruction="抓取红色方块到蓝色盒子里",
        attributes={"kind": "cube", "color": "red"}, destination_attributes={"kind": "bin", "color": "blue"})
    current_wrist_observation(runner)
    with pytest.raises(ValueError, match="head_destination_missing"):
        await runner.observe({'visual_plan': {'initial': 'initialize_inventory'}}, checkpoint="initial")
    assert runner.simulation.stage == 0 and not runner.simulation.closed


@pytest.mark.parametrize("change", ["missing", "hypothesis_only", "invalid_depth", "stale_point_sources",
                                    "stale_view", "uncalibrated"])
async def test_standalone_place_requires_current_measured_destination(tmp_path, change):
    runner, context = standalone_context(tmp_path, change=change)
    with pytest.raises(ValueError, match="^destination_missing$"):
        await runner.observe(context, checkpoint="standalone_place")
    assert not runner.state["facts"].get("views_ready", False)
    assert runner.simulation.stage == 2 and runner.simulation.closed


async def test_standalone_place_rejects_recycled_capture_identity(tmp_path):
    runner, context = standalone_context(tmp_path, change="stale_observation")
    with pytest.raises(ValueError, match="invalid_rgbd_observation_identity"):
        await runner.observe(context, checkpoint="standalone_place")
    assert runner.simulation.stage == 2 and runner.simulation.closed


async def test_wrist_destination_does_not_require_held_source_segmentation(tmp_path):
    runner, context = standalone_context(tmp_path, change="lost_holding")
    await runner.observe(context, checkpoint="standalone_place")
    assert runner.holding_state["verified"]
    assert runner.simulation.stage == 2 and runner.simulation.closed
