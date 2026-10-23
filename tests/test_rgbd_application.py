import copy

import numpy as np
import pytest

from luna_mujoco.runner import Runner


class RobotSimulation:
    def robot_snapshot(self):
        return {"generation": 1, "time": 2., "confirmed": True,
                "arms": {arm: {"joints_rad": [0.] * 7, "tcp": np.eye(4).tolist(),
                                "gripper_closed_fraction": .6,
                                "gripper_position_rad": .3} for arm in ("A", "B")}}

    def snapshot(self):
        raise AssertionError("control path accessed simulator environmental truth")


def observation():
    pose = np.eye(4)
    pose[:3, 3] = [.5, .1, .62]
    return {"source": "rgbd", "generation": 1, "time": 2., "observation_id": "rgbd-1",
            "objects": [{"id": "track-17", "kind": "cube", "color": "red",
                         "size": [.04] * 3, "position": pose[:3, 3].tolist(),
                         "quaternion": [1., 0., 0., 0.], "T_world_object": pose.tolist(),
                         "points_world_m": [[.5, .1, .62]], "confidence": .9,
                         "simulator_object_id": "secret", "coverage": "observed_surface_only"}],
            "table": {"position": [.5, 0., .56], "size": [.7, 1., .08], "source": "rgbd_plane"},
            "views": {"3": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}}}


def test_decision_before_observe_never_reads_environmental_truth(tmp_path):
    runner = Runner(RobotSimulation(), None, None, tmp_path)
    supplied = runner.snapshot_for_decision()
    assert supplied["simulation"]["generation"] == 1
    assert "objects" not in supplied["simulation"]
    assert "table" not in supplied["simulation"]
    assert supplied.get("observation") is None


def test_decision_uses_only_observed_fields_and_omits_pixels_and_hidden_ids(tmp_path):
    runner = Runner(RobotSimulation(), None, None, tmp_path)
    runner.state["facts"]["observed"] = True
    measured = observation()
    before = copy.deepcopy(measured)
    supplied = runner.snapshot_for_decision({"observation": measured})
    obj = supplied["observation"]["objects"][0]
    assert obj["id"] == "track-17"
    assert obj["color"] == "red"
    assert "points_world_m" not in obj
    assert "simulator_object_id" not in obj
    assert "views" not in supplied["observation"]
    assert measured["objects"] == before["objects"]




def test_object_release_tcp_goal_uses_measured_support_reference_without_full_extent():
    from luna_mujoco.placement import release_goal

    target = {"visible_centroid_world": [.5, .1, .63]}
    destination = {"id": "track-bin", "kind": "bin", "placement_reference": {
        "position_world_m": [.6, -.2, .61], "floor_z_m": .61, "source": "rgbd_floor"}}
    grasp = np.eye(4)
    grasp[:3, 3] = [.5, .1, .80]
    holding = {"arm": "A", "grasp_support_reference": {"T_world_tcp_at_grasp": grasp.tolist(), "table_height_m": .6}}
    result = release_goal(target, destination, holding)
    assert result["T_world_tcp"][0][3] == pytest.approx(.6)
    assert result["T_world_tcp"][1][3] == pytest.approx(-.2)
    assert result["T_world_tcp"][2][3] == pytest.approx(.825)
    assert "T_world_object" not in result


def test_release_goal_does_not_read_poison_capacity_or_object_geometry():
    from luna_mujoco.placement import release_goal

    target = {"visible_centroid_world": [.5, .1, .63], "size": "poison", "T_world_object": "poison"}
    destination = {"id": "bin", "kind": "bin", "cavity": "poison", "placement_reference": {
        "position_world_m": [.6, -.2, .61], "floor_z_m": .61}}
    holding = {"arm": "A", "grasp_support_reference": {"T_world_tcp_at_grasp": np.eye(4).tolist(), "table_height_m": .6}}
    result = release_goal(target, destination, holding)
    assert result["release_reference"]["release_height_m"] == .015
