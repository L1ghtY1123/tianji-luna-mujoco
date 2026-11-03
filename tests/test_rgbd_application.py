import copy

import numpy as np

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


def test_shared_primitives_have_no_stepwise_model_entrypoint(tmp_path):
    runner = Runner(RobotSimulation(), None, None, tmp_path)
    assert not hasattr(runner, 'snapshot_for_decision')


def test_decision_uses_only_observed_fields_and_omits_pixels_and_hidden_ids(tmp_path):
    runner = Runner(RobotSimulation(), None, None, tmp_path)
    runner.state["facts"]["observed"] = True
    measured = observation()
    before = copy.deepcopy(measured)
    from luna_mujoco.placement import public_observation
    supplied = public_observation(measured)
    obj = supplied["objects"][0]
    assert obj["id"] == "track-17"
    assert obj["color"] == "red"
    assert "points_world_m" not in obj
    assert "simulator_object_id" not in obj
    assert "views" not in supplied
    assert measured["objects"] == before["objects"]


