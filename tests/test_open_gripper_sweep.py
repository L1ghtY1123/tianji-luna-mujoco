"""Release checks the two requested endpoint states without an opening sweep."""
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from test_review_place_coverage_scope import Tensor
from test_visible_surface_gpu_contract import measured_binding

from luna_mujoco.backend_builder import ROOT
from luna_mujoco.carrying import carrying_touch_links, check_command_path
from luna_mujoco.motion_coverage import MotionCoverage


def boundary(monkeypatch, arm, failure=None):
    configs, calls = [], []
    names = [f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8)]
    driver = "gripL_finger_joint" if arm == "A" else "gripR_finger_joint"

    class Checker:
        def __init__(self, config):
            configs.append(config["robot_config"])
            self.kin = config["robot_config"]["robot_cfg"]["kinematics"]
            indices, count = {}, 0
            for link in self.kin['collision_link_names']:
                size = len(self.kin['collision_spheres'][link])
                indices[link] = list(range(count, count + size))
                count += size
            self.sphere_count = count
            self.kinematics = SimpleNamespace(joint_names=names + ['gripL_finger_joint', 'gripR_finger_joint'],
                kinematics_config=SimpleNamespace(get_sphere_index_from_link_name=lambda link: Tensor(indices[link])))
            self.collision_cost = self.self_collision_cost = object()

        def get_kinematics(self, q):
            self.angle = q.values[0, 0, self.kinematics.joint_names.index(driver)]
            self.count = q.values.shape[1]
            balls = np.tile([0., 0., 1., .02], (1, self.count, self.sphere_count, 1))
            return SimpleNamespace(robot_spheres=Tensor(balls))

        def cost(self, kind):
            calls.append((self.angle, kind))
            # Both requested endpoints are free; an interior finger position is not.
            value = .001 if ((failure == kind and .009 < self.angle < .031)
                             or (failure == "endpoint_" + kind and self.angle == 0.)) else 0.
            return Tensor(np.full((1, self.count, 1), value))

        def get_collision_distance(self, state): return self.cost("world")
        def get_self_collision_distance(self, spheres): return self.cost("self")
        def get_bound(self, q): return self.cost("bounds")

    for name, exports in {
        "torch": {"Tensor": Tensor, "as_tensor": lambda value, **kwargs: Tensor(value), "no_grad": nullcontext},
        "curobo": {}, "curobo.collision_checking": {"RobotCollisionChecker": Checker,
            "RobotCollisionCheckerCfg": SimpleNamespace(load_from_config=lambda **kwargs: kwargs)},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    runtime = SimpleNamespace(_robot_config_path=ROOT / "resources/curobo/marvin_gui_dual.yml",
        _urdf_path=ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf", _asset_root_path=ROOT / "assets",
        _command_checkers={}, planning_settings=SimpleNamespace(trajectory_collision_clearance_m=.01),
        geometry_scene={'scene_token': 'scene', 'collision_world_source': 'measured_semantic_esdf',
                        'exclusion_state': 'held'},
        motion_coverage=MotionCoverage({'head': {'depth_left_m': np.full((21, 21), 3.),
            'valid_left_mask': np.ones((21, 21)), 'left_intrinsics': [[10, 0, 10], [0, 10, 10], [0, 0, 1]],
            'T_base_left_camera': np.eye(4)}}, 'scene'))
    planner = SimpleNamespace(device_cfg=SimpleNamespace(as_torch_dict=dict), scene_collision_checker=object())
    binding = replace(measured_binding(), arm=arm)
    payload = {"phase": "open", "arm": arm, "scene_token": "scene", "positions_rad": [[0.] * 7],
        "joint_names": names[:7] if arm == "A" else names[7:], "other_joints_deg": [0.] * 7,
        "gripper_joint_positions_rad": {arm: .04, "B" if arm == "A" else "A": .3},
        "gripper_start_position_rad": .04, "gripper_end_position_rad": 0., "carrying_binding": asdict(binding)}
    return runtime, planner, payload, binding, configs, calls


@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("failure", ["world", "self", None])
def test_release_checks_only_closed_attached_and_open_detached(monkeypatch, arm, failure):
    runtime, planner, payload, binding, configs, calls = boundary(monkeypatch, arm, failure)
    result = check_command_path(runtime, planner, payload, binding)
    assert result["collision_free"]
    assert result['collision_world_source'] == 'measured_semantic_esdf'
    assert "gripper_sweep" not in result
    proof = result["gripper_endpoints"]
    assert proof["driver_samples_rad"] == [.04, 0.]
    assert proof["continuous_sweep_checked"] is False
    assert {angle for angle, _ in calls} == {.04, 0.}
    assert len(configs) == 2
    closed, opened = [config["robot_cfg"]["kinematics"] for config in configs]
    attached = f"jev_carried_object_{arm}"
    assert set(closed["self_collision_ignore"][attached]) == set(carrying_touch_links(arm))
    assert attached not in opened["collision_link_names"]


@pytest.mark.parametrize("changes", [{"gripper_start_position_rad": .03}, {"gripper_end_position_rad": -.01},
                                     {"gripper_end_position_rad": .6}])
def test_open_sweep_rejects_stale_or_out_of_range_driver_command_before_gpu(monkeypatch, changes):
    runtime, planner, payload, binding, configs, _ = boundary(monkeypatch, "A")
    with pytest.raises(ValueError):
        check_command_path(runtime, planner, payload | changes, binding)
    assert not configs

@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("failure", ["endpoint_world", "endpoint_self"])
def test_fully_open_endpoint_collision_is_still_rejected(monkeypatch, arm, failure):
    runtime, planner, payload, binding, _, _ = boundary(monkeypatch, arm, failure)
    result = check_command_path(runtime, planner, payload, binding)
    assert not result["collision_free"]
    assert result["gripper_endpoints"]["first_invalid_driver_rad"] == 0.


def test_numpy_sample_index_is_serializable_in_collision_service_response(monkeypatch):
    from tianji_curobo import grasp_lift
    runtime, planner, payload, binding, _, _ = boundary(monkeypatch, 'B', 'endpoint_world')
    original = grasp_lift.collision_samples
    def numpy_indices(positions):
        samples, indices = original(positions)
        return samples, np.asarray(indices, dtype=np.int64)
    monkeypatch.setattr(grasp_lift, 'collision_samples', numpy_indices)
    result = check_command_path(runtime, planner, payload, binding)
    assert not result['collision_free']
    assert json.loads(json.dumps(result, allow_nan=False))['first_invalid_index'] == 0
