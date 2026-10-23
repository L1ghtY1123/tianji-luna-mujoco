"""Independent CPU boundary checks; no CUDA or physical robot is instantiated."""

import sys
from contextlib import nullcontext
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from test_trajectory_postprocessing import Tensor as BaseTensor
from test_visible_surface_gpu_contract import measured_binding

from luna_mujoco import carrying
from luna_mujoco.backend_builder import ROOT
from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard


class Tensor(BaseTensor):
    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def __add__(self, other):
        return Tensor(self.values + other.values)

    def contiguous(self):
        return self

    def tolist(self):
        return self.values.tolist()


def command_boundary(monkeypatch, phase, failure=None):
    """Replace only tensor/GPU endpoints; retain native CPU config and verdicts."""
    checkers, calls = [], []
    names = [f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8)]

    class Checker:
        def __init__(self, config):
            self.index = len(checkers)
            checkers.append(self)
            kin = config["robot_config"]["robot_cfg"]["kinematics"]
            indices, balls = {}, []
            for link in kin["collision_link_names"]:
                indices[link] = list(range(len(balls), len(balls) + len(kin["collision_spheres"][link])))
                balls.extend([[0., 0., 1., .02]] * len(indices[link]))
            self.balls = np.asarray(balls)
            self.indices = indices
            self.kinematics = SimpleNamespace(joint_names=names + ['gripL_finger_joint', 'gripR_finger_joint'], kinematics_config=SimpleNamespace(
                get_sphere_index_from_link_name=lambda link: Tensor(indices[link])))
            self.collision_cost = self.self_collision_cost = object()

        def get_kinematics(self, q):
            self.q = q.values
            count = q.values.shape[1]
            pose = SimpleNamespace(position=Tensor(np.zeros((1, count, 3))),
                quaternion=Tensor(np.tile([1., 0., 0., 0.], (1, count, 1))))
            return SimpleNamespace(robot_spheres=Tensor(np.tile(self.balls, (1, count, 1, 1))),
                tool_poses=SimpleNamespace(get_link_pose=lambda link: pose))

        def cost(self, kind):
            calls.append((self.index, kind))
            values = np.zeros((1, self.q.shape[1], len(self.balls)))
            selected = failure == kind and self.index == 0
            selected |= failure == "sweep_world" and kind == "world" and self.index == len(checkers) - 1
            selected |= failure == "driver_self" and kind == "self" and self.index == 1
            if selected:
                # A positive cost occurs only between controller endpoints.
                interior = (self.q[0, :, 0] > .003) & (self.q[0, :, 0] < .018)
                values[0, interior, -1] = .001
            return Tensor(values)

        def get_collision_distance(self, state):
            return self.cost("world")

        def get_self_collision_distance(self, spheres):
            return self.cost("self")

        def get_bound(self, q):
            return self.cost("bounds")

    for name, exports in {
        "torch": {"Tensor": Tensor, "as_tensor": lambda value, **kwargs: Tensor(value), "no_grad": nullcontext},
        "curobo": {},
        "curobo.collision_checking": {"RobotCollisionChecker": Checker,
            "RobotCollisionCheckerCfg": SimpleNamespace(load_from_config=lambda **kwargs: kwargs)},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)

    view = {"depth_left_m": np.full((21, 21), .5), "valid_left_mask": np.ones((21, 21)),
        "left_intrinsics": [[10, 0, 10], [0, 10, 10], [0, 0, 1]],
        "T_base_left_camera": np.eye(4), "instance_masks": {}}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": view["depth_left_m"]}, "scene")
    guard.capture_spheres = np.empty((0, 4))
    guard.capture_joint_inputs = {}
    runtime = SimpleNamespace(_robot_config_path=ROOT / "resources/curobo/marvin_gui_dual.yml",
        _urdf_path=ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf", _asset_root_path=ROOT / "assets",
        _command_checkers={}, planning_settings=SimpleNamespace(trajectory_collision_clearance_m=.01),
        observed_free_guard=guard, geometry_scene={"collision_world_source": "measured_semantic_esdf"})
    planner = SimpleNamespace(device_cfg=SimpleNamespace(as_torch_dict=dict), scene_collision_checker=object())
    payload = {"phase": phase, "arm": "A", "scene_token": "scene", "positions_rad": [[0.] * 7, [.02] * 7],
        "joint_names": names[:7], "other_joints_deg": [0.] * 7,
        "gripper_joint_positions_rad": {"A": .04, "B": 0.}}
    binding = measured_binding() if phase in {"approach", "place", "open"} else None
    if binding is not None:
        payload["carrying_binding"] = asdict(binding)
    if phase == "open":
        payload.update(gripper_start_position_rad=.04, gripper_end_position_rad=0.)
    if phase == "approach_grasp":
        payload["gripper_geometry_mode"] = "fixed_open"
    return runtime, planner, payload, binding, calls


@pytest.mark.parametrize("phase", ["approach", "place", "retreat", "observation", "approach_grasp"])
@pytest.mark.parametrize("failure", [None, "world", "self", "bounds"])
def test_fixed_models_check_world_self_and_bounds(monkeypatch, phase, failure):
    runtime, planner, payload, binding, calls = command_boundary(monkeypatch, phase, failure)
    runtime._command_checkers = {}
    result = carrying.check_command_path(runtime, planner, payload, binding)
    assert result["collision_free"] is (failure is None)
    assert result["attached"] is (binding is not None)
    assert {kind for _, kind in calls} == {"world", "self", "bounds"}
    assert all(index == 0 for index, _ in calls)
    assert result["checked_point_count"] > result["command_point_count"]


@pytest.mark.parametrize('failure', [None, 'world', 'self', 'bounds'])
def test_dual_command_checks_both_moving_chains_and_measured_grippers(monkeypatch, failure):
    runtime, planner, payload, _, calls = command_boundary(monkeypatch, 'observation', failure)
    payload.update(arm='AB', positions_rad=[[0.] * 14, [.02] * 7 + [-.04] * 7],
                   joint_names=[f'Joint{i}_{side}' for side in ('L', 'R') for i in range(1, 8)])
    payload.pop('other_joints_deg')
    result = carrying.check_command_path(runtime, planner, payload, None)
    checker = next(iter(runtime._command_checkers.values()))
    np.testing.assert_allclose(checker.q[0, -1], [.02] * 7 + [-.04] * 7 + [.04, 0.])
    assert result['collision_free'] is (failure is None)
    assert {kind for _, kind in calls} == {'world', 'self', 'bounds'}
