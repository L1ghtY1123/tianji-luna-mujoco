from dataclasses import replace
from types import SimpleNamespace

import pytest

from luna_mujoco.backend_builder import (
    create_solver_configs,
    validate_grasp_batch,
)
from luna_mujoco.public_planner import PublicPlanner


def profile(**updates):
    values = {"ik_num_seeds": 24, "ik_seed_solver_num_seeds": 40,
              "ik_position_threshold_m": .003, "ik_rotation_threshold_rad": .04,
              "random_seed": 321, "self_collision_check": True, "trajopt_num_seeds": 3,
              "trajectory_collision_clearance_m": .02,
              "optimizer_collision_activation_distance_m": .04,
              "plan_max_attempts": 2, "finetune_attempts": 3, "finetune_dt_scale": .7,
              "freq_hz": 50}
    return SimpleNamespace(**(values | updates))


class ConfigFactory:
    @staticmethod
    def create(**kwargs):
        return SimpleNamespace(**kwargs, robot_config=kwargs['robot'])


def test_planning_reuses_parsed_robot_instead_of_parsing_same_dict_twice():
    parsed_robot = object()
    class IkConfig:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(robot_config=parsed_robot)
    class MotionConfig:
        @staticmethod
        def create(**kwargs):
            assert kwargs['robot'] is parsed_robot
            return SimpleNamespace()
    ik, motion = create_solver_configs(profile(), robot={'robot_cfg': {}},
        scene_model={}, collision_cache={}, batch_size=16,
        ik_config_type=IkConfig, motion_config_type=MotionConfig)
    assert motion.ik_solver_config is ik


def test_standalone_and_planning_ik_share_constructed_config():
    settings = profile()
    ik, motion = create_solver_configs(
        settings, robot={"robot_cfg": {}}, scene_model={"cuboid": {}},
        collision_cache={}, batch_size=16, ik_config_type=ConfigFactory,
        motion_config_type=ConfigFactory, graph_rollout={},
    )
    assert motion.ik_solver_config is ik
    assert ik.num_seeds == 24 and ik.seed_solver_num_seeds == 40
    assert ik.optimizer_configs == ["ik/lbfgs_ik.yml"]
    assert ik.optimizer_collision_activation_distance == .02
    assert motion.optimizer_collision_activation_distance == .04
    assert ik.position_tolerance == motion.position_tolerance == .003
    assert ik.orientation_tolerance == motion.orientation_tolerance == .04
    assert ik.self_collision_check == motion.self_collision_check is True
    assert ik.random_seed == motion.random_seed == 321
    assert ik.use_cuda_graph == motion.use_cuda_graph is True
    assert motion.num_trajopt_seeds == 3
    assert motion.interpolation_dt == .02


def test_public_plan_pose_receives_configured_attempts_and_finetuning():
    class Planner:
        def plan_pose(self, goal, state, *args, **kwargs):
            return kwargs

    planner = PublicPlanner(Planner(), profile(), batch=True)
    result = planner.plan_pose("goal", "start")
    assert result == {"max_attempts": 2, "finetune_attempts": 3, "finetune_dt_scale": .7}


def native_batch():
    from tianji_curobo.contracts import NativeGraspBatchItem, NativeGraspBatchPlan, NativeTrajectory
    trajectory = NativeTrajectory(positions_rad=((0.,) * 7, (.1,) * 7),
                                  joint_names=tuple(f"joint{i}" for i in range(7)),
                                  interpolation_dt_s=.02, planning_time_s=0.,
                                  position_error_m=0., rotation_error_rad=0., scene_token="scene")
    item = NativeGraspBatchItem(0, True, True, True, True, True, "success",
                               trajectory, trajectory, trajectory, trajectory.positions_rad[-1])
    return NativeGraspBatchPlan((item, replace(item, candidate_index=1)), 2, 2, .1, "scene")


@pytest.mark.parametrize('no_solution', [False, True])
def test_batch_reports_outer_cost_without_intercepting_internal_pose_calls(no_solution):
    from types import SimpleNamespace

    calls = []
    class OfficialPlanner:
        def plan_pose(self, *args, **kwargs):
            calls.append(kwargs)
            return None if no_solution else SimpleNamespace(total_time=.125)
        def plan_grasp(self, *args, **kwargs):
            self.plan_pose('goal', 'start')
            self.plan_pose('approach', 'start')
            return 'done'
    planner = PublicPlanner(OfficialPlanner(), profile(), batch=True)
    assert planner.plan_grasp() == 'done'
    assert calls == [{}, {}]
    assert planner.last_grasp_wall_s >= 0
    planner.plan_grasp()
    assert calls == [{}, {}, {}, {}]


def test_batch_hard_check_rejects_one_candidate_without_losing_other():
    from tianji_curobo.contracts import CuroboToolError
    calls = []

    def validate(request):
        calls.append(request.points_deg)
        if len(calls) == 1:
            raise CuroboToolError("hard clearance", stage="validation",
                                 code="CUROBO_TRAJECTORY_COLLISION", status=422)

    result = validate_grasp_batch(native_batch(), validate)
    assert len(calls) == 2
    assert len(calls[0]) == 6  # Includes all three phases and their boundaries.
    assert result.items[0].success is False
    assert result.items[0].approach is None
    assert result.items[0].grasp_end_joints_rad is None
    assert "CUROBO_TRAJECTORY_COLLISION" in result.items[0].status
    assert result.items[1].success is True


def test_batch_hard_check_does_not_hide_service_failures():
    def validate(request):
        raise RuntimeError("CUDA broken")

    with pytest.raises(RuntimeError, match="CUDA broken"):
        validate_grasp_batch(native_batch(), validate)


def test_batch_hard_check_uses_only_raw_points_even_if_commands_are_present():
    native = native_batch()
    phase = replace(native.items[0].approach,
                    command_positions_rad=((0.,) * 7, (.8,) * 7, (.9,) * 7, (.1,) * 7),
                    trajectory_point_scale=3)
    native = replace(native, items=tuple(replace(item, approach=phase, grasp=phase, lift=phase)
                                        for item in native.items))
    checked = []
    result = validate_grasp_batch(native, lambda request: checked.append(request.points_deg))
    assert result == native
    assert len(checked) == 2
    assert all(len(points) == 6 for points in checked)


def test_standalone_config_uses_same_profile_with_different_batch_capacity():
    kwargs = {"robot": {}, "scene_model": {}, "collision_cache": {},
              "ik_config_type": ConfigFactory}
    settings = profile(self_collision_check=False, freq_hz=100)
    standalone, absent = create_solver_configs(settings, batch_size=100, **kwargs)
    inside_plan, motion = create_solver_configs(settings, batch_size=16, **kwargs,
                                                motion_config_type=ConfigFactory, graph_rollout={})
    left = vars(standalone) | {"max_batch_size": 16}
    assert left == vars(inside_plan)
    assert absent is None and motion.interpolation_dt == .01
    assert motion.self_collision_check is False


@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("ik_only", [True, False])
@pytest.mark.parametrize("direct_factory", [False, True])
def test_optimizer_and_hard_checker_receive_original_open_gripper_spheres(monkeypatch, arm, ik_only, direct_factory):
    import sys
    from types import ModuleType

    import yaml

    from luna_mujoco import backend_builder

    root = backend_builder.ROOT
    robot_path = root / "resources/curobo/marvin_gui_dual.yml"
    original = yaml.safe_load(robot_path.read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    captured = {}

    class PublicCheckerBoundaryReached(Exception):
        pass

    class CheckerConfig:
        @staticmethod
        def load_from_config(**kwargs):
            captured["checker"] = kwargs
            # The actual application and vendor adapters have run. Stop at the
            # public CUDA constructor: CPU tests must not instantiate a GPU.
            raise PublicCheckerBoundaryReached

    class Planner:
        def __init__(self, config):
            self.config = config
            self.device_cfg = object()
            self.scene_collision_checker = object()
            captured["planner"] = self

    modules = {
        "curobo": {}, "torch": {},
        "curobo.collision_checking": {"RobotCollisionChecker": object,
                                      "RobotCollisionCheckerCfg": CheckerConfig},
        "curobo.inverse_kinematics": {"InverseKinematics": Planner, "InverseKinematicsCfg": ConfigFactory},
        "curobo.kinematics": {"Kinematics": object, "KinematicsCfg": object},
        "curobo.motion_planner": {"MotionPlannerCfg": ConfigFactory, "MotionPlanner": Planner},
        "curobo.batch_motion_planner": {"BatchMotionPlanner": Planner},
        "curobo.scene": {"Cuboid": object},
        "curobo.types": dict.fromkeys(("GoalToolPose", "JointState", "Pose", "ToolPoseCriteria"), object),
        "tianji_curobo.grasp_planner": {"GraspBatchMotionPlanner": Planner},
    }
    for name, exports in modules.items():
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(backend_builder, "_graph_planner_rollout_config", dict)
    runtime = SimpleNamespace(planning_settings=profile(trajectory_collision_clearance_m=.01),
                              _robot_config_path=robot_path,
                              _urdf_path=root / "assets/marvin_urdf/marvin_with_grippers.urdf",
                              _asset_root_path=root / "assets", _excluded_collision_links=(),
                              _scene_model=None, _collision_cache={})
    with pytest.raises(PublicCheckerBoundaryReached):
        backend_builder.build_geometry_backend(runtime, arm, batch_size=100 if ik_only else 16,
                                               ik_only=ik_only,
                                               batch_planner_type=Planner if direct_factory else None)

    planner = captured["planner"]
    checker = captured["checker"]
    if not ik_only:
        assert planner.config.ik_solver_config.robot is planner.config.robot
        assert type(planner) is Planner
    for config in (planner.config.robot, checker["robot_config"]):
        actual = config["robot_cfg"]["kinematics"]
        for link, spheres in actual["collision_spheres"].items():
            if link.startswith('jev_carried_object_'):
                assert len(spheres) == 31 and all(s['radius'] < 0 for s in spheres)
                continue
            for sphere, source in zip(spheres, original["collision_spheres"][link], strict=True):
                assert sphere['center'] == source['center']
                padding = 1e-9 if link.startswith(('gripL_', 'gripR_')) else 0.
                assert sphere['radius'] == source['radius'] + padding
        assert {k: v for k, v in actual["collision_sphere_buffer"].items()
                if not k.startswith('jev_carried_object_')} == original["collision_sphere_buffer"]
        assert actual["lock_joints"]["gripL_finger_joint"] == 0
        assert actual["lock_joints"]["gripR_finger_joint"] == 0
    assert checker["scene_collision_checker"] is planner.scene_collision_checker
    assert checker["collision_activation_distance"] == .01


def test_esdf_radius_uses_authored_spheres_and_buffer_without_closure_inflation(tmp_path):
    import yaml

    from luna_mujoco import backend_builder

    root = backend_builder.ROOT
    original = yaml.safe_load((root / "resources/curobo/marvin_gui_dual.yml").read_text(encoding="utf-8"))
    kinematics = original["robot_cfg"]["kinematics"]
    # A dominant gripper sphere makes extra inflation observable even when
    # the production model's largest sphere belongs to the robot base.
    kinematics["collision_spheres"]["gripL_Link3_R"][0]["radius"] = .2
    kinematics["collision_sphere_buffer"]["gripL_Link3_R"] = .005
    config_path = tmp_path / "robot.yml"
    config_path.write_text(yaml.safe_dump(original), encoding="utf-8")
    runtime = SimpleNamespace(_robot_config_path=config_path,
                              _urdf_path=root / "assets/marvin_urdf/marvin_with_grippers.urdf",
                              _asset_root_path=root / "assets")
    assert backend_builder.maximum_world_sphere_radius(runtime) == pytest.approx(.205)


def test_support_departure_passes_public_linear_scale_without_hidden_multiplier():
    from dataclasses import dataclass

    from luna_mujoco.backend_builder import support_departure_criteria

    @dataclass
    class Criteria:
        terminal_pose_axes_weight_factor: list
        non_terminal_pose_axes_weight_factor: list
        project_distance_to_goal: bool

        @staticmethod
        def linear_motion(*, axis, non_terminal_scale, project_distance_to_goal):
            assert axis == "z"
            return Criteria([1.] * 6, [non_terminal_scale] * 2 + [0.] + [non_terminal_scale] * 3,
                            project_distance_to_goal)

    actual = support_departure_criteria(Criteria, SimpleNamespace(axis="z", reference_frame="world"), 2.)
    assert actual.terminal_pose_axes_weight_factor == [1.] * 6
    assert actual.non_terminal_pose_axes_weight_factor == [2., 2., 0., 2., 2., 2.]
    assert actual.project_distance_to_goal is False


def test_support_frame_config_is_temporary_fixed_tcp_child_with_no_collision_changes():
    from copy import deepcopy

    from luna_mujoco.backend_builder import create_support_frame_robot_config

    original = {"robot_cfg": {"kinematics": {"extra_links": {}, "tool_frames": ["TCP_Link_L"],
                                             "collision_spheres": {"arm": [1]}, "collision_link_names": ["arm"]}}}
    before = deepcopy(original)
    frame = {"arm": "A", "link_name": "jev_support_axis_A", "parent_link_name": "TCP_Link_L",
             "fixed_transform": [0, 0, 0, 1, 0, 0, 0]}
    actual = create_support_frame_robot_config(original, frame)
    kin = actual["robot_cfg"]["kinematics"]
    assert original == before and kin["tool_frames"] == ["jev_support_axis_A"]
    assert kin["extra_links"]["jev_support_axis_A"]["parent_link_name"] == "TCP_Link_L"
    assert kin["collision_spheres"] == before["robot_cfg"]["kinematics"]["collision_spheres"]
    assert kin["collision_link_names"] == ["arm"]
