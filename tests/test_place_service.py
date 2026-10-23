from dataclasses import asdict
from types import SimpleNamespace

import pytest
from test_carrying import binding

from luna_mujoco import geometry_service
from luna_mujoco.planning_settings import PlanningSettings


def test_place_maps_offsets_and_returns_only_honest_pre_release_phases():
    from tianji_curobo.contracts import PlanPlaceRequest

    from luna_mujoco.trajectory_postprocessing import PreReleasePlacePlan, RawTrajectory

    settings = PlanningSettings(place_approach_offset_m=.12, place_retreat_offset_m=.15)
    phase = RawTrajectory(positions_rad=((0.,) * 7, (.1,) * 7), joint_names=tuple(f"Joint{i}_L" for i in range(1, 8)),
                             velocities_rad_s=((0.,) * 7,) * 2, accelerations_rad_s2=((0.,) * 7,) * 2,
                             interpolation_dt_s=.02, planning_time_s=.1, position_error_m=0., rotation_error_rad=0., scene_token="scene-2")
    class Runtime:
        geometry_solver_profile = settings.solver_payload()
        def __init__(self):
            self.effective_config = {"A": {"plan": {"max_batch_size": 1}}}
        def _scene_snapshot_unlocked(self):
            return {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2, "target_id": "sam-source"}
        def configure_planning(self, settings):
            pass
        def plan_place_scoped(self, request, carrying):
            assert isinstance(request, PlanPlaceRequest)
            assert request.approach_offset_m == .12 and request.approach_in_tool_frame is False
            assert request.lift_offset_m == .15 and request.lift_in_tool_frame is False
            return PreReleasePlacePlan(approach=phase, place=phase), {"optimizer_attachment": True}
    payload = {"arm": "A", "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
               "goal_position_m": [.5, 0, .7], "goal_quaternion_wxyz": [1, 0, 0, 0],
               "scene_token": "scene-2", "planning_settings": settings.model_dump(), "carrying_binding": binding()}
    result = geometry_service.run_operation(Runtime(), "/tools/plan_place", payload)
    assert result["retreat_requires_replan"] is True
    assert set(result["phases"]) == {"approach", "place"}
    assert "retreat_provisional" not in result
    assert result["phases"]["place"]["positions_rad"] == asdict(phase)["positions_rad"]
    assert result["collision_evidence"]["optimizer_attachment"] is True


def test_detached_pose_request_rejects_old_attachment():
    runtime = geometry_service.GeometryRuntime()
    with pytest.raises(ValueError, match="detach"):
        geometry_service.run_operation(runtime, "/tools/plan_to_pose", {"carrying_binding": binding()})


def test_reset_clear_invalidates_scene_but_preserves_resident_backends():
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "old"}
    runtime.geometry_model = {"voxel": {}}
    destroyed = []
    runtime._compound_backends["A"] = SimpleNamespace(destroy=lambda: destroyed.append(True))
    result = geometry_service.run_operation(runtime, "/carrying/clear", {})
    assert result["attached"] is False
    assert destroyed == [] and runtime._compound_backends
    with pytest.raises(ValueError, match="RGBD"):
        runtime._scene_snapshot_unlocked()


def test_compound_derivatives_survive_later_sdk_buffer_reuse():
    import numpy as np
    from test_trajectory_postprocessing import Tensor
    from tianji_curobo.contracts import NativePlacePlan, NativeTrajectory

    from luna_mujoco.trajectory_postprocessing import add_place_derivatives, build_execution_phases

    phase = NativeTrajectory(positions_rad=((0.,) * 7, (.1,) * 7), joint_names=tuple(f"joint{i}" for i in range(7)),
                             interpolation_dt_s=.02, planning_time_s=.1, position_error_m=0., rotation_error_rad=0.)
    q = np.array([[[[0.] * 7, [.1] * 7]]])
    trajectory = SimpleNamespace(position=Tensor(q), velocity=Tensor(np.ones_like(q)),
                                 acceleration=Tensor(np.zeros_like(q)), joint_names=list(phase.joint_names))
    raw = SimpleNamespace(**{f"{name}_interpolated_trajectory": trajectory for name in ("approach", "grasp", "lift")})
    retained = add_place_derivatives(NativePlacePlan.create(approach=phase, place=phase, retreat=phase), raw)
    trajectory.velocity.values[:] = 99
    assert retained.place.velocities_rad_s == ((1.,) * 7,) * 2
    commands = build_execution_phases({name: asdict(getattr(retained, name)) for name in ("approach", "place", "retreat")}, 3)
    assert set(commands) == {"approach", "place", "retreat"}
    assert commands["place"]["raw_point_count"] == 2 and commands["place"]["command_point_count"] == 4


def test_single_compound_builder_reaches_public_solver_with_carried_geometry(monkeypatch):
    import sys
    from types import ModuleType

    from test_backend_builder import ConfigFactory, profile

    from luna_mujoco import backend_builder
    from luna_mujoco.carrying import CarryingBinding

    constructed = []
    class BoundaryReached(Exception):
        pass
    class Planner:
        def __init__(self, config):
            constructed.append(config)
            self.device_cfg = object()
            self.scene_collision_checker = object()
    class CheckerConfig:
        @staticmethod
        def load_from_config(**kwargs):
            assert "jev_carried_object_A" in kwargs["robot_config"]["robot_cfg"]["kinematics"]["collision_spheres"]
            raise BoundaryReached
    modules = {"curobo": {}, "torch": {},
        "curobo.collision_checking": {"RobotCollisionChecker": object, "RobotCollisionCheckerCfg": CheckerConfig},
        "curobo.inverse_kinematics": {"InverseKinematics": Planner, "InverseKinematicsCfg": ConfigFactory},
        "curobo.kinematics": {"Kinematics": object, "KinematicsCfg": object},
        "curobo.motion_planner": {"MotionPlannerCfg": ConfigFactory, "MotionPlanner": Planner},
        "curobo.batch_motion_planner": {"BatchMotionPlanner": Planner}, "curobo.scene": {"Cuboid": object},
        "curobo.types": dict.fromkeys(("GoalToolPose", "JointState", "Pose", "ToolPoseCriteria"), object),
        "tianji_curobo.grasp_planner": {"GraspBatchMotionPlanner": Planner, "GraspMotionPlanner": Planner}}
    for name, exports in modules.items():
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(backend_builder, "_graph_planner_rollout_config", dict)
    root = backend_builder.ROOT
    scene = {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2, "target_id": "sam-source"}
    runtime = SimpleNamespace(planning_settings=profile(), _robot_config_path=root / "resources/curobo/marvin_gui_dual.yml",
        _urdf_path=root / "assets/marvin_urdf/marvin_with_grippers.urdf", _asset_root_path=root / "assets",
        _excluded_collision_links=(), _scene_model=None, _collision_cache={}, gripper_feedback=None,
        carrying_binding=CarryingBinding.from_mapping(binding(), scene=scene, arm="A"))
    with pytest.raises(BoundaryReached):
        backend_builder.build_geometry_backend(runtime, "A", batch_size=None, ik_only=False)
    assert len(constructed) == 1
    assert constructed[0].max_batch_size == 1
    assert constructed[0].ik_solver_config.robot is constructed[0].robot


def test_pinned_single_planner_forwards_only_supported_public_arguments():
    from test_backend_builder import profile

    from luna_mujoco.public_planner import PublicPlanner

    class PinnedPlanner:
        def plan_pose(self, goal, state, use_implicit_goal=True, max_attempts=5, enable_graph_attempt=1):
            return {"max_attempts": max_attempts, "enable_graph_attempt": enable_graph_attempt}
    planner = PublicPlanner(PinnedPlanner(), profile(plan_max_attempts=3, finetune_attempts=7))
    assert planner.plan_pose("goal", "state") == {"max_attempts": 3, "enable_graph_attempt": 1}


def test_attachment_detach_error_still_releases_public_planner_buffers():
    from luna_mujoco.backend_builder import ValidatedSingleBackend

    destroyed = []
    def broken_detach(**kwargs):
        raise RuntimeError("detach failed")
    backend = ValidatedSingleBackend.__new__(ValidatedSingleBackend)
    backend.carrying_link = "jev_carried_object_A"
    backend.motion_planner = SimpleNamespace(attachment_manager=SimpleNamespace(detach=broken_detach),
                                             destroy=lambda: destroyed.append(True))
    backend._peer_baseline_inverse_by_link = {}
    backend._torch = SimpleNamespace()
    with pytest.raises(RuntimeError, match="detach failed"):
        backend.destroy()
    assert destroyed == [True]
    assert backend.motion_planner is None and backend.carrying_link is None


def test_carried_pose_endpoint_replans_lift_with_attachment_and_measured_grippers():
    from tianji_curobo.contracts import NativeTrajectory, PlanToPoseRequest

    settings = PlanningSettings()
    phase = NativeTrajectory(positions_rad=((0.,) * 7, (.1,) * 7), joint_names=tuple(f"Joint{i}_L" for i in range(1, 8)),
                             interpolation_dt_s=.02, planning_time_s=.1, position_error_m=0., rotation_error_rad=0., scene_token="scene-2")
    class Runtime:
        geometry_solver_profile = settings.solver_payload()
        def _scene_snapshot_unlocked(self):
            return {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2,
                    "target_id": "sam-source", "exclusion_state": "held"}
        def configure_planning(self, settings):
            pass
        def plan_carry_pose_scoped(self, request, carrying, feedback):
            assert isinstance(request, PlanToPoseRequest)
            assert carrying.owner_id == "hold-1" and carrying.state == "held"
            assert feedback == {"A": .1, "B": 0.}
            assert request.goal_position_m == (.5, .1, .8)
            return phase, {"optimizer_attachment": True}
    payload = {"arm": "A", "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
               "goal_position_m": [.5, .1, .8], "goal_quaternion_wxyz": [1, 0, 0, 0],
               "scene_token": "scene-2", "planning_settings": settings.model_dump(), "carrying_binding": binding(),
               "gripper_joint_positions_rad": {"A": .1, "B": 0.}}
    result = geometry_service.run_operation(Runtime(), "/tools/plan_carried_pose", payload)
    assert result["planner"] == "curobo_carried_pose" and result["attached"] is True
    assert result["trajectory"]["positions_rad"] == phase.positions_rad
    assert result["collision_evidence"]["optimizer_attachment"] is True


@pytest.mark.parametrize("exclusion_state", ["released", None])
def test_carried_pose_endpoint_rejects_released_or_unbound_map(exclusion_state):
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2,
                              "target_id": "sam-source", "exclusion_state": exclusion_state}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    payload = {"arm": "A", "scene_token": "scene-2", "planning_settings": runtime.planning_settings.model_dump(),
               "carrying_binding": binding(), "gripper_joint_positions_rad": {"A": .1, "B": 0.}}
    with pytest.raises(ValueError, match="held.*map"):
        geometry_service.run_operation(runtime, "/tools/plan_carried_pose", payload)


@pytest.mark.parametrize("missing", ["carrying_binding", "gripper_joint_positions_rad"])
def test_carried_pose_endpoint_requires_bound_geometry_and_actual_driver_feedback(missing):
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2,
                              "target_id": "sam-source", "exclusion_state": "held"}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    payload = {"arm": "A", "scene_token": "scene-2", "planning_settings": runtime.planning_settings.model_dump(),
               "carrying_binding": binding(), "gripper_joint_positions_rad": {"A": .1, "B": 0.}}
    payload.pop(missing)
    with pytest.raises(ValueError, match="requires.*(binding|feedback)"):
        geometry_service.run_operation(runtime, "/tools/plan_carried_pose", payload)


def test_carried_pose_endpoint_forwards_only_explicit_observed_support_policy():
    from tianji_curobo.contracts import NativeTrajectory

    settings = PlanningSettings()
    phase = NativeTrajectory(positions_rad=((0.,) * 7, (.1,) * 7), joint_names=tuple(f"Joint{i}_L" for i in range(1, 8)),
                             interpolation_dt_s=.02, planning_time_s=.1, position_error_m=0., rotation_error_rad=0., scene_token="scene-2")
    class Runtime:
        geometry_solver_profile = settings.solver_payload()
        def _scene_snapshot_unlocked(self):
            return {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2,
                    "target_id": "sam-source", "exclusion_state": "selected"}
        def configure_planning(self, settings):
            pass
        def plan_carry_pose_scoped(self, request, carrying, feedback, support_contact_policy=None):
            assert support_contact_policy == "observed_plane_normal_departure"
            return phase, {"optimizer_attachment": True}
    payload = {"arm": "A", "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
               "goal_position_m": [.5, .1, .8], "goal_quaternion_wxyz": [1, 0, 0, 0],
               "scene_token": "scene-2", "planning_settings": settings.model_dump(), "carrying_binding": binding(),
               "gripper_joint_positions_rad": {"A": .55, "B": 0.}, "support_contact_policy": "observed_plane_normal_departure"}
    assert geometry_service.run_operation(Runtime(), "/tools/plan_carried_pose", payload)["attached"]


def test_support_frame_scope_clears_request_and_keeps_exact_configuration_after_error():
    runtime = geometry_service.GeometryRuntime()
    frame = {"arm": "A", "link_name": "jev_support_axis_A"}
    destroyed = []
    with pytest.raises(RuntimeError), runtime.carrying_scope(None, {"A": .55, "B": 0.}, support_frame=frame):
        assert runtime.support_frame is frame
        runtime._compound_backends["A"] = SimpleNamespace(destroy=lambda: destroyed.append(True))
        raise RuntimeError("native prefix failed")
    assert runtime.support_frame is None and runtime.gripper_feedback is None
    assert runtime._compound_backends and destroyed == []
    runtime.clear_carrying()
    assert runtime._compound_backends and destroyed == []
