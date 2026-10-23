import sys
from types import SimpleNamespace

from luna_mujoco import carrying


def test_feedback_updates_reuse_checker_but_keep_exact_kinematics(monkeypatch):
    created = []

    class Checker:
        def __init__(self, cfg):
            created.append(cfg)
            self.kinematics = SimpleNamespace()

    monkeypatch.setitem(sys.modules, 'curobo.collision_checking', SimpleNamespace(
        RobotCollisionChecker=Checker, RobotCollisionCheckerCfg=SimpleNamespace(load_from_config=lambda **kw: kw)))
    robot = {'robot_cfg': {'kinematics': {'lock_joints': {'gripL_finger_joint': .1, 'gripR_finger_joint': .2},
                                          'cspace': {'joint_names': ['arm'], 'default_joint_position': [0.]},
                                          'collision_spheres': {'finger': [1]}}}}
    runtime = SimpleNamespace(_command_checkers={}, planning_settings=SimpleNamespace(trajectory_collision_clearance_m=.01))
    planner = SimpleNamespace(device_cfg=object(), scene_collision_checker=object())
    first = carrying.command_checker(runtime, planner, robot, 'scene')
    robot['robot_cfg']['kinematics']['lock_joints']['gripL_finger_joint'] = .100001
    second = carrying.command_checker(runtime, planner, robot, 'scene')
    assert first is second and len(created) == 1
    dynamic = created[0]['robot_config']['robot_cfg']['kinematics']
    assert not dynamic['lock_joints']
    assert dynamic['cspace']['joint_names'] == ['arm', 'gripL_finger_joint', 'gripR_finger_joint']
    assert robot['robot_cfg']['kinematics']['lock_joints']['gripL_finger_joint'] == .100001
    robot['robot_cfg']['kinematics']['collision_spheres']['carried'] = [2]
    assert carrying.command_checker(runtime, planner, robot, 'scene') is not first
    assert carrying.command_checker(runtime, planner, robot, 'new-scene') is not second
