"""Offline GPU parity against fresh public SDK instances; no simulation motion."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.perception import Mapper, MapperCfg
from curobo.types import CameraObservation, DeviceCfg, Pose
from tianji_curobo.curobo_robot_config import create_dual_arm_robot_config

from luna_mujoco.carrying import command_checker, configure_gripper_feedback, create_carrying_robot_config
from luna_mujoco.instance_mapping import InstanceMapper

ROOT = Path(__file__).resolve().parents[1]


def mapping_parity():
    device = 'cuda'
    cfg = MapperCfg(extent_meters_xyz=(.4, .4, .4), extent_esdf_meters_xyz=(.8, .8, .8),
        voxel_size=.02, esdf_voxel_size=.02, grid_center=torch.tensor([0., 0., 1.], device=device),
        truncation_distance=.03, minimum_tsdf_weight=.5, depth_minimum_distance=.05,
        depth_maximum_distance=3., decay_factor=1., frustum_decay_factor=1., enable_static=False,
        num_cameras=3, image_height=48, image_width=64, seeding_method='scatter')
    depth = torch.zeros((3, 48, 64), device=device)
    depth[:, 15:33, 23:41] = 1.
    observation = CameraObservation(depth_image=depth,
        rgb_image=torch.zeros((3, 48, 64, 3), device=device, dtype=torch.uint8), depth_to_meter=1.,
        intrinsics=torch.tensor([[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]], device=device).repeat(3, 1, 1),
        pose=Pose(position=torch.zeros((3, 3), device=device),
                  quaternion=torch.tensor([[1., 0., 0., 0.]], device=device).repeat(3, 1)))
    direct, wrapped = Mapper(cfg), Mapper(cfg)
    direct.integrate(observation)
    expected = direct.compute_esdf().feature_tensor.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    timing = InstanceMapper.integrate(wrapped, stream, observation, None)
    actual_grid = wrapped.integrator.get_voxel_grid()
    torch.testing.assert_close(expected, actual_grid.feature_tensor, rtol=0., atol=0.)
    assert abs(float(actual_grid.voxel_size) - .02) < 1e-9
    return {'voxel_size_m': .02, 'same_esdf_cells': expected.numel(),
            'maximum_esdf_difference_m': 0., 'timings': timing}


def checker_parity():
    from audit_planning_geometry import binding_for, cases

    device = DeviceCfg()
    urdf = ROOT / 'assets/marvin_urdf/marvin_with_grippers.urdf'
    original = yaml.safe_load((ROOT / 'resources/curobo/marvin_gui_dual.yml').read_text())
    base = create_dual_arm_robot_config(original, urdf_path=urdf, asset_root_path=ROOT / 'assets')
    world = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=base, device_cfg=device, collision_activation_distance=.01,
        scene_model={'cuboid': {'floor': {'pose': [0., 0., -.1, 1., 0., 0., 0.], 'dims': [2., 2., .1]}}}))
    planner = SimpleNamespace(device_cfg=device, scene_collision_checker=world.scene_model)
    runtime = SimpleNamespace(_command_checkers={}, planning_settings=SimpleNamespace(trajectory_collision_clearance_m=.01))
    results = []
    for case in cases():
        for delta in (0., .00001):
            feedback = {name: angle + delta for name, angle in case['feedback'].items()}
            config = configure_gripper_feedback(base, feedback, urdf)
            binding = binding_for(case)
            if binding is not None:
                config, _ = create_carrying_robot_config(config, binding)
            reused = command_checker(runtime, planner, config, 'audit')
            direct = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                robot_config=config, device_cfg=device, scene_collision_checker=planner.scene_collision_checker,
                collision_activation_distance=.01))
            positions = dict(zip([f'Joint{i}_{s}' for s in ('L', 'R') for i in range(1, 8)],
                                 np.concatenate([case['joints']['A'], case['joints']['B']]), strict=True))
            positions.update(gripL_finger_joint=feedback['A'], gripR_finger_joint=feedback['B'])
            q = torch.tensor([[[positions[name] for name in direct.kinematics.joint_names]]],
                             **device.as_torch_dict())
            active_q = torch.tensor([[[positions[name] for name in reused.kinematics.joint_names]]],
                                    **device.as_torch_dict())
            a, b = reused.get_kinematics(active_q), direct.get_kinematics(q)
            torch.testing.assert_close(a.robot_spheres, b.robot_spheres, rtol=0., atol=2e-6)
            torch.testing.assert_close(reused.get_self_collision_distance(a.robot_spheres),
                                       direct.get_self_collision_distance(b.robot_spheres), rtol=0., atol=2e-6)
            torch.testing.assert_close(reused.get_collision_distance(a), direct.get_collision_distance(b),
                                       rtol=0., atol=2e-6)
            results.append({'arm': case['arm'], 'phase': case['phase'], 'feedback_delta': delta,
                            'maximum_sphere_error_m': (a.robot_spheres - b.robot_spheres).abs().max().item()})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'mapping': mapping_parity(), 'checker': checker_parity()}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'mapping': report['mapping'], 'checker_cases': len(report['checker']),
                      'maximum_sphere_error_m': max(r['maximum_sphere_error_m'] for r in report['checker'])}))
