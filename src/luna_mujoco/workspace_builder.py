"""Offline-only public cuRobo IK sweep; production selection only reads its atlas."""
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import TOOL_PROFILE
from .workspace import MODEL_POLICY, model_fingerprint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vendor'))


def workspace_robot(root, arm, *, urdf_path=None):
    import yaml
    from tianji_curobo.curobo_robot_config import create_single_arm_robot_config

    robot = create_single_arm_robot_config(
        yaml.safe_load((root / 'resources/curobo/marvin_gui_dual.yml').read_text(encoding='utf8')), arm,
        urdf_path=urdf_path or root / 'assets/marvin_urdf/marvin_with_grippers.urdf',
        asset_root_path=root / 'assets/marvin_urdf')
    kin = robot['robot_cfg']['kinematics']
    tool = 'jev_workspace_center_' + arm
    kin.setdefault('extra_links', {})[tool] = {'fixed_transform': [0., 0., TOOL_PROFILE[arm]['tcp_ee_z_m'], 1., 0., 0., 0.],
        'joint_name': tool + '_joint', 'joint_type': 'FIXED', 'link_name': tool,
        'parent_link_name': kin['tool_frames'][0]}
    kin['tool_frames'] = [tool]
    return robot


def workspace_bound(root, arm):
    """Conservative sphere from first active joint and remaining link lengths."""
    tree = ET.parse(root / 'assets/marvin_urdf/marvin_with_grippers.urdf')
    parents = {j.find('child').get('link'): j for j in tree.findall('joint')}
    link = 'TCP_Link_' + ('L' if arm == 'A' else 'R')
    chain = []
    while link in parents:
        joint = parents[link]
        chain.append(joint)
        link = joint.find('parent').get('link')
    transform, center, radius = np.eye(4), None, TOOL_PROFILE[arm]['tcp_ee_z_m']
    for joint in reversed(chain):
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        if center is not None:
            radius += np.linalg.norm(xyz)
        frame = np.eye(4)
        frame[:3, 3] = xyz
        if origin is not None:
            frame[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        transform = transform @ frame
        if center is None and joint.get('type') != 'fixed':
            center = transform[:3, 3].copy()
    return center, float(radius)


class WorkspaceIK:
    def __init__(self, root, arm, *, batch_size=1024, num_seeds=16):
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose, Pose, ToolPoseCriteria
        from tianji_curobo.curobo_service import materialize_urdf

        urdf = materialize_urdf(root / 'assets/marvin_urdf/marvin_with_grippers.urdf',
            marvin_meshes=root / 'assets/marvin_urdf/meshes', rag_meshes=root / 'assets/marvin_urdf/rag100_050/meshes')
        robot = workspace_robot(root, arm, urdf_path=urdf)
        self.collision_links = robot['robot_cfg']['kinematics']['collision_link_names']
        self.tool = robot['robot_cfg']['kinematics']['tool_frames'][0]
        self.batch_size, self.num_seeds = batch_size, num_seeds
        self.torch, self.Pose, self.GoalToolPose = torch, Pose, GoalToolPose
        cfg = InverseKinematicsCfg.create(robot=robot, num_seeds=num_seeds, seed_solver_num_seeds=num_seeds * 2,
            max_batch_size=batch_size, self_collision_check=True, scene_model=None,
            position_tolerance=.002, orientation_tolerance=3.15, random_seed=123)
        self.solver = InverseKinematics(cfg)
        self.solver.update_tool_pose_criteria({self.tool: ToolPoseCriteria.track_position()})

    def solve(self, positions):
        torch = self.torch
        count = len(positions)
        padded = np.pad(positions, ((0, self.batch_size - count), (0, 0)), mode='edge')
        tensor = torch.as_tensor(padded, device=self.solver.device_cfg.device, dtype=self.solver.device_cfg.dtype)
        quaternions = torch.zeros((self.batch_size, 4), device=tensor.device, dtype=tensor.dtype)
        quaternions[:, 0] = 1
        goal = self.GoalToolPose.from_poses({self.tool: self.Pose(position=tensor, quaternion=quaternions)}, num_goalset=1)
        result = self.solver.solve_pose(goal)
        active = self.solver.get_active_js(result.js_solution)
        actual = self.solver.compute_kinematics(active).tool_poses[self.tool].position.reshape(self.batch_size, 3)
        error = torch.linalg.vector_norm(actual - tensor, dim=-1)
        valid = (result.success.flatten() & (error <= .002)).detach().cpu().numpy()[:count]
        return valid, active.position.reshape(self.batch_size, -1).detach().cpu().numpy()[:count]


def build_workspace(root, arm, output, *, resolution=.01, batch_size=1024, num_seeds=16, max_batches=None):
    if (not np.isfinite(resolution) or resolution <= 0 or batch_size < 1 or num_seeds < 1
            or (max_batches is not None and max_batches < 1)):
        raise ValueError('invalid_workspace_build_budget')
    center, radius = workspace_bound(root, arm)
    origin = np.floor((center - radius) / resolution) * resolution
    shape = np.ceil((center + radius - origin) / resolution).astype(int) + 1
    indices = np.indices(shape).reshape(3, -1).T
    points = origin + indices * resolution
    candidates = np.flatnonzero(np.linalg.norm(points - center, axis=1) <= radius)
    started = time.monotonic()
    solver = WorkspaceIK(root, arm, batch_size=batch_size, num_seeds=num_seeds)
    reachable = np.zeros(len(points), dtype=bool)
    witness_indices, witness_joints = [], []
    total = len(candidates) if max_batches is None else min(len(candidates), max_batches * batch_size)
    for start in range(0, total, batch_size):
        subset = candidates[start:min(start + batch_size, total)]
        valid, joints = solver.solve(points[subset])
        reachable[subset] = valid
        witness_indices.append(subset[valid])
        witness_joints.append(joints[valid])
        if start == 0 or (start // batch_size + 1) % 20 == 0 or start + batch_size >= total:
            print(json.dumps({'arm': arm, 'checked': min(start + batch_size, total), 'total': total,
                'reachable': int(reachable.sum()), 'elapsed_s': round(time.monotonic() - started, 2)}), flush=True)
    metadata = {'schema_version': 1, 'arm': arm, 'frame': 'Link_Base', 'policy': MODEL_POLICY,
        'model_fingerprint': model_fingerprint(root), 'origin_m': origin.tolist(), 'resolution_m': resolution,
        'bound_center_m': center.tolist(), 'bound_radius_m': radius, 'collision_links': solver.collision_links,
        'reference_offset_m': [0., 0., TOOL_PROFILE[arm]['tcp_ee_z_m']], 'orientation_constrained': False,
        'self_collision_checked': True, 'peer_arm_included': False, 'environment_included': False,
        'num_seeds': num_seeds, 'seed_solver_num_seeds': num_seeds * 2, 'position_tolerance_m': .002,
        'sdk_commit': '8e734f3ced1df898990bcd92de40abce475907db',
        'checked_points': total, 'reachable_points': int(reachable.sum()),
        'complete': total == len(candidates), 'elapsed_s': time.monotonic() - started}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'arm_{arm}.npz', metadata=np.array(json.dumps(metadata)), reachable=reachable.reshape(shape))
    np.savez_compressed(output / f'arm_{arm}_witness.npz', grid_indices=np.concatenate(witness_indices),
        joints=np.concatenate(witness_joints))
    (output / f'arm_{arm}.json').write_text(json.dumps(metadata, indent=2))
    return metadata
