"""Real pinned GPU joint planning and full robot collision validation."""
import json

import numpy as np
import torch
from curobo.types import JointState
from tianji_curobo.contracts import CuroboToolError
from tianji_curobo.curobo_service import materialize_urdf

from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity
from luna_mujoco.planning_settings import PlanningSettings

urdf = materialize_urdf(ROOT / 'assets/marvin_urdf/marvin_with_grippers.urdf',
    marvin_meshes=ROOT / 'assets/marvin_urdf/meshes', rag_meshes=ROOT / 'assets/marvin_urdf/rag100_050/meshes')
runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
settings = PlanningSettings()
runtime.configure_planning(settings)
runtime.geometry_scene = {'scene_token': 'dual-test', 'observation_id': 'test', 'generation': 1,
                          'target_id': 'source', 'disabled_obstacle_names': (), 'exclusion_state': 'observation'}
runtime.geometry_model = {'cuboid': {}, 'voxel': {}}
runtime.geometry_grids = []
runtime.geometry_solver_profile = settings.solver_payload()
feedback = {'A': 0., 'B': 0.}
with runtime.carrying_scope(None, feedback):
    runtime._prepare_scene_cache_unlocked()
    backend = runtime._dual_backend_unlocked()
    planner = backend.motion_planner
    q = planner.default_joint_state.position.reshape(-1).detach().cpu().numpy()
    assert len(q) == 14 and set(planner.tool_frames) == {'TCP_Link_L','TCP_Link_R'}
    starts = {a: np.rad2deg([q[planner.joint_names.index(f'Joint{i}_{suffix}')] for i in range(1,8)]).tolist()
              for a,suffix in [('A','L'),('B','R')]}
    goal = q.copy()
    for suffix in ('L','R'):
        goal[planner.joint_names.index('Joint1_'+suffix)] += .03
    state = JointState.from_position(torch.as_tensor(goal[None], **planner.device_cfg.as_torch_dict()), joint_names=planner.joint_names)
    poses = planner.compute_kinematics(state).tool_poses
    matrices = {a: poses.get_link_pose('TCP_Link_'+suffix).get_matrix().reshape(-1,4,4)[0].detach().cpu().tolist()
                for a,suffix in [('A','L'),('B','R')]}
base = {'scene_token': 'dual-test', 'planning_settings': settings.model_dump(), 'gripper_joint_positions_rad': feedback}
print(json.dumps({'stage':'start', 'q':starts}),flush=True)
start_check = run_operation(runtime, '/tools/validate_dual_observation', base | {'points_deg':{a:[starts[a],starts[a]] for a in ('A','B')}, 'expected_scene_token':'dual-test'})
print(json.dumps({'stage':'start_valid', 'check':start_check}),flush=True)
result = run_operation(runtime, '/tools/plan_dual_observation', base | {'start_joints_deg': starts, 'goal_transforms_world': matrices})
assert len(result['trajectories']['A']) == len(result['trajectories']['B'])
checked = run_operation(runtime, '/tools/validate_dual_observation', base | {'points_deg': result['trajectories'], 'expected_scene_token':'dual-test'})
assert checked['collision_free']
# An observed obstacle at the moving wrist must be rejected by the same full-robot checker.
point = np.asarray(matrices['A'])[:3,3]
runtime.geometry_model = {'cuboid': {'wrist_obstacle': {'pose': [*point.tolist(),1,0,0,0], 'dims':[.12]*3}}, 'voxel':{}}
# A new measured world is published with a new identity; mutating a model under
# the previous token deliberately leaves the existing upload/cache valid.
runtime.geometry_scene = runtime.geometry_scene | {'scene_token': 'dual-test-obstacle'}
base = base | {'scene_token': 'dual-test-obstacle'}
try:
    run_operation(runtime, '/tools/validate_dual_observation', base | {'points_deg':result['trajectories'], 'expected_scene_token':'dual-test-obstacle'})
except CuroboToolError:
    rejected = True
else:
    raise AssertionError('full robot world collision was not rejected')
print(json.dumps({'sdk':sdk_identity()[0], 'joint_count':14, 'synchronized_samples':len(result['trajectories']['A']),
                  'dt':result['sample_dt_s'], 'world_obstacle_rejected':rejected, 'torch':torch.__version__, 'cuda':torch.version.cuda}),flush=True)
runtime.clear_carrying()
