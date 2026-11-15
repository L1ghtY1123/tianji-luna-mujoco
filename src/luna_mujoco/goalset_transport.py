"""Serialize a single official grasp goalset request and its selected trajectory."""
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import rigid_matrix
from .trajectory_postprocessing import _phase_samples


def world_z_lift_evidence(start_tcp, lift_tcp_matrices, height_m, tolerance_m):
    start = rigid_matrix(start_tcp)[:3, 3]
    matrices = np.asarray(lift_tcp_matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or len(matrices) < 2 or not np.isfinite(matrices).all():
        raise ValueError('lift_requires_finite_fk_path')
    offsets = matrices[:, :3, 3] - start
    lateral = float(np.linalg.norm(offsets[:, :2], axis=1).max())
    height = float(offsets[-1, 2])
    lowest = float(offsets[:, 2].min())
    downward_step = max(0., float(-np.diff(offsets[:, 2]).min()))
    accepted = (lateral <= tolerance_m and abs(height - height_m) <= tolerance_m
                and lowest >= -tolerance_m and downward_step <= tolerance_m)
    return {'accepted': accepted, 'frame': 'world', 'axis': [0., 0., 1.],
            'source': 'curobo_public_compute_kinematics', 'checked_sample_count': len(matrices),
            'requested_height_m': height_m, 'actual_height_m': height,
            'maximum_horizontal_deviation_m': lateral, 'minimum_height_m': lowest,
            'maximum_downward_step_m': downward_step, 'position_tolerance_m': tolerance_m}


def grasp_result(backend, payload):
    matrices = np.asarray(payload['goal_matrices_world'], dtype=float)
    count = len(matrices)
    if matrices.shape != (count, 4, 4) or not count or not np.isfinite(matrices).all():
        raise ValueError('invalid_grasp_goalset')
    for matrix in matrices:
        rigid_matrix(matrix)
    quaternion = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    poses = backend._pose_type(
        position=backend._torch.as_tensor(matrices[:, :3, 3], device=backend._device, dtype=backend._dtype),
        quaternion=backend._torch.as_tensor(quaternion, device=backend._device, dtype=backend._dtype))
    goal = backend._goal_tool_pose_type.from_poses({backend.tool_frame: poses},
        ordered_tool_frames=[backend.tool_frame], num_goalset=count)
    state = backend._state(payload['start_joints_deg'])
    planner = backend.motion_planner
    settings = planner.planning_settings
    with planner.optimizer_scene():
        try:
            result = planner.sdk.plan_grasp(goal, state,
                grasp_lift_axis=settings.lift_axis,
                grasp_lift_offset=settings.lift_offset_m,
                grasp_lift_in_tool_frame=settings.lift_in_tool_frame,
                disable_collision_links=list(planner.grasp_contact_links))
        finally:
            planner.collision_restore_failed = True
            planner.sdk.enable_link_collision(list(planner.grasp_contact_links))
            planner.collision_restore_failed = False
    flags = {name + '_success': bool(getattr(result, name + '_success').reshape(-1)[0].item())
             for name in ('approach', 'grasp', 'lift')}
    goalset = result.goalset_result
    flags['goalset_success'] = goalset is not None and bool(goalset.success.any().item())
    selected = int(result.goalset_index.reshape(-1)[0].item()) if flags['goalset_success'] else None
    response = {'success': bool(result.success.reshape(-1)[0].item()), **flags,
                'status': str(result.status), 'goalset_index': selected,
                'candidate_count': count, 'max_batch_size': 1, 'phases': {},
                'scene_token': payload['scene_token']}
    response['effective_config'] = {
        'max_batch_size': planner.ik_solver.config.max_batch_size,
        'max_goalset': planner.ik_solver.config.max_goalset,
        'num_ik_seeds': planner.ik_solver.config.num_seeds,
        'num_trajopt_seeds': planner.trajopt_solver.config.num_seeds,
        'use_cuda_graph': planner.ik_solver.config.use_cuda_graph,
        'use_trajopt_for_goalset_selection': False,
        'use_double_pass_trajopt_for_constrained_phases': False,
        'grasp_lift_axis': settings.lift_axis, 'grasp_lift_offset_m': settings.lift_offset_m,
        'grasp_lift_in_tool_frame': settings.lift_in_tool_frame,
    }
    if not response['success']:
        return response
    if selected is None or not 0 <= selected < count:
        raise ValueError('official_goalset_index_out_of_range')
    for name in ('approach', 'grasp', 'lift'):
        q, v, a, names = _phase_samples(result, name, 1)
        length = int(getattr(result, name + '_interpolated_last_tstep').reshape(-1)[0].item())
        if not 2 <= length <= q.shape[2]:
            raise ValueError('invalid_official_grasp_trajectory_length')
        order = [names.index(joint) for joint in backend.joint_names]
        response['phases'][name] = {
            'positions_rad': q[0, 0, :length][:, order].tolist(),
            'velocities_rad_s': v[0, 0, :length][:, order].tolist(),
            'accelerations_rad_s2': a[0, 0, :length][:, order].tolist(),
            'joint_names': list(backend.joint_names),
            'interpolation_dt_s': float(planner.trajopt_solver.config.interpolation_dt)}
    response['grasp_endpoint_fk'] = backend.tool_fk_evidence(
        response['phases']['grasp']['positions_rad'][-1], payload['scene_token'])
    lift_fk = backend.tool_fk_batch_evidence(response['phases']['lift']['positions_rad'], payload['scene_token'])
    response['lift_direction'] = world_z_lift_evidence(response['grasp_endpoint_fk']['T_world_tcp'],
        [row['T_world_tcp'] for row in lift_fk], settings.lift_offset_m, settings.ik_position_threshold_m)
    if not response['lift_direction']['accepted']:
        response.update(success=False, lift_success=False, status='Lift FK path violates the world +Z constraint.')
    return response
