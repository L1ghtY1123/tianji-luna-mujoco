"""Destination robot configurations screen release geometry, never task order."""
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .carrying import configure_gripper_feedback
from .demo_placement import sphere_array
from .geometry import rigid_matrix
from .kinematics import robot_model
from .model_builder import ARM_JOINTS


def configuration_spheres(configuration):
    """Own calibrated robot collision spheres; no scene assets or simulator state."""
    if not isinstance(configuration, dict):
        raise TypeError('placement_requires_complete_robot_configuration')
    arms = set(configuration.get('joints_deg', {}))
    partial = configuration.get('scope') == 'specified_arms'
    if (not configuration.get('id') or not arms or not arms <= {'A', 'B'}
            or (not partial and arms != {'A', 'B'})
            or set(configuration.get('gripper_positions_rad', {})) != arms):
        raise ValueError('placement_requires_complete_robot_configuration')
    # Unspecified chains only initialize FK; their dependent bodies are removed
    # below and never become predicted robot occupancy or a motion certificate.
    joints = np.asarray([configuration['joints_deg'].get(arm, [0.] * 7) for arm in ('A', 'B')], float)
    grippers = np.asarray([configuration['gripper_positions_rad'].get(arm, 0.) for arm in ('A', 'B')], float)
    if (joints.shape != (2, 7) or grippers.shape != (2,)
            or not np.isfinite(joints).all() or not np.isfinite(grippers).all()):
        raise ValueError('placement_requires_finite_robot_configuration')
    spheres, links = _configuration_spheres(tuple(joints.reshape(-1)), tuple(grippers), tuple(sorted(arms)))
    return np.asarray(spheres), links


@lru_cache(maxsize=16)
def _configuration_spheres(joints_deg, grippers_rad, required_arms):
    root = Path(__file__).resolve().parents[2]
    urdf = root / 'assets/marvin_urdf/marvin_with_grippers.urdf'
    config = yaml.safe_load((root / 'resources/curobo/marvin_gui_dual.yml').read_text(encoding='utf-8'))
    config = configure_gripper_feedback(config, dict(zip(('A', 'B'), grippers_rad)), urdf)
    kin = config['robot_cfg']['kinematics']
    model = robot_model()
    data = mujoco.MjData(model)
    positions = dict(kin['lock_joints'])
    for joint in ET.parse(urdf).getroot().findall('joint'):
        mimic = joint.find('mimic')
        if mimic is not None:
            positions[joint.get('name')] = (positions[mimic.get('joint')] * float(mimic.get('multiplier', 1))
                                            + float(mimic.get('offset', 0)))
    for name, value in zip((*ARM_JOINTS['A'], *ARM_JOINTS['B']), np.deg2rad(joints_deg)):
        positions[name] = value
    for name, value in positions.items():
        joint = model.joint(name)
        if joint.limited[0] and not joint.range[0] <= value <= joint.range[1]:
            raise ValueError('placement_robot_configuration_outside_joint_limits:' + name)
        data.qpos[int(joint.qposadr[0])] = value
    mujoco.mj_fwdPosition(model, data)
    excluded_roots = {int(model.joint(ARM_JOINTS[arm][0]).bodyid[0])
                      for arm in ('A', 'B') if arm not in required_arms}
    spheres, links = [], []
    for link in kin['collision_link_names']:
        body = model.body(link).id
        ancestor = body
        while ancestor and ancestor not in excluded_roots:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor in excluded_roots:
            continue
        rotation = data.xmat[body].reshape(3, 3)
        for sphere in kin['collision_spheres'][link]:
            center = data.xpos[body] + rotation @ np.asarray(sphere['center'])
            radius = float(sphere['radius']) + kin['collision_sphere_buffer'].get(link, 0.)
            if radius > 0:
                spheres.append((*center.tolist(), radius))
                links.append(link)
    return tuple(spheres), tuple(links)


def release_configuration_clearance(goal, holding, configurations, *, surface_voxel_size_m):
    """Screen the bound carrying spheres throughout a bounded vertical settling drop.

    This predicts compatibility with a destination's required robot pose. The
    later actual map and motion checks still decide whether that pose is safe.
    """
    transform = rigid_matrix(goal['T_world_tcp'])
    reference = goal['release_reference']
    attachment = sphere_array(reference['attachment_spheres_tool_m'])
    world = attachment[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    points = np.asarray(holding['points_tool_m'], float) @ transform[:3, :3].T + transform[:3, 3]
    drop = max(0., float(points[:, 2].min() - reference['floor_z_m']))
    voxel = 0. if surface_voxel_size_m is None else float(surface_voxel_size_m)
    if not np.isfinite(voxel) or voxel < 0 or (voxel == 0 and surface_voxel_size_m is not None):
        raise ValueError('placement_requires_positive_surface_voxel_size')
    # A measured point can lie anywhere in a native surface cell. Reserve the
    # full cell diagonal without adding invented surfaces to the collision map.
    # Explicit None denotes a primitive/mesh world with no surface-cell error;
    # the uploaded attachment still includes its measured/model uncertainty.
    reserve = float(np.sqrt(3.) * voxel)
    required = float(reference['required_clearance_m'])
    rows = []
    for configuration in configurations:
        robot, links = configuration_spheres(configuration)
        delta = world[:, None, :] - robot[None, :, :3]
        delta[:, :, 2] -= np.clip(delta[:, :, 2], 0., drop)
        separation = np.linalg.norm(delta, axis=2) - attachment[:, None, 3] - robot[None, :, 3]
        index = np.unravel_index(np.argmin(separation), separation.shape)
        gap = float(separation[index])
        rows.append({'configuration_id': configuration['id'], 'compatible': gap >= required + reserve,
            'configuration_scope': configuration.get('scope', 'whole_robot'),
            'required_arms': sorted(configuration.get('joints_deg', {})),
            'minimum_sphere_separation_m': gap, 'required_clearance_m': required,
            'surface_cell_reserve_m': reserve, 'settling_drop_bound_m': drop,
            'closest_robot_link': links[index[1]], 'attachment_sphere_index': int(index[0])})
    source = 'measured_carried_surface_and_own_robot_kinematics'
    assumption = 'observed_surface_only'
    if holding.get('cad_tool') is not None:
        source = 'registered_cad_carried_volume_and_own_robot_kinematics'
        assumption = 'registered_cad_model_and_validated_pose'
    elif holding.get('obb_tool') is not None:
        source = 'observed_obb_carried_volume_and_own_robot_kinematics'
        assumption = 'observed_obb_solid_approximation'
    return {'compatible': all(row['compatible'] for row in rows), 'configurations': rows,
        'source': source,
        'scope': 'destination_configuration_screening', 'motion_collision_certified': False,
        'assumptions': ['rigid_object_vertical_settling', assumption],
        'target_reference_world_xy_m': reference['target_reference_world_xy_m']}
