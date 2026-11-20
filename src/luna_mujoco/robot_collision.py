"""Own-model collision spheres evaluated from explicit joint configurations."""
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .carrying import configure_gripper_feedback
from .kinematics import link_poses


def configuration_spheres(configuration):
    """Own calibrated robot collision spheres; no scene assets or simulator state."""
    if not isinstance(configuration, dict):
        raise TypeError('robot_configuration_requires_complete_robot_configuration')
    arms = set(configuration.get('joints_deg', {}))
    partial = configuration.get('scope') == 'specified_arms'
    if (not configuration.get('id') or not arms or not arms <= {'A', 'B'}
            or (not partial and arms != {'A', 'B'})
            or set(configuration.get('gripper_positions_rad', {})) != arms):
        raise ValueError('robot_configuration_requires_complete_robot_configuration')
    # Unspecified chains only initialize FK; their dependent bodies are removed
    # below and never become predicted robot occupancy or a motion certificate.
    joints = np.asarray([configuration['joints_deg'].get(arm, [0.] * 7) for arm in ('A', 'B')], float)
    grippers = np.asarray([configuration['gripper_positions_rad'].get(arm, 0.) for arm in ('A', 'B')], float)
    if (joints.shape != (2, 7) or grippers.shape != (2,)
            or not np.isfinite(joints).all() or not np.isfinite(grippers).all()):
        raise ValueError('robot_configuration_requires_finite_robot_configuration')
    spheres, links = _configuration_spheres(tuple(joints.reshape(-1)), tuple(grippers), tuple(sorted(arms)))
    return np.asarray(spheres), links


@lru_cache(maxsize=16)
def _configuration_spheres(joints_deg, grippers_rad, required_arms):
    root = Path(__file__).resolve().parents[2]
    urdf = root / 'assets/marvin_urdf/marvin_with_grippers.urdf'
    config = yaml.safe_load((root / 'resources/robot/marvin_gui_dual.yml').read_text(encoding='utf-8'))
    config = configure_gripper_feedback(config, dict(zip(('A', 'B'), grippers_rad)), urdf)
    kin = config['robot_cfg']['kinematics']
    arms = {arm: {'joints_rad': np.deg2rad(joints_deg[i * 7:(i + 1) * 7]).tolist(),
                  'gripper_position_rad': grippers_rad[i]} for i, arm in enumerate(('A', 'B'))}
    poses = link_poses(arms, kin['collision_link_names'])
    parents = {joint.find('child').get('link'): (joint.find('parent').get('link'), joint.get('name'))
               for joint in ET.parse(urdf).getroot().findall('joint')}
    excluded_joints = {'Joint1_L' if arm == 'A' else 'Joint1_R' for arm in ('A', 'B') if arm not in required_arms}
    spheres, links = [], []
    for link in kin['collision_link_names']:
        ancestor = link
        excluded = False
        while ancestor in parents:
            ancestor, joint = parents[ancestor]
            if joint in excluded_joints:
                excluded = True
                break
        if excluded:
            continue
        rotation = poses[link][:3, :3]
        for sphere in kin['collision_spheres'][link]:
            center = poses[link][:3, 3] + rotation @ np.asarray(sphere['center'])
            buffer = kin['collision_sphere_buffer']
            if isinstance(buffer, dict):
                buffer = buffer.get(link, 0.)
            radius = float(sphere['radius']) + buffer
            if radius > 0:
                spheres.append((*center.tolist(), radius))
                links.append(link)
    return tuple(spheres), tuple(links)
