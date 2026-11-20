"""Application FK through cuRoboV2-Fast; never imports a physics engine."""
import numpy as np
from tianji_curobo.robot_geometry import robot_geometry_client

from .geometry import front_tcp_pose


def link_poses(arms, links=None):
    return {name: np.asarray(value) for name, value in
            robot_geometry_client().query(arms, links=links)['link_poses'].items()}


def zero_configuration(grippers=None):
    return {arm: {'joints_rad': [0.] * 7,
                  'gripper_position_rad': (grippers or {}).get(arm, 0.)} for arm in ('A', 'B')}


def robot_tcp(arms, arm):
    tool = 'TCP_Link_L' if arm == 'A' else 'TCP_Link_R'
    return front_tcp_pose(link_poses(arms, [tool])[tool], arm)
