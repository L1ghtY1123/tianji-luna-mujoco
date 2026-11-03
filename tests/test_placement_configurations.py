"""A valid drop must not obstruct a destination's fixed processing posture."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.placement_constraints import configuration_spheres, release_configuration_clearance
from luna_mujoco.vla_configuration import vla_unpack_configuration


def fixture():
    from luna_mujoco.geometry import front_tcp_pose
    value = json.loads((Path(__file__).parent / 'fixtures/vla_placement_configuration.json').read_text())
    holding = value['holding']
    holding['frame'] = 'gripper_front'
    holding['points_tool_m'] = (np.asarray(holding['points_tool_m']) - [0, 0, .185]).tolist()
    ref = holding['grasp_support_reference']
    ref['T_world_tcp_at_grasp'] = front_tcp_pose(ref['T_world_tcp_at_grasp'], holding['arm']).tolist()
    for balls in (holding['attachment_evidence']['attachment_spheres_tool_m'], value['gripper_spheres_tool_m']):
        for ball in balls:
            ball[2] -= .185
    return value


def configurations():
    return vla_unpack_configuration()['input_region']['required_robot_configurations']


def test_robot_fk_matches_archived_public_curobo_collision_sphere():
    spheres, links = configuration_spheres(configurations()[0])
    # Public RobotCollisionChecker run36 goal sphere336, independently archived.
    reference = [.43474501371383667, .24988383054733276, .7091424465179443, .017005769535899162]
    candidates = spheres[np.array(links) == 'gripL_Base']
    assert np.min(np.max(np.abs(candidates - reference), axis=1)) < 5e-7


def test_vertical_settling_cannot_tunnel_past_required_robot_pose(monkeypatch):
    monkeypatch.setattr('luna_mujoco.placement_constraints.configuration_spheres',
        lambda value: (np.array([[0., 0., .5, .03]]), ('fixture_link',)))
    goal = {'T_world_tcp':np.eye(4).tolist(), 'release_reference': {
        'attachment_spheres_tool_m':[[0.,0.,1.,.02]], 'floor_z_m':0.,
        'required_clearance_m':.01, 'target_reference_world_xy_m':[0.,0.]}}
    result = release_configuration_clearance(goal, {'points_tool_m':[[0.,0.,1.]]},
        [{'id':'posture'}], surface_voxel_size_m=.02)
    assert not result['compatible']
    assert result['configurations'][0]['settling_drop_bound_m'] == 1.


def test_primitive_scene_has_no_voxel_reserve_but_still_checks_robot_clearance(monkeypatch):
    monkeypatch.setattr('luna_mujoco.placement_constraints.configuration_spheres',
        lambda value: (np.array([[0., 0., .5, .03]]), ('fixture_link',)))
    goal = {'T_world_tcp':np.eye(4).tolist(), 'release_reference': {
        'attachment_spheres_tool_m':[[0.,0.,1.,.02]], 'floor_z_m':0.,
        'required_clearance_m':.01, 'target_reference_world_xy_m':[0.,0.]}}
    result = release_configuration_clearance(goal, {'points_tool_m':[[0.,0.,1.]]},
        [{'id':'posture'}], surface_voxel_size_m=None)
    assert result['configurations'][0]['surface_cell_reserve_m'] == 0.
    assert not result['compatible']


@pytest.mark.parametrize('change', ['missing_arm', 'nan', 'joint_limit'])
def test_invalid_required_robot_configuration_is_not_ignored(change):
    config = deepcopy(configurations()[0])
    if change == 'missing_arm':
        del config['joints_deg']['A']
    else:
        config['joints_deg']['A'][0] = float('nan') if change == 'nan' else 100000.
    with pytest.raises(ValueError):
        configuration_spheres(config)


def test_vla_destination_constraints_are_explicit_and_other_regions_are_unchanged():
    from luna_mujoco.demo_scene import demo_region
    from luna_mujoco.robot_skills.operations import configured_destination
    observation = {'observation_id':'capture','table':{'surface_height_m':.5}}
    config = vla_unpack_configuration()
    destination = configured_destination(observation, config['input_region'])
    constraints = destination['placement_reference']['required_robot_configurations']
    assert constraints[0]['joints_deg'] == config['preparation_joints_deg']
    assert 'required_robot_configurations' not in configured_destination(
        observation, demo_region())['placement_reference']
    assert destination['visual_membership'] == 'excluded'
