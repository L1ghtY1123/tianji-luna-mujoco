import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from luna_mujoco.geometry import TOOL_PROFILE, front_tcp_pose, grasp_targets
from luna_mujoco.kinematics import robot_model, robot_tcp
from luna_mujoco.model_builder import ARM_JOINTS, TCP_BODIES


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_graspgen_conversion_targets_front_without_subtracting_mount_length(arm):
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler('xyz', [21, -34, 65], degrees=True).as_matrix()
    grasp[:3, 3] = [.4, -.2, .6]
    expected = grasp.copy()
    if arm == 'A':
        expected[:3, :2] *= -1
    expected[:3, 3] += expected[:3, :3] @ [0, 0, TOOL_PROFILE['grasp_ee_z_m']]
    np.testing.assert_allclose(grasp_targets(grasp[None], arm)[0], expected)


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_own_fk_reports_front_not_mount(arm):
    import mujoco

    model = robot_model()
    data = mujoco.MjData(model)
    arms = {a: {'joints_rad': [0.] * 7} for a in ARM_JOINTS}
    mujoco.mj_forward(model, data)
    body = model.body(TCP_BODIES[arm]).id
    mount = np.eye(4)
    mount[:3, :3] = data.xmat[body].reshape(3, 3)
    mount[:3, 3] = data.xpos[body]
    np.testing.assert_allclose(robot_tcp(arms, arm), front_tcp_pose(mount, arm))


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_attachment_link_is_front_frame_with_measured_points_unchanged(arm):
    from luna_mujoco.resident_robot import reserve_attachment_links

    robot = {'robot_cfg': {'kinematics': {'collision_link_names': [], 'collision_spheres': {}}}}
    kin = reserve_attachment_links(robot)['robot_cfg']['kinematics']
    link = kin['extra_links']['jev_carried_object_' + arm]
    np.testing.assert_allclose(link['fixed_transform'][:3], [0, 0, .185])
    # Sphere coordinates are front-local; the parent remains the physical mount
    # with one explicit fixed transform, rather than an implicit zero offset.
    assert link['parent_link_name'] == TCP_BODIES[arm]


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_migrated_measured_points_and_spheres_keep_the_same_world_geometry(arm):
    from dataclasses import replace

    from luna_mujoco.carrying import CarryingBinding, carrying_spheres

    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_euler('xyz', [31, -57, 112], degrees=True).as_matrix()
    mount[:3, 3] = [.42, .3 if arm == 'A' else -.3, .81]
    points = np.random.default_rng(23).uniform(-.02, .02, (100, 3)) + [0, 0, .18]
    old = CarryingBinding(arm, 'owner', 'obs', 1, 'scene', 'target', 'tool', 'held',
                           tuple(map(tuple, points)), .005)
    new = replace(old, frame='gripper_front', points_tool_m=tuple(map(tuple, points - [0, 0, .185])))
    front = front_tcp_pose(mount, arm)
    np.testing.assert_allclose(points @ mount[:3, :3].T + mount[:3, 3],
        np.asarray(new.points_tool_m) @ front[:3, :3].T + front[:3, 3], atol=1e-12)
    old_balls, new_balls = (np.asarray(carrying_spheres(b)) for b in (old, new))
    np.testing.assert_allclose(old_balls[:, :3] @ mount[:3, :3].T + mount[:3, 3],
        new_balls[:, :3] @ front[:3, :3].T + front[:3, 3], atol=1e-12)
    np.testing.assert_allclose(old_balls[:, 3], new_balls[:, 3], atol=1e-12)


def test_legacy_holding_cannot_be_relabelled_as_front():
    from luna_mujoco.carrying import held_collision_binding

    with pytest.raises(ValueError, match='holding_tcp_frame_mismatch'):
        held_collision_binding({'frame': 'tool', 'points_tool_m': [[0, 0, .18]]}, {}, 'A', 'target')
