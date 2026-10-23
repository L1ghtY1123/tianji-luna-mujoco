"""Direction filtering preserves GraspGen positions without extra insertion."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from luna_mujoco import demo_grasp
from luna_mujoco.geometry import grasp_targets


def candidate(index, tilt_deg, azimuth_deg=0.):
    azimuth = np.deg2rad(azimuth_deg)
    axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.])
    rotation = Rotation.from_rotvec(
        np.cross([0., 0., -1.], axis) * np.deg2rad(tilt_deg)).as_matrix() @ np.diag([1., -1., -1.])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [.3, .2, .8]
    return {"index": index, "score": .9 - index * .01, "T_world_tcp": pose.tolist()}


def test_cone_boundary_and_radial_direction_reject_before_ik():
    candidates = [candidate(0, 0.), candidate(1, 45.), candidate(2, 45.1),
                  candidate(3, 20., 180.), candidate(4, 90.)]
    original = deepcopy(candidates)
    shifted, accepted, evidence = demo_grasp.prepare_grasp_candidates(
        candidates, robot_base_world_m=[0., 0., 0.], target_reference_world_m=[.4, 0., .5])
    assert [c["index"] for c in accepted] == [0, 1]
    assert evidence["maximum_tilt_deg"] == 45.
    assert evidence["rejected_tilt_count"] == 2
    assert evidence["rejected_toward_robot_count"] == 1
    assert candidates == original
    for source, prepared in zip(original, shifted):
        before, after = np.array(source["T_world_tcp"]), np.array(prepared["T_world_tcp"])
        np.testing.assert_array_equal(after[:3, 3], before[:3, 3])
        np.testing.assert_array_equal(after[:3, :3], before[:3, :3])
        assert not prepared.get('insertion_applied', False)
        assert 'T_world_tcp_before_insertion' not in prepared
    assert evidence['insertion_m'] == 0.
    assert evidence['operation_order'] == ['direction_filter', 'arm_jaw_symmetry', 'batched_ik']


@pytest.mark.parametrize("side", [-1., 1.])
def test_toward_robot_rule_is_radial_for_either_side(side):
    _, accepted, _ = demo_grasp.prepare_grasp_candidates(
        [candidate(0, 30., side * 90.), candidate(1, 30., -side * 90.)],
        robot_base_world_m=[0., 0., 0.], target_reference_world_m=[0., side * .5, .6])
    assert [c["index"] for c in accepted] == [0]


def test_existing_arm_profiles_preserve_z_and_reverse_xy():
    world = np.eye(4)[None]
    left, right = grasp_targets(world, "A")[0], grasp_targets(world, "B")[0]
    np.testing.assert_allclose(left[:3, 2], right[:3, 2], atol=1e-12)
    np.testing.assert_allclose(left[:3, 0], -right[:3, 0], atol=1e-12)
    np.testing.assert_allclose(left[:3, 1], -right[:3, 1], atol=1e-12)
