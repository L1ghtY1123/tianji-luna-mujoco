"""Insert once along the grasp's local approach axis before numerical checks."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from luna_mujoco import demo_grasp
from luna_mujoco.geometry import grasp_targets

POLICY = {"grasp_insertion_m": .01, "grasp_maximum_tilt_deg": 45., "grasp_require_away_from_robot": True}


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
        candidates, planning_settings=POLICY, robot_base_world_m=[0., 0., 0.], target_reference_world_m=[.4, 0., .5])
    assert [c["index"] for c in accepted] == [0, 1]
    assert evidence["maximum_tilt_deg"] == 45.
    assert evidence["rejected_tilt_count"] == 2
    assert evidence["rejected_toward_robot_count"] == 1
    assert candidates == original
    for source, prepared in zip(original, shifted):
        before, after = np.array(source["T_world_tcp"]), np.array(prepared["T_world_tcp"])
        np.testing.assert_allclose(after[:3, 3], before[:3, 3] + .01 * before[:3, 2])
        np.testing.assert_array_equal(after[:3, :3], before[:3, :3])
        assert prepared['insertion_applied'] is True
        assert prepared['T_world_tcp_before_insertion'] == source['T_world_tcp']
        np.testing.assert_allclose(prepared['insertion_offset_world_m'], .01 * before[:3, 2])
    assert evidence['insertion_m'] == .01
    assert evidence['operation_order'] == ['local_z_insertion', 'direction_filter', 'arm_jaw_symmetry', 'batched_ik']


def test_rotated_insertion_is_local_z_and_reprocessing_does_not_accumulate():
    source = candidate(0, 30.)
    before = np.array(source['T_world_tcp'])
    reference = before.copy()
    reference[:3, :2] *= -1
    kwargs = {'robot_base_world_m': [0., 0., 0.], 'target_reference_world_m': [.4, 0., .5],
              'reference_tcp': reference, 'planning_settings': POLICY}
    prepared, accepted, _ = demo_grasp.prepare_grasp_candidates([source], **kwargs)
    after = np.array(accepted[0]['T_world_tcp'])
    np.testing.assert_allclose(after[:3, 3] - before[:3, 3], [.005, 0., -.01 * np.cos(np.pi / 6)])
    np.testing.assert_allclose(after[:3, 2], before[:3, 2])
    np.testing.assert_allclose(after[:3, :2], -before[:3, :2])
    repeated, _, _ = demo_grasp.prepare_grasp_candidates(prepared, **kwargs)
    np.testing.assert_allclose(repeated[0]['T_world_tcp'], after)
    assert repeated[0]['T_world_tcp_before_insertion'] == source['T_world_tcp']


def test_generate_passes_inserted_pose_to_reach_and_actual_ik_request(monkeypatch):
    from luna_mujoco.algorithms import Algorithms
    from luna_mujoco.demo_algorithms import DemoAlgorithms
    from luna_mujoco.geometry import TOOL_PROFILE
    from luna_mujoco.object_obb import observed_object_obb

    raw = candidate(0, 30.)
    raw['T_object_grasp'] = np.eye(4).tolist()
    original = deepcopy(raw)
    target = {'id': 'target', 'points_world_m': [[.4, 0., .5]]}
    snapshot = {'generation': 1, 'observation_id': 'capture', 'objects': [target],
                'arms': {arm: {'joints_rad': [0.] * 7, 'gripper_position_rad': 0.} for arm in ('A', 'B')}}
    monkeypatch.setattr(Algorithms, 'generate_grasps', lambda *a, **k: {
        'object_obb': observed_object_obb(target),
        'graspgen_input_source': 'observed_obb_mesh_surface_samples',
        'candidates': [raw], 'all_generated_candidates': [raw], 'batches': [{'candidate_count': 1}],
        'arm': 'A', 'tcp_frame': 'gripper_front_A', 'tool_profile_revision': TOOL_PROFILE['revision'], 'T_world_cloud': np.eye(4).tolist()})
    monkeypatch.setattr('luna_mujoco.demo_algorithms.fixed_slot_reach_context', lambda *a: {
        'robot_base_world_m': [0., 0., 0.], 'measured_reference_world_m': [.4, 0., .5]})
    seen = []
    monkeypatch.setattr('luna_mujoco.demo_algorithms.filter_fixed_slot_reach',
                        lambda candidates, context: (seen.extend(deepcopy(candidates)) or candidates, {}))
    algorithms = DemoAlgorithms()
    monkeypatch.setattr(algorithms, '_completed_object_geometry', lambda *a: np.array(target['points_world_m']))
    monkeypatch.setattr(algorithms, '_binding', lambda *a: None)
    monkeypatch.setattr(algorithms, '_target', lambda *a: target)
    monkeypatch.setattr(algorithms, '_cloud_frame', lambda *a: np.eye(4))
    generated = algorithms.generate_grasps(snapshot, 'target', arm='A', planning_settings=POLICY)
    expected = np.array(original['T_world_tcp'])
    expected[:3, 3] += .01 * expected[:3, 2]
    np.testing.assert_allclose(seen[0]['T_world_tcp'], expected)
    assert raw == original
    assert generated['candidate_replay']['all_generated_candidates'] == [original]

    class RequestCaptured(Exception):
        pass

    def capture_request(path, payload):
        assert path == '/tools/solve_ik_batch'
        np.testing.assert_allclose(payload['goal_matrices_world'], [expected])
        raise RequestCaptured

    monkeypatch.setattr(algorithms, 'request', capture_request)
    with pytest.raises(RequestCaptured):
        algorithms.evaluate_arms(snapshot, 'target', generated, {'scene_token': 'scene'})


@pytest.mark.parametrize("side", [-1., 1.])
def test_toward_robot_rule_is_radial_for_either_side(side):
    _, accepted, _ = demo_grasp.prepare_grasp_candidates(
        [candidate(0, 30., side * 90.), candidate(1, 30., -side * 90.)],
        planning_settings=POLICY,
        robot_base_world_m=[0., 0., 0.], target_reference_world_m=[0., side * .5, .6])
    assert [c["index"] for c in accepted] == [0]


def test_existing_arm_profiles_preserve_z_and_reverse_xy():
    world = np.eye(4)[None]
    left, right = grasp_targets(world, "A")[0], grasp_targets(world, "B")[0]
    np.testing.assert_allclose(left[:3, 2], right[:3, 2], atol=1e-12)
    np.testing.assert_allclose(left[:3, 0], -right[:3, 0], atol=1e-12)
    np.testing.assert_allclose(left[:3, 1], -right[:3, 1], atol=1e-12)
