import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_rgbd_algorithms import observed_scene

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.geometry import TOOL_PROFILE, grasp_targets


class Generator:
    def generate(self, request):
        grasp = np.eye(4)
        grasp[:3, :3] = Rotation.from_euler('xyz', [25, 15, -30], degrees=True).as_matrix()
        return {'ok': True, 'model': 'measured-grasp-test', 'candidates': [
            {'index': 0, 'score': .9, 'T_camera_grasp': grasp.tolist()}]}


def test_generation_records_selected_arm_tcp_before_batched_ik():
    observation = observed_scene()
    algorithms = Algorithms(graspgen_client=Generator())
    generated = {arm: algorithms.generate_grasps(observation, 'seen-red', arm=arm) for arm in ('A', 'B')}
    for arm, result in generated.items():
        assert result['arm'] == arm
        assert result['tool_profile_revision'] == TOOL_PROFILE['revision']
        original = np.asarray(result['T_world_cloud']) @ result['candidates'][0]['T_object_grasp']
        tcp = np.asarray(result['candidates'][0]['T_world_tcp'])
        np.testing.assert_allclose(tcp, grasp_targets([original], arm)[0])
    a, b = [np.asarray(generated[arm]['candidates'][0]['T_world_tcp']) for arm in ('A', 'B')]
    np.testing.assert_allclose(a[:3, 2], b[:3, 2])
    np.testing.assert_allclose(a[:3, :2], -b[:3, :2])
    np.testing.assert_allclose(a[:3, 3], b[:3, 3])
    assert np.linalg.det(a[:3, :3]) == pytest.approx(1.)
    assert np.linalg.det(b[:3, :3]) == pytest.approx(1.)


def test_candidates_bound_to_one_arm_cannot_be_reinterpreted_by_other_arm():
    observation = observed_scene()
    algorithms = Algorithms(graspgen_client=Generator(), curobo_request=lambda *args: pytest.fail('IK must not run'))
    grasps = algorithms.generate_grasps(observation, 'seen-red', arm='A')
    scene = {'observation_id': observation['observation_id'], 'target_id': 'seen-red',
             'generation': observation['generation'], 'scene_token': 'scene'}
    with pytest.raises(AlgorithmError, match='grasp_candidate_arm_mismatch'):
        algorithms.evaluate_arms(observation, 'seen-red', grasps, scene, requested_arm='B')


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_ik_receives_generated_tcp_without_second_tool_transform(arm):
    observation = observed_scene()
    received = []

    def request(path, body):
        assert path == '/tools/solve_ik_batch'
        assert body['arm'] == arm
        received.extend(body['goal_matrices_world'])
        return {'scene_token': 'scene', 'batch_size': len(body['goal_matrices_world']),
                'items': [{'candidate_index': index, 'success': True}
                          for index in range(len(body['goal_matrices_world']))]}

    algorithms = Algorithms(graspgen_client=Generator(), curobo_request=request)
    grasps = algorithms.generate_grasps(observation, 'seen-red', arm=arm)
    scene = {'observation_id': observation['observation_id'], 'target_id': 'seen-red',
             'generation': observation['generation'], 'scene_token': 'scene'}
    algorithms.evaluate_arms(observation, 'seen-red', grasps, scene, requested_arm=arm)
    np.testing.assert_allclose(received, [item['T_world_tcp'] for item in grasps['candidates']])
    grasps['tool_profile_revision'] = 'outdated'
    with pytest.raises(AlgorithmError, match='grasp_candidate_tool_profile_changed'):
        algorithms.evaluate_arms(observation, 'seen-red', grasps, scene, requested_arm=arm)
