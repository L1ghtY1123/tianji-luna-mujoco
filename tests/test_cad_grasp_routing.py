from copy import deepcopy

import numpy as np
import pytest
from test_rgbd_algorithms import observed_scene

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.demo_grasp import prepare_grasp_candidates


class OfflineLibrary:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {'source': 'cad_offline_library', 'library_id': 'fixture', 'library_sha256': 'f' * 64,
            'asset_id': kwargs['asset_id'], 'source_sha256': kwargs['source_sha256'],
            'T_world_source': kwargs['T_world_source'], 'candidate_count': 256,
            'candidates': [{'index': i, 'source': 'cad_offline_library', 'score': .9,
                'source_batch': 1, 'source_index': i, 'T_world_tcp': np.eye(4).tolist()} for i in range(256)]}


class NoInference:
    def generate(self, request):
        pytest.fail('known CAD must not call GraspGen')


def cad_observation():
    observation = observed_scene()
    observation['objects'][0]['cad'] = {'instance_id': 'seen-red', 'asset_id': 'known-part',
        'source_sha256': 'a' * 64, 'observation_id': observation['observation_id'],
        'generation': observation['generation'], 'pose_validation': {'accepted': True},
        'T_world_source': np.eye(4).tolist()}
    return observation


def test_known_cad_uses_full_requested_arm_library_without_obb_or_inference():
    library = OfflineLibrary()
    algorithm = Algorithms(graspgen_client=NoInference(), cad_grasp_library=library)
    result = algorithm.generate_grasps(cad_observation(), 'seen-red', arm='B', candidate_count=10)
    assert len(result['candidates']) == 256
    assert library.calls[0]['arm'] == 'B'
    assert result['source'] == 'cad_offline_library'
    assert result['requested_candidate_count'] == 256
    assert result['tcp_frame'] == 'gripper_front_B'
    assert 'object_obb' not in result and 'graspgen_input' not in result


@pytest.mark.parametrize('field,value', [('observation_id', 'old'), ('generation', 99), ('instance_id', 'other')])
def test_stale_cad_cannot_supply_grasps(field, value):
    observation = cad_observation()
    observation['objects'][0]['cad'][field] = value
    algorithm = Algorithms(graspgen_client=NoInference(), cad_grasp_library=OfflineLibrary())
    with pytest.raises(AlgorithmError, match='cad_pose_binding_stale'):
        algorithm.generate_grasps(observation, 'seen-red', arm='A')


def test_offline_arm_pose_is_not_shifted_or_jaw_flipped():
    pose = np.eye(4)
    pose[:3, :3] = np.diag([-1., -1., 1.])
    candidate = {'index': 0, 'source': 'cad_offline_library', 'T_world_tcp': pose.tolist()}
    original = deepcopy(candidate)
    _, accepted, _ = prepare_grasp_candidates([candidate], robot_base_world_m=[0, 0, 0],
        target_reference_world_m=[.5, .1, .6], reference_tcp=np.eye(4),
        planning_settings={'grasp_insertion_m': .01})
    np.testing.assert_array_equal(accepted[0]['T_world_tcp'], pose)
    assert candidate == original


def test_offline_pool_has_one_fixed_budget_and_pose_bound_identity():
    from luna_mujoco.demo_sampling import grasp_input_identity, validate_grasp_budget

    result = Algorithms(graspgen_client=NoInference(), cad_grasp_library=OfflineLibrary()).generate_grasps(
        cad_observation(), 'seen-red', arm='A', candidate_count=10)
    result['sampling_round'] = 1
    validate_grasp_budget(result, 1)
    identity = grasp_input_identity(result)
    assert identity['library_sha256'] == 'f' * 64
    assert identity['T_world_source'] == np.eye(4).tolist()
    with pytest.raises(AlgorithmError, match='cad_grasp_library_cannot_resample'):
        validate_grasp_budget(result, 2)


def test_known_grasp_contact_geometry_retains_actual_cad_surface_and_pose():
    from luna_mujoco.cad_perception import CadCatalog
    from luna_mujoco.demo_algorithms import DemoAlgorithms

    geometry = CadCatalog.from_file().geometry('black_core')
    snapshot = cad_observation()
    target = snapshot['objects'][0]
    pose = np.eye(4)
    pose[:3, 3] = [.4, .1, .6]
    target['cad'].update({key: geometry[key] for key in ('asset_id', 'source_sha256', 'mesh_sha256')})
    target['cad'].update(mesh_key='black_core', T_world_object=pose.tolist())
    points = DemoAlgorithms._completed_object_geometry(snapshot, target)
    expected = np.unique(geometry['vertices_m'], axis=0) + pose[:3, 3]
    np.testing.assert_allclose(points, expected)
    assert len(points) > 8  # Exact low-poly CAD vertices checked above, not OBB corners.
    target['cad']['observation_id'] = 'stale'
    with pytest.raises(ValueError, match='cad_pose_binding_stale'):
        DemoAlgorithms._completed_object_geometry(snapshot, target)
