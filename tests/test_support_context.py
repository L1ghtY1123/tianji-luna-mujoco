import numpy as np
import pytest

from luna_mujoco.runtime.support import support_context


def example():
    capture = {'observation_id': 'now', 'generation': 1}
    observation = {**capture, 'objects': [{'id': 'object', 'points_world_m': np.tile([.4, -.3, .58], (35, 1))}],
        'visual_inventory': {'evidence': {'source': 'openai_decisions', 'request_hash': 'identity'}}}
    return capture, observation, {'id': 'object', 'description': 'workpiece'}, {'table': {'id': 'table'}}


def test_support_context_uses_measured_current_surface_without_task_history():
    result = support_context(*example())
    assert result['current_measured_geometry']['visible_bounds_world_m'] == [[.4, -.3, .58]] * 2
    assert result['current_measured_geometry']['identity_request_hash'] == 'identity'
    assert 'task' not in result and 'previous_observed_support' not in result


def test_support_question_context_excludes_descriptions_with_old_contents_or_goals():
    capture, observation, subject, candidates = example()
    subject.update(kind='object', color='green', description='green object originally inside the source box')
    candidates['table'].update(kind='table', description='the requested destination for this object')
    result = support_context(capture, observation, subject, candidates)
    assert result['subject'] == {'id': 'object', 'kind': 'object', 'color': 'green'}
    assert result['support_candidates'] == {'table': {'id': 'table', 'kind': 'table'}}


@pytest.mark.parametrize('fault', ['stale', 'generation', 'identity', 'missing', 'nan'])
def test_support_context_rejects_unbound_or_missing_measurements(fault):
    capture, observation, subject, candidates = example()
    if fault == 'stale': observation['observation_id'] = 'before'
    elif fault == 'generation': observation['generation'] = 0
    elif fault == 'identity': observation['visual_inventory']['evidence'].clear()
    elif fault == 'missing': observation['objects'].clear()
    else: observation['objects'][0]['points_world_m'][0, 0] = np.nan
    with pytest.raises(ValueError, match='observed_support_current_geometry_required'):
        support_context(capture, observation, subject, candidates)


@pytest.mark.parametrize('height,within', [(.571, True), (.58, False)])
def test_validated_cad_bottom_relation_is_numeric_evidence_not_visible_surface(height, within):
    capture, observation, subject, candidates = example()
    pose = np.eye(4).tolist()
    observation['objects'][0]['cad'] = {'instance_id': 'object', 'observation_id': 'now', 'generation': 1,
        'asset_id': 'cad', 'mesh_sha256': 'mesh', 'T_world_object': pose,
        'pose_validation': {'status': 'validated', 'accepted': True, 'surface': {'tolerance_m': .0022}},
        'shape_bounds': {'minimum_world_m': [.38, -.35, height], 'maximum_world_m': [.46, -.28, .62],
                         'mesh_sha256': 'mesh', 'T_world_object': pose}}
    observation['demo_tables'] = {'source_observation_id': 'older', 'tables': {'table': {
        'model': {'pose': [.5, 0, .422, 1, 0, 0, 0], 'dims': [.6, .8, .3]}, 'surface_height_m': .572}}}
    result = support_context(capture, observation, subject, candidates)['current_measured_geometry']
    relation = result['candidate_table_relationships']['table']
    assert relation['cad_bottom_minus_table_surface_m'] == pytest.approx(height - .572)
    assert relation['cad_xy_bounds_inside_table_xy_bounds']
    assert relation['bottom_gap_within_pose_residual_tolerance'] is within
    assert relation['table_source_observation_id'] == 'older'
    assert result['registered_cad']['geometry_role'].startswith('Registered CAD shape prior')
    assert result['visible_bounds_world_m'][0][2] == .58


@pytest.mark.parametrize('fault', ['generation', 'instance', 'pose', 'mesh', 'validation'])
def test_stale_or_unvalidated_cad_cannot_supply_support_relationships(fault):
    capture, observation, subject, candidates = example()
    pose = np.eye(4).tolist()
    cad = {'instance_id': 'object', 'observation_id': 'now', 'generation': 1,
        'asset_id': 'cad', 'mesh_sha256': 'mesh', 'T_world_object': pose,
        'pose_validation': {'status': 'validated', 'accepted': True, 'surface': {'tolerance_m': .0022}},
        'shape_bounds': {'minimum_world_m': [.38, -.35, .571], 'maximum_world_m': [.46, -.28, .62],
                         'mesh_sha256': 'mesh', 'T_world_object': np.eye(4).tolist()}}
    if fault == 'generation': cad['generation'] = 0
    if fault == 'instance': cad['instance_id'] = 'different'
    if fault == 'pose': cad['T_world_object'][0][3] = 1
    if fault == 'mesh': cad['shape_bounds']['mesh_sha256'] = 'different'
    if fault == 'validation': cad['pose_validation']['accepted'] = False
    observation['objects'][0]['cad'] = cad
    with pytest.raises(ValueError, match='observed_support_cad_geometry_invalid'):
        support_context(capture, observation, subject, candidates)


def test_direct_foundationpose_support_does_not_require_a_geometry_residual():
    capture, observation, subject, candidates = example()
    pose = np.eye(4).tolist()
    observation['objects'][0]['cad'] = {'instance_id': 'object', 'observation_id': 'now', 'generation': 1,
        'asset_id': 'cad', 'mesh_sha256': 'mesh', 'T_world_object': pose,
        'pose_validation': {'status': 'accepted_by_policy', 'accepted': True,
                            'method': 'foundationpose_direct', 'surface': {}},
        'shape_bounds': {'minimum_world_m': [.38, -.35, .571], 'maximum_world_m': [.46, -.28, .62],
                         'mesh_sha256': 'mesh', 'T_world_object': pose}}
    observation['demo_tables'] = {'source_observation_id': 'older', 'tables': {'table': {
        'model': {'pose': [.5, 0, .422, 1, 0, 0, 0], 'dims': [.6, .8, .3]}, 'surface_height_m': .572}}}
    result = support_context(capture, observation, subject, candidates)['current_measured_geometry']
    assert result['registered_cad']['pose_validation_status'] == 'accepted_by_policy'
    relation = result['candidate_table_relationships']['table']
    assert relation['cad_bottom_minus_table_surface_m'] == pytest.approx(-.001)
    assert 'bottom_gap_within_pose_residual_tolerance' not in relation
    assert 'cad_pose_surface_tolerance_m' not in relation
    assert 'no residual check is required' in relation['meaning']
