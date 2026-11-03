"""Independent physical QA counts completed inputs and verifies their output lineage."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('unpack_acceptance', ROOT / 'scripts/audit_unpack_run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    box_pose = np.eye(4)
    box_pose[:3, 3] = [.2, -.6, .4725]
    product_pose = box_pose.copy()
    product_pose[2, 3] += -.0675 + .0745 / 2
    creation = {'observation_id': 'unpack', 'request_hash': 'verified', 'action_id': 'unpack-action'}
    records = [
        {'id': 'pink-1', 'status': 'consumed', 'output_entity_id': 'product-1', 'transfers': [
            {'transfer_index': 0, 'destination_id': 'work-area', 'arm': 'A', 'observation_id': 'place'}]},
        {'id': 'product-1', 'status': 'completed', 'input_entity_id': 'pink-1', 'processing_skill': 'vla_unpack',
            'creation_evidence': creation, 'current_region': 'right-bin', 'transfers': [
                {'transfer_index': 0, 'destination_id': 'right-bin', 'arm': 'B', 'observation_id': 'drop'}]}]
    progress = {'objects': records, 'selected_input_ids': ['pink-1'], 'completed_input_ids': ['pink-1'],
        'transformations': {'process-1': {'id': 'process-1', 'skill': 'vla_unpack', 'input_entity_id': 'pink-1',
            'output_entity_id': 'product-1', 'evidence': deepcopy(creation)}}}
    state = {'run': {'run_id': 'test', 'status': 'succeeded', 'holding': None,
        'task_progress': progress, 'task_inventory': {'destination_id': 'right-bin'}, 'metrics': {'responses_call_count': 1}},
        'simulation': {'objects': [
            {'id': 'right-bin', 'asset_id': 'blue_box', 'size': [.410, .310, .145],
             'color': '#123456', 'T_world_object': box_pose.tolist()},
            {'id': 'physical-part', 'asset_id': 'black_core', 'size': [.044, .056, .0745],
             'color': '#112233', 'T_world_object': product_pose.tolist()}]},
        'events': [{'kind': 'skill_start', 'title': 'build_scene'}] * 2}
    return state, {'right_bin': {'id': 'right-bin', 'asset_id': 'blue_box'}}


def test_new_entity_ids_count_one_finished_process_with_native_cad_and_custom_color():
    state, config = sample()
    result = module.audit(state, 1, config)
    assert result['passed']
    assert result['ledger_completed'] == 1
    assert result['physical_black_inside_right_bin'] == 1


@pytest.mark.parametrize('missing', ['place', 'unpack', 'drop'])
def test_process_status_without_each_observation_is_not_acceptance(missing):
    state, config = sample()
    progress = state['run']['task_progress']
    if missing == 'unpack':
        progress['objects'][1]['creation_evidence'] = {}
    else:
        progress['objects'][0 if missing == 'place' else 1]['transfers'] = []
    assert not module.audit(state, 1, config)['passed']


def test_physical_part_outside_bin_rejects_successful_ledger():
    state, config = sample()
    state['simulation']['objects'][1]['T_world_object'][0][3] += .3
    assert not module.audit(state, 1, config)['passed']


def test_actual_bin_pose_not_configured_pose_is_used():
    state, config = sample()
    for obj in state['simulation']['objects']:
        pose = np.asarray(obj['T_world_object'])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        pose[:3, :3] = rotation @ pose[:3, :3]
        pose[:3, 3] = rotation @ pose[:3, 3] + [.4, .2, 0.]
        obj['T_world_object'] = pose.tolist()
    assert module.audit(state, 1, config)['passed']


def test_duplicate_transformation_output_cannot_count_twice():
    state, config = sample()
    progress = state['run']['task_progress']
    progress['transformations']['process-2'] = deepcopy(progress['transformations']['process-1'])
    progress['transformations']['process-2'].update(id='process-2', input_entity_id='other-input')
    assert not module.audit(state, 1, config)['passed']


def test_acceptance_does_not_require_a_fixed_number_of_mapping_calls_or_transfers():
    state, config = sample()
    state['events'] *= 3
    source = state['run']['task_progress']['objects'][0]
    source['transfers'].insert(0, {'transfer_index': 0, 'destination_id': 'buffer',
        'arm': 'B', 'observation_id': 'buffer-placement'})
    source['transfers'][1]['transfer_index'] = 1
    result = module.audit(state, 1, config)
    assert result['passed']
    assert result['verified_entity_lineage']
    assert result['build_scene_calls'] == 6


@pytest.mark.parametrize('fault', ['creation_mismatch', 'cycle', 'unselected', 'wrong_destination',
    'unobserved_transfer', 'output_mismatch', 'skill_mismatch', 'duplicate_object'])
def test_inconsistent_generic_completion_evidence_is_rejected(fault):
    state, config = sample()
    progress = state['run']['task_progress']
    if fault == 'creation_mismatch':
        progress['transformations']['process-1']['evidence']['action_id'] = 'different-action'
    elif fault == 'cycle':
        progress['transformations']['process-1']['output_entity_id'] = 'pink-1'
    elif fault == 'unselected':
        progress['selected_input_ids'] = []
    elif fault == 'wrong_destination':
        progress['objects'][1]['current_region'] = 'elsewhere'
    elif fault == 'output_mismatch':
        progress['objects'][0]['output_entity_id'] = 'different-product'
    elif fault == 'skill_mismatch':
        progress['objects'][1]['processing_skill'] = 'different-skill'
    elif fault == 'duplicate_object':
        progress['objects'].append(deepcopy(progress['objects'][1]))
    else:
        progress['objects'][1]['transfers'][0].pop('observation_id')
    assert not module.audit(state, 1, config)['passed']


def test_additional_verified_transformation_can_precede_unpack():
    state, config = sample()
    progress = state['run']['task_progress']
    creation = {'observation_id': 'inspect', 'request_hash': 'inspected', 'action_id': 'inspection-action'}
    progress['objects'][0]['output_entity_id'] = 'inspected-1'
    progress['objects'].append({'id': 'inspected-1', 'status': 'consumed', 'input_entity_id': 'pink-1',
        'output_entity_id': 'product-1', 'processing_skill': 'inspect_part', 'creation_evidence': creation})
    progress['objects'][1]['input_entity_id'] = 'inspected-1'
    progress['transformations']['process-1']['input_entity_id'] = 'inspected-1'
    progress['transformations']['inspection'] = {'id': 'inspection', 'skill': 'inspect_part',
        'input_entity_id': 'pink-1', 'output_entity_id': 'inspected-1', 'evidence': deepcopy(creation)}
    assert module.audit(state, 1, config)['passed']


def test_polygon_containment_does_not_fill_cut_corners():
    polygon = [[-.1, -.1], [.08, -.1], [.1, -.08], [.1, .1], [-.1, .1]]
    assert module.points_inside_polygon([[0, 0], [.099, -.099]], polygon).tolist() == [True, False]
