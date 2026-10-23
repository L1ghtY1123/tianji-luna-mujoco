"""Independent physical QA must not confuse process count with entity count."""
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
    records = [
        {'id': 'pink-1', 'status': 'consumed', 'completed_stages': [
            {'stage_index': 0, 'observation_id': 'place'}]},
        {'id': 'product-1', 'status': 'completed', 'completed_stages': [
            {'stage_index': 1, 'observation_id': 'drop'}]}]
    progress = {'objects': records, 'completed_process_ids': ['process-1'],
        'process_records': {'process-1': {'id': 'process-1', 'input_entity_id': 'pink-1',
            'output_entity_id': 'product-1', 'status': 'completed', 'phase': 'completed',
            'creation_evidence': {'observation_id': 'unpack', 'request_hash': 'verified'}}}}
    state = {'run': {'run_id': 'test', 'status': 'succeeded', 'holding': None,
        'task_progress': progress, 'metrics': {'responses_call_count': 1}},
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
        progress['process_records']['process-1']['creation_evidence'] = {}
    else:
        progress['objects'][0 if missing == 'place' else 1]['completed_stages'] = []
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


def test_duplicate_process_output_cannot_count_twice():
    state, config = sample()
    progress = state['run']['task_progress']
    progress['process_records']['process-2'] = deepcopy(progress['process_records']['process-1'])
    progress['completed_process_ids'].append('process-2')
    assert not module.audit(state, 2, config)['passed']


def test_polygon_containment_does_not_fill_cut_corners():
    polygon = [[-.1, -.1], [.08, -.1], [.1, -.08], [.1, .1], [-.1, .1]]
    assert module.points_inside_polygon([[0, 0], [.099, -.099]], polygon).tolist() == [True, False]
