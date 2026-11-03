"""Measured table evidence remains separate from the visual judgment."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from luna_mujoco.runtime.controller import PolicyController
from luna_mujoco.runtime.diagnostics import DiagnosticError
from luna_mujoco.task_progress import TaskProgress


def table_observation():
    return {
        'observation_id': 'capture-1', 'objects': [],
        'views': {'1': {}, '2': {}, '3': {}},
        'scene_entities': {'support': {'id': 'support', 'kind': 'table',
            'description': 'Central tabletop', 'reference_camera_id': '3',
            'bbox': [.2, .3, .8, .9]}},
        'demo_tables': {'source_observation_id': 'capture-1', 'tables': {
            'support': {'source': 'initial_inventory_ransac_table',
                'surface_height_m': .572, 'inlier_count': 167402,
                'fit': {'multicamera_fusion': {
                    'accepted_point_count_by_camera': {'1': 58375, '2': 46577}}}}}},
    }


def test_table_measurement_updates_already_initialized_inventory():
    observation = table_observation()
    progress = TaskProgress('use the table')
    proposal = {key: deepcopy(observation[key]) for key in ('observation_id', 'objects', 'scene_entities')}
    progress.initialize_scene_inventory(proposal)
    assert progress.scene_entities['support']['geometry_status'] == 'missing'
    progress.initialize_scene_inventory(observation)
    support = progress.scene_entities['support']
    assert support['geometry_status'] == 'measured'
    assert support['observation_id'] == 'capture-1'
    assert support['measurement_source'] == 'initial_inventory_ransac_table'
    assert support['camera_ids'] == ['1', '2', '3']
    later = deepcopy(observation)
    later['observation_id'] = 'capture-2'
    progress.initialize_scene_inventory(later)
    assert progress.scene_entities['support']['geometry_status'] != 'measured'
    assert progress.scene_entities['support']['observation_id'] == 'capture-1'


@pytest.mark.parametrize('missing', ['tables', 'old_capture', 'no_capture', 'no_inliers', 'invalid_height'])
def test_unmeasured_table_is_not_promoted(missing):
    observation = table_observation()
    if missing == 'tables':
        observation.pop('demo_tables')
    elif missing == 'old_capture':
        observation['demo_tables']['source_observation_id'] = 'older'
    elif missing == 'no_capture':
        observation['demo_tables'].pop('source_observation_id')
    elif missing == 'no_inliers':
        observation['demo_tables']['tables']['support']['inlier_count'] = 0
    else:
        observation['demo_tables']['tables']['support']['surface_height_m'] = float('nan')
    progress = TaskProgress('use the table')
    progress.initialize_scene_inventory(observation)
    assert progress.scene_entities['support']['geometry_status'] == 'missing'


@pytest.mark.parametrize('current_measurement', [True, False])
async def test_task_binding_includes_current_table_without_overriding_unknown(current_measurement):
    observation = table_observation()
    if not current_measurement:
        observation['demo_tables']['source_observation_id'] = 'older'
    observation['objects'].append({'id': 'region', 'kind': 'marker',
        'placement_reference': {'source': 'configured_xy_measured_support',
            'support_entity_id': 'support', 'observation_id': 'capture-1'}})
    result = {'judgments': {'work_area_support_visible': {'status': 'unknown', 'probability': .72}},
        'evidence': {'source': 'openai_decisions', 'observation_id': 'capture-1', 'request_hash': 'judge'}}
    controller = SimpleNamespace(object_tasks={},
        task_spec=SimpleNamespace(source_ref=None, destination_ref=None, work_area_ref='region', scope='count',
                                  model_dump=lambda **kwargs: {}),
        compiler=SimpleNamespace(configured_regions={'region': {}}),
        task_inventory={'source_object_ids': [], 'support_id': 'support'},
        task_progress=SimpleNamespace(records={}, scene_entities=observation['scene_entities']),
        state={}, emit=lambda *args: None, _judge=AsyncMock(return_value=result))
    with pytest.raises(DiagnosticError, match='task_binding_unconfirmed:work_area_support_visible'):
        await PolicyController._confirm_task_binding(controller, {'observation': observation})
    call = controller._judge.call_args
    rows = call.kwargs['extra']['current_entities']
    supports = [row for row in rows if row['id'] == 'support']
    assert bool(supports) is current_measurement
    if supports:
        assert supports[0]['observation_id'] == 'capture-1'
        assert supports[0]['measurement_source'] == 'initial_inventory_ransac_table'
        assert supports[0]['description'] == 'Central tabletop'
        assert supports[0]['camera_ids'] == ['1', '2', '3']
        assert supports[0]['inlier_count'] == 167402
    assert len(call.args[1]) == 1
    assert call.args[1][0].name == 'work_area_support_visible'
    assert 'Central tabletop' in call.args[1][0].instructions
    assert 'visible in at least one CURRENT RGB camera' in call.args[1][0].instructions
    assert controller.state['task_binding_evidence']['judgments']['work_area_support_visible']['probability'] == .72


async def test_source_role_question_includes_locating_description_but_unknown_still_stops():
    observation = table_observation()
    result = {'judgments': {'source_identity': {'status': 'unknown', 'probability': .61}},
        'evidence': {'source': 'openai_decisions', 'observation_id': 'capture-1', 'request_hash': 'judge'}}
    controller = SimpleNamespace(object_tasks={},
        task_spec=SimpleNamespace(source_ref='support', destination_ref=None, work_area_ref=None,
                                  scope='ids', target_selector='black part on the table', model_dump=lambda **kwargs: {}),
        compiler=SimpleNamespace(configured_regions={}), task_inventory={'source_object_ids': []},
        state={}, emit=lambda *args: None, _judge=AsyncMock(return_value=result))
    with pytest.raises(DiagnosticError, match='task_binding_unconfirmed:source_identity'):
        await PolicyController._confirm_task_binding(controller, {'observation': observation})
    question = controller._judge.call_args.args[1][0]
    assert 'Central tabletop' in question.instructions
    assert 'black part on the table' in question.instructions
    assert 'alone do not establish the visual answer' in question.instructions
