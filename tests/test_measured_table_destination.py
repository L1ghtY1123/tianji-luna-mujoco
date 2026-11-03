from copy import deepcopy

import numpy as np
import pytest
from test_agent_model_layer import task_data
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.agent.schema import TaskSpec
from luna_mujoco.runtime.controller import PolicyController
from luna_mujoco.task_progress import TaskProgress


def table_observation():
    return {'observation_id': 'view-7', 'generation': 4, 'objects': [
        {'id': 'red-cylinder', 'kind': 'object', 'position': [.4, -.3, .72]}],
        'scene_entities': {'side-table': {'id': 'side-table', 'kind': 'table',
            'roles': ['table', 'collision'], 'description': 'round-edged side table'}},
        'demo_tables': {'source_observation_id': 'initial-view', 'primary_table_id': 'side-table',
            'tables': {'side-table': {'source': 'initial_inventory_ransac_table',
                'inlier_count': 220, 'surface_height_m': .7,
                'model': {'pose': [.73, .31, .55, 1., 0., 0., 0.], 'dims': [.38, .26, .3]}}}}}


def test_table_region_is_measured_and_preserves_support_identity():
    from luna_mujoco.robot_skills.placement_region import measured_table_region

    observation = table_observation()
    before = deepcopy(observation)
    region = measured_table_region(observation, 'side-table')
    assert observation == before
    assert region['id'] == 'table-region:side-table'
    assert region['kind'] == 'marker'
    assert region['roles'] == ['target_region']
    assert region['position'] == [.73, .31, .7]
    ref = region['placement_reference']
    assert ref['support_entity_id'] == 'side-table'
    assert ref['measurement_sources']['observation_id'] == 'initial-view'
    assert ref['observation_id'] == 'view-7'
    np.testing.assert_allclose(ref['region_polygon_world_xy_m'],
                               [[.54, .18], [.92, .18], [.92, .44], [.54, .44]])


@pytest.mark.parametrize('fault', ['missing', 'simulator', 'bad_size', 'bad_floor'])
def test_table_region_requires_bound_measured_geometry(fault):
    from luna_mujoco.robot_skills.placement_region import measured_table_region

    observation = table_observation()
    measured = observation['demo_tables']['tables']['side-table']
    if fault == 'missing':
        observation['demo_tables']['tables'].clear()
    elif fault == 'simulator':
        measured['source'] = 'simulator_truth'
    elif fault == 'bad_size':
        measured['model']['dims'][0] = 0.
    else:
        measured['surface_height_m'] = float('nan')
    with pytest.raises(ValueError, match='measured_table_region'):
        measured_table_region(observation, 'side-table')


async def test_controller_table_binding_changes_physical_destination_only(tmp_path):
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.task_spec = TaskSpec.model_validate(task_data() | {'target_refs': ['red-cylinder'],
        'source_ref': None, 'destination_ref': 'side-table'})
    original_task = runner.task_spec.model_dump()
    runner.demo_task = {'destination_id': 'side-table', 'destination_kind': 'support_region'}
    runner.state['target_id'] = 'red-cylinder'
    progress = runner.task_progress = TaskProgress('Move red cylinder onto side table')
    progress.current_id = 'red-cylinder'
    progress.records['red-cylinder'] = {'id': 'red-cylinder', 'display_id': 6}
    progress.membership['red-cylinder'] = 'required'
    observation = table_observation()
    progress.scene_entities = deepcopy(observation['scene_entities'])
    original_table = deepcopy(observation['scene_entities']['side-table'])
    result = await runner.execute('identify_destination', {'observation': observation},
                                  {'destination_id': 'side-table', 'destination_kind': 'support_region'})
    assert result['success']
    assert runner.state['facts']['destination_is_container'] is False
    assert runner.state['destination_id'] == 'table-region:side-table'
    assert runner.task_spec.model_dump() == original_task
    assert runner.demo_task['destination_id'] == 'side-table'
    assert observation['scene_entities']['side-table'] == original_table
    assert progress.scene_entities['side-table'] == original_table
    request = runner._bind_region(observation)
    assert request['target_count'] == 1 and request['slot_id'] == 1
    assert request['layout']['support_entity_id'] == 'side-table'
    assert request['source'] == 'measured_table_region'
    assert len([obj for obj in observation['objects'] if obj['kind'] == 'marker']) == 1


async def test_controller_bin_binding_keeps_container_identity(tmp_path):
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.state['target_id'] = 'red-cylinder'
    runner.task_progress = TaskProgress('Move red cylinder into green box')
    runner.task_progress.records['red-cylinder'] = {'id': 'red-cylinder', 'status': 'selected'}
    observation = {'objects': [{'id': 'green-box', 'kind': 'bin'}]}
    result = await runner.execute('identify_destination', {'observation': observation},
                                  {'destination_id': 'green-box', 'destination_kind': 'container'})
    assert result == {'destination_id': 'green-box', 'success': True,
                      'placement_preference': 'clearance',
                      'selection_scope': 'bound_region_not_yet_validated_motion_goal'}
    assert runner.state['facts']['destination_is_container'] is True
    assert runner.state['destination_id'] == 'green-box'
    assert not runner.task_progress.placement_requests


async def test_policy_controller_rejects_retired_container_transport(tmp_path):
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    for name in ('plan_transport', 'execute_transport', 'drop', 'verify_drop'):
        with pytest.raises(ValueError, match='unregistered_skill'):
            await runner.execute(name, {}, {})
    assert runner.simulation.moves == []
