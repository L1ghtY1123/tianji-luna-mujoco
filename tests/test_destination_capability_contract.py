"""A support surface cannot authorize container-only operations."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision
from graph_contract_fixtures import call, check, robot_catalog

from luna_mujoco.runtime.controller import PolicyController


@pytest.mark.parametrize('operation', ['plan_transport', 'execute_transport', 'drop', 'verify_drop'])
def test_support_destination_rejects_container_operation_before_execution(operation):
    steps = [call('destination', 'identify_destination', destination_id='table-region',
                  destination_kind='support_region'), call('container_operation', operation)]
    original = deepcopy(steps)
    with pytest.raises(ValueError, match='unknown_skill'):
        check(steps, robot_catalog(), facts={'observed': True, 'target_selected': True})
    assert steps == original


@pytest.mark.parametrize('kind,operation', [('support_region', 'plan_place'), ('container', 'plan_place')])
def test_compatible_destination_does_not_constrain_model_route(kind, operation):
    steps = [call('destination', 'identify_destination', destination_id='measured-destination',
                  destination_kind=kind), call('zone', 'perceive_placement_zone'), call('arm', 'choose_arm', arm='A'),
             call('resolve', 'resolve_placement'), call('selected_plan', operation)]
    graph = check(steps, robot_catalog(), facts={'observed': True, 'target_selected': True})
    assert graph.model_dump()['nodes'][:-2] == steps


@pytest.mark.parametrize('measured,declared', [('marker', 'container'), ('bin', 'support_region')])
async def test_false_destination_type_is_rejected_without_binding_or_motion(tmp_path, measured, declared):
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.state.update(target_id='item', destination_id='previous')
    runner.task_progress = SimpleNamespace(records={'item': {}})
    before = deepcopy(runner.state)
    context = {'observation': {'objects': [{'id': 'destination', 'kind': measured}]}}
    with pytest.raises(ValueError, match='destination_kind_mismatch'):
        await runner.execute('identify_destination', context,
                             {'destination_id': 'destination', 'destination_kind': declared})
    before['facts'].update(placement_zone_ready=False, placement_resolved=False, place_plan_ready=False)
    assert runner.state == before
    assert runner.task_progress.records == {'item': {}}
    assert runner.simulation.moves == []


async def test_failed_support_binding_revokes_previous_container_type(tmp_path):
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.state.update(target_id='item', destination_id='previous-bin')
    runner.state['facts']['destination_is_container'] = True
    runner.task_progress = SimpleNamespace(records={'item': {}})
    runner._bind_region = Mock(side_effect=ValueError('measured_support_missing'))
    context = {'observation': {'objects': [{'id': 'surface', 'kind': 'marker'}]}}
    with pytest.raises(ValueError, match='measured_support_missing'):
        await runner.execute('identify_destination', context,
                             {'destination_id': 'surface', 'destination_kind': 'support_region'})
    assert runner.state['facts']['destination_is_container'] is False
    assert runner.simulation.moves == []


def test_new_target_does_not_inherit_old_container_type():
    steps = [call('target', 'identify_target', target_id='next-object'),
             call('zone', 'perceive_placement_zone')]
    with pytest.raises(ValueError, match='planning_precondition_not_satisfied:zone:.*destination_selected'):
        check(steps, robot_catalog(), facts={'destination_is_container': True})
