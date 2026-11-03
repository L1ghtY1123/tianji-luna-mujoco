from copy import deepcopy

import pytest
from test_observation_motion import ObservationAlgorithms, context

from luna_mujoco.task_progress import TaskProgress, source_region_confirmed


def test_entity_arm_binding_does_not_fix_another_entity_or_future_transfer():
    ledger = TaskProgress('move the selected entities')
    ledger.select({'id': 'first'}, 'destination')
    ledger.bind_arm('A')
    ledger.complete('first', observation_id='released-first')
    ledger.select({'id': 'second'}, 'destination')
    assert ledger.bound_arm() is None
    ledger.bind_arm('B')
    assert ledger.bound_arm() == 'B'
    assert ledger.records['first']['transfers'][0]['arm'] == 'A'
    assert not {'route', 'stage_index', 'stage_arms', 'pending_route_id', 'route_destinations'} & ledger.snapshot().keys()


def test_requested_arm_is_enforced_per_selected_entity():
    ledger = TaskProgress('use arm A', requested_arm='A')
    ledger.select({'id': 'first'}, 'destination')
    with pytest.raises(ValueError, match='requested_arm_cannot_switch'):
        ledger.bind_arm('B')
    ledger.bind_arm('A')
    assert ledger.bound_arm() == 'A'
    assert ledger.records['first']['execution_arm'] == 'A'


def test_transfer_index_preserves_completed_layout_until_next_explicit_selection():
    ledger = TaskProgress('move the entity')
    ledger.select({'id': 'item', 'display_id': 1}, 'region')
    ledger.bind_arm('B')
    request = ledger.placement_request('region')
    assert request['transfer_index'] == 0
    ledger.complete('item', observation_id='released')
    assert ledger.transfer_index('item') == 0
    assert ledger.visual_progress()['item']['slot_id'] == 1
    assert ledger.records['item']['transfers'][0]['transfer_index'] == 0
    # A task-specific controller can explicitly permit another transfer.
    ledger.records['item']['status'] = 'observed'
    ledger.select({'id': 'item'}, 'other-region')
    assert ledger.transfer_index() == 1
    assert ledger.bound_arm() is None
    assert ledger.placement_request('other-region')['transfer_index'] == 1


def test_declared_source_evidence_is_checked_without_requiring_a_bin_route():
    assert source_region_confirmed({'id': 'item'})
    record = {'id': 'item', 'initial_source': {'measurement_observation_id': 'before',
        'task_source_evidence': {'source': 'openai_decisions', 'status': 'yes',
            'target_id': 'item', 'source_id': 'support', 'source_kind': 'table',
            'request_hash': 'hash', 'observation_id': 'before'}}}
    assert source_region_confirmed(record)
    record['initial_source']['task_source_evidence']['observation_id'] = 'stale'
    assert not source_region_confirmed(record)


async def test_mapping_checks_explicit_observation_pose_against_current_feedback(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    state['observation_pose_capture_id'] = state['observation']['observation_id']
    state['observation_pose_reference'] = deepcopy(state['observation'])
    runner.simulation.state['arms']['A']['joints_rad'][0] += .2
    with pytest.raises(ValueError, match='observation_pose_analysis_required'):
        await runner.build_scene(state, 'selected')
    assert algorithms.map_calls == []


async def test_mapping_keeps_released_objects_required_but_excludes_consumed_inputs(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    runner.task_progress = TaskProgress('move selected entities')
    runner.task_progress.records['released'] = {'id': 'released', 'status': 'observed',
        'transfers': [{'destination_id': 'region', 'observation_id': 'after'}]}
    with pytest.raises(ValueError, match='placed_object_not_observed:released'):
        await runner.build_scene(state, 'selected')
    assert algorithms.map_calls == []
    runner.task_progress.records['released']['status'] = 'consumed'
    await runner.build_scene(state, 'selected')
    assert len(algorithms.map_calls) == 1


@pytest.mark.parametrize('center_reachability', [False, True])
async def test_retired_runner_arm_selection_is_rejected_for_each_evaluator(tmp_path, center_reachability):
    runner, state = context(tmp_path, ObservationAlgorithms())
    runner.task_progress = TaskProgress('use arm A', requested_arm='A')
    runner.task_progress.select({'id': 'black'}, 'destination')
    runner.state['facts']['center_reachability_enabled'] = center_reachability
    state['evaluations'] = {arm: {'reachable': True, 'valid_candidates': [0]} for arm in ('A', 'B')}
    with pytest.raises(ValueError, match='unknown_skill:choose_arm'):
        await runner.execute('choose_arm', state, {'arm': 'B'})
    assert runner.state['arm'] is None
    with pytest.raises(ValueError, match='unknown_skill:choose_arm'):
        await runner.execute('choose_arm', state, {'arm': 'A'})
    assert runner.state['arm'] is None
