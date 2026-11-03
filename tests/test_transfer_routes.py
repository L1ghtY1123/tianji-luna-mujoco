from types import SimpleNamespace

import pytest

from luna_mujoco.runtime.inventory import PolicyTaskProgress
from luna_mujoco.task_progress import TaskProgress, original_source_bin


def test_original_source_requires_one_dominant_measured_bin_relationship():
    relations = [{'relation': 'above_observed_bin_floor', 'object_id': 'source', 'visible_xy_fraction': .99},
                 {'relation': 'above_observed_bin_floor', 'object_id': 'other', 'visible_xy_fraction': .01}]
    record = {'initial_source': {'measured_relationships': relations}}
    assert original_source_bin(record) == 'source'
    relations[1]['visible_xy_fraction'] = .95
    assert original_source_bin(record) is None
    relations[0]['visible_xy_fraction'] = relations[1]['visible_xy_fraction'] = .01
    assert original_source_bin(record) is None


def source(identifier='red'):
    return {'id': identifier, 'kind': 'cube', 'color': 'red', 'position': [.45, .1, .65],
            'measured_relationships': [{'relation': 'above_observed_bin_floor', 'object_id': 'source-box',
                                        'visible_xy_fraction': 1.}]}


@pytest.mark.parametrize('fraction,accepted', [(.49, False), (.5, True), (.75, True)])
def test_box_source_uses_fifty_percent_visible_xy_overlap(fraction, accepted):
    obj = source()
    obj['measured_relationships'][0]['visible_xy_fraction'] = fraction
    assert original_source_bin({'initial_source': obj}) == ('source-box' if accepted else None)


def test_two_half_overlap_bins_remain_ambiguous():
    obj = source()
    obj['measured_relationships'][0]['visible_xy_fraction'] = .5
    obj['measured_relationships'].append({'relation': 'above_observed_bin_floor', 'object_id': 'other-box',
                                        'visible_xy_fraction': .5})
    assert original_source_bin({'initial_source': obj}) is None


def test_explicit_source_membership_rejects_unknown_required_atomically():
    progress = TaskProgress('move the entities from the selected source', 'collection')
    outside = {**source('table-object'), 'measured_relationships': []}
    progress.observe([source('inside'), outside], 'initial')
    for identifier, status in [('inside', 'yes'), ('table-object', 'unknown')]:
        progress.records[identifier]['initial_source']['task_source_evidence'] = {
            'source': 'openai_decisions', 'status': status, 'target_id': identifier,
            'source_id': 'source-box', 'source_kind': 'bin', 'observation_id': 'initial', 'request_hash': 'hash'}
    with pytest.raises(ValueError, match='task_source_region_unconfirmed'):
        progress.assess_membership({'inside': 'required', 'table-object': 'required'})
    assert progress.membership == {'inside': 'unknown', 'table-object': 'unknown'}
    # Unknown evidence does not prove exclusion.
    progress.assess_membership({'inside': 'required', 'table-object': 'unknown'})
    assert progress.snapshot()['membership_pending_ids'] == ['table-object']
    progress.membership['table-object'] = 'required'  # Backstop even with a stale/bad ledger.
    with pytest.raises(ValueError, match='task_source_region_unconfirmed'):
        progress.select(outside)
    assert progress.current_id is None


def test_explicit_transfer_keeps_original_source_after_intermediate_place():
    progress = PolicyTaskProgress('move the entity through a work area')
    progress.observe([source()], 'initial')
    progress.assess_membership({'red': 'required'})
    progress.configure_task(SimpleNamespace(destination_ref='other-box', processing=[]), ['red'])
    progress.select(source(), 'marker-yellow')
    progress.complete('red', observation_id='at-table')
    progress.select({**source(), 'measured_relationships': []}, 'other-box')
    assert original_source_bin(progress.records['red']) == 'source-box'


def test_frozen_inventory_does_not_add_new_wrist_detections_after_initial_list():
    from test_task_progress import inventory_observation
    progress = TaskProgress('所有方块放进盒子', 'collection')
    progress.observe([source('one'), source('two')], 'head-initial')
    progress.assess_membership({'one': 'required', 'two': 'required'})
    progress.freeze_inventory(inventory_observation([source('one'), source('two')]))
    fresh = {'objects': [source('one'), source('two'), source('box-wall-alias')]}
    progress.observe(fresh['objects'], 'new-geometry')
    progress.assign_display_ids(fresh)
    assert set(progress.records) == {'one', 'two'}
    assert len(fresh['objects']) == 3  # Still measured obstacle evidence.
    assert progress.snapshot()['inventory_initialized'] is True
    with pytest.raises(ValueError, match='target_not_in_initial_inventory'):
        progress.select(source('box-wall-alias'))
    progress.observe([source('replacement'), source('unrelated')], 'unlisted-observation')
    assert 'replacement' not in progress.records and 'unrelated' not in progress.records


def test_intermediate_place_preserves_id_without_completing_final_goal():
    progress = PolicyTaskProgress('move the entity through a work area')
    progress.observe([source()], 'initial')
    progress.assess_membership({'red': 'required'})
    progress.configure_task(SimpleNamespace(destination_ref='box-right', processing=[]), ['red'])
    progress.select(source(), 'marker-yellow')
    progress.bind_arm('A')
    progress.complete('red', observation_id='at-table')
    assert progress.completed_ids == []
    assert progress.records['red']['status'] == 'observed'
    assert progress.transfer_index('red') == 0
    assert progress.terminal_status('complete') == 'needs_attention'
    progress.complete('red', observation_id='at-table')
    assert len(progress.records['red']['transfers']) == 1
    progress.select(source(), 'box-right')
    progress.bind_arm('B')
    progress.complete('red', observation_id='at-box')
    assert progress.completed_ids == ['red']
    assert [step['destination_id'] for step in progress.records['red']['transfers']] == ['marker-yellow', 'box-right']
    assert [step['arm'] for step in progress.records['red']['transfers']] == ['A', 'B']


def test_arm_binding_belongs_to_the_current_entity():
    progress = TaskProgress('全部放进盒子', 'collection')
    progress.select(source(), 'box')
    progress.bind_arm('A')
    progress.complete('red', observation_id='done')
    progress.select(source('second'), 'box')
    assert progress.bound_arm() is None
    progress.bind_arm('B')
    assert progress.bound_arm() == 'B'


def test_pending_transfer_does_not_select_a_fixed_next_entity():
    progress = PolicyTaskProgress('move the entities through selected regions', 'collection')
    progress.observe([source(), source('second')], 'initial')
    progress.assess_membership({'red': 'required', 'second': 'required'})
    progress.configure_task(SimpleNamespace(destination_ref='final', processing=[]), ['red', 'second'])
    progress.select(source(), 'marker-yellow')
    progress.complete('red', observation_id='first')
    progress.select(source('second'), 'marker-yellow')
    assert progress.current_id == 'second'
    assert progress.completed_ids == []


def test_explicit_arm_constraint_applies_to_selected_transfer():
    progress = TaskProgress('用左臂中转', requested_arm='A')
    progress.select(source(), 'marker-yellow')
    with pytest.raises(ValueError, match='requested_arm'):
        progress.bind_arm('B')
