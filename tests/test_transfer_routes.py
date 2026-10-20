import pytest

from jev_mujoco.task_progress import TaskProgress, original_source_bin


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
    progress = TaskProgress('盒子中物体放到桌面', route='box_table')
    progress.observe([obj], 'head')
    if accepted:
        progress.assess_membership({'red': 'required'})
        progress.select(obj)
    else:
        with pytest.raises(ValueError, match='task_source_region_unconfirmed'):
            progress.assess_membership({'red': 'required'})


def test_two_half_overlap_bins_remain_ambiguous():
    obj = source()
    obj['measured_relationships'][0]['visible_xy_fraction'] = .5
    obj['measured_relationships'].append({'relation': 'above_observed_bin_floor', 'object_id': 'other-box',
                                        'visible_xy_fraction': .5})
    assert original_source_bin({'initial_source': obj}) is None


@pytest.mark.parametrize('route', ['box_table', 'box_table_box'])
def test_box_source_membership_rejects_unsupported_required_atomically(route):
    progress = TaskProgress('把盒子中物体拿到桌面黄色区域', 'collection', route=route)
    outside = {**source('table-object'), 'measured_relationships': []}
    progress.observe([source('inside'), outside], 'initial')
    with pytest.raises(ValueError, match='task_source_region_unconfirmed'):
        progress.assess_membership({'inside': 'required', 'table-object': 'required'})
    assert progress.membership == {'inside': 'unknown', 'table-object': 'unknown'}
    # Missing evidence stays unknown until JEV resolves membership; it is not exclusion.
    progress.assess_membership({'inside': 'required', 'table-object': 'unknown'})
    assert progress.snapshot()['membership_pending_ids'] == ['table-object']
    progress.membership['table-object'] = 'required'  # Backstop even with a stale/bad ledger.
    with pytest.raises(ValueError, match='task_source_region_unconfirmed'):
        progress.select(outside)
    assert progress.current_id is None


def test_box_route_keeps_original_source_after_intermediate_place():
    progress = TaskProgress('盒子到桌面再到另一盒子', route='box_table_box')
    progress.select(source(), 'marker-yellow')
    progress.complete('red', observation_id='at-table')
    progress.select({**source(), 'measured_relationships': []}, 'other-box')
    assert original_source_bin(progress.records['red']) == 'source-box'


def test_route_inventory_does_not_add_new_wrist_detections_after_initial_list():
    from test_task_progress import inventory_observation
    progress = TaskProgress('所有方块放进盒子', 'collection', route='table_box')
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


def test_intermediate_place_preserves_id_and_does_not_complete_route():
    progress = TaskProgress('盒子经桌面到另一盒子', route='box_table_box')
    progress.select(source(), 'marker-yellow')
    progress.bind_arm('A')
    progress.complete('red', observation_id='at-table')
    assert progress.completed_ids == []
    assert progress.snapshot()['pending_route_id'] == 'red'
    assert progress.stage_index('red') == 1
    assert progress.terminal_status('complete') == 'needs_attention'
    progress.complete('red', observation_id='at-table')
    assert progress.stage_index('red') == 1  # Re-verifying one release is not another transfer.
    progress.select(source(), 'box-right')
    progress.bind_arm('B')
    progress.complete('red', observation_id='at-box')
    assert progress.completed_ids == ['red']
    assert [step['destination_id'] for step in progress.records['red']['completed_stages']] == ['marker-yellow', 'box-right']
    assert progress.snapshot()['stage_arms'] == {'0': 'A', '1': 'B'}


def test_arm_binding_applies_to_entire_leg_including_other_objects():
    progress = TaskProgress('全部放进盒子', 'collection', route='table_box')
    progress.select(source(), 'box')
    progress.bind_arm('A')
    progress.complete('red', observation_id='done')
    progress.select(source('second'), 'box')
    assert progress.bound_arm() == 'A'
    with pytest.raises(ValueError, match='stage_arm_is_fixed'):
        progress.bind_arm('B')


def test_same_object_must_finish_second_leg_before_starting_another():
    progress = TaskProgress('全部从盒子经桌面到另一盒子', 'collection', route='box_table_box')
    progress.select(source(), 'marker-yellow')
    progress.complete('red', observation_id='first')
    with pytest.raises(ValueError, match='pending_route_requires_same_object'):
        progress.select(source('second'), 'marker-yellow')


def test_explicit_arm_cannot_change_between_legs():
    progress = TaskProgress('用左臂中转', requested_arm='A', route='box_table_box')
    progress.select(source(), 'marker-yellow')
    with pytest.raises(ValueError, match='requested_arm'):
        progress.bind_arm('B')
