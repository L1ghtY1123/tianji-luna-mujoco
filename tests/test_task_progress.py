import numpy as np
import pytest

from jev_mujoco.task_progress import TaskProgress


@pytest.mark.parametrize('route', ['direct', 'box_table'])
def test_jev_membership_separates_requested_objects_and_unknowns_from_obstacles(route):
    progress = TaskProgress('把盒子中物体放到桌面黄色区域', 'collection', route=route)
    objects = [{'id': key, 'kind': 'cube', 'position': [.4, i * .1, .6]}
               for i, key in enumerate(('source', 'table-object', 'unclear'))]
    objects[0]['measured_relationships'] = [
        {'relation': 'above_observed_bin_floor', 'object_id': 'box', 'visible_xy_fraction': 1.}]
    progress.observe(objects, 'head')
    assert progress.snapshot()['objects'] == []
    assert progress.snapshot()['membership_pending_ids'] == ['source', 'table-object', 'unclear']
    progress.assess_membership({'source': 'required', 'table-object': 'excluded', 'unclear': 'unknown'})
    state = progress.snapshot()
    assert [obj['id'] for obj in state['objects']] == ['source']
    assert [obj['id'] for obj in state['unresolved_membership']] == ['unclear']
    assert state['membership'] == {'source': 'required', 'table-object': 'excluded', 'unclear': 'unknown'}
    assert set(progress.records) == {'source', 'table-object', 'unclear'}
    assert progress.terminal_status('complete') == 'needs_attention'
    observation = inventory_observation(objects)
    assert progress.assign_display_ids(observation) == {}
    assert [obj['id'] for obj in progress.rank_candidates(observation, arm='A')] == ['source']
    for obj in objects[1:]:
        with pytest.raises(ValueError, match='target_not_in_task_inventory'):
            progress.select(obj)
    progress.observe(objects, 'next')
    assert [obj['id'] for obj in progress.snapshot()['objects']] == ['source']
    progress.assess_membership({'unclear': 'excluded'})
    progress.freeze_inventory(observation)
    progress.select(objects[0], 'marker')
    progress.complete('source', observation_id='released')
    assert progress.terminal_status('complete') == 'succeeded'


def test_membership_rejects_unknown_ids_and_changes_to_selected_history():
    progress = TaskProgress('only source', 'collection', route='box_table')
    source = {'id': 'source', 'kind': 'cube', 'measured_relationships': [
        {'relation': 'above_observed_bin_floor', 'object_id': 'box', 'visible_xy_fraction': 1.}]}
    progress.observe([source], 'head')
    with pytest.raises(ValueError, match='invalid_task_membership'):
        progress.assess_membership({'unmeasured': 'required'})
    progress.assess_membership({'source': 'required'})
    progress.select(source)
    with pytest.raises(ValueError, match='invalid_task_membership'):
        progress.assess_membership({'source': 'excluded'})
    assert progress.snapshot()['current']['id'] == 'source'


def test_explicit_selection_keeps_membership_when_observed_after_completion():
    progress = TaskProgress('move source', route='table_box')
    source = {'id': 'source', 'kind': 'cube'}
    progress.select(source, 'bin')
    progress.complete('source', observation_id='released')
    progress.observe([source], 'after')
    assert [obj['id'] for obj in progress.snapshot()['objects']] == ['source']
    assert progress.snapshot()['unresolved_membership'] == []


def test_direct_single_retains_measured_distances_for_jev_identity_selection():
    progress = TaskProgress('抓取距离手最近的方块', scope='single')
    objects = [{'id': 'near', 'kind': 'cube', 'position': [0, 0, 2.9]},
               {'id': 'far', 'kind': 'cube', 'position': [0, 0, 1.]}]
    progress.observe(objects, 'before')
    assert progress.snapshot()['objects'] == []
    ranked = progress.rank_candidates(inventory_observation(objects), arm='A')
    assert [item['id'] for item in ranked] == ['near', 'far']
    assert ranked[0]['tcp_distance_m'] == pytest.approx(.1)
    assert progress.membership == {'near': 'unknown', 'far': 'unknown'}


def test_completion_is_idempotent_and_preserves_task_constraints():
    progress = TaskProgress('用左臂放两个红色方块和一个蓝球，黑色不要动', 'collection', 'A')
    source = {'id': 'red-1', 'kind': 'cube', 'color': 'red', 'position': [.4, .1, .65]}
    progress.select(source, 'bin-blue')
    progress.complete('red-1', observation_id='after-1')
    progress.complete('red-1', observation_id='after-2')
    state = progress.snapshot()
    assert len(state['completed']) == 1
    assert state['completed'][0]['destination_id'] == 'bin-blue'
    assert state['instruction'] == '用左臂放两个红色方块和一个蓝球，黑色不要动'
    assert state['requested_arm'] == 'A'
    with pytest.raises(ValueError, match='already_completed'):
        progress.select(source, 'bin-blue')


def test_failure_records_evidence_without_selecting_any_recovery_action():
    progress = TaskProgress('把全部红色放进去', 'collection', None)
    progress.select({'id': 'red-1', 'color': 'red'}, 'bin')
    progress.fail('red-1', 'verify_grasp', 'empty', {'status': 'empty'})
    state = progress.snapshot()
    assert state['current_id'] == 'red-1'
    assert state['current']['color'] == 'red'
    assert state['deferred'] == []
    assert state['completed'] == []
    assert state['attempts']['red-1'] == 1
    assert 'recovery_action' not in state


def test_repeated_failure_count_is_evidence_not_a_recovery_decision():
    progress = TaskProgress('把红色和绿色放进去', 'collection', 'A')
    progress.fail(None, 'observe', 'instance association ambiguous', {'observation_id': 'first'})
    progress.fail(None, 'recover_task', 'instance association ambiguous', {'observation_id': 'second'})
    assert progress.failure['consecutive_same_code_count'] == 2
    assert progress.failure['evidence']['observation_id'] == 'second'
    assert 'recovery_action' not in progress.failure
    progress.fail('red', 'plan_grasp', 'no_path', {})
    assert progress.failure['consecutive_same_code_count'] == 1


def test_partial_result_cannot_be_reported_as_full_success():
    progress = TaskProgress('全部放进去', 'collection', None)
    progress.select({'id': 'a', 'color': 'red'}, 'bin')
    progress.defer('a', 'unreachable')
    progress.select({'id': 'b', 'color': 'blue'}, 'bin')
    progress.complete('b', observation_id='after')
    assert progress.terminal_status('complete') == 'partial_success'
    assert progress.snapshot()['deferred'][0]['reason'] == 'unreachable'


def test_unknown_visibility_does_not_remove_an_unfinished_instance():
    progress = TaskProgress('所有物体放进去', 'collection', None)
    progress.observe([{'id': 'a', 'color': 'red'}, {'id': 'b', 'color': 'blue'}], 'o1')
    progress.assess_membership({'a': 'required', 'b': 'required'})
    progress.observe([{'id': 'a', 'color': 'red'}], 'o2')
    assert progress.snapshot()['unseen_ids'] == ['b']
    assert progress.terminal_status('complete') == 'needs_attention'


def test_completion_preserves_original_region_for_natural_language_quotas():
    progress = TaskProgress('把盒里的红色放到标记区', 'collection')
    source = {'id': 'red', 'color': 'red', 'visual_semantics': {'inside_id': 'bin'}}
    progress.observe([source], 'before')
    progress.assess_membership({'red': 'required'})
    progress.select(source, 'marker')
    progress.complete('red', observation_id='after')
    progress.observe([{'id': 'red', 'visual_semantics': {'on_marker_id': 'marker'}}], 'after')
    item = progress.snapshot()['completed'][0]
    assert item['visual_semantics'] == {'inside_id': 'bin'}
    assert item['source_observation_id'] == 'before'


def test_only_jev_can_exclude_unseen_unrequested_objects_from_completion():
    progress = TaskProgress('只放红色，黑色不动', 'collection')
    progress.observe([{'id': 'red'}, {'id': 'black'}], 'before')
    progress.assess_membership({'red': 'required'})
    progress.select({'id': 'red'}, 'bin')
    progress.complete('red', observation_id='after')
    progress.observe([{'id': 'red'}], 'after')
    assert progress.terminal_status('complete') == 'needs_attention'
    progress.assess_membership({'black': 'excluded'})
    assert progress.terminal_status('complete') == 'succeeded'


def test_completed_history_uses_execution_order_instead_of_detector_order():
    progress = TaskProgress('先红后绿', 'collection')
    progress.observe([{'id': 'green'}, {'id': 'red'}], 'before')
    progress.assess_membership({'green': 'required', 'red': 'required'})
    for identifier in ('red', 'green'):
        progress.select({'id': identifier}, 'bin')
        progress.complete(identifier, observation_id='after-' + identifier)
    progress.complete('red', observation_id='reverified-red')
    assert [item['id'] for item in progress.snapshot()['completed']] == ['red', 'green']


def inventory_observation(objects, identifier='before'):
    masks = {}
    for index, obj in enumerate(objects):
        mask = np.zeros((10, max(10, len(objects) * 5)), dtype=bool)
        mask[1:5, index * 5:index * 5 + 4] = True
        masks[obj['id']] = mask
    return {'observation_id': identifier, 'objects': objects,
            'views': {'3': {'T_world_camera': np.eye(4).tolist(), 'instance_masks': masks}},
            'arms': {'A': {'tcp': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]}}}


def test_head_image_numbering_is_stable_after_completion():
    progress = TaskProgress('全部放进盒子', 'collection', 'A')
    objects = [{'id': name, 'kind': 'cube', 'position': [0, 0, distance]}
               for name, distance in [('far', 3), ('near', 1), ('middle', 2)]]
    observation = inventory_observation([objects[1], objects[2], objects[0]])
    progress.observe(objects, 'before')
    progress.assess_membership({'near': 'required', 'middle': 'required', 'far': 'required'})
    progress.freeze_inventory(observation)
    assert progress.assign_display_ids(observation) == {'near': 1, 'middle': 2, 'far': 3}
    progress.select(objects[1], 'box')
    progress.complete('near', observation_id='after')
    progress.observe(objects[:1] + objects[2:], 'after')
    progress.assign_display_ids(inventory_observation(objects[:1] + objects[2:], 'after'))
    records = {item['id']: item for item in progress.snapshot()['objects']}
    assert {key: value['display_id'] for key, value in records.items()} == {'near': 1, 'middle': 2, 'far': 3}
    assert records['near']['status'] == 'completed'
    assert records['near']['visibility'] == 'unseen'


def test_new_detection_stays_an_obstacle_after_inventory_is_frozen():
    progress = TaskProgress('全部', 'collection')
    first = {'id': 'one', 'kind': 'cube', 'position': [0, 0, 1]}
    progress.observe([first], 'first')
    progress.assess_membership({'one': 'required'})
    progress.freeze_inventory(inventory_observation([first]))
    progress.select(first)
    progress.complete('one', observation_id='done')
    new = {'id': 'two', 'kind': 'cube', 'position': [0, 0, .5]}
    progress.observe([new], 'second')
    with pytest.raises(ValueError, match='invalid_task_membership'):
        progress.assess_membership({'two': 'required'})
    assert progress.assign_display_ids(inventory_observation([new], 'second')) == {}
    assert progress.snapshot()['completed'][0]['display_id'] == 1


def test_tcp_ranking_does_not_change_display_ids_or_choose_a_target():
    progress = TaskProgress('全部', 'collection', 'A')
    objects = [{'id': name, 'kind': 'cube', 'position': [0, 0, distance]}
               for name, distance in [('near_head', 1), ('near_tcp', 2.9)]]
    observation = inventory_observation(objects)
    progress.observe(objects, 'before')
    progress.assess_membership({'near_head': 'required', 'near_tcp': 'required'})
    progress.freeze_inventory(observation)
    ranked = progress.rank_candidates(observation, arm='A')
    assert [item['id'] for item in ranked] == ['near_tcp', 'near_head']
    assert [item['display_id'] for item in ranked] == [2, 1]
    assert ranked[0]['tcp_distance_m'] == pytest.approx(.1)
    assert progress.current_id is None


def test_completed_object_keeps_source_semantics_but_updates_measured_position():
    progress = TaskProgress('桌上的红色', 'collection')
    source = {'id': 'red', 'kind': 'cube', 'position': [1, 0, 0],
              'visual_semantics': {'on_table': True}}
    progress.observe([source], 'before')
    progress.assess_membership({'red': 'required'})
    progress.freeze_inventory(inventory_observation([source]))
    progress.select(source, 'box')
    progress.complete('red', observation_id='after')
    progress.observe([{**source, 'position': [2, 0, 0], 'visual_semantics': {'inside_id': 'box'}}], 'after')
    record = progress.snapshot()['completed'][0]
    assert record['position'] == [1, 0, 0]
    assert record['last_observed_position'] == [2, 0, 0]
    assert record['last_seen_observation_id'] == 'after'
    assert record['display_id'] == 1

