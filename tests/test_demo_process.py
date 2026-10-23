"""Input entities, process records and observed products have separate IDs."""
from copy import deepcopy

import pytest


def task():
    return {'workflow': 'unpack_transfer', 'scope': 'all', 'transport_ids': ['pink-1', 'pink-2'],
        'source_object_ids': ['pink-1', 'pink-2'], 'right_bin_id': 'bin-right',
        'item_lifecycle': {key: {'phase': 'wrapped_source', 'appearance': 'wrapped',
            'appearance_revision': 0, 'to_table_arm': 'A', 'to_right_bin_arm': 'B'} for key in ('pink-1', 'pink-2')}}


def test_pending_unpack_includes_the_pink_item_already_placed_on_table():
    from luna_mujoco.demo_process import initialize_processes, process_snapshot
    value = task()
    initialize_processes(value, {})
    value['item_lifecycle']['pink-1']['phase'] = 'unpack_pending'
    result = process_snapshot(value)
    assert result['pending_unpack_ids'] == ['pink-1', 'pink-2']
    assert result['not_started_input_ids'] == ['pink-2']
    assert result['pending_delivery_product_ids'] == []


def test_current_entities_do_not_double_count_historical_products_or_target_regions():
    from luna_mujoco.demo_process import inventory_views
    entities = {
        'pink': {'kind': 'object', 'geometry_status': 'historical_consumed'},
        'black': {'kind': 'object', 'geometry_status': 'historical'},
        'collection': {'kind': 'collection', 'members': ['black']},
        'region': {'kind': 'marker', 'roles': ['target_region']},
        'table': {'kind': 'table'},
    }
    result = inventory_views(entities)
    assert set(result['current_entities']) == {'table', 'collection'}
    assert set(result['historical_entities']) == {'pink', 'black'}
    assert set(result['target_regions']) == {'region'}
    result['current_entities']['table']['kind'] = 'changed'
    assert entities['table']['kind'] == 'table'


def test_process_id_is_separate_and_product_is_not_created_for_pending_input():
    from luna_mujoco.demo_process import initialize_processes, process_snapshot
    value = task()
    entities = {key: {'id': key, 'display_id': i} for i, key in enumerate(value['source_object_ids'], 1)}
    initialize_processes(value, entities)
    before = deepcopy(value)
    initialize_processes(value, entities)
    assert value == before
    assert set(value['process_records']).isdisjoint(value['source_object_ids'])
    assert all(row['output_entity_id'] is None for row in value['process_records'].values())
    assert process_snapshot(value)['pending_unpack_ids'] == ['pink-1', 'pink-2']
    assert value['product_entities'] == {}


def test_confirmed_visible_result_creates_separate_product_without_rewriting_source():
    from luna_mujoco.demo_process import create_product, initialize_processes, process_for_input
    value = task()
    initialize_processes(value, {'pink-1': {'display_id': 1}, 'pink-2': {'display_id': 2}})
    source = value['item_lifecycle']['pink-1']
    source.update(phase='unpack_pending', mock_action={'action_id': 'unpack', 'target_id': 'pink-1'})
    result = {'success': True, 'status': 'confirmed', 'target_id': 'pink-1', 'observation_id': 'current',
        'generation': 2, 'request_hash': 'visual-request', 'appearance_revision': 1,
        'visual_evidence': {'judgments': {'unpack_result': 'yes'}, 'confidence': .95, 'view_scope': 'fused_current_three_rgb'}}
    product = create_product(value, 'pink-1', result, {'observation_id': 'current', 'generation': 2}, owner_id='run')
    process = process_for_input(value, 'pink-1')
    assert product['id'] not in value['source_object_ids'] and product['id'] != process['id']
    assert process['input_entity_id'] == 'pink-1' and process['output_entity_id'] == product['id']
    assert value['source_object_ids'] == ['pink-1', 'pink-2']
    assert source['phase'] == 'consumed' and source['appearance'] == 'wrapped'
    assert value['item_lifecycle'][product['id']]['unpacking_evidence']['target_id'] == 'pink-1'
    assert product['display_label'] == 'P1' and product['input_display_id'] == 1
    assert create_product(value, 'pink-1', result, {'observation_id': 'current', 'generation': 2}, owner_id='run') == product


@pytest.mark.parametrize('change', ['unknown', 'wrong_target', 'wrong_frame'])
def test_unconfirmed_or_mismatched_result_does_not_mint_product(change):
    from luna_mujoco.demo_process import create_product, initialize_processes
    value = task()
    initialize_processes(value, {})
    value['item_lifecycle']['pink-1']['phase'] = 'unpack_pending'
    result = {'success': True, 'status': 'confirmed', 'target_id': 'pink-1', 'observation_id': 'current',
        'generation': 2, 'request_hash': 'hash', 'visual_evidence': {'judgments': {'unpack_result': 'yes'}, 'confidence': .95, 'view_scope': 'fused_current_three_rgb'}}
    result[{'unknown': 'success', 'wrong_target': 'target_id', 'wrong_frame': 'observation_id'}[change]] = False if change == 'unknown' else 'other'
    with pytest.raises(ValueError, match='product'):
        create_product(value, 'pink-1', result, {'observation_id': 'current', 'generation': 2}, owner_id='run')
    assert not value['product_entities']


def test_completed_products_form_explicit_collection_without_aliasing_members():
    from luna_mujoco.demo_lifecycle_perception import lifecycle_entities
    from luna_mujoco.demo_process import completed_product_collection, initialize_processes, process_snapshot
    value = task()
    initialize_processes(value, {})
    value['product_entities'] = {key: {'id': key, 'kind': 'object', 'color': 'black'} for key in ('product-a', 'product-b')}
    for row, product in zip(value['process_records'].values(), value['product_entities'], strict=True):
        original = row['input_entity_id']
        value['item_lifecycle'][original]['phase'] = 'consumed'
        value['item_lifecycle'][product] = {**value['item_lifecycle'][original], 'phase': 'completed',
            'appearance': 'unpacked', 'appearance_revision': 1, 'input_entity_id': original}
        row['output_entity_id'] = product
    collection = completed_product_collection(value)
    assert collection['kind'] == 'collection' and collection['members'] == ['product-a', 'product-b']
    assert collection['id'] not in collection['members']
    entities = lifecycle_entities({key: {'id': key, 'kind': 'object'} for key in value['source_object_ids']}, value)
    assert set(entities) == {collection['id']}
    result = process_snapshot(value)
    assert len(result['completed_process_ids']) == 2 and result['pending_unpack_ids'] == []
    assert result['product_ids'] == ['product-a', 'product-b']


def test_ledger_registers_product_as_new_table_source_and_preserves_pink_number():
    from luna_mujoco.demo_process import ProcessTaskProgress, initialize_processes
    value = task()
    initialize_processes(value, {})
    ledger = ProcessTaskProgress('unwrap', route='box_table_box')
    ledger.process_task = value
    ledger.records = {'pink-1': {'id': 'pink-1', 'display_id': 1, 'status': 'pending', 'stage_index': 1,
                               'destination_visual_evidence': {'status': 'at_destination'}}}
    ledger.scene_entities = {'pink-1': {'id': 'pink-1', 'display_id': 1}}
    ledger.next_display_id = 3
    product = {'id': 'product-a', 'input_entity_id': 'pink-1', 'process_id': value['process_order'][0]}
    value['product_entities']['product-a'] = deepcopy(product)
    ledger.register_product(product)
    assert ledger.records['pink-1']['display_id'] == 1 and ledger.records['pink-1']['status'] == 'consumed'
    assert ledger.records['product-a']['display_id'] == value['product_entities']['product-a']['display_id'] == 3
    assert ledger.records['product-a']['initial_source']['initial_parent_kind'] == 'table'
    assert ledger.pending_route_id() == 'product-a'
    assert 'collision' not in ledger.scene_entities['pink-1'].get('roles', [])


def test_scene_catalog_retires_consumed_and_completed_geometry_and_registers_aggregate():
    from luna_mujoco.demo_process import ProcessTaskProgress
    value = task()
    value['product_entities'] = {'product-a': {'id': 'product-a', 'kind': 'object', 'color': 'black'}}
    value['item_lifecycle']['pink-1']['phase'] = 'consumed'
    value['item_lifecycle']['product-a'] = {'phase': 'completed'}
    ledger = ProcessTaskProgress('unwrap')
    ledger.process_task = value
    ledger.scene_entities = {key: {'id': key, 'roles': ['collision'], 'segmentation_query': {'prompt': key}}
                             for key in ('pink-1', 'pink-2', 'product-a')}
    ledger.sync_process_geometry()
    assert all('collision' not in ledger.scene_entities[key]['roles'] for key in ('pink-1', 'product-a'))
    assert ledger.scene_entities['pink-2']['roles'] == ['collision']
    collection = ledger.scene_entities['product-collection:bin-right']
    assert collection['kind'] == 'collection' and collection['members'] == ['product-a']
    assert 'collision' in collection['roles']


def test_program_target_switches_to_product_and_completion_counts_processes():
    import asyncio
    from types import MethodType, SimpleNamespace

    from luna_mujoco.demo_process import initialize_processes
    from luna_mujoco.demo_program import DemoProgram
    from luna_mujoco.demo_runner import DemoRunner
    value = task()
    initialize_processes(value, {})
    runner = object.__new__(DemoRunner)
    runner.demo_task, runner.program_steps, runner.state = value, 0, {'facts': {}}
    runner.task_progress = SimpleNamespace(stage_arms={}, completed_ids=[])
    runner.reset_transfer = lambda context: None
    runner.check_generation = lambda: None
    calls = []
    async def step(self, name, context, arguments):
        row = value['process_records'][self.state['process_id']]
        if name == 'verify_unpack':
            product = 'product-' + row['input_entity_id']
            row['output_entity_id'] = product
            value['item_lifecycle'][product] = {**value['item_lifecycle'][row['input_entity_id']], 'phase': 'unpacked_confirmed'}
        elif name == 'verify_drop':
            value['item_lifecycle'][row['output_entity_id']]['phase'] = 'completed'
        calls.append((name, arguments, self.state['process_id']))
        return {'success': True}
    runner._step = MethodType(step, runner)
    nodes = DemoProgram.model_validate({'version': 1, 'steps': [{'type': 'for_each_target', 'between_targets': [],
        'body': [{'type': 'skill', 'skill': name, 'arguments': []} for name in
                 ('identify_target', 'verify_unpack', 'identify_target', 'verify_drop')]}]}).steps
    asyncio.run(runner._execute_program(nodes, {}))
    assert [args['target_id'] for name, args, _ in calls if name == 'identify_target'] == [
        'pink-1', 'product-pink-1', 'pink-2', 'product-pink-2']
    runner._confirm_program_completion({})
    assert runner.state['status'] == 'succeeded'
