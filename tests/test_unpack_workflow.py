from luna_mujoco.task_progress import TaskProgress


def test_effect_invalidates_geometry_without_adding_an_unmeasured_obstacle():
    from test_scene_update import mapped

    from luna_mujoco.geometry_service import run_operation
    runtime, _ = mapped()
    grids = list(runtime.geometry_grids)
    result = run_operation(runtime, '/perception/update', {
        'scene_token': 'old', 'generation': 1, 'target_id': 'one',
        'operation': 'invalidate', 'exclusion_state': 'released',
        'action_id': 'mock-1', 'source': 'mock_vla'})
    assert all(a is b for a, b in zip(grids, runtime.geometry_grids, strict=True))
    assert result['observation_id'] == 'initial'
    assert result['layer_evidence']['rgbd_remapped'] is False
    assert result['layer_evidence']['measured'] is False
    assert set(runtime.geometry_model['cuboid']) == {'semantic_table::table'}
    assert result['inventory_map']['entities']['one']['status'] == 'stale'
    assert result['coverage']['inventory_complete'] is False
    assert result['layer_evidence']['action_id'] == 'mock-1'


def test_unpack_proposal_leaves_arm_selection_to_decisions_and_no_invented_cores():
    from test_demo_perception import demo_discovery, discovery_output
    from test_perception import capture_fixture
    body = discovery_output()
    source_bin = body['task']['source_ref']
    bin_template = next(item for item in body['entities'] if item['ref'] == source_bin)
    body['entities'].append({**bin_template, 'ref': 'right-bin', 'reference_camera_id': '1'})
    body['task'].update(processing=['vla_unpack'], destination_ref='right-bin', work_area_ref='large')
    capture, _ = capture_fixture()
    result = demo_discovery(body, []).discover(capture, instruction='逐件VLA拆包后放右盒')
    inventory = result['task_inventory']
    assert len({inventory['source_id'], inventory['destination_id'], inventory['work_area_id']}) == 3
    assert inventory['selected_ids'] == inventory['source_object_ids']
    assert result['task_spec']['processing'] == ['vla_unpack']
    assert result['policy_graph'] == body['graph']
    assert 'demo_task' not in result and 'item_lifecycle' not in inventory
    assert all('execution_arm' not in item for item in result['entities'].values())
    assert not any(entity['color'] == 'black' for entity in result['entities'].values())


def test_shared_work_area_uses_same_full_region_center_for_each_item():
    ledger = TaskProgress('unwrap')
    ledger.records = {name: {'display_id': index, 'transfer_index': 0} for index, name in enumerate(['a', 'b'], 1)}
    ledger.membership = {'a': 'required', 'b': 'required'}
    destination = {'id': 'zone', 'placement_reference': {'floor_z_m': .56,
        'position_world_m': [.36, 0, .56],
        'region_polygon_world_xy_m': [[.2, -.16], [.52, -.16], [.52, .16], [.2, .16]]}}
    requests = []
    for name in ['a', 'b']:
        ledger.current_id = name
        requests.append(ledger.bind_shared_work_area(destination, {'observation_id': 'obs', 'time': 0}))
    assert all(row['layout']['method'] == 'shared_work_area' for row in requests)
    assert all(len(row['layout']['cells']) == 1 for row in requests)
    assert all(row['layout']['cells'][0]['center_world_xy_m'] == [.36, 0.] for row in requests)
    assert [row['slot_id'] for row in requests] == [1, 1]
    assert [row['object_display_id'] for row in requests] == [1, 2]


def test_intermediate_transfer_requires_verified_processing_and_product_delivery():
    from test_policy_entity_ledger import ledger, product, release

    progress = ledger(('vla_unpack',))
    assert release(progress, 'item', 'bench', 1)
    assert progress.completed_input_ids() == set()
    assert progress.records['item']['status'] == 'observed'
    progress.register_product(product())
    assert progress.completed_input_ids() == set()
    assert progress.records['item']['status'] == 'consumed'
    assert release(progress, 'output', 'tray', 2)
    assert progress.completed_input_ids() == {'item'}
    assert progress.records['output']['status'] == 'completed'
