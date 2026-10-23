from luna_mujoco.demo_program import DemoProgram
from luna_mujoco.task_progress import TaskProgress


def test_mock_envelope_keeps_measured_grid_tensor_and_exposes_action_provenance():
    from test_scene_update import mapped

    from luna_mujoco.geometry_service import run_operation
    runtime, _ = mapped()
    grids = list(runtime.geometry_grids)
    result = run_operation(runtime, '/perception/update', {
        'scene_token': 'old', 'generation': 1, 'target_id': 'one',
        'operation': 'action_envelope', 'exclusion_state': 'released',
        'bounds': {'minimum_world_m': [.3, -.2, .56], 'maximum_world_m': [.4, -.1, .66]},
        'action_id': 'mock-1', 'source': 'mock_vla'})
    assert all(a is b for a, b in zip(grids, runtime.geometry_grids, strict=True))
    assert result['observation_id'] == 'initial'
    assert result['layer_evidence']['rgbd_remapped'] is False
    assert result['layer_evidence']['measured'] is False
    assert 'action_envelope::one' in runtime.geometry_model['cuboid']
    from luna_mujoco.sdk_scene import sdk_scene_model
    public = sdk_scene_model(runtime.geometry_model)
    assert set(public['cuboid']['action_envelope::one']) == {'dims', 'pose'}
    assert runtime.geometry_model['cuboid']['action_envelope::one']['action_id'] == 'mock-1'


def test_unpack_discovery_binds_two_arms_without_inventing_visible_black_cores():
    from test_demo_perception import demo_discovery, discovery_output
    from test_perception import capture_fixture
    body = discovery_output()
    sources = [item for item in body['entities'] if item['color'] == 'pink']
    source_bin = body['task']['source_bin_ref']
    bin_template = next(item for item in body['entities'] if item['ref'] == source_bin)
    body['entities'].append({**bin_template, 'ref': 'right-bin', 'reference_camera_id': '1'})
    body['task'].update(workflow='unpack_transfer', right_bin_ref='right-bin', transport_arms=[
        {'object_ref': item['ref'], 'to_table_arm': 'A', 'to_right_bin_arm': 'B'} for item in sources])
    capture, _ = capture_fixture()
    result = demo_discovery(body, []).discover(capture, instruction='逐件VLA拆包后放右盒')
    task = result['demo_task']
    assert task['workflow'] == 'unpack_transfer'
    assert task['right_bin_id'] != task['source_bin_id']
    assert all(item['to_table_arm'] == 'A' and item['to_right_bin_arm'] == 'B'
               for item in task['item_lifecycle'].values())
    assert not any(entity['color'] == 'black' for entity in result['entities'].values())


def test_program_can_express_unpack_and_attached_drop_without_arbitrary_poses():
    calls = ['prepare_unpack', 'mock_unpack', 'verify_unpack', 'move_to_observe',
             'observe_scene', 'plan_transport', 'execute_transport', 'drop', 'verify_drop']
    program = DemoProgram.model_validate({'version': 1, 'steps': [
        {'type': 'skill', 'skill': name, 'arguments': []} for name in calls]})
    assert [node.skill for node in program.steps] == calls


def test_shared_work_area_uses_same_full_region_center_for_each_item():
    ledger = TaskProgress('unwrap', route='box_table_box')
    ledger.records = {name: {'display_id': index, 'stage_index': 0} for index, name in enumerate(['a', 'b'], 1)}
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


def test_table_stage_never_completes_whole_unpack_item():
    ledger = TaskProgress('unwrap', route='box_table_box')
    ledger.records = {'a': {'id': 'a', 'status': 'selected', 'stage_index': 0, 'destination_id': 'zone'}}
    ledger.current_id = 'a'
    ledger.complete('a', observation_id='table-confirmed')
    assert ledger.completed_ids == []
    assert ledger.records['a']['stage_index'] == 1
    ledger.records['a']['destination_id'] = 'right-bin'
    ledger.complete('a', observation_id='drop-confirmed')
    assert ledger.completed_ids == ['a']
