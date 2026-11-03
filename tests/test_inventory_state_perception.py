from copy import deepcopy

from test_generic_perception import generic_observation
from test_visual_inventory import client_with_answers, fixture

from luna_mujoco.demo_perception import DemoPerception, DemoVisualInventory
from luna_mujoco.perception import Perception


def test_observation_needs_only_task_graph_and_inventory(monkeypatch):
    observation = generic_observation()
    reference = observation['visual_reference']
    reference['policy_graph'] = {'nodes': []}
    reference['inventory_state'] = {'entities': reference['entities'], 'retired_entity_ids': [], 'transformations': {}}
    monkeypatch.setattr(Perception, 'observe', lambda *args, **kwargs: deepcopy(observation))
    monkeypatch.setattr('luna_mujoco.demo_perception.reference_with_container_tracks',
                        lambda observation, capture: observation['visual_reference'])
    engine = object.__new__(DemoPerception)
    engine.demo_tables = None
    result = engine.observe({'observation_id': 'next'}, source_hint={
        'scene_entities': observation['scene_entities'], 'visual_reference': reference})
    assert result['task_spec'] == reference['task_spec']
    assert 'demo_task' not in result


def test_retired_input_and_product_have_independent_identity_options():
    capture, detection = fixture()
    calls = []
    inventory = DemoVisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'product', 'visibility_product': 'visible', 'scene_inventory_closure': 'none'}, calls))
    initial = {'input': {'kind': 'object', 'color': 'red', 'description': 'original red parcel'}}
    product = {'kind': 'object', 'color': 'black', 'description': 'verified black product',
               'input_entity_id': 'input', 'process_id': 'process-1'}
    reference = {'entities': initial, 'task_spec': {'target_refs': ['input'], 'source_ref': 'tray'},
                 'inventory_state': {'entities': {**initial, 'product': product},
                     'retired_entity_ids': ['input'], 'transformations': {
                         'process-1': {'input_entity_id': 'input', 'output_entity_id': 'product'}}}}
    result = inventory.resolve(capture, [detection], entities=initial, reference=reference, instruction='process')
    assert result['groups'][0][0]['entity_id'] == 'product'
    question_names = [row['name'] for row in calls[0]['questions']]
    assert 'visibility_input' not in question_names
    assert 'visibility_product' in question_names
    options = [row['value'] for row in calls[0]['questions'][0]['choices']]
    assert 'product' in options and 'input' not in options


def release_fixture():
    import numpy as np
    capture, _ = fixture()
    view = capture['views']['3']
    view.update(T_world_camera=np.eye(4), intrinsics={'fx': 10., 'fy': 10., 'cx': 16., 'cy': 12.})
    capture.update(time=2., arms={'A': {'gripper_closed_command': False,
        'gripper_position_rad': 0., 'gripper_limits_rad': [0., 1.]}})
    capture['views'] = {camera: deepcopy(view) for camera in ('1', '2', '3')}
    slot = {'slot_id': 1, 'center_world_xy_m': [.4, 0.], 'polygon_world_xy_m': [
        [.3, -.1], [.5, -.1], [.5, .1], [.3, .1]], 'floor_z_m': .6, 'layout_revision': 'layout'}
    tcp = np.eye(4)
    tcp[:3, 3] = [.4, 0., .72]
    tcp = tcp.tolist()
    binding = {'arm': 'A', 'target_id': 'product', 'generation': 1, 'observation_id': 'before',
        'binding_id': 'proof', 'release_feedback_id': 'feedback-1-1.0', 'release_feedback_time': 1.,
        'destination_id': 'shelf', 'region_id': 'region:shelf', 'slot_id': 1, 'layout_revision': 'layout',
        'assigned_slot': slot, 'identity_source': 'execution_chain', 'needs_visible_inventory_check': False,
        'release_tcp': tcp, 'evidence': {'release_endpoint': {
            'source': 'measured_object_reference_attachment_gripper_and_actual_tcp',
            'at_release_goal': True, 'footprint_inside_region': True, 'actual_tcp_world': tcp,
            'tool_reference_point_m': [0., 0., 0.], 'actual_reference_world_m': [.4, 0., .72]}}}
    reference = {'observation_id': 'before', 'views': deepcopy(capture['views']),
        'entities': {'input': {'kind': 'object', 'description': 'red parcel'}},
        'task_spec': {'target_refs': ['input'], 'source_ref': 'tray', 'destination_ref': 'shelf'},
        'inventory_state': {'entities': {'product': {'kind': 'object', 'description': 'black product'}},
            'retired_entity_ids': ['input'], 'transformations': {'process': {
                'input_entity_id': 'input', 'output_entity_id': 'product',
                'evidence': {'private_simulator_pose': [1., 2., 3.]}}}}}
    return capture, binding, reference


def test_product_release_uses_current_identity_and_sanitized_lineage():
    import json

    from luna_mujoco.demo_transfer_verification import release_result_payload, verify_release_result
    capture, binding, reference = release_fixture()
    payload = release_result_payload(capture, binding, reference, 'place product on shelf')
    context = json.loads(payload['input'][0]['content'][0]['text'])
    assert context['target_id'] == 'product'
    assert context['appearance'] == {'kind': 'object', 'description': 'black product'}
    assert 'red parcel' not in json.dumps(payload)
    assert len([p for p in payload['input'][0]['content'] if p['type'] == 'input_image']) == 3
    assert 'private_simulator_pose' not in json.dumps(payload)
    assert 'pink' not in json.dumps(payload)
    assert 'release_position_check' not in context
    calls = []
    inventory = DemoVisualInventory(api_key='test', client=client_with_answers({
        'place_result': .99, 'workpiece_still_held': .01, 'observation_sufficient': .99}, calls))
    result = verify_release_result(inventory, capture, binding, reference)
    assert result['visual_evidence']['target_id'] == 'product'
    assert result['visual_evidence']['status'] == 'at_destination'
    assert result['geometry_updated'] is False
    assert result['visual_evidence']['release_position_check']['reference_inside_region'] is True


def test_completed_tracking_has_no_undeclared_height_limit():
    import numpy as np

    from luna_mujoco.demo_completed_tracking import _inside_slot
    _, binding, _ = release_fixture()
    assert _inside_slot(np.array([[.4, 0., 1.]]), binding['assigned_slot'])
    assert not _inside_slot(np.array([[.4, 0., 1.]]), {**binding['assigned_slot'], 'height_prior_m': .1})
