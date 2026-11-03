from copy import deepcopy

from luna_mujoco.inventory_map import reconcile_inventory, require_native_inventory_geometry
from luna_mujoco.scene_update import prepare_scene_update


def test_hybrid_wire_binds_geometry_to_scene_token_and_preserves_registered_depth():
    from test_perception_mapping import wire

    from luna_mujoco.perception_mapping import decode_mapping_payload

    rows = [{'id': 'source', 'kind': 'cube', 'obb': {'fixture': 1}}]
    decoded, scene = decode_mapping_payload(wire() | {'geometry_objects': rows})
    assert decoded['geometry_objects'] == rows
    assert scene['geometry_policy'] == 'registered_cad_or_observed_obb'
    assert scene['collision_world_source'] == 'inventory_cad_meshes_obb_and_measured_table_cuboids'
    _, changed = decode_mapping_payload(wire() | {'geometry_objects': [{'id': 'source', 'kind': 'cube', 'obb': {'fixture': 2}}]})
    assert changed['scene_token'] != scene['scene_token']
    assert decoded['views']['3']['depth_left_m'].shape == (2, 3)


def model():
    return {'voxel': {}, 'cuboid': {'semantic_instance::unknown': {
        'dims': [.1, .04, .03], 'pose': [.4, .1, .6, 1, 0, 0, 0],
        'geometry_source': 'observed_obb', 'instance_id': 'unknown', 'observation_id': 'one', 'generation': 1}},
        'mesh': {'semantic_instance::known': {'file_path': 'catalog/part.stl',
        'pose': [.5, .2, .6, 1, 0, 0, 0], 'geometry_source': 'registered_cad',
        'asset_id': 'part', 'source_sha256': 'a' * 64, 'mesh_sha256': 'b' * 64,
        'instance_id': 'known', 'observation_id': 'one', 'generation': 1}}}


def identity(observation='one'):
    return {'scene_token': observation, 'observation_id': observation, 'generation': 1,
            'target_id': 'known', 'exclusion_state': 'observation'}


def test_verified_hybrid_primitives_are_accepted_and_unseen_entities_retained():
    geometry = model()
    require_native_inventory_geometry(geometry)
    first = reconcile_inventory(None, identity(), [], geometry)
    second = reconcile_inventory(first, identity('two'), [], {'voxel': {}, 'mesh': {}, 'cuboid': {}})
    assert second[2] == first[2]
    assert second[0]['inventory_map']['entities']['known']['status'] == 'retained'


def test_new_registered_pose_replaces_previous_cad_without_ghost_history():
    first = reconcile_inventory(None, identity(), [], model())
    newer = deepcopy(model())
    newer['mesh']['semantic_instance::known']['pose'][0] += .001
    newer['mesh']['semantic_instance::known']['observation_id'] = 'two'
    second = reconcile_inventory(first, identity('two'), [], newer)
    assert list(second[2]['mesh']) == ['semantic_instance::known']
    assert second[2]['mesh']['semantic_instance::known'] == newer['mesh']['semantic_instance::known']


def test_cad_selection_and_release_prediction_preserve_mesh_then_invalidate():
    initial = identity()
    request = {'scene_token': 'one', 'generation': 1, 'target_id': 'known',
               'operation': 'select', 'exclusion_state': 'selected',
               'selected_surface_world_m': [[.5, .2, .6], [.5, .2, .61]]}
    selected, _, selected_model = prepare_scene_update(initial, [], model(), request)
    assert selected['disabled_obstacle_names'] == ('semantic_instance::known',)
    request.update(scene_token=selected['scene_token'], operation='transfer', exclusion_state='released',
        source='release_feedback_rigid_prediction', after_capture_id='two',
        T_new_world_from_old_world=[[1, 0, 0, .1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    released, _, moved = prepare_scene_update(selected, [], selected_model, request)
    assert moved['mesh']['semantic_instance::known']['pose'][0] == .6
    assert released['layer_evidence']['geometry_invalidated'] is True
