"""Collision display must come from published map products, not raw display points."""
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest


def test_preview_preserves_actual_layers_distances_and_inferred_table():
    from luna_mujoco import collision_mapping

    centers = np.array([[.3, .1, .7], [.4, .2, .8]], dtype='<f4')
    table = {'pose': [.5, 0., .45, 1., 0., 0., 0.], 'dims': [.8, .6, .305]}
    plane = {'normal_xyz': [0, 0, 1], 'offset_m': -.6,
             'observed_polygon_xy_m': [[.2, -.2], [.8, -.2], [.8, .2]], 'revision': 'plane-1'}
    candidate = SimpleNamespace(integration=SimpleNamespace(pointcloud_data=struct.pack('<I', 2) + centers.tobytes() + bytes(6), surface_voxel_size_m=.01),
        table=SimpleNamespace(model=table, support_plane=plane, height_m=.6, normal_xyz=(0, 0, 1), inlier_count=50),
        camera_metrics={'3': {'removed_robot_point_count': 8}})
    grids = [SimpleNamespace(name=name, pose=[.5, 0, .8, 1, 0, 0, 0], dims=[.02, .02, .02],
        voxel_size=.01, feature_tensor=np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 100 - .03)
        for name in ('block_sparse_esdf_grid', 'semantic_instance::source')]
    model = {'cuboid': {'semantic_table_forbidden': table}, 'mesh': {}, 'voxel': {
        'semantic_instance::source': {'centers_base_m': [[.6, 0, .7]], 'surface_voxel_size': .01}}}
    identity = {'scene_token': 'rgbd:map', 'observation_id': 'capture-4', 'generation': 4,
                'disabled_obstacle_names': ('semantic_instance::source',)}
    result = collision_mapping.collision_scene_preview(candidate, grids, model, identity)
    assert result['scene_token'] == 'rgbd:map' and result['frame'] == 'world'
    assert result['table']['provenance'] == 'inferred_ransac_support_completion'
    assert result['table']['model'] == table and result['table']['support_plane'] == plane
    assert result['coverage']['unknown_space_motion_gate'] is False
    assert result['coverage']['full_environment_coverage'] is False
    background, source = result['voxel_layers']
    np.testing.assert_allclose(background['surface_centers_world_m'], centers)
    assert source['surface_centers_world_m'] == [[.6, 0, .7]]
    assert background['enabled'] and not source['enabled']
    np.testing.assert_array_equal(background['esdf']['sample_flat_indices'], np.arange(8))
    np.testing.assert_allclose(background['esdf']['sample_distances_m'], grids[0].feature_tensor.ravel())
    assert background['esdf']['shape_xyz'] == [2, 2, 2]
    json.dumps(result, allow_nan=False)
    grids[0].feature_tensor[:] = 99
    table['dims'][0] = 99
    assert result['table']['model']['dims'][0] == .8
    assert max(background['esdf']['sample_distances_m']) < .1


@pytest.mark.parametrize('include_background', [False, True])
def test_instance_preview_uses_named_surface_when_background_is_omitted_or_reordered(include_background):
    from luna_mujoco.collision_mapping import collision_scene_preview

    background = np.array([[.2, -.1, .7]], dtype='<f4') if include_background else np.empty((0, 3), dtype='<f4')
    table = {'pose': [.5, 0., .45, 1., 0., 0., 0.], 'dims': [.8, .6, .3]}
    candidate = SimpleNamespace(integration=SimpleNamespace(
        pointcloud_data=struct.pack('<I', len(background)) + background.tobytes(), surface_voxel_size_m=.02),
        table=SimpleNamespace(model=table, height_m=.6, normal_xyz=(0, 0, 1), inlier_count=50),
        camera_metrics={})
    names = ['semantic_instance::bin', 'semantic_instance::cube']
    if include_background:
        names.append('block_sparse_esdf_grid')
    grids = [SimpleNamespace(name=name, pose=[.5, 0, .8, 1, 0, 0, 0], dims=[.02, .02, .02],
        voxel_size=.01, feature_tensor=np.arange(8, dtype=np.float32).reshape(2, 2, 2)) for name in names]
    model = {'voxel': {
        'semantic_instance::bin': {'centers_base_m': [[.6, 0, .7], [.61, 0, .7]], 'surface_voxel_size': .01},
        'semantic_instance::cube': {'centers_base_m': [[.4, .1, .8]], 'surface_voxel_size': .01}}}
    identity = {'scene_token': 'rgbd:map', 'observation_id': 'capture-4', 'generation': 4,
                'disabled_obstacle_names': ('semantic_instance::cube',),
                'mapping_diagnostics': {'background_grid_omitted': not include_background}}
    result = collision_scene_preview(candidate, grids, model, identity)
    bin_layer, cube_layer = result['voxel_layers'][:2]
    assert bin_layer['role'] == cube_layer['role'] == 'semantic_instance'
    assert bin_layer['surface_centers_world_m'] == [[.6, 0, .7], [.61, 0, .7]]
    assert bin_layer['mapper_surface_preview_count'] == bin_layer['displayed_surface_count'] == 2
    assert bin_layer['surface_voxel_size_m'] == .01
    assert cube_layer['surface_centers_world_m'] == [[.4, .1, .8]]
    assert bin_layer['enabled'] and not cube_layer['enabled']
    if include_background:
        background_layer = result['voxel_layers'][2]
        assert background_layer['role'] == 'background'
        np.testing.assert_allclose(background_layer['surface_centers_world_m'], background)
        assert background_layer['surface_voxel_size_m'] == .02


def test_mapping_input_preview_keeps_camera_and_pixel_provenance():
    from test_perception_mapping import wire

    from luna_mujoco.perception_mapping import decode_mapping_payload, filter_mapping_views

    decoded, _ = decode_mapping_payload(wire())
    filtering = filter_mapping_views(decoded, {})
    preview = filtering['preview']
    assert preview['camera_ids'] == ['1'] * 6 + ['2'] * 6 + ['3'] * 6
    assert preview['pixels_rc'][:6] == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert preview['cameras']['1']['valid_depth_count'] == 6
    assert preview['cameras']['1']['invalid_depth_count'] == 0
    assert preview['cameras']['1']['intrinsics'] == decoded['views']['1']['left_intrinsics']


def test_transport_preserves_original_sensor_pixels_through_letterboxing():
    from test_rgbd_algorithms import observed_scene

    from luna_mujoco.algorithms import Algorithms
    from luna_mujoco.perception_mapping import decode_mapping_payload, filter_mapping_views

    sent = []
    Algorithms(curobo_request=lambda path, body: (sent.append(body) or {'scene_token': 'test'})).build_scene(
        observed_scene(), 'seen-red')
    decoded, _ = decode_mapping_payload(sent[0])
    preview = filter_mapping_views(decoded, {})['preview']
    assert preview['cameras']['3']['captured_sensor']['image_shape_hw'] == [8, 8]
    assert preview['cameras']['3']['image_shape_hw'] == [480, 640]
    assert preview['cameras']['3']['captured_sensor']['valid_depth_count'] == 64
    for point, (row, column) in zip(preview['points_world_m'], preview['pixels_rc'], strict=True):
        assert 0 <= row < 8 and 0 <= column < 8
        np.testing.assert_allclose(point, [(column - 4) / 10, (row - 4) / 10, 1.])
