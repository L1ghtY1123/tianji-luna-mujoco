"""CAD shells remain complete, current, hollow and SDK-compatible downstream."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from luna_mujoco.cad_assets import cad_asset
from luna_mujoco.collision_mapping import collision_layer_bounds
from luna_mujoco.demo_bin_geometry import fit_bin_shell, published_bin_shell_layers


@pytest.fixture(scope='module')
def completion():
    asset = cad_asset('blue_box')
    points = np.load(Path(__file__).parent/'fixtures/cad_bin_registered_depth.npz')['right']
    prior = {'kind': 'bin', 'color': 'blue', 'instance_id': 'right', 'observation_id': 'capture',
        'generation': 3, 'source': 'user_supplied_cad_shape', 'asset_id': 'blue_box',
        'cad_sha256': asset['source']['sha256'], 'outer_dimensions_m': asset['size_m']}
    return fit_bin_shell(points, instance_id='right', observation_id='capture', generation=3, prior=prior)


def scene_and_model(completion):
    scene = {'scene_token': 'current', 'observation_id': 'capture', 'generation': 3,
        'mapping_diagnostics': {'bin_shell_completions': {'right': completion}}}
    model = {'cuboid': {}, 'voxel': {}, 'mesh': {
        f'semantic_instance::right::bin_shell::{name}': copy.deepcopy(mesh)
        for name, mesh in completion['meshes'].items()}}
    scene['collision_layer_bounds'] = collision_layer_bounds([], model, scene)
    return scene, model


def test_published_meshes_use_public_sdk_fields_and_do_not_fill_inner_cavity(completion):
    _, model = scene_and_model(completion)
    fields = json.loads((Path(__file__).parent/'fixtures/curobo_scene_constructor_fields.json').read_text())['mesh']
    assert len(model['mesh']) == 1
    for mesh in model['mesh'].values():
        assert set(mesh) <= set(fields)
        shape = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'], process=True)
        assert shape.is_watertight and shape.is_winding_consistent
        assert len(shape.faces) == 33082
        assert not shape.contains([[0, 0, 0]]).any()
        assert shape.contains([[0, 0, -.070]]).all()


@pytest.mark.parametrize('damage', ['missing', 'hash', 'source', 'generation', 'token', 'disabled', 'wrong_id'])
def test_cad_bounds_reject_partial_stale_unknown_or_disabled_parts(completion, damage):
    scene, _ = scene_and_model(completion)
    layers = scene['collision_layer_bounds']['layers']
    first = next(iter(layers))
    assert len(published_bin_shell_layers(scene, 'right')) == 1
    if damage == 'missing':
        layers.pop(first)
    elif damage == 'hash':
        layers[first]['part_sha256'] = 'unverified'
    elif damage == 'source':
        layers[first]['geometry_source'] = 'rgbd_measured'
    elif damage == 'generation':
        layers[first]['generation'] -= 1
    elif damage == 'token':
        scene['scene_token'] = 'new'
    elif damage == 'disabled':
        scene['disabled_obstacle_names'] = [first]
    else:
        layers[first]['instance_id'] = 'left'
    with pytest.raises(ValueError, match='demo_bin_shell'):
        published_bin_shell_layers(scene, 'right')


def test_builder_keeps_exact_clean_measurements_and_both_complete_cad_surfaces():
    from dataclasses import replace

    from test_demo_mapping import table_evidence
    from test_semantic_builder import builder, request

    value, segmenter, mapper, _ = builder()
    arrays = np.load(Path(__file__).parent/'fixtures/cad_bin_registered_depth.npz')
    points = {'source': arrays['source'], 'other': arrays['right']}
    segmenter.filter_isolated_surfaces = lambda views: tuple(replace(view, instance_points_base_m={
        name: cloud if view.camera_id == camera else np.empty((0, 3))
        for name, cloud, camera in [('source', points['source'], '2'), ('other', points['other'], '1')]})
        for view in views)
    asset = cad_asset('blue_box')
    priors = {name: {'kind': 'bin', 'color': 'blue', 'instance_id': name,
        'observation_id': 'capture-42', 'generation': 4, 'source': 'user_supplied_cad_shape',
        'asset_id': 'blue_box', 'cad_sha256': asset['source']['sha256'],
        'outer_dimensions_m': asset['size_m']} for name in points}
    result = value.build(request('observation') | {'demo_mapping': True, 'scene_policy': 'demo_inventory',
        'demo_tables': table_evidence(), 'demo_bin_priors': priors})
    assert len(result.scene_model['mesh']) == 2
    assert len(result.scene_model['voxel']) == 2
    np.testing.assert_array_equal(mapper.instances['source'][0]['2'], points['source'])
    np.testing.assert_array_equal(mapper.instances['other'][0]['1'], points['other'])


def test_cache_reserves_complete_meshes_and_same_names_replace_old_geometry(completion):
    from tianji_curobo.curobo_scene import collision_cache_for_scene
    from tianji_curobo.mixed_esdf_scene import update_mixed_esdf_scene

    from luna_mujoco.geometry_service import GeometryRuntime

    scene, model = scene_and_model(completion)
    model['mesh'].update({name.replace('::right::', '::left::'): copy.deepcopy(mesh)
                         for name, mesh in list(model['mesh'].items())})
    assert collision_cache_for_scene(model)['mesh'] >= 2
    runtime = GeometryRuntime()
    runtime.geometry_scene, runtime.geometry_model = scene, model
    runtime._prepare_scene_cache_unlocked()
    assert runtime._collision_cache['mesh'] >= 2
    calls = []
    planner = SimpleNamespace(scene_collision_checker=SimpleNamespace(data=SimpleNamespace(
        meshes=SimpleNamespace(clear=lambda **kw: calls.append(('clear', kw))))),
        update_world=lambda value: calls.append(('replace', value)))
    first = SimpleNamespace(mesh=list(model['mesh'].values()), voxel=[])
    second = SimpleNamespace(mesh=first.mesh[:1], voxel=[])
    update_mixed_esdf_scene(planner, first)
    update_mixed_esdf_scene(planner, second)
    assert calls == [('clear', {'clear_warp_cache': True}), ('replace', first),
                     ('clear', {'clear_warp_cache': True}), ('replace', second)]


def cad_drop_region(completion):
    from test_drop_region_presentation import measured_drop_inputs

    from luna_mujoco.demo_unpack_workflow import right_bin_occupied_bounds

    capture, _, reference = measured_drop_inputs()
    current = copy.deepcopy(completion)
    current.update(instance_id='right-bin', observation_id=reference['observation_id'])
    scene = {'scene_token': 'current', 'observation_id': reference['observation_id'], 'generation': 3,
        'mapping_diagnostics': {'bin_shell_completions': {'right-bin': current}}}
    model = {'mesh': {'semantic_instance::right-bin::bin_shell::'+name: mesh
                     for name, mesh in current['meshes'].items()}}
    scene['collision_layer_bounds'] = collision_layer_bounds([], model, scene)
    scene['collision_layer_bounds']['layers']['semantic_instance::right-bin'] = {
        'kind': 'voxel', 'enabled': True, 'source': 'native_surface_voxel_cell_bounds',
        'bounds_source': 'all_native_surface_cells_before_preview_sampling',
        'cell_count': 100, 'voxel_size_m': .02,
        'minimum_world_m': [.07, -.79, .41], 'maximum_world_m': [.33, -.42, .53]}
    destination = {'id': 'right-bin', 'kind': 'bin', 'placement_reference': {
        'observation_id': reference['observation_id'],
        'region_polygon_world_xy_m': [[.05, -.81], [.35, -.81], [.35, -.40], [.05, -.40]]}}
    reference['drop_bin_region'] = right_bin_occupied_bounds(destination, scene, reference)
    return capture, reference, destination, scene


def test_transport_and_drop_crop_use_the_same_complete_cad_native_union(completion):
    from luna_mujoco.demo_lifecycle_perception import _drop_bin_zooms
    from luna_mujoco.demo_unpack_workflow import right_bin_transport_goal

    capture, reference, destination, scene = cad_drop_region(completion)
    parts, evidence = _drop_bin_zooms(capture, reference, 'right-bin')
    assert len(parts) == 8 and len(evidence) == 4
    assert {(row['frame_role'], row['source_camera_id']) for row in evidence} == {
        ('BEFORE', '1'), ('CURRENT', '1'), ('BEFORE', '3'), ('CURRENT', '3')}
    region = reference['drop_bin_region']
    assert len(region['layer_names']) == 2 and len(region['derived_shell_components']) == 1
    assert .544 < region['maximum_world_m'][2] < .547
    prompt_region = {key: value for key, value in region.items()
                     if key not in {'derived_shell_components', 'layer_names'}}
    assert all(row['measured_region'] == prompt_region for row in evidence)
    held = {'points_tool_m': [[-.02, -.01, -.07], [.02, .01, -.01]]}
    goal, proof = right_bin_transport_goal(np.eye(4), held, destination, scene, reference)
    minimum = region['maximum_world_m'][2] + .10 + .07
    assert goal[2, 3] >= minimum
    assert proof['minimum_tcp_z_m'] == pytest.approx(minimum)
    assert proof['derived_shell_components'] == region['derived_shell_components']
    assert all(row['synthetic_geometry_is_measured'] is False
               for row in region['derived_shell_components'].values())


@pytest.mark.parametrize('damage', ['missing_part', 'stale', 'hash', 'filled_cuboid'])
def test_drop_crop_rejects_incomplete_or_relabelled_cad_union_before_model(completion, damage):
    from luna_mujoco.demo_lifecycle_perception import _drop_bin_zooms
    from luna_mujoco.visual_api import VisualInventoryError

    capture, reference, _, _ = cad_drop_region(completion)
    layers = reference['drop_bin_region']['derived_shell_components']
    name = next(iter(layers))
    if damage == 'missing_part':
        layers.pop(name)
    elif damage == 'stale':
        layers[name]['observation_id'] = 'old'
    elif damage == 'hash':
        layers[name]['part_sha256'] = 'different'
    else:
        layers[name]['kind'] = 'cuboid'
    with pytest.raises(VisualInventoryError, match='drop_bin_region_invalid'):
        _drop_bin_zooms(capture, reference, 'right-bin')
