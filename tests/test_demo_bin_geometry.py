"""The frozen source-wall gap requires an explicit hollow, measured-anchored prior."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def frozen_points():
    return np.load(Path(__file__).parent / 'fixtures' / '9d005_source_bin_points.npz')['points']


def fit(points):
    assert importlib.util.find_spec('luna_mujoco.demo_bin_geometry'), 'measured bin shell completion is missing'
    from luna_mujoco import demo_bin_geometry as geometry
    return geometry.fit_bin_shell(points, instance_id='source', observation_id='capture', generation=2)


def contains(model, point):
    pose = np.asarray(model['pose'])
    local = Rotation.from_quat(pose[3:], scalar_first=True).inv().apply(np.asarray(point)-pose[:3])
    return bool(np.all(np.abs(local) <= np.asarray(model['dims'])/2 + 1e-8))


def test_frozen_occluded_wall_is_covered_without_filling_the_cavity_or_changing_measurements():
    points = frozen_points()
    original = points.copy()
    contact = [.0595014407, .6538195689, .5999932705]  # Offline failure witness, never a fit input.
    assert np.linalg.norm(points-contact, axis=1).min() > .08
    result = fit(points)
    assert any(contains(model, contact) for model in result['cuboids'].values())
    assert not any(contains(model, [.20, .60, .50]) for model in result['cuboids'].values())
    assert len(result['cuboids']) == 5
    assert result['source'] == 'measured_planes_and_user_bin_dimensions'
    assert result['synthetic_geometry_is_measured'] is False
    assert result['observation_id'] == 'capture' and result['generation'] == 2
    np.testing.assert_allclose(result['wall_thickness_xy_m'], [.012, .012], atol=.001)
    np.testing.assert_array_equal(points, original)


def test_rotated_translated_registered_bin_uses_measured_pose_not_configured_scene_xy():
    points = frozen_points().astype(float)
    rotation = Rotation.from_euler('z', 31, degrees=True)
    translation = np.array([.34, -.24, .08])
    result = fit(rotation.apply(points)+translation)
    contact = rotation.apply([.0595014407, .6538195689, .5999932705])+translation
    assert any(contains(model, contact) for model in result['cuboids'].values())
    assert not any(contains(model, rotation.apply([.20, .60, .50])+translation)
                   for model in result['cuboids'].values())


def test_sparse_high_hand_contamination_does_not_define_bin_top():
    points = frozen_points()
    xx, yy = np.meshgrid(np.linspace(.15, .22, 12), np.linspace(.55, .62, 12))
    hand = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, .88)))
    result = fit(np.vstack((points, hand)))
    assert result['rim_height_m'] == pytest.approx(.60, abs=.001)
    assert result['model_agreement_fraction'] < 1.


@pytest.mark.parametrize('change', ['missing_wall', 'wrong_size', 'tilted', 'solid_clutter'])
def test_missing_or_contradictory_plane_evidence_is_rejected(change):
    points = frozen_points().astype(float)
    if change == 'missing_wall':
        points = points[points[:, 0] > .08]
    elif change == 'wrong_size':
        points[:, 0] *= 1.3
    elif change == 'tilted':
        points = Rotation.from_euler('x', 12, degrees=True).apply(points)
    else:
        rng = np.random.default_rng(1)
        points = np.vstack((points, rng.uniform([.08, .43, .43], [.32, .77, .58], (len(points), 3))))
    with pytest.raises(ValueError, match='demo_bin_shell'):
        fit(points)


def test_measured_complete_bin_preserves_same_hollow_bounds():
    points = frozen_points()
    yy, zz = np.meshgrid(np.linspace(.414, .786, 70), np.linspace(.414, .60, 40))
    near = np.column_stack((np.full(yy.size, .062), yy.ravel(), zz.ravel()))
    result = fit(np.vstack((points, near)))
    assert result['rim_height_m'] == pytest.approx(.6, abs=.001)
    assert not any(contains(model, [.2, .6, .5]) for model in result['cuboids'].values())


def test_builder_fits_only_post_robot_filter_points_and_preserves_native_measurements():
    from dataclasses import replace

    from test_demo_mapping import table_evidence
    from test_semantic_builder import builder, request

    value, segmenter, mapper, _ = builder()
    clean = frozen_points()
    def filtered(views):
        return tuple(replace(view, instance_points_base_m={
            'source': clean if view.camera_id == '2' else np.empty((0, 3)),
            'other': clean + [0, -1.2, 0] if view.camera_id == '1' else np.empty((0, 3))}) for view in views)
    segmenter.filter_isolated_surfaces = filtered
    priors = {name: {'kind': 'bin', 'color': 'blue', 'instance_id': name,
        'observation_id': 'capture-42', 'generation': 4, 'outer_dimensions_m': [.30, .40, .20],
        'source': 'user_specified_bin_outer_dimensions'} for name in ('source', 'other')}
    result = value.build(request('observation') | {'demo_mapping': True, 'scene_policy': 'demo_inventory',
        'demo_tables': table_evidence(), 'demo_bin_priors': priors})
    shell = {k: v for k, v in result.scene_model['cuboid'].items() if '::bin_shell::' in k}
    assert len(shell) == 10
    np.testing.assert_array_equal(mapper.instances['source'][0]['2'], clean)
    assert value.diagnostics['bin_shell_completions']['source']['measurement_source'] == (
        'registered_inventory_points_after_robot_and_surface_filter')
    priors['source']['observation_id'] = 'stale'
    with pytest.raises(ValueError, match='demo_bin_shell_prior'):
        value.build(request('observation') | {'demo_mapping': True, 'demo_tables': table_evidence(),
                                            'demo_bin_priors': priors})


def test_transport_bounds_union_includes_enabled_exact_id_shell_and_binds_capture():
    import json

    from luna_mujoco.collision_mapping import collision_layer_bounds
    from luna_mujoco.demo_bin_geometry import fit_bin_shell
    from luna_mujoco.demo_unpack_workflow import right_bin_occupied_bounds

    fixture = json.loads((Path(__file__).parent/'fixtures/unpack_bin_occupied_bounds.json').read_text())
    scene, destination = fixture['scene'], fixture['destination']
    identifier = destination['id']
    completion = fit_bin_shell(frozen_points()+[0, -1.2, .05], instance_id=identifier,
                               observation_id=scene['observation_id'], generation=scene['generation'])
    scene['mapping_diagnostics'] = {'bin_shell_completions': {identifier: completion}}
    model = {'cuboid': {f'semantic_instance::{identifier}::bin_shell::{face}': value
                       for face, value in completion['cuboids'].items()}}
    layers = collision_layer_bounds([], model, scene)['layers']
    assert all(layer.get('geometry_source') == 'measured_planes_and_user_bin_dimensions'
               for layer in layers.values())
    scene['collision_layer_bounds']['layers'].update(layers)
    result = right_bin_occupied_bounds(destination, scene, fixture['observation'])
    assert result['maximum_world_m'][2] == pytest.approx(.65, abs=.001)
    assert result['bounds_source'] == 'exact_inventory_native_cells_and_derived_shell_union'
    assert len(result['layer_names']) == 6
    first = next(iter(layers))
    scene['collision_layer_bounds']['layers'][first]['generation'] += 1
    with pytest.raises(ValueError, match='bin_shell'):
        right_bin_occupied_bounds(destination, scene, fixture['observation'])


def test_shell_preview_is_explicit_cuboids_not_measured_voxels():
    import struct
    from types import SimpleNamespace

    from luna_mujoco.collision_mapping import collision_scene_preview

    completion = fit(frozen_points())
    identity = {'scene_token': 'scene', 'observation_id': 'capture', 'generation': 2,
                'mapping_diagnostics': {'bin_shell_completions': {'source': completion}}}
    model = {'cuboid': {f'semantic_instance::source::bin_shell::{face}': value
                        for face, value in completion['cuboids'].items()}}
    candidate = SimpleNamespace(integration=SimpleNamespace(pointcloud_data=struct.pack('<I', 0)),
        table=SimpleNamespace(model={'pose': [0, 0, 0, 1, 0, 0, 0], 'dims': [1, 1, .1]},
                              height_m=0., normal_xyz=[0, 0, 1], inlier_count=10), camera_metrics={})
    result = collision_scene_preview(candidate, [], model, identity)
    assert not result['voxel_layers']
    assert len(result['cuboid_layers']) == 5
    assert all(layer['provenance'] == 'measured_planes_and_user_bin_dimensions'
               and layer['instance_id'] == 'source' for layer in result['cuboid_layers'])


def test_source_lift_includes_shell_top_beyond_native_bounds():
    from test_demo_lift_clearance import derive, frozen

    from luna_mujoco.collision_mapping import collision_layer_bounds
    from luna_mujoco.demo_bin_geometry import fit_bin_shell

    value = frozen()
    scene, snapshot = value['scene'], value['snapshot']
    identifier = snapshot['scene_entities'][value['target_id']]['initial_parent_id']
    completion = fit_bin_shell(frozen_points()+[0, 0, .06], instance_id=identifier,
                               observation_id=scene['observation_id'], generation=scene['generation'])
    scene['mapping_diagnostics'] = {'bin_shell_completions': {identifier: completion}}
    model = {'cuboid': {f'semantic_instance::{identifier}::bin_shell::{face}': model
                        for face, model in completion['cuboids'].items()}}
    scene['collision_layer_bounds']['layers'].update(collision_layer_bounds([], model, scene)['layers'])
    _, proof = derive(value)
    assert proof['source_occupied_top_world_z_m'] == pytest.approx(.66, abs=.001)
    assert len([layer for layer in proof['retained_source_layers'] if '::bin_shell::' in layer['name']]) == 5
    scene['collision_layer_bounds']['layers'].pop(next(iter(model['cuboid'])))
    with pytest.raises(ValueError, match='demo_bin_shell'):
        derive(value)


def test_transport_encodes_only_current_fixed_blue_bin_prior_and_rejects_stale_prior():
    from test_demo_mapping import table_evidence
    from test_rgbd_algorithms import observed_scene

    from luna_mujoco.algorithms import Algorithms
    from luna_mujoco.perception_mapping import decode_mapping_payload

    snapshot = observed_scene()
    snapshot.update(demo_mapping=True, demo_tables=table_evidence(), demo_task={'source_bin_id': 'seen-red'})
    snapshot['objects'][0].update(kind='bin', color='blue')
    sent = []
    Algorithms(curobo_request=lambda path, body: (sent.append(body) or {'scene_token': 'map'})).build_scene(
        snapshot, 'seen-red')
    decoded, _ = decode_mapping_payload(sent[0])
    assert set(decoded['demo_bin_priors']) == {'seen-red'}
    decoded_prior = decoded['demo_bin_priors']['seen-red']
    from luna_mujoco.cad_assets import cad_asset
    assert decoded_prior['outer_dimensions_m'] == cad_asset('blue_box')['size_m']
    assert decoded_prior['source'] == 'user_supplied_cad_shape'
    assert decoded_prior['observation_id'] == snapshot['observation_id']
    sent[0]['demo_bin_priors']['seen-red']['generation'] += 1
    with pytest.raises(ValueError, match='demo_bin_shell_prior'):
        decode_mapping_payload(sent[0])


@pytest.mark.parametrize('damage', ['disabled', 'missing_face', 'wrong_id', 'wrong_source', 'unlabeled'])
def test_shell_bounds_never_silently_ignore_incomplete_or_untrusted_completion(damage):
    from luna_mujoco.collision_mapping import collision_layer_bounds
    from luna_mujoco.demo_bin_geometry import published_bin_shell_layers

    completion = fit(frozen_points())
    scene = {'scene_token': 'map', 'observation_id': 'capture', 'generation': 2,
             'mapping_diagnostics': {'bin_shell_completions': {'source': completion}}}
    model = {'cuboid': {f'semantic_instance::source::bin_shell::{face}': value
                       for face, value in completion['cuboids'].items()}}
    scene['collision_layer_bounds'] = collision_layer_bounds([], model, scene)
    first = next(iter(model['cuboid']))
    layer = scene['collision_layer_bounds']['layers'][first]
    if damage == 'disabled':
        layer['enabled'] = False
    elif damage == 'missing_face':
        scene['collision_layer_bounds']['layers'].pop(first)
    elif damage == 'wrong_id':
        layer['instance_id'] = 'other'
    elif damage == 'wrong_source':
        layer['geometry_source'] = 'rgbd_measured'
    else:
        scene['mapping_diagnostics'] = {}
    with pytest.raises(ValueError, match='demo_bin_shell'):
        published_bin_shell_layers(scene, 'source')
