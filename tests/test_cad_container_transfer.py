from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.cad_perception import CadCatalog
from luna_mujoco.collision_mapping import collision_layer_bounds
from luna_mujoco.robot_skills.container_geometry import container_occupied_bounds, container_transport_region
from luna_mujoco.runtime.controller import PolicyController


@pytest.fixture(params=['registered_cad', 'observed_obb'])
def published_container(request):
    identity = {'observation_id': 'rgbd-current', 'generation': 1, 'scene_token': 'hybrid-scene',
        'disabled_obstacle_names': [], 'inventory_map': {'entities': {'bin': {
            'status': 'observed', 'last_observation_id': 'rgbd-current', 'last_generation': 1,
            'layer_names': ['semantic_instance::bin']}}}}
    geometry = {'geometry_source': request.param, 'instance_id': 'bin',
        'observation_id': identity['observation_id'], 'generation': identity['generation'],
        'pose': [.5, -.4, .65, 1., 0., 0., 0.]}
    if request.param == 'registered_cad':
        mesh = CadCatalog.from_file().geometry('blue_box')
        geometry.update({key: mesh[key] for key in ('asset_id', 'source_sha256', 'mesh_sha256')})
        geometry['file_path'] = str(mesh['path'])
        model = {'mesh': {'semantic_instance::bin': geometry}}
    else:
        geometry.update(dims=[.41, .31, .145], geometry_observation_ids=['rgbd-before', 'rgbd-current'])
        model = {'cuboid': {'semantic_instance::bin': geometry}}
    scene = {**identity, 'voxel_size_m': None,
        'collision_layer_bounds': collision_layer_bounds([], model, identity)}
    destination = {'id': 'bin', 'kind': 'bin', 'placement_reference': {
        'source': 'measured_visible_floor_patch', 'observation_id': identity['observation_id'],
        'generation': identity['generation'], 'floor_z_m': .59,
        'region_polygon_world_xy_m': [[.35, -.5], [.65, -.5], [.65, -.3], [.35, -.3]]}}
    observation = {key: identity[key] for key in ('observation_id', 'generation')}
    observation['objects'] = [destination]
    return destination, scene, observation


def test_published_model_bounds_supply_container_region_without_voxel_metadata(published_container):
    destination, scene, observation = published_container
    before = deepcopy(published_container)
    result = container_transport_region(destination, scene, observation)
    layer = scene['collision_layer_bounds']['layers']['semantic_instance::bin']
    assert result['occupied_top_z_m'] == layer['maximum_world_m'][2]
    assert result['occupancy']['geometry_source'] == layer['geometry_source']
    assert result['occupancy']['bounds_source'] == layer['source']
    assert result['occupancy']['geometry_observation_id'] == observation['observation_id']
    assert 'cell_count' not in result['occupancy'] and 'voxel_size_m' not in result['occupancy']
    assert published_container == before


def test_registered_cad_region_preserves_model_provenance(published_container):
    destination, scene, observation = published_container
    destination['placement_reference'].update(source='validated_cad_support_plane',
        region_measurement={'surface_source': 'registered_model_prior', 'unseen_area_observed': False,
                            'collision_checked': False})
    result = container_transport_region(destination, scene, observation)
    assert result['source'] == 'validated_cad_support_plane'
    assert result['placement_extent'] == 'registered_cad_support_facet'
    assert result['support_provenance']['unseen_area_observed'] is False


async def test_normal_generate_grasps_accepts_published_model_bin_destination(published_container, tmp_path):
    destination, scene, observation = published_container
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.state.update(target_id='workpiece', destination_id=destination['id'], arm='B')
    runner.state['facts']['views_ready'] = True
    runner.require_grasp_view = Mock()
    runner.emit = Mock()
    runner.algorithms = Mock()
    runner.call_algorithm = AsyncMock(return_value={'source': 'test_grasp_algorithm'})
    context = {'observation': observation, 'scene': scene, 'planning_settings': TEST_SETTINGS.model_dump()}
    result = await runner.execute('generate_grasps', context)
    assert result == {'source': 'test_grasp_algorithm'}
    runner.call_algorithm.assert_awaited_once()
    call = runner.call_algorithm.call_args
    assert call.args == (runner.algorithms.generate_grasps, observation, 'workpiece')
    assert call.kwargs['placement_request']['transport_region']['occupancy']['layer_name'] == 'semantic_instance::bin'
    assert context['grasps_observation_id'] == observation['observation_id']
    assert 'plan_transport' not in runner.registry.describe()


@pytest.mark.parametrize('damage', ['instance', 'generation', 'observation', 'missing_observation', 'source', 'disabled',
    'disabled_name', 'stale_record', 'record_generation', 'record_layer', 'nonfinite'])
def test_published_model_bounds_reject_changed_binding(published_container, damage):
    destination, scene, observation = published_container
    layer = scene['collision_layer_bounds']['layers']['semantic_instance::bin']
    record = scene['inventory_map']['entities']['bin']
    if damage == 'instance':
        layer['instance_id'] = 'other'
    elif damage == 'generation':
        layer['generation'] = 2
    elif damage == 'observation':
        layer['observation_id'] = 'unbound-capture'
    elif damage == 'missing_observation':
        layer['observation_id'] = record['last_observation_id'] = None
    elif damage == 'source':
        layer['source'] = 'simulation_bounds'
    elif damage == 'disabled':
        layer['enabled'] = False
    elif damage == 'disabled_name':
        scene['disabled_obstacle_names'].append('semantic_instance::bin')
    elif damage == 'stale_record':
        record['status'] = 'stale'
    elif damage == 'record_generation':
        record['last_generation'] = 2
    elif damage == 'record_layer':
        record['layer_names'] = ['semantic_instance::other']
    else:
        layer['maximum_world_m'][2] = float('nan')
    with pytest.raises(ValueError, match='drop_requires_complete_enabled_bin_occupancy'):
        container_occupied_bounds(destination, scene, observation)


def test_retained_model_bound_keeps_original_geometry_observation(published_container):
    destination, scene, observation = published_container
    layer = scene['collision_layer_bounds']['layers']['semantic_instance::bin']
    layer['observation_id'] = 'rgbd-before'
    scene['inventory_map']['entities']['bin'].update(status='retained', last_observation_id='rgbd-before')
    result = container_occupied_bounds(destination, scene, observation)
    assert result['observation_id'] == 'rgbd-current'
    assert result['geometry_observation_id'] == 'rgbd-before'


@pytest.mark.parametrize('published_container', ['registered_cad'], indirect=True)
@pytest.mark.parametrize('field,value', [('asset_id', ''), ('source_sha256', 'bad'), ('mesh_sha256', None)])
def test_cad_bounds_keep_model_identity_evidence(published_container, field, value):
    destination, scene, observation = published_container
    layer = scene['collision_layer_bounds']['layers']['semantic_instance::bin']
    layer[field] = value
    with pytest.raises(ValueError, match='drop_requires_complete_enabled_bin_occupancy'):
        container_occupied_bounds(destination, scene, observation)
