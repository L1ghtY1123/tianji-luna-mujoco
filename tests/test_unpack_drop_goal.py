import copy
import json
from pathlib import Path

import pytest

from luna_mujoco.robot_skills.container_geometry import container_transport_region


@pytest.fixture
def frozen():
    value = json.loads((Path(__file__).parent / 'fixtures/unpack_bin_occupied_bounds.json').read_text())
    value['holding']['arm'] = 'B'  # This frozen test exercises the recorded right-arm transport.
    return value


def test_free_container_placement_uses_measured_floor_hull_not_only_its_rectangle(frozen):
    reference = frozen['destination']['placement_reference']
    measured_hull = [[.4, -.54], [.4, -.63], [.6, -.73], [.61, -.54]]
    reference['observed_floor_hull_world_xy_m'] = measured_hull
    before = copy.deepcopy(frozen)
    proof = container_transport_region(frozen['destination'], frozen['scene'], frozen['observation'])
    assert proof['polygon_world_xy_m'] == measured_hull
    assert proof['placement_extent'] == 'observed_floor_hull'
    assert frozen == before


@pytest.mark.parametrize('location,key,value', [
    ('header', 'scene_token', 'stale'), ('header', 'observation_id', 'stale'),
    ('header', 'generation', 99), ('observation', 'observation_id', 'stale'),
    ('observation', 'generation', 99), ('scene', 'scene_token', None),
    ('scene', 'generation', None), ('region', 'observation_id', 'stale'),
])
def test_bin_bounds_reject_stale_or_missing_binding(frozen, location, key, value):
    locations = {'header': frozen['scene']['collision_layer_bounds'],
        'scene': frozen['scene'], 'observation': frozen['observation'],
        'region': frozen['destination']['placement_reference']}
    locations[location][key] = value
    code = 'container_placement_region_unavailable' if location == 'region' else 'drop_requires_current_bin_occupancy'
    with pytest.raises(ValueError, match=code):
        container_transport_region(frozen['destination'], frozen['scene'], frozen['observation'])


def test_missing_region_is_distinguished_from_valid_container_collision_occupancy(frozen):
    from luna_mujoco.robot_skills.container_geometry import container_transport_region
    from luna_mujoco.runtime.diagnostics import DiagnosticError

    frozen['destination'].pop('placement_reference')
    with pytest.raises(DiagnosticError, match='container_placement_region_unavailable') as caught:
        container_transport_region(frozen['destination'], frozen['scene'], frozen['observation'])
    details = caught.value.details
    assert details['component'] == 'placement_geometry'
    assert details['observation_id'] == frozen['observation']['observation_id']
    assert details['container_collision_occupancy_valid'] is True
    assert details['diagnostic_checks'][0]['actual'] is None
    assert details['diagnostic_checks'][0]['status'] == 'unknown'
    assert 'full_box_bottom_required' in details and details['full_box_bottom_required'] is False


@pytest.mark.parametrize('field,value', [
    ('enabled', False), ('kind', 'cuboid'), ('source', 'published_cuboid_bounds'),
    ('bounds_source', 'preview_cells'), ('bounds_source', None),
    ('cell_count', 0), ('cell_count', True), ('voxel_size_m', float('nan')),
    ('minimum_world_m', [0, 0]), ('maximum_world_m', [0, 0, float('inf')]),
    ('maximum_world_m', [0, 0, 0]),
])
def test_bin_bounds_reject_partial_disabled_or_invalid_native_layer(frozen, field, value):
    name = 'semantic_instance::' + frozen['destination']['id']
    frozen['scene']['collision_layer_bounds']['layers'][name][field] = value
    with pytest.raises(ValueError, match='drop_requires_complete_enabled_bin_occupancy'):
        container_transport_region(frozen['destination'], frozen['scene'], frozen['observation'])


@pytest.mark.parametrize('missing', ['header', 'exact_layer', 'disabled_names'])
def test_bounds_never_fall_back_to_raw_bin_or_another_layer(frozen, missing):
    scene = frozen['scene']
    name = 'semantic_instance::' + frozen['destination']['id']
    if missing == 'header':
        scene.pop('collision_layer_bounds')
    elif missing == 'exact_layer':
        scene['collision_layer_bounds']['layers'].pop(name)
    else:
        scene['disabled_obstacle_names'] = [name]
    with pytest.raises(ValueError, match='drop_requires_'):
        container_transport_region(frozen['destination'], scene, frozen['observation'])


