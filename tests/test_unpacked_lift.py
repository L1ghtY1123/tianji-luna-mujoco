"""An unpacked target's current measured support replaces its historic source bin."""
from copy import deepcopy

import numpy as np
import pytest
from test_demo_lift_clearance import frozen

from luna_mujoco.demo_lift import source_clearance_lift


def table_case():
    value = frozen()
    value['settings']['attachment']['sphere_budget'] = 24
    snapshot = value['snapshot']
    snapshot['stage'] = 'unpacked'
    # The immutable task identity still remembers the original bin. Current
    # support is explicitly measured in this capture, not inferred from that ID.
    snapshot['table'] = {'surface_height_m': .56, 'position': [.36, 0., .557],
                         'size': [.60, .80, .006], 'source': 'cached_ransac_table'}
    target = next(obj for obj in snapshot['objects'] if obj['id'] == value['target_id'])
    target['points_world_m'] = np.array([[.35+x, y, z] for x in (-.02, 0., .02)
                                       for y in (-.015, 0., .015) for z in (.56, .59, .62)])
    target.pop('completed_geometry_points_world_m', None)
    target.pop('geometry_completion', None)
    bounds = value['scene']['collision_layer_bounds']['layers']
    # Keep the published native layer name used by the semantic table builder.
    bounds.clear()
    bounds['semantic_table::table'] = {'enabled': True, 'source': 'published_cuboid_bounds',
        'minimum_world_m': [.06, -.4, .554], 'maximum_world_m': [.66, .4, .56]}
    for candidate in value['candidates']:
        pose = np.diag([1., -1., -1., 1.]); pose[:3, 3] = [.35, 0., .8]
        candidate['T_world_tcp'] = pose.tolist()
    return value


def test_unpacked_lift_uses_current_table_not_original_bin_and_preserves_world():
    value = table_case()
    before = deepcopy(value['scene'])
    settings, evidence = source_clearance_lift(value['snapshot'], value['target_id'], 'B',
        value['candidates'], value['scene'], value['settings'])
    assert evidence['support_kind'] == 'table'
    assert evidence['source_container_id'] is None
    assert evidence['source_occupied_top_world_z_m'] == .56
    assert settings['lift_offset_m'] >= value['settings']['lift_offset_m']
    for candidate in evidence['candidates']:
        assert candidate['lowest_sphere_world_z_m'] + settings['lift_offset_m'] >= .565
        assert candidate['sphere_count'] <= 24
    assert value['scene'] == before


@pytest.mark.parametrize('fault', ['missing_table', 'missing_layer', 'stale', 'outside_table'])
def test_unpacked_lift_does_not_invent_missing_current_support(fault):
    value = table_case()
    if fault == 'missing_table': value['snapshot'].pop('table')
    elif fault == 'missing_layer': value['scene']['collision_layer_bounds']['layers'].pop('semantic_table::table')
    elif fault == 'stale': value['scene']['collision_layer_bounds']['observation_id'] = 'old'
    else:
        target = next(obj for obj in value['snapshot']['objects'] if obj['id'] == value['target_id'])
        target['points_world_m'] += [2., 0., 0.]
    with pytest.raises(ValueError):
        source_clearance_lift(value['snapshot'], value['target_id'], 'B',
            value['candidates'], value['scene'], value['settings'])
