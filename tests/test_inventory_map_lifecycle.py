import importlib.util
from dataclasses import dataclass, field

import numpy as np
import pytest


@dataclass
class Grid:
    name: str
    feature_tensor: object
    pose: list = field(default_factory=lambda: [0, 0, 0, 1, 0, 0, 0])
    dims: list = field(default_factory=lambda: [1, 1, 1])
    voxel_size: float = .01


def api():
    assert importlib.util.find_spec('luna_mujoco.inventory_map'), 'inventory map lifecycle is missing'
    from luna_mujoco import inventory_map
    return inventory_map


def scene(capture, ids):
    identity = {'scene_token': capture, 'observation_id': capture, 'generation': 1,
                'target_id': '', 'exclusion_state': 'observation', 'disabled_obstacle_names': ()}
    grids = [Grid('semantic_instance::' + name, np.array([1.])) for name in ids]
    model = {'voxel': {grid.name: {'pose': [0, 0, 0, 1, 0, 0, 0]} for grid in grids},
             'cuboid': {'semantic_table::table': {'dims': [1, 1, .1]}}, 'mesh': {}}
    return identity, grids, model


def test_partial_capture_retains_previously_measured_entity_and_freshness():
    module = api()
    first = module.reconcile_inventory(None, *scene('first', ['a', 'b']), required_entity_ids=['a', 'b'])
    second = module.reconcile_inventory(first, *scene('second', ['a']), required_entity_ids=['a', 'b'])
    identity, grids, _ = second
    assert 'semantic_instance::b' in {grid.name for grid in grids}
    assert identity['inventory_map']['entities']['b']['last_observation_id'] == 'first'
    assert identity['inventory_map']['entities']['b']['status'] == 'retained'
    assert identity['coverage']['inventory_complete']
    assert identity['coverage']['workspace_status'] == 'unknown'
    assert not identity['coverage']['full_environment_coverage']


def test_model_local_scope_changes_only_requested_entities_and_retains_other_evidence():
    module = api()
    first = module.reconcile_inventory(None, *scene('first', ['a', 'b']))
    second = module.reconcile_inventory(first, *scene('second', ['a', 'b']),
        requested_scope='local', mapping_entity_ids=['a'], strategy_source='decisions',
        scope_reason='Only a moved', invalidated_entity_ids=['a'])
    records = second[0]['inventory_map']['entities']
    assert records['a']['last_observation_id'] == 'second'
    assert records['b']['last_observation_id'] == 'first'
    assert records['b']['status'] == 'retained'
    assert second[0]['mapping_strategy']['requested_scope'] == 'local'
    assert not second[0]['coverage']['full_environment_coverage']


def test_model_global_request_does_not_erase_unseen_entities_or_certify_blind_space():
    module = api()
    first = module.reconcile_inventory(None, *scene('first', ['a', 'b']))
    second = module.reconcile_inventory(first, *scene('second', ['a']),
        requested_scope='global', strategy_source='responses', scope_reason='Refresh scene')
    assert set(second[0]['inventory_map']['entities']) == {'a', 'b'}
    assert second[0]['coverage']['workspace_status'] == 'unknown'
    assert not second[0]['coverage']['full_environment_coverage']


@pytest.mark.parametrize('scope,ids,source', [('local', [], 'decisions'),
    ('decide', ['a'], 'decisions'), ('global', [], 'heuristic')])
def test_unresolved_or_non_model_mapping_strategy_cannot_reach_mapper(scope, ids, source):
    with pytest.raises(ValueError, match='mapping_strategy_invalid'):
        api().reconcile_inventory(None, *scene('capture', ['a']), requested_scope=scope,
            mapping_entity_ids=ids, strategy_source=source, scope_reason='test')


def test_same_entity_new_partial_surface_does_not_erase_old_geometry():
    module = api()
    first = module.reconcile_inventory(None, *scene('first', ['a']))
    incoming = scene('second', ['a'])
    incoming[1][0].pose[0] = .1
    second = module.reconcile_inventory(first, *incoming)
    assert len(second[1]) == 2
    assert len({grid.name for grid in second[1]}) == 2


def test_invalidated_missing_entity_is_stale_and_blocks_motion():
    module = api()
    first = module.reconcile_inventory(None, *scene('first', ['a', 'b']))
    second = module.reconcile_inventory(first, *scene('second', ['a']), invalidated_entity_ids=['b'])
    assert second[0]['inventory_map']['entities']['b']['status'] == 'stale'
    assert module.inventory_admission(second[0])['reason'] == 'inventory_geometry_stale'


def test_reobserved_moved_entity_replaces_old_pose_without_retiring_other_entities():
    module = api()
    first = module.reconcile_inventory(None, *scene('first', ['a', 'b']))
    second = module.reconcile_inventory(first, *scene('second', ['a']), invalidated_entity_ids=['a'])
    assert len(second[1]) == 2
    assert second[0]['inventory_map']['entities']['a']['status'] == 'observed'
    assert second[0]['inventory_map']['entities']['b']['status'] == 'retained'


def test_only_explicit_retirement_removes_an_unseen_entity():
    module = api()
    first = module.reconcile_inventory(None, *scene('first', ['a', 'b']))
    second = module.reconcile_inventory(first, *scene('second', ['a']), retired_entity_ids=['b'])
    assert set(second[0]['inventory_map']['entities']) == {'a'}
    assert [grid.name for grid in second[1]] == ['semantic_instance::a']


def test_missing_required_entity_and_uncertified_swept_workspace_are_rejected():
    module = api()
    current = module.reconcile_inventory(None, *scene('first', ['a']), required_entity_ids=['a', 'b'])
    assert module.inventory_admission(current[0])['reason'] == 'inventory_geometry_missing'
    current = module.reconcile_inventory(None, *scene('first', ['a']))
    bounds = {'minimum_world_m': [0, 0, 0], 'maximum_world_m': [1, 1, 1]}
    assert module.inventory_admission(current[0], swept_workspace_bounds=bounds)['reason'] == 'motion_coverage_required'


def test_archive_copies_mutable_mapper_tensors_before_rebuild():
    module = api()
    current = module.reconcile_inventory(None, *scene('first', ['a']))
    frozen = module.freeze_inventory_scene(current)
    current[1][0].feature_tensor[:] = 9
    assert frozen[1][0].feature_tensor.tolist() == [1.]


def test_twelve_partial_captures_union_compatible_esdf_without_losing_occupied_samples():
    module = api()
    current = None
    for i in range(12):
        incoming = scene(str(i), ['a'])
        incoming[1][0].feature_tensor = np.full(12, 1.)
        incoming[1][0].feature_tensor[i] = -.1
        current = module.reconcile_inventory(current, *incoming)
    assert len(current[1]) == 1
    np.testing.assert_array_equal(current[1][0].feature_tensor, np.full(12, -.1))
    assert current[0]['inventory_map']['entities']['a']['observation_ids'] == [str(i) for i in range(12)]


def test_distinct_lattices_respect_configurable_bytes_without_dropping_old_surfaces():
    module = api()
    first = module.reconcile_inventory(None, *scene('first', ['a']))
    incoming = scene('second', ['a'])
    incoming[1][0].pose[0] = .1
    with pytest.raises(ValueError, match='inventory_voxel_byte_budget_exceeded'):
        module.reconcile_inventory(first, *incoming, max_retained_voxel_bytes=8)
    assert first[1][0].feature_tensor.tolist() == [1.]
    merged = module.reconcile_inventory(first, *incoming, max_retained_voxel_bytes=16)
    assert len(merged[1]) == 2


def test_select_preserves_freshness_but_unmeasured_release_invalidates_it():
    module = api()
    identity, _, model = module.reconcile_inventory(None, *scene('first', ['a']))
    module.apply_inventory_update(identity, {'operation': 'select', 'target_id': 'a'}, model)
    assert module.inventory_admission(identity)['accepted']
    module.apply_inventory_update(identity, {'operation': 'invalidate', 'target_id': 'a',
        'source': 'drop_release_action_bounds', 'action_id': 'release',
        'bounds': {'minimum_world_m': [0, 0, 0], 'maximum_world_m': [1, 1, 1]}}, model)
    assert not module.inventory_admission(identity)['accepted']
    record = identity['inventory_map']['entities']['a']
    assert record['status'] == 'stale'
    assert record['invalidation_reason'] == 'invalidate'
    assert record['last_observation_id'] == 'first'


def test_complete_workspace_requires_configured_region_and_registered_geometry_proof():
    from test_motion_coverage import guard

    module = api()
    workspace = {'minimum_world_m': [-.1, -.1, .8], 'maximum_world_m': [.1, .1, 1.2]}
    identity, _, _ = module.reconcile_inventory(None, *scene('capture-map', ['a']), workspace_bounds=workspace)
    assert identity['coverage']['workspace_status'] == 'partial'
    module.certify_workspace(identity, guard())
    assert identity['coverage']['workspace_status'] == 'complete'
    module.certify_workspace(identity, guard(np.zeros((101, 101))))
    assert identity['coverage']['workspace_status'] == 'partial'
    assert not identity['coverage']['full_environment_coverage']


def test_retained_history_changes_map_version_even_for_identical_latest_capture():
    module = api()
    original = module.reconcile_inventory(None, *scene('latest', ['a']))
    prior = module.reconcile_inventory(None, *scene('prior', ['b']))
    retained = module.reconcile_inventory(prior, *scene('latest', ['a']))
    assert original[0]['scene_token'] != retained[0]['scene_token']


def test_unseen_action_changed_geometry_stays_stale_until_pose_is_remeasured():
    module = api()
    identity, grids, model = module.reconcile_inventory(None, *scene('first', ['a', 'b']))
    module.apply_inventory_update(identity, {'operation': 'invalidate', 'target_id': 'b',
        'source': 'drop_release_action_bounds', 'action_id': 'release',
        'bounds': {'minimum_world_m': [0, 0, 0], 'maximum_world_m': [1, 1, 1]}}, model)
    unseen = module.reconcile_inventory((identity, grids, model), *scene('second', ['a']))
    assert unseen[0]['inventory_map']['entities']['b']['status'] == 'stale'
    remeasured = module.reconcile_inventory(unseen, *scene('third', ['b']))
    assert remeasured[0]['inventory_map']['entities']['b']['pose_replaced_after_invalidation']
