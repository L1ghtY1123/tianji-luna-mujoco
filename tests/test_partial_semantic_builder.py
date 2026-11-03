"""Exercise partial inventory requests through decoding, the builder and retained layers."""
from copy import deepcopy

import numpy as np
import pytest
from test_inventory_map_lifecycle import Grid
from test_rgbd_algorithms import observed_scene
from test_semantic_builder import builder

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.inventory_map import inventory_admission, reconcile_inventory
from luna_mujoco.perception_mapping import decode_mapping_payload


def captured_inventory(*, visible_obstacle=True, scope='global'):
    capture = deepcopy(observed_scene())
    target = capture['objects'][0]
    obstacle = deepcopy(target) | {'id': 'obstacle', 'color': 'blue', 'observation_id': 'initial'}
    target['observation_id'] = 'initial'
    capture['scene_entities'] = {'seen-red': deepcopy(target), 'obstacle': obstacle}
    capture['task_member_ids'] = ['seen-red']
    capture['declared_environment'] = [{'color': 'blue', 'observed_ids': ['obstacle'], 'status': 'observed'}]
    capture['mapping_lifecycle'] = {'requested_scope': scope, 'strategy_source': 'responses',
        'scope_reason': 'Graph-selected measured update', 'mapping_entity_ids': ['seen-red'] if scope == 'local' else []}
    for view in capture['views'].values():
        target_mask = np.zeros((8, 8), bool)
        target_mask[:2, :2] = True
        obstacle_mask = np.zeros((8, 8), bool)
        obstacle_mask[2:4, 2:4] = True
        view['instance_masks'] = {'seen-red': target_mask}
        if visible_obstacle:
            view['instance_masks']['obstacle'] = obstacle_mask
    if visible_obstacle:
        capture['objects'].append(deepcopy(obstacle))
    else:
        capture['mapping_unresolved'] = [{'previous_id': 'obstacle', 'status': 'unseen', 'candidates': [],
                                           'reason': 'fixed_collision_entity_not_observed'}]
    return capture


class MappingBoundary:
    def __init__(self):
        self.builder, self.segmenter, self.mapper, self.fits = builder()
        self.previous = None
        self.calls = []

    def __call__(self, path, body):
        assert path == '/perception/build'
        self.calls.append(body)
        decoded, identity = decode_mapping_payload(body)
        candidate = self.builder.build(decoded)
        grids = [Grid(name, np.array([1.])) for name in candidate.scene_model.get('voxel', {})]
        model = {name: dict(value) for name, value in candidate.scene_model.items()}
        self.previous = reconcile_inventory(self.previous, identity, grids, model,
                                            **decoded['mapping_lifecycle'])
        return self.previous[0]


@pytest.mark.parametrize('scope', ['global', 'local'])
@pytest.mark.parametrize('stale', [False, True])
def test_partial_model_map_retains_declared_obstacle_and_strict_motion_admission(scope, stale):
    boundary = MappingBoundary()
    algorithms = Algorithms(curobo_request=boundary)
    initial = algorithms.build_scene(captured_inventory(), None, exclusion_state='observation')
    assert len(boundary.fits) == 1
    assert len(boundary.fits[0][0]) == 1
    capture = captured_inventory(visible_obstacle=False, scope=scope)
    capture['observation_id'] = 'partial'
    if stale:
        capture['mapping_lifecycle']['invalidated_entity_ids'] = ['obstacle']
    result = algorithms.build_scene(capture, None, exclusion_state='observation')
    record = result['inventory_map']['entities']['obstacle']
    assert record['last_observation_id'] == initial['observation_id']
    assert record['status'] == ('stale' if stale else 'retained')
    assert 'semantic_instance::obstacle' in boundary.previous[2]['voxel']
    assert len(boundary.fits) == 1
    assert [item.camera_id for item in boundary.segmenter.requests] == ['1', '2', '3'] * 2
    assert not result['coverage']['full_environment_coverage']
    if stale:
        assert inventory_admission(result)['reason'] == 'inventory_geometry_stale'
    else:
        assert inventory_admission(result)['accepted']
        assert inventory_admission(result, swept_workspace_bounds={
            'minimum_world_m': [0, 0, 0], 'maximum_world_m': [1, 1, 1]})['reason'] == 'motion_coverage_required'


def test_historical_identity_without_prior_map_cannot_certify_required_geometry():
    boundary = MappingBoundary()
    capture = captured_inventory(visible_obstacle=False, scope='local')
    result = Algorithms(curobo_request=boundary).build_scene(capture, None, exclusion_state='observation')
    assert result['coverage']['missing_entity_ids'] == ['obstacle']
    assert inventory_admission(result)['reason'] == 'inventory_geometry_missing'


@pytest.mark.parametrize('missing', ['measurement', 'bound_id', 'only_target', 'ambiguous', 'geometry_rejected', 'conflicting_rows'])
def test_unknown_declared_identity_or_missing_measurement_is_not_historical_evidence(missing):
    capture = captured_inventory(visible_obstacle=False)
    capture['mapping_unresolved'] = []
    if missing == 'measurement':
        capture['scene_entities']['obstacle'].pop('position')
    elif missing == 'bound_id':
        capture['declared_environment'][0]['observed_ids'] = []
    elif missing == 'only_target':
        capture['declared_environment'][0].update(color='red', observed_ids=['seen-red'])
    elif missing == 'ambiguous':
        capture['mapping_unresolved'] = [{'previous_id': 'obstacle', 'status': 'ambiguous'}]
    elif missing == 'geometry_rejected':
        capture['mapping_unresolved'] = [{'previous_id': 'obstacle', 'status': 'geometry_rejected', 'candidates': []}]
    else:
        capture['mapping_unresolved'] = [
            {'previous_id': 'obstacle', 'status': 'unseen'},
            {'previous_id': 'obstacle', 'status': 'ambiguous'}]
    calls = []
    with pytest.raises(AlgorithmError, match='semantic_scene_identity_unresolved|declared_obstacle_not_observed'):
        Algorithms(curobo_request=lambda *args: calls.append(args)).build_scene(capture, None,
                                                                            exclusion_state='observation')
    assert not calls
