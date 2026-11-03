import json
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
from test_visual_inventory import client_with_answers, fixture

from luna_mujoco.demo_perception import DemoPerception

from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory
from luna_mujoco.perception import Perception, PerceptionError
from luna_mujoco.visual_inventory import VisualInventoryError


@pytest.mark.parametrize('closure', ['none', 'unknown', 'unlisted'])
def test_generic_closure_covers_nonpink_subjects_in_all_observed_regions(closure):
    capture, detection = fixture()
    calls = []
    inventory = DemoVisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'red', 'visibility_red': 'visible', 'scene_inventory_closure': closure}, calls))
    reference = {'task_spec': {'goal': 'Move the red cube to the shelf'},
                 'task_inventory': {'source_object_ids': ['red']}}
    entities = {'red': {'kind': 'cube', 'color': 'red', 'visual_membership': 'required'}}
    if closure == 'unlisted':
        with pytest.raises(VisualInventoryError, match='scene_unlisted_object') as failure:
            inventory.resolve(capture, [detection], reference=reference, entities=entities, instruction='move red')
        assert failure.value.evidence['demo_inventory_check']['status'] == 'unlisted'
    else:
        result = inventory.resolve(capture, [detection], reference=reference, entities=entities, instruction='move red')
        assert result['demo_inventory_check']['status'] == closure
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    assert 'robot-left box containing the pink' not in context['geometric_location_rule']
    assert 'large table and both bins' not in context['rules']
    question = calls[0]['questions'][-1]
    assert question['name'] == 'scene_inventory_closure'
    assert 'all observed regions' in question['instructions']
    assert 'pink' not in question['instructions']
    assert 'task_spec.target_selector' in question['instructions']
    assert 'Unlisted non-task objects are intentionally filtered' in question['instructions']
    assert 'absent from source_object_ids' in question['instructions']
    assert 'For ids scope, only the explicitly requested subset' in question['instructions']


def generic_observation():
    view = {'depth_m': np.ones((3, 3)), 'valid_depth': np.ones((3, 3), bool), 'instance_masks': {}}
    reference = {'task_spec': {'goal': 'Move red cube'}, 'task_inventory': {'source_object_ids': ['red']},
                 'policy_graph': {}, 'entities': {'red': {'color': 'red'}}, 'observation_id': 'initial'}
    return {'visual_reference': reference, 'visual_inventory': {}, 'scene_entities': reference['entities'],
            'objects': [{'id': 'red', 'color': 'red', 'points_world_m': np.array([[.2, .1, .7]])}],
            'views': {'1': view}}


def test_generic_source_has_no_undeclared_cad_prior_and_maps_only_inventory(monkeypatch):
    observation = generic_observation()
    monkeypatch.setattr(Perception, 'observe', lambda *args, **kwargs: deepcopy(observation))
    monkeypatch.setattr('luna_mujoco.demo_perception.reference_with_container_tracks',
                        lambda observation, capture: observation['visual_reference'])
    engine = object.__new__(DemoPerception)
    engine.demo_tables = None
    result = engine.observe({'observation_id': 'next'}, source_hint={
        'scene_entities': observation['scene_entities'], 'visual_reference': observation['visual_reference']})
    assert 'dimension_prior' not in result['objects'][0]
    assert 'geometry_completion' not in result['objects'][0]
    assert not result['views']['1']['valid_depth'].any()
    assert np.all(result['views']['1']['depth_m'] == 0.)
    assert result['views']['1']['coverage_valid_depth'].all()
    assert np.all(result['views']['1']['coverage_depth_m'] == 1.)


def test_generic_observation_requires_controller_inventory_before_segmentation(monkeypatch):
    observe = Mock()
    monkeypatch.setattr(Perception, 'observe', observe)
    engine = object.__new__(DemoPerception)
    with pytest.raises(PerceptionError, match='controller_scene_inventory_required'):
        engine.observe({}, source_hint={'visual_reference': {'task_spec': {'goal': 'inspect'}}})
    observe.assert_not_called()


def test_generic_identity_coverage_does_not_require_the_left_wrist():
    capture, detection = fixture()
    capture['views']['2'] = deepcopy(capture['views']['3'])
    unknown = {**detection, 'camera_id': '2', 'view': capture['views']['2']}
    calls = []
    inventory = DemoVisualInventory(api_key='test', client=client_with_answers({
        'identity_0': 'red', 'identity_1': 'unknown', 'visibility_red': 'visible',
        'scene_inventory_closure': 'none'}, calls))
    result = inventory.resolve(capture, [detection, unknown], instruction='move red',
        purpose='associate_inventory',
        reference={'task_spec': {'goal': 'Move red'}, 'task_inventory': {'source_object_ids': ['red']}},
        entities={'red': {'kind': 'cube', 'color': 'red', 'visual_membership': 'required'}})
    assert result['identity_coverage']['accepted_entity_ids'] == ['red']
    assert result['unassociated_candidates'][0]['camera_id'] == '2'


def test_generic_primary_table_uses_a_confirmed_measured_support(monkeypatch):
    capture, _ = fixture()
    capture['views']['3']['valid_depth'][:] = True
    engine = object.__new__(DemoPerception)
    engine.demo_tables = None
    engine.robot_surface_classifier = lambda *args: {'status': 'measured', 'masks': {}}
    monkeypatch.setattr('luna_mujoco.demo_perception.deproject_view', lambda *args: np.zeros((3, 3)))
    monkeypatch.setattr('luna_mujoco.demo_perception.fit_inventory_table', lambda *args, **kwargs: {
        'surface_height_m': .8, 'model': {'pose': [.5, .1, .8], 'dims': [1., .6, .01]},
        'source': 'measured_rgbd'})
    result = engine._table_for_entities(capture, {}, {
        'shelf': {'kind': 'table', 'reference_camera_id': '3', 'bbox': [0., 0., 1., 1.]}},
        {'observation_id': capture['observation_id'], 'task_spec': {'goal': 'Move red'},
         'task_inventory': {'destination_id': 'tray'}}, [])
    assert result['surface_height_m'] == .8
    assert engine.demo_tables['primary_table_id'] == 'shelf'
