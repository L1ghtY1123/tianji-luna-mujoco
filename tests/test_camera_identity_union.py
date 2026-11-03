"""Camera symmetry and explicit conflicts in Decisions identity evidence."""
from copy import deepcopy

import numpy as np
import pytest
from test_demo_partial_identity import clients, inputs

from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError


def container_inputs():
    capture, detections, entities, reference = inputs(('3', '3', '3', '2', '2', '1'))
    entities.update({identifier: {'id': identifier, 'kind': 'bin', 'color': 'blue', 'roles': ['collision']}
                     for identifier in ('source', 'destination')})
    subjects = ('destination', 'source', 'pink', 'source', 'pink', 'destination')
    for detection, subject in zip(detections, subjects, strict=True):
        mask = np.zeros_like(detection['mask'])
        if subject == 'destination':
            mask[1:20, 20:30] = True
        elif subject == 'source':
            mask[1:20, :18] = True
        else:
            mask[4:14, 10:15] = True
        detection.update(mask=mask.copy(), silhouette_mask=mask.copy(),
                         pixels_rc=np.argwhere(mask), pixels=np.zeros((mask.sum(), 3)),
                         points=np.tile([.5, -.6 if subject == 'destination' else .6, .55], (mask.sum(), 1)),
                         kind=entities[subject]['kind'], prompt='blue container' if subject != 'pink' else 'pink object')
    return capture, detections, entities, reference


@pytest.mark.parametrize('trusted_camera', ['1', '3'])
def test_head_and_wrist_have_equal_identity_authority(trusted_camera):
    capture, detections, entities, reference = container_inputs()
    answers = [('destination', .99 if trusted_camera == '3' else .37), ('source', 1.), ('pink', .98),
               ('source', .75), ('pink', .59), ('destination', .99 if trusted_camera == '1' else .26)]
    result = DemoVisualInventory(api_key='test', client=clients(answers, extra='none')).resolve(
        capture, detections, entities=entities, reference=reference,
        instruction='move the requested object', purpose='associate_inventory')
    groups = {group[0]['entity_id']: group for group in result['groups']}
    assert set(groups) == {'source', 'destination', 'pink'}
    assert [row['camera_id'] for row in groups['destination']] == [trusted_camera]
    expected_index = 0 if trusted_camera == '3' else 5
    np.testing.assert_array_equal(groups['destination'][0]['points'], detections[expected_index]['points'])
    assert len(result['unassociated_candidates']) == 3
    assert result['identity_coverage']['accepted_entity_ids'] == ['destination', 'pink', 'source']


def test_run15_confidences_do_not_establish_destination_identity_in_any_camera():
    capture, detections, entities, reference = container_inputs()
    answers = [('destination', .37), ('source', 1.), ('pink', .98),
               ('source', .75), ('pink', .59), ('source', .26)]
    inventory = DemoVisualInventory(api_key='test', client=clients(answers, extra='none'))
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown') as caught:
        inventory.resolve(capture, detections, entities=entities, reference=reference,
                          instruction='move the requested object', purpose='associate_inventory')
    assert caught.value.evidence['candidate'] == 'candidate-0'
    assert caught.value.evidence['confidence'] < .8
    assert caught.value.evidence['minimum_confidence'] == .8
    assert caught.value.evidence['missing_geometry_ids'] == ['destination']
    initial = caught.value.evidence['initial_identity_failure']['evidence']
    assert initial['confidence'] == .37
    assert initial['answers'][5]['choice'] == 'source'
    assert initial['answers'][5]['confidence'] == .26


def test_same_camera_identical_surface_cannot_confirm_two_distinct_object_ids():
    capture, detections, entities, reference = inputs(('1', '1'))
    entities['other'] = {**deepcopy(entities['pink']), 'id': 'other'}
    reference['task_spec']['target_refs'] = ['pink', 'other']
    inventory = DemoVisualInventory(api_key='test', client=clients([('pink', .99), ('other', .99)], extra='none'))
    with pytest.raises(VisualInventoryError, match='visual_identity_conflict') as caught:
        inventory.resolve(capture, detections, entities=entities, reference=reference,
                          instruction='move both objects', purpose='associate_inventory')
    assert caught.value.evidence['candidate_ids'] == ['candidate-0', 'candidate-1']
    assert caught.value.evidence['camera_id'] == '1'
    assert caught.value.evidence['entity_ids'] == ['pink', 'other']
    assert caught.value.evidence['shared_pixel_count'] == 132
    assert [answer['confidence'] for answer in caught.value.evidence['answers'][:2]] == [.99, .99]


@pytest.mark.parametrize('case', ['same_identity_alias', 'different_camera', 'container_contents'])
def test_conflict_guard_does_not_infer_identity_from_cross_camera_pixels_or_container_overlap(case):
    capture, detections, entities, reference = inputs(('1', '2' if case == 'different_camera' else '1'))
    second_id = 'pink' if case == 'same_identity_alias' else 'other'
    if second_id == 'other':
        entities['other'] = {**deepcopy(entities['pink']), 'id': 'other',
                             'kind': 'bin' if case == 'container_contents' else 'object'}
    result = DemoVisualInventory(api_key='test', client=clients([('pink', .99), (second_id, .99)])).resolve(
        capture, detections, entities=entities, reference=reference,
        instruction='observe requested objects', purpose='associate_inventory')
    assert {group[0]['entity_id'] for group in result['groups']} == {'pink', second_id}
    if case == 'same_identity_alias':
        assert len(result['groups'][0][0]['points']) == 132
    elif case == 'container_contents':
        workpiece = next(group for group in result['groups'] if group[0]['entity_id'] == 'pink')
        assert len(workpiece[0]['points']) == 132
