"""Post-delivery uncertainty cannot erase independently measured inventory."""
import pytest
from test_demo_completed_tracking import resolve

from luna_mujoco.visual_api import VisualInventoryError


@pytest.mark.parametrize('answer,confidence', [('unknown', .5), ('no', .99), ('yes', .79)])
def test_frozen_delivered_identity_can_remain_missing_in_a_partial_observation(answer, confidence):
    def freeze(capture, detections, entities, reference, progress):
        reference['inventory_state'] = {'inventory_finalized': True}
    result, calls, (_, _, _, _, progress) = resolve(
        {'tracked_0': (answer, confidence), 'identity_1': ('remaining', .95)}, mutate=freeze)
    assert len(calls) == 2
    assert result['missing_geometry_ids'] == ['pink']
    assert result['unobserved_completed_ids'] == ['pink']
    assert result['identity_coverage']['complete'] is False
    assert result['identity_coverage']['accepted_entity_ids'] == ['remaining']
    assert {group[0]['entity_id'] for group in result['groups']} == {'remaining'}
    assert result['completed_tracking']['candidate_mapping'] == {}
    assert result['completed_tracking']['response']['answers'][0]['choice'] == answer
    assert progress['pink']['completed_identity_track']['capture']['observation_id'] == 'post-place'


def test_an_intermediate_placement_still_needs_its_current_correspondence():
    def intermediate(capture, detections, entities, reference, progress):
        reference['inventory_state'] = {'inventory_finalized': True}
        progress['pink']['status'] = 'observed'
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown'):
        resolve({'tracked_0': ('unknown', .5), 'identity_1': ('remaining', .95)}, mutate=intermediate)
