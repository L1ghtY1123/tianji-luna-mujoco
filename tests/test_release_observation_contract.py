"""Placement uncertainty belongs to the two actual required visual facts."""

import pytest
from test_demo_release_result import decisions, release_inputs

from luna_mujoco.demo_transfer_verification import release_result_payload, verify_release_result


@pytest.mark.parametrize('kind', ['table', 'container', 'object_surface', 'relative', 'language'])
def test_release_verification_never_calls_segmentation_registration_or_scene_analysis(monkeypatch, kind):
    from unittest.mock import Mock

    from luna_mujoco.demo_perception import DemoPerception
    engine = object.__new__(DemoPerception)
    engine.visual_inventory = Mock()
    engine.observe = Mock(side_effect=AssertionError('verify_place must not remeasure geometry'))
    zone = {'destination_id': 'box', 'support_entity_id': 'support',
            'intent': {'relation': 'on', 'reference_ids': []}, 'semantic_scope': 'entire_region'}
    if kind == 'container': zone['intent']['relation'] = 'inside'
    if kind == 'object_surface': zone['geometry_source'] = 'measured_object_surface'
    if kind == 'relative': zone['intent']['reference_ids'] = ['reference']
    if kind == 'language': zone['semantic_scope'] = 'candidate_selection'
    verifier = Mock(return_value={'visual_evidence': {'status': 'at_destination'}})
    monkeypatch.setattr('luna_mujoco.demo_transfer_verification.verify_release_result', verifier)
    reference = {'entities': {name: {'id': name} for name in ('target', 'box', 'support', 'reference')}}
    capture, binding = {}, {'target_id': 'target', 'placement_zone': zone, 'release_feedback_id': 'release'}
    result = engine.verify_release(capture, binding, reference)
    engine.observe.assert_not_called()
    verifier.assert_called_once_with(engine.visual_inventory, capture, binding, reference, '')
    assert result['visual_evidence']['status'] == 'at_destination'


def test_release_has_only_the_two_actual_completion_criteria():
    payload=release_result_payload(*release_inputs())
    assert [q['name'] for q in payload['questions']]==['place_result','workpiece_still_held']


@pytest.mark.parametrize('on_support,held,status,missing', [
    (.91,0.,'at_destination',[]),(.5,0.,'unknown',['place_result']),
    (.91,.5,'unknown',['workpiece_still_held']),(.1,0.,'rejected',[]),(.91,.99,'rejected',[])])
def test_each_visual_fact_keeps_its_uncertainty_and_rejection(on_support,held,status,missing):
    rows=[{'type':'predicate','name':name,'probability':p}
          for name,p in [('place_result',on_support),('workpiece_still_held',held)]]
    result=verify_release_result(decisions(rows,[]),*release_inputs())
    assert result['visual_evidence']['status']==status
    assert result['observation_assessment']['missing_evidence']==missing
    assert result['observation_assessment']['needs_observation']==bool(missing)


