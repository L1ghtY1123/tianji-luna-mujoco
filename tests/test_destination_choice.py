"""Destination selection is evidence, not permission to silently change the graph."""
from copy import deepcopy

import pytest

from luna_mujoco.agent.task_binding import (
    bind_destination_choice,
    destination_candidates,
    destination_question,
    target_membership_question,
)


def candidates():
    return [{'value': 'ID1', 'instance_id': 'left', 'kind': 'bin'},
            {'value': 'ID2', 'instance_id': 'right', 'kind': 'bin'}]


def test_options_use_current_image_labels_and_exclude_configured_markers():
    entities = [{'id': 'right', 'kind': 'bin'}, {'id': 'configured', 'kind': 'marker',
                 'entity_type': 'configured_region'}, {'id': 'left', 'kind': 'bin'}]
    locations = {'views': {'3': {'items': [{'id': 'left', 'image_label': 'ID1'},
                                          {'id': 'right', 'image_label': 'ID2'}]}}}
    assert destination_candidates(entities, locations) == candidates()
    request = destination_question('put it in the left box', 'black object', candidates())
    assert request.selection_mode == 'evidence'
    assert request.choices == ('ID1', 'ID2', 'none_of_listed', 'unknown')


@pytest.mark.parametrize('value, confidence, selected, expected', [
    ('ID1', .92, 'left', 'yes'), ('ID2', .99, 'right', 'no'),
    ('none_of_listed', .99, None, 'no'), ('unknown', .99, None, 'unknown'),
    ('ID1', .79, None, 'unknown')])
def test_only_confident_choice_of_the_proposed_instance_confirms_binding(value, confidence, selected, expected):
    known = value != 'unknown' and confidence >= .8
    judgment = {'status': 'known' if known else 'unknown',
                'value': value if known else 'unknown', 'confidence': confidence}
    result = {'judgments': {'destination_requirement_matches': judgment},
              'evidence': {'answers': [{'choice': value, 'confidence': confidence}], 'request_hash': 'original'}}
    original = deepcopy(result['evidence'])
    bind_destination_choice(result, candidates(), 'left')
    answer = result['judgments']['destination_requirement_matches']
    assert answer['status'] == expected
    assert answer['selected_instance_id'] == selected
    assert answer['proposed_instance_id'] == 'left'
    assert result['evidence']['answers'] == original['answers']
    assert result['evidence']['request_hash'] == original['request_hash']


def test_predicate_style_yes_cannot_impersonate_a_typed_choice():
    result = {'judgments': {'destination_requirement_matches': {'status': 'yes', 'probability': .99}},
              'evidence': {}}
    bind_destination_choice(result, candidates(), 'left')
    assert result['judgments']['destination_requirement_matches']['status'] == 'unknown'


def test_option_building_is_independent_of_proposed_destination_and_keeps_tables():
    rows = destination_candidates([{'id': 'table', 'kind': 'table'}, {'id': 'object', 'kind': 'object'}], {})
    assert {r['instance_id'] for r in rows} == {'table', 'object'}
    assert len({r['value'] for r in rows}) == 2


def test_current_measured_locations_are_attached_to_destination_choices():
    entities = [{'id': 'right', 'kind': 'bin', 'color': 'blue',
        'visible_centroid_world': [.46, -.62, .55],
        'observation_provenance': {'observation_id': 'current'}},
        {'id': 'left', 'kind': 'bin', 'color': 'blue',
         'position': [.46, .54, .55]}]
    rows = destination_candidates(entities, {'observation_id': 'current'})
    right = next(row for row in rows if row['instance_id'] == 'right')
    left = next(row for row in rows if row['instance_id'] == 'left')
    assert right['visible_centroid_world'] == [.46, -.62, .55]
    assert 'visible_centroid_world' not in left
    request = destination_question('put it in the right box', 'object', rows)
    assert '-0.62' in request.instructions
    assert '+Y is robot LEFT' in request.instructions
    assert 'explicitly named viewpoint' in request.instructions
    assert 'position' not in left


def test_historical_position_is_not_presented_as_current_destination_measurement():
    rows = destination_candidates([{'id': 'old', 'kind': 'bin',
        'visible_centroid_world': [.46, -.62, .55],
        'observation_provenance': {'observation_id': 'previous'}}], {'observation_id': 'current'})
    assert 'visible_centroid_world' not in rows[0]


def test_production_question_matches_both_frozen_real_diagnostics_exactly():
    import json
    from pathlib import Path
    cases = json.loads((Path(__file__).parent / 'fixtures/destination_choice_a13_a14.json').read_text('utf-8'))
    for case in cases:
        request = destination_question(case['instruction'], case['target_selector'], case['candidates'])
        wire = {'type': 'choice', 'name': request.name, 'instructions': request.instructions,
                'choices': [{'value': value} for value in request.choices]}
        assert wire == case['question']


def test_processed_input_destination_question_distinguishes_final_output_from_work_area():
    request = destination_question(
        'Put the pink package on the work area, unpack it, put its black product in the right box.',
        'pink package', candidates(), processing=['vla_unpack'])
    assert 'FINAL destination' in request.instructions
    assert 'selector identifies the input, not its destination' in request.instructions
    assert 'intermediate processing area' in request.instructions
    assert 'product need not exist yet' in request.instructions
    assert 'historical reference descriptions do not override' in request.instructions
    assert 'put its black product in the right box' in request.instructions
    for row in candidates():
        assert row['value'] in request.instructions and row['instance_id'] in request.instructions
    assert request.choices == ('ID1', 'ID2', 'none_of_listed', 'unknown')
    assert request.selection_mode == 'evidence'


def test_processing_membership_checks_current_input_and_preserves_original_constraints():
    instruction = 'Unpack the left package and deliver its black product to the right box.'
    question = target_membership_question('target_0', 'input-id', 'package and its product',
        instruction, processing=['vla_unpack'])
    assert instruction in question.instructions
    assert 'CURRENT INPUT membership' in question.instructions
    assert 'not current input candidates' in question.instructions
    assert 'does not authorize additional input objects' in question.instructions
    from luna_mujoco.agent.graph_prompt import GRAPH_RULES
    assert 'target_selector describes the CURRENT INPUT' in GRAPH_RULES
