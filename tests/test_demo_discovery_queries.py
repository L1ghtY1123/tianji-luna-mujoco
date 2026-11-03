"""Generic planning preserves supplied appearance proposals and registry-controlled arguments."""
from copy import deepcopy

import pytest
from test_demo_perception import demo_discovery, discovery_output
from test_perception import capture_fixture

from luna_mujoco.agent.schema import EntityProposal


@pytest.mark.parametrize('color', ['red', 'green', 'unknown', '蓝色', 'pinkish'])
def test_inventory_color_is_a_proposal_not_a_hardcoded_workflow_gate(color):
    body = discovery_output()
    body['entities'][2]['color'] = color
    body['entities'][2]['segmentation_prompt'] = color + ' open container'
    result = demo_discovery(body, []).discover(capture_fixture()[0], instruction='move selected objects')
    source = result['entities'][result['task_spec']['source_ref']]
    assert source['color'] == color
    assert source['segmentation_query']['prompt'] == color + ' open container'


@pytest.mark.parametrize('kind,prompt', [('cube', 'pink cube'), ('sphere', 'pink ball'), ('object', 'pink toy')])
def test_inventory_keeps_observed_shape_and_supplied_segmentation_phrase(kind, prompt):
    body = discovery_output()
    body['entities'][5].update(kind=kind, segmentation_prompt=prompt)
    result = demo_discovery(body, []).discover(capture_fixture()[0], instruction='move selected objects')
    item = next(item for item in result['entities'].values() if item['description'] == 'bottom')
    assert item['kind'] == kind
    assert item['segmentation_query']['prompt'] == prompt
    assert 'segmentation_prompts' not in item


@pytest.mark.parametrize('changes', [{'kind': 'teapot'}, {'segmentation_prompt': ''}, {'bbox': [0, 0, 2, 1]}])
def test_invalid_entity_proposals_are_rejected(changes):
    with pytest.raises(ValueError):
        EntityProposal.model_validate(discovery_output()['entities'][5] | changes)


def test_initial_inventory_and_quantity_support_more_than_sixty_four_objects():
    body = discovery_output()
    source = deepcopy(body['entities'][3])
    body['entities'] = body['entities'][:3]
    body['task'].update(scope='count', quantity=65, target_refs=[])
    for index in range(65):
        ref = 'item-' + str(index)
        body['entities'].append(source | {'ref': ref})
        body['task']['target_refs'].append(ref)
    result = demo_discovery(body, []).discover(capture_fixture()[0], instruction='move 65 objects')
    assert len(result['task_inventory']['source_object_ids']) == len(result['task_inventory']['selected_ids']) == 65
    assert result['task_spec']['quantity'] == 65


def test_count_subset_retains_all_candidates_numbering_and_collision_proposals():
    result = demo_discovery(discovery_output(1), []).discover(capture_fixture()[0], instruction='move one')
    identifiers = result['task_spec']['target_refs']
    assert [result['entities'][key]['display_id'] for key in identifiers] == [1, 2, 3]
    assert len(result['task_inventory']['selected_ids']) == 1
    assert all('collision' in result['entities'][key]['roles'] for key in identifiers)
    # The controller confirms semantic membership and chooses the measured nearest subset.
    assert all(result['entities'][key]['identity_source'] == 'openai_responses_proposal' for key in identifiers)


def test_uncertain_additional_entity_is_retained_for_decisions_instead_of_dropped():
    body = discovery_output()
    body['task']['source_inventory_complete'] = 'unknown'
    body['entities'].append(body['entities'][3] | {'ref': 'possible-extra', 'membership': 'unknown'})
    body['task']['target_refs'].append('possible-extra')
    result = demo_discovery(body, []).discover(capture_fixture()[0], instruction='move all relevant objects')
    assert len(result['task_spec']['target_refs']) == 4
    assert result['task_spec']['source_inventory_complete'] == 'unknown'
    assert any(item['visual_membership'] == 'unknown' for item in result['entities'].values())
