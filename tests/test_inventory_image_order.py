import copy

import numpy as np
import pytest

from jev_mujoco.task_progress import TaskProgress


def head_observation():
    positions = {'br': (70, 70), 'tl': (10, 10), 'bl': (70, 10), 'tr': (12, 70)}
    masks = {}
    objects = []
    for index, (identifier, (row, column)) in enumerate(positions.items()):
        mask = np.zeros((100, 100), dtype=bool)
        mask[row:row + 10, column:column + 10] = True
        masks[identifier] = mask
        objects.append({'id': identifier, 'kind': 'cube', 'position': [0, 0, index + 1.]})
    objects.extend([{'id': 'obstacle', 'kind': 'sphere'}, {'id': 'bin', 'kind': 'bin'},
                    {'id': 'region', 'kind': 'marker'}])
    return {'observation_id': 'head-1', 'objects': objects,
            'views': {'3': {'instance_masks': masks}}}


def progress_for(observation, count=4, **kwargs):
    progress = TaskProgress('搬运指定物体', route='table_box', requested_count=count, **kwargs)
    progress.observe(observation['objects'], observation['observation_id'])
    progress.assess_membership({'br': 'required', 'tl': 'required', 'bl': 'required',
                                'tr': 'required', 'obstacle': 'excluded'})
    return progress


def test_freeze_uses_head_rows_instead_of_distance_and_keeps_obstacles_out():
    observation = head_observation()
    progress = progress_for(observation)
    assert hasattr(progress, 'freeze_inventory')
    result = progress.freeze_inventory(observation)
    assert result['inventory_finalized'] is True
    assert result['initial_head_observation_id'] == 'head-1'
    assert {item['id']: item['display_id'] for item in result['objects']} == {
        'tl': 1, 'tr': 2, 'bl': 3, 'br': 4}
    assert set(result['scene_entities']) == {'tl', 'tr', 'bl', 'br', 'obstacle', 'bin', 'region'}
    later = copy.deepcopy(observation)
    later['views']['3']['instance_masks']['tl'] = later['views']['3']['instance_masks']['br']
    assert progress.assign_display_ids(later)['tl'] == 1
    assert progress.snapshot()['inventory_revision'] == result['inventory_revision']


def test_unknown_membership_cannot_freeze_or_consume_numbers():
    observation = head_observation()
    progress = progress_for(observation)
    progress.assess_membership({'tr': 'unknown'})
    assert hasattr(progress, 'freeze_inventory')
    with pytest.raises(ValueError, match='inventory_membership_unknown'):
        progress.freeze_inventory(observation)
    assert all('display_id' not in item for item in progress.records.values())
    progress.assess_membership({'tr': 'required'})
    assert progress.freeze_inventory(observation)['objects'][0]['display_id'] == 1


def test_missing_required_quantity_does_not_freeze_a_smaller_inventory():
    observation = head_observation()
    progress = progress_for(observation, count=5)
    assert hasattr(progress, 'freeze_inventory')
    with pytest.raises(ValueError, match='inventory_quantity_mismatch'):
        progress.freeze_inventory(observation)
    assert progress.snapshot()['inventory_finalized'] is False


def test_numbering_never_truncates_membership_to_fit_the_requested_total():
    observation = head_observation()
    progress = progress_for(observation, count=3, selection_order='inventory')
    before = dict(progress.membership)
    progress.assign_display_ids(observation)
    assert progress.membership == before


def test_missing_head_mask_reports_evidence_gap_instead_of_guessing_order():
    observation = head_observation()
    progress = progress_for(observation)
    del observation['views']['3']['instance_masks']['tr']
    assert hasattr(progress, 'freeze_inventory')
    with pytest.raises(ValueError, match='inventory_head_mask_missing:tr'):
        progress.freeze_inventory(observation)


def test_group_quota_rejects_three_red_members_instead_of_two_red_and_one_green():
    observation = head_observation()
    groups = [{'group_id': 'group_0', 'source_text': '两个红色', 'requested_count': 2},
              {'group_id': 'group_1', 'source_text': '一个绿色', 'requested_count': 1}]
    progress = TaskProgress('两个红色和一个绿色', requested_count=3, quantity_groups=groups)
    progress.observe(observation['objects'], 'head-1')
    progress.assess_membership({'tl': 'required', 'tr': 'required', 'bl': 'required',
                                'br': 'excluded', 'obstacle': 'excluded'})
    progress.assess_groups({'group_0': {'tl': 'required', 'tr': 'required', 'bl': 'required'},
                            'group_1': {}})
    with pytest.raises(ValueError, match='inventory_group_quantity_mismatch'):
        progress.freeze_inventory(observation)
    assert progress.inventory_finalized is False
    progress.assess_groups({'group_0': {'tl': 'required', 'tr': 'required'},
                            'group_1': {'bl': 'required'}})
    assert progress.freeze_inventory(observation)['inventory_finalized'] is True


def test_build_scene_contract_rejects_unfinalized_inventory_and_missing_region():
    from jev_mujoco.skills import Registry, SkillSelectionError
    registry = Registry()
    facts = {'observed': True, 'target_selected': True, 'arm_selected': True}
    with pytest.raises(SkillSelectionError) as failure:
        registry.validate_selection('build_scene', facts)
    assert set(failure.value.details['unmet_requirements']) == {'inventory_finalized', 'destination_ready'}
