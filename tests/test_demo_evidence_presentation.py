import copy
import json

import numpy as np
import pytest

from luna_mujoco.demo_grasp import prepare_grasp_candidates
from luna_mujoco.geometry import grasp_targets
from luna_mujoco.observation_labels import group_candidate_boxes
from luna_mujoco.visual_image_selection import compact_decision_images


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_equivalent_jaw_pose_uses_active_arm_with_one_local_z_insertion(arm):
    canonical = np.diag([1., -1., -1., 1.])
    canonical[:3, 3] = [.5, .2, .7]
    pose = grasp_targets(np.array([canonical]), arm)[0]
    reference = pose.copy()
    reference[:3, :2] *= -1
    candidates = [{'index': 4, 'T_world_tcp': pose.tolist()}]
    before = copy.deepcopy(candidates)
    _, accepted, evidence = prepare_grasp_candidates(candidates, robot_base_world_m=[0, 0, 0],
        target_reference_world_m=[.5, .2, .6], reference_tcp=reference, planning_settings={'grasp_insertion_m': .01, 'grasp_maximum_tilt_deg': 45., 'grasp_require_away_from_robot': True})
    actual = np.array(accepted[0]['T_world_tcp'])
    np.testing.assert_allclose(actual[:3, :3], reference[:3, :3])
    np.testing.assert_allclose(actual[:3, 3], pose[:3, 3] + .01 * pose[:3, 2])
    assert evidence['insertion_m'] == .01
    assert candidates == before


def test_image_compaction_preserves_questions_and_unique_current_cameras():
    content = []
    for role in ['INITIAL', 'CURRENT']:
        for camera in (1, 2, 3):
            content.extend([{'type': 'input_text', 'text': f'{role} RGB camera {camera}'},
                {'type': 'input_image', 'image_url': f'data:image/jpeg;base64,camera{camera}'}])
    content.extend([{'type': 'input_text', 'text': json.dumps({
        'image_role': 'current_annotations', 'items': [], 'regions': []})},
        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,emptyannotation'}])
    payload = {'input': [{'content': content}], 'questions': [{'name': 'identity_1'}]}
    before = copy.deepcopy(payload)
    compact, counts = compact_decision_images(payload)
    assert counts == {'original_image_count': 7, 'image_count': 3, 'removed_image_count': 4}
    assert compact['questions'] == payload['questions']
    assert payload == before
    images = [part for part in compact['input'][0]['content'] if part['type'] == 'input_image']
    assert len({part['image_url'] for part in images}) == 3
    parts = compact['input'][0]['content']
    assert [parts[index - 1]['text'] for index, part in enumerate(parts)
            if part['type'] == 'input_image'] == [f'CURRENT RGB camera {camera}' for camera in (1, 2, 3)]


def test_shared_candidate_outline_does_not_merge_adjacent_objects_or_permanent_ids():
    items = [{'id': 'candidate-1', 'display_label': 'C1', 'box': [.1, .1, .3, .3]},
             {'id': 'candidate-2', 'display_label': 'C2', 'box': [.101, .1, .3, .3]},
             {'id': 'candidate-3', 'display_label': 'C3', 'box': [.31, .1, .5, .3]},
             {'id': 'permanent-4', 'box': [.1, .1, .3, .3]}]
    before = copy.deepcopy(items)
    grouped = group_candidate_boxes(items)
    assert len(grouped) == 3
    assert grouped[0]['display_label'] == 'C1 / C2'
    assert items == before


