import numpy as np

from luna_mujoco.observation_labels import responses_reference_labels


def test_original_boxes_are_archived_as_proposals_without_sam_or_identity_claims():
    capture = {'observation_id': 'initial', 'generation': 5,
               'views': {'3': {'rgb': np.zeros((10, 10, 3), dtype=np.uint8)}}}
    reference = {'entities': {'permanent-id': {'reference_camera_id': '3',
                 'bbox': [.1, .2, .6, .8], 'kind': 'bin', 'color': 'blue'}}}
    labels = responses_reference_labels(capture, reference)
    assert labels['analysis_status'] == 'responses_proposal'
    assert labels['observation_id'] == 'initial'
    assert not labels['used_as_model_input']
    item = labels['items'][0]
    assert item['id'] == 'permanent-id'
    assert item['status'] == 'unverified'
    assert item['box_source'] == 'responses_reference_proposal'
    assert np.allclose([item['x'], item['y'], item['width'], item['height']], [.1, .2, .5, .6])
