"""Descriptions propose pixels, not permanent identities."""
import numpy as np

from luna_mujoco.demo_perception import DemoPerception


def test_nearly_identical_masks_merge_but_distinct_objects_and_cameras_do_not():
    mask = np.zeros((20, 30), bool)
    mask[:10, :10] = True
    changed = mask.copy()
    changed[0, 0] = False
    other = np.roll(mask, 15, axis=1)
    rows = [{'camera_id': camera, 'kind': 'object', 'requested_color': 'pink',
             'silhouette_mask': pixels, 'prompt': prompt, 'score': .9}
            for camera, pixels, prompt in [('2', mask, 'pink object'),
                ('2', changed, 'pink foam wrapped object'), ('2', other, 'pink object'),
                ('1', mask, 'pink object')]]
    result = DemoPerception()._deduplicate_detections(rows)
    assert len(result) == 3
    assert result[0]['prompt_proposals'] == ['pink object', 'pink foam wrapped object']
    assert all('entity_id' not in row for row in result)
