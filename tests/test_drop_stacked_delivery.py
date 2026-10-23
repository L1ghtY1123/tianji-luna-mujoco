"""A random drop may overlap a previous product; compare visible contents."""
import json
from copy import deepcopy

from test_drop_region_presentation import measured_drop_inputs, request, zooms


def test_stacked_delivery_uses_visual_addition_not_mandatory_separate_count():
    capture, action, reference = measured_drop_inputs()
    for frame in (capture, reference):
        frame['views']['3'] = deepcopy(frame['views']['1'])
    _, payload = request(capture, action, reference)
    question = payload['questions'][0]['instructions']
    assert 'additional black workpiece' in question
    assert 'overlap or stack' in question
    assert 'viewpoint change alone' in question
    assert 'individually distinguishable' not in question
    details = zooms(payload)
    assert {(d['frame_role'], d['source_camera_id']) for d in details} == {
        ('BEFORE', '1'), ('CURRENT', '1'), ('BEFORE', '3'), ('CURRENT', '3')}
    assert 'derived_shell_components' not in json.dumps(payload)
