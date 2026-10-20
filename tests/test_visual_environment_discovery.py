from geometry_perception_fixture import GeometryPerception
from test_broad_perception import PromptSegmenter
from test_perception import capture_fixture


def test_unmentioned_object_is_observed_alongside_explicit_target():
    capture, masks = capture_fixture()
    for view in capture['views'].values():
        view['rgb'][masks[1]] = [10, 10, 10]
    detector = PromptSegmenter({'red cube': [masks[0]], 'object': [masks[0], masks[1]]})
    result = GeometryPerception(detector, identities=('first', 'first', 'second'),
        prompts={'first': 'red cube', 'second': 'object'}).observe(capture, checkpoint='fine', source_hint={
        'kind': 'cube', 'color': 'red', 'camera_ids': ['3']})
    assert 'object' in detector.calls
    assert {obj['color'] for obj in result['objects']} == {'red', 'black'}
    assert len(result['objects']) == 2  # source query and generic query do not double-count


def test_generic_query_miss_still_discovers_unmentioned_workpiece():
    capture, masks = capture_fixture()
    for view in capture['views'].values():
        view['rgb'][masks[1]] = [10, 10, 10]
    detector = PromptSegmenter({'red cube': [masks[0]], 'cube': masks, 'object': []})
    result = GeometryPerception(detector, identities=('first', 'first', 'second'),
        prompts={'first': 'red cube', 'second': 'cube'}).observe(capture, checkpoint='fine', source_hint={
        'kind': 'cube', 'color': 'red', 'camera_ids': ['3']})
    assert {obj['color'] for obj in result['objects']} == {'red', 'black'}
