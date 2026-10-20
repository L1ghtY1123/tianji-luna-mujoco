from geometry_perception_fixture import GeometryPerception
from test_perception import FrozenSegmenter, capture_fixture


def test_initial_segmentation_queries_requested_roles_and_generic_obstacles():
    capture, masks = capture_fixture()
    calls = []

    class Detector(FrozenSegmenter):
        def segment(self, rgb, text, timestamp_us):
            calls.append(text)
            return super().segment(rgb, text.removeprefix('red '), timestamp_us)

    observed = GeometryPerception(Detector(masks), identities=('first', 'second')).observe(
        capture, checkpoint='initial', source_hint={'kind': 'cube', 'color': 'red',
                                                    'destination': {'color': 'blue'}})
    assert calls == ['red cube'] * 3
    assert len(observed['objects']) == 2
    assert abs(observed['table']['surface_height_m'] - .6) < .005


def test_same_frame_head_analysis_is_reused_for_geometry_without_new_sam3_call():
    capture, masks = capture_fixture()
    calls = []
    class Detector(FrozenSegmenter):
        def segment(self, rgb, text, timestamp_us):
            calls.append(text)
            return super().segment(rgb, text, timestamp_us)
    perception = GeometryPerception(Detector(masks))
    hint = {'kind': 'cube', 'color': 'any', 'destination': {'color': 'blue'}}
    first = perception.observe(capture, checkpoint='initial', source_hint=hint | {'camera_ids': ['3']})
    count = len(calls)
    second = perception.observe(capture, checkpoint='initial', source_hint=hint)
    assert second['sam3_request_count'] == 2
    assert len(calls) == count + 2
    assert first['sam3_request_count'] == 1
    assert all(query['cache_hit'] for query in second['sam3_queries'] if query['camera_id'] == '3')
