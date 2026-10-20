"""Broad source requests combine bounded SAM prompts without duplicating bodies."""

import numpy as np
import pytest
from geometry_perception_fixture import GeometryPerception
from test_perception import capture_fixture


class PromptSegmenter:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def segment(self, rgb, text, timestamp_us):
        self.calls.append(text)
        return [{"mask": mask, "score": .9} for mask in self.answers.get(text, [])]


@pytest.mark.parametrize("kind", ["any", "object"])
def test_broad_source_recovers_two_cubes_when_generic_query_misses_them(kind):
    capture, cubes = capture_fixture()
    container = np.zeros((80, 80), bool)
    container[8:28, 30:60] = True
    for view in capture["views"].values():
        view["rgb"][cubes[1]] = [20, 180, 30]
        view["rgb"][container] = [20, 60, 210]
        view["depth_m"][container] = .35
    detector = PromptSegmenter({"cube": [*cubes, container], "object": [container], "open box": [container]})
    observed = GeometryPerception(detector, identities=('container', 'first', 'second', 'container'),
        kinds={'container': 'bin'}).observe(capture, checkpoint="initial", source_hint={
        "kind": kind, "color": "any", "camera_ids": ["3"], "include_containers": True})
    assert detector.calls == ["open box", "cube"]
    assert sorted((item["kind"], item["color"]) for item in observed["objects"]) == [
        ("bin", "blue"), ("cube", "green"), ("cube", "red")]
    assert observed["sam3_request_count"] == 2
    assert all(item["coverage"]["sam3_camera_ids"] == ["3"] for item in observed["objects"])


def test_specific_source_keeps_target_query_and_discovers_environment():
    capture, cubes = capture_fixture()
    detector = PromptSegmenter({"red cube": cubes})
    observed = GeometryPerception(detector).observe(capture, source_hint={"kind": "cube", "color": "red", "camera_ids": ["3"]})
    assert detector.calls == ["red cube"]
    assert len(observed["objects"]) == 2
