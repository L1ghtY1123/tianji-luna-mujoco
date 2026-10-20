"""Measured marker identities cross perception, task selection and verification."""


import numpy as np
import pytest
from geometry_perception_fixture import GeometryPerception
from test_perception import capture_fixture

from jev_mujoco.observation_motion import initial_view_evidence, matching_instances
from jev_mujoco.perception import PerceptionError, public_observation
from jev_mujoco.placement import public_observation as public_placement


def marked_capture():
    capture, source_masks = capture_fixture()
    marker = np.zeros((80, 80), bool)
    marker[8:28, 35:65] = True
    for view in capture["views"].values():
        view["rgb"][marker] = [242, 184, 31]
    return capture, source_masks, marker


class MarkedSegmenter:
    def __init__(self, masks, marker):
        self.masks, self.marker = masks, marker
        self.calls = []

    def segment(self, rgb, text, timestamp_us):
        self.calls.append(text)
        if text in {"cube", "object", "red cube"}:
            return [{"mask": mask | self.marker, "score": .92} for mask in self.masks]
        return []


def test_marker_destination_uses_measured_masks_and_is_removed_from_generic_source_masks():
    capture, masks, marker = marked_capture()
    segmenter = MarkedSegmenter(masks, marker)
    observed = GeometryPerception(segmenter, identities=('first', 'second')).observe(capture, checkpoint="initial", source_hint={
        "kind": "any", "destination": {"kind": "marker", "color": "yellow"}})
    targets = [obj for obj in observed["objects"] if obj["kind"] != "marker"]
    destination, = [obj for obj in observed["objects"] if obj["kind"] == "marker"]
    assert len(targets) == 2
    assert destination["id"] == "marker-yellow"
    assert segmenter.calls == ["cube"] * 3
    for view in observed["views"].values():
        np.testing.assert_array_equal(view["instance_masks"][destination["id"]], marker)
        assert all(not np.any(view["instance_masks"][source["id"]] & marker) for source in targets)
    assert destination["point_sources"]["observation_id"] == capture["observation_id"]
    assert destination["observation_provenance"]["checkpoint"] == "initial"
    assert "SIM_SECRET" not in str(public_observation(observed))


def test_marker_ambiguity_is_a_perception_error_and_editor_fields_are_not_fallback():
    capture, masks, marker = marked_capture()
    for view in capture["views"].values():
        view["rgb"][55:75, 30:50] = [242, 184, 31]
    with pytest.raises(PerceptionError, match="table_marker_color_ambiguous"):
        GeometryPerception(MarkedSegmenter(masks, marker), identities=('first', 'second')).observe(capture)


def test_requested_source_color_is_checked_after_marker_pixels_are_removed():
    capture, masks, marker = marked_capture()
    observed = GeometryPerception(MarkedSegmenter(masks, marker), identities=('first', 'second')).observe(capture, source_hint={"kind": "cube", "color": "red"})
    assert len([obj for obj in observed["objects"] if obj["kind"] == "cube" and obj["color"] == "red"]) == 2


def test_legacy_include_containers_does_not_expand_fixed_rgb_inventory():
    capture, masks, marker = marked_capture()
    detector = MarkedSegmenter(masks, marker)
    GeometryPerception(detector, identities=('first', 'second')).observe(capture, checkpoint="initial", source_hint={"kind": "cube", "include_containers": True})
    assert detector.calls == ['cube'] * 3


@pytest.mark.parametrize("missing_edge_pixels", [0, 30])
def test_decisions_container_and_contents_groups_keep_distinct_measured_pixels(missing_edge_pixels):
    capture, _ = capture_fixture()
    container = np.zeros((80, 80), bool)
    container[20:60, 20:60] = True
    contents = np.zeros_like(container)
    contents[32:42, 32:42] = True
    duplicate = container.copy()
    duplicate[20, 20:20 + missing_edge_pixels] = False
    for view in capture["views"].values():
        view["rgb"][:] = 140
        view["depth_m"][:] = .4
        view["rgb"][container] = [20, 60, 210]
        view["depth_m"][container] = .35
        view["depth_m"][contents] = .32

    class Segmenter:
        def segment(self, rgb, prompt, timestamp_us):
            masks = [container] if prompt == "open box" else [duplicate, contents] if prompt == "object" else []
            return [{"mask": mask, "score": .9} for mask in masks]

    observed = GeometryPerception(Segmenter(), identities=('bin', 'bin', 'contents'), kinds={'bin': 'bin', 'contents': 'object'}).observe(capture, checkpoint="fine",
        source_hint={"kind": "any", "include_containers": True})
    assert len(observed["objects"]) == 2
    source, = [obj for obj in observed["objects"] if obj["kind"] == "object"]
    container_object, = [obj for obj in observed["objects"] if obj["kind"] == "bin"]
    contents_interior = np.zeros_like(contents)
    contents_interior[33:41, 33:41] = True
    for view in observed["views"].values():
        np.testing.assert_array_equal(view["instance_masks"][source["id"]], contents_interior)
        np.testing.assert_array_equal(view["instance_masks"][container_object["id"]], container & ~contents)


def test_measured_marker_head_evidence_does_not_require_a_sam_detection():
    source = {"id": "source", "kind": "cube", "color": "red", "coverage": {
        "camera_ids": ["1", "2", "3"], "sam3_camera_ids": ["3"]}}
    source['points_world_m'] = [[x, y, .6] for x in [0., .01, .02, .03] for y in [0., .01, .02, .03]]
    marker = {"id": "marker-yellow", "kind": "marker", "color": "yellow", "coverage": {
        "camera_ids": ["3"], "sam3_camera_ids": []}}
    observation = {"objects": [source, marker]}
    assert matching_instances(observation, {"kind": "any"}) == [source]
    assert matching_instances(observation, {"kind": "marker"}, destination=True) == [marker]
    evidence = initial_view_evidence(observation, "grasp_place", {"kind": "cube"}, {"kind": "marker"})
    assert evidence["views_ready"] is True
    marker["coverage"]["camera_ids"] = ["1"]
    with pytest.raises(ValueError, match="head_destination_missing"):
        initial_view_evidence(observation, "grasp_place", {"kind": "cube"}, {"kind": "marker"})


@pytest.mark.parametrize("projection", [public_observation, public_placement])
def test_public_marker_and_visual_semantics_are_bounded_whitelists(projection):
    capture, masks, marker = marked_capture()
    observed = GeometryPerception(MarkedSegmenter(masks, marker), identities=('first', 'second')).observe(capture)
    destination = next(obj for obj in observed["objects"] if obj["kind"] == "marker")
    destination["visual_semantics"] = {"label": "yellow placement area", "visibility": "visible",
        "ambiguous": False, "on_table": True, "inside_id": None, "on_marker_id": None,
        "evidence": "Color region on table.", "points_world_m": "poison", "T_world_object": "poison"}
    visible = next(obj for obj in projection(observed)["objects"] if obj["kind"] == "marker")
    assert len(visible["placement_reference"]["region_polygon_world_xy_m"]) >= 3
    assert visible["visual_semantics"]["label"] == "yellow placement area"
    assert "poison" not in str(visible)
    assert "masks_by_camera" not in visible and "point_sources" not in visible
