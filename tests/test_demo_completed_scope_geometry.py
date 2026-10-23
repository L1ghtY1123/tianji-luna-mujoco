"""Completed correspondence scopes use object returns, not mixed support pixels."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_demo_completed_tracking import resolve, tracking_inputs

from luna_mujoco.demo_completed_tracking import completed_candidate_scopes, completed_identity_tracks
from luna_mujoco.visual_api import VisualInventoryError


@pytest.mark.parametrize("name,matching_count", [("pink_object", 702), ("elliptical_pink_object", 701)])
def test_frozen_f9bd_support_returns_do_not_hide_the_completed_object(name, matching_count):
    fixture = Path(__file__).parent / "fixtures/f9bd-completed-proposals.npz"
    metadata = json.loads(fixture.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(fixture) as arrays:
        points = arrays[name + "_points_world_m"]
        mask = arrays[name + "_mask"]
        pixels = arrays[name + "_pixels_rc"]
    original = points.copy()
    detection = {"points": points, "camera_id": "3", "silhouette_mask": mask, "pixels_rc": pixels}
    scopes = completed_candidate_scopes({"observation_id": metadata["observation_id"]},
                                      [detection], metadata["tracks"])
    assert set(scopes) == {0}
    scope = scopes[0]
    assert scope["raw_point_count"] == 729
    assert scope["matching_point_count"] == scope["point_count"] == matching_count
    assert scope["support_plane_removed_point_count"] == 729 - matching_count
    assert scope["support_plane_tolerance_m"] == .006
    assert scope["owner_id"] == next(iter(metadata["tracks"]))
    assert scope["location_is_identity_proof"] is False
    np.testing.assert_array_equal(points, original)
    assert detection["silhouette_mask"] is mask and detection["pixels_rc"] is pixels


def test_support_filter_preserves_raw_candidate_for_fresh_decisions_and_final_geometry():
    def mutate(capture, detections, entities, reference, progress):
        detections[0]["points"][0] = [.6, .2, .56]  # Table outside slot, not object ownership.
    result, calls, (_, detections, _, _, _) = resolve(mutate=mutate)
    assert [question["name"] for question in calls[0]["questions"]] == ["tracked_0"]
    assert result["identity_coverage"]["required_entity_ids"] == ["pink", "remaining"]
    group = next(group for group in result["groups"] if group[0]["entity_id"] == "pink")
    np.testing.assert_array_equal(group[0]["points"], detections[0]["points"])
    assert result["completed_tracking"]["candidate_scopes"][0]["support_plane_removed_point_count"] == 1


@pytest.mark.parametrize("case", [
    "only_support", "outside_xy", "outside_height", "outside_support_band", "nonfinite", "ambiguous",
])
def test_support_filter_never_crops_object_geometry_to_make_a_unique_match(case):
    capture, detections, entities, reference, progress = tracking_inputs()
    tracks = completed_identity_tracks(capture, reference, progress, entities)
    points = detections[0]["points"]
    points[0] = [.6, .2, .56]
    if case == "only_support":
        points[:, 2] = .56
    elif case == "outside_xy":
        points[1] = [.6, .2, .63]
    elif case == "outside_height":
        points[1] = [.4, 0., .70]
    elif case == "outside_support_band":
        points[1] = [.6, .2, .567]
    elif case == "nonfinite":
        points[1, 0] = np.nan
    else:
        tracks["other"] = deepcopy(tracks["pink"])
    original = points.copy()
    assert completed_candidate_scopes(capture, detections, tracks) == {}
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("value,confidence", [("yes", .79), ("unknown", .99), ("no", .99)])
def test_support_removal_cannot_promote_negative_or_uncertain_visual_correspondence(value, confidence):
    def mutate(capture, detections, entities, reference, progress):
        detections[0]["points"][0] = [.6, .2, .56]
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        resolve({"tracked_0": (value, confidence), "identity_1": ("remaining", .95)}, mutate=mutate)
