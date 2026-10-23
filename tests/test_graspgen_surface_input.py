"""Inference density must preserve the labeled height through the official filter."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_demo_planning import GraspClient, grasp_placement
from test_grasp_height_completion import completed_scene

from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.demo_grasp_input import sample_completed_side_surface


def test_graspgen_side_samples_survive_fixed_official_filter_without_changing_geometry():
    snapshot = completed_scene()
    target = snapshot["objects"][0]
    original = deepcopy(target)
    client = GraspClient()
    result = DemoAlgorithms(graspgen_client=client).generate_grasps(
        snapshot, target["id"], arm="A", placement_request=grasp_placement())
    request = client.calls[0]["object_points_camera_m"]
    origin = np.asarray(original["points_world_m"], dtype=float).mean(axis=0)
    original_input = (original["completed_geometry_points_world_m"] - origin).astype(np.float32)
    assert len(request) > len(original_input)
    np.testing.assert_array_equal(request[:len(original_input)], original_input)
    # Exact official public preprocessing criterion: exclude self, average
    # the 20 nearest L1 distances, retain samples strictly below 14 mm.
    keep = cKDTree(request).query(request, k=21, p=1)[0][:, 1:].mean(axis=1) < .014
    assert np.ptp(request[keep, 2]) >= np.ptp(original_input[:, 2]) - 1e-7
    np.testing.assert_array_equal(target["points_world_m"], original["points_world_m"])
    np.testing.assert_array_equal(target["completed_geometry_points_world_m"],
                                  original["completed_geometry_points_world_m"])
    assert target["geometry_completion"] == original["geometry_completion"]
    replay = result["candidate_replay"]
    evidence = replay["graspgen_input_sampling"]
    assert evidence["spacing_m"] == .002
    assert evidence["source"] == "measured_convex_hull_side_surface_sampling"
    assert evidence["actual_point_count"] == len(request) <= evidence["maximum_point_count"]
    assert evidence["source_geometry_point_count"] == len(original_input)
    assert evidence["added_surface_point_count"] == len(request) - len(original_input)
    assert replay["graspgen_input_source"] == "height_completed"
    assert result["cloud"]["synthetic_point_count"] == len(request) - len(original["points_world_m"])
    assert replay["completion_provenance"] == original["geometry_completion"]


def test_inference_surface_budget_failure_does_not_coarsen_or_call_service():
    snapshot = completed_scene()
    target = snapshot["objects"][0]
    target["points_world_m"][:, :2] *= 100
    target["completed_geometry_points_world_m"][:, :2] *= 100
    client = GraspClient()
    with pytest.raises((ValueError, RuntimeError), match="graspgen_surface_sampling_budget"):
        DemoAlgorithms(graspgen_client=client).generate_grasps(
            snapshot, target["id"], arm="A", placement_request=grasp_placement())
    assert not client.calls


@pytest.mark.parametrize("capture", ["new9bc1", "olda979", "historicalc6b5"])
def test_frozen_inference_preserves_full_height_and_original_points(capture):
    with np.load(Path(__file__).parent / "fixtures/graspgen_surface_inputs.npz") as arrays:
        measured = arrays[capture + "_measured"]
        completed = arrays[capture + "_completed"]
    points = sample_completed_side_surface(measured, completed)
    np.testing.assert_array_equal(points[:len(completed)], completed)
    assert len(points) <= 12000
    mean_distances = cKDTree(points).query(points, k=21, p=1)[0][:, 1:].mean(axis=1)
    retained = points[mean_distances < .014]
    assert np.ptp(retained[:, 2]) == pytest.approx(np.ptp(completed[:, 2]), abs=1e-7)
    assert np.ptp(retained[:, 2]) >= .1
    assert len(retained) == len(points)
    np.testing.assert_allclose(points.min(axis=0), completed.min(axis=0), atol=1e-7)
    np.testing.assert_allclose(points.max(axis=0), completed.max(axis=0), atol=1e-7)
    if capture == "new9bc1":
        old_keep = cKDTree(completed).query(completed, k=21, p=1)[0][:, 1:].mean(axis=1) < .014
        assert np.ptp(completed[old_keep, 2]) == pytest.approx(.0166667, abs=1e-6)
