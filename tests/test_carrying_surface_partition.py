"""Historical measured surfaces retain full coverage under the new coarse budget."""
from pathlib import Path

import numpy as np
import pytest
from test_tight_carrying import assert_covered, measured_binding

from luna_mujoco.carrying import carrying_geometry


@pytest.mark.parametrize("name", ["luna_first_carried_surface", "luna_second_carried_surface"])
@pytest.mark.parametrize("budget", [24, 31])
def test_small_budget_preserves_both_frozen_surfaces_and_original_padding(name, budget):
    # The archives retain the old Link3-only collision classification. Current
    # exact owning-hand touch pairs are covered in test_carrying_touch_policy;
    # coarse envelope overlap with those nine links is no longer a fit failure.
    with np.load(Path(__file__).parent / f"fixtures/{name}.npz", allow_pickle=False) as fixture:
        points = fixture["points_tool_m"]
        measured_count = int(fixture["measured_point_count"])
        padding = float(fixture["uncertainty_m"] + fixture["margin_m"])
    original = points.copy()
    provenance = {"source": "registered_rgbd_surface_with_explicit_height_completion",
        "point_count": len(points), "measured_point_count": measured_count,
        "synthetic_point_count": len(points) - measured_count,
        "measured_reference_tool_m": points[:measured_count].mean(axis=0).tolist(),
        "geometry_completion": {"source": "measured_surface_and_user_height_prior",
                                "synthetic_points_are_measured": False}}
    result = carrying_geometry(measured_binding(points, sphere_budget=budget, measurement_provenance=provenance))
    spheres = np.asarray(result["spheres"])
    assert 0 < result["sphere_count"] <= result["sphere_budget"] == budget < 32
    assert result["minimum_partition_size_m"] == .01
    assert result["fit_method"] == "bounded_principal_splits_tight_centers"
    assert result["partition_direction_source"] == "point_group_principal_component"
    assert padding == pytest.approx(.006)
    assert_covered(points, spheres, padding)
    np.testing.assert_array_equal(points, original)
    assert sum(row["measured"] for row in result["source_counts_by_sphere"]) == measured_count
    assert sum(row["height_completion"] for row in result["source_counts_by_sphere"]) == len(points) - measured_count
