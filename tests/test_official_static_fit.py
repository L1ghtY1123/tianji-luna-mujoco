"""The offline static fitter must not silently accept a degraded SDK fallback."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/fit_official_static.py"
    spec = importlib.util.spec_from_file_location("official_static", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_fitter_rejects_surface_fallback_instead_of_hiding_missing_geometry():
    module = load_module()
    result = SimpleNamespace(
        debug_info={"fallback_used": True},
        centers=np.zeros((8, 3)),
        radii=np.full(8, .005),
    )
    with pytest.raises(ValueError, match="fallback"):
        module.final_spheres(result, np.eye(4), 8)


def test_final_result_changes_only_rigid_frame_and_preserves_all_radii():
    module = load_module()
    centers = np.array([[.01123456789, .02, .03], [.04, -.05, .06]])
    radii = np.array([.0234567891, .0456789123])
    original = centers.copy()
    transform = np.array([[0., -1., 0., .02], [1., 0., 0., -.04], [0., 0., 1., .1], [0., 0., 0., 1.]])
    result = SimpleNamespace(debug_info={"fallback_used": False}, centers=centers, radii=radii)
    spheres = module.final_spheres(result, transform, 2)
    actual = np.array([sphere["center"] for sphere in spheres])
    np.testing.assert_allclose(actual @ transform[:3, :3].T + transform[:3, 3], centers, atol=1e-15)
    np.testing.assert_array_equal([sphere["radius"] for sphere in spheres], radii)
    np.testing.assert_array_equal(centers, original)
    with pytest.raises(ValueError, match="budget"):
        module.final_spheres(result, transform, 1)


def test_nonrigid_fit_frame_is_rejected():
    module = load_module()
    result = SimpleNamespace(debug_info={}, centers=np.zeros((1, 3)), radii=np.ones(1))
    transform = np.diag([2., 1., 1., 1.])
    with pytest.raises(ValueError, match="rigid"):
        module.final_spheres(result, transform, 1)


def test_retained_column_is_matched_to_exact_source_spheres_not_recipe_row_order():
    module = load_module()
    columns = [{"center": [0., 0., .1 + i * .05], "radius": .09389} for i in range(16)]
    rib = {"center": [.1, 0., .06], "radius": .045}
    source = [rib] + deepcopy(list(reversed(columns)))
    archive = {"calls": [{"label": f"column_segment_{i}", "spheres": [s]} for i, s in enumerate(columns)]}
    selected, indices = module.retained_column(source, archive)
    assert selected == columns
    assert indices == list(range(16, 0, -1))
    selected[0]["radius"] = 1.
    assert source[-1]["radius"] == .09389
    source[-1]["radius"] = .094
    with pytest.raises(ValueError, match="exactly once"):
        module.retained_column(source, archive)


def test_archived_body_replays_only_thirty_public_final_cells_without_old_body_union():
    module = load_module()
    project = Path(__file__).resolve().parents[1]
    archive = project / "data/collision-correction-2026-09-30/static-fit.json"
    parts = module.load_archived_body_parts(project, archive)
    assert len(parts) == 30
    assert len({part["name"] for part in parts}) == 30
    assert all(part["name"].startswith("body_cell_") for part in parts)
    for part in parts:
        record = part["archived_record"]
        assert record["result_source"] == "official_final"
        assert record["parameters"]["coverage_weight"] == 10000.
        assert len(record["spheres"]) == 1
        np.testing.assert_array_equal(record["fit_from_link"], np.eye(4))
        assert record["final_centers_fit_frame_m"] == [record["spheres"][0]["center"]]
