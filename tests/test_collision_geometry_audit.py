"""Geometric quality must measure the exposed union, not hidden sphere surfaces."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_collision_geometry.py"


def load_geometry(monkeypatch):
    assert SCRIPT.exists(), "the exposed geometry quality audit is not implemented"
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("collision_geometry_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sphere_union_boundary_discards_contained_and_overlapping_sphere_surfaces(monkeypatch):
    audit = load_geometry(monkeypatch)
    spheres = np.array([[0, 0, 0, 2.], [0, 0, 0, .5], [2., 0, 0, 1.]])
    points = np.array([[2., 0, 0], [.5, 0, 0], [3., 0, 0], [-2., 0, 0]])
    assert audit.exposed_sphere_points(points, spheres).tolist() == [False, False, True, True]


def test_sphere_sampling_weights_represent_area_not_number_of_spheres(monkeypatch):
    audit = load_geometry(monkeypatch)
    points, owners, weights = audit.sample_sphere_surfaces(np.array([[0, 0, 0, 1.], [5, 0, 0, 2.]]), 100)
    assert points.shape == (200, 3)
    assert weights[owners == 1].sum() / weights[owners == 0].sum() == pytest.approx(4.)
    np.testing.assert_allclose(np.linalg.norm(points[owners == 0], axis=1), 1.)


def test_weighted_distance_summary_has_explicit_millimetres_and_worst_point(monkeypatch):
    audit = load_geometry(monkeypatch)
    result = audit.distance_summary(np.array([0., .001, .020]), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                                    np.array([.96, .03, .01]))
    assert result["p95_mm"] == 0.
    assert result["maximum_mm"] == 20.
    assert result["within_1mm_fraction"] == pytest.approx(.99)
    assert result["worst_point_world_m"] == [2., 0., 0.]


def test_nearest_surface_checks_adjacent_parts_not_only_the_owning_mesh(monkeypatch):
    pytest.importorskip("rtree")
    audit = load_geometry(monkeypatch)
    first = trimesh.creation.box(extents=[2, 2, 2])
    second = first.copy()
    second.apply_translation([3, 0, 0])
    nearest, distance, owners = audit.nearest_surfaces({"a": first, "b": second}, np.array([[1.9, 0, 0]]))
    np.testing.assert_allclose(nearest, [[2., 0, 0]])
    np.testing.assert_allclose(distance, [.1])
    assert owners == ["b"]


def test_nearest_query_preserves_points_on_small_metre_scale_triangles(monkeypatch):
    pytest.importorskip("rtree")
    audit = load_geometry(monkeypatch)
    vertices = np.array([[.017046, -.039125, -.05929], [.01706, -.039222, -.059425],
                         [.019137, -.038294, -.059828]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    point = np.array([.29503802, .07241868, .63254330]) @ vertices
    nearest, distance, _ = audit.nearest_surfaces({"thin": mesh}, np.array([point]))
    assert distance[0] < 1e-12
    np.testing.assert_allclose(nearest[0], point, atol=1e-12)


def test_closed_geometry_inside_and_open_geometry_unknown_are_not_reported_as_outward(monkeypatch):
    pytest.importorskip("rtree")
    audit = load_geometry(monkeypatch)
    box = trimesh.creation.box(extents=[2, 2, 2])
    open_box = box.copy()
    open_box.update_faces(np.arange(len(open_box.faces)) != 0)
    open_box.apply_translation([4, 0, 0])
    parts = audit.solid_components({"closed": box, "open": open_box})
    inside, unknown = audit.classify_points(parts, np.array([[0, 0, 0], [4, 0, 0], [7, 0, 0]]))
    assert inside.tolist() == [True, False, False]
    assert unknown.tolist() == [False, True, False]


def test_visibility_excludes_internal_assembly_surface(monkeypatch):
    pytest.importorskip("rtree")
    audit = load_geometry(monkeypatch)
    box = trimesh.creation.box(extents=[2, 2, 2])
    points = np.array([[1., 0, 0], [.25, 0, 0]])
    normals = np.array([[1., 0, 0], [1., 0, 0]])
    visible = audit.visible_surface_points({"box": box}, points, normals)
    assert visible.tolist() == [True, False]


def test_per_sphere_report_marks_hidden_spheres_without_claiming_a_zero_protrusion_bound(monkeypatch):
    pytest.importorskip("rtree")
    audit = load_geometry(monkeypatch)
    assert hasattr(audit, "measure_sphere_boundaries"), "per-sphere exposed boundary reports are missing"
    meshes = {"part": trimesh.creation.box(extents=[.1, .1, .1])}
    spheres = {"part": np.array([[0, 0, 0, 2.], [0, 0, 0, .5]])}
    result = audit.measure_sphere_boundaries("part", meshes, spheres, audit.solid_components(meshes), 64)
    outer, hidden = result["per_sphere"]
    assert outer["outward"]["maximum_mm"] > 1000.
    assert hidden["union_boundary_samples"] == 0
    assert hidden["outward"]["maximum_mm"] is None
