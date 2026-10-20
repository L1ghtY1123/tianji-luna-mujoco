"""Budgeted offline fitting consumes only the official final result."""

import importlib.util
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_module(filename):
    spec = importlib.util.spec_from_file_location("collision_fit_test", SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeMetrics:
    coverage: float = 0.9


class FakeResult:
    def __init__(self, count):
        self.centers = np.arange(count * 3, dtype=float).reshape(count, 3) / 1000
        self.radii = np.full(count, 0.01)
        self.num_spheres = count
        self.metrics = FakeMetrics()
        self.fit_time_s = 0.1
        self.debug_info = {"used_convex_hull": False}

    @property
    def history(self):
        raise AssertionError("Iteration history is not a final collision model")


def fake_official_api(monkeypatch, count):
    calls = []

    def fit(mesh, **kwargs):
        calls.append((mesh, kwargs))
        return FakeResult(count)

    monkeypatch.setitem(sys.modules, "curobo.sphere_fit", SimpleNamespace(fit_spheres_to_mesh=fit))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    return calls


def test_fitting_uses_one_official_final_result_without_history_or_extra_spheres(monkeypatch):
    module = load_module("fit_motion_collision_model.py")
    calls = fake_official_api(monkeypatch, count=2)
    mesh = trimesh.creation.box(extents=[0.04, 0.03, 0.02])
    mesh.apply_translation([0.2, -0.1, 0.3])
    original_vertices = mesh.vertices.copy()

    result = module.fit_motion_link("test_link", mesh, num_spheres=4, seed=7)

    assert len(calls) == 1
    assert calls[0][1] == {"num_spheres": 4, "iterations": 200, "compute_metrics": True}
    assert result["spheres"] == [
        {"center": [0.0, 0.001, 0.002], "radius": 0.01},
        {"center": [0.003, 0.004, 0.005], "radius": 0.01},
    ]
    assert result["sphere_count"] == 2
    assert result["sphere_budget"] == 4
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(calls[0][0].bounds, mesh.bounds)


def test_fitting_rejects_an_official_result_over_the_link_budget(monkeypatch):
    module = load_module("fit_motion_collision_model.py")
    fake_official_api(monkeypatch, count=3)
    with pytest.raises(ValueError, match="exceeds.*budget"):
        module.fit_motion_link("test_link", trimesh.creation.box(), num_spheres=2, seed=7)


def test_candidate_changes_only_selected_sphere_arrays():
    module = load_module("fit_collision_model.py")
    config = {
        "robot_cfg": {
            "kinematics": {
                "collision_link_names": ["left", "right"],
                "collision_spheres": {"left": [{"center": [0, 0, 0], "radius": 0.1}], "right": []},
                "collision_sphere_buffer": {"left": 0.005},
                "self_collision_ignore": {"left": ["right"]},
                "lock_joints": {"gripper": 0.0},
                "urdf_path": "original.urdf",
            }
        },
        "other_config": {"untouched": True},
    }
    original = deepcopy(config)
    records = {
        "left": {
            "result_source": "official_final",
            "spheres": [{"center": [0.01, 0.02, 0.03], "radius": 0.02}],
        }
    }
    candidate = module.build_candidate(config, records, sphere_budget=2)
    assert config == original
    assert candidate["robot_cfg"]["kinematics"]["collision_spheres"]["left"] == records["left"]["spheres"]
    candidate["robot_cfg"]["kinematics"]["collision_spheres"]["left"] = original["robot_cfg"]["kinematics"][
        "collision_spheres"
    ]["left"]
    assert candidate == original


def test_resume_rejects_history_pipeline_artifacts():
    module = load_module("fit_collision_model.py")
    with pytest.raises(ValueError, match="official_final_v1"):
        module.validate_resume({"api": "SphereFitResult.history"}, {"pipeline": "official_final_v1"})


def test_resume_rejects_changed_visual_bytes_even_when_urdf_is_unchanged():
    module = load_module("fit_collision_model.py")
    selection = {"kind": "all_link_visuals", "visual_count": 1}
    sources = [{"path": "link.obj", "sha256": "original-mesh"}]
    record = {"geometry_sources": sources, "geometry_selection": selection}
    module.validate_record_geometry(record, sources, selection)
    with pytest.raises(ValueError, match="visual geometry differs"):
        module.validate_record_geometry(record, [{"path": "link.obj", "sha256": "edited-mesh"}], selection)
