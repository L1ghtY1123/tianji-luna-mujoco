"""Independent checks for the reviewed, final-only collision fitting workflow."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/audit_official_collision.py"


def load_audit():
    assert SCRIPT.exists(), "The official collision audit has not been implemented"
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("official_collision_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sphere(radius=0.01):
    return {"center": [0.0, 0.0, 0.0], "radius": radius}


def official_part(spheres, scope="whole_link"):
    return {
        "name": "body",
        "method": "MORPHIT",
        "result_source": "official_final",
        "input_geometry": {"scope_id": scope},
        "parameters": {"num_spheres": len(spheres)},
        "spheres": spheres,
    }


def test_archived_stand_old_and_new_body_are_not_a_replacement():
    """Both real archived sets replay correctly, yet fit the same body twice."""
    archive = PROJECT / "data/collision-correction-2026-09-30"
    original = yaml.safe_load((PROJECT / "data/collision-sphere-source-2026-09-30.yml").read_text())
    source = original["robot_cfg"]["kinematics"]["collision_spheres"]["Link_Stand"]
    manifest = json.loads((archive / "static-fit.json").read_text())
    call = next(item for item in manifest["calls"] if item["label"] == "body_cell_000")
    retained = {"indices": [0], "scope_id": "component:19", "reason": "old body restored"}
    fitted = official_part(call["spheres"], "component:19/cell:000")
    balls = call["spheres"] + [source[0]]
    errors = load_audit().check_link_sources(balls, source, {"parts": [fitted], "retained_source": retained})
    assert any("overlapping geometry scopes" in error for error in errors)


def test_tighter_official_replacement_need_not_contain_the_old_air_envelope():
    source, fitted = [sphere(0.1)], [sphere(0.02)]
    assert load_audit().check_link_sources(fitted, source, {"parts": [official_part(fitted)]}) == []


def test_explicit_unchanged_source_is_allowed_without_refitting():
    source = [sphere()]
    record = {
        "parts": [],
        "retained_source": {
            "indices": [0],
            "scope_id": "whole_link",
            "reason": "reviewed simple one-sphere geometry",
        },
    }
    assert load_audit().check_link_sources(source, source, record) == []


@pytest.mark.parametrize("change", ["edited_radius", "history", "patch", "duplicate_scope"])
def test_rejects_unexplained_or_accumulated_outputs(change):
    fitted = [sphere()]
    record = {"parts": [official_part(copy.deepcopy(fitted))]}
    if change == "edited_radius":
        fitted[0]["radius"] += 0.001
    elif change == "history":
        record["parts"][0]["result_source"] = "history"
    elif change == "patch":
        record["parts"][0]["input_geometry"]["selection"] = "gap_patch"
    else:
        record["parts"] *= 2
        fitted *= 2
    assert load_audit().check_link_sources(fitted, [], record)


def test_rigid_input_rotation_must_reproduce_centres_and_keep_radii():
    output = [{"center": [0.0, 1.0, 0.0], "radius": 0.1}]
    part = official_part(output)
    part["input_geometry"]["link_to_fit_rotation"] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    part["official_final_in_fit_frame"] = [{"center": [-1.0, 0.0, 0.0], "radius": 0.1}]
    audit = load_audit()
    assert audit.check_link_sources(output, [], {"parts": [part]}) == []
    part["spheres"][0]["radius"] += 0.001
    assert any("official final" in error for error in audit.check_link_sources(output, [], {"parts": [part]}))


@pytest.mark.parametrize("invalid", ["translation", "radius", "bottom_row", "reflection", "scale"])
def test_static_fit_transform_rejects_nonrigid_or_edited_final(invalid):
    output = [{"center": [0.0, 1.0, 0.0], "radius": 0.1}]
    part = official_part(output)
    part["fit_from_link"] = [
        [0.0, -1.0, 0.0, 0.3],
        [1.0, 0.0, 0.0, -0.2],
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 0.0, 0.0, 1.0],
    ]
    part["final_centers_fit_frame_m"] = [[-0.7, -0.2, 0.1]]
    part["final_radii_m"] = [0.1]
    audit = load_audit()
    assert audit.check_link_sources(output, [], {"parts": [part]}) == []
    if invalid == "translation":
        part["fit_from_link"][0][3] += 0.001
    elif invalid == "radius":
        part["final_radii_m"][0] += 0.001
    elif invalid == "bottom_row":
        part["fit_from_link"][3][0] = 0.1
    elif invalid == "reflection":
        part["fit_from_link"][2][2] = -1.0
    else:
        part["fit_from_link"][0][1] = -2.0
    assert audit.check_link_sources(output, [], {"parts": [part]})


def test_motion_reflection_is_not_a_rotation():
    output = [sphere()]
    part = official_part(output)
    part["input_geometry"]["link_to_fit_rotation"] = np.diag([1.0, 1.0, -1.0]).tolist()
    part["official_final_in_fit_frame"] = copy.deepcopy(output)
    assert load_audit().check_link_sources(output, [], {"parts": [part]})


def test_source_and_final_order_does_not_require_rewriting_the_candidate():
    original, replacement = sphere(0.02), sphere(0.01)
    record = {
        "parts": [official_part([replacement], "component:new")],
        "retained_source": {
            "indices": [0],
            "scope_id": "component:old",
            "reason": "reviewed separate component",
        },
    }
    assert load_audit().check_link_sources([original, replacement], [original], record) == []
    assert load_audit().check_link_sources([original, replacement, replacement], [original], record)


def test_complete_surface_report_preserves_unproven_samples():
    mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    points = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    balls = np.array([[0.0, 0.0, 0.0, 0.005]])
    result = load_audit().surface_gap_report(mesh, points, np.array([True, False]), balls)
    assert result["all_area_samples"]["maximum_mm"] == pytest.approx(5.0)
    assert result["visibility_unproven_samples"] == 1
    assert result["all_vertices"]["maximum_mm"] == pytest.approx((np.sqrt(3) * 0.01 - 0.005) * 1000)
    assert result["geometry_verified"] is False


def test_non_sphere_settings_are_checked_independently():
    source = {
        "robot_cfg": {"kinematics": {"collision_spheres": {"arm": [sphere()]}, "lock_joints": {"grip": 0.0}}}
    }
    candidate = copy.deepcopy(source)
    candidate["robot_cfg"]["kinematics"]["lock_joints"]["grip"] = 0.5
    report = load_audit().audit_provenance(candidate, source, {"schema_version": 1, "links": {}})
    assert any("non-sphere" in error for error in report["errors"])
    assert report["geometry_verified"] is False
