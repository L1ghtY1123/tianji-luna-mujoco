"""Frozen fit outputs must reconstruct a model without changing planning settings."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_replay():
    path = ROOT / "scripts/replay_official_collision.py"
    assert path.exists(), "Official replacement replay is not implemented"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("official_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_archive(path):
    old = {"center": [0.0, 0.0, 0.0], "radius": 0.1}
    new = {"center": [0.0, 0.0, 0.0], "radius": 0.02}
    source = {
        "robot_cfg": {
            "kinematics": {
                "collision_spheres": {"arm": [old]},
                "lock_joints": {"grip": 0.0},
            }
        }
    }
    manifest = {
        "schema_version": 1,
        "sdk_commit": "8e734f3ced1df898990bcd92de40abce475907db",
        "links": {
            "arm": {
                "spheres": [new],
                "parts": [
                    {
                        "name": "body",
                        "method": "MORPHIT",
                        "result_source": "official_final",
                        "input_geometry": {"scope_id": "whole_link"},
                        "parameters": {"num_spheres": 1},
                        "spheres": [new],
                    }
                ],
            }
        },
    }
    (path / "source.yml").write_text(yaml.safe_dump(source, sort_keys=False))
    manifest["source_config_sha256"] = hashlib.sha256((path / "source.yml").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    source["robot_cfg"]["kinematics"]["collision_spheres"]["arm"] = [new]
    expected = yaml.safe_dump(source, sort_keys=False).encode()
    recipe = {
        "source": "source.yml",
        "manifest": "manifest.json",
        "archive_files": {
            name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ("source.yml", "manifest.json")
        },
        "model_sha256": hashlib.sha256(expected).hexdigest(),
    }
    (path / "recipe.json").write_text(json.dumps(recipe))
    return expected


def test_replay_accepts_tighter_geometry_and_preserves_other_settings(tmp_path):
    expected = make_archive(tmp_path)
    assert load_replay().rebuild(tmp_path) == expected


def test_replay_rejects_modified_frozen_output(tmp_path):
    make_archive(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Recorded input has changed"):
        load_replay().rebuild(tmp_path)


def test_replay_rejects_an_unrecorded_assembled_sphere(tmp_path):
    make_archive(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["links"]["arm"]["spheres"][0]["radius"] = 0.04
    path.write_text(json.dumps(manifest))
    recipe_path = tmp_path / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["archive_files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    recipe_path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="unexplained spheres"):
        load_replay().rebuild(tmp_path)


def test_replay_rejects_a_different_source_in_manifest(tmp_path):
    make_archive(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_config_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    recipe_path = tmp_path / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["archive_files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    recipe_path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="Manifest source"):
        load_replay().rebuild(tmp_path)


def test_recorded_official_outputs_reproduce_the_symmetry_source_model():
    assert load_replay().rebuild() == (ROOT / "data/collision-symmetry-2026-10-08/source.yml").read_bytes()
