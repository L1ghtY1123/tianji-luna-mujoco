"""Exact replay of public static fit evidence, never edited sphere parameters."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_static_collision_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("static_fit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inverse_fit_frame_changes_only_coordinates():
    module = load_module()
    transform = np.array(
        [[0.0, -1.0, 0.0, 0.03], [1.0, 0.0, 0.0, -0.02], [0.0, 0.0, 1.0, 0.04], [0.0, 0.0, 0.0, 1.0]]
    )
    centers = np.array([[0.02, 0.08, 0.05], [-0.01, 0.01, 0.06]])
    radii = np.array([0.007123456789, 0.013987654321])
    original_centers, original_radii = centers.copy(), radii.copy()
    result = module.centers_in_link_frame(centers, transform)
    np.testing.assert_allclose(result @ transform[:3, :3].T + transform[:3, 3], centers, atol=1e-15)
    np.testing.assert_array_equal(centers, original_centers)
    np.testing.assert_array_equal(radii, original_radii)
    transform[0, 0] = 2.0
    with pytest.raises(ValueError, match="rigid"):
        module.centers_in_link_frame(centers, transform)


def test_replay_uses_iteration_and_sphere_id_not_row_position():
    module = load_module()
    source = [{"center": [0.0, 0.0, 0.3], "radius": 0.088}]
    original = deepcopy(source)
    outputs = {
        "rib": {
            "centers": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "radii": np.array([0.010987654321, 0.017123456789]),
            "iterations": np.array([15, 10]),
            "sphere_indices": np.array([2, 0]),
        }
    }
    selections = [
        {"source": "original_source", "index": 0},
        {"source": "official_history", "call": "rib", "iteration": 10, "sphere_index": 0},
    ]
    result = module.replay_spheres(source, outputs, selections)
    assert result == [source[0], {"center": [0.4, 0.5, 0.6], "radius": 0.017123456789}]
    assert source == original
    result[0]["center"][2] = -5.0
    assert source == original
    selections[1]["iteration"] = 11
    with pytest.raises(ValueError, match="Missing official"):
        module.replay_spheres(source, outputs, selections)


def test_replay_rejects_duplicate_public_history_identity():
    module = load_module()
    outputs = {
        "part": {
            "centers": np.zeros((2, 3)),
            "radii": np.ones(2),
            "iterations": np.array([5, 5]),
            "sphere_indices": np.array([0, 0]),
        }
    }
    with pytest.raises(ValueError, match="Duplicate"):
        module.replay_spheres(
            [], outputs, [{"source": "official_history", "call": "part", "iteration": 5, "sphere_index": 0}]
        )


def test_static_archive_rebuilds_accepted_spheres_without_curobo(tmp_path):
    import yaml

    module = load_module()
    archive = json.loads((ROOT / module.ARCHIVE_PATH / "manifest.json").read_text())
    config = yaml.safe_load((ROOT / archive["source_config"]).read_text())
    for link in ["Link_Base", "Link_Stand"]:
        source = config["robot_cfg"]["kinematics"]["collision_spheres"][link]
        original = deepcopy(source)
        record = module.fit_static_link(ROOT, link, source, tmp_path / link, 27)
        expected = archive["links"][link]
        assert len(record["spheres"]) == expected["sphere_count"]
        assert module.sphere_digest(record["spheres"]) == expected["sphere_sha256"]
        assert record["generation_mode"] == "archived_public_fit_replay"
        assert record["requested_seed"] == 27 and record["seed_used_for_replay"] is None
        assert len(record["provenance"]) == len(record["spheres"])
        assert source == original
        if link == "Link_Base":
            body = [s for s in record["spheres"] if s["center"][:2] == [0.0, 0.0] and s["radius"] == 0.088]
            assert body == source[149:164]
        changed = deepcopy(source)
        changed[0]["radius"] += 0.001
        with pytest.raises(ValueError, match="source spheres"):
            module.fit_static_link(ROOT, link, changed, tmp_path / "changed", 27)


def test_replay_rejects_corrupt_evidence_before_output(tmp_path):
    module = load_module()
    path = tmp_path / "evidence.npz"
    path.write_bytes(b"not the expected evidence")
    with pytest.raises(ValueError, match="hash"):
        module.checked_file(tmp_path, {"path": "evidence.npz", "sha256": "0" * 64})


def test_archived_input_is_one_component_and_keeps_frame_transform():
    module = load_module()
    archive = ROOT / module.ARCHIVE_PATH
    manifest = json.loads((archive / "manifest.json").read_text())
    main_block = manifest["calls"]["Link_Stand:component19"]
    with np.load(module.checked_file(archive, main_block["input"])) as mesh:
        np.testing.assert_allclose(mesh["vertices"].min(axis=0), [-0.06, -0.04, 0.02], atol=1e-8)
        np.testing.assert_allclose(mesh["vertices"].max(axis=0), [0.06, 0.04, 0.2], atol=1e-8)
    for name in ["Link_Base:rotated_component22", "Link_Base:rotated_component25"]:
        call = manifest["calls"][name]
        with np.load(module.checked_file(archive, call["input"])) as mesh:
            vertices = module.centers_in_link_frame(mesh["vertices"], call["fit_frame_from_link"])
        assert np.ptp(vertices[:, 1]) == pytest.approx(0.008002, abs=1e-8)
        assert np.ptp(vertices[:, 2]) == pytest.approx(0.35, abs=1e-8)
        assert call["stored_output_frame"] == "link"
