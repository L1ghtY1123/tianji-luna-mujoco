"""Reject changed geometry and non-final fitting results before replay."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_fitting_input_requires_the_frozen_geometry_hash(tmp_path):
    from refit_transfer_spheres import load_job_mesh

    path = tmp_path / "part.ply"
    trimesh.creation.box().export(path)
    job = {"mesh": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert len(load_job_mesh(tmp_path, job).faces) == 12
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_job_mesh(tmp_path, job)


def test_fitting_input_cannot_escape_its_archive(tmp_path):
    from refit_transfer_spheres import load_job_mesh

    with pytest.raises(ValueError, match="archive"):
        load_job_mesh(tmp_path, {"mesh": "../outside.ply", "sha256": "0" * 64})


def test_public_final_conversion_rejects_fallback_and_preserves_radii():
    from refit_transfer_spheres import convert_final

    result = SimpleNamespace(centers=np.array([[0.1, 0.2, 0.3]]), radii=np.array([0.04]),
                             num_spheres=1, debug_info={"fallback_used": False})
    assert convert_final(result, np.eye(4), 2) == [{"center": [0.1, 0.2, 0.3], "radius": 0.04}]
    result.debug_info["fallback_used"] = True
    with pytest.raises(ValueError, match="fallback"):
        convert_final(result, np.eye(4), 2)


def test_historical_293_refit_replays_and_preserves_every_non_sphere_setting():
    from replay_transfer_spheres import ARCHIVE, rebuild

    payload = rebuild()
    assert payload == (ROOT / "data/next-revision-audit-20261012/spheres/source.yml").read_bytes()
    baseline = yaml.safe_load((ARCHIVE / "source.yml").read_bytes())
    current = yaml.safe_load(payload)
    old = baseline["robot_cfg"]["kinematics"].pop("collision_spheres")
    new = current["robot_cfg"]["kinematics"].pop("collision_spheres")
    assert baseline == current
    assert set(old) == set(new)
    assert sum(map(len, new.values())) == 293
    assert {link for link in old if old[link] != new[link]} == {
        "Link_Base", "Link_Stand", "Link4_L", "Link4_R", "TCP_Link_L", "TCP_Link_R",
    }


def test_historical_309_refit_replays_within_350_budget_and_preserves_all_other_settings():
    from replay_transfer_spheres import rebuild

    archive = ROOT / "data/next-revision-audit-20261012/spheres"
    payload = rebuild(archive, recipe_name="recipe-v6.json")
    record = json.loads((ROOT / "resources/curobo/link4-refit-20261016.json").read_text())
    assert hashlib.sha256(payload).hexdigest() == record["source_model_sha256"]
    baseline = yaml.safe_load((archive / "source.yml").read_bytes())
    current = yaml.safe_load(payload)
    old = baseline["robot_cfg"]["kinematics"].pop("collision_spheres")
    new = current["robot_cfg"]["kinematics"].pop("collision_spheres")
    assert baseline == current
    assert set(old) == set(new)
    assert sum(map(len, new.values())) == 309
    assert sum(map(len, new.values())) <= 350
    assert {link for link in old if old[link] != new[link]} == {
        "Link_Base", "Link_Stand", "Link2_L", "Link2_R", "Link4_L", "Link4_R", "TCP_Link_L", "TCP_Link_R",
        "gripL_Base", "gripR_Base",
    }
    for link in old:
        if link.startswith("grip") and "_Link" in link:
            assert new[link] == old[link]


def test_refit_replay_rejects_total_robot_budget_overflow():
    from replay_transfer_spheres import apply_refits

    sphere = {"center": [0, 0, 0], "radius": .01}
    source = {"robot_cfg": {"kinematics": {"collision_spheres": {"part": [sphere] * 351}}}}
    with pytest.raises(ValueError, match="350"):
        apply_refits(source, {}, [])


def test_refit_replay_rejects_manual_radius_edits():
    from replay_transfer_spheres import ARCHIVE, apply_refits

    source = yaml.safe_load((ARCHIVE / "source.yml").read_bytes())
    manifest = json.loads((ARCHIVE / "round4/manifest.json").read_text())
    changed = copy.deepcopy(manifest)
    changed["jobs"]["Link_Base_lower_body"]["spheres"][0]["radius"] += 0.001
    with pytest.raises(ValueError, match="official final"):
        apply_refits(source, {"round4": changed}, [])
