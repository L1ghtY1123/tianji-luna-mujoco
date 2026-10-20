"""Runtime invariants and reproducible historical geometry checks, without CUDA."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "resources/curobo/marvin_gui_dual.yml"
PREVIOUS_REVIEWED_CONFIG = CONFIG_PATH.parents[2] / "data/collision-avoidance-refit-2026-09-30/source.yml"


def test_web_mounts_the_same_collision_resources_as_the_planner():
    project = CONFIG_PATH.parents[2]
    compose = yaml.safe_load((project / "deploy/web.compose.yaml").read_text(encoding="utf-8"))
    mounts = {item["target"]: item for item in compose["services"]["web"]["volumes"]}
    resource_mount = mounts["/opt/jev/resources"]
    assert resource_mount["source"] == "../resources"
    assert resource_mount["read_only"] is True
    assert resource_mount["bind"]["create_host_path"] is False


@pytest.mark.parametrize("point,maximum_gap_m", [
    ([-0.060, -0.010785, 0.163894], 1e-6),
    ([0.048147314209338946, 0.03799648696755311, 0.1938324096529315], 0.001),
    ([0.0577346, 0.04, 0.110149], 1e-6),
])
def test_previous_model_reproduces_its_audited_visual_mesh_points(point, maximum_gap_m):
    # Historical CAD-fit witnesses remain reproducible, not new obstacle-body design requirements.
    config = yaml.safe_load(PREVIOUS_REVIEWED_CONFIG.read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    point = np.array(point)
    spheres = config["collision_spheres"]["Link_Stand"]
    centers = np.array([sphere["center"] for sphere in spheres])
    radii = np.array([sphere["radius"] for sphere in spheres])
    radii += config["collision_sphere_buffer"].get("Link_Stand", 0.0)
    gap = np.min(np.linalg.norm(centers - point, axis=1) - radii)
    assert gap <= maximum_gap_m, f"Audited support point remains outside planning spheres by {gap * 1000:.3f} mm"


def test_previous_static_model_reproduces_its_reviewed_axis_bounds():
    """Preserve the previous model's result without imposing its CAD-fit policy on new bodies."""
    project = CONFIG_PATH.parents[2]
    spec = importlib.util.spec_from_file_location(
        "collision_fit", project / "scripts/fit_collision_model.py"
    )
    fitting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fitting)
    config = yaml.safe_load(PREVIOUS_REVIEWED_CONFIG.read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    for link in ("Link_Stand", "Link_Base"):
        mesh, _, _ = fitting.load_visual_mesh(project, link)
        spheres = config["collision_spheres"][link]
        centers = np.array([sphere["center"] for sphere in spheres])
        radii = np.array([sphere["radius"] for sphere in spheres])[:, None]
        # This checks raw geometry. Existing planning buffers have a separate purpose.
        lower = mesh.bounds[0] - (centers - radii).min(axis=0)
        upper = (centers + radii).max(axis=0) - mesh.bounds[1]
        overhang = max(float(lower.max()), float(upper.max()))
        # Static component fitting permits up to 15 mm beyond its own outer shape.
        # This is a geometric envelope bound, not the planner's collision clearance.
        assert overhang <= 0.015001, f"{link}: raw sphere bounds protrude by {overhang * 1000:.3f} mm"


def test_geometry_refit_preserves_buffers_collision_pairs_and_open_gripper_locks():
    project = CONFIG_PATH.parents[2]
    source = project / "data/collision-sphere-source-2026-09-30.yml"
    before = yaml.safe_load(source.read_text(encoding="utf-8"))
    after = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    before["robot_cfg"]["kinematics"].pop("collision_spheres")
    after["robot_cfg"]["kinematics"].pop("collision_spheres")
    assert after == before


def load_correction_replay():
    path = CONFIG_PATH.parents[2] / "scripts/replay_collision_correction.py"
    spec = importlib.util.spec_from_file_location("collision_correction_replay", path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    return replay


def test_previous_correction_remains_reproducible_for_rollback():
    replay = load_correction_replay()
    assert replay.rebuild() == (replay.ARCHIVE / "reviewed-model.yml").read_bytes()


def test_replay_rejects_changed_official_output(tmp_path, monkeypatch):
    replay = load_correction_replay()
    archive = tmp_path / "archive"
    shutil.copytree(replay.ARCHIVE, archive)
    path = archive / "static-fit.json"
    manifest = json.loads(path.read_text())
    manifest["calls"][0]["spheres"][0]["radius"] += 0.01
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(replay, "ARCHIVE", archive)
    with pytest.raises(ValueError, match="Recorded input has changed"):
        replay.rebuild()
