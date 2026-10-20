"""The avoidance audit preserves configuration and detects blocked task space."""

import copy
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_avoidance_model.py"


def load_audit(monkeypatch):
    assert SCRIPT.exists(), "the avoidance audit is not implemented"
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("avoidance_model_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration():
    return {
        "robot_cfg": {
            "load_dynamics": True,
            "kinematics": {
                "base_link": "part",
                "collision_link_names": ["part"],
                "lock_joints": {},
                "collision_spheres": {"part": [{"center": [0, 0, 0], "radius": 0.1}]},
                "self_collision_ignore": {},
                "cspace": {"joint_names": [], "default_joint_position": []},
            },
        }
    }


def test_configuration_check_rejects_changed_ignore_and_missing_link(monkeypatch):
    audit = load_audit(monkeypatch)
    baseline = configuration()
    candidate = copy.deepcopy(baseline)
    candidate["robot_cfg"]["kinematics"]["collision_spheres"]["part"][0]["radius"] = 0.08
    audit.validate_configuration(baseline, candidate)
    candidate["robot_cfg"]["kinematics"]["self_collision_ignore"] = {"part": ["peer"]}
    with pytest.raises(ValueError, match="non-sphere"):
        audit.validate_configuration(baseline, candidate)
    candidate = copy.deepcopy(baseline)
    candidate["robot_cfg"]["kinematics"]["collision_spheres"] = {}
    with pytest.raises(ValueError, match="sphere links"):
        audit.validate_configuration(baseline, candidate)


def test_body_envelope_reports_missing_end_and_outward_air_without_quality_verdict(monkeypatch):
    audit = load_audit(monkeypatch)
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    result = audit.axis_envelope(mesh, np.array([[0.02, 0, 0, 0.04]]))
    np.testing.assert_allclose(result["missing_extent_m"], [[0.03, 0.01, 0.01], [0, 0.01, 0.01]])
    np.testing.assert_allclose(result["outward_extent_m"], [[0, 0, 0], [0.01, 0, 0]])
    assert "passed" not in result


def test_target_box_channel_detects_an_air_sphere_between_open_fingers(monkeypatch):
    audit = load_audit(monkeypatch)
    fingers = np.array([[-0.04, 0, 0.1, 0.01], [0.04, 0, 0.1, 0.01]])
    target_from_tcp = np.eye(4)
    target_from_tcp[2, 3] = -0.1
    np.testing.assert_allclose(
        audit.target_box_clearances(fingers, target_from_tcp, [0.04] * 3), [0.01, 0.01]
    )
    blocked = np.vstack([fingers, [0, 0, 0.1, 0.015]])
    assert audit.target_box_clearances(blocked, target_from_tcp, [0.04] * 3).min() == pytest.approx(-0.035)


def test_target_box_channel_uses_full_rigid_frame_not_world_axis_assumption(monkeypatch):
    audit = load_audit(monkeypatch)
    target_from_tcp = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    spheres = np.array([[0.06, 0, 0, 0.01]])
    # Rotating the sphere to target Y puts it 50 mm beyond the 10 mm half extent.
    assert audit.target_box_clearances(spheres, target_from_tcp, [0.2, 0.02, 0.02])[0] == pytest.approx(0.04)


def test_scene_pose_uses_mujoco_wxyz_quaternion(monkeypatch):
    audit = load_audit(monkeypatch)
    transform = audit.obstacle_transform(
        {"position": [1, 2, 3], "quaternion": [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]}
    )
    np.testing.assert_allclose(transform @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)


def test_historical_gzip_input_keeps_defaults_and_requires_all_unlocked_joints(monkeypatch, tmp_path):
    audit = load_audit(monkeypatch)
    config = configuration()["robot_cfg"]["kinematics"]
    config["cspace"] = {"joint_names": ["joint"], "default_joint_position": [0.2]}
    path = tmp_path / "poses.json.gz"
    payload = {
        "poses": [{"name": "recorded", "joint_positions_rad": {"joint": 0.3}}],
        "table": {"position": [1, 2, 3], "size": [1, 1, 0.1]},
    }
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    poses, table = audit.read_audit_poses(path, config)
    assert [p["name"] for p in poses] == ["default", "zero", "recorded"]
    assert table == payload["table"]
    payload["poses"][0]["joint_positions_rad"] = {}
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="unlocked"):
        audit.read_audit_poses(path, config)


def test_visual_world_check_detects_real_mesh_collision_even_when_spheres_are_small(monkeypatch):
    import xml.etree.ElementTree as ET

    audit = load_audit(monkeypatch)
    config = configuration()["robot_cfg"]["kinematics"]
    config["collision_spheres"]["part"][0]["radius"] = 0.001
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    scene = {
        "target_id": "held",
        "table": {"position": [0, 0, -0.055], "size": [0.2, 0.2, 0.04]},
        "objects": [{"id": "held", "kind": "cube", "size": [0.2] * 3, "T_world_object": np.eye(4).tolist()}],
    }
    result = audit.audit_visual_world(
        ET.fromstring('<robot><link name="part"/></robot>'),
        config,
        [{"name": "zero", "joint_positions_rad": {}}],
        scene,
        {"part": mesh},
        samples=20,
    )
    witness = result["poses"][0]["nearest"]
    assert witness["obstacle_id"] == "table"
    assert witness["clearance_m"] < -0.014
    assert result["excluded_target_id"] == "held"


def test_cli_compares_every_link_without_changing_models_or_claiming_full_coverage(monkeypatch, tmp_path):
    load_audit(monkeypatch)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot><link name="part"><visual><geometry><box size=".1 .1 .1"/></geometry></visual></link></robot>'
    )
    baseline = tmp_path / "source.yml"
    candidate = tmp_path / "candidate.yml"
    baseline.write_text(yaml.safe_dump(configuration()))
    candidate.write_bytes(baseline.read_bytes())
    before = {p: p.read_bytes() for p in [urdf, baseline, candidate]}
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--config",
            str(candidate),
            "--urdf",
            str(urdf),
            "--samples",
            "20",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["non_sphere_configuration_unchanged"] is True
    assert list(report["links"]) == ["part"]
    assert report["pose_count"] == 2
    assert report["quality_verdict"] == "requires_geometric_and_task_review"
    assert all(p.read_bytes() == value for p, value in before.items())


def test_running_model_default_and_zero_preserve_self_and_table_clearance(monkeypatch):
    import xml.etree.ElementTree as ET

    audit = load_audit(monkeypatch)
    config = yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_text())["robot_cfg"][
        "kinematics"
    ]
    robot = ET.parse(audit.DEFAULT_URDF).getroot()
    poses, table = audit.read_audit_poses(None, config)
    for pose in poses:
        result = audit.audit_pose(robot, config, pose["joint_positions_rad"], table)
        assert not result["self_collision"]["overlapping_link_pairs"], pose["name"]
        assert result["table"]["minimum_clearance_m"] >= 0.01, pose["name"]
