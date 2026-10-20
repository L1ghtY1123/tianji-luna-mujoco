"""Analytic checks for the offline model audit, without CUDA or mesh repair."""

import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_collision_model.py"


def load_audit():
    assert SCRIPT.is_file(), "the CPU collision model audit is not implemented"
    spec = importlib.util.spec_from_file_location("collision_model_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sphere_gap_keeps_signed_distance_and_buffer_semantics():
    audit = load_audit()
    spheres = np.array([[0., 0., 0., 1.]])
    points = np.array([[0., 0., 0.], [1., 0., 0.], [1.5, 0., 0.]])
    np.testing.assert_allclose(audit.sphere_gaps(points, spheres), [-1., 0., .5])
    config = {"collision_spheres": {"part": [{"center": [0, 0, 0], "radius": 1.}]},
              "collision_sphere_buffer": {"part": .2}, "self_collision_buffer": {"part": -.1}}
    modes = audit.sphere_modes(config, "part")
    assert modes["raw"][0, 3] == 1.
    assert modes["world_buffered"][0, 3] == pytest.approx(1.2)
    assert modes["self_effective"][0, 3] == pytest.approx(1.1)


def test_box_sdf_handles_outside_corners_and_inside_points():
    audit = load_audit()
    points = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [1, 0, 0]])
    np.testing.assert_allclose(audit.box_sdf(points, [0, 0, 0], [2, 2, 2]),
                               [-1., 1., np.sqrt(2), 0.])


def test_visual_scale_is_applied_before_origin_rotation_and_translation(tmp_path):
    audit = load_audit()
    (tmp_path / "part.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot><link name="part"><visual><origin xyz="1 2 3" '
                    'rpy="0 0 1.5707963267948966"/><geometry><mesh filename="part.obj" '
                    'scale="2 3 4"/></geometry></visual></link></robot>')
    meshes, assets = audit.load_visual_meshes(urdf)
    actual = meshes["part"].vertices
    expected = np.array([[1, 2, 3], [1, 4, 3], [-2, 2, 3]])
    for point in expected:
        assert np.min(np.linalg.norm(actual - point, axis=1)) < 1e-12
    assert assets[0]["sha256"]


def test_static_component_filter_preserves_low_central_geometry_and_rejects_unknown_high_parts():
    audit = load_audit()
    central = trimesh.creation.box(extents=[.1, .1, .02])
    central.apply_translation([0, 0, .02])
    foot = trimesh.creation.box(extents=[.04, .04, .02])
    foot.apply_translation([.3, 0, .02])
    mesh = trimesh.util.concatenate([central, foot])
    retained, classification = audit.retain_central_base(mesh)
    assert classification["retained_components"] == 1
    assert classification["excluded_components"] == 1
    np.testing.assert_allclose(retained.bounds, [[-.05, -.05, .01], [.05, .05, .03]])
    foot.apply_translation([0, 0, .2])
    with pytest.raises(ValueError, match="unclassified"):
        audit.retain_central_base(trimesh.util.concatenate([central, foot]))


def test_current_base_components_have_the_reviewed_static_boundary():
    audit = load_audit()
    mesh = trimesh.load_mesh(ROOT / "assets/marvin_urdf/meshes/visual/Link_Base.obj", process=False)
    retained, classification = audit.retain_central_base(mesh)
    assert classification["component_count"] == 46
    assert classification["retained_components"] == 15
    assert classification["excluded_components"] == 31
    assert retained.bounds[0, 2] == pytest.approx(.002)
    assert retained.bounds[1, 2] == pytest.approx(.981)


def test_fk_uses_parent_joint_transform_mimic_and_configured_open_lock():
    audit = load_audit()
    robot = ET.fromstring('''<robot><link name="base"/><link name="arm"/><link name="finger"/>
      <joint name="turn" type="revolute"><parent link="base"/><child link="arm"/>
      <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="finger_slide" type="prismatic"><parent link="arm"/><child link="finger"/>
      <axis xyz="1 0 0"/><mimic joint="driver" multiplier="2" offset=".1"/></joint>
      </robot>''')
    config = {"base_link": "base", "lock_joints": {"driver": 0.}}
    transforms = audit.forward_kinematics(robot, config, {"turn": np.pi / 2, "driver": 1.})
    np.testing.assert_allclose(transforms["finger"][:3, 3], [1., .1, 0.], atol=1e-12)


def test_pose_report_respects_reverse_ignore_and_exact_sphere_table_clearance():
    audit = load_audit()
    robot = ET.fromstring('''<robot><link name="a"/><link name="b"/>
      <joint name="fixed" type="fixed"><parent link="a"/><child link="b"/>
      <origin xyz="1 0 0"/></joint></robot>''')
    config = {"base_link": "a", "lock_joints": {}, "collision_link_names": ["a", "b"],
              "collision_spheres": {name: [{"center": [0, 0, 0], "radius": .6}] for name in ["a", "b"]},
              "self_collision_ignore": {"b": ["a"]}}
    table = {"position": [3, 0, 0], "size": [2, 2, 2]}
    result = audit.audit_pose(robot, config, {}, table)
    assert result["self_collision"]["overlapping_link_pairs"] == []
    assert result["table"]["minimum_clearance_m"] == pytest.approx(.4)
    config["self_collision_ignore"] = {}
    result = audit.audit_pose(robot, config, {}, table)
    pair = result["self_collision"]["overlapping_link_pairs"][0]
    assert pair["gap_m"] == pytest.approx(-.2)
    assert result["self_collision"]["potential_sphere_pairs"] == 1


def test_containment_statistics_do_not_treat_duplicates_or_overlap_as_strict_containment():
    audit = load_audit()
    spheres = np.array([[0, 0, 0, 1], [.1, 0, 0, .2], [0, 0, 0, 1], [1.5, 0, 0, 1.]])
    result = audit.strict_containment(spheres)
    assert [value["sphere_index"] for value in result] == [1]
    assert result[0]["slack_m"] == pytest.approx(.7)


def test_sampling_is_reproducible_and_coverage_includes_all_vertices():
    audit = load_audit()
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    spheres = np.array([[0, 0, 0, 2.]])
    first = audit.audit_surface(mesh, {"raw": spheres}, sample_count=100, seed=42)
    second = audit.audit_surface(mesh, {"raw": spheres}, sample_count=100, seed=42)
    assert first == second
    assert first["raw"]["vertices"]["covered_fraction"] == 1.
    assert first["raw"]["vertices"]["maximum_gap_m"] == 0.
    assert first["raw"]["samples"]["count"] == 100


def test_cli_reads_candidate_and_external_poses_without_rewriting_inputs(tmp_path):
    load_audit()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot><link name="part"><visual><geometry><box size=".1 .1 .1"/>'
                    '</geometry></visual></link></robot>')
    config = tmp_path / "candidate.yml"
    config.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": {
        "base_link": "part", "collision_link_names": ["part"], "lock_joints": {},
        "collision_spheres": {"part": [{"center": [0, 0, 0], "radius": .1}]},
        "cspace": {"joint_names": [], "default_joint_position": []}}}}))
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps([{"name": "recorded", "joint_positions_rad": {}}]))
    before = {path: path.read_bytes() for path in [urdf, config, poses]}
    output = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--output", str(output),
                             "--urdf", str(urdf), "--poses", str(poses)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert [pose["name"] for pose in report["poses"]] == ["default", "zero", "recorded"]
    assert report["links"]["part"]["surface"]["raw"]["vertices"]["maximum_gap_m"] == 0.
    assert all(path.read_bytes() == content for path, content in before.items())
