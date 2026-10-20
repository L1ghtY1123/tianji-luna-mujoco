"""Mesh containment and the complete measured holding-frame regression, without CUDA."""

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CONFIG = ROOT / "resources/curobo/marvin_gui_dual.yml"
RECORD = ROOT / "resources/curobo/link4-refit-20261016.json"


def array(rows):
    return np.array([[*row["center"], row["radius"]] for row in rows])


def minimum_gap(first, second):
    return (np.linalg.norm(first[:, None, :3] - second[None, :, :3], axis=-1)
            - first[:, None, 3] - second[None, :, 3]).min()


def test_measured_attachment_false_collision_is_removed_without_shrinking_attachment():
    record = json.loads(RECORD.read_text())
    config = yaml.safe_load(CONFIG.read_bytes())["robot_cfg"]["kinematics"]
    attachment = np.asarray(record["regression"]["attachment_spheres_m"])
    previous = array(record["source_link_spheres"]["gripL_Link4_R"])
    current = array(config["collision_spheres"]["gripL_Link4_R"])
    assert minimum_gap(previous, attachment) < -.003
    assert minimum_gap(current, attachment) > 0.
    assert config["self_collision_buffer"].get("gripL_Link4_R", 0.) == 0.


def test_refit_changes_only_four_mirrored_link4_lists_and_stays_within_budget():
    from replay_transfer_spheres import rebuild

    previous = yaml.safe_load(rebuild(ROOT / "data/next-revision-audit-20261012/spheres",
                                     recipe_name="recipe-v6.json"))
    current = yaml.safe_load(CONFIG.read_bytes())
    before = previous["robot_cfg"]["kinematics"].pop("collision_spheres")
    after = current["robot_cfg"]["kinematics"].pop("collision_spheres")
    assert previous == current
    assert {link for link in before if before[link] != after[link]} == {
        f"grip{hand}_Link4_{side}" for hand in "LR" for side in "LR"
    }
    assert sum(map(len, after.values())) == 349
    for side in "LR":
        assert after[f"gripL_Link4_{side}"] == after[f"gripR_Link4_{side}"]
        assert len(after[f"gripL_Link4_{side}"]) == 14
        assert max(sphere["radius"] for sphere in after[f"gripL_Link4_{side}"]) < .020


@pytest.mark.parametrize("side", ["L", "R"])
def test_complete_collision_volume_and_visual_surface_are_covered(side):
    from audit_link4_coverage import certify_cells

    record = json.loads(RECORD.read_text())
    for name, expected in record["geometry_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    config = yaml.safe_load(CONFIG.read_bytes())["robot_cfg"]["kinematics"]
    link = f"gripL_Link4_{side}"
    spheres = array(config["collision_spheres"][link])
    mesh_root = ROOT / "assets/marvin_urdf/meshes"
    collision = trimesh.load_mesh(mesh_root / "collision" / f"{link}.stl", process=True)
    visual = trimesh.load_mesh(mesh_root / "visual" / f"{link}.obj", process=True)
    for kind, suffix, reference in (("collision", "stl", collision), ("visual", "obj", visual)):
        peer = trimesh.load_mesh(mesh_root / kind / f"gripR_Link4_{side}.{suffix}", process=True)
        np.testing.assert_array_equal(peer.triangles, reference.triangles)
    assert collision.is_convex and collision.is_watertight and collision.is_winding_consistent
    anchor = collision.center_mass
    assert np.max(np.sum((anchor - collision.triangles_center) * collision.face_normals, axis=1)) <= 0
    tetrahedra = np.concatenate([np.broadcast_to(anchor, (len(collision.faces), 1, 3)),
                                collision.triangles], axis=1)
    for cells in (collision.triangles, tetrahedra, visual.triangles):
        certificate = certify_cells(cells, spheres)
        assert certificate["complete"], certificate


def test_recorded_cpu_fit_reproduces_the_configuration():
    from refit_link4_collision import build_candidate

    config = yaml.safe_load(CONFIG.read_bytes())
    reproduced = build_candidate(config, json.loads(RECORD.read_text()))
    for side in "LR":
        link = f"gripL_Link4_{side}"
        np.testing.assert_allclose(array(config["robot_cfg"]["kinematics"]["collision_spheres"][link]),
                                   array(reproduced["robot_cfg"]["kinematics"]["collision_spheres"][link]),
                                   rtol=0, atol=1e-8)  # 10 nm: allow cross-platform optimizer rounding, not geometry changes.


def test_raw_open_gripper_link4_pairs_do_not_intersect_before_feedback_derivation():
    from audit_collision_model import forward_kinematics, sphere_modes

    config = yaml.safe_load(CONFIG.read_bytes())["robot_cfg"]["kinematics"]
    assert config["lock_joints"]["gripL_finger_joint"] == 0.
    assert config["lock_joints"]["gripR_finger_joint"] == 0.
    transforms = forward_kinematics(ET.parse(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf").getroot(),
                                   config, {})
    for hand in "LR":
        pair = []
        for side in "LR":
            link = f"grip{hand}_Link4_{side}"
            sphere = sphere_modes(config, link)["self_effective"]
            transform = transforms[link]
            sphere[:, :3] = sphere[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            pair.append(sphere)
        # Raw templates must work for offline WorkspaceIK too. Its gripper
        # geometry does not use the held-feedback Link4 opposing-pair exemption.
        assert minimum_gap(*pair) > 0.


def test_postprocessing_cannot_publish_intersection_or_radius_overflow():
    from refit_link4_collision import validate_final_template

    record = json.loads(RECORD.read_text())
    spheres = array(record["template_spheres"])
    attachment = np.asarray(record["regression"]["attachment_spheres_m"])
    generation = record["generation"]
    validate_final_template(spheres, attachment, generation)
    enlarged = spheres.copy()
    enlarged[0, 3] = generation["split_until_radius_m"] + 1e-6
    with pytest.raises(ValueError, match="radius"):
        validate_final_template(enlarged, attachment, generation)
    overlapping = spheres.copy()
    overlapping[0, :3] = attachment[0, :3]
    with pytest.raises(ValueError, match="attachment"):
        validate_final_template(overlapping, attachment, generation)
    plane = generation["open_gripper_base_halfspace"]
    crossing = spheres.copy()
    crossing[:, :3] -= np.asarray(plane["normal"]) * .2
    with pytest.raises(ValueError, match="midplane"):
        validate_final_template(crossing, attachment, generation)


@pytest.mark.parametrize("driver", [0., .4, .8])
def test_closed_gripper_derivation_retains_all_link4_spheres_and_attachment_checks(driver):
    from jev_mujoco.carrying import (
        CarryingBinding,
        configure_gripper_feedback,
        create_carrying_robot_config,
    )

    config = yaml.safe_load(CONFIG.read_bytes())
    locked = configure_gripper_feedback(config, {"A": driver, "B": driver},
                                        ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf")
    scene = {"scene_token": "scene", "observation_id": "rgbd", "generation": 1, "target_id": "source"}
    binding = CarryingBinding.from_mapping(
        {"arm": "A", "owner_id": "held", "scene_token": "scene", "observation_id": "rgbd",
         "generation": 1, "instance_id": "source", "frame": "tool",
         "state": "held", "points_tool_m": [[0., 0., .15], [.01, .01, .17]],
         "uncertainty_m": .005, "coverage": .6}, scene=scene, arm="A")
    carried, attached_link = create_carrying_robot_config(locked, binding)
    kin = carried["robot_cfg"]["kinematics"]
    for hand in "LR":
        for side in "LR":
            link = f"grip{hand}_Link4_{side}"
            assert kin["collision_spheres"][link] == config["robot_cfg"]["kinematics"]["collision_spheres"][link]
            assert len(kin["collision_spheres"][link]) == 14
            assert attached_link not in kin["self_collision_ignore"].get(link, [])
            assert link not in kin["self_collision_ignore"][attached_link]
    assert set(kin["self_collision_ignore"][attached_link]) == {"gripL_Link3_L", "gripL_Link3_R"}
