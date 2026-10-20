"""Shared geometry must retain radii, preserve settings and replay deterministically."""

import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def source_config():
    return {"robot_cfg": {"kinematics": {
        "collision_spheres": {
            "left": [{"center": [0.01, 0.02, 0.03], "radius": 0.004}],
            "right": [{"center": [0.0, 0.0, 0.0], "radius": 0.008}],
        },
        "lock_joints": {"gripper": 0.0},
        "self_collision_ignore": {"left": ["right"]},
        "collision_sphere_buffer": 0.001,
    }}}


def mapping(transform):
    return [{"target_link": "right", "parts": [{
        "source_link": "left", "indices": [0], "transform": transform,
    }]}]


def test_mirror_preserves_radius_and_maps_the_link_frame():
    from replay_symmetric_collision import apply_templates

    source = source_config()
    original = copy.deepcopy(source)
    transform = np.diag([1.0, -1.0, 1.0, 1.0])
    transform[:3, 3] = [0.1, 0.0, 0.0]
    candidate = apply_templates(source, mapping(transform.tolist()))
    sphere = candidate["robot_cfg"]["kinematics"]["collision_spheres"]["right"][0]
    assert sphere["center"] == pytest.approx([0.11, -0.02, 0.03])
    assert sphere["radius"] == 0.004
    assert source == original
    candidate["robot_cfg"]["kinematics"].pop("collision_spheres")
    original["robot_cfg"]["kinematics"].pop("collision_spheres")
    assert candidate == original


@pytest.mark.parametrize("entry,value", [((0, 0), 2.0), ((0, 1), 0.2), ((3, 0), 0.1)])
def test_rejects_scale_shear_and_non_affine_transforms(entry, value):
    from replay_symmetric_collision import apply_templates

    transform = np.eye(4)
    transform[entry] = value
    with pytest.raises(ValueError, match="isometry"):
        apply_templates(source_config(), mapping(transform.tolist()))


def test_all_templates_read_the_frozen_source_not_previous_replacements():
    from replay_symmetric_collision import apply_templates

    rules = mapping(np.eye(4).tolist())
    rules.append({"target_link": "left", "parts": [{
        "source_link": "right", "indices": [0], "transform": np.eye(4).tolist(),
    }]})
    result = apply_templates(source_config(), rules)["robot_cfg"]["kinematics"]["collision_spheres"]
    assert result["left"][0]["radius"] == 0.008
    assert result["right"][0]["radius"] == 0.004


def test_duplicate_target_is_rejected():
    from replay_symmetric_collision import apply_templates

    rules = mapping(np.eye(4).tolist())
    with pytest.raises(ValueError, match="Duplicate target"):
        apply_templates(source_config(), rules + rules)


def test_one_sphere_can_form_a_symmetric_pair_with_distinct_isometries():
    from replay_symmetric_collision import apply_templates

    rules = mapping(np.eye(4).tolist())
    rules[0]["parts"].append({"source_link": "left", "indices": [0],
                              "transform": np.diag([1.0, -1.0, 1.0, 1.0]).tolist()})
    values = apply_templates(source_config(), rules)["robot_cfg"]["kinematics"]["collision_spheres"]["right"]
    assert [v["radius"] for v in values] == [0.004, 0.004]
    assert [v["center"][1] for v in values] == [0.02, -0.02]


def write_archive(path):
    from replay_symmetric_collision import apply_templates

    source = source_config()
    rules = mapping(np.diag([1.0, -1.0, 1.0, 1.0]).tolist())
    (path / "source.yml").write_bytes(yaml.safe_dump(source, sort_keys=False).encode())
    (path / "rules.json").write_text(json.dumps({"rules": rules}))
    expected = yaml.safe_dump(apply_templates(source, rules), sort_keys=False).encode()
    recipe = {"archive_files": {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("source.yml", "rules.json")
    }, "model_sha256": hashlib.sha256(expected).hexdigest()}
    (path / "recipe.json").write_text(json.dumps(recipe))
    return expected


def test_replay_verifies_frozen_templates_and_the_final_model(tmp_path):
    from replay_symmetric_collision import rebuild

    expected = write_archive(tmp_path)
    assert rebuild(tmp_path) == expected
    (tmp_path / "rules.json").write_text("{}")
    with pytest.raises(ValueError, match="Recorded input has changed"):
        rebuild(tmp_path)


def test_replay_rejects_a_wrong_model_digest(tmp_path):
    from replay_symmetric_collision import rebuild

    write_archive(tmp_path)
    path = tmp_path / "recipe.json"
    recipe = json.loads(path.read_text())
    recipe["model_sha256"] = "0" * 64
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="differs from the reviewed model"):
        rebuild(tmp_path)


def test_rotation_and_translation_follow_the_target_from_source_convention():
    from replay_symmetric_collision import apply_templates

    matrix = [[0, -1, 0, 0.1], [1, 0, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]]
    result = apply_templates(source_config(), mapping(matrix))
    center = result["robot_cfg"]["kinematics"]["collision_spheres"]["right"][0]["center"]
    assert center == pytest.approx([0.08, 0.21, 0.33])


def current_model():
    return yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_bytes())["robot_cfg"]["kinematics"]


def centers(spheres):
    return np.array([sphere["center"] for sphere in spheres])


def test_corresponding_arm_and_gripper_links_share_templates():
    from audit_collision_model import DEFAULT_URDF, forward_kinematics

    config = current_model()
    spheres = config["collision_spheres"]
    for name in ("Base", "Link1", "Link2", "Link3", "Link4", "Link6"):
        assert spheres[f"{name}_L"] == spheres[f"{name}_R"]
    for left in [link for link in spheres if link.startswith("gripL")]:
        assert spheres[left] == spheres[left.replace("gripL", "gripR")]
    robot = ET.parse(DEFAULT_URDF).getroot()
    fk = forward_kinematics(robot, config, {})
    for name in ("Link5", "Link7"):
        pair = [spheres[f"{name}_{side}"] for side in ("L", "R")]
        assert [s["radius"] for s in pair[0]] == [s["radius"] for s in pair[1]]
        world = [centers(values) @ fk[f"{name}_{side}"][:3, :3].T + fk[f"{name}_{side}"][:3, 3]
                 for side, values in zip(("L", "R"), pair)]
        np.testing.assert_allclose(world[0], world[1] * [1, -1, 1], rtol=0, atol=1e-12)


def test_finger_templates_reflect_in_gripper_base_and_cameras_keep_actual_mounts():
    from audit_collision_model import DEFAULT_URDF, forward_kinematics, origin_transform

    config = current_model()
    spheres = config["collision_spheres"]
    robot = ET.parse(DEFAULT_URDF).getroot()
    fk = forward_kinematics(robot, config, {})
    for number in range(1, 5):
        positions = []
        for side in ("L", "R"):
            name = f"gripL_Link{number}_{side}"
            matrix = np.linalg.inv(fk["gripL_Base"]) @ fk[name]
            positions.append(centers(spheres[name]) @ matrix[:3, :3].T + matrix[:3, 3])
        np.testing.assert_allclose(positions[1], positions[0] * [1, -1, 1], rtol=0, atol=1e-12)
    mount_count = len(spheres["TCP_Link_L"]) - 2
    for visual_index, selection in ((0, slice(0, mount_count)), (1, slice(mount_count, None))):
        positions = []
        for side in ("L", "R"):
            name = f"TCP_Link_{side}"
            visual = robot.find(f"link[@name='{name}']").findall("visual")[visual_index]
            matrix = np.linalg.inv(origin_transform(visual))
            positions.append(centers(spheres[name][selection]) @ matrix[:3, :3].T + matrix[:3, 3])
        np.testing.assert_allclose(positions[0], positions[1], rtol=0, atol=1e-12)


def test_current_central_parts_preserve_requested_rotation_and_reflection_symmetry():
    from scipy.spatial.transform import Rotation

    spheres = current_model()["collision_spheres"]
    transforms = [
        ("Link_Base", slice(0, 10), Rotation.from_euler("z", 120, degrees=True).as_matrix()),
        ("Link_Stand", slice(0, 16), np.diag([-1, 1, 1])),
        ("Link_Stand", slice(0, 16), np.diag([1, -1, 1])),
        ("Link_Stand", slice(16, 25), Rotation.from_euler("z", 90, degrees=True).as_matrix()),
    ]
    for link, selection, transform in transforms:
        group = spheres[link][selection]
        points = centers(group)
        distances = np.linalg.norm((points @ transform.T)[:, None, :] - points[None, :, :], axis=-1)
        mapping = distances.argmin(axis=1)
        assert len(set(mapping.tolist())) == len(group)
        assert distances[np.arange(len(group)), mapping].max() < 1e-12
        radii = np.array([sphere["radius"] for sphere in group])
        np.testing.assert_array_equal(radii, radii[mapping])


def test_historical_model_replays_from_shared_templates_and_preserves_the_official_baseline():
    from replay_official_collision import rebuild as rebuild_official
    from replay_symmetric_collision import ARCHIVE, rebuild

    assert rebuild_official() == (ARCHIVE / "source.yml").read_bytes()
    assert rebuild() == (ARCHIVE / "candidate.yml").read_bytes()

