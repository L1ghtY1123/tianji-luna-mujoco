"""Offline library contracts use tiny, real files instead of a grasp service."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cad_library(tmp_path):
    root = tmp_path / "CAD"
    source = root / "cad/part/part.stl"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture CAD with millimetre source coordinates")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(root / "cad/index.json", {"schema_version": 1, "assets": [
        {"asset_id": "part", "source": {"path": "part/part.stl", "units": "mm", "sha256": digest}},
        {"asset_id": "blue_box", "source": {"path": "box.step", "units": "mm", "sha256": "0" * 64}},
    ]})
    write_json(root / "grasps/index.json", {"schema_version": 1, "libraries": [
        {"asset_id": "part", "gripper_profile": "tianji", "path": "part/tianji/index.json"}
    ]})
    rows = []
    for arm, quaternion in (("A", [0, 0, 0, 1]), ("B", [0, 0, 1, 0])):
        for number in (1, 2):
            rows.append({"grasp_id": f"{arm}_{number}", "arm": arm,
                         "pair_id": f"P_{number}",
                         "paired_grasp_id": f"{'B' if arm == 'A' else 'A'}_{number}",
                         "eef_frame": f"gripper_front_{arm}",
                         "reference_frame": "original_CAD", "position_unit": "m",
                         "eef_position_m": [.01 * number, .02, -.03],
                         "eef_quaternion_xyzw": quaternion,
                         "source_id": f"source_{number}", "source_score": .9 / number,
                         "valid_table_placements": []})
    poses = {"schema": "tianji_front_eef_grasp_library_v1", "cad_sha256": digest,
             "pose_editing": False, "arm_convention": {"A": "left", "B": "right", "pair_count": 2},
             "selection": {"count_per_arm": 2, "pair_count": 2, "pose_editing": False},
             "runtime_candidate_policy": {"prefilter_by_valid_table_placements": False}, "grasps": rows}
    pose_path = root / "grasps/part/tianji/grasp_poses.json"
    pose_hash = write_json(pose_path, poses)
    index = {"schema_version": 1, "library_id": "test-paired-v1", "asset_id": "part",
             "gripper_profile": "tianji", "pair_count": 2,
             "pairing_key": "pair_id", "paired_grasp_key": "paired_grasp_id",
             "cad_source": {"path": "../../../cad/part/part.stl", "units": "mm", "sha256": digest},
             "grasps": {"path": "grasp_poses.json", "sha256": pose_hash,
                        "count_per_arm": {"A": 2, "B": 2}, "reference_frame": "original_CAD",
                        "position_unit": "m", "quaternion_order": "xyzw",
                        "eef_frames": ["gripper_front_A", "gripper_front_B"]},
             "runtime_candidate_policy": {"prefilter_by_valid_table_placements": False}}
    index_path = root / "grasps/part/tianji/index.json"
    write_json(index_path, index)
    return {"root": root, "source": source, "source_sha256": digest, "index": index,
            "index_path": index_path, "poses": poses, "pose_path": pose_path}


def generate(fixture, **changes):
    from luna_mujoco.cad_grasps import CadGraspLibrary

    arguments = {"asset_id": "part", "source_sha256": fixture["source_sha256"],
                 "arm": "A", "T_world_source": np.eye(4)}
    arguments.update(changes)
    return CadGraspLibrary(fixture["root"]).generate(**arguments)


def save_library(fixture):
    fixture["index"]["grasps"]["sha256"] = write_json(fixture["pose_path"], fixture["poses"])
    write_json(fixture["index_path"], fixture["index"])


@pytest.mark.parametrize("arm,expected_rotation", [
    ("A", [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ("B", [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
])
def test_returns_every_pose_in_world_without_scaling_tcp_offset_or_symmetry(cad_library, arm, expected_rotation):
    world = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    result = generate(cad_library, arm=arm, T_world_source=world)

    assert result["candidate_count"] == 2
    assert [row["grasp_id"] for row in result["candidates"]] == [f"{arm}_1", f"{arm}_2"]
    assert result["source"] == "cad_offline_library"
    assert result["source_unit"] == "mm"
    assert result["position_unit"] == "m"
    assert result["tcp_frame"] == f"gripper_front_{arm}"
    assert result["library_sha256"] == cad_library["index"]["grasps"]["sha256"]
    first = result["candidates"][0]
    np.testing.assert_allclose(np.asarray(first["T_world_tcp"])[:3, 3], [.98, 2.01, 2.97])
    np.testing.assert_allclose(np.asarray(first["T_world_tcp"])[:3, :3], expected_rotation, atol=1e-15)
    np.testing.assert_allclose(np.asarray(first["T_source_front"])[:3, 3], [.01, .02, -.03])
    assert first["index"] == 0 and first["score"] == .9
    assert first["valid_table_placements"] == []
    assert first["source"] == "cad_offline_library"


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_renamed_asset_uses_explicit_original_library_without_changing_poses(cad_library, arm):
    previous = generate(cad_library, arm=arm)
    renamed = generate(cad_library, arm=arm, asset_id='black_workpiece', source_asset_id='part')
    assert renamed['asset_id'] == 'black_workpiece'
    assert renamed['source_asset_id'] == 'part'
    assert renamed['candidates'] == previous['candidates']
    assert renamed['library_sha256'] == previous['library_sha256']


@pytest.mark.parametrize("arm", [None, "auto", "left", "C"])
def test_requires_explicit_supported_arm(cad_library, arm):
    with pytest.raises(ValueError, match="cad_grasp_invalid_arm"):
        generate(cad_library, arm=arm)


@pytest.mark.parametrize("world", [np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]),
                                   np.full((4, 4), np.nan), np.eye(3)])
def test_rejects_nonrigid_world_pose(cad_library, world):
    with pytest.raises(ValueError, match="cad_grasp_invalid_pose"):
        generate(cad_library, T_world_source=world)


def test_missing_library_is_explicit_for_known_cad(cad_library):
    with pytest.raises(ValueError, match="cad_grasp_library_unavailable"):
        generate(cad_library, asset_id="blue_box", source_sha256="0" * 64)


@pytest.mark.parametrize("digest", [None, "", "x" * 64, "0" * 64])
def test_rejects_missing_or_mismatched_model_source_binding(cad_library, digest):
    with pytest.raises(ValueError, match="cad_grasp_source_mismatch"):
        generate(cad_library, source_sha256=digest)


@pytest.mark.parametrize("file_key", ["source", "pose_path"])
def test_rejects_file_bytes_that_no_longer_match_index(cad_library, file_key):
    path = cad_library[file_key]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="cad_grasp_hash_mismatch"):
        generate(cad_library)


@pytest.mark.parametrize("digest", [None, "", "x" * 64])
def test_rejects_invalid_declared_library_hash(cad_library, digest):
    cad_library["index"]["grasps"]["sha256"] = digest
    write_json(cad_library["index_path"], cad_library["index"])
    with pytest.raises(ValueError, match="cad_grasp_hash_mismatch"):
        generate(cad_library)


@pytest.mark.parametrize("field,value", [
    ("reference_frame", "canonical"), ("position_unit", "mm"),
    ("quaternion_order", "wxyz"), ("eef_frames", ["TCP_Link_L", "TCP_Link_R"]),
])
def test_rejects_incompatible_index_pose_contract(cad_library, field, value):
    cad_library["index"]["grasps"][field] = value
    save_library(cad_library)
    with pytest.raises(ValueError, match="cad_grasp_invalid_contract"):
        generate(cad_library)


@pytest.mark.parametrize("field,value", [
    ("reference_frame", "canonical"), ("position_unit", "mm"), ("eef_frame", "gripper_front_B"),
    ("eef_quaternion_xyzw", [0, 0, 0, 2]), ("eef_position_m", [0, 0, float("nan")]),
    ("source_score", float("nan")), ("arm", "C"),
])
def test_rejects_incompatible_row_even_with_updated_checksum(cad_library, field, value):
    cad_library["poses"]["grasps"][0][field] = value
    save_library(cad_library)
    with pytest.raises(ValueError, match="cad_grasp_invalid_contract"):
        generate(cad_library)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_pair", "wrong_source_hash"])
def test_rejects_incomplete_or_misbound_library(cad_library, change):
    rows = cad_library["poses"]["grasps"]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1]["grasp_id"] = rows[0]["grasp_id"]
    elif change == "wrong_pair":
        rows[0]["paired_grasp_id"] = "B_missing"
    else:
        cad_library["poses"]["cad_sha256"] = "0" * 64
    save_library(cad_library)
    with pytest.raises(ValueError, match="cad_grasp_invalid_contract"):
        generate(cad_library)


@pytest.mark.parametrize("path", ["../../../../outside.json", "C:/outside.json", "\\\\host\\share\\poses.json"])
def test_rejects_library_paths_outside_repository(cad_library, path):
    cad_library["index"]["grasps"]["path"] = path
    save_library(cad_library)
    with pytest.raises(ValueError, match="cad_grasp_invalid_path"):
        generate(cad_library)


def test_rejects_source_units_disagreement(cad_library):
    cad_library["index"]["cad_source"]["units"] = "m"
    save_library(cad_library)
    with pytest.raises(ValueError, match="cad_grasp_source_mismatch"):
        generate(cad_library)


def test_allows_declared_metre_cad_source_without_pose_scaling(cad_library):
    path = cad_library["root"] / "cad/index.json"
    catalog = json.loads(path.read_text())
    catalog["assets"][0]["source"]["units"] = "m"
    write_json(path, catalog)
    cad_library["index"]["cad_source"]["units"] = "m"
    save_library(cad_library)
    result = generate(cad_library)
    np.testing.assert_allclose(np.asarray(result["candidates"][0]["T_world_tcp"])[:3, 3], [.01, .02, -.03])


def test_library_returns_fresh_rows_on_each_call(cad_library):
    first = generate(cad_library)
    first["candidates"][0]["eef_position_m"][0] = 999
    second = generate(cad_library)
    assert second["candidates"][0]["eef_position_m"][0] == .01


def test_published_library_preserves_all_paired_arm_frames_when_available():
    from luna_mujoco.cad_grasps import CadGraspLibrary

    root = Path(__file__).resolve().parents[2] / "CAD"
    index_path = root / "grasps/THREADED_VISUAL_SAMPLE_MM/tianji/index.json"
    if not index_path.is_file():
        pytest.skip("Published sibling CAD repository is not installed")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    reader = CadGraspLibrary(root)
    matrices = {}
    for arm in ("A", "B"):
        result = reader.generate(asset_id="THREADED_VISUAL_SAMPLE_MM",
                                 source_sha256=index["cad_source"]["sha256"],
                                 arm=arm, T_world_source=np.eye(4))
        assert result["candidate_count"] == index["grasps"]["count_per_arm"][arm]
        assert len({row["grasp_id"] for row in result["candidates"]}) == result["candidate_count"]
        matrices[arm] = np.asarray([row["T_world_tcp"] for row in result["candidates"]])
    np.testing.assert_allclose(matrices["A"][:, :3, 3], matrices["B"][:, :3, 3], atol=1e-12)
    np.testing.assert_allclose(matrices["A"][:, :3, 2], matrices["B"][:, :3, 2], atol=1e-12)
    np.testing.assert_allclose(matrices["A"][:, :3, :2], -matrices["B"][:, :3, :2], atol=1e-12)
