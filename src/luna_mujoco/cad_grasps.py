"""Read indexed Tianji CAD grasps without inference or scene-dependent filtering."""
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path, PureWindowsPath

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import rigid_matrix


class CadGraspError(ValueError):
    def __init__(self, code, message=""):
        self.code = code
        super().__init__(code + (": " + message if message else ""))


def _require(condition, code="cad_grasp_invalid_contract"):
    if not condition:
        raise CadGraspError(code)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class CadGraspLibrary:
    """Project-side reader; the asset repository and its reader remain read-only.

    The caller binds asset_id/source_sha256 to the identified model and supplies
    its current original-CAD pose in metres. Freshness and inventory identity
    belong to the caller; a rigid matrix alone cannot establish either fact.
    """

    def __init__(self, cad_root):
        self.root = Path(cad_root).resolve()

    def _path(self, parent, value):
        _require(isinstance(value, str) and bool(value), "cad_grasp_invalid_path")
        windows = PureWindowsPath(value)
        _require(not windows.drive and not windows.root and not Path(value).is_absolute(),
                 "cad_grasp_invalid_path")
        path = (parent / value).resolve()
        _require(path.is_relative_to(self.root), "cad_grasp_invalid_path")
        return path

    @staticmethod
    def _read(path, expected_hash=None):
        try:
            contents = path.read_bytes()
        except OSError as error:
            raise CadGraspError("cad_grasp_library_unavailable", str(path)) from error
        if expected_hash is not None:
            _require(_sha256(expected_hash) and hashlib.sha256(contents).hexdigest() == expected_hash,
                     "cad_grasp_hash_mismatch")
        return contents

    def _json(self, path, expected_hash=None):
        result = json.loads(self._read(path, expected_hash))
        _require(isinstance(result, dict))
        return result

    def _load(self, asset_id, source_sha256):
        catalog_path = self._path(self.root, "grasps/index.json")
        catalog = self._json(catalog_path)
        _require(catalog["schema_version"] == 1 and isinstance(catalog["libraries"], list))
        matches = [item for item in catalog["libraries"]
                   if item["asset_id"] == asset_id and item["gripper_profile"] == "tianji"]
        _require(bool(matches), "cad_grasp_library_unavailable")
        _require(len(matches) == 1)
        index_path = self._path(catalog_path.parent, matches[0]["path"])
        index = self._json(index_path)
        _require(index["schema_version"] == 1 and index["asset_id"] == asset_id
                 and index["gripper_profile"] == "tianji"
                 and isinstance(index["library_id"], str) and bool(index["library_id"]))

        cad_path = self._path(self.root, "cad/index.json")
        cad_catalog = self._json(cad_path)
        _require(cad_catalog["schema_version"] == 1 and isinstance(cad_catalog["assets"], list))
        sources = [item["source"] for item in cad_catalog["assets"] if item["asset_id"] == asset_id]
        _require(len(sources) == 1, "cad_grasp_source_mismatch")
        source = sources[0]
        bound_source = index["cad_source"]
        _require(source["sha256"] == bound_source["sha256"] == source_sha256
                 and source["units"] == bound_source["units"] and source["units"] in ("mm", "m"),
                 "cad_grasp_source_mismatch")
        source_path = self._path(cad_path.parent, source["path"])
        _require(source_path == self._path(index_path.parent, bound_source["path"]),
                 "cad_grasp_source_mismatch")
        self._read(source_path, source_sha256)

        contract = index["grasps"]
        _require(_sha256(contract["sha256"]), "cad_grasp_hash_mismatch")
        _require(contract["reference_frame"] == "original_CAD" and contract["position_unit"] == "m"
                 and contract["quaternion_order"] == "xyzw"
                 and contract["eef_frames"] == ["gripper_front_A", "gripper_front_B"]
                 and index["runtime_candidate_policy"]["prefilter_by_valid_table_placements"] is False)
        poses = self._json(self._path(index_path.parent, contract["path"]), contract["sha256"])
        _require(poses["schema"] == "tianji_front_eef_grasp_library_v1"
                 and poses["cad_sha256"] == source_sha256 and poses["pose_editing"] is False
                 and poses["arm_convention"]["A"] == "left" and poses["arm_convention"]["B"] == "right"
                 and poses["runtime_candidate_policy"]["prefilter_by_valid_table_placements"] is False)
        self._validate_rows(index, poses)
        return index, poses["grasps"]

    @staticmethod
    def _source_pose(row):
        position = np.asarray(row["eef_position_m"], dtype=float)
        quaternion = np.asarray(row["eef_quaternion_xyzw"], dtype=float)
        _require(position.shape == (3,) and np.isfinite(position).all()
                 and quaternion.shape == (4,) and np.isfinite(quaternion).all()
                 and np.isclose(np.linalg.norm(quaternion), 1., atol=1e-6, rtol=0))
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(quaternion, scalar_first=False).as_matrix()
        pose[:3, 3] = position
        return pose

    def _validate_rows(self, index, poses):
        counts = index["grasps"]["count_per_arm"]
        _require(set(counts) == {"A", "B"})
        for count in counts.values():
            _require(type(count) is int and count > 0)
        pair_count = index["pair_count"]
        _require(type(pair_count) is int and pair_count == counts["A"] == counts["B"]
                 and poses["arm_convention"]["pair_count"] == pair_count
                 and poses["selection"]["count_per_arm"] == pair_count
                 and poses["selection"]["pair_count"] == pair_count
                 and poses["selection"]["pose_editing"] is False
                 and index["pairing_key"] == "pair_id" and index["paired_grasp_key"] == "paired_grasp_id")
        rows = poses["grasps"]
        _require(isinstance(rows, list) and len(rows) == sum(counts.values()))
        by_id = {}
        for row in rows:
            arm = row["arm"]
            grasp_id = row["grasp_id"]
            _require(arm in ("A", "B") and isinstance(grasp_id, str) and bool(grasp_id)
                     and grasp_id not in by_id and row["eef_frame"] == "gripper_front_" + arm
                     and row["reference_frame"] == "original_CAD" and row["position_unit"] == "m"
                     and type(row["source_score"]) in (int, float) and np.isfinite(row["source_score"]))
            self._source_pose(row)
            by_id[grasp_id] = row
        for arm in ("A", "B"):
            selected = [row for row in rows if row["arm"] == arm]
            _require(len(selected) == counts[arm]
                     and len({row["pair_id"] for row in selected}) == counts[arm])
        for row in rows:
            paired = by_id.get(row["paired_grasp_id"])
            _require(paired is not None and paired["arm"] != row["arm"]
                     and paired["paired_grasp_id"] == row["grasp_id"]
                     and paired["pair_id"] == row["pair_id"] and paired["source_id"] == row["source_id"])

    def generate(self, *, asset_id, source_sha256, arm, T_world_source, source_asset_id=None):
        """Return every indexed pose of one arm in original order, already front TCP.

        No millimetre scaling, mount offset, jaw-symmetry choice, insertion,
        table-label filtering, candidate budget, IK or collision claim is applied.
        Missing libraries and invalid bindings raise CadGraspError; no fallback.
        """
        _require(arm in ("A", "B"), "cad_grasp_invalid_arm")
        _require(isinstance(asset_id, str) and bool(asset_id) and _sha256(source_sha256),
                 "cad_grasp_source_mismatch")
        try:
            world_pose = rigid_matrix(T_world_source).copy()
        except (TypeError, ValueError) as error:
            raise CadGraspError("cad_grasp_invalid_pose") from error
        try:
            index, rows = self._load(source_asset_id or asset_id, source_sha256)
            candidates = []
            for source_index, row in enumerate(rows):
                if row["arm"] != arm:
                    continue
                source_pose = self._source_pose(row)
                candidates.append({**deepcopy(row), "index": len(candidates), "source_batch": 1,
                                   "source_index": source_index, "score": float(row["source_score"]),
                                   "source": "cad_offline_library", "T_source_front": source_pose.tolist(),
                                   "T_world_tcp": (world_pose @ source_pose).tolist()})
        except CadGraspError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise CadGraspError("cad_grasp_invalid_contract") from error
        return {"source": "cad_offline_library", "asset_id": asset_id, "arm": arm,
                "source_asset_id": source_asset_id or asset_id,
                "source_sha256": source_sha256, "source_unit": index["cad_source"]["units"],
                "library_id": index["library_id"], "library_sha256": index["grasps"]["sha256"],
                "reference_frame": "original_CAD", "position_unit": "m", "quaternion_order": "xyzw",
                "tcp_frame": "gripper_front_" + arm, "T_world_source": world_pose.tolist(),
                "candidate_count": len(candidates), "candidates": candidates}
