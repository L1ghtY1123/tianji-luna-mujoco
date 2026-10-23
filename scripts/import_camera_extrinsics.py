"""Import user-owned millimetre extrinsics without changing frozen intrinsics."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {"1": ("right_wrist", "TCP_Link_R", "T_flange_cam"),
           "2": ("left_wrist", "TCP_Link_L", "T_flange_cam"),
           "3": ("head", "world", "T_base_cam")}


def import_extrinsics(source_directory):
    calibration_path = ROOT / "resources/cameras/calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    destination = ROOT / "resources/cameras/user-extrinsics"
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for camera_id, (role, parent, expected_name) in CAMERAS.items():
        source_path = Path(source_directory) / f"camera_{role}_extrinsic.json"
        raw = source_path.read_bytes()
        source = json.loads(raw)
        if (source["cam_id"] != role or source["extrinsic_name"] != expected_name
                or source["translation_unit"] != "mm" or source["base_frame"] != "base_link"):
            raise ValueError(f"unexpected calibration frames or units: {role}")
        matrix = np.asarray(source["extrinsic"], dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid calibration matrix: {role}")
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(matrix[3], [0, 0, 0, 1], atol=1e-12)
        matrix[:3, 3] *= .001
        target = destination / source_path.name
        shutil.copyfile(source_path, target)
        digest = hashlib.sha256(raw).hexdigest()
        hashes[role] = digest
        camera = calibration["cameras"][camera_id]
        camera["parent_body"] = parent
        camera["T_parent_rgb_m"] = matrix.tolist()
        camera.pop("historical_extrinsic", None)
        camera["extrinsic_provenance"] = {
            "source_path": target.relative_to(ROOT).as_posix(), "source_sha256": digest,
            "original_source_path": str(source_path), "source_frame": source["base_frame"],
            "extrinsic_name": expected_name, "source_translation_unit": "mm",
            "translation_scale_to_m": .001, "T_world_base_applied": False,
            "camera_serial_number": source["camera_serial_number"],
            "intrinsic_revision_from_source": source.get("intrinsic_revision"),
            "frame_interpretation": "flange-local optical frame" if parent != "world"
                                    else "T_base_cam is already in robot world/base_link",
        }
    calibration["calibration_kind"] = "user_extrinsics_with_frozen_registered_rgb_intrinsics"
    calibration["extrinsics_source"] = "resources/cameras/user-extrinsics/"
    calibration["extrinsics_source_sha256"] = hashes
    calibration.pop("extrinsics_sha256", None)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    calibration["calibration_id"] = "user-extrinsics-" + digest[:16]
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return {"calibration_id": calibration["calibration_id"], "sources": hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(import_extrinsics(args.source_directory), indent=2))


if __name__ == "__main__":
    main()
