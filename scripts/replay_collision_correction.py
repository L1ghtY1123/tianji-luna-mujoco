"""Rebuild the reviewed YAML from recorded source indices and official final outputs.

This is an exact archive replay, not a new cuRobo optimization or geometry approval.
The output must be a new file outside the mounted robot resources.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/collision-correction-2026-09-30"
SOURCE = ROOT / "data/collision-sphere-source-2026-09-30.yml"


def check_hash(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Recorded input has changed: {path}")


def rebuild():
    recipe = json.loads((ARCHIVE / "recipe.json").read_text(encoding="utf-8"))
    check_hash(SOURCE, recipe["source_sha256"])
    for name, digest in recipe["archive_files"].items():
        path = (ARCHIVE / name).resolve()
        if not path.is_relative_to(ARCHIVE.resolve()):
            raise ValueError("Archive file escapes its directory")
        check_hash(path, digest)
    manifest = json.loads((ARCHIVE / "static-fit.json").read_text(encoding="utf-8"))
    geometry = manifest["geometry_input_lock"]
    for record in [geometry["urdf"], *geometry["visual_assets"]]:
        check_hash(ROOT / record["path"], record["sha256"])
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    original = source["robot_cfg"]["kinematics"]["collision_spheres"]
    static_spheres = {}
    for link, records in manifest["provenance"].items():
        values = []
        for record in records:
            if record["kind"] == "original_source":
                sphere = original[link][record["source_index"]]
            elif record["kind"] in {"official_final", "official_final_rigid_frame"}:
                sphere = manifest["calls"][record["call"]]["spheres"][record["result_index"]]
            else:
                raise ValueError(f"Unsupported sphere provenance: {record['kind']}")
            values.append(sphere)
        static_spheres[link] = values
    result = deepcopy(source)
    spheres = result["robot_cfg"]["kinematics"]["collision_spheres"]
    for link, records in recipe["sphere_origins"].items():
        values = []
        for record in records:
            if record["source"] == "original":
                pool = original[link]
            elif record["source"] == "static_final":
                pool = static_spheres[link]
            else:
                raise ValueError(f"Unsupported sphere source: {record['source']}")
            values.append(deepcopy(pool[record["index"]]))
        spheres[link] = values
    # Preserve the reviewed Windows file's bytes when replaying on Linux too.
    payload = yaml.safe_dump(result, sort_keys=False).replace("\n", recipe["yaml_newline"]).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != recipe["sha256"]:
        raise ValueError("Replayed configuration differs from the reviewed model")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(
        output.is_relative_to(ROOT / name) for name in ("assets", "resources", "data")
    ):
        raise ValueError("Choose a new output file outside assets, resources and data")
    payload = rebuild()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    print(f"Exact archived model reproduced: {output}")


if __name__ == "__main__":
    main()
