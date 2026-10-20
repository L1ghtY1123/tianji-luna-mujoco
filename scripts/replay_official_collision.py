"""Reconstruct the reviewed model from frozen public fitting outputs.

This replays recorded results. It does not rerun optimization or certify geometry.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml
from audit_official_collision import audit_provenance

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/collision-avoidance-refit-2026-09-30"


def archived_path(archive, name):
    path = (archive / name).resolve()
    if not path.is_relative_to(archive.resolve()):
        raise ValueError("Archive file escapes its directory")
    return path


def rebuild(archive=ARCHIVE):
    archive = Path(archive)
    recipe = json.loads((archive / "recipe.json").read_text(encoding="utf-8"))
    for name, digest in recipe["archive_files"].items():
        path = archived_path(archive, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Recorded input has changed: {path}")
    for name in (recipe["source"], recipe["manifest"]):
        if name not in recipe["archive_files"]:
            raise ValueError("Source and manifest must have frozen input hashes")
    source_bytes = archived_path(archive, recipe["source"]).read_bytes()
    source = yaml.safe_load(source_bytes)
    manifest = json.loads(archived_path(archive, recipe["manifest"]).read_text(encoding="utf-8"))
    if manifest.get("source_config_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("Manifest source does not match the frozen source configuration")
    candidate = deepcopy(source)
    spheres = candidate["robot_cfg"]["kinematics"]["collision_spheres"]
    for link, record in manifest["links"].items():
        if link not in spheres:
            raise ValueError(f"Unknown replacement link: {link}")
        spheres[link] = deepcopy(record["spheres"])
    report = audit_provenance(candidate, source, manifest)
    if not report["procedural_verified"]:
        raise ValueError("; ".join(report["errors"]))
    payload = yaml.safe_dump(candidate, sort_keys=False).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != recipe["model_sha256"]:
        raise ValueError("Replayed configuration differs from the reviewed model")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(ROOT / name) for name in ("assets", "resources", "data")):
        raise ValueError("Choose a new output file outside assets, resources and data")
    payload = rebuild()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    print(f"Recorded official model reproduced: {output}")


if __name__ == "__main__":
    main()
