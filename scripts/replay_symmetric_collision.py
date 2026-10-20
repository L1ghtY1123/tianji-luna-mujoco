"""Replay shared sphere templates in their actual link frames, without fitting.

Templates come from frozen cuRobo fitting results. Orthogonal transformations may
reflect or rotate the centers, but never scale or modify their radii.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/collision-symmetry-2026-10-08"


def apply_templates(source, rules):
    candidate = copy.deepcopy(source)
    original = source["robot_cfg"]["kinematics"]["collision_spheres"]
    result = candidate["robot_cfg"]["kinematics"]["collision_spheres"]
    replaced = set()
    for rule in rules:
        target = rule["target_link"]
        if target not in original:
            raise ValueError(f"Unknown target link: {target}")
        if target in replaced:
            raise ValueError(f"Duplicate target link: {target}")
        replaced.add(target)
        spheres = []
        used = set()
        for part in rule["parts"]:
            source_link = part["source_link"]
            if source_link not in original:
                raise ValueError(f"Unknown source link: {source_link}")
            matrix = np.asarray(part["transform"], dtype=float)
            if (
                matrix.shape != (4, 4)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-10)
            ):
                raise ValueError(f"Template transform must be an isometry: {target}")
            for index in part["indices"]:
                key = (source_link, index, tuple(matrix.ravel()))
                if (
                    not isinstance(index, int) or isinstance(index, bool)
                    or not 0 <= index < len(original[source_link]) or key in used
                ):
                    raise ValueError(f"Invalid or repeated source sphere: {key}")
                used.add(key)
                sphere = original[source_link][index]
                center = np.asarray(sphere["center"], dtype=float)
                radius = sphere["radius"]
                if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
                    raise ValueError(f"Invalid source sphere: {key}")
                mapped = matrix[:3, :3] @ center + matrix[:3, 3]
                spheres.append({"center": mapped.tolist(), "radius": radius})
        if not spheres:
            raise ValueError(f"Empty template: {target}")
        result[target] = spheres
    return candidate


def rebuild(archive=ARCHIVE):
    archive = Path(archive).resolve()
    recipe = json.loads((archive / "recipe.json").read_text(encoding="utf-8"))
    for name, digest in recipe["archive_files"].items():
        path = (archive / name).resolve()
        if not path.is_relative_to(archive):
            raise ValueError("Recorded input escapes the archive")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Recorded input has changed: {name}")
    if not {"source.yml", "rules.json"}.issubset(recipe["archive_files"]):
        raise ValueError("Source and template rules require frozen hashes")
    source = yaml.safe_load((archive / "source.yml").read_bytes())
    rules = json.loads((archive / "rules.json").read_text(encoding="utf-8"))
    candidate = apply_templates(source, rules["rules"])
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
    print(f"Shared collision templates reproduced: {output}")


if __name__ == "__main__":
    main()
