"""Replay frozen public final fits plus rigid shared templates, without CUDA."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from refit_transfer_spheres import convert_final
from replay_symmetric_collision import apply_templates

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/continuous-transfer-20261012/spheres"


def apply_refits(source, manifests, rules):
    pool = copy.deepcopy(source)
    sphere_map = pool["robot_cfg"]["kinematics"]["collision_spheres"]
    original_links = set(sphere_map)
    for name, manifest in manifests.items():
        for job, record in manifest["jobs"].items():
            key = f"fit/{name}/{job}"
            result = SimpleNamespace(centers=record["final_centers_fit_frame_m"],
                                     radii=record["final_radii_m"], debug_info=record["debug_info"])
            spheres = convert_final(result, record["fit_from_link"], record["parameters"]["num_spheres"])
            recorded = record["spheres"]
            if ([sphere["radius"] for sphere in spheres] != [sphere["radius"] for sphere in recorded]
                    or not np.allclose([sphere["center"] for sphere in spheres],
                                       [sphere["center"] for sphere in recorded], rtol=0, atol=1e-14)):
                raise ValueError("Stored fit differs from the official final arrays")
            # Preserve the frozen serialization across Linux/Windows BLAS rounding.
            sphere_map[key] = copy.deepcopy(recorded)
    if any(rule["target_link"] not in original_links for rule in rules):
        raise ValueError("Refit target must be an original robot link")
    candidate = apply_templates(pool, rules)
    result = candidate["robot_cfg"]["kinematics"]["collision_spheres"]
    for key in set(result) - original_links:
        del result[key]
    if sum(map(len, result.values())) > 350:
        raise ValueError("Refit exceeds the total robot budget of 350 spheres")
    return candidate


def rebuild(archive=ARCHIVE, recipe_name="recipe.json"):
    archive = Path(archive).resolve()
    recipe_path = (archive / recipe_name).resolve()
    if not recipe_path.is_relative_to(archive):
        raise ValueError("Recipe escapes archive")
    recipe = json.loads(recipe_path.read_text())
    rules_name = recipe.get("rules_file", "rules.json")
    required = {"source.yml", rules_name, *recipe["manifests"].values()}
    if not required.issubset(recipe["archive_files"]):
        raise ValueError("Every replay input requires a frozen hash")
    for name, expected in recipe["archive_files"].items():
        path = (archive / name).resolve()
        if not path.is_relative_to(archive):
            raise ValueError("Input escapes archive")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Recorded input has changed: {name}")
    source = yaml.safe_load((archive / "source.yml").read_bytes())
    manifests = {key: json.loads((archive / path).read_text()) for key, path in recipe["manifests"].items()}
    rules = json.loads((archive / rules_name).read_text())["rules"]
    candidate = apply_refits(source, manifests, rules)
    spheres = candidate["robot_cfg"]["kinematics"]["collision_spheres"]
    for link, indices in recipe.get("remove_contained_spheres", {}).items():
        values = spheres[link]
        if len(set(indices)) != len(indices) or any(type(index) is not int or index < 0 or index >= len(values)
                                                    for index in indices):
            raise ValueError("Invalid contained sphere index")
        for index in indices:
            center = np.asarray(values[index]["center"], dtype=float)
            radius = values[index]["radius"]
            if not any(other != index and other not in indices and
                       np.linalg.norm(center - np.asarray(values[other]["center"], dtype=float)) + radius
                       < values[other]["radius"] - 1e-6 for other in range(len(values))):
                raise ValueError("Removed sphere is not strictly contained by a retained sphere")
        spheres[link] = [value for index, value in enumerate(values) if index not in indices]
    payload = yaml.safe_dump(candidate, sort_keys=False).encode()
    if hashlib.sha256(payload).hexdigest() != recipe["model_sha256"]:
        raise ValueError("Replayed model differs from reviewed configuration")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--recipe", default="recipe.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(ROOT / path) for path in ("resources", "assets", "data")):
        raise ValueError("Choose a new output outside resources, assets and data")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(rebuild(args.archive, recipe_name=args.recipe))
    print(f"Reproduced collision configuration: {output}")


if __name__ == "__main__":
    main()
