"""Reproduce reviewed static spheres from unedited public cuRobo fit evidence.

The default is an exact archive replay, not a fresh optimizer run. --refit-call
explicitly reruns one public SDK call into a new directory for separate review.
Neither mode writes the deployed robot model or changes buffers/ignore rules.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

ARCHIVE_PATH = Path("data/collision-static-fit-correction-2026-09-30")


def sphere_digest(spheres):
    payload = json.dumps(spheres, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def checked_file(directory, record):
    directory = Path(directory).resolve()
    path = (directory / record["path"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Evidence path must stay inside its archive")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Evidence hash differs: {record['path']}")
    return path


def centers_in_link_frame(centers, fit_frame_from_link):
    """Inverse rigid mapping; distances and sphere radii are unchanged."""
    transform = np.asarray(fit_frame_from_link, dtype=float)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12, rtol=0.0)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("The fit frame must be a proper rigid transform")
    return (np.asarray(centers, dtype=float) - transform[:3, 3]) @ transform[:3, :3]


def replay_spheres(source_spheres, outputs, selections):
    lookup = {}
    for call, values in outputs.items():
        count = len(values["radii"])
        if (
            np.asarray(values["centers"]).shape != (count, 3)
            or len(values["iterations"]) != count
            or len(values["sphere_indices"]) != count
            or not np.isfinite(values["centers"]).all()
            or not np.isfinite(values["radii"]).all()
            or np.any(values["radii"] <= 0)
        ):
            raise ValueError(f"Invalid official sphere output: {call}")
        for row in range(count):
            identity = (call, int(values["iterations"][row]), int(values["sphere_indices"][row]))
            if identity in lookup:
                raise ValueError(f"Duplicate official history identity: {identity}")
            lookup[identity] = {
                "center": np.asarray(values["centers"][row], dtype=float).tolist(),
                "radius": float(values["radii"][row]),
            }
    spheres = []
    for selection in selections:
        if selection["source"] == "original_source":
            index = selection["index"]
            if not isinstance(index, int) or not 0 <= index < len(source_spheres):
                raise ValueError("Invalid original source sphere index")
            sphere = source_spheres[index]
        elif selection["source"] in ["official_history", "official_surface"]:
            identity = (selection["call"], selection["iteration"], selection["sphere_index"])
            if identity not in lookup:
                raise ValueError(f"Missing official sphere identity: {identity}")
            sphere = lookup[identity]
        else:
            raise ValueError(f"Unknown sphere provenance: {selection['source']}")
        spheres.append(deepcopy(sphere))
    return spheres


def load_archive(project):
    archive = project / ARCHIVE_PATH
    manifest = json.loads((archive / "manifest.json").read_text())
    for record in manifest["geometry_sources"]:
        checked_file(project, record)
    return archive, manifest


def fit_static_link(project, link_name, source_spheres, output_dir, seed):
    """Exact reviewed selection from source indices and archived public fit rows.

    Seed is recorded for the caller but deliberately does not regenerate the
    approved selection. Fresh SDK fitting is an explicit separate operation.
    """
    project, output_dir = Path(project).resolve(), Path(output_dir).resolve()
    archive, manifest = load_archive(project)
    if link_name not in manifest["links"]:
        raise ValueError(f"Unsupported static link: {link_name}")
    selection = manifest["links"][link_name]
    if sphere_digest(source_spheres) != selection["source_spheres_sha256"]:
        raise ValueError("Static replay source spheres differ from the reviewed original source")
    outputs, calls = {}, []
    for name in selection["calls"]:
        call = manifest["calls"][name]
        path = checked_file(archive, call["output"])
        checked_file(archive, call["input"])
        with np.load(path, allow_pickle=False) as values:
            outputs[name] = {key: values[key] for key in values.files}
        calls.append(deepcopy(call))
    spheres = replay_spheres(source_spheres, outputs, selection["provenance"])
    if sphere_digest(spheres) != selection["sphere_sha256"]:
        raise ValueError("Replayed static sphere digest differs from the reviewed selection")
    record = {
        "link": link_name,
        "spheres": spheres,
        "sphere_count": len(spheres),
        "source_sphere_count": len(source_spheres),
        "refitted": False,
        "generation_mode": "archived_public_fit_replay",
        "source": manifest["source_config"],
        "sdk_commit": manifest["sdk_commit"],
        "archive_manifest_sha256": hashlib.sha256((archive / "manifest.json").read_bytes()).hexdigest(),
        "requested_seed": seed,
        "seed_used_for_replay": None,
        "fit_calls": calls,
        "provenance": deepcopy(selection["provenance"]),
        "selection": deepcopy(selection["selection"]),
        "metrics": deepcopy(selection["metrics"]),
        "limitations": manifest["limitations"],
    }
    if output_dir.is_relative_to(archive) or any(
        output_dir.is_relative_to(project / name) for name in ["assets", "resources"]
    ):
        raise ValueError("Replay output must not overwrite robot assets or archived evidence")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=2))
    return record


def rerun_public_call(project, call_name, output_dir):
    """Rerun the archived input/parameters; do not reuse its accepted selection."""
    import torch
    import trimesh
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
    from curobo.types import DeviceCfg

    archive, manifest = load_archive(Path(project).resolve())
    call = manifest["calls"][call_name]
    with np.load(checked_file(archive, call["input"]), allow_pickle=False) as values:
        mesh = trimesh.Trimesh(values["vertices"], values["faces"], process=False)
    output_dir = Path(output_dir).resolve()
    project = Path(project).resolve()
    if output_dir.is_relative_to(project) or output_dir == project:
        raise ValueError("Fresh fits must use a new directory outside the project")
    output_dir.mkdir(parents=True, exist_ok=False)
    np.random.seed(call["seed"])
    torch.manual_seed(call["seed"])
    torch.set_num_threads(4)
    parameters = dict(call["parameters"])
    parameters["fit_type"] = SphereFitType[parameters["fit_type"]]
    parameters["device_cfg"] = DeviceCfg(device=call["device"])
    result = fit_spheres_to_mesh(mesh, **parameters)
    centers, radii, iterations, indices = [], [], [], []
    history = list(enumerate(result.history))
    if not history:
        history = [(-1, (result.centers.cpu().numpy(), result.radii.cpu().numpy()))]
    for step, (cc, rr) in history:
        if step >= 0 and step % call["history_stride"] and step != len(result.history) - 1:
            continue
        cc = centers_in_link_frame(cc, call["fit_frame_from_link"])
        centers.extend(cc)
        radii.extend(np.asarray(rr, dtype=float))
        iterations.extend([step] * len(rr))
        indices.extend(range(len(rr)))
    np.savez_compressed(
        output_dir / "result.npz", centers=centers, radii=radii, iterations=iterations, sphere_indices=indices
    )
    record = {
        "call": call,
        "generation_mode": "fresh_public_sdk_fit",
        "debug_info": result.debug_info,
        "output_rows": len(radii),
        "selection_reused": False,
        "requires_new_geometry_review": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refit-call", help="Archive call key, e.g. Link_Stand:component19")
    parser.add_argument("--link", choices=["Link_Base", "Link_Stand"])
    args = parser.parse_args()
    if args.refit_call:
        record = rerun_public_call(args.project, args.refit_call, args.output)
        print(json.dumps({"rows": record["output_rows"], "requires_new_geometry_review": True}))
        return
    _, archive = load_archive(args.project)
    config = yaml.safe_load((args.project / archive["source_config"]).read_text())
    for link in [args.link] if args.link else ["Link_Base", "Link_Stand"]:
        record = fit_static_link(
            args.project,
            link,
            config["robot_cfg"]["kinematics"]["collision_spheres"][link],
            args.output / link,
            seed=None,
        )
        print(
            json.dumps(
                {
                    "link": link,
                    "sphere_count": record["sphere_count"],
                    "generation_mode": record["generation_mode"],
                }
            )
        )


if __name__ == "__main__":
    main()
