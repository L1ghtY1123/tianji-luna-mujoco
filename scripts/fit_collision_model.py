"""Generate unreviewed, budgeted collision-model candidates with public cuRobo."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh
import yaml

SDK_COMMIT = "8e734f3ced1df898990bcd92de40abce475907db"
PIPELINE = "official_final_v1"


def load_visual_parts(project: Path, link_name: str):
    urdf_path = project / "assets/marvin_urdf/marvin_with_grippers.urdf"
    robot = ET.parse(urdf_path).getroot()
    link = robot.find(f"link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Missing URDF link: {link_name}")
    meshes = []
    sources = []
    for visual in link.findall("visual"):
        mesh_element = visual.find("geometry/mesh")
        if mesh_element is None:
            raise ValueError(f"Visual is not a mesh: {link_name}")
        path = (urdf_path.parent / mesh_element.attrib["filename"]).resolve()
        mesh = trimesh.load_mesh(path, process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected one triangle mesh: {path}")
        mesh.apply_scale(np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" "))
        origin = visual.find("origin")
        if origin is not None:
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
            transform[:3, 3] = xyz
            mesh.apply_transform(transform)
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No visual geometry: {link_name}")
    return meshes, sources


def load_visual_mesh(project: Path, link_name: str):
    meshes, sources = load_visual_parts(project, link_name)
    mesh = trimesh.util.concatenate(meshes)
    selection = {"kind": "all_link_visuals", "visual_count": len(meshes)}
    if link_name == "Link_Base":
        components = mesh.split(only_watertight=False)
        retained = []
        excluded = []
        for component in components:
            max_radius = float(np.linalg.norm(component.vertices[:, :2], axis=1).max())
            if max_radius <= 0.161:
                retained.append(component)
            else:
                excluded.append(component)
        if len(components) != 46 or len(retained) != 15:
            raise ValueError("Link_Base components differ from the approved 46 / 15 audit")
        if any(float(component.bounds[1, 2]) > 0.060001 for component in excluded):
            raise ValueError("Foot exclusion includes geometry above the audited 60 mm limit")
        mesh = trimesh.util.concatenate(retained)
        selection = {
            "kind": "central_components_only",
            "max_component_radial_extent_m": 0.161,
            "retained_components": len(retained),
            "excluded_components": len(excluded),
        }
    return mesh, sources, selection


def build_candidate(config, records, sphere_budget):
    """Replace only the selected sphere arrays, leaving all other settings intact."""
    candidate = deepcopy(config)
    kinematics = candidate["robot_cfg"]["kinematics"]
    for name, record in records.items():
        if name not in kinematics["collision_link_names"]:
            raise ValueError(f"Unknown collision link: {name}")
        if record.get("result_source") != "official_final":
            raise ValueError(f"{name}: only an official final result is accepted")
        if not 1 <= len(record["spheres"]) <= sphere_budget:
            raise ValueError(f"{name}: result exceeds or violates per-link sphere budget")
        kinematics["collision_spheres"][name] = deepcopy(record["spheres"])
    return candidate


def validate_resume(previous, expected):
    if previous.get("pipeline") != PIPELINE:
        raise ValueError(
            "Only official_final_v1 artifacts can be resumed; old history candidates are rejected"
        )
    for key in [
        "source_config_sha256",
        "urdf_sha256",
        "script_sha256",
        "motion_helper_sha256",
        "generator_parameters",
    ]:
        if previous.get(key) != expected.get(key):
            raise ValueError(f"Resume inputs differ for {key}; choose a new output directory")


def validate_record_geometry(record, sources, selection):
    if record.get("geometry_sources") != sources or record.get("geometry_selection") != selection:
        raise ValueError("Resume visual geometry differs; choose a new output directory")


def wait_for_live_algorithm(url, link):
    if not url:
        return
    while True:
        with urllib.request.urlopen(url, timeout=10) as response:
            state = json.load(response)
        if not state.get("run", {}).get("algorithm_pending", False):
            return
        print(json.dumps({"waiting_for_live_algorithm": link}), flush=True)
        time.sleep(5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-config", type=Path, help="Defaults to the archived audited source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--links", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--num-spheres", type=int, help="Required upper bound for EACH selected link")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--runtime-url", help="Optional live /api/state guard before each offline link")
    args = parser.parse_args(argv)
    project, output = args.project.resolve(), args.output.resolve()
    source = (args.source_config or project / "data/collision-sphere-source-2026-09-30.yml").resolve()
    if output == project or any(output.is_relative_to(project / name) for name in ["assets", "resources"]):
        raise ValueError("Output must not overwrite project assets or resources")
    if source == output / "marvin_gui_dual.yml":
        raise ValueError("Output must not overwrite the input configuration")
    if output.exists() and not args.resume:
        raise FileExistsError("Choose a new output directory; existing artifacts are not overwritten")
    if bool(args.all) == bool(args.links):
        raise ValueError("Specify either --all or explicit --links")
    if args.num_spheres is None or args.num_spheres <= 0:
        raise ValueError("--num-spheres must specify a positive per-link sphere budget")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    config = yaml.safe_load(source.read_text())
    kinematics = config["robot_cfg"]["kinematics"]
    names = list(kinematics["collision_link_names"]) if args.all else args.links
    if len(set(names)) != len(names):
        raise ValueError("Selected links must not contain duplicates")
    if any(name not in kinematics["collision_link_names"] for name in names):
        raise ValueError("Selected links must be enabled collision links")
    script_path = Path(__file__)
    helper_path = script_path.with_name("fit_motion_collision_model.py")
    urdf_path = project / "assets/marvin_urdf/marvin_with_grippers.urdf"
    manifest = {
        "pipeline": PIPELINE,
        "sdk_expected_commit": SDK_COMMIT,
        "api": "curobo.sphere_fit.fit_spheres_to_mesh",
        "result_source": "official_final",
        "candidate_only": True,
        "geometry_reviewed": False,
        "source_config_path": str(source),
        "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "motion_helper_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        "generator_parameters": {
            "links": names,
            "num_spheres_per_link": args.num_spheres,
            "seed": args.seed,
            "iterations": args.iterations,
        },
        "sphere_output_postprocessing": "None; official final centers/radii only",
        "links": {},
    }
    manifest_path = output / "manifest.json"
    if args.resume and output.exists():
        if not manifest_path.exists():
            raise ValueError("Cannot resume existing artifacts without their manifest")
        validate_resume(json.loads(manifest_path.read_text()), manifest)

    import curobo
    import torch
    from curobo.logging import setup_logger
    from fit_motion_collision_model import fit_motion_link

    if output.is_relative_to(Path(curobo.__file__).resolve().parent):
        raise ValueError("Output must not modify the installed cuRobo SDK")
    setup_logger("warn")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for name in names:
        record_path = output / f"{name}.json"
        mesh, sources, selection = load_visual_mesh(project, name)
        if args.resume and record_path.exists():
            record = json.loads(record_path.read_text())
            validate_record_geometry(record, sources, selection)
        else:
            wait_for_live_algorithm(args.runtime_url, name)
            index = kinematics["collision_link_names"].index(name)
            record = fit_motion_link(
                name,
                mesh,
                num_spheres=args.num_spheres,
                seed=args.seed + 1000 * index,
                iterations=args.iterations,
            )
            record["geometry_sources"] = sources
            record["geometry_selection"] = selection
            record_path.write_text(json.dumps(record, indent=2))
        build_candidate(config, {name: record}, args.num_spheres)
        manifest["links"][name] = record
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(
            json.dumps(
                {
                    "link": name,
                    "sphere_budget": args.num_spheres,
                    "sphere_count": len(record["spheres"]),
                    "candidate_only": True,
                }
            ),
            flush=True,
        )
    candidate = build_candidate(config, manifest["links"], args.num_spheres)
    (output / "marvin_gui_dual.yml").write_text(yaml.safe_dump(candidate, sort_keys=False))
    manifest["candidate_count"] = sum(
        len(spheres) for spheres in candidate["robot_cfg"]["kinematics"]["collision_spheres"].values()
    )
    manifest["all_enabled_links_processed"] = set(names) == set(kinematics["collision_link_names"])
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
