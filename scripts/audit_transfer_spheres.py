"""Compare fixed mesh surface samples and certified exposed sphere boundaries.

The central base uses the already reviewed tube/rib/top-flange components 21:29.
The previously omitted bottom flange and feet are reported as an unchanged scope
limitation. This measures finite local geometry, not continuous collision safety.
"""

import argparse
import hashlib
import json
from pathlib import Path

import trimesh
import yaml
from audit_avoidance_model import validate_configuration
from audit_collision_geometry import measure_sphere_boundaries, solid_components
from audit_collision_model import gap_statistics, sphere_modes
from fit_collision_model import load_visual_mesh, load_visual_parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-samples", type=int, default=12000)
    parser.add_argument("--sphere-samples", type=int, default=256)
    parser.add_argument("--links", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence file")
    project = Path(__file__).resolve().parents[1]
    models = [yaml.safe_load(path.read_bytes()) for path in (args.baseline, args.candidate)]
    validate_configuration(*models)
    configs = [model["robot_cfg"]["kinematics"] for model in models]
    links = args.links or [link for link in configs[0]["collision_spheres"]
                           if configs[0]["collision_spheres"][link] != configs[1]["collision_spheres"][link]]
    report = {"models": [hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.baseline, args.candidate)],
              "surface_samples": args.surface_samples, "sphere_samples": args.sphere_samples,
              "seed": 20261012, "non_sphere_configuration_unchanged": True,
              "sphere_counts": [sum(map(len, config["collision_spheres"].values())) for config in configs],
              "links": {}, "limitations": [
                  "Finite area samples, vertices and sphere directions do not prove continuous coverage or extrema.",
                  "Boundary distance is conditional on certified exterior and finite visible rays; open meshes remain ambiguous.",
                  "Static base scope matches previously reviewed tube/ribs/top flange; old bottom flange/feet omission retained.",
                  "Measurements are own-link local geometry; neighbouring link surfaces can cover additional boundaries.",
              ]}
    for link in links:
        mesh, assets, selection = load_visual_mesh(project, link)
        if link == "Link_Base":
            visuals, assets = load_visual_parts(project, link)
            components = visuals[0].split(only_watertight=False)
            if len(components) != 46:
                raise ValueError("Reviewed base component inventory changed")
            mesh = trimesh.util.concatenate(components[21:29])
            selection = {"component_indices": list(range(21, 29)), "source": "2026-10-08 static audit"}
        points, _ = trimesh.sample.sample_surface(mesh, args.surface_samples, seed=20261012)
        parts = solid_components({link: mesh})
        row = {"assets": assets, "selection": selection}
        for name, config in zip(("baseline", "candidate"), configs, strict=True):
            modes = sphere_modes(config, link)
            row[name] = {"count": len(modes["raw"]),
                         "raw_surface": gap_statistics(points, modes["raw"]),
                         "raw_vertices": gap_statistics(mesh.vertices, modes["raw"]),
                         "world_surface": gap_statistics(points, modes["world_buffered"]),
                         "raw_boundary": measure_sphere_boundaries(
                             link, {link: mesh}, {link: modes["raw"]}, parts,
                             sphere_samples=args.sphere_samples)}
        report["links"][link] = row
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"link": link, "counts": [row[name]["count"] for name in ("baseline", "candidate")],
                          "gap_mm": [row[name]["raw_surface"]["maximum_gap_m"] * 1000
                                     for name in ("baseline", "candidate")],
                          "overreach_mm": [row[name]["raw_boundary"]
                                           ["outward_distance_conditional_on_certified_exterior"]["maximum_mm"]
                                           for name in ("baseline", "candidate")]}), flush=True)


if __name__ == "__main__":
    main()
