"""Cover the authored palm collision boxes, without changing other robot geometry.

Each sphere encloses one rectangular cell. The eight cell vertices certify the
whole convex cell, and the regular grid tiles the complete URDF collision box.
This is a project-owned analytic primitive cover, not a replacement SDK fitter.
"""

import argparse
import copy
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
LINKS = ("gripL_Base", "gripR_Base")
DIVISIONS = np.array([2, 2, 3])
MAXIMUM_ROBOT_SPHERES = 365
NUMERICAL_PADDING_M = 1e-7


def cover_box(urdf, link):
    collisions = ET.parse(urdf).getroot().findall(f"link[@name='{link}']/collision")
    if len(collisions) != 1 or collisions[0].find("geometry/box") is None:
        raise ValueError(f"Expected one authored collision box for {link}")
    collision = collisions[0]
    size = np.fromstring(collision.find("geometry/box").get("size", ""), sep=" ")
    origin = collision.find("origin")
    center = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    if (any(value.shape != (3,) or not np.isfinite(value).all() for value in (size, center, rpy))
            or np.any(size <= 0)):
        raise ValueError(f"Invalid authored collision box for {link}")
    rotation = Rotation.from_euler("xyz", rpy).as_matrix()
    cell_size = size / DIVISIONS
    radius = float(np.linalg.norm(cell_size / 2) + NUMERICAL_PADDING_M)
    spheres = []
    cells = []
    for index in itertools.product(*(range(int(n)) for n in DIVISIONS)):
        lower = -size / 2 + np.asarray(index) * cell_size
        upper = lower + cell_size
        local_center = (lower + upper) / 2
        sphere_center = rotation @ local_center + center
        spheres.append({"center": sphere_center.tolist(), "radius": radius})
        cells.append({"minimum_box_frame_m": lower.tolist(), "maximum_box_frame_m": upper.tolist()})
    return spheres, {
        "link": link, "source": "authored_urdf_collision_box", "size_m": size.tolist(),
        "origin_xyz_m": center.tolist(), "origin_rpy_rad": rpy.tolist(),
        "divisions": DIVISIONS.tolist(), "numerical_padding_m": NUMERICAL_PADDING_M,
        "sphere_count": len(spheres), "radius_m": radius, "cells": cells,
        "proof": "Each sphere contains all eight vertices of its convex cell; cells tile the entire box.",
    }


def build_candidate(source, urdf):
    candidate = copy.deepcopy(source)
    spheres = candidate["robot_cfg"]["kinematics"]["collision_spheres"]
    certificates = {}
    for link in LINKS:
        if link not in spheres:
            raise ValueError(f"Missing configured collision link {link}")
        spheres[link], certificates[link] = cover_box(urdf, link)
    count = sum(map(len, spheres.values()))
    if count > MAXIMUM_ROBOT_SPHERES:
        raise ValueError(f"Palm coverage exceeds the explicit {MAXIMUM_ROBOT_SPHERES}-sphere robot budget")
    return candidate, {"robot_sphere_count": count, "maximum_robot_spheres": MAXIMUM_ROBOT_SPHERES,
                       "links": certificates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "resources/curobo/marvin_gui_dual.yml")
    parser.add_argument("--urdf", type=Path, default=ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--certificate", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.certificate.exists():
        raise ValueError("Choose new output paths; source and existing evidence are never overwritten")
    candidate, certificate = build_candidate(yaml.safe_load(args.source.read_bytes()), args.urdf)
    args.output.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    args.certificate.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
