"""Rebuild the three user CAD assets using public CPU mesh interfaces.

Requires the existing Python numpy/scipy/trimesh stack and Node. STEP import
uses the pinned official occt-import-js release, downloaded only by this tool.
The application consumes generated files and never downloads or converts CAD.
"""
import argparse
import base64
import hashlib
import io
import json
import subprocess
import tarfile
import urllib.request
from itertools import pairwise
from pathlib import Path

import numpy as np
import scipy
import trimesh

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "resources/cad"
CONVERTER_VERSION = "0.0.23"
CONVERTER_URL = "https://registry.npmjs.org/occt-import-js/-/occt-import-js-0.0.23.tgz"
CONVERTER_INTEGRITY = "sha512-RFfYQXYFX5C1mB1Aywm0ShcUKzXOr/VzTnlzhBSDJOR6YCAPt1HYCzeXWg1vwwjn/cUxwqRNhhtf1dlewoZYCQ=="
CONVERTER_HASHES = {
    "occt-import-js.js": "3fb44ce11d00611f9b3f3c5775d520ebab48930c1f08279b7b1316f05f0d3379",
    "occt-import-js.wasm": "33391fc9d94ea5c869a6718488bf0a9a464222bac9bdc764dfe1690cef281952",
}
SOURCES = {"blue_box": "blue_box_410x310x145.step",
           "pink_wrapped": "tripo_convert_891be992-2e91-4d83-b19a-dbbfcfa1acd1_flat_bottom_110mm.stl",
           "black_core": "THREADED_VISUAL_SAMPLE_MM.stl"}
LABELS = {"blue_box": "蓝色物料盒", "pink_wrapped": "粉色包裹", "black_core": "黑色拆包工件"}
COLORS = {"blue_box": "blue", "pink_wrapped": "pink", "black_core": "black"}
MASSES = {"blue_box": .5, "pink_wrapped": .08, "black_core": .06}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def converter(directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "occt-import-js.js"
    if target.is_file() and (directory / "occt-import-js.wasm").is_file():
        return target
    with urllib.request.urlopen(CONVERTER_URL, timeout=60) as response:
        raw = response.read()
    integrity = "sha512-" + base64.b64encode(hashlib.sha512(raw).digest()).decode()
    if integrity != CONVERTER_INTEGRITY:
        raise ValueError("occt-import-js archive integrity mismatch")
    (directory / "occt-import-js-0.0.23.tgz").write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for name in ("package/dist/occt-import-js.js", "package/dist/occt-import-js.wasm"):
            (directory / Path(name).name).write_bytes(archive.extractfile(name).read())
    return target


def step_mesh(path, node, module, temporary):
    script = temporary / "convert-step.cjs"
    script.write_text("""const fs=require('node:fs');
const create=require(process.argv[2]);
create().then(occt=>{
 const data=occt.ReadStepFile(fs.readFileSync(process.argv[3]),{
  linearUnit:'millimeter',linearDeflectionType:'absolute_value',
  linearDeflection:0.1,angularDeflection:0.15});
 if(!data.success)throw Error('STEP conversion failed');
 fs.writeFileSync(process.argv[4],JSON.stringify(data));
});
""", encoding="utf-8")
    result = temporary / "step-tessellation.json"
    subprocess.run([str(node), str(script), str(module), str(path), str(result)], check=True)
    parts = json.loads(result.read_text())["meshes"]
    return trimesh.util.concatenate([trimesh.Trimesh(
        np.array(part["attributes"]["position"]["array"]).reshape(-1, 3),
        np.array(part["index"]["array"]).reshape(-1, 3), process=False) for part in parts])


def clipped(mesh, normal, origin):
    return trimesh.intersections.slice_mesh_plane(mesh, normal, origin, cap=False)


def trim_pink_side_tab(raw):
    """User-requested removal of the negative-X tab; keep the full-height body.

    Source units are mm. The body above/below the tab starts at X >= -23.74;
    X=-24 therefore preserves it while removing the projecting strip. The
    original non-watertight STL remains unchanged and separately fingerprinted.
    """
    return clipped(raw, [1, 0, 0], [-24., 0, 0])


def bin_hulls(mesh):
    """Partition the CAD surface before convexification; never hull the cavity.

    The 5mm solid floor is separate. Twenty-four angular wall sectors, split
    below the lip, retain the empty center. Ribs/handle recesses within one
    sector can be conservatively filled; the imported visible mesh is exact.
    """
    floor = -.0675
    pieces = [clipped(mesh, [0, 0, -1], [0, 0, floor]).convex_hull]
    for low, high in ((floor, .0625), (.0625, .0725)):
        band = clipped(mesh, [0, 0, 1], [0, 0, low])
        band = clipped(band, [0, 0, -1], [0, 0, high])
        for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
            end = angle + 2*np.pi/24
            part = clipped(band, [-np.sin(angle), np.cos(angle), 0], [0, 0, 0])
            part = clipped(part, [np.sin(end), -np.cos(end), 0], [0, 0, 0])
            if len(part.vertices) >= 4:
                pieces.append(part.convex_hull)
    return pieces


def object_hulls(mesh):
    """Eight axial convex bands approximate irregular parts, including threads.

    This closes the non-watertight input for contact/inertia only. It does not
    repair the original visual triangles or claim the CAD material volume.
    """
    pieces = []
    edges = np.linspace(*mesh.bounds[:, 2], 9)
    for low, high in pairwise(edges):
        part = clipped(mesh, [0, 0, 1], [0, 0, low])
        part = clipped(part, [0, 0, -1], [0, 0, high])
        pieces.append(part.convex_hull)
    return pieces


def cavity_section(mesh, z):
    """Extract the innermost closed CAD cut loop containing the XY origin."""
    segments = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, z])[:, :, :2]
    neighbors = {}
    for start, end in np.round(segments, 10):
        first, second = tuple(start), tuple(end)
        if first == second:
            continue
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    if any(len(adjacent) != 2 for adjacent in neighbors.values()):
        raise ValueError("CAD cavity cut is not a union of closed loops")
    remaining = set(neighbors)
    containing = []
    while remaining:
        start = min(remaining)
        loop = [start]
        previous, current = None, start
        while True:
            following = min(neighbors[current] - {previous})
            if following == start:
                break
            loop.append(following)
            previous, current = current, following
        remaining.difference_update(loop)
        points = np.asarray(loop)
        following = np.roll(points, -1, axis=0)
        crossing = (points[:, 1] > 0) != (following[:, 1] > 0)
        edges = np.flatnonzero(crossing)
        x_at_zero = points[edges, 0] - points[edges, 1] * (
            following[edges, 0] - points[edges, 0]) / (following[edges, 1] - points[edges, 1])
        area = .5*np.sum(points[:, 0]*following[:, 1] - following[:, 0]*points[:, 1])
        if np.count_nonzero(x_at_zero > 0) % 2:
            containing.append((abs(area), points if area > 0 else points[::-1]))
    if not containing:
        raise ValueError("no closed cavity contour around CAD center")
    area, points = min(containing, key=lambda item: item[0])
    return {"z_m": z, "polygon_xy_m": points.tolist(), "area_m2": float(area),
            "source": "closed intersection of source CAD tessellation, including inner corner rounds",
            "scope": "shape prior / independent geometry QA; not an observed bin pose"}


def build(node, module, temporary, asset_ids=None):
    if asset_ids is None or 'blue_box' in asset_ids:
        for name, expected in CONVERTER_HASHES.items():
            if digest(module.parent / name) != expected:
                raise ValueError(f"unexpected occt-import-js 0.0.23 runtime: {name}")
    # A replacement asset must not rewrite unrelated meshes or their metadata.
    records = (json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))["assets"]
               if asset_ids is not None else {})
    for asset_id in SOURCES if asset_ids is None else dict.fromkeys(asset_ids):
        filename = SOURCES[asset_id]
        source = ASSETS / "source" / filename
        raw = (step_mesh(source, node, module, temporary) if source.suffix == ".step"
               else trimesh.load(source, force="mesh", process=False))
        raw.merge_vertices()
        processed = trim_pink_side_tab(raw) if asset_id == 'pink_wrapped' else raw
        center = processed.bounds.mean(axis=0)
        mesh = processed.copy()
        mesh.vertices = (mesh.vertices - center) * .001
        directory = ASSETS / asset_id
        directory.mkdir(parents=True, exist_ok=True)
        visual = directory / "visual.stl"
        mesh.export(visual)
        pieces = bin_hulls(mesh) if asset_id == "blue_box" else object_hulls(mesh)
        collisions = []
        for index, piece in enumerate(pieces):
            if not piece.is_watertight or not piece.is_winding_consistent or piece.volume <= 0:
                raise ValueError(f"invalid collision hull {asset_id}/{index}")
            path = directory / f"collision-{index:02d}.stl"
            piece.export(path)
            collisions.append({"path": path.relative_to(ASSETS).as_posix(), "sha256": digest(path),
                               "volume_m3": float(piece.volume), "vertices": len(piece.vertices)})
        records[asset_id] = {
            "id": asset_id, "label": LABELS[asset_id], "default_color": COLORS[asset_id],
            "size_m": mesh.extents.tolist(), "demo_mass_kg": MASSES[asset_id],
            "mass_basis": "explicit demo approximation; not CAD material density",
            "source": {"path": source.relative_to(ASSETS).as_posix(), "sha256": digest(source),
                       "units": "mm", "unit_evidence": "STEP SI_UNIT or STL authored header",
                       "bounds_mm": raw.bounds.tolist(), "aabb_center_mm": raw.bounds.mean(axis=0).tolist(),
                       "watertight": raw.is_watertight, "winding_consistent": raw.is_winding_consistent},
            "frame": "source axes retained; subtract source AABB center then multiply by 0.001; +Z upright",
            "visual": {"path": visual.relative_to(ASSETS).as_posix(), "sha256": digest(visual),
                       "triangles": len(mesh.faces)},
            "collision": collisions,
            "collision_approximation": ("CAD floor hull plus 24 wall sectors in two Z bands; cavity remains open"
                if asset_id == "blue_box" else "eight axial convex bands; concavities/threads locally filled"),
            "inertia_basis": "uniform density of convex pieces; mass split by piece volume to explicit total mass",
        }
        if asset_id == 'pink_wrapped':
            trimmed = directory / 'trimmed-source-mm.stl'
            processed.export(trimmed)
            records[asset_id]['preprocessing'] = {
                'operation': 'remove_negative_x_side_tab_then_tight_aabb',
                'plane_source_mm': {'normal': [1, 0, 0], 'origin': [-24., 0, 0]},
                'trimmed_source': {'path': trimmed.relative_to(ASSETS).as_posix(), 'sha256': digest(trimmed)},
                'processed_bounds_mm': processed.bounds.tolist(),
                'aabb_center_mm': center.tolist(),
                'original_source_preserved': True,
            }
            records[asset_id]['frame'] = 'source axes retained; subtract trimmed AABB center then multiply by 0.001; +Z upright'
        if asset_id == "blue_box":
            records[asset_id]["cavity_floor_z_m"] = -.0675
            records[asset_id]["cavity_axis"] = "+Z"
            records[asset_id]["cavity_bottom_section"] = cavity_section(mesh, -.0675 + .000001)
            records[asset_id]["cavity_floor_plus_1mm_section"] = cavity_section(mesh, -.0675 + .001)
            records[asset_id]["cavity_center_sections_m"] = [
                {"z": z*.001-.0725, "half_size_xy": [x*.001, y*.001]}
                for z, x, y in ((20,188.0357142857,133.0357142857),
                               (70,189.8214285714,134.8214285714),
                               (120,191.6071428571,136.6071428571),
                               (144,192.4642857143,137.4642857143))]
    manifest = {"schema_version": 1, "units": "m", "assets": records,
                "generator": {"script": "scripts/build_cad_assets.py", "script_sha256": digest(__file__),
                    "trimesh_version": trimesh.__version__, "scipy_version": scipy.__version__,
                    "converter": {"name": "occt-import-js", "version": CONVERTER_VERSION,
                                  "url": CONVERTER_URL, "integrity": CONVERTER_INTEGRITY,
                                  "files_sha256": CONVERTER_HASHES,
                                  "linear_deflection_mm": .1, "angular_deflection_rad": .15},
                    "sdk_modified": False}}
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: {"size_m": value["size_m"], "hulls": len(value["collision"])}
                      for key, value in records.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="node")
    parser.add_argument("--occt-module", type=Path)
    parser.add_argument("--asset", action="append", choices=tuple(SOURCES), help="Rebuild only selected assets in an existing manifest")
    args = parser.parse_args()
    work = ROOT / "data/cad-build"
    work.mkdir(parents=True, exist_ok=True)
    needs_step = args.asset is None or 'blue_box' in args.asset
    module_path = args.occt_module or (converter(work / "converter") if needs_step else work / 'unused-step-module')
    build(args.node, module_path.resolve(), work, args.asset)
