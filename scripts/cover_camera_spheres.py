"""Cover every authored camera triangle while preserving other robot spheres.

Input centers/radii come from a reviewed public cuRobo fit. Each camera triangle
is subdivided to at most 10 mm edges before assignment to one sphere; its radius
covers all three vertices with 0.5 mm reserve. Convexity covers the complete
surface, without enlarging spheres just to enclose a long triangulation edge.
The unchanged stand-body spheres remain separate from its camera geometry.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml


def camera_mesh(root, link):
    urdf = root / 'assets/marvin_urdf/marvin_with_grippers.urdf'
    node = ET.parse(urdf).getroot().find(f'link[@name="{link}"]')
    meshes = []
    for visual in node.findall('visual'):
        asset = visual.find('geometry/mesh')
        if asset is None or 'Cam' not in asset.get('filename', ''):
            continue
        mesh = trimesh.load(urdf.parent / asset.get('filename'), force='mesh', process=False)
        mesh.apply_scale(np.fromstring(asset.get('scale', '1 1 1'), sep=' '))
        origin = visual.find('origin')
        if origin is not None:
            transform = trimesh.transformations.euler_matrix(*np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            mesh.apply_transform(transform)
        meshes.append(mesh)
    if len(meshes) != 3:
        raise ValueError('Expected one mount, housing and optical window per camera')
    return trimesh.util.concatenate(meshes)


def cover(root, baseline, fitted, output):
    config = yaml.safe_load(baseline.read_bytes())
    candidate = yaml.safe_load(fitted.read_bytes())
    original = deepcopy(config['robot_cfg']['kinematics']['collision_spheres'])
    records = {}
    for link in ('Link_Stand', 'TCP_Link_L', 'TCP_Link_R'):
        mesh = camera_mesh(root, link)
        source_triangle_count = len(mesh.faces)
        vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=.01, max_iter=12)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        fitted_spheres = candidate['robot_cfg']['kinematics']['collision_spheres'][link]
        centers = np.array([s['center'] for s in fitted_spheres])
        radii = np.array([s['radius'] for s in fitted_spheres])
        distances = np.linalg.norm(mesh.triangles[:, :, None, :] - centers[None, None], axis=-1).max(axis=1)
        assignment = (distances - radii[None]).argmin(axis=1)
        retained, increases = [], []
        for index in np.unique(assignment):
            radius = max(radii[index], distances[assignment == index, index].max() + .0005)
            retained.append({'center': centers[index].tolist(), 'radius': float(radius)})
            increases.append(float(radius-radii[index]))
        spheres = deepcopy(original[link]) + retained if link == 'Link_Stand' else retained
        final_centers = np.array([s['center'] for s in spheres])
        final_radii = np.array([s['radius'] for s in spheres])
        gaps = (np.linalg.norm(mesh.triangles[:, :, None] - final_centers[None, None], axis=-1).max(axis=1) - final_radii).min(axis=1)
        assert gaps.max() <= -.000499 and len(spheres) <= 31
        config['robot_cfg']['kinematics']['collision_spheres'][link] = spheres
        records[link] = dict(source_triangle_count=source_triangle_count, covered_triangle_count=len(mesh.faces), sphere_count=len(spheres),
            maximum_triangle_gap_m=float(gaps.max()), maximum_radius_increase_m=max(increases),
            unchanged_body_spheres=len(original[link]) if link == 'Link_Stand' else 0)
    assert all(config['robot_cfg']['kinematics']['collision_spheres'][n] == v
               for n,v in original.items() if n not in records)
    output.mkdir(parents=True, exist_ok=True)
    (output/'marvin_gui_dual.yml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (output/'coverage.json').write_text(json.dumps(dict(source_fit_sha256=hashlib.sha256(fitted.read_bytes()).hexdigest(),
        source_baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(), reserve_m=.0005,
        scope='all camera mount/housing/window triangles; other link spheres unchanged', links=records),indent=2)+'\n')
    print(json.dumps(records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--fitted',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cover(args.root,args.baseline,args.fitted,args.output)
