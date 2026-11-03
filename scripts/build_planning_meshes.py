"""Build small CAD meshes for optimization; original CAD remains the hard-check world."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from build_registration_meshes import deviation

from luna_mujoco.cad_paths import cad_resource_root
from luna_mujoco.cad_perception import CadCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = CadCatalog.from_file()
    registration = cad_resource_root() / 'registration_meshes'
    reduced_records = json.loads((registration / 'manifest.json').read_text())['models']
    records = {}
    for record in catalog.records.values():
        geometry = catalog.geometry(record['mesh_key'])
        original = trimesh.Trimesh(geometry['vertices_m'], geometry['faces'], process=True)
        reduced_record = reduced_records[record['mesh_key']]
        reduced_path = registration / reduced_record['path']
        if (reduced_record['source_sha256'] != geometry['source_sha256']
                or reduced_record['units'] != 'mm'
                or reduced_record['frame_transform'] != np.eye(4).tolist()
                or hashlib.sha256(reduced_path.read_bytes()).hexdigest() != reduced_record['mesh_sha256']):
            raise ValueError('planning_mesh_source_mismatch')
        reduced = trimesh.load(reduced_path, force='mesh')
        reduced.apply_scale(.001)
        if not np.allclose(reduced.bounds, original.bounds, atol=.0001):
            raise ValueError('planning_mesh_canonical_frame_mismatch')
        path = args.output / (record['asset_id'] + '.ply')
        reduced.export(path)
        reduced = trimesh.load(path, force='mesh')
        source_mm, reduced_mm = original.copy(), reduced.copy()
        source_mm.apply_scale(1000.)
        reduced_mm.apply_scale(1000.)
        row = {'asset_id': record['asset_id'], 'source_mesh_sha256': geometry['mesh_sha256'],
            'source_sha256': geometry['source_sha256'], 'path': path.name,
            'mesh_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'units': 'm', 'frame': 'source_visual_canonical', 'frame_transform': np.eye(4).tolist(),
            'original_faces': len(original.faces), 'faces': len(reduced.faces),
            'bounds_difference_m': (reduced.bounds - original.bounds).tolist(),
            'original_to_reduced': deviation(source_mm, reduced_mm),
            'reduced_to_original': deviation(reduced_mm, source_mm),
            'error_measurement': 'sampled vertices, edge midpoints and face centroids; not a Hausdorff bound',
            'use': 'optimizer_only_original_CAD_required_for_execution_validation'}
        records[geometry['mesh_sha256']] = row
        print(json.dumps(row), flush=True)
    (args.output / 'manifest.json').write_text(json.dumps({'version': 1, 'meshes': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
