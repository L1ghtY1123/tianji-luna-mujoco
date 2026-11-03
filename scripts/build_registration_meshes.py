"""Offline decimation of frozen registry meshes without moving their source frame."""
import argparse
import hashlib
import json
from pathlib import Path

import igl
import numpy as np
import trimesh


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surface_samples(mesh):
    edges = mesh.vertices[mesh.edges_unique].mean(axis=1)
    return np.vstack([mesh.vertices, mesh.triangles_center, edges])


def deviation(source, target):
    points = surface_samples(source)
    squared, _, _ = igl.point_mesh_squared_distance(points, np.asarray(target.vertices, dtype=np.float64),
                                                   np.asarray(target.faces, dtype=np.int64))
    distance = np.sqrt(np.maximum(squared, 0))
    return {'sample_count': len(points), 'max_mm': float(distance.max()),
            'p99_mm': float(np.quantile(distance, .99)), 'rms_mm': float(np.sqrt(squared.mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Frozen MODEL.json and canonical MODEL.ply in mm')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--faces', type=int, required=True)
    parser.add_argument('--aggression', type=int, default=7)
    parser.add_argument('--prepared', type=Path, help='Audit existing Open3D MODEL-FACES.ply candidates')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = {}
    for path in sorted(args.input.glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        model_id = record['model_id']
        original = args.input / (model_id + '.ply')
        if digest(original) != record['mesh_sha256']:
            raise ValueError('original_registration_mesh_hash_mismatch')
        mesh = trimesh.load(original, force='mesh')
        if args.prepared:
            reduced = trimesh.load(args.prepared / f'{model_id}-{args.faces}.ply', force='mesh')
            recipe = {'open3d': '0.19.0', 'method': 'simplify_quadric_decimation',
                      'boundary_weight': 1.0, 'maximum_error': 'default infinity'}
        else:
            reduced = mesh.simplify_quadric_decimation(face_count=args.faces, aggression=args.aggression)
            recipe = {'trimesh': trimesh.__version__, 'fast_simplification': '0.1.13',
                      'aggression': args.aggression}
        output = args.output / (model_id + '.ply')
        reduced.export(output)
        # Validate the serialized mesh actually consumed by the service.
        reduced = trimesh.load(output, force='mesh')
        records[model_id] = {
            'path': output.name, 'source_sha256': record['source_sha256'],
            'original_mesh_sha256': record['mesh_sha256'], 'mesh_sha256': digest(output),
            'source': 'offline_quadric_decimation', 'units': 'mm', 'frame': 'original_registry_canonical',
            'frame_transform': np.eye(4).tolist(), 'original_face_count': len(mesh.faces),
            'face_count': len(reduced.faces), 'requested_face_count': args.faces,
            'original_watertight': mesh.is_watertight, 'reduced_watertight': reduced.is_watertight,
            'bounds_difference_mm': (reduced.bounds - mesh.bounds).tolist(),
            'original_to_reduced': deviation(mesh, reduced), 'reduced_to_original': deviation(reduced, mesh),
            'deviation_method': 'all vertices, unique edge midpoints and face centroids; sampled not certified Hausdorff',
            'recipe': recipe}
        print(json.dumps({'model_id': model_id, **records[model_id]}), flush=True)
    (args.output / 'manifest.json').write_text(json.dumps({'version': 1, 'models': records}, indent=2))


if __name__ == '__main__':
    main()
