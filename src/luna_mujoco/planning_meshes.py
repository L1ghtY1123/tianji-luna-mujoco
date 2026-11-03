"""Static CAD proxies for optimizer costs only; full CAD remains validation geometry."""
import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from .cad_paths import cad_resource_root


@lru_cache(maxsize=8)
def _mesh(path, digest):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('planning_mesh_hash_mismatch')
    mesh = trimesh.load(path, force='mesh', process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('planning_mesh_invalid')
    return mesh.vertices.tolist(), mesh.faces.tolist()


def optimizer_scene(model):
    root = cad_resource_root() / 'planning_meshes'
    manifest = root / 'manifest.json'
    if not model.get('mesh') or not manifest.exists():
        return model, []
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('version') != 1:
        raise ValueError('planning_mesh_manifest_invalid')
    result, evidence = deepcopy(model), []
    for name, geometry in model['mesh'].items():
        record = data['meshes'].get(geometry.get('mesh_sha256'))
        if geometry.get('geometry_source') != 'registered_cad' or record is None:
            continue
        path = (root / record['path']).resolve()
        if (not path.is_relative_to(root.resolve()) or record['units'] != 'm'
                or record['frame'] != 'source_visual_canonical'
                or record['frame_transform'] != np.eye(4).tolist()
                or record['asset_id'] != geometry['asset_id']
                or record['source_sha256'] != geometry['source_sha256']):
            raise ValueError('planning_mesh_source_mismatch')
        vertices, faces = _mesh(str(path), record['mesh_sha256'])
        # The pinned SDK caches Warp meshes by name, not path. These temporary
        # SDK cache keys never change the map or the permanent inventory IDs.
        optimizer_name = name + '::optimizer::' + record['mesh_sha256']
        del result['mesh'][name]
        result['mesh'][optimizer_name] = {**geometry, 'vertices': vertices, 'faces': faces,
                                          'file_path': str(path)}
        evidence.append({'obstacle': name, 'optimizer_obstacle': optimizer_name,
            'source_mesh_sha256': geometry['mesh_sha256'],
            'optimizer_mesh_sha256': record['mesh_sha256'], 'optimizer_faces': len(faces),
            'original_faces': record['original_faces'], 'validation_geometry': 'original_cad'})
    return result, evidence
