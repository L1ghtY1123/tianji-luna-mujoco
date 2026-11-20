"""Luna-owned CAD selection for manual simulation setup, never task evidence."""
import hashlib
import json

import numpy as np
import trimesh

from .cad_perception import CadCatalog


class SimulationSceneEditor:
    def __init__(self, gateway, *, catalog=None):
        self.gateway = gateway
        self._catalog = catalog

    @property
    def catalog(self):
        if self._catalog is None:
            self._catalog = CadCatalog.from_file()
        return self._catalog

    def assets(self):
        return [{'id': record['asset_id'], 'label': record.get('display_name', record.get('label', record['asset_id']))}
                for record in self.catalog.records.values()]

    def import_mesh(self, asset_id):
        record = self.catalog.record(asset_id)
        geometry = self.catalog.geometry(record['mesh_key'])
        root = self.catalog.resource_root / 'cad'
        asset = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))['assets'][record['mesh_key']]
        parts = []
        for part in asset['collision']:
            path = (root / part['path']).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError('cad_collision_path_invalid')
            if hashlib.sha256(path.read_bytes()).hexdigest() != part['sha256']:
                raise ValueError('cad_collision_hash_mismatch')
            mesh = trimesh.load(path, force='mesh', process=False)
            parts.append({'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist()})
        resource = {'geometry': {'vertices': geometry['vertices_m'].tolist(), 'faces': geometry['faces'].tolist()},
                    'collision_parts': parts}
        identifier = self.gateway.register_mesh(resource)
        vertices = np.asarray(resource['geometry']['vertices'])
        colors = {'blue': [.05, .28, .8, 1.], 'black': [.03, .03, .03, 1.], 'pink': [.9, .3, .55, 1.]}
        return {'shape': 'mesh', 'asset_id': identifier, 'size_m': (vertices.max(axis=0)-vertices.min(axis=0)).tolist(),
                'color_rgba': colors.get(asset.get('default_color'), [.6, .6, .6, 1.]),
                'label': record.get('display_name', record.get('label',asset_id)), 'fixed': False, 'mass_kg': .2,
                'gravity_enabled': True, 'pose_wxyz': [.5, 0., .7, 1., 0., 0., 0.]}
