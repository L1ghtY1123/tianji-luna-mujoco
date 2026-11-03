"""Published low-poly assets keep the original coordinate and grasp contracts."""
import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1] / 'resources'


def test_every_active_cad_visual_uses_at_most_one_thousand_faces():
    manifest = json.loads((ROOT / 'cad/manifest.json').read_text(encoding='utf-8'))
    for record in manifest['assets'].values():
        mesh = trimesh.load(ROOT / 'cad' / record['visual']['path'], force='mesh')
        assert 0 < len(mesh.faces) <= 1000, record['id']
        assert np.isfinite(mesh.vertices).all()
        np.testing.assert_allclose(mesh.extents, record['size_m'], atol=1e-6, rtol=0)


def test_registration_and_collision_use_the_same_published_surface():
    from luna_mujoco.cad_perception import CadCatalog

    catalog = CadCatalog.from_file(ROOT / 'cad_catalog.json')
    registration = json.loads((ROOT / 'registration_meshes/manifest.json').read_text(encoding='utf-8'))
    for record in catalog.records.values():
        geometry = catalog.geometry(record['mesh_key'])
        metadata = registration['models'][record['model_id']]
        assert metadata['source_sha256'] == geometry['source_sha256']
        mesh = trimesh.load(ROOT / 'registration_meshes' / metadata['path'], force='mesh', process=False)
        assert len(mesh.faces) <= 1000
        np.testing.assert_allclose(mesh.triangles * .001,
            geometry['vertices_m'][geometry['faces']], atol=2e-8, rtol=0)


def test_no_superseded_high_poly_mesh_remains_in_active_assets():
    for directory in ('cad', 'registration_meshes', 'planning_meshes'):
        for path in (ROOT / directory).rglob('*'):
            if path.suffix.lower() in {'.stl', '.ply', '.obj'}:
                assert len(trimesh.load(path, force='mesh', process=False).faces) <= 1000, path
    assert not (ROOT / 'planning_meshes/manifest.json').exists()
