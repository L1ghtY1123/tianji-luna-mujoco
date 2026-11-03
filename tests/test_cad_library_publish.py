import hashlib
import json
from pathlib import Path

from scripts.sync_cad_library_assets import publish


def test_publish_preserves_existing_manifest_bytes_and_adds_planning_assets(tmp_path, monkeypatch):
    project, library = tmp_path / 'project', tmp_path / 'library'
    resources, originals, runtime = project / 'resources', library / 'cad', library / 'runtime'
    for path in (resources / 'cad', resources / 'planning_meshes', originals, runtime / 'cad'):
        path.mkdir(parents=True)
    original = b'original CAD geometry'
    (originals / 'part.stl').write_bytes(original)
    source = {'path': 'part.stl', 'sha256': hashlib.sha256(original).hexdigest()}
    (originals / 'index.json').write_text(json.dumps({'assets': [{'source': source}]}))
    manifest = {'assets': {'part': {'source': source}}}
    (resources / 'cad/manifest.json').write_text(json.dumps(manifest))
    catalog = b'{\r\n  "assets": []\r\n}\r\n'
    (resources / 'cad_catalog.json').write_bytes(catalog)
    (resources / 'planning_meshes/part.ply').write_bytes(b'derived optimizer mesh')
    manifest['assets']['part']['source'] = {'sha256': source['sha256'], 'library_path': 'cad/part.stl'}
    existing = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').replace('\n', '\r\n').encode()
    (runtime / 'cad/manifest.json').write_bytes(existing)
    (runtime / 'cad_catalog.json').write_bytes(catalog)
    def no_manifest_rewrite(path, content, **kwargs):
        raise AssertionError('Existing equivalent manifests must not be rewritten across operating systems')
    monkeypatch.setattr(Path, 'write_text', no_manifest_rewrite)
    publish(project, library)
    assert (runtime / 'cad/manifest.json').read_bytes() == existing
    assert (runtime / 'cad_catalog.json').read_bytes() == catalog
    assert (originals / 'part.stl').read_bytes() == original
    assert (runtime / 'planning_meshes/part.ply').read_bytes() == b'derived optimizer mesh'
