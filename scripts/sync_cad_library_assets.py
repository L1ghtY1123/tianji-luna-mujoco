"""Publish derived runtime assets into the independent CAD library, preserving originals."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def publish(project, library):
    source = project / 'resources'
    destination = library / 'runtime'
    index = json.loads((library / 'cad/index.json').read_text(encoding='utf-8'))
    sources = {row['source']['sha256']: row['source'] for row in index['assets']}
    manifest = json.loads((source / 'cad/manifest.json').read_text(encoding='utf-8'))
    for asset in manifest['assets'].values():
        original = sources[asset['source']['sha256']]
        path = library / 'cad' / original['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != original['sha256']:
            raise ValueError('cad_library_source_hash_mismatch:' + str(path))
        asset['source']['library_path'] = 'cad/' + original['path']
        asset['source'].pop('path', None)
    for folder in ('cad', 'cad_catalog', 'registration_meshes', 'planning_meshes'):
        for path in (source / folder).rglob('*'):
            relative = path.relative_to(source)
            if not path.is_file() or (folder == 'cad' and ('source' in relative.parts or path.name == 'manifest.json')):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != path.read_bytes():
                raise ValueError('existing_library_asset_differs:' + str(target))
            shutil.copy2(path, target)
    for relative, content in {
        'cad/manifest.json': json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        'cad_catalog.json': (source / 'cad_catalog.json').read_text(encoding='utf-8'),
    }.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_text(encoding='utf-8') != content:
                raise ValueError('existing_library_manifest_differs:' + str(target))
            continue
        target.write_text(content, encoding='utf-8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(publish(args.project.resolve(), args.library.resolve()))
