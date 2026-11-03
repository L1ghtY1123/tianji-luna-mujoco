"""Prepare this project's CPU CAD registry using the existing service package.

Run with jodell-foundationpose:0.2.2-guangyu, or its unchanged service package
on PYTHONPATH. This script never imports the inference backend or uses a GPU.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

SDK_COMMIT = 'a1b694b83e633c2cb6115b9063d940a687759392'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scoped_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('CAD file is missing or outside its declared root')
    return path


def expected_models(project, cad_root):
    assets = cad_root / 'runtime' if (cad_root / 'runtime').is_dir() else project / 'resources'
    catalog = json.loads((assets / 'cad_catalog.json').read_text(encoding='utf-8'))
    manifest = json.loads((assets / 'cad/manifest.json').read_text(encoding='utf-8'))
    source_root = cad_root / 'cad'
    index = json.loads((source_root / 'index.json').read_text(encoding='utf-8'))
    sources = {row['asset_id']: row['source'] for row in index['assets']}
    models = {}
    for row in catalog['assets']:
        if row['geometry'] != 'rigid_reference':
            raise ValueError('Catalog model is not a rigid reference')
        source = sources[row.get('source_asset_id', row['asset_id'])]
        model = manifest['assets'][row['mesh_key']]['source']
        path = scoped_file(source_root, source['path'])
        if (source['units'] != 'mm' or model['units'] != 'mm'
                or source['sha256'] != row['source_sha256'] or model['sha256'] != row['source_sha256']
                or digest(path) != row['source_sha256']):
            raise ValueError('CAD source hash or units mismatch: ' + row['asset_id'])
        models[row['model_id']] = {'model_id': row['model_id'], 'asset_id': row['asset_id'],
            'source': path, 'source_sha256': row['source_sha256'],
            'center_m': [value / 1000 for value in model['aabb_center_mm']],
            'extents_mm': [hi - lo for lo, hi in zip(*model['bounds_mm'], strict=True)]}
    if set(models) != {'blue_box', 'black_core'}:
        raise ValueError('Expected exactly the configured blue_box and black_core models')
    return models


def verify_registration(record, expected):
    for field, value in {'model_id': expected['model_id'], 'source_sha256': expected['source_sha256'],
                         'units': 'mm', 'geometry': 'rigid_reference', 'sdk_commit': SDK_COMMIT}.items():
        if record.get(field) != value:
            raise ValueError('Registered model metadata mismatch: ' + field)
    expected_transform = [[1, 0, 0, expected['center_m'][0]], [0, 1, 0, expected['center_m'][1]],
                          [0, 0, 1, expected['center_m'][2]], [0, 0, 0, 1]]
    actual = record.get('T_source_canonical_m', [])
    if (len(actual) != 4 or any(len(row) != 4 for row in actual)
            or any(not math.isclose(a, b, abs_tol=1e-7, rel_tol=0)
                   for row, wanted in zip(actual, expected_transform, strict=True)
                   for a, b in zip(row, wanted, strict=True))):
        raise ValueError('Registered canonical frame mismatch')
    extents = record.get('extents_mm', [])
    if len(extents) != 3 or any(not math.isclose(a, b, abs_tol=.001, rel_tol=0)
                               for a, b in zip(extents, expected['extents_mm'], strict=True)):
        raise ValueError('Registered model extents mismatch')


def copy_seed_model(seed, destination, expected):
    record, cache = seed.get('blue_box')
    verify_registration(record, expected)
    models = destination.root / 'models'
    models.mkdir(parents=True, exist_ok=True)
    if (models / 'blue_box.json').exists():
        current, _ = destination.get('blue_box')
        verify_registration(current, expected)
        return
    lock = destination.root / '.registration-lock'
    lock.mkdir()  # Existing work is never overwritten or unlocked by this script.
    try:
        target_cache = destination.root / 'cache' / record['cache_key']
        target_cache.parent.mkdir(exist_ok=True)
        if not target_cache.exists():
            with tempfile.TemporaryDirectory(prefix='copy-blue-', dir=destination.root) as temporary:
                staged = Path(temporary) / 'cache'
                shutil.copytree(cache, staged)
                if digest(staged / 'model.ply') != record['mesh_sha256']:
                    raise ValueError('Seed mesh changed during copy')
                staged.rename(target_cache)
        # Atomic no-overwrite creation, matching the registry's immutable-model contract.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=models, delete=False,
                                         encoding='utf-8') as stream:
            temporary_record = Path(stream.name)
            json.dump(record, stream, indent=2)
        temporary_record.chmod(0o644)
        try:
            os.link(temporary_record, models / 'blue_box.json')
        finally:
            temporary_record.unlink()
    finally:
        lock.rmdir()
    current, _ = destination.get('blue_box')
    verify_registration(current, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--seed-registry', type=Path, required=True)
    parser.add_argument('--cad-root', type=Path, required=True)
    args = parser.parse_args()
    project, cad_root = args.project.resolve(), args.cad_root.resolve()
    destination_root, seed_root = args.registry.resolve(), args.seed_registry.resolve()
    if (destination_root == seed_root or destination_root.is_relative_to(seed_root)
            or destination_root.is_relative_to(cad_root)):
        raise ValueError('Destination registry must be independent of read-only seed and CAD roots')
    models = expected_models(project, cad_root)
    from tianji_foundationpose.registry import Registry

    from tianji_foundationpose.contracts import SDK_COMMIT as installed_commit

    if installed_commit != SDK_COMMIT:
        raise ValueError('Installed FoundationPose adapter commit differs from the pinned release')
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = Registry(destination_root)
    copy_seed_model(Registry(seed_root), destination, models['blue_box'])
    black = models['black_core']
    registered = destination.register('black_core', black['source'], 'mm', 'rigid_reference')
    verify_registration(registered, black)
    records = []
    for name in ('blue_box', 'black_core'):
        record, _ = destination.get(name)
        verify_registration(record, models[name])
        records.append(record)
    print(json.dumps({'registry': str(destination_root), 'sdk_commit': SDK_COMMIT,
                      'operation': 'copy_existing_blue_and_register_black_CPU_only', 'models': records}, indent=2))


if __name__ == '__main__':
    main()
