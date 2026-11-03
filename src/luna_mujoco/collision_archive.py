"""Immutable published collision scenes, with content-addressed static CAD assets."""
import json
import re
from copy import deepcopy
from pathlib import Path

ASSET = re.compile(r'collision/assets/v1-([a-f0-9]{64})\.json')


def freeze_collision_snapshot(scene):
    """Freeze the published scene without writing to the geometry worker filesystem."""
    snapshot = deepcopy({key: value for key, value in scene.items() if key != 'mesh_layers'})
    snapshot['mesh_layers'] = {}
    snapshot['mesh_assets'] = {}
    for name, layer in scene.get('mesh_layers', {}).items():
        digest = layer['mesh_sha256']
        if not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('collision_asset_hash_invalid')
        artifact = f'collision/assets/v1-{digest}.json'
        if artifact not in snapshot['mesh_assets']:
            snapshot['mesh_assets'][artifact] = deepcopy({key: layer[key] for key in ('vertices_local_m', 'faces')})
        snapshot['mesh_layers'][name] = {
            **deepcopy({key: value for key, value in layer.items() if key not in {'vertices_local_m', 'faces'}}),
            'mesh_artifact': artifact}
    return snapshot


def archive_collision_assets(directory, snapshot):
    """Write static assets once, including when referenced by raw algorithm records."""
    directory = Path(directory)
    for layer in snapshot.get('mesh_layers', {}).values():
        artifact = layer['mesh_artifact']
        if not ASSET.fullmatch(artifact):
            raise ValueError('collision_asset_path_invalid')
        destination = directory / artifact
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(snapshot['mesh_assets'][artifact],
                separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return {key: value for key, value in snapshot.items() if key != 'mesh_assets'}


def archive_collision_snapshot(directory, name, snapshot):
    """Run on the experiment writer; never fetch the mutable latest scene."""
    directory = Path(directory)
    archived = archive_collision_assets(directory, snapshot)
    path = directory / (name + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(archived, ensure_ascii=False, allow_nan=False), encoding='utf-8')
