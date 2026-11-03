"""Shared CAD assets live in the independent library; fixtures remain portable."""
import os
from pathlib import Path


def cad_resource_root():
    project = Path(__file__).resolve().parents[2]
    configured = os.getenv('LUNA_CAD_ROOT')
    root = Path(configured) if configured else project.parent / 'CAD'
    assets = root / 'runtime'
    if configured:
        if not (assets / 'cad_catalog.json').is_file():
            raise ValueError('cad_library_runtime_assets_missing:' + str(assets))
        return assets.resolve()
    return project / 'resources'
