"""CAD assets for MuJoCo scene construction, not perception or planning."""
import json
from copy import deepcopy
from functools import lru_cache

from .cad_paths import cad_resource_root

CAD_ROOT = cad_resource_root() / "cad"


@lru_cache(maxsize=1)
def _manifest():
    return json.loads((CAD_ROOT / "manifest.json").read_text(encoding="utf-8"))


def cad_asset(asset_id):
    """Return a copy of declared dimensions, provenance and collision approximation."""
    if not isinstance(asset_id, str) or asset_id not in _manifest()["assets"]:
        raise ValueError(f"unknown CAD asset {asset_id!r}")
    record = deepcopy(_manifest()["assets"][asset_id])
    if asset_id == 'pink_wrapped':
        record['simulation_geometry'] = 'cad_bounds_box'
        record['label'] = '粉色长方体（CAD 去凸条后包围盒）'
    return record


def cad_catalog():
    """Public UI catalog; dimensions are native CAD metres, not editable scale."""
    return {"assets": [{key: record[key] for key in
                        ("id", "label", "size_m", "default_color", "demo_mass_kg")}
                       for record in (cad_asset(key) for key in _manifest()['assets'])]}


def _mesh_path(relative_path):
    path = (CAD_ROOT / relative_path).resolve()
    if not path.is_relative_to(CAD_ROOT.resolve()) or not path.is_file():
        raise ValueError("invalid generated CAD mesh path")
    return path
