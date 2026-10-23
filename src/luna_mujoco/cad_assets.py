"""Explicit user CAD priors and generated assets, never observed scene poses."""
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

CAD_ROOT = Path(__file__).resolve().parents[2] / "resources/cad"


@lru_cache(maxsize=1)
def _manifest():
    return json.loads((CAD_ROOT / "manifest.json").read_text(encoding="utf-8"))


def cad_asset(asset_id):
    """Return a copy of declared dimensions, provenance and collision approximation."""
    if not isinstance(asset_id, str) or asset_id not in _manifest()["assets"]:
        raise ValueError(f"unknown CAD asset {asset_id!r}")
    return deepcopy(_manifest()["assets"][asset_id])


def cad_catalog():
    """Public UI catalog; dimensions are native CAD metres, not editable scale."""
    return {"assets": [{key: record[key] for key in
                        ("id", "label", "size_m", "default_color", "demo_mass_kg")}
                       for record in deepcopy(_manifest()["assets"]).values()]}


def workpiece_height_prior(asset_id):
    """Declared demo posture: pink upright; unpacked black long axis horizontal."""
    return cad_asset(asset_id)['size_m'][0 if asset_id == 'black_core' else 2]


def _mesh_path(relative_path):
    path = (CAD_ROOT / relative_path).resolve()
    if not path.is_relative_to(CAD_ROOT.resolve()) or not path.is_file():
        raise ValueError("invalid generated CAD mesh path")
    return path
