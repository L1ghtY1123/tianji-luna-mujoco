"""Archived CAD shell geometry is rejected before measured lift planning."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.demo_lift import source_clearance_lift


def frozen():
    directory = Path(__file__).parent / "fixtures"
    value = json.loads((directory / "cad_lift_dafba328.json").read_text(encoding="utf-8"))
    with np.load(directory / "cad_lift_dafba328.npz", allow_pickle=False) as arrays:
        for obj in value["snapshot"]["objects"]:
            for key in ("points_world_m", "completed_geometry_points_world_m"):
                if key in obj:
                    obj[key] = arrays[obj[key]["fixture_array"]]
    return value


def derive(value):
    return source_clearance_lift(value["snapshot"], value["target_id"], value["arm"],
                                 value["candidates"], value["scene"], value["settings"])


def test_archived_cad_source_bounds_require_rebuild_instead_of_influencing_lift():
    value = frozen()
    before = copy.deepcopy(value['scene'])
    with pytest.raises(ValueError, match='scene_geometry_policy_requires_rebuild:cad_shell'):
        derive(value)
    assert value['scene'] == before


@pytest.mark.parametrize('damage', ['missing_part', 'stale_part', 'hash', 'disabled', 'wrong_id'])
def test_cad_shell_is_rejected_regardless_of_registration_metadata(damage):
    value = frozen()
    layers = value['scene']['collision_layer_bounds']['layers']
    name = next(name for name in layers if '::bin_shell::' in name)
    if damage == 'missing_part':
        layers.pop(name)
    elif damage == 'disabled':
        value['scene']['disabled_obstacle_names'].append(name)
    else:
        layers[name][damage] = 'changed'
    with pytest.raises(ValueError, match='scene_geometry_policy_requires_rebuild:cad_shell'):
        derive(value)
