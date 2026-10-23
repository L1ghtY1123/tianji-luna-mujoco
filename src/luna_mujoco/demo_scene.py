"""Demo configuration and scene factory, separate from runtime visual evidence.

The scene factory is only for simulation composition. The controller may use
the configured destination intent and camera mapping, never source object truth.
"""

import json
from copy import deepcopy
from pathlib import Path

from .cad_assets import cad_asset
from .model_builder import normalize_scene
from .observation_motion import observation_poses

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "resources/demo-config.json"


def load_demo_config(path=CONFIG_PATH):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def demo_scene(object_count=None, config=None):
    """Build the configured pair, single/three-object row, or four-object grid."""
    config = load_demo_config() if config is None else config
    object_count = config["object_count"] if object_count is None else object_count
    if isinstance(object_count, bool) or not isinstance(object_count, int) or not 1 <= object_count <= 4:
        raise ValueError("object_count must be an integer from 1 to 4")
    bin_item = deepcopy(config["source_bin"])
    definition = config["source_objects"]
    source_asset = cad_asset(definition["asset_id"])
    size = source_asset["size_m"]
    floor_z = bin_item["position"][2] + cad_asset(bin_item["asset_id"])["cavity_floor_z_m"]
    objects = [bin_item, deepcopy(config["right_bin"])]
    if object_count == 1:
        offsets = [definition['single_object_offset_xy_m']]
    elif object_count == 4:
        grid = definition["four_object_grid"]
        offsets = [(x, y) for y in grid["y_offsets_from_bin_center_m"]
                   for x in grid["x_offsets_from_bin_center_m"]]
        if len(grid["x_offsets_from_bin_center_m"]) != 2 or len(grid["y_offsets_from_bin_center_m"]) != 2:
            raise ValueError("four_object_grid must contain two X and two Y offsets")
    elif object_count == 2:
        offsets = definition["two_object_offsets_xy_m"]
        if len(offsets) != 2 or any(len(offset) != 2 for offset in offsets):
            raise ValueError("two_object_offsets_xy_m must contain two XY offsets")
    else:
        start_x, start_y = definition["row_start_offset_xy_m"]
        step_x, step_y = definition["row_step_xy_m"]
        offsets = [(start_x + index * step_x, start_y + index * step_y)
                   for index in range(object_count)]
    for index, (offset_x, offset_y) in enumerate(offsets, start=1):
        objects.append({
            "id": f"pink_cylinder_{index}", "kind": "cad", "asset_id": definition["asset_id"],
            "color": "pink", "mass_kg": definition["mass_kg"],
            "position": [bin_item["position"][0] + offset_x, bin_item["position"][1] + offset_y,
                         floor_z + size[2] / 2 + definition["settling_gap_m"]],
            "size": list(size), "yaw_deg": definition.get("yaw_deg", 0),
        })
    return normalize_scene({"table": config["main_table"],
                            "support_tables": [],
                            "objects": objects})


def demo_region(config=None):
    """Return operator-defined intent for an unmarked square, not measured support."""
    config = load_demo_config() if config is None else config
    table, region = config["main_table"], config["destination"]
    offset = region["center_offset_from_main_table_m"]
    center = [table["position"][0] + offset[0], table["position"][1] + offset[1]]
    half = region["size_m"] / 2
    polygon = [[center[0] + x, center[1] + y]
               for x, y in ((-half, -half), (half, -half), (half, half), (-half, half))]
    return {"region_id": region["region_id"], "center_world_xy_m": center, "size_m": region["size_m"],
            "region_polygon_world_xy_m": polygon,
            "physical_marker": False, "source": "configured_destination_intent"}


def demo_observation_poses(config=None):
    """Load calibrated endpoints; existing validator checks joint ranges."""
    config = load_demo_config() if config is None else config
    path = ROOT / config["observation_poses_resource"]
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("model_validated") is not True:
        raise ValueError("demo_observation_poses_require_model_validation")
    return observation_poses({"observation": record["joints_rad"]})
