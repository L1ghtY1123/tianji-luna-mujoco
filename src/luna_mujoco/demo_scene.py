"""Operator task regions and observation poses; physical scenes run in Simulation."""
import json
from pathlib import Path

from .observation_motion import observation_poses

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "resources/demo-config.json"


def load_demo_config(path=CONFIG_PATH):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def demo_region(config=None):
    """Return operator-defined intent for an unmarked square, not measured support."""
    config = load_demo_config() if config is None else config
    region = config['destination']
    center = list(region['center_world_xy_m'])
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
