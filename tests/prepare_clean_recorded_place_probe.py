"""Prepare separate GPU packets from reviewed recorded observations/bindings."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from jev_mujoco.algorithms import Algorithms
from jev_mujoco.perception_mapping import decode_mapping_payload
from jev_mujoco.placement import public_observation, release_goal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / options.recording
    observation_path, binding_path = root / options.observation, root / options.binding
    observation = np.load(observation_path, allow_pickle=True)["observation"].item()
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    stage_path = folder / "stage-1.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    original_path = folder / "gpu-requests/019-carrying-prepare.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))["body"]
    settings = original["planning_settings"]
    arm = binding["arm"]
    assert settings["place_release_height_m"] == .04
    assert binding["observation_id"] == observation["observation_id"]
    for field in ("voxel_size_m", "margin_m", "sphere_budget", "uncertainty_m"):
        assert binding[field] == original["carrying_binding"][field]
    source = next(obj for obj in observation["objects"] if obj["id"] == binding["instance_id"])
    bins = [obj for obj in observation["objects"] if obj["kind"] == "bin" and obj["color"] == "blue"]
    assert len(bins) == 1 and bins[0].get("placement_reference")
    packets = {}
    algorithms = Algorithms.__new__(Algorithms)

    def capture(path, body):
        packets[path] = body
        return decode_mapping_payload(body)[1]

    algorithms.request = capture
    scene = algorithms.build_scene(observation, source["id"], planning_settings=settings, exclusion_state="held")
    binding = deepcopy(binding) | {"scene_token": scene["scene_token"]}
    holding = deepcopy(stage["holding"]) | binding
    holding["observation"] = public_observation(observation) | {"arms": observation["arms"]}
    holding["attached"] = False
    goal = release_goal(source, bins[0], holding, release_height_m=settings["place_release_height_m"])
    prepare = Algorithms._motion_request(observation, arm, scene, observation["arms"][arm]["tcp"], settings)
    prepare["carrying_binding"] = binding
    place = Algorithms._motion_request(observation, arm, scene, goal["T_world_tcp"], settings)
    place.update(carrying_binding=binding,
        approach_axis=settings["place_approach_axis"], approach_offset_m=settings["place_approach_offset_m"],
        approach_in_tool_frame=settings["place_approach_in_tool_frame"],
        lift_axis=settings["place_retreat_axis"], lift_offset_m=settings["place_retreat_offset_m"],
        lift_in_tool_frame=settings["place_retreat_in_tool_frame"])
    manifest = {"physical_motion": False, "oracle_used": False, "arm": arm,
        "observation_id": observation["observation_id"], "scene_token": scene["scene_token"],
        "binding_point_count": len(binding["points_tool_m"]), "current_source_point_count": len(source["points_world_m"]),
        "goal": goal, "historic_capture_pixel_convention": "original_pre_sensor_correction_preserved",
        "input_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (observation_path, binding_path, stage_path, original_path)}}
    output = root / options.output
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("mapping.json", {"path": "/perception/build", "body": packets["/perception/build"]}),
                        ("prepare.json", {"path": "/carrying/prepare", "body": prepare}),
                        ("place.json", {"path": "/tools/plan_place", "body": place}),
                        ("observation.json", holding["observation"]), ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(value), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("arm", "observation_id", "scene_token", "binding_point_count", "current_source_point_count")}))
    print("goal", place["goal_position_m"], "approach", place["approach_offset_m"])


if __name__ == "__main__":
    main()
