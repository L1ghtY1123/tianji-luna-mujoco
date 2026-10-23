"""Prepare the actual recorded release boundary; no motors, JEV, services or GPU."""

import asyncio
import copy
import json
from pathlib import Path

import numpy as np

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.cameras import letterbox_view
from luna_mujoco.perception_mapping import decode_mapping_payload
from luna_mujoco.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/visible-cloud-work/grasp-place-right-9"
OUTPUT = ROOT / "data/visible-cloud-work/b9-release-boundary"
FRAME = "rgbd-2-2e55c22c6c7c4baba4b6618201990f59"


class PreparedRequest(Exception):
    """Stop before sending a request; never manufacture a successful response."""


class RecordedRobot:
    def __init__(self, capture):
        self.capture = capture

    def capture_rgbd(self):
        return copy.deepcopy(self.capture)

    def robot_snapshot(self):
        return {key: copy.deepcopy(self.capture[key]) for key in ("generation", "time", "arms")}


async def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage = json.loads((DATA / "stage-2.json").read_text(encoding="utf-8"))
    old_mapping = json.loads((DATA / "gpu-requests/020-perception-build.json").read_text(encoding="utf-8"))["body"]
    old_decoded, _ = decode_mapping_payload(old_mapping)
    previous = np.load(DATA / f"perception-{old_mapping['observation_set_id']}.npz", allow_pickle=True)["observation"].item()
    current = np.load(ROOT / "data/visible-cloud-work/b9-release-replay-observation.npz", allow_pickle=True)["observation"].item()
    capture = np.load(DATA / f"rgbd-{FRAME}.npz", allow_pickle=True)["capture"].item()
    assert current["observation_id"] == capture["observation_id"] == FRAME
    assert current["generation"] == capture["generation"] == stage["run"]["generation"]
    names, parity = {}, {}
    old_ids = set().union(*(set(view["instance_masks"]) for view in old_decoded["views"].values()))
    for obj in previous["objects"]:
        candidates = []
        for identifier in old_ids:
            differences, evidence = [], 0
            for camera, view in previous["views"].items():
                original = letterbox_view(view)["instance_masks"].get(obj["id"])
                recorded = old_decoded["views"][camera]["instance_masks"].get(identifier)
                shape = np.asarray(old_decoded["views"][camera]["depth_left_m"]).shape
                first = np.zeros(shape, bool) if original is None else np.asarray(original, bool)
                second = np.zeros(shape, bool) if recorded is None else np.asarray(recorded, bool)
                differences.append(int(np.count_nonzero(first ^ second)))
                evidence += int(np.count_nonzero(first))
            if not any(differences) and evidence:
                candidates.append(identifier)
        assert len(candidates) == 1, "prior semantic identity requires exact recorded mask support"
        names[obj["id"]] = candidates[0]
        parity[obj["id"]] = {"recorded_id": candidates[0], "all_view_masks_byte_equal": True}
    assert len(set(names.values())) == len(names)
    for obj in previous["objects"]:
        obj["id"] = names[obj["id"]]
    for view in previous["views"].values():
        view["instance_masks"] = {names.get(identifier, identifier): mask for identifier, mask in view["instance_masks"].items()}
    source = next(obj for obj in previous["objects"] if obj["id"] == stage["run"]["target_id"])
    assert np.array_equal(source["points_world_m"], np.asarray(old_mapping["selected_surface_world_m"], np.float32))
    opening = json.loads((DATA / "gpu-requests/026-carrying-validate.json").read_text(encoding="utf-8"))["body"]
    holding = copy.deepcopy(stage["holding"])
    holding["points_tool_m"] = copy.deepcopy(opening["carrying_binding"]["points_tool_m"])
    assert holding["state"] == "release_pending" and not holding["attached"] and not holding["verified"]
    plan = next(event["data"]["result"] for event in stage["events"]
        if event["kind"] == "skill_complete" and event.get("title") == "plan_place")
    algorithms = Algorithms()
    def recorded_observe(measured, *, instruction, checkpoint, source_hint):
        assert measured["observation_id"] == FRAME and checkpoint == "release"
        assert source_hint == {"kind": source["kind"], "color": source["color"]}
        return copy.deepcopy(current)  # Actual frozen Perception replay, not synthesized points.
    algorithms.observe = recorded_observe
    robot = RecordedRobot(capture)
    runner = Runner(robot, algorithms, None, OUTPUT / "logs")
    runner.state = copy.deepcopy(stage["run"])
    runner.holding_state = copy.deepcopy(holding)
    context = {"observation": previous, "holding": holding, "place_plan": plan,
        "planning_settings": copy.deepcopy(stage["run"]["planning_settings"])}
    observation = await runner.observe(context, checkpoint="release")
    assert next(obj for obj in observation["objects"] if obj["id"] == runner.state["target_id"])
    assert next(obj for obj in observation["objects"] if obj["id"] == runner.state["destination_id"])
    prepared = {}
    def prepare_request(path, body):
        prepared.update(path=path, body=body)
        raise PreparedRequest("CPU preparation ends before the service boundary")
    algorithms.request = prepare_request
    try:
        algorithms.build_scene(observation, runner.state["target_id"],
            planning_settings=context["planning_settings"], exclusion_state="released")
    except PreparedRequest:
        pass
    assert prepared["path"] == "/perception/build" and prepared["body"]["exclusion_state"] == "released"
    _, identity = decode_mapping_payload(prepared["body"])
    settings = context["planning_settings"]
    goal = runner.offset_pose(holding["release_tcp"], settings["place_retreat_axis"],
        settings["place_retreat_offset_m"], settings["place_retreat_in_tool_frame"])
    motion = algorithms._motion_request(observation, runner.state["arm"], identity, goal, settings)
    assert "carrying_binding" not in motion
    packet = {"mapping": prepared, "expected_scene_identity": identity,
        "retreat": {"path": "/tools/plan_to_pose", "body": motion}, "goal": goal,
        "provenance": {"physical_motion": False, "oracle_used": False, "service_used": False,
            "frame": FRAME, "prior_mask_identity_proof": parity, "prior_source_points_byte_equal": True,
            "release_tcp_source": "actual recorded gripper-opening feedback",
            "current_observation_source": "fixed Perception replay of actual release RGBD/SAM3",
            "retreat_goal_source": "Runner.offset_pose(actual recorded releaseTCP, applied retreat settings)",
            "runner_release_association_passed": True,
            "objects": [{"id": obj["id"], "kind": obj["kind"], "point_count": len(obj["points_world_m"])}
                for obj in observation["objects"]]}}
    (OUTPUT / "prepared-requests.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")
    np.savez_compressed(OUTPUT / "associated-observation.npz", observation=np.array(observation, dtype=object))
    print(json.dumps(packet["provenance"] | {"prepared_scene_token": identity["scene_token"],
        "goal_tcp_position_m": np.asarray(goal)[:3, 3].tolist()}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
