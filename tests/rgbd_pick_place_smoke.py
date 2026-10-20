"""Real simulation + RGBD/SAM3/GraspGen/cuRobo/JEV acceptance, never fake decisions."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("JEV_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor")]

from dotenv import load_dotenv
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

from jev_mujoco.algorithms import Algorithms
from jev_mujoco.decisions import Decisions
from jev_mujoco.planning_settings import PlanningSettings
from jev_mujoco.runner import Runner, public_summary
from jev_mujoco.simulation import Simulation
from jev_mujoco.visual_assistance import VisualAssistance


class RecordedAlgorithms(Algorithms):
    """Keep exact simulated sensor/GPU inputs for independent failure replay."""
    def __init__(self, output):
        self.output = output
        self.sequence = 0
        (output / "gpu-requests").mkdir(parents=True, exist_ok=True)
        super().__init__()

    def _request(self, path, body=None):
        if body is not None:
            self.sequence += 1
            artifact = self.output / "gpu-requests" / f"{self.sequence:03d}-{path.strip('/').replace('/', '-')}.json"
            artifact.write_text(json.dumps({"path": path, "body": body}, allow_nan=False), encoding="utf-8")
        return super()._request(path, body)

    def observe(self, capture, instruction="", *, checkpoint=None, source_hint=None):
        capture_path = self.output / f"rgbd-{capture['observation_id']}.npz"
        np.savez_compressed(capture_path, capture=np.array(capture, dtype=object))
        detector = self.perception.detector
        records = []

        class RecordedDetector:
            def segment(self, rgb, text, timestamp_us):
                camera = next(key for key, view in capture["views"].items() if view["rgb"] is rgb)
                result = detector.segment(rgb, text, timestamp_us)
                records.append({"camera_id": camera, "text": text, "timestamp_us": timestamp_us,
                                "detections": result})
                return result

        self.perception.detector = RecordedDetector()
        try:
            observation = super().observe(capture, instruction, checkpoint=checkpoint, source_hint=source_hint)
            np.savez_compressed(self.output / f"perception-{capture['observation_id']}.npz",
                                observation=np.array(observation, dtype=object))
            return observation
        finally:
            self.perception.detector = detector
            np.savez_compressed(self.output / f"sam3-{capture['observation_id']}.npz",
                                records=np.array(records, dtype=object), source_hint=np.array(source_hint, dtype=object))


class ScoredSimulation(Simulation):
    """Keep physical truth only in the acceptance scorer, outside sensor returns."""

    def __init__(self):
        self.gripper_oracle = []
        super().__init__()

    def set_gripper(self, arm, closed, generation):
        record = {"arm": arm, "closed_command": closed, "before": self.snapshot()}
        self.gripper_oracle.append(record)
        result = super().set_gripper(arm, closed, generation)
        record["after"] = self.snapshot()
        return result


async def main(output, timeout_s, side, task, retry_release_height_m=None, bin_size_xy_m=(.18, .16),
               release_height_m=None, attachment_voxel_size_m=None, bin_position_xy_m=None, source_position_xy_m=None, instruction_override=None, planning_attempts=1):
    load_dotenv(ROOT / ".env", override=False)
    output.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY_not_configured")
    model = TypeSafeModel(os.environ.get("JEV_MODEL", "jev-1.13.0"), provider=TypeSafeProvider(
        api_key=key, base_url=os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")))
    simulation = ScoredSimulation()
    simulation.start()
    runner = None
    try:
        sign = 1 if side == "left" else -1
        simulation.configure({"objects": [
            {"id": "fixture_red_cube", "kind": "cube", "color": "red", "position": [*(source_position_xy_m or (.43, sign * .15)), .625],
             "size": [.05, .05, .05]},
            {"id": "fixture_blue_bin", "kind": "bin", "color": "blue", "position": [*(bin_position_xy_m or (.65, sign * .15)), .65],
             "size": [*bin_size_xy_m, .10], "wall_thickness_m": .01}]})
        await asyncio.to_thread(simulation.confirm)
        # This truth snapshot is an independent scorer only; never passed to Runner or algorithms.
        before_oracle = simulation.snapshot()
        overrides = {} if release_height_m is None else {"place_release_height_m": release_height_m}
        if attachment_voxel_size_m is not None:
            overrides["attachment"] = {"voxel_size_m": attachment_voxel_size_m}
        settings = PlanningSettings(plan_timeout_s=300., place_timeout_s=300., plan_max_attempts=planning_attempts, **overrides)
        if retry_release_height_m is not None:
            # Validate the explicit test override before any robot task starts.
            retry_settings = PlanningSettings.model_validate(
                {**settings.model_dump(), "place_release_height_m": retry_release_height_m})
        algorithms = RecordedAlgorithms(output)
        # This harness owns a fresh physical session; never inherit the last
        # completed CLI session's GPU attachment. Keep it intact between stages.
        cleared = await asyncio.to_thread(algorithms.clear_carrying)
        if cleared.get("ok") is not True or cleared.get("attached") is not False:
            raise RuntimeError("fresh_session_gpu_cleanup_not_confirmed")
        runner = Runner(simulation, algorithms, lambda record: Decisions(model, record), output / "runs",
                        planning_settings=settings, visual_assistance=VisualAssistance.from_environment())
        instructions = {
            "grasp": ["抓取红色方块"],
            "grasp_place": ["抓取红色方块到蓝色盒子里"],
            "grasp_then_place": ["抓取红色方块", "把已抓取的物体放到蓝色盒子里"],
        }[task]
        if instruction_override is not None:
            instructions = [instruction_override]
        completed_runs = []
        retried_release = False
        for instruction in instructions:
            await runner.start(instruction)
            try:
                await asyncio.wait_for(asyncio.shield(runner.worker), timeout_s)
            except TimeoutError:
                # Match product Reset order, invalidating physical actions before coroutine cancellation.
                await asyncio.to_thread(simulation.reset)
                await runner.cancel()
            completed_runs.append({"run": public_summary(runner.state), "events": list(runner.events),
                                   "holding": public_summary(runner.holding_state)})
            (output / f"stage-{len(completed_runs)}.json").write_text(
                json.dumps(completed_runs[-1], ensure_ascii=False, indent=2), encoding="utf-8")
            if runner.state["status"] != "succeeded":
                if (not retried_release and retry_release_height_m is not None
                        and runner.state.get("error_code") == "release_clearance_conflict"
                        and runner.holding_state and runner.holding_state.get("attached")):
                    # This explicit harness option mirrors applying a new advanced
                    # release height. Product defaults and the held model are unchanged.
                    runner.planning_settings = retry_settings
                    instructions.append("把已抓取的物体放到蓝色盒子里")
                    retried_release = True
                    continue
                break
        payload = {"input_boundary": "controller uses only registered RGBD and robot feedback",
                   "fixture_side": side, "task": task,
                   "fixture_bin_size_xy_m": list(bin_size_xy_m),
                   "fixture_bin_position_xy_m": list(bin_position_xy_m or (.65, sign * .15)),
                   "fixture_source_position_xy_m": list(source_position_xy_m or (.43, sign * .15)),
                   "instruction_override": instruction_override,
                   "explicit_initial_release_height_m": release_height_m,
                   "explicit_attachment_voxel_size_m": attachment_voxel_size_m,
                   "explicit_retry_release_height_m": retry_release_height_m,
                   "release_height_retry_used": retried_release,
                   "run": public_summary(runner.state), "events": runner.events,
                   "stages": completed_runs}
        (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "independent-oracle.json").write_text(json.dumps(
            {"before": before_oracle, "gripper_commands": simulation.gripper_oracle,
             "after": simulation.snapshot()}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": runner.state["status"], "error": runner.state["error"],
                          "run_id": runner.run_id, "output": str(output)}, ensure_ascii=False))
        return 0 if runner.state["status"] == "succeeded" else 1
    finally:
        if runner and runner.active:
            await asyncio.to_thread(simulation.reset)
            await runner.cancel()
        await asyncio.to_thread(simulation.close)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=900.)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--task", choices=("grasp", "grasp_place", "grasp_then_place"), default="grasp_place")
    parser.add_argument("--retry-release-height-m", type=float, default=None,
                        help="Explicitly retry only a release_clearance_conflict with this advanced height")
    parser.add_argument("--release-height-m", type=float, default=None,
                        help="Explicit initial advanced release height; omission preserves the product default")
    parser.add_argument("--attachment-voxel-size-m", type=float, default=None,
                        help="Explicit initial attachment resolution; omission preserves the product default")
    parser.add_argument("--bin-size-xy-m", type=float, nargs=2, default=(.18, .16),
                        help="Explicit physical fixture dimensions; never passed to perception or planning")
    parser.add_argument("--bin-position-xy-m", type=float, nargs=2, default=None,
                        help="Explicit world XY of physical fixture; never passed to perception or planning")
    parser.add_argument("--source-position-xy-m", type=float, nargs=2, default=None,
                        help="Explicit world XY of source fixture; controller still uses RGBD only")
    parser.add_argument("--instruction", default=None, help="Explicit natural-language test instruction to JEV")
    parser.add_argument("--planning-attempts", type=int, default=1, help="Explicit planner attempt budget for this test")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.output, args.timeout_s, args.side, args.task,
                                     args.retry_release_height_m, args.bin_size_xy_m, args.release_height_m,
                                     args.attachment_voxel_size_m, args.bin_position_xy_m, args.source_position_xy_m, args.instruction, args.planning_attempts)))
