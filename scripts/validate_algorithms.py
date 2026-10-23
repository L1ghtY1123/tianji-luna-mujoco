"""Run real GPU inference and planning against a separately stepped simulation."""
import json
import os
import time
from pathlib import Path

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.simulation import Simulation


def main():
    output = Path(os.getenv("LUNA_GPU_PROBE_OUTPUT", "data/algorithm-validation.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"started": time.time(), "steps": {}}
    sim = Simulation()
    sim.start()
    algorithms = Algorithms()
    try:
        sim.settle()
        sim.confirm()
        snapshot = sim.snapshot()
        evidence["snapshot"] = snapshot
        target = snapshot["objects"][0]["id"]
        for name, operation in [
            ("health", lambda: algorithms.health()),
            ("scene", lambda: algorithms.build_scene(snapshot, target)),
            ("grasps", lambda: algorithms.generate_grasps(snapshot, target)),
            ("evaluations", lambda: algorithms.evaluate_arms(snapshot, target, evidence["steps"]["grasps"], evidence["steps"]["scene"]))]:
            start = time.monotonic()
            value = operation()
            evidence["steps"][name] = value
            evidence.setdefault("durations_s", {})[name] = time.monotonic() - start
            output.write_text(json.dumps(evidence, indent=2))
            print(name, "complete", evidence["durations_s"][name], flush=True)
        for arm in ("A", "B"):
            evaluation = evidence["steps"]["evaluations"][arm]
            print(arm, "reachable", len(evaluation["valid_candidates"]), flush=True)
            if evaluation["valid_candidates"]:
                start = time.monotonic()
                try:
                    plan = algorithms.plan_grasp(snapshot, target, arm, evidence["steps"]["evaluations"], evidence["steps"]["scene"])
                    evidence["steps"]["plan_" + arm] = plan
                except Exception as error:  # noqa: BLE001 - preserve the real service failure as probe evidence
                    evidence["steps"]["plan_" + arm] = {"error": str(error)}
                evidence["durations_s"]["plan_" + arm] = time.monotonic() - start
                output.write_text(json.dumps(evidence, indent=2))
                print("plan", arm, "complete", evidence["durations_s"]["plan_" + arm], flush=True)
    except Exception as error:
        evidence["error"] = str(error)
        output.write_text(json.dumps(evidence, indent=2))
        raise
    finally:
        sim.close()


if __name__ == "__main__":
    main()
