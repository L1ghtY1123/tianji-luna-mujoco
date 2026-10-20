"""Live JEV language acceptance over synthetic measured-shaped evidence; no robot calls."""
import argparse
import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from httpx import HTTPError
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from test_rgbd_workflow import SensorRobot

from jev_mujoco.decisions import DecisionError, Decisions
from jev_mujoco.runner import Runner
from jev_mujoco.task_progress import TaskProgress


def instance(identifier, kind, color, position, *, inside_id=None):
    return {"id": identifier, "kind": kind, "color": color, "position": position,
            "visible_centroid_world": position, "source": "rgbd", "confidence": .98,
            "coverage": {"observed_surface_only": True, "camera_ids": ["1", "2", "3"], "point_count": 600},
            "visual_semantics": {"label": color + " " + kind, "visibility": "visible", "ambiguous": False,
                                 "inside_id": inside_id, "on_table": inside_id is None,
                                 "on_marker_id": None, "evidence": "Current registered camera evidence."}}


def objects():
    return [instance("red-1", "cube", "red", [.40, .18, .66]),
            instance("red-2", "cube", "red", [.49, .18, .66]),
            instance("red-3", "cube", "red", [.58, .18, .66]),
            instance("blue-sphere", "sphere", "blue", [.47, -.18, .67], inside_id="red-bin"),
            instance("black-cube", "cube", "black", [.44, .32, .66]),
            instance("green-cube", "cube", "green", [.58, .31, .66]),
            instance("red-bin", "bin", "red", [.47, -.18, .62]),
            instance("blue-bin", "bin", "blue", [.69, -.20, .62]),
            instance("yellow-marker", "marker", "yellow", [.71, .22, .60])]


def state_for(instruction, *, selected=None, completed=()):
    visible = objects()
    by_id = {obj["id"]: obj for obj in visible}
    progress = TaskProgress(instruction, "collection", "A")
    progress.observe(visible, "fixture-current")
    for identifier in completed:
        progress.select(by_id[identifier], "blue-bin")
        progress.complete(identifier, observation_id="verified-" + identifier)
        by_id[identifier]["position"] = [.69, -.20, .67]
        by_id[identifier]["visual_semantics"].update(inside_id="blue-bin", on_table=False)
    if selected:
        progress.select(by_id[selected], "blue-bin")
    return {"instruction": instruction, "task": "grasp_place", "requested_arm": "A", "execution_arm": "A",
            "attributes": {"kind": "any", "color": "any", "position": "any"},
            "destination_attributes": {"kind": "any", "color": "any", "position": "any"},
            "target_id": selected, "destination_id": "blue-bin" if selected else None,
            "facts": {"collection_task": True, "observed": True, "target_selected": selected is not None,
                      "recovery_needed": False, "grasp_held": False, "verified": False, "attached": False},
            "task_progress": progress.snapshot(),
            "observation": {"source": "rgbd", "generation": 1, "observation_id": "fixture-current",
                            "objects": visible, "table": {"surface_height_m": .6}},
            "observation_budget": {"used": 1, "limit": 8}, "visual_call_budget": {"used": 1, "limit": 8}}


def recovery_state(held):
    state = state_for("Use the left arm to move all red cubes into the blue bin; if one cannot be moved, continue with the others.",
                      selected="red-1")
    verdict = "held" if held else "empty"
    position = {"success": held, "status": verdict, "source": "gripper_position",
                "position_rad": .48 if held else .8, "empty_closed_rad": .8,
                "tolerance_rad": .016, "stable_duration_s": .25}
    state["facts"].update(recovery_needed=True, grasp_held=held, verified=held, attached=held,
                          released=False, lifted=True, grasp_completed=held)
    state["verification"] = {"success": held, "status": verdict, "source": "gripper_position", "position": position,
                             "attachment": {"ok": held, "attached": held, "scene_token": "fixture-scene"}}
    stage = "plan_place" if held else "verify_grasp"
    state["failure"] = {"target_id": "red-1", "stage": stage,
                        "code": "place_no_valid_candidate" if held else "grasp_verification_failed",
                        "attempt_count": 3, "retry_limit": 3, "evidence": copy.deepcopy(position)}
    state["task_progress"].update(failure=copy.deepcopy(state["failure"]), attempts={"red-1": 3})
    state["observation_budget"]["used"] = 8
    state["visual_call_budget"]["used"] = 8
    state["simulation"] = {"generation": 1, "time": 10., "arms": {
        "A": {"joints_rad": [0.] * 7, "gripper_position_rad": .48 if held else .8,
              "gripper_closed_command": True, "gripper_limits_rad": [0., .8]},
        "B": {"joints_rad": [0.] * 7, "gripper_position_rad": 0.,
              "gripper_closed_command": False, "gripper_limits_rad": [0., .8]}}}
    if held:
        current = next(obj for obj in state["observation"]["objects"] if obj["id"] == "red-1")
        current["position"] = [.53, .08, .85]
        current["visual_semantics"].update(on_table=False, evidence="Visibly held between active gripper fingers.")
    return state


BIND_CASES = [
    ("subset_explicit_left", "Use the left arm to put only the red cube into the blue bin; leave the green cube untouched.",
     {"task": "grasp_place", "arm": "A", "kind": "cube", "color": "red"}),
    ("count_combination", "First put two red cubes into the blue bin, then one blue sphere; do not move black objects.",
     {"task": "grasp_place", "scope": "collection", "kind": "any", "color": "any"}),
    ("source_container_right", "用右臂把红色盒子里的所有物体放到黄色标记区，桌面其他物体不要动。",
     {"task": "grasp_place", "arm": "B", "scope": "collection", "color": "any", "destination_kind": "marker"}),
    ("between_relation", "Move the cylinder between the red cubes into the blue bin.",
     {"task": "grasp_place", "kind": "cylinder", "position": "any"}),
    ("negative_position", "Move every cube except the leftmost black one into the blue bin.",
     {"task": "grasp_place", "scope": "collection", "color": "any", "position": "any"}),
    ("grouped_destinations", "First move red cubes into the blue bin, then move blue spheres onto the yellow marker.",
     {"task": "grasp_place", "scope": "collection", "kind": "any", "color": "any", "destination_kind": "any"}),
    ("semantic_category", "Move the cup beside the red bin onto the yellow marker.",
     {"task": "grasp_place", "kind": "any", "position": "any", "destination_kind": "marker"}),
]


class FixturePerception:
    """Apply actual source query hints to fixed measured-shaped candidates."""

    def __init__(self):
        self.source_hint = None
        self.visible_ids = []

    def observe(self, capture, *, source_hint, **kwargs):
        self.source_hint = copy.deepcopy(source_hint)
        visible = objects()
        cup = instance("white-cup", "cylinder", "white", [.36, -.18, .67])
        cup["visual_semantics"].update(label="white cup", evidence="Visible cup with handle beside the red bin.")
        visible.append(cup)
        visible = [obj for obj in visible if obj["kind"] in {"bin", "marker"}
                   or all(source_hint.get(key, "any") in ("any", obj[key]) for key in ("kind", "color"))]
        for obj in visible:
            obj["points_world_m"] = np.array([np.asarray(obj["position"]) + [x, y, .01]
                                             for x in (-.01, 0., .01) for y in (-.01, 0., .01)])
        self.visible_ids = [obj["id"] for obj in visible]
        return {**copy.deepcopy(capture), "objects": visible,
                "table": {"position": [.55, 0., .595], "size": [.7, 1., .01], "surface_height_m": .6}}


class SelectionOnlyDecisions(Decisions):
    """Exercise real binding and selection; stop the fixture before planning or motion."""

    async def choose_skill(self, state, registry):
        if state.get("failure") or state["facts"].get("target_selected"):
            return "stop"
        if not state["facts"].get("observed"):
            return "observe"
        if not state["facts"].get("task_assessed"):
            return "assess_task"
        return "identify_target"

    async def assess_task(self, state):
        # All chain cases explicitly have unfinished requested sources in the fixture.
        return "continue"


async def chained_selection(model, record, instruction, expected_targets, *, broad_query):
    perception = FixturePerception()
    with tempfile.TemporaryDirectory(prefix="jev-language-fixture-") as directory:
        runner = Runner(SensorRobot(), perception,
                        lambda emit: SelectionOnlyDecisions(model, record), directory)
        await runner.start(instruction)
        await runner.worker
        assert runner.state["target_id"] in expected_targets, runner.state.get("error")
        assert runner.state["instruction"] == instruction
        if broad_query:
            assert runner.state["attributes"] == dict.fromkeys(("kind", "color", "position"), "any")
            assert "blue-sphere" in perception.visible_ids
        return {"target_id": runner.state["target_id"], "source_query": runner.state["attributes"],
                "visible_ids": perception.visible_ids, "instruction_preserved": True}


async def run_checks(model, output):
    report = {"model": os.getenv("JEV_MODEL", "jev-1.13.0"), "fixture_source": "synthetic measured-shaped evidence",
              "uses_robot_or_gpu": False, "cases": []}
    records = []
    decisions = Decisions(model, lambda kind, title, data: records.append(data),
                          minimum_confidence=float(os.getenv("JEV_MINIMUM_CONFIDENCE", ".6")))

    async def check(name, operation, expected, *, category="decision"):
        before = len(records)
        actual, error = None, None
        try:
            actual = await operation()
            passed = (all(actual.get(key) == value for key, value in expected.items())
                      if isinstance(expected, dict) else actual in expected)
        except (AssertionError, ValueError, TypeError, ModelAPIError, HTTPError) as exception:
            error = str(exception) if isinstance(exception, (AssertionError, DecisionError)) else type(exception).__name__
            passed = False
        traces = []
        for item in records[before:]:
            traces.append({"state": item["state"], "questions": item["questions"], "output": item.get("output"),
                           "confidence": item.get("provider_details", {}).get("confidence"),
                           "usage": item.get("usage"), "duration_s": item.get("duration_s")})
        row = {"name": name, "category": category, "passed": passed, "expected": expected, "actual": actual, "error": error,
               "traces": traces}
        report["cases"].append(row)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: row[key] for key in ("name", "passed", "actual", "error")}, ensure_ascii=False), flush=True)

    for name, instruction, expected in BIND_CASES:
        await check("bind_" + name, lambda instruction=instruction: decisions.bind_request(instruction), expected,
                    category="raw_binding")

    subset = state_for(BIND_CASES[0][1])
    subset["observation"]["objects"] = [obj for obj in subset["observation"]["objects"]
                                       if obj["id"] not in {"red-2", "red-3"}]
    await check("select_requested_subset", lambda: decisions.choose_target(subset, subset["observation"]["objects"]), ["red-1"])

    quantity = state_for(BIND_CASES[1][1], completed=("red-1",))
    await check("respect_count_and_group_order", lambda: decisions.choose_target(quantity, quantity["observation"]["objects"]),
                ["red-2", "red-3"])
    finished = state_for(BIND_CASES[1][1], completed=("red-1", "red-2", "blue-sphere"))
    await check("quota_done_with_extra_red_remaining", lambda: decisions.assess_task(finished), ["complete"])

    source_bin = state_for(BIND_CASES[2][1])
    source_bin["requested_arm"] = source_bin["execution_arm"] = "B"
    source_bin["task_progress"]["requested_arm"] = "B"
    await check("source_bin_excludes_tabletop", lambda: decisions.choose_target(source_bin, source_bin["observation"]["objects"]),
                ["blue-sphere"])

    for selected, expected in (("red-2", "blue-bin"), ("blue-sphere", "yellow-marker")):
        grouped = state_for(BIND_CASES[5][1], selected=selected)
        original = copy.deepcopy(grouped["task_progress"]["current"])
        # The source's current location changes during carrying; its original grouping attributes remain in the ledger.
        for obj in grouped["observation"]["objects"]:
            if obj["id"] == selected:
                obj["position"] = [.54, 0., .88]
                obj["visual_semantics"].update(inside_id=None, on_table=False)
        await check("grouped_destination_" + selected,
                    lambda grouped=grouped: decisions.choose_destination(grouped, grouped["observation"]["objects"]), [expected])
        assert grouped["task_progress"]["current"] == original

    await check("exhausted_empty_collection_defer", lambda: decisions.choose_recovery(recovery_state(False)), ["defer"])
    await check("exhausted_held_collection_stop", lambda: decisions.choose_recovery(recovery_state(True)), ["stop"])
    chain_cases = [("source_bin", BIND_CASES[2][1], {"blue-sphere"}, True),
                   ("mixed_count", BIND_CASES[1][1], {"red-1", "red-2", "red-3"}, True),
                   ("semantic_cup", BIND_CASES[6][1], {"white-cup"}, False)]
    for name, instruction, targets, broad in chain_cases:
        await check("chain_" + name,
                    lambda instruction=instruction, targets=targets, broad=broad: chained_selection(
                        model, lambda kind, title, data: records.append(data), instruction, targets, broad_query=broad),
                    {"instruction_preserved": True}, category="runner_binding_to_selection")
    report["passed"] = all(row["passed"] for row in report["cases"])
    report["pass_count"] = sum(row["passed"] for row in report["cases"])
    report["category_results"] = {category: {"passed": sum(row["passed"] for row in report["cases"] if row["category"] == category),
                                            "total": sum(row["category"] == category for row in report["cases"])}
                                  for category in ("raw_binding", "decision", "runner_binding_to_selection")}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "pass_count": report["pass_count"], "total": len(report["cases"])}), flush=True)
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model = TypeSafeModel(os.getenv("JEV_MODEL", "jev-1.13.0"), provider=TypeSafeProvider(
        api_key=os.environ["TYPESAFE_API_KEY"], base_url=os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai")))
    if not asyncio.run(run_checks(model, args.output)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
