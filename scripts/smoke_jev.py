"""Run a small actual JEV classification probe with the configured server credential."""
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

from jev_mujoco.decisions import DecisionError, Decisions

# These assertions exercise the actual model; fake Agent unit tests only verify the confidence gate.
CASES = [
    ("抓取左侧黑色方块", "grasp", {"kind": "cube", "color": "black", "position": "left"}),
    ("抓取黑色物体", "grasp", {"kind": "any", "color": "black", "position": "any"}),
    ("把黑色方块放到盒子里", "grasp_place", None),
    ("Pick the blue cylinder", "grasp", {"kind": "cylinder", "color": "blue", "position": "any"}),
    ("抓取蓝色圆柱", "grasp", {"kind": "cylinder", "color": "blue", "position": "any"}),
    ("Pick the blue cylinder on the left", "grasp", {"kind": "cylinder", "color": "blue", "position": "left"}),
    ("Pick the blue cylinder on the right", "grasp", {"kind": "cylinder", "color": "blue", "position": "right"}),
    ("Pick the blue cylinder at the front", "grasp", {"kind": "cylinder", "color": "blue", "position": "front"}),
    ("Pick the blue cylinder at the back", "grasp", {"kind": "cylinder", "color": "blue", "position": "back"}),
    ("Pick the blue cylinder in the middle", "grasp",
     {"kind": "cylinder", "color": "blue", "position": "unsupported"}),
    ("Pick the blue cylinder between the cubes", "grasp",
     {"kind": "cylinder", "color": "blue", "position": "unsupported"}),
    ("Pick the blue cylinder, not the one on the left", "grasp",
     {"kind": "cylinder", "color": "blue", "position": "unsupported"}),
]


async def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    model = TypeSafeModel(os.getenv("JEV_MODEL", "jev-1.13.0"), provider=TypeSafeProvider(
        api_key=os.environ["TYPESAFE_API_KEY"],
        base_url=os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai")))
    records = []
    decisions = Decisions(model, lambda kind, title, data: records.append(data),
                          minimum_confidence=float(os.getenv("JEV_MINIMUM_CONFIDENCE", ".6")))
    failures = 0
    try:
        for instruction, expected_task, expected_attrs in CASES:
            task, attrs, error = None, None, None
            expected_error = ("unsupported_object_attributes" if expected_attrs
                              and "unsupported" in expected_attrs.values() else None)
            try:
                task = await decisions.bind_task(instruction)
                if task == "grasp":
                    attrs = await decisions.bind_attributes(instruction)
            except DecisionError as exception:
                error = str(exception)
                if task == "grasp":
                    attrs = records[-1].get("output")
            passed = task == expected_task and attrs == expected_attrs and error == expected_error
            failures += not passed
            print(json.dumps({"instruction": instruction, "task": task, "attributes": attrs,
                              "error": error, "passed": passed,
                              "confidence": records[-1].get("provider_details", {}).get("confidence")},
                             ensure_ascii=False), flush=True)
    finally:
        (root / "data").mkdir(exist_ok=True)
        (root / "data" / "jev-smoke.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JEV semantic checks: {len(CASES) - failures}/{len(CASES)} passed")
    if failures:
        raise SystemExit(1)


asyncio.run(main())
