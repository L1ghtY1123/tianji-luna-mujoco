"""Compare captured and current JEV skill questions without running any skills."""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from httpx import HTTPError
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_mujoco.decisions import Decisions
from jev_mujoco.skills import Registry

POLICIES = ("captured_questions", "all_skills")


def load_cases(paths):
    """Accept run JSONL or an /api/state JSON snapshot containing events."""
    cases = []
    for path in paths:
        path = Path(path).resolve()
        contents = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".jsonl":
            events = [json.loads(line) for line in contents.splitlines() if line.strip()]
        else:
            snapshot = json.loads(contents)
            events = snapshot["events"]
        if not isinstance(events, list):
            raise TypeError(f"events must be a list: {path}")
        for index, event in enumerate(events):
            if event.get("kind") != "jev" or event.get("title") != "选择技能":
                continue
            data = event["data"]
            state, questions = data.get("state"), data.get("questions")
            if (not isinstance(state, dict) or not isinstance(state.get("facts"), dict)
                    or not isinstance(questions, dict) or "skill" not in questions):
                raise ValueError(f"missing captured skill state/questions: {path}, event {index + 1}")
            cases.append({"case_id": len(cases) + 1, "source": str(path),
                          "event_id": event.get("id", index + 1),
                          "state": state, "questions": questions,
                          "captured_output": data.get("output")})
    if not cases:
        raise ValueError("no captured JEV skill decisions found")
    return cases


def assess_selection(registry, state, selection):
    """Offline assessment after the answer: any eligible skill is accepted."""
    eligible = registry.eligible(state["facts"])
    valid = selection in eligible
    incorrect_stop = selection == "stop" and bool(eligible)
    correct_stop = selection == "stop" and not eligible
    denied = selection is not None and selection != "stop" and not valid
    return {"eligible_skills": list(eligible), "valid_selection": valid,
            "incorrect_stop": incorrect_stop, "correct_stop": correct_stop,
            "denied_choice": denied}


def aggregate(records):
    result = {}
    for policy in POLICIES:
        selected = [row for row in records if row["policy"] == policy]
        result[policy] = {
            "request_count": len(selected),
            "valid_selection_count": sum(row["assessment"]["valid_selection"] for row in selected),
            "correct_stop_count": sum(row["assessment"]["correct_stop"] for row in selected),
            "incorrect_stop_count": sum(row["assessment"]["incorrect_stop"] for row in selected),
            "denied_choice_counts": dict(Counter(row["selection"] for row in selected
                                               if row["assessment"]["denied_choice"])),
            "error_count": sum(row["error"] is not None for row in selected),
            "errors": dict(Counter(row["error"] for row in selected if row["error"])),
            "api_errors": dict(Counter(row["error"] for row in selected if row["api_error"])),
            "duration_s": sum(row["record"].get("duration_s", 0.) for row in selected),
            "input_tokens": sum(row["record"].get("usage", {}).get("input_tokens", 0) or 0
                                for row in selected),
            "output_tokens": sum(row["record"].get("usage", {}).get("output_tokens", 0) or 0
                                 for row in selected),
        }
    return result


def save_report(report, output):
    report["aggregates"] = aggregate(report["records"])
    report["updated_at"] = datetime.now(UTC).isoformat()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)


async def replay(cases, model, registry, repeats, output, *, minimum_confidence=.6):
    report = {"model": os.getenv("JEV_MODEL", "jev-1.13.0"),
              "started_at": datetime.now(UTC).isoformat(),
              "repeats": repeats, "case_count": len(cases), "status": "running",
              "assessment_rule": "Accept any skill whose current contract matches captured facts. "
                                 "A stop is incorrect when an eligible skill exists. "
                                 "Eligibility is assessed only after model selection.",
              "records": []}
    save_report(report, output)
    try:
        for repeat in range(1, repeats + 1):
            for case in cases:
                for policy in POLICIES:
                    captured = []
                    decisions = Decisions(model, lambda kind, title, data, records=captured: records.append(data),
                                          minimum_confidence=minimum_confidence)
                    selection, error, api_error = None, None, False
                    try:
                        if policy == "captured_questions":
                            answer = await decisions.ask("选择技能", deepcopy(case["state"]),
                                                         deepcopy(case["questions"]), require_confidence=False)
                            selection = answer["skill"]
                        else:
                            selection = await decisions.choose_skill(deepcopy(case["state"]), registry)
                    except Exception as exception:  # noqa: BLE001 - preserve every failed API case in the report.
                        # Exception bodies can contain server responses or credentials.
                        error = type(exception).__name__
                        api_error = isinstance(exception, (ModelAPIError, HTTPError, TimeoutError))
                    finally:
                        record = captured[-1] if captured else {}
                        if selection is None:
                            selection = record.get("output", {}).get("skill")
                            error = error or "Interrupted"
                        assessment = assess_selection(registry, case["state"], selection)
                        report["records"].append({
                            "case_id": case["case_id"], "source": case["source"],
                            "event_id": case["event_id"], "repeat": repeat, "policy": policy,
                            "captured_output": case["captured_output"], "selection": selection,
                            "error": error, "api_error": api_error,
                            "assessment": assessment, "record": record,
                        })
                        save_report(report, output)
                        print(json.dumps({"case": case["case_id"], "repeat": repeat, "policy": policy,
                                          "selection": selection, "valid": assessment["valid_selection"],
                                          "incorrect_stop": assessment["incorrect_stop"], "error": error,
                                          "duration_s": round(record.get("duration_s", 0.), 3)},
                                         ensure_ascii=False), flush=True)
        report["status"] = "completed"
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted"
        save_report(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Captured run .jsonl or /api/state snapshot .json")
    parser.add_argument("--repeats", type=int, default=2, help="Replays per policy and case (default: 2)")
    parser.add_argument("--output", type=Path, default=ROOT / "data/skill-decision-replay.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = args.output.resolve()
    if output in {path.resolve() for path in args.paths}:
        parser.error("the report must not overwrite an input capture")
    cases = load_cases(args.paths)
    load_dotenv(ROOT / ".env")
    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        parser.error("TYPESAFE_API_KEY is missing; configure the project .env")
    model = TypeSafeModel(os.getenv("JEV_MODEL", "jev-1.13.0"), provider=TypeSafeProvider(
        api_key=key, base_url=os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai")))
    report = asyncio.run(replay(cases, model, Registry(), args.repeats, output,
                               minimum_confidence=float(os.getenv("JEV_MINIMUM_CONFIDENCE", ".6"))))
    print(json.dumps({"report": str(output), "aggregates": report["aggregates"]},
                     ensure_ascii=False), flush=True)
    if any(summary["error_count"] for summary in report["aggregates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
