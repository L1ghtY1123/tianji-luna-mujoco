"""Exercise the deployed application's real JEV/GPU/physics workflow over its public API."""
import argparse
import json
import time
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://192.168.3.4:8340")
parser.add_argument("--instruction", default="抓取左侧黑色方块")
args = parser.parse_args()
with httpx.Client(base_url=args.url, timeout=20) as client:
    state = client.get("/api/state").json()
    if state["run"]["status"] == "running":
        raise SystemExit("Existing run preserved; wait for it to finish")
    if state["simulation"]["confirmed"]:
        client.post("/api/reset").raise_for_status()
    else:
        client.post("/api/settle").raise_for_status()
        client.post("/api/confirm").raise_for_status()
    response = client.post("/api/start", json={"instruction": args.instruction})
    response.raise_for_status()
    started = time.monotonic()
    last_id = 0
    while time.monotonic() - started < 900:
        state = client.get("/api/state").json()
        for event in state["events"]:
            if event["id"] > last_id:
                data = event["data"]
                print(json.dumps({"id": event["id"], "kind": event["kind"], "title": event["title"],
                                  "output": data.get("output"), "error": data.get("error"),
                                  "sim_time": state["simulation"]["time"]}, ensure_ascii=False), flush=True)
        last_id = len(state["events"])
        if state["run"]["status"] != "running":
            break
        time.sleep(2)
    output = Path(__file__).resolve().parents[1] / "data" / ("end-to-end-" + state["run"]["run_id"] + ".json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state["run"], ensure_ascii=False), flush=True)
    if state["run"]["status"] != "succeeded":
        raise SystemExit(1)
