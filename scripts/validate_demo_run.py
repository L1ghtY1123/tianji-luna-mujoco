"""Run the isolated simulator API and save independent acceptance evidence."""
import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://192.168.3.4:8350")
    parser.add_argument("--count", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--requested", type=int, choices=range(1, 5))
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requested is not None and args.requested > args.count:
        parser.error("requested count must not exceed scene count")
    quantity = f'{args.requested}个' if args.requested is not None else '所有'
    instruction = (f'对于左侧盒子中的{quantity}粉色待拆包物体，逐个拿出放到桌面目标区域，'
                   '用VLA拆包取出黑色物体，再把黑色物体放到右侧盒子里。一个物体完成后再处理下一个。')
    with httpx.Client(base_url=args.url, timeout=60,
                      limits=httpx.Limits(max_keepalive_connections=0)) as client:
        response = client.post("/api/demo/preset", json={"object_count": args.count})
        response.raise_for_status()
        for endpoint in ("/api/settle", "/api/confirm"):
            response = client.post(endpoint)
            response.raise_for_status()
        # Preset queues cleanup of the previous planner world. Starting a task
        # before that cleanup finishes correctly returns algorithm_request_pending.
        cleanup_deadline = time.monotonic() + 60
        while True:
            response = client.get("/api/state")
            response.raise_for_status()
            readiness = response.json()["run"]
            if readiness.get("algorithm_cleanup_error"):
                raise RuntimeError(readiness["algorithm_cleanup_error"])
            if not readiness.get("algorithm_pending"):
                break
            if time.monotonic() >= cleanup_deadline:
                raise TimeoutError("Planner reset did not finish within 60 seconds")
            time.sleep(.25)
        response = client.post("/api/start", json={"instruction": instruction,
            "experiment": {"name": f"demo-{args.count}-requested-{args.requested or 'all'}",
                           "record_video": args.record_video, "save_mode": "auto"}})
        if response.is_error:
            print(json.dumps({"start_status": response.status_code, "detail": response.text}), flush=True)
        response.raise_for_status()
        started = time.monotonic()
        previous = None
        while True:
            response = client.get("/api/run-progress")
            response.raise_for_status()
            state = response.json()
            run = state["run"]
            progress = (run.get("status"), (run.get("current_step") or {}).get("skill"), run.get("target_id"))
            if progress != previous:
                print(json.dumps({"elapsed_s": round(time.monotonic() - started, 1),
                                  "status": progress[0], "skill": progress[1],
                                  "target": progress[2]}, ensure_ascii=False), flush=True)
                previous = progress
            if run["status"] != "running" or time.monotonic() - started > args.timeout:
                response = client.get('/api/state')
                response.raise_for_status()
                state = response.json()
                run = state['run']
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"status": run["status"], "error": run.get("error"),
                                  "metrics": run.get("metrics"), "evidence": str(args.output)},
                                 ensure_ascii=False), flush=True)
                return 0 if run["status"] == "succeeded" else 1
            time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
