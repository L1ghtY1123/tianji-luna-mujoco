"""One optional live VLM call against a trusted, finished RGBD acceptance recording.

No simulation, GPU, planner, or robot service is accessed. Provider configuration
is loaded exactly as the acceptance harness loads it; credentials are never logged.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
from dotenv import load_dotenv

from luna_mujoco.visual_assistance import VisualAssistance, VisualCallBudget
from luna_mujoco.visual_scene import VisualScene, capture_fingerprint, grounded_scene, scene_legend


class TimedStream(httpx.SyncByteStream):
    def __init__(self, source, report, started, body):
        self.source, self.report, self.started, self.body = source, report, started, body

    def __iter__(self):
        try:
            for chunk in self.source:
                self.report.setdefault("first_body_s", time.monotonic() - self.started)
                self.report["last_body_s"] = time.monotonic() - self.started
                self.report["received_bytes"] = self.report.get("received_bytes", 0) + len(chunk)
                self.body.extend(chunk[:max(0, 1048576 - len(self.body))])
                yield chunk
        except httpx.HTTPError as error:
            self.report["stream_error"] = type(error).__name__
            raise

    def close(self):
        self.source.close()


class TimedTransport(httpx.BaseTransport):
    def __init__(self):
        self.transport = httpx.HTTPTransport(retries=0)
        self.report, self.body = {}, bytearray()

    def handle_request(self, request):
        started = time.monotonic()
        self.report.update(request_bytes=len(request.content), request_count=self.report.get("request_count", 0) + 1)
        try:
            response = self.transport.handle_request(request)
        except httpx.HTTPError as error:
            self.report.update(header_error=type(error).__name__, header_error_s=time.monotonic() - started)
            raise
        self.report.update(status_code=response.status_code, headers_s=time.monotonic() - started,
                           content_type=response.headers.get("content-type", ""))
        response.stream = TimedStream(response.stream, self.report, started, self.body)
        return response

    def close(self):
        self.transport.close()


def summarize_response(body, legend):
    """Retain timing/schema diagnostics, never raw provider headers or error text."""
    report, completed = {"event_types": []}, None
    try:
        text = body.decode("utf-8")
        if text.lstrip().startswith("{"):
            completed = json.loads(text)
        else:
            for line in text.splitlines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                event = json.loads(line[5:])
                event_type = event.get("type", "unknown")
                if event_type not in report["event_types"]:
                    report["event_types"].append(event_type)
                if "response" in event:
                    report["last_response_status"] = event["response"].get("status")
                if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                    completed = event["response"]
        if completed is not None:
            report["response_status"] = completed.get("status")
            report["provider_error_code"] = (completed.get("error") or {}).get("code")
            report["incomplete_reason"] = (completed.get("incomplete_details") or {}).get("reason")
            texts = [entry["text"] for item in completed.get("output", []) for entry in item.get("content", [])
                     if entry.get("type") == "output_text"]
            if texts:
                try:
                    parsed = VisualScene.model_validate_json("".join(texts))
                    grounded_scene(parsed, legend)
                    report["grounding_validation"] = "passed"
                except ValueError as error:
                    report["grounding_validation"] = type(error).__name__
                    # Our identity failure strings contain no network/configuration data.
                    if str(error).startswith("ungrounded_visual_"):
                        report["grounding_failure"] = str(error)
    except (ValueError, KeyError, TypeError, UnicodeError) as error:
        report["response_decode_error"] = type(error).__name__
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--observation-id")
    parser.add_argument("--project-root", type=Path, default=Path("/opt/jev"))
    parser.add_argument("--call", action="store_true", help="Spend at most one configured provider request")
    args = parser.parse_args()
    load_dotenv(args.project_root / ".env", override=False)
    candidates = []
    for path in sorted(args.recording.glob("perception-*.npz")):
        with np.load(path, allow_pickle=True) as saved:
            observed = saved["observation"].item()
        if args.observation_id is None or observed["observation_id"] == args.observation_id:
            candidates.append(observed)
    if not candidates:
        raise ValueError("saved_perception_missing")
    # First sensor time gives the initial scene, before manipulation obscures IDs.
    observed = min(candidates, key=lambda item: item["time"])
    with np.load(args.recording / f"rgbd-{observed['observation_id']}.npz", allow_pickle=True) as saved:
        raw = saved["capture"].item()
    assert raw["observation_id"] == observed["observation_id"] and raw["generation"] == observed["generation"]
    for camera in ("3", "2", "1"):
        assert np.array_equal(raw["views"][camera]["rgb"], observed["views"][camera]["rgb"])
    legend = scene_legend(observed, observed["objects"])
    report = {"recording": str(args.recording), "observation_id": observed["observation_id"],
              "capture_fingerprint": capture_fingerprint(observed), "legend": legend,
              "same_saved_rgb_verified": True, "gpu_used": False, "oracle_used": False}
    if args.call:
        visual = VisualAssistance.from_environment()
        endpoint = urlsplit(visual.base_url)
        report.update(endpoint_host=endpoint.hostname, endpoint_path=endpoint.path, model=visual.model,
                      configured=visual.enabled, timeout_s=visual.timeout_s)
        transport = TimedTransport()
        budget = VisualCallBudget(limit=1)
        with httpx.Client(transport=transport, timeout=visual.timeout_s) as client:
            visual.client = client
            report["result"] = visual.inspect_scene(observed, observed["objects"],
                "Identify the visible supplied objects, their colors and kinds, and whether they are on the table "
                "or visibly inside another supplied object. State ambiguity and occlusion; use only supplied IDs.",
                call_budget=budget)
        report.update(calls_used=budget.used, transport=transport.report,
                      response_diagnostics=summarize_response(transport.body, legend))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
