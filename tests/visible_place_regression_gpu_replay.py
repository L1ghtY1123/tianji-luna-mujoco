"""Replay measured B7 planning/opening requests; never send physical commands."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np


def recorded_dependencies(folder):
    """Fail before service mutation if B7 files are from different captures."""
    names = ("020-perception-build.json", "021-tools-plan_place.json", "026-carrying-validate.json")
    values = [json.loads((folder / name).read_text(encoding="utf-8")) for name in names]
    mapping, planning, opening = values
    assert tuple(value["path"] for value in values) == ("/perception/build", "/tools/plan_place", "/carrying/validate")
    assert mapping["body"]["exclusion_state"] == "held" and opening["body"]["phase"] == "open"
    first, second = planning["body"]["carrying_binding"], opening["body"]["carrying_binding"]
    for key in ("instance_id", "observation_id", "generation", "scene_token", "arm", "owner_id", "points_tool_m"):
        assert first[key] == second[key], "mismatched recorded binding: " + key
    assert first["instance_id"] == mapping["body"]["confirmed_instance_id"]
    assert first["observation_id"] == mapping["body"]["observation_set_id"]
    assert first["generation"] == mapping["body"]["selection_generation"]
    assert first["scene_token"] == planning["body"]["scene_token"] == opening["body"]["scene_token"]
    evidence = {"files": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names},
        "scene_token": first["scene_token"], "source_id": first["instance_id"],
        "observation_id": first["observation_id"], "generation": first["generation"],
        "arm": first["arm"], "measured_binding_point_count": len(first["points_tool_m"]),
        "opening_configuration": "unchanged recorded physical B7 release configuration",
        "new_plan_scope": "native attached approach/place, no executed physical commands",
        "opening_reconstructed_from_new_plan": False,
        "detached_retreat_replayed": False, "detached_retreat_requires_actual_release_RGBD": True}
    return values, evidence


def check_two_phase_result(plan, scene_token, arm):
    assert plan["ok"] is True and plan["scene_token"] == scene_token
    assert set(plan["phases"]) == {"approach", "place"}
    assert plan["retreat_requires_replan"] is True
    assert plan["collision_evidence"]["optimizer_attachment"] is True
    expected = {f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)}
    for phase in plan["phases"].values():
        names = phase["joint_names"]
        assert len(names) == len(set(names)) == 7 and set(names) == expected
        assert phase["scene_token"] == scene_token
        arrays = [np.asarray(phase[key], dtype=float) for key in
            ("positions_rad", "velocities_rad_s", "accelerations_rad_s2")]
        assert arrays[0].ndim == 2 and arrays[0].shape[1] == 7 and len(arrays[0]) >= 2
        assert all(value.shape == arrays[0].shape and np.isfinite(value).all() for value in arrays)
        assert np.isfinite(phase["interpolation_dt_s"]) and phase["interpolation_dt_s"] > 0


def main(folder, output, service_url):
    report = {"physical_motion": False, "oracle_used": False, "requests": [], "completed": False}
    (mapping, planning, opening), report["input_dependencies"] = recorded_dependencies(folder)

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def call(path, body, name):
        encoded = json.dumps(body, allow_nan=False).encode()
        started = time.perf_counter()
        request = Request(service_url.rstrip("/") + path, data=encoded,
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=600) as response:
                payload = json.load(response)
        except HTTPError as error:
            payload = json.loads(error.read())
            report["requests"].append({"name": name, "path": path, "http_status": error.code,
                                       "response": payload, "wall_s": time.perf_counter() - started})
            save()
            raise
        report["requests"].append({"name": name, "path": path, "body_sha256": hashlib.sha256(encoded).hexdigest(),
                                   "response": payload, "wall_s": time.perf_counter() - started})
        save()
        print(json.dumps({"step": name, "wall_s": report["requests"][-1]["wall_s"],
                          "ok": payload.get("ok"), "collision_free": payload.get("collision_free"),
                          "phase_names": list(payload.get("phases", {}))}), flush=True)
        return payload

    try:
        assert call("/carrying/clear", {}, "start_cleanup").get("attached") is False
        scene = call(mapping["path"], mapping["body"], "recorded_map")
        assert scene["scene_token"] == planning["body"]["scene_token"] == opening["body"]["scene_token"]
        plan = call(planning["path"], planning["body"], "two_phase_attached_place")
        check_two_phase_result(plan, scene["scene_token"], planning["body"]["arm"])
        validation = call(opening["path"], opening["body"], "actual_opening")
        assert validation["collision_free"] is True and validation["motion_collision_checks_passed"] is True
        assert validation["attached"] is True and validation["gripper_internal_self_checked"] is True
        assert validation["gripper_driver_samples_checked"] == 26
        assert validation["gripper_sweep_self_checked"] is False
        report["completed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    finally:
        report["cleanup_confirmed"] = call("/carrying/clear", {}, "final_cleanup").get("attached") is False
        save()
    assert report["cleanup_confirmed"]
    print(json.dumps({"completed": report["completed"], "cleanup_confirmed": report["cleanup_confirmed"],
                      "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8310")
    options = parser.parse_args()
    main(options.input_folder, options.output, options.service_url)
