"""Check HTTP reset cancels an in-flight JEV run and restores the confirmed scene."""
import json
import time
from pathlib import Path

import httpx
import numpy as np

from jev_mujoco.model_builder import DEFAULT_SCENE

with httpx.Client(base_url="http://192.168.3.4:8340", timeout=20) as client:
    state = client.get("/api/state").json()
    if state["run"]["status"] == "running":
        raise SystemExit("Existing run preserved")
    client.post("/api/reset").raise_for_status()
    client.post("/api/scene", json=DEFAULT_SCENE).raise_for_status()
    client.post("/api/settle").raise_for_status()
    baseline = client.post("/api/confirm").json()
    started = client.post("/api/start", json={"instruction": "抓取左侧黑色方块"})
    started.raise_for_status()
    reset = client.post("/api/reset")
    reset.raise_for_status()
    restored = reset.json()
    time.sleep(2)
    state = client.get("/api/state").json()
    assert state["run"]["status"] == "idle" and state["events"] == []
    assert state["simulation"]["active_commands"] == []
    assert restored["generation"] > baseline["generation"]
    assert state["simulation"]["time"] > restored["time"]
    errors = {}
    for arm in ("A", "B"):
        error = float(np.max(np.abs(np.array(restored["arms"][arm]["joints_rad"])
                                     - baseline["arms"][arm]["joints_rad"])))
        assert error < 1e-10
        errors[arm] = error
    for expected, actual in zip(baseline["objects"], restored["objects"], strict=True):
        np.testing.assert_allclose(expected["T_world_object"], actual["T_world_object"], atol=1e-10)
    report = {"reset_passed": True, "late_jev_result_cannot_restore_run": True,
              "arm_restore_error_rad": errors, "physics_time_after_reset": restored["time"],
              "physics_time_two_seconds_later": state["simulation"]["time"],
              "confirmed": state["simulation"]["confirmed"]}
    path = Path(__file__).resolve().parents[1] / "docs" / "reset-validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
