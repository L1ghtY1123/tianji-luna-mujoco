"""Adversarial cleanup ordering and same-generation holding lifecycle review."""

import asyncio
import copy
import threading

import pytest
from fastapi.testclient import TestClient
from test_api import Algorithms, Simulation

from luna_mujoco.app import create_app
from luna_mujoco.runner import Runner


@pytest.mark.parametrize("new_generation", [False, True])
async def test_latest_successful_reset_is_not_poisoned_by_superseded_cleanup_failure(tmp_path, new_generation):
    class SupersededFailure(Algorithms):
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def clear_carrying(self):
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
                assert self.release.wait(5)
                raise RuntimeError("superseded_reset_transport_failure")
            return {"ok": True, "attached": False}

    simulation = Simulation()
    algorithms = SupersededFailure()
    runner = Runner(simulation, algorithms, lambda record: None, tmp_path)
    runner.clear_algorithm_state()
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        if new_generation:
            simulation.reset()
        runner.clear_algorithm_state()
        assert runner.algorithm_pending
    finally:
        algorithms.release.set()
        await asyncio.wait_for(runner.algorithm_cleanup_worker, 3)
    assert algorithms.calls == 2
    assert not runner.algorithm_pending
    assert runner.algorithm_cleanup_error is None


async def test_reset_cleanup_waits_for_old_request_then_clears_its_late_attachment(tmp_path):
    class LateAttachment(Algorithms):
        attached = False
        clears = 0

        def clear_carrying(self):
            assert self.attached
            self.attached = False
            self.clears += 1
            return {"ok": True, "attached": False}

    simulation = Simulation()
    simulation.confirm()
    algorithms = LateAttachment()
    runner = Runner(simulation, algorithms, lambda record: None, tmp_path)
    release = asyncio.Event()

    async def old_request():
        await release.wait()
        algorithms.attached = True

    runner.planning_worker = asyncio.create_task(old_request())
    runner.clear_algorithm_state()
    try:
        with pytest.raises(ValueError, match="algorithm_request_pending"):
            await runner.start("抓取红色方块")
        assert algorithms.clears == 0
    finally:
        release.set()
        await asyncio.wait_for(runner.algorithm_cleanup_worker, 3)
    assert algorithms.clears == 1 and not algorithms.attached
    assert not runner.algorithm_pending
    assert runner.algorithm_cleanup_error is None


async def test_only_latest_generation_cleans_when_two_resets_precede_pending_request(tmp_path):
    class CountingAlgorithms(Algorithms):
        calls = 0

        def clear_carrying(self):
            self.calls += 1
            return {"ok": True, "attached": False}

    simulation = Simulation()
    algorithms = CountingAlgorithms()
    runner = Runner(simulation, algorithms, lambda record: None, tmp_path)
    release = asyncio.Event()
    runner.planning_worker = asyncio.create_task(release.wait())
    runner.clear_algorithm_state()
    simulation.reset()
    runner.clear_algorithm_state()
    release.set()
    await asyncio.wait_for(runner.algorithm_cleanup_worker, 3)
    assert algorithms.calls == 1
    assert not runner.algorithm_pending


@pytest.mark.parametrize("endpoint", ["/api/confirm", "/api/settle", "/api/scene"])
@pytest.mark.parametrize("holding_state", ["provisional", "held", "release_pending"])
def test_same_generation_scene_action_does_not_forget_a_nonempty_hand(tmp_path, endpoint, holding_state):
    class SettledSimulation(Simulation):
        def settle(self):
            return {**self.snapshot(), "settled": True}

    simulation = SettledSimulation()
    app = create_app(simulation, Algorithms(), lambda record: None, tmp_path)
    holding = {"state": holding_state, "generation": 1, "arm": "A", "instance_id": "visible-source",
               "verified": holding_state == "held", "attached": holding_state == "held"}
    with TestClient(app) as client:
        app.state.runner.holding_state = copy.deepcopy(holding)
        response = client.post(endpoint, json={"objects": []})
        assert response.status_code == 409
        assert response.json()["detail"] == "holding_state_reset_required"
        assert simulation.generation == 1
        assert app.state.runner.holding_state == holding


async def test_failed_cleanup_blocks_start_until_a_later_confirmed_cleanup(tmp_path):
    class UnconfirmedCleanup(Algorithms):
        cleared = False

        def clear_carrying(self):
            return {"ok": True, "attached": not self.cleared}

    simulation = Simulation()
    simulation.confirm()
    algorithms = UnconfirmedCleanup()
    runner = Runner(simulation, algorithms, lambda record: None, tmp_path)
    runner.clear_algorithm_state()
    await asyncio.wait_for(runner.algorithm_cleanup_worker, 3)
    assert not runner.algorithm_pending
    with pytest.raises(ValueError, match="algorithm_cleanup_failed_reset_to_retry"):
        await runner.start("抓取红色方块")
    algorithms.cleared = True
    runner.clear_algorithm_state()
    await asyncio.wait_for(runner.algorithm_cleanup_worker, 3)
    assert runner.algorithm_cleanup_error is None


def test_public_release_pending_keeps_unknown_detach_state_and_bounds_nested_evidence(tmp_path):
    app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path)
    with TestClient(app) as client:
        app.state.runner.holding_state = {
            "state": "release_pending", "arm": "A", "verified": False, "attached": True,
            "attachment_detach_pending": True,
            "attachment_evidence": {"points_tool_m": [[1, 2, 3]] * 100,
                                    "timings": list(range(20)), "prepare_wall_ms": 12.5},
            "observation": {"views": {"rgb": [1] * 100}},
        }
        holding = client.get("/api/state").json()["run"]["holding"]
        assert holding["state"] == "release_pending"
        assert holding["attached"] is True and holding["attachment_detach_pending"] is True
        assert not holding["verified"]
        assert "observation" not in holding
        assert "points_tool_m" not in holding["attachment_evidence"]
        assert holding["attachment_evidence"]["timings"] == {"count": 20, "first": [0, 1, 2]}
