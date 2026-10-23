"""Real MuJoCo checks: no GPU, planner, or network required."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


@pytest.fixture(scope="module")
def simulation():
    from luna_mujoco.simulation import Simulation

    instance = Simulation()
    instance.start()
    yield instance
    instance.close()


def test_gravity_settles_objects_on_a_fixed_table(simulation):
    initial = simulation.snapshot()
    result = simulation.settle(timeout_s=4.0)
    assert result["settled"] is True
    assert result["time"] > initial["time"]
    assert result["table"] == initial["table"]
    table_top = result["table"]["position"][2] + result["table"]["size"][2] / 2
    for item in result["objects"]:
        assert item["position"][2] == pytest.approx(table_top + item["size"][2] / 2, abs=0.003)
    assert initial["objects"][0]["position"][2] > result["objects"][0]["position"][2] + 0.025


def test_reset_restores_confirmed_scene_and_rejects_old_generation(simulation):
    from luna_mujoco.simulation import StaleCommandError

    baseline = simulation.confirm()
    scene = {"objects": [dict(baseline["objects"][0], position=[0.45, 0.15, 0.85], yaw_deg=35)]}
    simulation.configure(scene)
    simulation.settle(timeout_s=3)
    restored = simulation.reset()
    assert restored["confirmed"] is True
    assert restored["generation"] > baseline["generation"]
    assert len(restored["objects"]) == len(baseline["objects"])
    for expected, actual in zip(baseline["objects"], restored["objects"], strict=True):
        np.testing.assert_allclose(actual["T_world_object"], expected["T_world_object"], atol=1e-10)
    with pytest.raises(StaleCommandError):
        simulation.set_gripper("A", True, baseline["generation"])


def test_reset_cancels_active_motor_trajectory_and_restores_controls(simulation):
    from luna_mujoco.simulation import StaleCommandError

    baseline = simulation.confirm()
    start = np.array(baseline["arms"]["A"]["joints_rad"])
    finish = start.copy()
    finish[0] += 0.16
    path = np.linspace(start, finish, 101).tolist()
    with ThreadPoolExecutor() as executor:
        future = executor.submit(simulation.execute_trajectory, "A", path, 0.025, baseline["generation"])
        time.sleep(0.20)
        during = simulation.snapshot()
        assert during["time"] > baseline["time"] + 0.03
        assert during["arms"]["A"]["joints_rad"][0] < finish[0] - 0.03
        restored = simulation.reset()
        with pytest.raises(StaleCommandError):
            future.result(timeout=2)
    np.testing.assert_allclose(restored["arms"]["A"]["joints_rad"], start, atol=1e-10)
    time.sleep(0.12)
    np.testing.assert_allclose(simulation.snapshot()["arms"]["A"]["joints_rad"], start, atol=0.01)


def test_motor_trajectory_and_gripper_report_measured_state(simulation):
    state = simulation.reset()
    start = np.array(state["arms"]["B"]["joints_rad"])
    finish = start.copy()
    finish[0] -= 0.07
    result = simulation.execute_trajectory("B", np.linspace(start, finish, 61).tolist(), 0.01, state["generation"])
    assert result["max_final_error_rad"] < 0.03
    np.testing.assert_allclose(simulation.snapshot()["arms"]["B"]["joints_rad"], finish, atol=0.03)
    closed = simulation.set_gripper("B", True, state["generation"])
    assert closed["gripper_closed_fraction"] > 0.8
    opened = simulation.set_gripper("B", False, state["generation"])
    assert opened["gripper_closed_fraction"] < 0.1


def test_render_manifest_and_snapshot_share_geometry_ids(simulation):
    manifest = simulation.manifest()
    snapshot = simulation.snapshot()
    assert manifest["revision"] == snapshot["manifest_revision"]
    assert {item["id"] for item in manifest["geometries"]} == {
        item["id"] for item in snapshot["geometry_poses"]
    }
    assert any(item["type"] == "mesh" and item["mesh_url"].startswith("/assets/") for item in manifest["geometries"])
    assert set(snapshot["arms"]) == {"A", "B"}
    assert np.asarray(snapshot["arms"]["A"]["tcp"]).shape == (4, 4)


def test_dwell_uses_wall_time_and_reset_interrupts_it(simulation):
    from luna_mujoco.simulation import StaleCommandError

    baseline = simulation.reset()
    result = simulation.dwell(.05, baseline["generation"])
    assert .05 <= result["duration_s"] < .1
    assert result["simulation_duration_s"] >= 0
    with ThreadPoolExecutor() as executor:
        pending = executor.submit(simulation.dwell, 10, baseline["generation"])
        time.sleep(.04)
        simulation.reset()
        with pytest.raises(StaleCommandError):
            pending.result(timeout=2)


def test_dwell_does_not_stretch_when_physics_runs_below_real_time(monkeypatch):
    import threading
    from types import SimpleNamespace

    from luna_mujoco.simulation import Simulation

    clock = SimpleNamespace(now=0.)
    data = SimpleNamespace(time=10.)
    test_thread = threading.get_ident()
    monotonic = time.monotonic

    class Condition:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self, timeout):
            clock.now += timeout
            data.time += timeout * .5

    instance = SimpleNamespace(_condition=Condition(), _data=data, _check_generation=lambda _: None)
    # The module fixture has a live physics thread; its clock must keep advancing.
    monkeypatch.setattr(time, 'monotonic',
                        lambda: clock.now if threading.get_ident() == test_thread else monotonic())
    result = Simulation.dwell(instance, 1., 1)
    assert clock.now == pytest.approx(1.)
    assert result['duration_s'] == pytest.approx(1.)
    assert result['simulation_duration_s'] == pytest.approx(.5)


def test_physics_step_refreshes_cartesian_poses_only_when_read(monkeypatch):
    import mujoco

    from luna_mujoco.simulation import Simulation

    instance = Simulation()
    step = mujoco.mj_step
    forward_position = mujoco.mj_fwdPosition
    calls = []

    def step_once(model, data):
        step(model, data)
        if data is instance._data:
            instance._stop.set()

    def record_forward(model, data):
        if data is instance._data:
            calls.append(float(data.time))
        forward_position(model, data)

    monkeypatch.setattr(mujoco, 'mj_step', step_once)
    monkeypatch.setattr(mujoco, 'mj_fwdPosition', record_forward)
    try:
        instance._run()
        assert instance._error is None
        assert calls == []
        with instance._condition:
            instance._data.qpos[instance._arm_q['A'][0]] += .2
            expected = mujoco.MjData(instance._model)
            mujoco.mj_copyData(expected, instance._model, instance._data)
            forward_position(instance._model, expected)
            robot = instance.robot_snapshot()
            snapshot = instance.snapshot()
        assert len(calls) == 2
        body = instance._tcp_ids['A']
        for result in (robot, snapshot):
            np.testing.assert_allclose(np.asarray(result['arms']['A']['tcp'])[:3, 3],
                                       expected.xpos[body], atol=1e-12)
            np.testing.assert_allclose(np.asarray(result['arms']['A']['tcp'])[:3, :3],
                                       expected.xmat[body].reshape(3, 3), atol=1e-12)
    finally:
        instance.close()


def test_invalid_scene_does_not_change_live_scene(simulation):
    previous = simulation.snapshot()
    with pytest.raises(ValueError, match="size"):
        simulation.configure({"objects": [{"id": "invalid", "kind": "box", "size": [-1, 1, 1]}]})
    assert simulation.snapshot()["generation"] == previous["generation"]


def test_confirm_waits_for_falling_objects_before_saving_baseline(simulation):
    scene = {"objects": [{"id": "falling", "kind": "cube", "color": "black",
                           "position": [0.45, 0.15, 0.88], "size": [0.04] * 3}]}
    simulation.configure(scene)
    confirmed = simulation.confirm()
    assert confirmed["objects"][0]["position"][2] == pytest.approx(0.62, abs=0.003)


def test_reset_restores_integration_inputs_and_motor_references(simulation):
    with simulation._condition:
        simulation.confirm()
        expected_controls = simulation._data.ctrl.copy()
        expected_qpos = simulation._data.qpos.copy()
        expected_qvel = simulation._data.qvel.copy()
        expected_targets = {arm: target.copy() for arm, target in simulation._desired_q.items()}
        simulation._data.ctrl[:] = 99
        simulation._data.qfrc_applied[:] = 10
        simulation._data.xfrc_applied[:] = 15
        simulation._desired_q["A"] += 0.1
        simulation.reset()
        np.testing.assert_array_equal(simulation._data.ctrl, expected_controls)
        np.testing.assert_array_equal(simulation._data.qpos, expected_qpos)
        np.testing.assert_array_equal(simulation._data.qvel, expected_qvel)
        assert not np.any(simulation._data.qfrc_applied)
        assert not np.any(simulation._data.xfrc_applied)
        for arm in ("A", "B"):
            np.testing.assert_array_equal(simulation._desired_q[arm], expected_targets[arm])
        assert simulation._commands == {}


def test_objects_keep_physical_contacts_with_articulated_robot(simulation):
    with simulation._condition:
        model = simulation._model
        object_geom = model.geom("object_geom_" + simulation._scene["objects"][0]["id"]).id
        finger_body = model.body("gripL_Link3_L").id
        finger_geoms = np.flatnonzero((model.geom_bodyid == finger_body) & (model.geom_group == 3))
        assert len(finger_geoms) == 1
        finger_geom = finger_geoms[0]
        assert (model.geom_contype[finger_geom] & model.geom_conaffinity[object_geom]
                or model.geom_contype[object_geom] & model.geom_conaffinity[finger_geom])
        table_geom = model.geom("table").id
        assert model.geom_bodyid[table_geom] == 0
        assert model.neq == 10  # Both physical grippers retain their five mimic constraints.


def test_dual_commands_have_same_start_time_and_reset_cancels_both(simulation):
    baseline = simulation.confirm()
    paths = {}
    for arm in ('A', 'B'):
        start = np.array(baseline['arms'][arm]['joints_rad'])
        end = start.copy()
        end[0] += .08
        paths[arm] = np.linspace(start, end, 101).tolist()
    with ThreadPoolExecutor() as executor:
        future = executor.submit(simulation.execute_dual_trajectory, paths, .025, baseline['generation'])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with simulation._condition:
                if set(simulation._commands) == {'A', 'B'}:
                    assert simulation._commands['A'].start_time == simulation._commands['B'].start_time
                    break
            time.sleep(.01)
        else:
            future.result(timeout=1)
            pytest.fail('dual commands were not submitted together')
        simulation.reset()
        from luna_mujoco.simulation import StaleCommandError
        with pytest.raises(StaleCommandError):
            future.result(timeout=3)
        assert not simulation._commands
