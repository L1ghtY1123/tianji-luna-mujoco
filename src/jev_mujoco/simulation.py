"""Continuous MuJoCo physics, motor commands, and confirmed scene baselines."""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .geometry import front_tcp_pose
from .model_builder import ARM_JOINTS, DEFAULT_SCENE, GRIP_DRIVERS, TCP_BODIES, ModelBuilder, normalize_scene


class StaleCommandError(RuntimeError):
    """The scene was reset or reconfigured while a request was in flight."""


@dataclass
class Trajectory:
    positions: np.ndarray
    dt: float
    generation: int
    start_time: float
    done: bool = False
    error: str | None = None
    result: dict = field(default_factory=dict)


def _transform(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation).reshape(3, 3)
    matrix[:3, 3] = position
    return matrix.tolist()


class Simulation:
    """The physics thread never performs network I/O or waits for decisions.

    Calls can block their caller while this thread integrates. The condition
    lock only protects short state accesses; model compilation and mesh export
    occur outside it. Commands change torque references, never robot qpos.
    """

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._build_lock = threading.Lock()
        self._motion_locks = {arm: threading.Lock() for arm in ARM_JOINTS}
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._error = None
        self._generation = 1
        self._revision = 1
        self._confirmed = False
        self._baseline = None
        self._commands = {}
        self._gripper_history = {arm: deque(maxlen=3000) for arm in ARM_JOINTS}
        self._builder = ModelBuilder()
        self._sensor_lock = threading.Lock()
        self._sensor_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgbd-render")
        self._sensor_renderer = None
        self._sensor_model = None
        self._latest_capture = None
        self._latest_preview = None
        self._preview_started = 0.
        self._task_capture_pending = 0
        self._scene = normalize_scene(DEFAULT_SCENE)
        self._model = self._builder.build(self._scene)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._geometries = self._builder.render_geometries(self._model, self._data)
        self._bind_indices()
        self._hold_current_pose()

    def _bind_indices(self):
        model = self._model
        self._arm_q = {arm: np.array([int(model.joint(name).qposadr[0]) for name in names])
                       for arm, names in ARM_JOINTS.items()}
        self._arm_v = {arm: np.array([int(model.joint(name).dofadr[0]) for name in names])
                       for arm, names in ARM_JOINTS.items()}
        self._arm_act = {arm: np.array([model.actuator("act_" + name).id for name in names])
                         for arm, names in ARM_JOINTS.items()}
        self._tcp_ids = {arm: model.body(name).id for arm, name in TCP_BODIES.items()}
        self._grip_q = {arm: int(model.joint(name).qposadr[0]) for arm, name in GRIP_DRIVERS.items()}
        self._grip_v = {arm: int(model.joint(name).dofadr[0]) for arm, name in GRIP_DRIVERS.items()}
        self._grip_act = {arm: model.actuator("act_" + name).id for arm, name in GRIP_DRIVERS.items()}
        self._object_ids = {item["id"]: model.body("object_" + item["id"]).id for item in self._scene["objects"]}
        self._geometry_ids = [int(item["id"]) for item in self._geometries]

    def _hold_current_pose(self):
        self._desired_q = {arm: self._data.qpos[indices].copy() for arm, indices in self._arm_q.items()}
        self._desired_v = {arm: np.zeros(7) for arm in ARM_JOINTS}
        self._desired_grip = {arm: float(self._data.qpos[index]) for arm, index in self._grip_q.items()}

    def start(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("simulation has been closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="mujoco-physics", daemon=True)
            self._thread.start()

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._commands.clear()
            self._stop.set()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._sensor_lock:
            self._sensor_executor.submit(self._close_sensor).result()
            self._sensor_executor.shutdown(wait=True)
        self._builder.close()

    def _close_sensor(self):
        if self._sensor_renderer is not None:
            self._sensor_renderer.close()
            self._sensor_renderer = None
            self._sensor_model = None

    def _render_capture(self, model, data, generation, arms, rgb_only=False):
        from .cameras import RgbdRenderer, clone_render_model

        if self._sensor_model is not model:
            self._close_sensor()
            self._sensor_renderer = RgbdRenderer(clone_render_model(model))
            self._sensor_model = model
        return self._sensor_renderer.capture(data, generation, arms, rgb_only=rgb_only)

    def capture_rgbd(self):
        """Freeze briefly, render on one GL thread, and reject a superseded frame."""
        with self._condition:
            self._task_capture_pending += 1
        try:
            with self._sensor_lock:
                with self._condition:
                    if self._closed:
                        raise RuntimeError("simulation has been closed")
                    model = self._model
                    data = mujoco.MjData(model)
                    mujoco.mj_copyData(data, model, self._data)
                    captured_monotonic_s = time.monotonic()
                    generation = self._generation
                    arms = self._robot_snapshot()["arms"]
                captured = self._sensor_executor.submit(self._render_capture, model, data, generation, arms).result()
                captured["captured_monotonic_s"] = captured_monotonic_s
                with self._condition:
                    self._check_generation(generation)
                    self._latest_capture = captured
                return captured
        finally:
            with self._condition:
                self._task_capture_pending -= 1

    def preview_rgb(self):
        """Demand-driven latest RGB group; busy frames are dropped, never queued."""
        with self._condition:
            if self._latest_preview and self._latest_preview["generation"] != self._generation:
                self._latest_preview = None
            if self._task_capture_pending or time.monotonic() - self._preview_started < .5:
                return self._latest_preview
        if not self._sensor_lock.acquire(blocking=False):
            return self._latest_preview
        try:
            with self._condition:
                if self._closed or self._task_capture_pending:
                    return self._latest_preview
                self._preview_started = time.monotonic()
                model = self._model
                data = mujoco.MjData(model)
                mujoco.mj_copyData(data, model, self._data)
                captured_monotonic_s = time.monotonic()
                generation = self._generation
            captured = self._sensor_executor.submit(self._render_capture, model, data, generation, {}, True).result()
            captured["captured_monotonic_s"] = captured_monotonic_s
            with self._condition:
                self._check_generation(generation)
                self._latest_preview = captured
            return captured
        finally:
            self._sensor_lock.release()

    def latest_rgbd(self):
        with self._condition:
            if self._latest_capture is None or self._latest_capture["generation"] != self._generation:
                return None
            return self._latest_capture

    def robot_snapshot(self):
        """Only proprioception and command ownership; no environment identities or poses."""
        with self._condition:
            return self._robot_snapshot()

    def _robot_snapshot(self):
        # mj_step integrates qpos after computing Cartesian poses; refresh at read time.
        mujoco.mj_fwdPosition(self._model, self._data)
        arms = {}
        for arm in ARM_JOINTS:
            body_id = self._tcp_ids[arm]
            arms[arm] = {"joints_rad": self._data.qpos[self._arm_q[arm]].tolist(),
                         "tcp": _transform(self._data.xpos[body_id], self._data.xmat[body_id]),
                         "gripper_closed_fraction": self._gripper_fraction(arm),
                         "gripper_position_rad": float(self._data.qpos[self._grip_q[arm]]),
                         "gripper_limits_rad": self._model.joint(GRIP_DRIVERS[arm]).range.tolist(),
                         "gripper_closed_command": self._desired_grip[arm] == float(self._model.joint(GRIP_DRIVERS[arm]).range[1]),
                         "gripper_position_samples": [[t, q] for g, t, q in self._gripper_history[arm]
                                                      if g == self._generation and t >= self._data.time - 1.1]}
            arms[arm]['tcp_front'] = front_tcp_pose(arms[arm]['tcp'], arm).tolist()
        return {"generation": self._generation, "time": float(self._data.time), "confirmed": self._confirmed,
                "arms": arms, "physics_error": self._error, "active_commands": list(self._commands)}

    def _check_running(self):
        if self._closed:
            raise RuntimeError("simulation has been closed")
        if self._error:
            raise RuntimeError(f"physics thread failed: {self._error}")
        if self._thread is None:
            raise RuntimeError("start the simulation before waiting for physics")

    def _check_generation(self, generation):
        if generation != self._generation:
            raise StaleCommandError("scene generation changed; the old motion request was cancelled")
        self._check_running()

    @staticmethod
    def _check_arm(arm):
        if arm not in ARM_JOINTS:
            raise ValueError("arm must be A or B")

    def _run(self):
        next_step = time.monotonic()
        try:
            while not self._stop.is_set():
                with self._condition:
                    self._update_trajectories()
                    self._apply_motor_controls()
                    mujoco.mj_step(self._model, self._data)
                    for arm in ARM_JOINTS:
                        self._gripper_history[arm].append((self._generation, float(self._data.time),
                                                          float(self._data.qpos[self._grip_q[arm]])))
                    if not np.isfinite(self._data.qpos).all() or not np.isfinite(self._data.qvel).all():
                        raise RuntimeError("non-finite physics state")
                    self._condition.notify_all()
                    dt = self._model.opt.timestep
                next_step += dt
                delay = next_step - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                elif delay < -0.1:
                    next_step = time.monotonic()
        except Exception as error:  # noqa: BLE001 - expose worker failures to every API caller.
            with self._condition:
                self._error = str(error)
                self._condition.notify_all()

    def _apply_motor_controls(self):
        for arm in ARM_JOINTS:
            indices = self._arm_v[arm]
            error = self._desired_q[arm] - self._data.qpos[self._arm_q[arm]]
            velocity_error = self._desired_v[arm] - self._data.qvel[indices]
            torque = 12000 * error + 600 * velocity_error
            limits = self._model.actuator_forcerange[self._arm_act[arm], 1]
            self._data.ctrl[self._arm_act[arm]] = np.clip(torque, -limits, limits)
            self._data.ctrl[self._grip_act[arm]] = self._desired_grip[arm]

    def _update_trajectories(self):
        for arm, command in list(self._commands.items()):
            if command.done:
                continue
            elapsed = float(self._data.time) - command.start_time
            duration = (len(command.positions) - 1) * command.dt
            if elapsed < duration:
                index = min(int(elapsed / command.dt), len(command.positions) - 2)
                fraction = (elapsed - index * command.dt) / command.dt
                self._desired_q[arm] = ((1 - fraction) * command.positions[index]
                                        + fraction * command.positions[index + 1])
                self._desired_v[arm] = (command.positions[index + 1] - command.positions[index]) / command.dt
                continue
            self._desired_q[arm] = command.positions[-1].copy()
            self._desired_v[arm] = np.zeros(7)
            error = float(np.max(np.abs(command.positions[-1] - self._data.qpos[self._arm_q[arm]])))
            speed = float(np.max(np.abs(self._data.qvel[self._arm_v[arm]])))
            if elapsed >= duration + 0.08 and error < 0.03 and speed < 0.08:
                command.done = True
                command.result = {"arm": arm, "generation": command.generation,
                                  "duration_s": elapsed, "max_final_error_rad": error,
                                  "joints_rad": self._data.qpos[self._arm_q[arm]].tolist()}
            elif elapsed > duration + 2.0:
                command.done = True
                command.error = f"arm {arm} failed to track trajectory: final error {error:.4f} rad"

    def snapshot(self):
        with self._condition:
            return self._snapshot()

    def _snapshot(self):
        mujoco.mj_fwdPosition(self._model, self._data)
        data = self._data
        objects = []
        for definition in self._scene["objects"]:
            body_id = self._object_ids[definition["id"]]
            item = deepcopy(definition)
            item["position"] = data.xpos[body_id].tolist()
            item["quaternion"] = data.xquat[body_id].tolist()
            item["T_world_object"] = _transform(data.xpos[body_id], data.xmat[body_id])
            quaternion = data.xquat[body_id]
            item["yaw_deg"] = float(np.degrees(np.arctan2(
                2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2))))
            objects.append(item)
        arms = {}
        for arm in ARM_JOINTS:
            body_id = self._tcp_ids[arm]
            arms[arm] = {"joints_rad": data.qpos[self._arm_q[arm]].tolist(),
                         "tcp": _transform(data.xpos[body_id], data.xmat[body_id]),
                         "gripper_closed_fraction": self._gripper_fraction(arm)}
            arms[arm]['tcp_front'] = front_tcp_pose(arms[arm]['tcp'], arm).tolist()
        geometry_poses = []
        for index in self._geometry_ids:
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.geom_xmat[index])
            geometry_poses.append({"id": str(index), "position": data.geom_xpos[index].tolist(),
                                   "quaternion": quaternion.tolist()})
        return {"generation": self._generation, "time": float(data.time), "confirmed": self._confirmed,
                "objects": objects, "arms": arms, "table": deepcopy(self._scene["table"]),
                "geometry_poses": geometry_poses, "manifest_revision": self._revision,
                "physics_error": self._error, "active_commands": list(self._commands)}

    def manifest(self):
        with self._condition:
            return {"revision": self._revision, "geometries": deepcopy(self._geometries),
                    "quaternion_order": "wxyz", "up_axis": "z", "units": "meters"}

    def collision_snapshot(self):
        from .collision_view import collision_sphere_snapshot

        with self._condition:
            result, poses = collision_sphere_snapshot(self._model, self._data, self._geometry_ids)
            snapshot = self._snapshot()
            snapshot["geometry_poses"] = poses
            for arm in snapshot["arms"].values():
                arm["gripper_closed_fraction"] = 0.
            return {**result, "simulation": snapshot}

    def configure(self, scene):
        normalized = normalize_scene(scene)
        with self._condition:
            if self._closed:
                raise RuntimeError("simulation has been closed")
            self._generation += 1
            generation = self._generation
            self._commands.clear()
            self._hold_current_pose()
            self._confirmed = False
            self._condition.notify_all()
        # Serial compilation avoids overlapping writes to render mesh cache.
        with self._build_lock:
            model = self._builder.build(normalized)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            geometries = self._builder.render_geometries(model, data)
        with self._condition:
            if generation != self._generation:
                raise StaleCommandError("scene edit superseded by a newer scene or reset")
            # Scene editing must not introduce an observation or startup arm pose.
            for name in ARM_JOINTS["A"] + ARM_JOINTS["B"] + list(GRIP_DRIVERS.values()):
                old_joint, new_joint = self._model.joint(name), model.joint(name)
                data.qpos[int(new_joint.qposadr[0])] = self._data.qpos[int(old_joint.qposadr[0])]
            # Keep all gripper follower positions, including loaded contacts.
            for index in range(self._model.njnt):
                name = self._model.joint(index).name
                if name.startswith("grip"):
                    data.qpos[int(model.joint(name).qposadr[0])] = self._data.qpos[int(self._model.jnt_qposadr[index])]
            data.time = self._data.time
            self._model, self._data, self._scene = model, data, normalized
            self._geometries = geometries
            self._revision += 1
            self._bind_indices()
            self._hold_current_pose()
            mujoco.mj_forward(model, data)
            self._condition.notify_all()
            return self._snapshot()

    def settle(self, timeout_s=5.0):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        with self._condition:
            self._check_running()
            generation = self._generation
            deadline = time.monotonic() + timeout_s
            stable_since = None
            while True:
                self._check_generation(generation)
                speed = 0.0
                for item in self._scene["objects"]:
                    if item['kind'] == 'marker':
                        continue
                    address = int(self._model.joint("free_" + item["id"]).dofadr[0])
                    speed = max(speed, float(np.max(np.abs(self._data.qvel[address:address + 6]))))
                if speed < 0.015:
                    stable_since = self._data.time if stable_since is None else stable_since
                else:
                    stable_since = None
                settled = stable_since is not None and self._data.time - stable_since >= 0.25
                if settled or time.monotonic() >= deadline:
                    result = self._snapshot()
                    result["settled"] = bool(settled)
                    result["max_object_speed"] = speed
                    return result
                self._condition.wait(timeout=0.02)

    def confirm(self):
        with self._condition:
            self._check_running()
            if self._commands:
                raise RuntimeError("cannot confirm a baseline while a motor trajectory is active")
        settled = self.settle(timeout_s=5.0)
        if not settled["settled"]:
            raise RuntimeError("objects are still moving; wait for the scene to settle before confirming")
        with self._condition:
            self._check_generation(settled["generation"])
            if self._commands:
                raise RuntimeError("cannot confirm a baseline while a motor trajectory is active")
            state_type = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(self._model, state_type))
            mujoco.mj_getState(self._model, self._data, state, state_type)
            self._baseline = {"model": self._model, "state": state, "scene": deepcopy(self._scene),
                              "geometries": deepcopy(self._geometries),
                              "desired_q": deepcopy(self._desired_q), "desired_v": deepcopy(self._desired_v),
                              "desired_grip": deepcopy(self._desired_grip)}
            self._confirmed = True
            return self._snapshot()

    def reset(self):
        with self._condition:
            self._check_running()
            if self._baseline is None:
                raise RuntimeError("no confirmed scene baseline; settle and confirm the scene first")
            baseline = self._baseline
            self._generation += 1
            self._commands.clear()
            self._model = baseline["model"]
            self._data = mujoco.MjData(self._model)
            mujoco.mj_setState(self._model, self._data, baseline["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(self._model, self._data)
            self._scene = deepcopy(baseline["scene"])
            self._geometries = deepcopy(baseline["geometries"])
            self._desired_q = deepcopy(baseline["desired_q"])
            self._desired_v = deepcopy(baseline["desired_v"])
            self._desired_grip = deepcopy(baseline["desired_grip"])
            self._revision += 1
            self._confirmed = True
            self._bind_indices()
            self._condition.notify_all()
            return self._snapshot()

    def execute_trajectory(self, arm, positions_rad, dt, generation):
        self._check_arm(arm)
        positions = np.asarray(positions_rad, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 7 or len(positions) < 1 or not np.isfinite(positions).all():
            raise ValueError("positions_rad must be a nonempty finite N-by-7 array")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("trajectory dt must be positive and finite")
        with self._motion_locks[arm], self._condition:
            self._check_generation(generation)
            if np.max(np.abs(self._data.qpos[self._arm_q[arm]] - positions[0])) > np.deg2rad(.05):
                raise ValueError("trajectory_start_changed_after_collision_check")
            for index, name in enumerate(ARM_JOINTS[arm]):
                lower, upper = self._model.joint(name).range
                if np.any(positions[:, index] < lower - 0.001) or np.any(positions[:, index] > upper + 0.001):
                    raise ValueError(f"trajectory exceeds {name} joint limits")
            command = Trajectory(positions.copy(), float(dt), generation, float(self._data.time))
            self._commands[arm] = command
            wall_deadline = time.monotonic() + max(10, len(positions) * dt * 4 + 5)
            try:
                while not command.done:
                    self._check_generation(generation)
                    if time.monotonic() > wall_deadline:
                        raise TimeoutError("physics did not complete the trajectory before the wall-clock timeout")
                    self._condition.wait(timeout=0.03)
                self._check_generation(generation)
                if command.error:
                    raise RuntimeError(command.error)
                return command.result
            finally:
                if self._commands.get(arm) is command:
                    self._commands.pop(arm)
                    if not command.done:
                        self._desired_q[arm] = self._data.qpos[self._arm_q[arm]].copy()
                        self._desired_v[arm] = np.zeros(7)

    def execute_dual_trajectory(self, paths, dt, generation):
        if set(paths) != {"A", "B"} or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dual trajectory requires A/B paths and positive dt")
        positions = {arm: np.asarray(paths[arm], dtype=float) for arm in ("A", "B")}
        for path in positions.values():
            if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all():
                raise ValueError("dual paths must be finite N-by-7 arrays")
        if positions["A"].shape != positions["B"].shape:
            raise ValueError("dual paths must have identical sample counts")
        with self._motion_locks["A"], self._motion_locks["B"], self._condition:
            self._check_generation(generation)
            for arm, path in positions.items():
                if np.max(np.abs(self._data.qpos[self._arm_q[arm]] - path[0])) > np.deg2rad(.05):
                    raise ValueError("trajectory_start_changed_after_collision_check")
                for index, name in enumerate(ARM_JOINTS[arm]):
                    lower, upper = self._model.joint(name).range
                    if np.any(path[:, index] < lower - .001) or np.any(path[:, index] > upper + .001):
                        raise ValueError(f"trajectory exceeds {name} joint limits")
            start_time = float(self._data.time)
            commands = {arm: Trajectory(path.copy(), float(dt), generation, start_time)
                        for arm, path in positions.items()}
            self._commands.update(commands)
            deadline = time.monotonic() + max(10, len(positions["A"]) * dt * 4 + 5)
            try:
                while not all(command.done for command in commands.values()):
                    self._check_generation(generation)
                    for command in commands.values():
                        if command.error:
                            raise RuntimeError(command.error)
                    if time.monotonic() > deadline:
                        raise TimeoutError("physics did not complete the dual trajectory")
                    self._condition.wait(timeout=.03)
                self._check_generation(generation)
                for command in commands.values():
                    if command.error:
                        raise RuntimeError(command.error)
                return {"synchronized": True, "start_time": start_time,
                        "arms": {arm: command.result for arm, command in commands.items()}}
            finally:
                for arm, command in commands.items():
                    if self._commands.get(arm) is command:
                        self._commands.pop(arm)
                        if not command.done:
                            self._desired_q[arm] = self._data.qpos[self._arm_q[arm]].copy()
                            self._desired_v[arm] = np.zeros(7)

    def dwell(self, duration_s, generation):
        """Hold current commanded joints for the requested wall-clock duration."""
        if not np.isfinite(duration_s) or not 0 <= duration_s <= 30:
            raise ValueError("dwell duration must be between 0 and 30 seconds")
        with self._condition:
            self._check_generation(generation)
            simulation_started = float(self._data.time)
            started = time.monotonic()
            deadline = started + duration_s
            while (remaining := deadline - time.monotonic()) > 0:
                self._check_generation(generation)
                self._condition.wait(timeout=min(.02, remaining))
            self._check_generation(generation)
            return {"completed": True, "requested_duration_s": duration_s,
                    "duration_s": time.monotonic() - started,
                    "simulation_duration_s": float(self._data.time) - simulation_started}

    def _gripper_fraction(self, arm):
        lower, upper = self._model.joint(GRIP_DRIVERS[arm]).range
        return float(np.clip((self._data.qpos[self._grip_q[arm]] - lower) / (upper - lower), 0, 1))

    def set_gripper(self, arm, closed, generation):
        self._check_arm(arm)
        if not isinstance(closed, bool):
            raise ValueError("closed must be a boolean")  # noqa: TRY004 - JSON validation uses ValueError.
        with self._motion_locks[arm], self._condition:
            self._check_generation(generation)
            lower, upper = self._model.joint(GRIP_DRIVERS[arm]).range
            self._desired_grip[arm] = float(upper if closed else lower)
            start_time = float(self._data.time)
            deadline = time.monotonic() + 8
            while True:
                self._check_generation(generation)
                elapsed = self._data.time - start_time
                speed = abs(self._data.qvel[self._grip_v[arm]])
                if elapsed >= 0.5 and speed < 0.015 or elapsed >= 2.0:
                    return {"arm": arm, "generation": generation, "closed_command": closed,
                            "gripper_closed_fraction": self._gripper_fraction(arm), "duration_s": elapsed}
                if time.monotonic() >= deadline:
                    raise TimeoutError("physics did not settle the gripper before the wall-clock timeout")
                self._condition.wait(timeout=0.02)
