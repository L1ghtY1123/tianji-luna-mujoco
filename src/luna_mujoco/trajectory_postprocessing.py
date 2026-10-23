"""Preserve raw cuRobo derivatives and prepare candidate execution trajectories."""

import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
from tianji_curobo.contracts import NativeTrajectory
from tianji_curobo.trajectory_interpolation import quintic_hermite_points


@dataclass(frozen=True, kw_only=True)
class RawTrajectory(NativeTrajectory):
    """Native samples plus exact derivatives, without execution point insertion."""

    velocities_rad_s: tuple[tuple[float, ...], ...]
    accelerations_rad_s2: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        super().__post_init__()
        shape = np.asarray(self.positions_rad).shape
        for name in ("velocities_rad_s", "accelerations_rad_s2"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{name} must match finite native trajectory positions")
            object.__setattr__(self, name, tuple(tuple(row) for row in values.tolist()))


@dataclass(frozen=True)
class PreReleasePlacePlan:
    """Real attached phases; retreat is planned after measured release."""

    approach: RawTrajectory
    place: RawTrajectory
    scene_token: str | None = field(init=False)

    def __post_init__(self):
        if not all(isinstance(phase, RawTrajectory) for phase in (self.approach, self.place)):
            raise ValueError("pre-release phases must retain native positions and derivatives")
        if self.approach.scene_token != self.place.scene_token:
            raise ValueError("pre-release phases must use the same scene token")
        if self.approach.joint_names != self.place.joint_names:
            raise ValueError("pre-release phases must use the same joint order")
        object.__setattr__(self, "scene_token", self.approach.scene_token)


def _phase_samples(raw_result, name, batch_size):
    trajectory = getattr(raw_result, f"{name}_interpolated_trajectory", None)
    if trajectory is None:
        raise ValueError(f"missing {name} positions and derivatives")
    arrays = []
    for sample_field in ("position", "velocity", "acceleration"):
        tensor = getattr(trajectory, sample_field, None)
        if tensor is None:
            raise ValueError(f"missing {name} positions or derivative: {sample_field}")
        try:
            arrays.append(np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid {name} positions or derivative: {sample_field}") from error
    q, v, a = arrays
    names = getattr(trajectory, "joint_names", None)
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(joint, str) or not joint for joint in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"invalid {name} joint names")
    if q.ndim != 4 or q.shape[:2] != (batch_size, 1) or q.shape[-1] != len(names):
        raise ValueError(f"invalid {name} positions shape; expected [B, 1, T, D]")
    if v.shape != q.shape or a.shape != q.shape:
        raise ValueError(f"invalid {name} derivative shape")
    return q, v, a, names


def add_trajectory_derivatives(native, raw_result):
    """Copy the SDK's q/v/a before the next solver call reuses its buffers."""
    if not any(item.success for item in native.items):
        return native
    samples = {name: _phase_samples(raw_result, name, native.batch_size)
               for name in ("approach", "grasp", "lift")}
    items = []
    for item in native.items:
        if not item.success:
            items.append(item)
            continue
        phases = {}
        for name in ("approach", "grasp", "lift"):
            phase = getattr(item, name)
            if phase is None:
                raise ValueError(f"successful candidate is missing {name} positions")
            q, v, a, joint_names = samples[name]
            order = [joint_names.index(joint) for joint in phase.joint_names]
            count = len(phase.positions_rad)
            if count < 2 or count > q.shape[2]:
                raise ValueError(f"invalid {name} positions length")
            selected = [values[item.candidate_index, 0, :count][:, order] for values in (q, v, a)]
            positions, velocities, accelerations = selected
            if not all(np.isfinite(values).all() for values in selected):
                raise ValueError(f"{name} positions and derivatives must be finite")
            if not np.array_equal(positions, np.asarray(phase.positions_rad)):
                raise ValueError(f"{name} positions do not match the native trajectory")
            phases[name] = RawTrajectory(
                **asdict(phase), velocities_rad_s=velocities,
                accelerations_rad_s2=accelerations,
            )
        items.append(replace(item, **phases))
    return replace(native, items=tuple(items))


def add_place_derivatives(native, raw_result):
    """Retain all compound place phase buffers before another planner call."""
    phases = {}
    for native_name, sdk_name in (("approach", "approach"), ("place", "grasp"), ("retreat", "lift")):
        phase = getattr(native, native_name)
        q, v, a, names = _phase_samples(raw_result, sdk_name, 1)
        order = [names.index(joint) for joint in phase.joint_names]
        count = len(phase.positions_rad)
        if count < 2 or count > q.shape[2]:
            raise ValueError(f"invalid {native_name} trajectory length")
        selected = [values[0, 0, :count][:, order].copy() for values in (q, v, a)]
        if not np.array_equal(selected[0], np.asarray(phase.positions_rad)):
            raise ValueError(f"{native_name} positions disagree with SDK buffers")
        phases[native_name] = RawTrajectory(**asdict(phase), velocities_rad_s=selected[1], accelerations_rad_s2=selected[2])
    return replace(native, **phases)


def pre_release_place_from_result(raw_result, joint_names, interpolation_dt_s):
    """Copy only the public approach/grasp samples, respecting valid lengths."""
    planning_time = float(getattr(raw_result, "planning_time", 0.) or 0.)
    if not np.isfinite(planning_time) or planning_time < 0:
        planning_time = 0.
    phases = {}
    for project_name, sdk_name in (("approach", "approach"), ("place", "grasp")):
        q, v, a, names = _phase_samples(raw_result, sdk_name, 1)
        length_tensor = getattr(raw_result, f"{sdk_name}_interpolated_last_tstep", None)
        try:
            length = np.asarray(length_tensor.detach().cpu().numpy(), dtype=float).reshape(-1)
            if (length.size != 1 or not np.isfinite(length[0])
                    or length[0] != int(length[0]) or not 2 <= length[0] <= q.shape[2]):
                raise ValueError("out of range")
            count = int(length[0])
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid {project_name} interpolation length") from error
        order = [names.index(joint) for joint in joint_names]
        positions, velocities, accelerations = [values[0, 0, :count][:, order].copy() for values in (q, v, a)]
        phases[project_name] = RawTrajectory(positions_rad=positions.tolist(), joint_names=tuple(joint_names),
            velocities_rad_s=velocities, accelerations_rad_s2=accelerations,
            interpolation_dt_s=interpolation_dt_s, planning_time_s=planning_time,
            position_error_m=0., rotation_error_rad=0.)
    return PreReleasePlacePlan(**phases)


def add_pose_derivatives(native, raw_result):
    """Copy a detached retreat's single public interpolated plan."""
    trajectory = raw_result.get_interpolated_plan()
    names = trajectory.joint_names
    arrays = []
    for name in ("position", "velocity", "acceleration"):
        tensor = getattr(trajectory, name)
        if tensor is None:
            raise ValueError(f"missing pose derivative {name}")
        values = np.asarray(tensor.detach().cpu().numpy(), dtype=float)
        if values.ndim != 4 or values.shape[:2] != (1, 1):
            raise ValueError("invalid pose trajectory shape")
        order = [names.index(joint) for joint in native.joint_names]
        arrays.append(values[0, 0, :len(native.positions_rad)][:, order].copy())
    if not np.array_equal(arrays[0], np.asarray(native.positions_rad)):
        raise ValueError("pose positions disagree with SDK buffers")
    return RawTrajectory(**asdict(native), velocities_rad_s=arrays[1], accelerations_rad_s2=arrays[2])


def build_execution_phases(raw_phases, scale):
    """Insert execution points for one candidate; do not run collision checks."""
    phases = {}
    for name in raw_phases:
        raw = raw_phases[name]
        positions = raw["positions_rad"]
        dt = raw["interpolation_dt_s"]
        command = quintic_hermite_points(
            positions, raw["velocities_rad_s"], raw["accelerations_rad_s2"],
            sample_dt_s=dt, point_scale=scale,
        )
        phases[name] = {
            "positions_rad": command.tolist(), "raw_dt": dt, "dt": dt,
            "raw_duration_s": (len(positions) - 1) * dt,
            "command_duration_s": (len(command) - 1) * dt,
            "raw_point_count": len(positions), "command_point_count": len(command),
        }
    return phases
