"""Copy the pinned public batch planner's two pre-release phase buffers."""

from dataclasses import asdict

import numpy as np

from .planning_batches import TRAJECTORY_BATCH_CAPACITY
from .trajectory_postprocessing import RawTrajectory, _phase_samples, phase_terminal_state

PLACE_BATCH_CAPACITY = TRAJECTORY_BATCH_CAPACITY


def place_batch_items_from_result(result, joint_names, interpolation_dt_s):
    """Keep every candidate index; failed phases never become executable paths."""
    flags = {}
    for name in ("approach", "grasp"):
        values = np.asarray(getattr(result, name + "_success").detach().cpu().numpy())
        if values.shape != (PLACE_BATCH_CAPACITY,) or not np.isin(values, [0, 1]).all():
            raise ValueError(f"invalid {name} batch success flags")
        flags[name] = values.astype(bool)
    successful = flags["approach"] & flags["grasp"]
    samples = {}
    if successful.any():
        for name in ("approach", "grasp"):
            q, v, a, names = _phase_samples(result, name, PLACE_BATCH_CAPACITY)
            lengths = np.asarray(getattr(result, name + "_interpolated_last_tstep").detach().cpu().numpy())
            if lengths.size != PLACE_BATCH_CAPACITY:
                raise ValueError(f"invalid {name} batch interpolation lengths")
            samples[name] = (q, v, a, names, lengths.reshape(-1))
    planning_time = float(getattr(result, "planning_time", 0.) or 0.)
    if not np.isfinite(planning_time) or planning_time < 0:
        planning_time = 0.
    items = []
    for index in range(PLACE_BATCH_CAPACITY):
        item = {"index": index, "success": bool(successful[index]),
                "approach_success": bool(flags["approach"][index]),
                "place_success": bool(flags["grasp"][index]), "phases": {},
                "status": "success" if successful[index] else str(result.status or "planning_failed")}
        if successful[index]:
            for phase_name, sdk_name in (("approach", "approach"), ("place", "grasp")):
                q, v, a, names, lengths = samples[sdk_name]
                length = lengths[index]
                if not np.isfinite(length) or length != int(length) or not 2 <= length <= q.shape[2]:
                    raise ValueError(f"invalid {phase_name} batch interpolation length")
                order = [names.index(joint) for joint in joint_names]
                positions, velocities, accelerations = [
                    values[index, 0, :int(length)][:, order].copy() for values in (q, v, a)]
                phase = RawTrajectory(positions_rad=positions.tolist(), joint_names=tuple(joint_names),
                    velocities_rad_s=velocities, accelerations_rad_s2=accelerations,
                    terminal_state=phase_terminal_state(result, sdk_name, joint_names, index),
                    interpolation_dt_s=interpolation_dt_s, planning_time_s=planning_time,
                    position_error_m=0., rotation_error_rad=0.)
                item["phases"][phase_name] = asdict(phase)
        items.append(item)
    return items
