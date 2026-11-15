"""Join checked phase trajectories on one time axis with scheduled gripper commands."""
import numpy as np


def build_sequence(phases, order, *, gripper_events, gripper_transition_s):
    positions, timestamps, events = [], [], []
    elapsed = 0.
    for phase in order:
        command = phases[phase]
        path = np.asarray(command['positions_rad'], dtype=float)
        dt = float(command['dt'])
        if (path.ndim != 2 or path.shape[1] != 7 or len(path) < 2
                or not np.isfinite(path).all() or not np.isfinite(dt) or dt <= 0):
            raise ValueError('invalid_sequence_phase:' + phase)
        if positions and not np.allclose(positions[-1], path[0], atol=1e-6, rtol=0):
            raise ValueError('sequence_phase_discontinuity:' + phase)
        start = 1 if positions else 0
        positions.extend(path[start:])
        timestamps.extend(elapsed + np.arange(start, len(path)) * dt)
        elapsed += (len(path) - 1) * dt
        if phase in gripper_events:
            event = {**gripper_events[phase], 'time_s': elapsed, 'phase': phase}
            events.append(event)
            # The actuator gets its configured command duration, without a
            # feedback verdict, network call or visual acquisition at this point.
            elapsed += gripper_transition_s
            positions.append(path[-1])
            timestamps.append(elapsed)
    return {'positions_rad': np.asarray(positions).tolist(),
            'times_s': np.asarray(timestamps).tolist(), 'events': events}
