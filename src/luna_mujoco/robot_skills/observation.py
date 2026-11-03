"""Bounded reuse of sensor evidence; no object truth or task policy."""
import math
import time

import numpy as np


def reusable_release_capture(context, feedback, *, epoch, binding):
    """A failed verdict does not invalidate an unchanged post-release exposure."""
    capture = context.get('latest_capture')
    if not capture:
        return None
    verified_at = binding.get('release_verified_perf_counter_ns')
    captured_at = capture.get('captured_perf_counter_ns')
    after_release = (type(captured_at) is int and captured_at > verified_at
                     if type(verified_at) is int else
                     capture.get('time', -math.inf) > binding['release_feedback_time'])
    if not after_release:
        return None
    reason = capture_reuse_rejection(capture, feedback,
        epoch_matches=epoch is not None and context.get('capture_epochs', {}).get(
            capture['observation_id']) == epoch, now=time.monotonic(), max_age_s=None)
    return capture if reason is None else None


def capture_reuse_rejection(capture, feedback, *, epoch_matches, now, max_age_s=10.):
    """Return why a photograph must be refreshed, or None for reusable evidence.

    This checks acquisition provenance and proprioception, not environmental
    safety. The original timestamp remains authoritative for every consumer.
    """
    if not capture or not capture.get('observation_id'):
        return 'no_previous_capture'
    if not epoch_matches:
        return 'action_since_capture'
    for field in ('generation', 'manifest_revision', 'command_revision'):
        if type(capture.get(field)) is not int or capture[field] != feedback.get(field):
            return field + '_changed_or_missing'
    if feedback.get('physics_error') or capture.get('active_commands') != [] or feedback.get('active_commands') != []:
        return 'robot_not_idle'
    captured_at = capture.get('captured_monotonic_s')
    if (type(captured_at) not in (int, float) or not math.isfinite(captured_at)
            or now < captured_at or max_age_s is not None and now - captured_at > max_age_s):
        return 'capture_outside_reuse_window'
    if set(capture.get('views', {})) != {'1', '2', '3'}:
        return 'incomplete_camera_group'
    for arm in ('A', 'B'):
        previous = capture.get('arms', {}).get(arm, {})
        current = feedback.get('arms', {}).get(arm, {})
        for field, shape, tolerance in (('joints_rad', (7,), np.deg2rad(.05)),
                                        ('gripper_position_rad', (), 1e-4),
                                        ('gripper_command_position_rad', (), 1e-4)):
            before = np.asarray(previous.get(field), dtype=float)
            after = np.asarray(current.get(field), dtype=float)
            if (before.shape != shape or after.shape != shape or not np.isfinite(before).all()
                    or not np.isfinite(after).all() or not np.allclose(before, after, atol=tolerance, rtol=0)):
                return arm + '_' + field + '_changed_or_missing'
    return None
