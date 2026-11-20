"""Bounded reuse of sensor evidence; no object truth or task policy."""
import math
import time

import numpy as np


def capture_reuse_rejection(capture, feedback, *, epoch_matches, now, max_age_s=10.):
    """Return why a photograph must be refreshed, or None for reusable evidence.

    This checks acquisition provenance and proprioception, not environmental
    safety. The original timestamp remains authoritative for every consumer.
    """
    if not capture or not capture.get('observation_id'):
        return 'no_previous_capture'
    if not epoch_matches:
        return 'action_since_capture'
    for field in ('generation', 'command_revision'):
        if type(capture.get(field)) is not int or capture[field] != feedback.get(field):
            return field + '_changed_or_missing'
    revision = capture.get('manifest_revision')
    if not isinstance(revision, str) or not revision or revision != feedback.get('manifest_revision'):
        return 'manifest_revision_changed_or_missing'
    if not isinstance(capture.get('active_commands'), list) or not isinstance(feedback.get('active_commands'), list):
        return 'command_state_unavailable'
    if capture['active_commands'] or feedback['active_commands']:
        return 'robot_not_idle'
    if feedback.get('physics_error'):
        return 'robot_feedback_error'
    captured_at = capture.get('captured_monotonic_s')
    if (type(captured_at) not in (int, float) or not math.isfinite(captured_at)
            or now < captured_at or max_age_s is not None and now - captured_at > max_age_s):
        return 'capture_outside_reuse_window'
    if set(capture.get('views', {})) != {'1', '2', '3'}:
        return 'incomplete_camera_group'
    return stationary_feedback_rejection(capture, feedback)


def stationary_feedback_rejection(capture, feedback):
    for arm in ('A', 'B'):
        previous = capture.get('arms', {}).get(arm, {})
        current = feedback.get('arms', {}).get(arm, {})
        fields = [('joints_rad', (7,), np.deg2rad(.05)), ('gripper_position_rad', (), 1e-4)]
        # A freshly started Base may have no issued gripper target. Compare
        # echoes whenever either observation has one; never infer a command
        # from the measured opening.
        if 'gripper_command_position_rad' in previous or 'gripper_command_position_rad' in current:
            fields.append(('gripper_command_position_rad', (), 1e-4))
        for field, shape, tolerance in fields:
            before = np.asarray(previous.get(field), dtype=float)
            after = np.asarray(current.get(field), dtype=float)
            if (before.shape != shape or after.shape != shape or not np.isfinite(before).all()
                    or not np.isfinite(after).all() or not np.allclose(before, after, atol=tolerance, rtol=0)):
                return arm + '_' + field + '_changed_or_missing'
    return None


def stationary_capture_reference(previous, current, *, stopped=None):
    """Authorize comparing fresh surfaces after unchanged feedback or one acknowledged stop."""
    if (not previous or not current.get('observation_id')
            or previous.get('observation_id') == current['observation_id']
            or current.get('time', 0.) <= previous.get('time', 0.)
            or any(not previous.get(field) or previous[field] != current.get(field)
                   for field in ('session_id', 'bridge_instance_id', 'geometry_revision'))):
        return None
    comparable = previous
    source = 'unchanged_command_revision'
    if stopped is not None:
        if (stopped.get('ok') is not True or stopped.get('robot_identity_changed') is not False
                or stopped.get('previous_generation') != previous.get('generation')
                or stopped.get('generation') != current.get('generation')
                or type(previous.get('command_revision')) is not int
                or current.get('command_revision') != previous['command_revision'] + 1):
            return None
        comparable = {**previous, 'generation': current['generation'], 'command_revision': current['command_revision']}
        source = 'stationary_acknowledged_stop'
    if capture_reuse_rejection(comparable, current, epoch_matches=True, now=time.monotonic(), max_age_s=None):
        return None
    return {'source': source, 'reference_observation_id': previous['observation_id'],
            'reference_generation': previous['generation'], 'observation_id': current['observation_id'],
            'generation': current['generation']}
