"""Measured support-goal ranking, separate from model policy and path validation."""

from ..algorithms import AlgorithmError


def check_goal_workspace(lookup, arm, position):
    """Check the final gripper-front XYZ, after release height has been resolved."""
    rows = lookup(arm, [position])
    if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
            or rows[0].get('position_world_m') != position or type(rows[0].get('reachable')) is not bool):
        raise AlgorithmError('invalid place workspace result', code='algorithm_invalid_response')
    if not rows[0]['reachable']:
        raise AlgorithmError('actual place goal is outside the calibrated workspace',
            code='place_goal_workspace_rejected', details={'workspace': rows})
    return rows[0]
