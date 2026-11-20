"""Execution failures shared by the controller and Base transport boundary."""
from copy import deepcopy

JOINT_COMMAND_PERIOD_S = .02
MAX_TRAJECTORY_TRACKING_ERROR_RAD = .03
CHECKED_START_TOLERANCE_DEG = .05


class StaleCommandError(RuntimeError):
    """Robot session, world or command epoch changed."""


class TrajectoryExecutionError(RuntimeError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = deepcopy(details)


class CheckedStartChanged(TrajectoryExecutionError):
    code = 'trajectory_start_changed_after_collision_check'


class GripperLimitError(ValueError):
    code = 'gripper_position_outside_authored_limits'

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = deepcopy(details or {})


class GripperStabilityError(TimeoutError):
    code = 'gripper_stability_timeout'

    def __init__(self, message, details):
        super().__init__(message)
        self.details = deepcopy(details)
