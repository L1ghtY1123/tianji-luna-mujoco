"""Application-owned immutable observation data and completeness errors."""
import copy
import numpy as np

CAMERA_ROLES = {"1": "right_wrist", "2": "left_wrist", "3": "head"}


class RGBDCaptureIncomplete(ValueError):
    def __init__(self, errors, *, readiness=None, synchronization=None):
        self.errors = copy.deepcopy(errors)
        self.details = {'camera_errors': self.errors, 'readiness': copy.deepcopy(readiness or {}),
                        'synchronization': copy.deepcopy(synchronization or {})}
        self.codes = tuple(str(error.get("error", "unknown")) for error in self.errors)
        super().__init__("rgbd_capture_incomplete:" + ",".join(self.codes))


class RGBDCaptureTimeout(TimeoutError):
    def __init__(self, last_error, attempts):
        super().__init__('rgbd_capture_timeout:' + str(last_error))
        self.details = {**last_error.details, 'attempts': attempts, 'message': str(self)}


class RobotFeedbackUnavailable(ValueError):
    def __init__(self, generation, snapshot, read_error=None):
        super().__init__('robot_feedback_unavailable')
        self.details = {key: snapshot.get(key) for key in
                        ('connected', 'feedback_id', 'session_id', 'available', 'error', 'readiness_errors')}
        self.details.update(generation=generation, telemetry_refresh_attempted=True)
        if read_error is not None:
            self.details['refresh_error'] = type(read_error).__name__ + ': ' + str(read_error)[:240]


def immutable(array):
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


def detached_capture(capture):
    """Give each consumer its own containers and read-only sensor arrays."""
    result = copy.deepcopy(capture)
    if result is not None:
        for view in result.get('views', {}).values():
            for name in ('rgb', 'depth_m', 'valid_depth'):
                if name in view:
                    view[name].setflags(write=False)
    return result
