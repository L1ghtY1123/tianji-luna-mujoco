"""GraspGen tool data contract; no runtime or transport dependency."""
from typing import TypedDict

class GraspCandidate(TypedDict):
    """T_camera_grasp maps grasp-frame points into the input camera frame."""
    index: int  # original concatenated official index, zero-based
    T_camera_grasp: list[list[float]]  # rigid 4x4, translation in metres
    score: float  # unchanged official confidence

class InvalidRequest(ValueError):
    pass


class InvalidOutput(RuntimeError):
    pass


class NoCandidates(RuntimeError):
    pass


class ModelIdentityMismatch(RuntimeError):
    pass
