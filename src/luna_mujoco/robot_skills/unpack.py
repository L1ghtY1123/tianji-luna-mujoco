"""Invoke Base's VLA capability; a receipt never establishes visual success."""
from typing import Protocol

class UnpackRobot(Protocol):
    def unpack_at_pose(self, work_area, task_id=None, *, generation,
                       target_id=None, owner_id=None): ...


class VLAUnpack:
    """A single synchronous effect; the caller owns pose preparation and verification."""

    def __init__(self, simulation: UnpackRobot):
        self.simulation = simulation

    def execute(self, work_area, *, generation, target_id, owner_id, task_id=None):
        receipt = self.simulation.unpack_at_pose(
            work_area,
            generation=generation, target_id=target_id, owner_id=owner_id, task_id=task_id)
        # Locate the product from fresh RGBD; Base's receipt supplies no pose.
        evidence_fields = {'kind', 'source', 'provenance', 'is_visual_evidence', 'target_id', 'owner_id',
                           'task_id', 'action_id', 'generation', 'completed', 'completed_time',
                           'completed_perf_counter_ns', 'completed_monotonic_ns',
                           'before_manifest_revision', 'manifest_revision'}
        return {key: value for key, value in receipt.items() if key in evidence_fields}
