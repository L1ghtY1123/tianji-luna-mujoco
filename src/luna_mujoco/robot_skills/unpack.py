"""The simulation VLA substitute changes physics, never perception or task success."""
from typing import Protocol

from pydantic import BaseModel, ConfigDict, FiniteFloat


class UnpackOutputPose(BaseModel):
    """Explicit simulator diagnostic pose; production effects sample their own position."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    position: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    yaw_deg: FiniteFloat


class UnpackSimulation(Protocol):
    def unpack_at_pose(self, work_area, output_pose=None, task_id=None, *, generation,
                       target_id=None, owner_id=None): ...


class VLAUnpack:
    """A single synchronous effect; the caller owns pose preparation and verification."""

    def __init__(self, simulation: UnpackSimulation, output_pose: UnpackOutputPose | None = None):
        self.simulation = simulation
        self.output_pose = output_pose

    def execute(self, work_area, *, generation, target_id, owner_id, task_id=None):
        receipt = self.simulation.unpack_at_pose(
            work_area, output_pose=None if self.output_pose is None else self.output_pose.model_dump(mode='json'),
            generation=generation, target_id=target_id, owner_id=owner_id, task_id=task_id)
        # Simulation rendering owns the physical pose. Task/perception receives
        # only action provenance and must locate the product from fresh RGBD.
        evidence_fields = {'kind', 'source', 'provenance', 'is_visual_evidence', 'target_id', 'owner_id',
                           'task_id', 'action_id', 'generation', 'completed', 'completed_time',
                           'completed_perf_counter_ns', 'completed_monotonic_ns',
                           'before_manifest_revision', 'manifest_revision'}
        return {key: value for key, value in receipt.items() if key in evidence_fields}
