"""Data accepted by object and spatial memory, independent of graph execution."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

ObjectId = Annotated[str, StringConstraints(pattern=r'^[1-9][0-9]*$')]
Vector3 = tuple[float, float, float]


class MemoryData(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)


class PositionMeasurement(MemoryData):
    observation_id: str = Field(min_length=1)
    position_world_m: Vector3
    source: str = Field(min_length=1)


class ObjectRetirement(MemoryData):
    entity_id: ObjectId
    reason: Literal['consumed', 'removed']
    observation_id: str = Field(min_length=1)
    source: Literal['openai_decisions', 'operator']
    request_hash: str = Field(min_length=1)


class ObjectMotionReceipt(MemoryData):
    entity_id: ObjectId
    # holding_confirmed remains readable in historical receipts, not emitted by current execution.
    kind: Literal['lift_started', 'gripper_load_feedback', 'release_opened', 'release_commanded', 'holding_confirmed']
    arm: Literal['A', 'B']
    generation: int
    feedback_time_s: float
    source: Literal['robot_feedback'] = 'robot_feedback'
    evidence: dict[str, JsonValue]


class ObjectRecord(MemoryData):
    """Physical instance memory. Task membership and current collision validity are separate."""
    id: ObjectId
    kind: Literal['table', 'bin', 'marker', 'cube', 'sphere', 'cylinder', 'object']
    color: str = 'unknown'
    description: str = ''
    status: Literal['active', 'consumed', 'removed'] = 'active'
    retirement: ObjectRetirement | None = None
    world_key: str | None = None
    roles: tuple[str, ...] = ()
    appearance_keywords: tuple[str, ...] = ()
    segmentation_query: dict[str, JsonValue] | None = None
    initial_parent_id: ObjectId | None = None
    reference_camera_id: Literal['1', '2', '3'] | None = None
    reference_observation_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    identity_source: str | None = None
    identity_request_hash: str | None = None
    current_identity_proposal: dict[str, JsonValue] | None = None
    asset_id: str | None = None
    asset_source: str | None = None
    geometry_policy: dict[str, JsonValue] | None = None
    geometry_policy_instruction: str | None = None
    visibility: Literal['visible', 'unseen', 'unknown', 'occluded'] = 'unknown'
    visual_visibility: str | None = None
    observation_id: str | None = None
    last_position_measurement: PositionMeasurement | None = None
    geometry_reference: dict[str, JsonValue] | None = None
    observed_support: dict[str, JsonValue] | None = None
    current_region: str | None = None
    confirmed_observation_id: str | None = None
    transport_status: str | None = None
    last_motion_receipt: ObjectMotionReceipt | None = None
    visual_release: dict[str, JsonValue] | None = None
    container_entry: dict[str, JsonValue] | None = None
    transfers: tuple[dict[str, JsonValue], ...] = ()
    processing_history: tuple[str, ...] = ()
    input_entity_id: ObjectId | None = None
    output_entity_id: ObjectId | None = None
    processing_skill: str | None = None
    process_id: str | None = None
    creation_evidence: dict[str, JsonValue] | None = None

    @model_validator(mode='after')
    def valid_reference(self):
        if self.retirement is not None and self.retirement.entity_id != self.id:
            raise ValueError('retirement_instance_mismatch')
        if self.last_motion_receipt is not None and self.last_motion_receipt.entity_id != self.id:
            raise ValueError('motion_receipt_instance_mismatch')
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                raise ValueError('invalid_object_reference_box')
        return self


class ObjectObservation(MemoryData):
    observation_id: str = Field(min_length=1)
    entities: tuple[ObjectRecord, ...]
    visible_ids: tuple[ObjectId, ...]
    positions: dict[ObjectId, PositionMeasurement] = Field(default_factory=dict)

    @model_validator(mode='after')
    def coherent_observation(self):
        identifiers = [entity.id for entity in self.entities]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('duplicate_observed_instance')
        if not set(self.visible_ids) <= set(identifiers) or not set(self.positions) <= set(self.visible_ids):
            raise ValueError('measurement_without_observed_identity')
        if any(position.observation_id != self.observation_id for position in self.positions.values()):
            raise ValueError('measurement_observation_mismatch')
        return self


class ObjectOperation(MemoryData):
    """A dated physical receipt. Current task status cannot be submitted here."""
    entity_id: ObjectId
    event_key: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    observation_id: str | None = None
    current_region: str | None = None
    confirmed_observation_id: str | None = None
    transport_status: str | None = None
    last_motion_receipt: ObjectMotionReceipt | None = None
    visual_release: dict[str, JsonValue] | None = None
    container_entry: dict[str, JsonValue] | None = None
    transfers: tuple[dict[str, JsonValue], ...] = ()
    observed_support: dict[str, JsonValue] | None = None
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class ProductRegistration(MemoryData):
    """Confirmed product facts submitted by execution to the permanent-ID registry."""
    event_key: str = Field(min_length=1)
    attributes: dict[str, JsonValue]


class CollisionUpdate(MemoryData):
    requested_scope: Literal['global']
    strategy_source: str = Field(min_length=1)
    scope_reason: str = Field(min_length=1)

class CollisionLifecycle(MemoryData):
    required_entity_ids: tuple[ObjectId, ...] = ()
    invalidated_entity_ids: tuple[ObjectId, ...] = ()
    retired_entity_ids: tuple[ObjectId, ...] = ()
    requested_scope: Literal['global'] | None = None
    strategy_source: str | None = None
    scope_reason: str | None = None
    workspace_bounds: tuple[Vector3, Vector3] | None = None
    max_retained_voxel_bytes: int | None = Field(default=None, gt=0)


class CollisionSnapshot(MemoryData):
    """Detached collision evidence for a consumer; no memory service or raw RGBD."""
    scene: dict[str, JsonValue]
    observation: dict[str, JsonValue]
    refresh_required: bool


class GeometryMeasurement(MemoryData):
    observation_id: str = Field(min_length=1)
    geometry_status: Literal['measured', 'cached', 'missing']
    id: ObjectId | None = None
    kind: str | None = None
    color: str | None = None
    description: str | None = None
    position: Vector3 | None = None
    measurement_source: str | None = None
    measurement_scope: str | None = None
    surface_height_m: float | None = None
    inlier_count: int | None = Field(default=None, gt=0)
    camera_ids: tuple[Literal['1', '2', '3'], ...] = ()


class GeometryObservation(MemoryData):
    observation_id: str = Field(min_length=1)
    measurements: dict[str, GeometryMeasurement]

    @model_validator(mode='after')
    def same_capture(self):
        if any(row.observation_id != self.observation_id for row in self.measurements.values()):
            raise ValueError('geometry_measurement_capture_mismatch')
        if any(row.id is not None and row.id != identifier for identifier, row in self.measurements.items()):
            raise ValueError('geometry_measurement_instance_mismatch')
        return self
