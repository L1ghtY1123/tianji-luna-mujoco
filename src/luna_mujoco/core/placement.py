"""Placement semantics contain no coordinates invented by a language model."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class ReleaseLedgerSnapshot(BaseModel):
    """Only the current transfer's ledger evidence, without a mutable ledger service."""
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    target_id: str
    current_id: str | None
    completed_ids: tuple[str, ...]
    record: dict[str, JsonValue]
    bound_arm: Literal['A', 'B'] | None
    placement_request: dict[str, JsonValue]
    visual_progress: dict[str, JsonValue]
    transfer_index: int = Field(ge=0)


class PlacementIntent(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    relation: Literal['on', 'inside']
    release_mode: Literal['surface_setdown', 'container_release'] = 'surface_setdown'
    relative: Literal['none', 'left_of', 'right_of', 'in_front_of', 'behind', 'between'] = 'none'
    zone: Literal['whole', 'left_half', 'right_half', 'front_half', 'back_half', 'center'] = 'whole'
    reference_ids: tuple[str, ...] = ()
    frame: Literal['robot_base', 'camera_1', 'camera_2', 'camera_3'] | None = None
    preference: Literal['clearance', 'reach'] = 'clearance'

    @model_validator(mode='after')
    def coherent_relations(self):
        if self.release_mode == 'container_release' and self.relation != 'inside':
            raise ValueError('container_release_requires_inside_relation')
        count = 0 if self.relative == 'none' else 2 if self.relative == 'between' else 1
        if len(self.reference_ids) != count or len(set(self.reference_ids)) != count:
            raise ValueError('placement_reference_count_mismatch')
        directional = self.relative not in {'none', 'between'} or self.zone not in {'whole', 'center'}
        if directional and self.frame is None:
            raise ValueError('placement_direction_requires_frame')
        return self


class ResolvePlacementArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    preference: Literal['decide', 'clearance', 'reach'] = Field(default='decide', description=(
        'decide keeps the bound destination ranking preference (clearance by default). Explicit ranking changes only '
        'candidate order, never the bound spatial relationship, grasp rotation or exact configured XY.'))
    capture_mode: Literal['reuse_current', 'fresh'] = Field(default='reuse_current', description=(
        'For candidate-level language checks, fresh explicitly acquires current three-camera RGBD without '
        'segmentation, CAD registration, scene reconstruction or movement. Geometry remains bound to the '
        'verified planning scene. Choose fresh after grasp motion invalidates the earlier camera capture. '
        'Entire-region permission needs no candidate images.'))
