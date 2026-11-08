"""Immutable identity and provenance shared by planning and execution."""
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MotionPlanBinding(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    target_id: str = Field(min_length=1)
    destination_id: str | None
    arm: Literal['A', 'B']
    generation: int
    observation_id: str = Field(min_length=1)
    scene_token: str = Field(min_length=1)
    settings_sha256: str = Field(pattern='^[a-f0-9]{64}$')

    @classmethod
    def capture(cls, *, target_id, destination_id, arm, observation, scene, planning_settings):
        if (scene.get('target_id') != target_id
                or scene.get('generation') != observation.get('generation')
                or scene.get('observation_id') != observation.get('observation_id')):
            raise ValueError('motion_plan_input_binding_mismatch')
        encoded = json.dumps(planning_settings, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return cls(target_id=target_id, destination_id=destination_id, arm=arm,
            generation=observation['generation'], observation_id=observation['observation_id'],
            scene_token=scene['scene_token'], settings_sha256=hashlib.sha256(encoded.encode()).hexdigest())

    def require_current(self, current):
        if self != current:
            raise ValueError('motion_plan_binding_changed')
