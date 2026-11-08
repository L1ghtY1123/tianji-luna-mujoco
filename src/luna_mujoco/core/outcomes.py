"""Graph-facing skill results contain control outputs and concise evidence."""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class GraphSkillResult(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    status: Literal['success', 'failure', 'unknown', 'empty']
    outputs: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode='after')
    def separate_control_fields(self):
        if {'status', 'success'} & self.outputs.keys():
            raise ValueError('skill_output_overwrites_control_status')
        return self

    def payload(self):
        return {**deepcopy(self.evidence), **deepcopy(self.outputs),
                'status': self.status, 'success': self.status == 'success'}
