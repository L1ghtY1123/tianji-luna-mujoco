"""Model-selected grasp preferences, frozen before sampling and planning."""
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..planning_settings import PlanningSettings


class GraspParameters(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    grasp_insertion_m: float | None = Field(default=None, ge=-.02, le=.03,
        description='One local +Z offset before IK, in metres. Zero uses the calibrated model pose.')
    grasp_maximum_tilt_deg: float | None = Field(default=None, ge=0., le=180.,
        description='Maximum deviation from world-down. 180 disables this preference; IK/collision still apply.')
    grasp_require_away_from_robot: bool | None = Field(default=None,
        description='Optional grasp direction filter, not a collision or reachability guarantee.')


class GraspArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    parameters: GraspParameters = Field(default_factory=GraspParameters,
        description='Optional Responses-selected preferences for this grasp. Omitted fields use UI defaults.')
    reason: str = Field(default='', max_length=1000)

    @model_validator(mode='after')
    def explained(self):
        if self.parameters.model_dump(exclude_none=True) and not self.reason.strip():
            raise ValueError('model_grasp_parameters_require_reason')
        return self


def apply_grasp_parameters(context, arguments):
    overrides = arguments.parameters.model_dump(exclude_none=True)
    base = context.setdefault('configured_planning_settings', deepcopy(context['planning_settings']))
    selected = PlanningSettings.model_validate({**base, **overrides}).model_dump()
    context['planning_settings'] = selected
    return {'source': 'responses_skill_arguments' if overrides else 'configured_defaults',
        'overrides': overrides, 'reason': arguments.reason,
        'effective_parameters': {key: selected[key] for key in GraspParameters.model_fields},
        'scope': 'current_grasp_sampling_planning_execution_and_transfer'}
