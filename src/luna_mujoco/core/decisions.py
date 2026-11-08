"""Typed single-question Decisions boundary, independent of API and execution."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class JudgmentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    name: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,63}$')
    instructions: str = Field(min_length=1)

    @model_validator(mode='after')
    def stated_question(self):
        if not self.instructions.strip():
            raise ValueError('decision_question_is_blank')
        return self


class ChoiceRequest(JudgmentRequest):
    choices: tuple[str, ...] = Field(min_length=2, max_length=64)
    selection_mode: Literal['evidence', 'proposal'] = 'evidence'

    @model_validator(mode='after')
    def distinct_choices(self):
        if len(set(self.choices)) != len(self.choices) or 'unknown' not in self.choices:
            raise ValueError('choices_require_distinct_values_and_unknown')
        return self


class DecisionQuery(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    observation_id: str = Field(min_length=1)
    generation: int | None = None
    question: ChoiceRequest | JudgmentRequest
    context: dict[str, JsonValue] = Field(default_factory=dict)


class GraphDecisionRequest(BaseModel):
    """A router's evidence snapshot; the receiver does not read executor state."""
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    condition: str = Field(min_length=1)
    bindings: dict[str, JsonValue]
    choices: tuple[str, ...] = ()
    branches: tuple[dict[str, JsonValue], ...] = ()
    last_result: dict[str, JsonValue] | None = None
    last_failure: dict[str, JsonValue] | None = None

    @model_validator(mode='after')
    def defined_choice(self):
        if self.choices:
            if len(set(self.choices)) != len(self.choices) or 'unknown' not in self.choices:
                raise ValueError('graph_choices_require_distinct_values_and_unknown')
        elif self.branches:
            raise ValueError('predicate_router_has_no_choice_branches')
        return self


class GraphDecisionResult(BaseModel):
    """Accepted judgment with its evidence; it cannot define a node or an edge."""
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    status: Literal['yes', 'no', 'unknown', 'proposed']
    value: str | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    selection_mode: Literal['proposal'] | None = None
    reason: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode='after')
    def stated_result(self):
        if self.status == 'proposed' and (not self.value or self.value == 'unknown'):
            raise ValueError('proposed_graph_branch_requires_value')
        if self.status in {'yes', 'no'} and self.value is not None:
            raise ValueError('predicate_result_has_no_choice_value')
        return self

    def payload(self):
        return self.model_dump(mode='json', exclude_none=True)


class DecisionOption(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    value: str = Field(min_length=1)
    description: str | None = None


class WireDecisionQuestion(BaseModel):
    """Validate every Decisions producer, including multimodal identity questions."""
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    type: Literal['predicate', 'choice']
    name: str = Field(min_length=1)
    instructions: str | list[dict[str, JsonValue]]
    choices: list[DecisionOption] | None = None

    @model_validator(mode='after')
    def defined_meaning(self):
        text = self.instructions if isinstance(self.instructions, str) else ' '.join(
            part.get('text', '') for part in self.instructions if part.get('type') == 'input_text')
        if not self.name.strip() or not text.strip():
            raise ValueError('decision_question_is_blank')
        if self.type == 'predicate' and self.choices is not None:
            raise ValueError('predicate_has_no_choices')
        if self.type == 'choice':
            options = [option.value for option in self.choices or []]
            if len(options) < 2 or len(set(options)) != len(options) or 'unknown' not in options:
                raise ValueError('choice_requires_distinct_options_and_unknown')
        return self
