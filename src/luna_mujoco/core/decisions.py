"""Typed single-question Decisions boundary, independent of API and execution."""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def predicate_status(probability):
    """Choose the more probable Boolean outcome; equal probabilities abstain."""
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
        return 'unknown'
    if probability > .5:
        return 'yes'
    if probability < .5:
        return 'no'
    return 'unknown'


def choice_value(answer, choices=None):
    """Select the probability maximum without a confidence acceptance threshold.

    Older valid wire answers may contain only the service-selected choice.
    A supplied distribution must cover the allowed options exactly; ties abstain.
    The caller retains the original answer for audit.
    """
    if answer.get('type') == 'refusal':
        return 'unknown'
    value = answer.get('choice')
    allowed = set(choices) if choices is not None else None
    if answer.get('type') != 'choice' or not isinstance(value, str) or (allowed is not None and value not in allowed):
        raise ValueError('invalid_answer_choice')
    confidence = answer.get('confidence')
    if confidence is not None and (type(confidence) not in (int, float)
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError('invalid_answer_confidence')
    if 'probabilities' not in answer:
        return value
    rows = answer['probabilities']
    if not isinstance(rows, list) or not rows:
        raise ValueError('invalid_choice_probabilities')
    probabilities = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid_choice_probabilities')  # noqa: TRY004 - wire validation error
        option, probability = row.get('value'), row.get('probability')
        if (not isinstance(option, str) or option in probabilities
                or type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            raise ValueError('invalid_choice_probabilities')
        probabilities[option] = probability
    if ((allowed is not None and set(probabilities) != allowed) or value not in probabilities
            or not math.isclose(sum(probabilities.values()), 1., abs_tol=.001)):
        raise ValueError('incomplete_choice_distribution')
    maximum = max(probabilities.values())
    winners = [option for option, probability in probabilities.items() if probability == maximum]
    return winners[0] if len(winners) == 1 else 'unknown'


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
    choices: tuple[str, ...] = Field(min_length=2, max_length=255)
    descriptions: dict[str, str] = Field(default_factory=dict)
    selection_mode: Literal['evidence', 'proposal'] = 'evidence'

    @model_validator(mode='after')
    def distinct_choices(self):
        if len(set(self.choices)) != len(self.choices) or 'unknown' not in self.choices:
            raise ValueError('choices_require_distinct_values_and_unknown')
        if set(self.descriptions) - set(self.choices) or any(not text.strip() for text in self.descriptions.values()):
            raise ValueError('choice_description_requires_declared_option')
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
    rule: Literal['visual', 'observation_recovery', 'failure_return'] = 'visual'
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
            if not 2 <= len(options) <= 255 or len(set(options)) != len(options) or 'unknown' not in options:
                raise ValueError('choice_requires_distinct_options_and_unknown')
        return self
