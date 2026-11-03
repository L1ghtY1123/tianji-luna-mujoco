"""Executable requirements live in contracts; instructions come from SKILL.md."""
import json
import re
from pathlib import Path
from typing import Literal

from jsonschema import Draft202012Validator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

ROOT = Path(__file__).resolve().parents[2]


def validate_result_bindings(value):
    protected = {'$target', '$source', '$destination', '$work_area'}
    if (not isinstance(value, dict) or any(
            not isinstance(name, str) or not re.fullmatch(r'\$[A-Za-z][A-Za-z0-9_]{0,63}', name)
            or name in protected or not isinstance(field, str) or not field.strip()
            for name, field in value.items())):
        raise ValueError('invalid_skill_result_bindings')
    return value


class OutcomeRule(BaseModel):
    """Possible returned outcomes under facts known before a skill invocation."""
    model_config = ConfigDict(extra='forbid')
    when: dict[str, StrictBool] = Field(min_length=1)
    statuses: list[Literal['success', 'failure', 'unknown']] = Field(min_length=1)


class ArgumentRule(BaseModel):
    """Literal argument conditions add requirements and successful fact effects."""
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    when_arguments: dict[str, StrictStr | StrictBool | StrictInt | StrictFloat] = Field(min_length=1)
    requires: dict[str, StrictBool]
    effects: dict[str, StrictBool | None]

    @model_validator(mode='after')
    def changes_contract(self):
        if not self.requires and not self.effects:
            raise ValueError('argument_rule_requires_a_requirement_or_effect')
        return self


def _same_argument(left, right):
    return (isinstance(left, bool) == isinstance(right, bool)) and left == right


def same_result_arguments(original, recovered, ignored=()):
    """Compare result semantics, excluding declared acquisition-only inputs."""
    if not isinstance(original, dict) or not isinstance(recovered, dict):
        return False
    return ({key: value for key, value in original.items() if key not in ignored}
            == {key: value for key, value in recovered.items() if key not in ignored})


def validate_argument_rules(contract):
    """Reject undeclared conditions and overlapping contradictory rules."""
    schema = contract.get('arguments', contract.get('argument_schema', {}))
    properties = schema.get('properties', {})
    rules = [ArgumentRule.model_validate(value).model_dump() for value in contract.get('argument_rules', [])]
    for index, rule in enumerate(rules):
        for key, value in rule['when_arguments'].items():
            if key not in properties or not Draft202012Validator(
                    {'$defs': schema.get('$defs', {}), **properties[key]}).is_valid(value):
                raise ValueError('invalid_argument_rule:' + key)
        if any(key in contract.get('requires', {}) and contract['requires'][key] != value
               for key, value in rule['requires'].items()):
            raise ValueError('conflicting_argument_rule_requirement')
        for previous in rules[:index]:
            overlap = all(key not in previous['when_arguments'] or _same_argument(value, previous['when_arguments'][key])
                          for key, value in rule['when_arguments'].items())
            if overlap and any(key in previous[field] and previous[field][key] != value
                               for field in ('requires', 'effects') for key, value in rule[field].items()):
                raise ValueError('conflicting_argument_rules')
    return rules


def argument_contract(contract, arguments):
    """Resolve defaults without rewriting the supplied graph or its arguments."""
    schema = contract.get('arguments', contract.get('argument_schema', {}))
    values = {key: value['default'] for key, value in schema.get('properties', {}).items() if 'default' in value}
    values.update(arguments)
    requires, effects = dict(contract.get('requires', {})), dict(contract.get('planning_effects', {}))
    for rule in validate_argument_rules(contract):
        if all(key in values and _same_argument(values[key], value) for key, value in rule['when_arguments'].items()):
            requires.update(rule['requires'])
            effects.update(rule['effects'])
    return {**contract, 'requires': requires, 'planning_effects': effects}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str
    capability: str = ''
    invalidates_observation: bool = True
    recovery_retry_safe: bool = False
    recovery_result_reusable: bool = False
    recovery_result_ignored_arguments: list[StrictStr] = Field(default_factory=list)
    processing_effect: bool = False
    inputs: list[str] = Field(default_factory=list)
    requires: dict[str, bool]
    provides: list[str]
    outputs: list[str] = Field(default_factory=list)
    result_bindings: dict[str, str] = Field(default_factory=dict)
    planning_effects: dict[str, StrictBool | None] = Field(default_factory=dict)
    argument_rules: list[ArgumentRule] = Field(default_factory=list)
    outcome_rules: list[OutcomeRule] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def reusable_result_is_nonmutating(self):
        if self.recovery_result_reusable and (self.invalidates_observation or not self.recovery_retry_safe):
            raise ValueError('recovery_result_reuse_requires_repeatable_sensing')
        if self.recovery_result_ignored_arguments and not self.recovery_result_reusable:
            raise ValueError('result_argument_equivalence_requires_reusable_result')
        return self

    @field_validator('result_bindings')
    @classmethod
    def valid_result_bindings(cls, value):
        return validate_result_bindings(value)


class NoArguments(BaseModel):
    """Explicit schema for a skill whose inputs come from the bound context."""
    model_config = ConfigDict(extra='forbid', strict=True)


class SkillSelectionError(ValueError):
    def __init__(self, code, skill, unmet_requirements=None):
        super().__init__(code + ":" + skill)
        self.code = code
        self.details = {"selected_skill": skill, "unmet_requirements": unmet_requirements or {}}


class Registry:
    def __init__(self, root=ROOT / "skill_catalog"):
        self.contracts = {}
        self.documents = {}
        self.argument_models = {}
        self.handlers = {}
        for path in sorted(Path(root).glob("*/contract.json")):
            contract = Contract.model_validate(json.loads(path.read_text(encoding="utf-8")))
            if contract.id in self.contracts or path.parent.name != contract.id:
                raise ValueError("invalid_skill_identity")
            self.contracts[contract.id] = contract
            self.documents[contract.id] = (path.parent / "SKILL.md").read_text(encoding="utf-8")
        if not self.contracts:
            raise ValueError("no_skill_contracts")

    def eligible(self, facts):
        return {name: contract for name, contract in self.contracts.items()
                if all(bool(facts.get(key, False)) == value for key, value in contract.requires.items())}

    def describe(self):
        """The model gets original contracts, without computed eligibility hints."""
        result = {}
        for name, contract in self.contracts.items():
            if self.handlers and name not in self.handlers:
                continue
            description = contract.model_dump()
            description['capability'] = contract.capability or contract.description
            description['outputs'] = contract.outputs or contract.provides
            description['arguments'] = self.argument_models.get(name, NoArguments).model_json_schema()
            result[name] = description
        return result

    def register(self, name, argument_model, handler):
        """Bind declared implementations at the composition root; never import by name."""
        if name not in self.contracts:
            raise SkillSelectionError('unknown_skill', name)
        if name in self.handlers:
            raise ValueError('duplicate_skill_handler:' + name)
        if not isinstance(argument_model, type) or not issubclass(argument_model, BaseModel):
            raise TypeError('skill_arguments_require_pydantic_model')
        if not callable(handler):
            raise TypeError('skill_handler_must_be_callable')
        ignored = self.contracts[name].recovery_result_ignored_arguments
        if len(ignored) != len(set(ignored)) or not set(ignored) <= set(argument_model.model_fields):
            raise ValueError('invalid_recovery_result_ignored_arguments')
        validate_argument_rules(self.contracts[name].model_dump() | {'arguments': argument_model.model_json_schema()})
        self.argument_models[name] = argument_model
        self.handlers[name] = handler

    def validate_arguments(self, name, arguments):
        if name not in self.contracts:
            raise SkillSelectionError('unknown_skill', name)
        if name not in self.argument_models:
            raise SkillSelectionError('skill_handler_not_registered', name)
        try:
            return self.argument_models[name].model_validate(arguments, strict=True)
        except ValidationError as error:
            failure = SkillSelectionError('invalid_skill_arguments', name)
            failure.details['validation_errors'] = error.errors(include_url=False)
            raise failure from error

    async def execute(self, name, facts, context, arguments):
        """Execute exactly one selected skill; no fallback, retries or next-step policy."""
        self.validate_selection(name, facts)
        validated = self.validate_arguments(name, arguments)
        self.validate_selection(name, facts, validated.model_dump())
        return await self.handlers[name](context, validated)

    def validate_selection(self, skill, facts, arguments=None):
        """Validate the next program skill without selecting a replacement."""
        if skill not in self.contracts:
            raise SkillSelectionError("unknown_skill", skill)
        unmet = {}
        contract = self.contracts[skill].model_dump()
        if arguments is not None:
            contract = argument_contract(contract | {
                'arguments': self.argument_models.get(skill, NoArguments).model_json_schema()}, arguments)
        for fact, expected in contract['requires'].items():
            actual = bool(facts.get(fact, False))
            if actual != expected:
                unmet[fact] = {"expected": expected, "actual": actual}
        if unmet:
            raise SkillSelectionError("skill_contract_not_satisfied", skill, unmet)
