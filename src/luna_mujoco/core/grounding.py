"""Task semantics and observation-local evidence, independent of storage and control."""
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from .graph import StrictModel

COLOR_CATEGORIES = ('black', 'blue', 'pink', 'red', 'green', 'yellow', 'orange', 'white', 'gray', 'brown')


class Relation(StrictModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    op: Literal['true', 'and', 'or', 'not', 'robot_left', 'robot_right',
                'left_of', 'right_of', 'in_front_of', 'behind', 'above', 'below',
                'inside', 'on', 'history'] = Field(description=(
        'robot_left/right mean the corresponding robot half-space (+Y/-Y). '
        'Use these for left-side/right-side descriptions, independent of image position. '
        'left_of/right_of compare against an explicitly named reference entity.'))
    ref: str | None = None
    value: str | None = None
    terms: tuple['Relation', ...] = ()

    @model_validator(mode='after')
    def valid_operands(self):
        if self.op in {'and', 'or', 'not'}:
            if not self.terms or self.op == 'not' and len(self.terms) != 1:
                raise ValueError('invalid_relation_operands')
        elif self.terms:
            raise ValueError('unexpected_relation_operands')
        relative = {'left_of', 'right_of', 'in_front_of', 'behind', 'above', 'below', 'inside', 'on'}
        if (self.op in relative) != bool(self.ref):
            raise ValueError('relation_reference_required_or_unexpected')
        if (self.op == 'history') != bool(self.value):
            raise ValueError('history_condition_required_or_unexpected')
        return self


class SelectionRule(StrictModel):
    """Immutable robot-frame selection semantics, separate from identity.

    Distinguish operated targets, destinations, sources and supporting references.
    Compile relations and ordinal/count requirements explicitly. For all/count
    inputs propose a collection selector, not one selector per visible member.
    """
    model_config = ConfigDict(extra='forbid', frozen=True)
    role: Literal['target', 'source', 'destination', 'support', 'reference']
    predicate: Relation
    quantifier: Literal['one', 'all', 'count', 'ordinal'] = Field(description=(
        'one for one specifically described subject, including a robot-side subject. '
        'ordinal only for an explicitly ranked subject (second, leftmost, rightmost), '
        'not for left-side/right-side. all/count for collection inputs.'))
    quantity: int | None = Field(default=None, ge=1, le=1024)
    axis: Literal['x', 'y', 'z'] = 'y'
    order: Literal['ascending', 'descending'] = Field(default='descending', description=(
        'Robot frame only. Left-to-right is decreasing Y; front-to-back decreasing X. '
        'This ranks the comparison set, independently of later action order.'))
    binding_mode: Literal['once', 'each_selection'] = Field(description=(
        'once freezes a specifically described physical identity after RGBD grounding. '
        'each_selection is only for collection inputs; never switches an active grasp/place target.'))

    @model_validator(mode='after')
    def valid_quantifier(self):
        if (self.quantifier in {'count', 'ordinal'}) != (self.quantity is not None):
            raise ValueError('selection_quantity_conflict')
        if self.binding_mode == 'each_selection' and self.role != 'target':
            raise ValueError('only_collection_targets_can_reselect')
        if self.quantifier in {'all', 'count'} and self.binding_mode != 'each_selection':
            raise ValueError('collection_requires_each_selection')
        if self.quantifier in {'one', 'ordinal'} and self.binding_mode != 'once':
            raise ValueError('individual_selection_requires_once_binding')
        return self


def relation_refs(relation):
    result = {relation.ref} if relation.ref else set()
    for term in relation.terms:
        result.update(relation_refs(term))
    return result


@dataclass(frozen=True)
class CandidateObservation:
    observation_id: str
    generation: int
    candidates: tuple[dict, ...]
    query_evidence: tuple[dict, ...] = ()
    proposal_overlap_evidence: tuple[dict, ...] = ()
    mask_pair_evidence: tuple[dict, ...] = ()
    coverage: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CurrentGroup:
    group_id: str
    observation_id: str
    candidate_indices: tuple[int, ...]
    geometry_evidence: dict


@dataclass(frozen=True)
class HistoryMatch:
    group_id: str
    outcome: Literal['existing', 'new', 'unknown', 'reject']
    entity_id: str | None
    evidence: dict


def identity_provenance(match):
    """Program-derived identity never fabricates a Decisions request or answer."""
    records = match.evidence.get('records', [])
    source = match.evidence.get('source')
    if source in {'confirmed_same_capture_masks', 'empty_registry_and_confirmed_current_group'}:
        return {'source': source, 'request_hash': None}
    requests = [record['request_hash'] for record in records if record.get('request_hash')
                and record.get('identity_stage') in {'joint_history_and_novelty', 'joint_current_conflict_review'}]
    return {'source': 'openai_decisions_rgbd_grounding' if requests else match.evidence.get('source', 'unknown'),
            'request_hash': requests[-1] if requests else None}


@dataclass(frozen=True)
class GroundingResult:
    status: Literal['success', 'unknown', 'failure']
    observation: CandidateObservation
    groups: tuple[CurrentGroup, ...]
    selected_groups_by_ref: dict[str, tuple[str, ...]] = field(default_factory=dict)
    history_matches: tuple[HistoryMatch, ...] = ()
    missing_evidence: tuple[dict, ...] = ()
    evidence: dict = field(default_factory=dict)
