"""Immutable task requirements and executable graph types; no model or robot calls."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TaskSpec(StrictModel):
    """Freeze the user's interpreted goal for this execution."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    instruction: str = Field(min_length=1)
    supported: bool
    reason: str
    scope: Literal['all', 'count', 'ids'] = Field(description=(
        'all selects every matching observed candidate; count selects quantity candidates; '
        'ids selects exactly target_refs. For ids, quantity may be null or the exact number of refs.'))
    target_selector: str = Field(min_length=1)
    target_refs: tuple[str, ...] = Field(description=(
        'Currently observed candidate refs for all/count, updated through sensing without freezing '
        'task membership. For ids, the exact requested instance refs, which must be nonempty.'))
    quantity: int | None = Field(ge=1, le=1024, description=(
        'Required for count, null for all. For ids, null or exactly len(target_refs); '
        'a consistent explicit count does not change the selected IDs.'))
    source_ref: str | None = Field(description=(
        'The original supporting surface, containing receptacle or configured source region of the '
        'selected INPUT objects. This is not the input object or its processing lineage. '
        'For an object taken from a box, reference the box; for an object taken from a table, '
        'reference the table. Never use a target itself as its own source. Null when the task '
        'does not specify or imply a shared source; current support is then established separately.'))
    destination_ref: str | None = Field(description=(
        'Final destination entity or configured region. Use the configured region ref when it is '
        'the requested final placement area; its supporting table remains a separate entity. '
        'In an object-owned subgraph this is the FINAL destination of that input and its processing '
        'lineage, including the confirmed output product. It is not the first drop-off point. '
        'A processing input area belongs in work_area_ref; destination_ref must agree with the '
        'required final product delivery in completion_conditions.'))
    work_area_ref: str | None = Field(default=None, description=(
        'Intermediate processing or staging area only. Null for direct pick-and-place with no '
        'intermediate operation; do not put the final placement area here.'))
    selection_order: Literal['inventory', 'nearest', 'explicit'] = Field(description=(
        'Use explicit by default: graph edges choose the object execution order. Use inventory or nearest '
        'only when the instruction explicitly requires that order. This field does not select an arm.'))
    processing: tuple[str, ...] = Field(description=(
        'Required object transformations only, declared processing_effect=true in the skill catalog. '
        'Ordinary observation, mapping, grasp, movement and placement are graph nodes, not processing requirements. '
        'Use an empty array for direct grasp-place without processing.'))
    completion_conditions: tuple[str, ...] = Field(description=(
        'Atomic task outcome requirements, one independently verifiable condition per item. '
        'Separate processing, final delivery and ordering; do not mix a historical milestone with '
        'current source emptiness. For all scope, verify_task_completion independently checks '
        'source_has_remaining_input, so do not repeat source closure in these conditions. '
        'Apply processing/delivery milestones to the selected inputs and their product lineages. '
        'Final robot posture is specified separately by completion_pose.'))
    completion_pose: Literal['observation', 'initial'] | None = Field(default=None, description=(
        'Requested final robot posture. initial means both arms at their measured task-start joints; '
        'observation means the configured observation pose. Honor explicit user or supplied '
        'completion_requirements; otherwise null. '
        'The runtime verifies actual joint feedback, never RGB judgment alone.'))
    constraints: tuple[str, ...]

    @model_validator(mode='after')
    def consistent_selection(self):
        if self.scope == 'count' and self.quantity is None:
            raise ValueError('task_count_scope_conflict: scope=count requires a positive quantity')
        if self.scope == 'all' and self.quantity is not None:
            raise ValueError('task_count_scope_conflict: scope=all requires quantity=null')
        if self.scope == 'ids' and self.quantity is not None and self.quantity != len(self.target_refs):
            raise ValueError('task_count_scope_conflict: scope=ids quantity must equal len(target_refs) or be null')
        if len(set(self.target_refs)) != len(self.target_refs):
            raise ValueError('duplicate_target_reference')
        if self.supported and not self.completion_conditions:
            raise ValueError('task_completion_required')
        if self.supported and self.scope == 'ids' and not self.target_refs:
            raise ValueError('ids_task_requires_target_refs')
        return self


class SkillCall(StrictModel):
    type: Literal['skill']
    id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')
    skill: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


Identifier = Annotated[str, Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]{0,95}$')]


class SubgraphNode(StrictModel):
    type: Literal['subgraph']
    id: Identifier
    subgraph_id: Identifier


class RouterNode(StrictModel):
    type: Literal['router']
    id: Identifier
    condition: str = Field(min_length=1, description=(
        'One atomic evidence question, or one choice among declared recovery branches. '
        'The no edge must be valid for the full negation of the question. '
        'Never combine recovery eligibility and selected arm in one condition: '
        'not (eligible AND arm A) does not mean arm B. Split independent questions '
        'into separate routers when both are necessary. Does not establish completion facts.'))
    choices: tuple[str, ...] = Field(default=(), max_length=64, description=(
        'For branch selection, list the exact outgoing edge.on values, including unknown. '
        'Explain each alternative in condition. Decisions selects one declared alternative. '
        'An empty array uses the factual yes/no/unknown predicate interface.'))

    @model_validator(mode='after')
    def declared_choices(self):
        if self.choices and (len(self.choices) < 2 or len(set(self.choices)) != len(self.choices)
                             or 'unknown' not in self.choices or any(not choice.strip() for choice in self.choices)):
            raise ValueError('router_choices_require_distinct_values_and_unknown')
        return self


class EndNode(StrictModel):
    type: Literal['end']
    id: Identifier
    status: Literal['success', 'failure', 'unknown'] = Field(description=(
        'Explicit return from this scope. Every subgraph requires an end reached by its final edge; '
        'the last skill never implicitly returns. Parent subgraph-call edges match this status. '
        'An owned subgraph success checks its verified delivery receipts; missing evidence '
        'returns unknown through an authored edge without replaying the subgraph.'))
    reason: str


Node = Annotated[SkillCall | SubgraphNode | RouterNode | EndNode,
                 Field(discriminator='type')]


class Edge(StrictModel):
    id: Identifier = Field(description=(
        'Unique label for this edge within its scope. This field names the edge, not its source node. '
        'Derive it from the exact source node ID and outcome, e.g. observe_input__success. '
        'Do not abbreviate different source IDs into the same edge-ID prefix.'))
    source: Identifier = Field(description='Exact ID of the node whose outcome triggers this edge, in this scope.nodes array.')
    target: Identifier = Field(description='Exact ID of the next node in this same scope.nodes array.')
    on: str = Field(min_length=1, description=(
        'Outcome of the SOURCE node, never the target. Skill and subgraph sources use success/failure/unknown; '
        'a skill declaring empty also emits empty when its current selection has no result. '
        'Predicate routers use yes/no/unknown. Choice routers use the exact values in their choices array. '
        'A success end may have only an unknown receipt-failure edge. '
        'Failure and unknown ends return immediately and have no outgoing edges, including self-loops.'))
    max_traversals: int = Field(default=1, ge=1, strict=True, description=(
        'Finite traversal budget for this edge within one scope invocation. '
        'Set the required retry bound explicitly on every edge participating in a loop. '
        'Exhaustion returns failure from this scope and the parent follows its declared failure edge.'))


class Subgraph(StrictModel):
    id: Identifier
    input_id: str | None = Field(description='Permanent inventory ID owning this subgraph, null for shared work.')
    task: TaskSpec | None
    entry: Identifier
    nodes: list[Node] = Field(min_length=1, max_length=512)
    edges: list[Edge] = Field(max_length=1536)

    @model_validator(mode='after')
    def owner_matches_task(self):
        if (self.input_id is None) != (self.task is None):
            raise ValueError('subgraph_task_owner_required')
        if self.task is not None and (self.task.scope != 'ids' or self.task.target_refs != (self.input_id,)
                                     or self.task.completion_pose is not None):
            raise ValueError('subgraph_task_owner_mismatch')
        return self


class Graph(StrictModel):
    """Executable task policy. Only edges determine control order."""
    version: Literal[2]
    id: Identifier
    entry: Identifier
    nodes: list[Node] = Field(min_length=1, max_length=512, description=(
        'Only nodes owned by the ROOT scope. This is NOT a global node index. '
        'Never copy subgraphs[i].nodes here. Include a subgraph CALL node with subgraph_id instead.'))
    edges: list[Edge] = Field(max_length=1536, description=(
        'Only edges between ROOT nodes. Subgraph body edges belong exclusively in that subgraph.edges; '
        'do not repeat them here or connect a root edge to a subgraph body node.'))
    subgraphs: list[Subgraph] = Field(max_length=1024, description=(
        'Separately owned scopes, each containing its own body nodes and edges exactly once. '
        'Node IDs are globally unique across all scopes, but node arrays are scope-local.'))

    @model_validator(mode='after')
    def coherent_topology(self):
        from .topology import validate_topology
        validate_topology(self.model_dump())
        return self


