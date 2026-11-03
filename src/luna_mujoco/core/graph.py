"""Immutable task requirements and executable graph types; no model or robot calls."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TaskSpec(StrictModel):
    """Freeze the user's interpreted goal; revisions can only change the graph."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    instruction: str = Field(min_length=1)
    supported: bool
    reason: str
    scope: Literal['all', 'count', 'ids'] = Field(description=(
        'all selects every matching observed candidate; count selects quantity candidates; '
        'ids selects exactly target_refs. For ids, quantity may be null or the exact number of refs.'))
    target_selector: str = Field(min_length=1)
    target_refs: tuple[str, ...] = Field(description=(
        'All observed candidate entity refs matching target_selector, also for count tasks. '
        'Must be nonempty for a supported task; quantity selects from these candidates later.'))
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
    selection_order: Literal['inventory', 'nearest', 'explicit']
    processing: tuple[str, ...] = Field(description=(
        'Required object transformations only, declared processing_effect=true in the skill catalog. '
        'Ordinary observation, mapping, grasp, movement and placement are graph nodes, not processing requirements. '
        'Use an empty array for direct grasp-place without processing.'))
    completion_conditions: tuple[str, ...]
    completion_pose: Literal['observation', 'initial'] | None = Field(default=None, description=(
        'Requested final robot posture. initial means both arms at their measured task-start joints; '
        'observation means the configured observation pose. Honor explicit user or supplied '
        'completion_requirements; otherwise null. '
        'The runtime verifies actual joint feedback, never RGB judgment alone.'))
    constraints: tuple[str, ...]
    source_inventory_complete: Literal['confirmed', 'unknown']

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
        if self.supported and not self.target_refs:
            raise ValueError('supported_task_requires_observed_target_refs')
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
    condition: str = Field(min_length=1, description='A graph-declared evidence question for Decisions to choose yes/no/unknown edges; does not establish completion facts.')


class EndNode(StrictModel):
    type: Literal['end']
    id: Identifier
    status: Literal['success', 'failure', 'unknown'] = Field(description=(
        'Explicit return from this scope. Every subgraph requires an end reached by its final edge; '
        'the last skill never implicitly returns. Parent subgraph-call edges match this status. '
        'An owned subgraph success checks verified delivery/observation receipts; missing evidence '
        'produces unknown and can request Decisions recovery of this end without replaying the subgraph.'))
    reason: str


class RecoveryNode(StrictModel):
    """Runtime-only Decisions router; never proposed by initial Responses."""
    type: Literal['recovery']
    id: Identifier
    failed_node_id: Identifier


Node = Annotated[SkillCall | SubgraphNode | RouterNode | EndNode | RecoveryNode,
                 Field(discriminator='type')]


class Edge(StrictModel):
    id: Identifier
    source: Identifier
    target: Identifier
    on: Literal['success', 'failure', 'unknown', 'yes', 'no', 'selected']


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
    """Executable inventory-owned graph. Only edges determine control order."""
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


