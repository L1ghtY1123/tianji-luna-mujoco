"""Data-only task and bounded policy graph contracts shared by model and runtime."""
from copy import deepcopy
from typing import Literal

from pydantic import Field, model_validator

from ..core.graph import Graph, StrictModel, TaskSpec


class EntityProposal(StrictModel):
    ref: str = Field(min_length=1)
    kind: Literal['table', 'bin', 'marker', 'cube', 'sphere', 'cylinder', 'object']
    color: str = Field(description=(
        'Short ENGLISH color adjective, e.g. pink, blue, black or unknown. '
        'Used literally in SAM3 appearance fallback prompts; never localize this field. '
        'Human-facing description may use the user language.'))
    description: str
    bbox: list[float] = Field(min_length=4, max_length=4, description=(
        'Visible subject box [x_min, y_min, x_max, y_max], normalized to [0,1] in the named camera. '
        'Both extents must be positive. Configured unmarked regions are referenced directly, not entities.'))
    reference_camera_id: Literal['1', '2', '3']
    parent_ref: str | None
    membership: Literal['required', 'excluded', 'unknown']
    segmentation_prompt: str = Field(description=(
        'Short English visual noun phrase for SAM3, even when the user task is Chinese. '
        'For example green block or blue box. Human descriptions may remain Chinese.'))
    appearance_keywords: tuple[str, ...] = Field(default=(), max_length=8, description=(
        'Visible appearance descriptors: color, shape, texture, pattern or material when observable. '
        'Proposals, not verified identity; do not invent hidden properties or task roles.'))
    segmentation_aliases: tuple[str, ...] = Field(default=(), max_length=2, description=(
        'Up to two distinct short English visual noun phrases, e.g. pink rectangular block, pink package. '
        'Complement segmentation_prompt; omit location, task relations and unsupported properties.'))
    asset_id: str | None = Field(default=None, description=(
        'Optional model identity from the configured known_assets catalog. Null when unrecognized. '
        'An asset identifies declared dimensions, never current position or orientation.'))
    geometry_policy: Literal['auto', 'unknown_obb'] = Field(default='auto', description=(
        'unknown_obb only when the USER explicitly declares this subject unknown/unmodeled/no CAD. '
        'Otherwise auto uses runtime Decisions CAD retrieval. An unrecognized appearance alone is not such a declaration.'))
    geometry_policy_instruction: str | None = Field(default=None, description=(
        'Exact nonempty quote from the user instruction supporting unknown_obb; null for auto. '
        'This is a representation requirement, not visual identity or a CAD match.'))

    @model_validator(mode='after')
    def valid_box(self):
        if self.geometry_policy == 'unknown_obb' and (
                not self.geometry_policy_instruction or not self.geometry_policy_instruction.strip()
                or self.asset_id is not None):
            raise ValueError('unknown_geometry_requires_instruction_and_no_asset')
        x1, y1, x2, y2 = self.bbox
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError('invalid_rgb_box')
        if self.kind not in {'table', 'marker'} and not self.segmentation_prompt.strip():
            raise ValueError('missing_segmentation_prompt')
        if any(not value.strip() or len(value) > 256
               for value in (*self.appearance_keywords, *self.segmentation_aliases)):
            raise ValueError('invalid_appearance_description')
        return self


def object_task_specs(graph):
    """Collect the immutable per-input task from each owned subgraph."""
    data = graph.model_dump() if isinstance(graph, Graph) else Graph.model_validate(graph).model_dump()
    return {scope['input_id']: TaskSpec.model_validate(scope['task'])
            for scope in data['subgraphs'] if scope['task'] is not None}


class InitialPlan(StrictModel):
    entities: list[EntityProposal] = Field(min_length=1, max_length=1024)
    task: TaskSpec
    graph: Graph

    @model_validator(mode='after')
    def coherent_entities(self):
        refs = {entity.ref for entity in self.entities}
        if len(refs) != len(self.entities):
            raise ValueError('duplicate_scene_reference')
        for entity in self.entities:
            if entity.parent_ref is not None and (entity.parent_ref not in refs or entity.parent_ref == entity.ref):
                raise ValueError('invalid_parent_reference')
        if not set(self.task.target_refs) <= refs:
            raise ValueError('unknown_task_target')
        if any(entity.ref in self.task.target_refs and entity.membership == 'excluded'
               for entity in self.entities):
            raise ValueError('task_target_membership_conflict')
        if not set(object_task_specs(self.graph)) <= set(self.task.target_refs):
            raise ValueError('object_task_outside_candidate_inventory')
        return self


def skill_catalog(registry):
    return registry.describe() if hasattr(registry, 'describe') else deepcopy(registry)


def argument_schema(contract):
    return contract.get('arguments', contract.get('argument_schema', {
        'type': 'object', 'properties': {}, 'additionalProperties': False}))


def validate_graph(graph, catalog, *, judgments=None, initial_bindings=(),
                   initial_facts=None, require_completion=False, required_final_facts=None):
    """Validate the graph and skill contracts without generating or rewriting nodes."""
    from .graph_validation import validate_capabilities

    graph = graph if isinstance(graph, Graph) else Graph.model_validate(graph)
    validate_capabilities(graph, skill_catalog(catalog), judgments=judgments, initial_bindings=initial_bindings,
                          initial_facts=initial_facts, require_completion=require_completion,
                          required_final_facts=required_final_facts)
    return graph


def response_schema(model, catalog):
    """Use a closed argument object for each registered skill in strict Responses JSON."""
    schema = model.model_json_schema()
    definitions = schema.get('$defs', {})
    processing = [name for name, contract in skill_catalog(catalog).items() if contract.get('processing_effect')]
    field = definitions['TaskSpec']['properties']['processing']
    field['items'] = {'type': 'string', 'enum': processing} if processing else {'type': 'string'}
    if not processing:
        field['maxItems'] = 0
    # Express ownership in the output schema, not only a post-hoc validator.
    # Shared subgraphs have neither an item owner nor an item TaskSpec.
    owned = deepcopy(definitions['Subgraph'])
    shared = deepcopy(definitions['Subgraph'])
    owned['properties']['input_id'] = {'type': 'string', 'minLength': 1}
    owned['properties']['task'] = {'$ref': '#/$defs/TaskSpec'}
    owned['description'] = (
        'Object-owned scope. Its caller must already have inventory_finalized=true. '
        'An observation or confirmation inside this scope cannot establish its call prerequisite.')
    shared['properties']['input_id'] = {'type': 'null'}
    shared['properties']['task'] = {'type': 'null'}
    shared['description'] = (
        'Shared scope without an object task. Its skills retain their own declared prerequisites; '
        'it may establish shared facts before an object-owned scope is called.')
    definitions['Subgraph'] = {'anyOf': [owned, shared]}
    definitions['SubgraphNode']['properties']['subgraph_id']['description'] = (
        'Existing subgraph ID. Calling an object-owned subgraph binds its task to confirmed inventory '
        'and requires inventory_finalized=true in the CALLER before entry. Initial inventory_finalized '
        'is false. Establish this fact through successful predecessor skills in the root or a shared '
        'scope; placing those skills inside the called object scope is too late.')
    skill_base = definitions['SkillCall']
    variants = []
    for name, contract in skill_catalog(catalog).items():
        variant = deepcopy(skill_base)
        variant['properties']['skill'] = {'type': 'string', 'const': name}
        arguments = deepcopy(argument_schema(contract))
        argument_definitions = arguments.pop('$defs', {})

        def inline(node, local_definitions=argument_definitions):
            if isinstance(node, dict):
                if '$ref' in node:
                    reference = node['$ref'].removeprefix('#/$defs/')
                    return inline(deepcopy(local_definitions[reference]))
                return {key: inline(value) for key, value in node.items()}
            if isinstance(node, list):
                return [inline(value) for value in node]
            return node
        variant['properties']['arguments'] = inline(arguments)
        variants.append(variant)
    if not variants:
        raise ValueError('empty_skill_catalog')
    definitions['SkillCall'] = {'anyOf': variants}
    definitions.pop('RecoveryNode', None)
    definitions.pop('JsonValue', None)

    def normalize(node):
        if isinstance(node, dict):
            node.pop('discriminator', None)
            if 'oneOf' in node:
                node['anyOf'] = node.pop('oneOf')
            if 'anyOf' in node:
                node['anyOf'] = [variant for variant in node['anyOf']
                                 if variant.get('$ref') != '#/$defs/RecoveryNode']
            if node.get('type') == 'object':
                node['additionalProperties'] = False
                node['required'] = list(node.get('properties', {}))
            for value in node.values():
                normalize(value)
        elif isinstance(node, list):
            for value in node:
                normalize(value)
    normalize(schema)
    return schema
