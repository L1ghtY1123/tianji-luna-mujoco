"""Expand explicitly authored route groups into the runtime's ordinary edges."""
from copy import deepcopy

from pydantic import Field, ValidationError, model_validator

from ..core.graph import Edge, Identifier, StrictModel


class RouteGroup(StrictModel):
    sources: tuple[Identifier, ...] = Field(min_length=1, max_length=512,
        description='Exact local source node IDs sharing the same declared routing and traversal limit.')
    outcomes: tuple[str, ...] = Field(min_length=1, max_length=64,
        description='Each listed SOURCE outcome routes to target for EVERY listed source. '
                    'Group failure and unknown only when both use the same continuation. '
                    'Each source/outcome pair must be declared exactly once.')
    target: Identifier
    max_traversals: int = Field(ge=1, strict=True,
        description='Independent finite limit on EACH expanded source/outcome edge within one scope invocation.')

    @model_validator(mode='after')
    def distinct_members(self):
        if (len(set(self.sources)) != len(self.sources) or len(set(self.outcomes)) != len(self.outcomes)
                or any(not outcome.strip() for outcome in self.outcomes)):
            raise ValueError('route_group_requires_distinct_sources_and_outcomes')
        return self


class ScopeFailureRoute(StrictModel):
    target: Identifier
    max_traversals: int = Field(ge=1, strict=True)


def expand_scope_routes(scope, *, catalog=None):
    """Compile route declarations and retain already-expanded edges in revision bases."""
    if not isinstance(scope, dict) or not isinstance(scope.get('edges', []), list):
        raise ValueError('graph_route_invalid:scope_requires_object_and_edge_array')
    result = deepcopy(scope)
    defaults = {outcome: result.pop('on_' + outcome, None) for outcome in ('failure', 'unknown')}
    edges = []
    for index, declaration in enumerate(scope.get('edges', [])):
        try:
            if not isinstance(declaration, dict):
                raise ValueError('route_requires_object')
            if 'sources' in declaration or 'outcomes' in declaration:
                group = RouteGroup.model_validate(declaration)
                if len(edges) + len(group.sources) * len(group.outcomes) > 1536:
                    raise ValueError('expanded_edge_limit_exceeded')
                for source_index, source in enumerate(group.sources):
                    for outcome_index, outcome in enumerate(group.outcomes):
                        edges.append(Edge(id=f'route_{index}_{source_index}_{outcome_index}', source=source,
                            on=outcome, target=group.target, max_traversals=group.max_traversals).model_dump(mode='json'))
            else:
                edges.append(Edge.model_validate(declaration).model_dump(mode='json'))
        except (ValidationError, ValueError, TypeError) as error:
            raise ValueError(f'graph_route_invalid:{scope.get("id")}:{index}:{error}') from None
    unique = {}
    for edge in edges:
        key = (edge['source'], edge['on'], edge['target'], edge['max_traversals'])
        unique.setdefault(key, edge)
    edges = list(unique.values())
    explicit = {(edge['source'], edge['on']) for edge in edges}
    for outcome, declaration in defaults.items():
        if declaration is None:
            continue
        try:
            route = ScopeFailureRoute.model_validate(declaration)
        except ValidationError as error:
            raise ValueError(f'graph_route_invalid:{scope.get("id")}:on_{outcome}:{error}') from None
        if route.target not in {node.get('id') for node in scope.get('nodes', [])}:
            raise ValueError(f'graph_route_invalid:{scope.get("id")}:on_{outcome}:unknown_target={route.target}')
        for index, node in enumerate(scope.get('nodes', [])):
            if node.get('type') not in {'skill', 'subgraph'} or (node['id'], outcome) in explicit:
                continue
            outcomes = (catalog or {}).get(node.get('skill'), {}).get('outcomes', ['success', 'failure', 'unknown'])
            if outcome in outcomes:
                edges.append(Edge(id=f'default_{outcome}_{index}', source=node['id'], on=outcome,
                    target=route.target, max_traversals=route.max_traversals).model_dump(mode='json'))
    result['edges'] = edges
    return result


def expand_graph_routes(graph, *, catalog=None):
    result = expand_scope_routes(graph, catalog=catalog)
    if not isinstance(graph.get('subgraphs', []), list):
        raise ValueError('graph_route_invalid:subgraphs_requires_array')
    result['subgraphs'] = [expand_scope_routes(scope, catalog=catalog) for scope in graph.get('subgraphs', [])]
    return result
