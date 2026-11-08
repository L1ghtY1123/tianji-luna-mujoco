"""Pure graph topology checks, independent of execution and models."""
from copy import deepcopy


def reachable_graph_for_diagnostics(graph):
    """Inspect active paths after unreachable declarations have already rejected a graph.

    Call only when topology errors are limited to unreachable nodes/scopes. The
    returned copy is used for additional error reporting, never for execution.
    """
    result = deepcopy(graph)
    scopes = {scope['id']: scope for scope in [result, *result['subgraphs']]}
    for scope in scopes.values():
        adjacency = {node['id']: [] for node in scope['nodes']}
        for edge in scope['edges']:
            adjacency[edge['source']].append(edge['target'])
        reached, pending = set(), [scope['entry']]
        while pending:
            identifier = pending.pop()
            if identifier in reached:
                continue
            reached.add(identifier)
            pending.extend(adjacency[identifier])
        scope['nodes'] = [node for node in scope['nodes'] if node['id'] in reached]
        scope['edges'] = [edge for edge in scope['edges'] if edge['source'] in reached]
    called, pending = set(), [result['id']]
    while pending:
        identifier = pending.pop()
        if identifier in called:
            continue
        called.add(identifier)
        pending.extend(node['subgraph_id'] for node in scopes[identifier]['nodes'] if node['type'] == 'subgraph')
    result['subgraphs'] = [scope for scope in result['subgraphs'] if scope['id'] in called]
    return result


class TopologyValidationError(ValueError):
    def __init__(self, issues):
        self.issues = list(dict.fromkeys(issues))
        super().__init__(' | '.join(self.issues))


def recovery_reentry_issues(scope, nodes):
    """A repaired checkpoint must be reachable without reusing an exhausted prefix."""
    incoming, outgoing = {key: [] for key in nodes}, {key: [] for key in nodes}
    for edge in scope['edges']:
        if edge['source'] not in nodes or edge['target'] not in nodes:
            return []  # The ordinary topology check reports invalid endpoints.
        outgoing[edge['source']].append(edge)
        if edge['on'] == 'success' and nodes[edge['source']]['type'] in {'skill', 'subgraph'}:
            incoming[edge['target']].append(edge)
    normal, pending = set(), [scope['entry']]
    while pending:
        identifier = pending.pop()
        if identifier in normal or identifier not in nodes:
            continue
        normal.add(identifier)
        pending.extend(edge['target'] for edge in outgoing[identifier]
                       if edge['on'] not in {'failure', 'unknown'})
    issues = {}
    for checkpoint, node in nodes.items():
        if node['type'] not in {'skill', 'subgraph'} or checkpoint not in normal:
            continue
        rejected = [edge for edge in outgoing[checkpoint] if edge['on'] in {'failure', 'unknown'}]
        if not rejected:
            continue
        prefixes, visited, pending = {}, {checkpoint}, [checkpoint]
        while pending:
            for edge in incoming[pending.pop()]:
                if edge['source'] not in normal:
                    continue
                if edge['max_traversals'] == 1:
                    prefixes[edge['source']] = edge
                if edge['source'] not in visited:
                    visited.add(edge['source'])
                    pending.append(edge['source'])
        reached, pending = {checkpoint}, [edge['target'] for edge in rejected]
        while pending:
            identifier = pending.pop()
            if identifier in reached:
                continue
            reached.add(identifier)
            if identifier in prefixes:
                edge = prefixes[identifier]
                issues.setdefault(edge['id'], 'graph_recovery_reuses_exhausted_success_edge:'
                    + scope['id'] + ':' + edge['id'] + ':source=' + identifier
                    + ':failed_checkpoint=' + checkpoint + ':required_minimum=2')
            source = nodes[identifier]
            if source['type'] == 'end':
                continue
            pending.extend(edge['target'] for edge in outgoing[identifier]
                if edge['on'] == 'success' or source['type'] == 'router' and edge['on'] != 'unknown')
    return list(issues.values())


def validate_topology(graph):
    scopes = [graph, *graph['subgraphs']]
    scope_ids = [scope['id'] for scope in scopes]
    owners = [scope['input_id'] for scope in graph['subgraphs'] if scope['input_id'] is not None]
    issues = []
    if len(set(scope_ids)) != len(scope_ids) or len(set(owners)) != len(owners):
        issues.append('duplicate_subgraph_or_inventory_owner')
    node_owners = {}
    references = {scope['id']: set() for scope in scopes}
    total = 0
    for scope in scopes:
        nodes = {node['id']: node for node in scope['nodes']}
        for node in scope['nodes']:
            if node['id'] in node_owners:
                issues.append('duplicate_graph_node_id:' + node['id']
                              + ':first_scope=' + node_owners[node['id']]
                              + ':duplicate_scope=' + scope['id'])
            else:
                node_owners[node['id']] = scope['id']
        total += len(nodes)
        if total > 8192:
            raise ValueError('graph_node_budget_exceeded')
        if scope['entry'] not in nodes:
            issues.append('graph_entry_unknown_node:' + scope['id'])
        edge_ids, routes, adjacency = {}, set(), {key: [] for key in nodes}
        for edge in scope['edges']:
            if edge['id'] in edge_ids:
                previous = edge_ids[edge['id']]
                issues.append('duplicate_graph_edge_id:' + scope['id'] + ':' + edge['id']
                              + ':first=' + previous['source'] + '/' + previous['on']
                              + ':duplicate=' + edge['source'] + '/' + edge['on'])
            else:
                edge_ids[edge['id']] = edge
            if edge['source'] not in nodes or edge['target'] not in nodes:
                missing = [name + '=' + edge[name] for name in ('source', 'target') if edge[name] not in nodes]
                issues.append('graph_edge_unknown_node:' + scope['id'] + ':' + edge['id']
                              + ':missing_' + ',missing_'.join(missing))
                continue
            route = (edge['source'], edge['on'])
            if route in routes:
                issues.append('graph_ambiguous_edge:' + edge['source'] + ':' + edge['on'])
            routes.add(route)
            source = nodes[edge['source']]
            allowed = {'router': set(source.get('choices') or ('yes', 'no', 'unknown')),
                       'skill': {'success', 'failure', 'unknown', 'empty'},
                       'end': {'unknown'} if source.get('status') == 'success' else set()}.get(source['type'], {'success', 'failure', 'unknown'})
            if edge['on'] not in allowed:
                issues.append('graph_edge_invalid_outcome:' + scope['id'] + ':' + edge['id']
                              + ':source=' + edge['source'] + ':source_type=' + source['type']
                              + ':target=' + edge['target'] + ':on=' + edge['on']
                              + ':allowed=' + ','.join(sorted(allowed)))
            adjacency[edge['source']].append(edge['target'])
        reachable, pending = set(), [scope['entry']] if scope['entry'] in nodes else []
        while pending:
            node_id = pending.pop()
            if node_id not in reachable:
                reachable.add(node_id)
                pending.extend(adjacency[node_id])
        if reachable != nodes.keys():
            issues.append('graph_unreachable_nodes:' + ','.join(sorted(nodes.keys() - reachable)))
        issues.extend(recovery_reentry_issues(scope, nodes))
        for node in nodes.values():
            if node['type'] == 'subgraph':
                if node['subgraph_id'] not in scope_ids[1:]:
                    issues.append('unknown_subgraph:' + node['subgraph_id'])
                    continue
                references[scope['id']].add(node['subgraph_id'])
    if issues:
        raise TopologyValidationError(issues)
    reached = set()
    def visit(name, ancestors):
        if name in ancestors:
            raise ValueError('recursive_subgraph_call')
        reached.add(name)
        for child in references[name]:
            visit(child, ancestors | {name})
    visit(graph['id'], set())
    if reached != set(scope_ids):
        raise TopologyValidationError(['uncalled_subgraph:' + ','.join(sorted(set(scope_ids) - reached))])
    owner_scopes = {scope['id'] for scope in graph['subgraphs'] if scope['input_id'] is not None}
    def has_nested_owner(name):
        return any(child in owner_scopes or has_nested_owner(child) for child in references[name])
    if any(has_nested_owner(name) for name in owner_scopes):
        raise ValueError('nested_inventory_owned_subgraph')


