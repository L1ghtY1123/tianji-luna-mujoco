"""Pure graph topology checks, independent of execution and models."""


def validate_topology(graph):
    scopes = [graph, *graph['subgraphs']]
    scope_ids = [scope['id'] for scope in scopes]
    owners = [scope['input_id'] for scope in graph['subgraphs'] if scope['input_id'] is not None]
    if len(set(scope_ids)) != len(scope_ids) or len(set(owners)) != len(owners):
        raise ValueError('duplicate_subgraph_or_inventory_owner')
    all_ids = set()
    references = {scope['id']: set() for scope in scopes}
    total = 0
    for scope in scopes:
        nodes = {node['id']: node for node in scope['nodes']}
        if len(nodes) != len(scope['nodes']) or all_ids & nodes.keys():
            raise ValueError('duplicate_graph_node_id')
        all_ids.update(nodes)
        total += len(nodes)
        if total > 8192:
            raise ValueError('graph_node_budget_exceeded')
        if scope['entry'] not in nodes:
            raise ValueError('graph_entry_unknown_node')
        edge_ids, routes, adjacency = set(), set(), {key: [] for key in nodes}
        for edge in scope['edges']:
            if edge['source'] not in nodes or edge['target'] not in nodes:
                raise ValueError('graph_edge_unknown_node:' + edge['id'])
            if edge['id'] in edge_ids:
                raise ValueError('duplicate_graph_edge_id')
            edge_ids.add(edge['id'])
            route = (edge['source'], edge['on'])
            if route in routes and edge['on'] != 'selected':
                raise ValueError('graph_ambiguous_edge:' + edge['source'])
            routes.add(route)
            source = nodes[edge['source']]
            allowed = {'router': {'yes', 'no', 'unknown'},
                       'end': {'unknown'} if source.get('status') == 'success' else set(),
                       'recovery': {'selected'}}.get(source['type'], {'success', 'failure', 'unknown'})
            if edge['on'] not in allowed:
                raise ValueError('graph_edge_invalid_outcome:' + edge['id'])
            adjacency[edge['source']].append(edge['target'])
        reachable, pending = set(), [scope['entry']]
        while pending:
            node_id = pending.pop()
            if node_id not in reachable:
                reachable.add(node_id)
                pending.extend(adjacency[node_id])
        if reachable != nodes.keys():
            raise ValueError('graph_unreachable_nodes:' + ','.join(sorted(nodes.keys() - reachable)))
        for node in nodes.values():
            if node['type'] == 'subgraph':
                if node['subgraph_id'] not in scope_ids[1:]:
                    raise ValueError('unknown_subgraph:' + node['subgraph_id'])
                references[scope['id']].add(node['subgraph_id'])
            if node['type'] == 'recovery' and node['failed_node_id'] not in nodes:
                raise ValueError('recovery_unknown_failed_node')
    reached = set()
    def visit(name, ancestors):
        if name in ancestors:
            raise ValueError('recursive_subgraph_call')
        reached.add(name)
        for child in references[name]:
            visit(child, ancestors | {name})
    visit(graph['id'], set())
    if reached != set(scope_ids):
        raise ValueError('uncalled_subgraph')
    owner_scopes = {scope['id'] for scope in graph['subgraphs'] if scope['input_id'] is not None}
    def has_nested_owner(name):
        return any(child in owner_scopes or has_nested_owner(child) for child in references[name])
    if any(has_nested_owner(name) for name in owner_scopes):
        raise ValueError('nested_inventory_owned_subgraph')


