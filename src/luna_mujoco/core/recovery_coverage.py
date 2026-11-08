"""Check authored observation-recovery coverage without adding nodes or selecting edges."""


def observation_recovery_outcomes(contract):
    requires = contract.get('requires', {})
    errors = set(contract.get('errors', []))
    outcomes = []
    if (requires.get('grasp_held') is False
            and {'observed_support_unknown', 'observed_instance_missing_or_ambiguous'} & errors):
        outcomes.append('unknown')
    return outcomes


def missing_observation_recovery(graph, catalog):
    """Normal empty-hand identity/support checkpoints offer an observation route.

    Follow explicit subgraph returns so a caller may own the recovery. Retry-only
    nodes are excluded: their failed result may terminate without another retry.
    Ordinary capability validation separately checks whether every step is legal.
    """
    if not {'move_to_observe', 'observe_scene'} <= catalog.keys():
        return []
    scopes = {s['id']: s for s in [graph, *graph['subgraphs']]}
    nodes = {s: {n['id']: n for n in scope['nodes']} for s, scope in scopes.items()}
    edges = {s: {} for s in scopes}
    for s, scope in scopes.items():
        for edge in scope['edges']:
            edges[s].setdefault(edge['source'], []).append(edge)

    def checkpoints(scope_id, parents):
        pending, seen = [scopes[scope_id]['entry']], set()
        while pending:
            identifier = pending.pop()
            if identifier in seen:
                continue
            seen.add(identifier)
            node = nodes[scope_id][identifier]
            if node['type'] == 'subgraph':
                yield from checkpoints(node['subgraph_id'], parents + ((scope_id, identifier),))
            elif node['type'] == 'skill':
                for outcome in observation_recovery_outcomes(catalog[node['skill']]):
                    yield scope_id, identifier, parents, outcome
            pending.extend(e['target'] for e in edges[scope_id].get(identifier, [])
                           if e['on'] not in {'failure', 'unknown'})

    def has_route(scope_id, identifier, parents, outcome):
        pending = [(scope_id, e['target'], parents, False)
                   for e in edges[scope_id].get(identifier, []) if e['on'] == outcome]
        seen = set()
        while pending:
            state = pending.pop()
            if state in seen:
                continue
            seen.add(state)
            scope, current, stack, moved = state
            node = nodes[scope][current]
            if node['type'] == 'end':
                if stack:
                    parent, call = stack[-1]
                    pending.extend((parent, e['target'], stack[:-1], moved)
                        for e in edges[parent].get(call, []) if e['on'] == node['status'])
                continue
            if node['type'] == 'subgraph':
                child = node['subgraph_id']
                pending.append((child, scopes[child]['entry'], stack + ((scope, current),), moved))
                continue
            if node['type'] == 'router':
                outcomes = set(node.get('choices') or ('yes', 'no')) - {'unknown'}
            else:
                if node['skill'] == 'move_to_observe':
                    moved = True
                if moved and node['skill'] == 'observe_scene':
                    return True
                outcomes = {'success'}
            pending.extend((scope, e['target'], stack, moved)
                           for e in edges[scope].get(current, []) if e['on'] in outcomes)
        return False

    return [f'graph_observation_recovery_missing:{scope}:{node}:{outcome}'
            for scope, node, parents, outcome in checkpoints(graph['id'], ())
            if not has_route(scope, node, parents, outcome)]
