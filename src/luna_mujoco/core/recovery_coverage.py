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
    """Normal empty-hand identity/support checkpoints offer a fresh observation route.

    Follow explicit subgraph returns so a caller may own the recovery. Retry-only
    nodes are excluded: their failed result may terminate without another retry.
    A fresh in-place capture qualifies; movement also invalidates reused captures.
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

    transfer = any(node.get('skill') == 'execute_place' for scope in nodes.values() for node in scope.values())
    grasp_steps = ('recover_failed_grasp', 'observe_scene', 'generate_grasps', 'plan_grasp', 'execute_grasp', 'verify_grasp')

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
                    yield scope_id, identifier, parents, outcome, False
                if transfer and node['skill'] == 'verify_grasp':
                    yield scope_id, identifier, parents, 'failure', True
            pending.extend(e['target'] for e in edges[scope_id].get(identifier, [])
                           if e['on'] not in {'failure', 'unknown'})

    def has_route(scope_id, identifier, parents, outcome, grasp_retry):
        pending = [(scope_id, e['target'], parents, False, 0)
                   for e in edges[scope_id].get(identifier, []) if e['on'] == outcome]
        seen = set()
        while pending:
            state = pending.pop()
            if state in seen:
                continue
            seen.add(state)
            scope, current, stack, moved, step = state
            node = nodes[scope][current]
            if node['type'] == 'end':
                if stack:
                    parent, call = stack[-1]
                    pending.extend((parent, e['target'], stack[:-1], moved, step)
                        for e in edges[parent].get(call, []) if e['on'] == node['status'])
                continue
            if node['type'] == 'subgraph':
                child = node['subgraph_id']
                pending.append((child, scopes[child]['entry'], stack + ((scope, current),), moved, step))
                continue
            if node['type'] == 'router':
                outcomes = set(node.get('choices') or ('yes', 'no')) - {'unknown'}
            else:
                if node['skill'] == 'move_to_observe':
                    moved = True
                fresh = node['skill'] == 'observe_scene' and (moved or node.get('arguments', {}).get('capture_mode') == 'fresh')
                if grasp_retry:
                    if node['skill'] == 'execute_grasp' and grasp_steps[step] != 'execute_grasp':
                        continue
                    if node['skill'] == grasp_steps[step] and (node['skill'] != 'observe_scene' or fresh):
                        step += 1
                        if step == len(grasp_steps):
                            return True
                elif fresh:
                    return True
                outcomes = {'success'}
            pending.extend((scope, e['target'], stack, moved, step)
                           for e in edges[scope].get(current, []) if e['on'] in outcomes)
        return False

    issues = []
    for scope, node, parents, outcome, grasp_retry in checkpoints(graph['id'], ()):
        if not has_route(scope, node, parents, outcome, grasp_retry):
            code = 'graph_empty_grasp_retry_missing' if grasp_retry else 'graph_observation_recovery_missing'
            issues.append(f'{code}:{scope}:{node}:{outcome}')
    return issues
