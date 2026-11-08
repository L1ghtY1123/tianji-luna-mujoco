"""Read-only graph evidence for a declared branch choice."""
from copy import deepcopy


def destination_reference_evidence(observation, configured_regions, task):
    """Resolve the locations named by the task without deciding semantic permission."""
    objects = {obj['id']: obj for obj in observation.get('objects', [])}
    objects.update(observation.get('scene_entities', {}))
    fields = ('id', 'kind', 'color', 'position', 'position_source', 'placement_reference')
    result = {}
    for role, identifier in (('work_area', task.work_area_ref), ('final_destination', task.destination_ref)):
        if identifier is None:
            continue
        entity = objects.get(identifier, {})
        result[role] = {'id': identifier, 'configuration': deepcopy(configured_regions.get(identifier)),
            'current_measurement': {key: deepcopy(entity[key]) for key in fields if key in entity},
            'observation_id': observation.get('observation_id')}
    return result


def available_branch_choices(choices, branches):
    """An exhausted edge remains in the trace but is no longer a selectable operation."""
    available = {branch['outcome'] for branch in branches if branch['remaining_traversals'] > 0}
    return tuple(choice for choice in choices if choice == 'unknown' or choice in available)


def branch_evidence(graph, branches, catalog, last_failure):
    scopes = {scope.id: scope.model_dump(mode='json') for scope in graph.subgraphs}
    node_scopes = {node.id: scope for scope in [graph, *graph.subgraphs] for node in scope.nodes}
    continuations = []
    for branch in branches:
        scope = node_scopes[branch['target']['id']]
        nodes = {node.id: node for node in scope.nodes}
        current, visited, steps, edges = branch['target']['id'], set(), [], []
        while current not in visited:
            visited.add(current)
            node = nodes[current]
            steps.append(node.model_dump(mode='json'))
            outgoing = [edge for edge in scope.edges if edge.source == current]
            edges.extend(edge.model_dump(mode='json') for edge in outgoing)
            if node.type in {'end', 'router'}:
                break
            success = [edge for edge in outgoing if edge.on == 'success']
            if not success:
                break
            current = success[0].target
        continuations.append({'choice': branch['outcome'], 'scope_id': scope.id,
            'remaining_entry_traversals': branch['remaining_traversals'],
            'success_continuation': steps, 'outcome_edges': edges,
            'meaning': 'Declared success continuation to the next router, end, or cycle. '
                'Other outcomes use the listed edges; subgraph calls return through their caller edges. '
                'These are available operations, not evidence that recovery will succeed.'})
    pending = [branch['target']['subgraph_id'] for branch in branches if branch['target']['type'] == 'subgraph']
    selected = {}
    skills = {branch['target']['skill'] for branch in branches if branch['target']['type'] == 'skill'}
    for route in continuations:
        for node in route['success_continuation']:
            if node['type'] == 'skill':
                skills.add(node['skill'])
            elif node['type'] == 'subgraph':
                pending.append(node['subgraph_id'])
    if (last_failure or {}).get('skill'):
        skills.add(last_failure['skill'])
    while pending:
        identifier = pending.pop()
        if identifier in selected:
            continue
        scope = scopes[identifier]
        selected[identifier] = deepcopy(scope)
        for node in scope['nodes']:
            if node['type'] == 'skill':
                skills.add(node['skill'])
            elif node['type'] == 'subgraph':
                pending.append(node['subgraph_id'])
    return {'declared_subgraphs': selected, 'declared_continuations': continuations,
            'skill_contracts': {name: deepcopy(contract) for name, contract in catalog.items() if name in skills}}
