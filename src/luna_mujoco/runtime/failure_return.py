"""An explicit bounded return graph for every exhausted or unexpected task failure."""
from copy import deepcopy


def terminal_graph():
    from ..agent.graph_authoring import expand_graph_routes

    skills = [('prepare', 'prepare_failure_return'), ('observe', 'observe_after_failure'),
              ('map', 'build_failure_scene'), ('return', 'return_after_failure')]
    nodes = [{'type': 'skill', 'id': identifier, 'skill': skill, 'arguments': {}}
             for identifier, skill in skills]
    nodes.append({'type': 'router', 'id': 'mode', 'condition': 'Apply measured terminal return state',
                  'rule': 'failure_return', 'choices': ['at_pose', 'empty', 'held', 'refresh', 'blocked', 'unknown']})
    nodes.extend({'type': 'end', 'id': status, 'status': status, 'reason': 'Terminal return ' + status}
                 for status in ('success', 'failure', 'unknown'))
    routes = [('prepare', ['success'], 'mode'), ('mode', ['at_pose'], 'success'),
              ('mode', ['empty', 'held'], 'return'), ('mode', ['refresh'], 'observe'),
              ('mode', ['blocked'], 'failure'), ('mode', ['unknown'], 'unknown'),
              ('observe', ['success'], 'map'), ('map', ['success'], 'return'), ('return', ['success'], 'success')]
    result = expand_graph_routes({'version': 2, 'id': 'terminal_failure_return', 'entry': 'prepare',
        'nodes': nodes, 'subgraphs': [], 'on_failure': {'target': 'failure', 'max_traversals': 1},
        'on_unknown': {'target': 'unknown', 'max_traversals': 1}, 'edges': [
            {'sources': [source], 'outcomes': outcomes, 'target': target, 'max_traversals': 1}
            for source, outcomes, target in routes]})
    for node in result['nodes']:
        node['id'] = 'terminal_' + node['id']
    result['entry'] = 'terminal_' + result['entry']
    for edge in result['edges']:
        edge['id'] = 'terminal_' + edge['id']
        edge['source'] = 'terminal_' + edge['source']
        edge['target'] = 'terminal_' + edge['target']
    return result


def wrap_failure_exits(graph):
    from ..core.failure_policy import TERMINAL_SKILLS

    graph = graph.model_dump(mode='json') if hasattr(graph, 'model_dump') else deepcopy(graph)
    if any(node.get('skill') in TERMINAL_SKILLS for scope in [graph, *graph['subgraphs']] for node in scope['nodes']):
        raise ValueError('terminal_capability_is_reserved')
    if any(node.get('rule') == 'failure_return' for scope in [graph, *graph['subgraphs']] for node in scope['nodes']):
        raise ValueError('terminal_router_is_reserved')
    fixed = terminal_graph()
    if fixed['id'] in {graph['id'], *(scope['id'] for scope in graph['subgraphs'])}:
        raise ValueError('terminal_scope_id_is_reserved')
    used = {item['id'] for scope in [graph, *graph['subgraphs']] for key in ('nodes', 'edges') for item in scope[key]}

    def free_id(stem):
        stem = stem[:80]
        candidate, index = stem, 0
        while candidate in used:
            index += 1
            candidate = stem + '_' + str(index)
        used.add(candidate)
        return candidate

    renamed = {node['id']: free_id(node['id']) for node in fixed['nodes']}
    for node in fixed['nodes']:
        node['id'] = renamed[node['id']]
    fixed['entry'] = renamed[fixed['entry']]
    for edge in fixed['edges']:
        edge.update(id=free_id(edge['id']), source=renamed[edge['source']], target=renamed[edge['target']])
    graph['subgraphs'].append({key: value for key, value in fixed.items() if key not in {'version', 'subgraphs'}}
                              | {'input_id': None, 'task': None})
    nodes = []
    for node in graph['nodes']:
        if node['type'] != 'end' or node['status'] == 'success':
            nodes.append(node)
            continue
        terminal = deepcopy(node)
        terminal['id'] = free_id(node['id'] + '_stopped')
        nodes.extend([{'type': 'subgraph', 'id': node['id'], 'subgraph_id': fixed['id']}, terminal])
        graph['edges'].extend({'id': free_id(node['id'] + '_return_' + status), 'source': node['id'],
                               'target': terminal['id'], 'on': status, 'max_traversals': 1}
                              for status in ('success', 'failure', 'unknown'))
    graph['nodes'] = nodes
    return graph
