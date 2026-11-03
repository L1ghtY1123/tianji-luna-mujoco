"""Small explicit graph fixtures for contract tests."""
from luna_mujoco.agent.schema import Graph, validate_graph


def call(identifier, skill, **arguments):
    return {'type': 'skill', 'id': identifier, 'skill': skill, 'arguments': arguments}


def linear_graph(nodes, *, identifier='test_graph', finish='test_done'):
    nodes = list(nodes)
    targets = [node['id'] for node in nodes] + [finish]
    return {'version': 2, 'id': identifier, 'entry': targets[0], 'subgraphs': [],
        'nodes': nodes + [{'type': 'end', 'id': finish, 'status': 'success', 'reason': ''}],
        'edges': [{'id': identifier + '_edge_' + str(i), 'source': source, 'target': target, 'on': 'success'}
                  for i, (source, target) in enumerate(zip(targets, targets[1:]))]}


def check(nodes, catalog, facts=None):
    data = linear_graph(nodes)
    data['nodes'].append({'type': 'end', 'id': 'failed', 'status': 'failure', 'reason': 'test failure edge'})
    data['edges'].extend({'id': node['id'] + '_failed', 'source': node['id'], 'target': 'failed', 'on': 'failure'}
                        for node in nodes)
    catalog = {name: {'arguments': {'type': 'object', 'properties': {}, 'additionalProperties': False}, **contract}
               for name, contract in catalog.items()}
    graph = Graph.model_validate(data)
    return validate_graph(graph, catalog, initial_facts=facts or {}, require_completion=True)


def robot_catalog():
    from luna_mujoco.runtime.controller import (
        ArmArguments,
        BuildSceneArguments,
        DestinationArguments,
        ObservationMotionArguments,
        TargetArguments,
    )
    from luna_mujoco.skills import Registry
    registry = Registry()
    registry.argument_models.update(identify_target=TargetArguments,
        identify_destination=DestinationArguments, choose_arm=ArmArguments, build_scene=BuildSceneArguments,
        move_to_observe=ObservationMotionArguments)
    return registry.describe()
