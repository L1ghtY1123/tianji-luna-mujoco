"""Bind compiled task references to separately registered permanent instances."""
from copy import deepcopy

from ..agent.schema import TaskSpec, object_task_specs
from .diagnostics import DiagnosticError


def bind_compiled_task(plan, registration, memory_entities):
    """Bind explicit task references without changing graph topology or object memory."""
    update = registration
    identifiers, proposed = update['identifiers'], update['entities']
    unresolved_refs = {row['proposal']['ref'] for row in update['unresolved_proposals']}

    def bind_literals(value):
        if isinstance(value, str):
            if value in unresolved_refs:
                raise DiagnosticError('task_reference_unconfirmed', {
                    'reference': value, 'unresolved_proposals': update['unresolved_proposals']})
            return identifiers.get(value, value)
        if isinstance(value, (tuple, list)):
            result = []
            for item in value:
                bound = bind_literals(item)
                result.extend(bound if isinstance(bound, list) else [bound])
            return result
        if isinstance(value, dict):
            return {key: bind_literals(item) for key, item in value.items()}
        return value

    entities = deepcopy(proposed)
    target_ids = bind_literals([ref for ref in plan.task.target_refs
                               if plan.task.scope == 'ids' or ref in identifiers])
    task_data = plan.task.model_dump(mode='json')
    task_data['target_refs'] = target_ids
    for name in ('source_ref', 'destination_ref', 'work_area_ref', 'allowed_intermediate_refs'):
        task_data[name] = bind_literals(task_data[name])
    task = TaskSpec.model_validate(task_data)

    graph = plan.graph.model_dump(mode='json')

    def bind_nodes(nodes):
        for node in nodes:
            if node['type'] == 'skill':
                node['arguments'] = bind_literals(node['arguments'])
    bind_nodes(graph['nodes'])
    for subgraph in graph['subgraphs']:
        if subgraph['task'] is not None:
            subgraph['input_id'] = bind_literals(subgraph['input_id'])
            for field in ('target_refs', 'source_ref', 'destination_ref', 'work_area_ref', 'allowed_intermediate_refs'):
                subgraph['task'][field] = bind_literals(subgraph['task'][field])
        bind_nodes(subgraph['nodes'])
    task_data = task.model_dump(mode='json')
    support_ref = plan.task.work_area_ref or plan.task.destination_ref
    proposals = {entity.ref: entity for entity in plan.task_selectors}
    if support_ref not in proposals or proposals[support_ref].kind != 'table':
        support_ref = next(iter(plan.support_table_refs), None)
    objects = {'source_id': task.source_ref, 'support_id': identifiers.get(support_ref),
                 'entity_ref_bindings': identifiers,
                 'destination_id': task.destination_ref, 'work_area_id': task.work_area_ref,
                 'object_tasks': {key: value.model_dump(mode='json')
                                  for key, value in object_task_specs(graph).items()}}
    return {'entities': entities, 'task_spec': task_data, 'policy_graph': graph,
            'task_bindings': objects, 'unresolved_proposals': deepcopy(update['unresolved_proposals'])}
