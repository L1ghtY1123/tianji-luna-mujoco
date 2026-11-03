"""Structural and conservative dataflow checks for executable graphs; no synthesis."""
from collections import deque

from jsonschema import Draft202012Validator

from ..skills import argument_contract


def validate_capabilities(graph, catalog, *, judgments=None, initial_bindings=(), initial_facts=None,
                          require_completion=False, required_final_facts=None):
    data = graph.model_dump()
    diagnostics = []
    scopes = {scope['id']: scope for scope in [data, *data['subgraphs']]}
    derived = {'$product'} | {key for contract in catalog.values() for key in contract.get('result_bindings', {})}
    def refs(value):
        if isinstance(value, str):
            return {value} if value.startswith('$') else set()
        if isinstance(value, dict):
            return set().union(*(refs(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(refs(item) for item in value))
        return set()
    for scope in scopes.values():
        if require_completion and not any(node['type'] == 'end' for node in scope['nodes']):
            leaves = [node['id'] for node in scope['nodes']
                      if not any(edge['source'] == node['id'] for edge in scope['edges'])]
            raise ValueError('graph_scope_missing_return:' + scope['id'] + ':leaves=' + ','.join(leaves))
        for node in scope['nodes']:
            if node['type'] == 'recovery':
                raise ValueError('initial_graph_cannot_claim_runtime_recovery')
            if node['type'] == 'skill':
                if node['skill'] not in catalog:
                    raise ValueError('unknown_skill:' + node['skill'])
                Draft202012Validator(catalog[node['skill']]['arguments']).validate(node['arguments'])
                unknown = refs(node['arguments']) - derived - {'$target', '$source', '$destination', '$work_area'}
                if unknown:
                    raise ValueError('invalid_argument_binding:' + ','.join(sorted(unknown)))
            # Native router predicates are declared by their graph nodes. They only
            # choose edges; they cannot establish skill effects or completion facts.
    def visit(scope, facts, bindings):
        nodes = {node['id']: node for node in scope['nodes']}
        routes = {(edge['source'], edge['on']): edge['target'] for edge in scope['edges']}
        pending = deque([(scope['entry'], facts, bindings)])
        seen, ends = set(), []
        while pending:
            identifier, state, available = pending.popleft()
            signature = (identifier, tuple(sorted(state.items())), tuple(sorted(available)))
            if signature in seen:
                continue
            seen.add(signature)
            if len(seen) > 8192:
                raise ValueError('graph_validation_budget_exceeded')
            node = nodes[identifier]
            kind = node['type']
            outcomes = []
            if kind == 'end':
                ends.append((node['status'], state, available))
                continue
            if kind == 'skill':
                missing = refs(node['arguments']) & (derived | {'$target'}) - available
                if missing:
                    raise ValueError('unbound_policy_reference:' + identifier + ':' + ','.join(sorted(missing)))
                contract = argument_contract(catalog[node['skill']], node['arguments'])
                conflicts = [key for key, expected in contract.get('requires', {}).items()
                             if key in state and state[key] != expected]
                if initial_facts is not None and conflicts:
                    if not contract.get('recovery_retry_safe') or (identifier, 'failure') in routes:
                        raise ValueError('planning_precondition_not_satisfied:' + identifier + ':' + ','.join(sorted(conflicts)))
                    # A non-actuating, repeatable node can report this precise entry failure to
                    # runtime Decisions. Reachability below is conditional on successful recovery;
                    # this is not a proof that the initial graph supplied the missing prerequisite.
                    diagnostic = {'node_id': identifier, 'missing_preconditions': sorted(conflicts),
                                  'handling': 'runtime_decisions_required_before_node_can_succeed'}
                    if diagnostic not in diagnostics:
                        diagnostics.append(diagnostic)
                    state = {key: value for key, value in state.items() if key not in conflicts}
                effects = contract.get('planning_effects', {})
                produced = set(contract.get('result_bindings', {}))
                statuses = {'success', 'failure', 'unknown'}
                for rule in contract.get('outcome_rules', []):
                    if all(key in state and state[key] == value for key, value in rule['when'].items()):
                        statuses &= set(rule['statuses'])
                for status in statuses:
                    changed = {key: value for key, value in state.items() if key not in effects}
                    if contract.get('invalidates_observation'):
                        changed['task_verified'] = False
                    changed.update(dict.fromkeys(contract.get('provides', []), status == 'success'))
                    if status == 'success':
                        changed.update({key: value for key, value in effects.items() if value is not None})
                    outcomes.append((status, changed, (available - produced) | (produced if status == 'success' else set())))
            elif kind == 'router':
                outcomes = [(status, state, available) for status in ('yes', 'no', 'unknown')]
            elif kind == 'subgraph':
                child = scopes[node['subgraph_id']]
                child_state, child_bindings = dict(state), set(available)
                if child['task'] is not None:
                    if state.get('inventory_finalized') is False:
                        raise ValueError('subgraph_requires_confirmed_inventory:' + identifier)
                    child_bindings = (child_bindings - derived) | {'$target'}
                    for key in ('target_identified', 'destination_bound', 'arm_selected', 'grasp_planned', 'place_planned', 'task_verified'):
                        child_state.pop(key, None)
                for status, changed, outputs in visit(child, child_state, child_bindings):
                    outcomes.append((status, changed, available if child['task'] is not None else outputs))
            for status, changed, outputs in outcomes:
                target = routes.get((identifier, status))
                if target is not None:
                    pending.append((target, changed, outputs))
                # Unhandled results are runtime Decisions recovery, never fictitious success.
        return ends
    final = visit(data, dict(initial_facts or {}), set(initial_bindings))
    successful = [state for status, state, _ in final if status == 'success']
    if require_completion and not successful:
        raise ValueError('graph_no_possible_completion')
    for state in successful:
        for fact, expected in (required_final_facts or {}).items():
            if state.get(fact) is not expected:
                raise ValueError('completion_fact_not_established:' + fact)
    return diagnostics
