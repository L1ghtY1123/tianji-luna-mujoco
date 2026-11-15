"""Structural and conservative dataflow checks for executable graphs; no synthesis."""
from collections import deque

from jsonschema import Draft202012Validator

from ..skills import argument_contract


class GraphValidationError(ValueError):
    """Report independent errors while preserving the first error code for callers."""

    def __init__(self, issues, counterexamples=()):
        self.issues = list(dict.fromkeys(issues))
        self.counterexamples = list(counterexamples)
        super().__init__(self.issues[0])


def fact_dependencies(facts, path, nodes, catalog):
    """Explain a rejected path from the same contracts used by the validator."""
    result = {}
    for fact, expected in facts.items():
        producers = []
        for skill, contract in catalog.items():
            effects = dict.fromkeys(contract.get('provides', []), True)
            effects.update(contract.get('planning_effects', {}))
            if effects.get(fact) is expected:
                producers.append({'skill': skill, 'arguments': {}})
            for rule in contract.get('argument_rules', []):
                if rule.get('effects', {}).get(fact) is expected:
                    producers.append({'skill': skill, 'arguments': rule['when_arguments']})
        last_change = None
        for step in reversed(path):
            identifier, _, outcome = step.rpartition('/')[2].rpartition(':')
            node = nodes.get(identifier, {})
            if node.get('type') != 'skill':
                continue
            contract = argument_contract(catalog[node['skill']], node['arguments'])
            effects = contract.get('planning_effects', {})
            if (outcome != 'success' and fact in contract.get('failure_preserves', [])
                    and fact not in contract.get('failure_effects', {})):
                continue
            changes = {key: None for key in effects}
            if contract.get('invalidates_observation'):
                changes['task_verified'] = False
            changes.update(dict.fromkeys(contract.get('provides', []), outcome == 'success'))
            if outcome == 'success':
                changes.update({key: value for key, value in effects.items() if value is not None})
            elif outcome in {'failure', 'unknown'}:
                changes.update(contract.get('failure_effects', {}))
            if fact in changes:
                last_change = {'node': identifier, 'skill': node['skill'], 'outcome': outcome,
                               'value': changes[fact]}
                break
        result[fact] = {'last_change': last_change, 'established_by': producers}
    return result


def validate_capabilities(graph, catalog, *, initial_bindings=(), initial_facts=None,
                          require_completion=False, required_final_facts=None):
    data = graph.model_dump()
    scopes = {scope['id']: scope for scope in [data, *data['subgraphs']]}
    all_nodes = {node['id']: node for scope in scopes.values() for node in scope['nodes']}
    issues = []
    counterexamples = {}
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
            issues.append('graph_scope_missing_return:' + scope['id'] + ':leaves=' + ','.join(leaves))
        for node in scope['nodes']:
            if node['type'] == 'skill':
                if node['skill'] not in catalog:
                    issues.append('unknown_skill:' + node['id'] + ':' + node['skill'])
                    continue
                allowed = set(catalog[node['skill']].get('outcomes', ['success', 'failure', 'unknown']))
                for edge in scope['edges']:
                    if edge['source'] == node['id'] and edge['on'] not in allowed:
                        issues.append('skill_outcome_not_declared:' + node['id'] + ':' + edge['on'])
                for error in Draft202012Validator(catalog[node['skill']]['arguments']).iter_errors(node['arguments']):
                    issues.append('invalid_skill_arguments:' + node['id'] + ':' + error.json_path + ':' + error.message)
                unknown = refs(node['arguments']) - derived - {'$target', '$source', '$destination', '$work_area'}
                if unknown:
                    issues.append('invalid_argument_binding:' + node['id'] + ':' + ','.join(sorted(unknown)))
            # Native router predicates are declared by their graph nodes. They only
            # choose edges; they cannot establish skill effects or completion facts.
    if issues:
        raise GraphValidationError(issues)

    def visit(scope, facts, bindings, path=()):
        nodes = {node['id']: node for node in scope['nodes']}
        routes = {(edge['source'], edge['on']): edge['target'] for edge in scope['edges']}
        pending = deque([(scope['entry'], facts, bindings, path)])
        seen, ends = set(), []
        while pending:
            identifier, state, available, path = pending.popleft()
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
                ends.append((node['status'], state, available, path + (scope['id'] + '/' + identifier,)))
                continue
            if kind == 'skill':
                missing = refs(node['arguments']) & (derived | {'$target'}) - available
                if missing:
                    issue = 'unbound_policy_reference:' + identifier + ':' + ','.join(sorted(missing))
                    issues.append(issue)
                    counterexamples.setdefault(issue, {'error': issue, 'scope': scope['id'], 'node': identifier,
                        'symbolic_path': list(path), 'required_bindings': sorted(missing),
                        'available_bindings': sorted(available),
                        'binding_dependencies': {binding: {'established_by': [
                            {'skill': name, 'result_field': contract['result_bindings'][binding]}
                            for name, contract in catalog.items() if binding in contract.get('result_bindings', {})],
                            'reset_by': [name for name, contract in catalog.items()
                                         if '$selected' in contract.get('result_bindings', {})]}
                            for binding in sorted(missing)}})
                contract = argument_contract(catalog[node['skill']], node['arguments'])
                conflicts = [key for key, expected in contract.get('requires', {}).items()
                             if state.get(key, False) != expected]
                if initial_facts is not None and conflicts:
                    issue = 'planning_precondition_not_satisfied:' + identifier + ':' + ','.join(sorted(conflicts))
                    issues.append(issue)
                    counterexamples.setdefault(issue, {'error': issue, 'scope': scope['id'], 'node': identifier,
                        'symbolic_path': list(path), 'required': {key: contract['requires'][key] for key in conflicts},
                        'actual': {key: state.get(key, False) for key in conflicts},
                        'fact_dependencies': fact_dependencies(
                            {key: contract['requires'][key] for key in conflicts}, path, all_nodes, catalog)})
                if missing or initial_facts is not None and conflicts:
                    # This path cannot establish effects. Other reachable paths can still be checked.
                    continue
                effects = contract.get('planning_effects', {})
                produced = set(contract.get('result_bindings', {}))
                retained = available - (derived if '$selected' in produced else produced)
                statuses = list(contract.get('outcomes', ['success', 'failure', 'unknown']))
                for rule in contract.get('outcome_rules', []):
                    if all(key in state and state[key] == value for key, value in rule['when'].items()):
                        statuses = [status for status in statuses if status in rule['statuses']]
                for status in statuses:
                    changed = {key: value for key, value in state.items() if key not in effects}
                    if contract.get('invalidates_observation'):
                        changed['task_verified'] = False
                    changed.update(dict.fromkeys(contract.get('provides', []), status == 'success'))
                    if status == 'success':
                        changed.update({key: value for key, value in effects.items() if value is not None})
                    else:
                        changed.update({key: state[key] for key in contract.get('failure_preserves', []) if key in state})
                        if status in {'failure', 'unknown'}:
                            for key, value in contract.get('failure_effects', {}).items():
                                if value is None:
                                    changed.pop(key, None)
                                else:
                                    changed[key] = value
                    outcomes.append((status, changed, retained | (produced if status == 'success' else set()), path))
            elif kind == 'router':
                outcomes = [(status, state, available, path) for status in (node.get('choices') or ('yes', 'no', 'unknown'))]
            elif kind == 'subgraph':
                child = scopes[node['subgraph_id']]
                child_state, child_bindings = dict(state), set(available)
                if child['task'] is not None:
                    child_bindings = (child_bindings - derived) | {'$target'}
                    for key in ('target_identified', 'destination_bound', 'arm_selected', 'grasp_planned', 'place_planned', 'task_verified'):
                        child_state.pop(key, None)
                for status, changed, outputs, child_path in visit(child, child_state, child_bindings,
                        path + (scope['id'] + '/' + identifier + ':call',)):
                    outcomes.append((status, changed, available if child['task'] is not None else outputs, child_path))
            for status, changed, outputs, previous_path in outcomes:
                target = routes.get((identifier, status))
                if target is not None:
                    pending.append((target, changed, outputs, previous_path + (scope['id'] + '/' + identifier + ':' + status,)))
                elif require_completion:
                    issues.append('graph_outcome_edge_missing:' + identifier + ':' + status)
        return ends
    final = visit(data, dict(initial_facts or {}), set(initial_bindings))
    successful = [state for status, state, _, _ in final if status == 'success']
    if require_completion and not successful:
        issues.append('graph_no_possible_completion')
    for state in successful:
        for fact, expected in (required_final_facts or {}).items():
            if state.get(fact) is not expected:
                issues.append('completion_fact_not_established:' + fact)
    if require_completion:
        from ..core.recovery_coverage import missing_observation_recovery
        issues.extend(missing_observation_recovery(data, catalog))
    if issues:
        raise GraphValidationError(issues, counterexamples.values())
