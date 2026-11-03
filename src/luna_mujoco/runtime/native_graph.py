"""Execute explicit edges and inventory-owned scopes, including audited Decisions patches."""
from copy import deepcopy

from ..core.graph import Graph
from ..skills import same_result_arguments
from .execution_state import ExecutionState, PolicyStopped, UnhandledSkillResult, result_status


class NativeGraphExecutor(ExecutionState):
    def __init__(self, *args, propose_recovery=None, repeatable_skills=(), reusable_result_skills=(),
                 reusable_result_ignored_arguments=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.propose_recovery = propose_recovery
        self.repeatable_skills = set(repeatable_skills)
        self.reusable_result_skills = set(reusable_result_skills)
        self.reusable_result_ignored_arguments = deepcopy(reusable_result_ignored_arguments or {})
        self.graph = None
        self.revision = 0
        self.recoveries = {}
        self.completed_actions = set()
        self.resolved_arguments = {}
        self.node_authors = {}
        self.last_failure = None

    def event(self, kind, title, data):
        if self.emit is not None:
            self.emit(kind, title, {'graph_id': self.graph['id'], 'graph_revision': self.revision, **deepcopy(data)})

    async def run(self, graph):
        self.graph = Graph.model_validate(graph.model_dump() if hasattr(graph, 'model_dump') else graph).model_dump()
        result = await self._scope(self.graph)
        if result_status(result) != 'success':
            raise PolicyStopped(result.get('reason', 'graph_end_failure'), self.last_failure or result)
        return deepcopy(self.results)

    def _patch(self, scope, nodes, edges, *, source, evidence):
        candidate = deepcopy(self.graph)
        target = candidate if scope['id'] == candidate['id'] else next(
            child for child in candidate['subgraphs'] if child['id'] == scope['id'])
        target['nodes'].extend(deepcopy(nodes))
        target['edges'].extend(deepcopy(edges))
        Graph.model_validate(candidate)  # No broken route may be published or executed.
        scope['nodes'].extend(deepcopy(nodes))
        scope['edges'].extend(deepcopy(edges))
        self.revision += 1
        self.node_authors.update({node['id']: source for node in nodes})
        self.event('policy_graph_patch', 'graph 更新', {'source': source, 'subgraph_id': scope['id'],
            'added_nodes': nodes, 'added_edges': edges, 'evidence': evidence, 'policy_graph': self.graph})

    def _new_id(self, prefix):
        identifiers = {item['id'] for scope in [self.graph, *self.graph['subgraphs']]
                       for key in ('nodes', 'edges') for item in scope[key]}
        number = self.revision + 1
        while prefix + str(number) in identifiers:
            number += 1
        return prefix + str(number)

    def _recovery_entry(self, scope, node, result):
        if self.propose_recovery is None or node['type'] not in {'skill', 'router', 'end'}:
            raise UnhandledSkillResult(node['id'], result)
        router_id = self._new_id('recovery_')
        prior = next((key for key, value in self.recoveries.items() if value['node']['id'] == node['id']), None)
        if prior is not None:
            router_id = prior
        else:
            self.recoveries[router_id] = {'node': deepcopy(node), 'history': [], 'result': deepcopy(result)}
        self.recoveries[router_id]['result'] = deepcopy(result)
        outcome = result['status'] if node['type'] == 'router' else result_status(result)
        edge = {'id': self._new_id('dynamic_edge_'), 'source': node['id'],
                'target': router_id, 'on': outcome}
        nodes = [] if prior else [{'id': router_id, 'type': 'recovery', 'failed_node_id': node['id']}]
        self._patch(scope, nodes, [edge], source='runtime_control', evidence={
            'reason': 'unhandled_outcome_requests_decisions', 'result': result})
        return edge

    async def _recovery_choice(self, scope, router):
        state = self.recoveries[router['id']]
        if state.get('pending_node'):
            identifier = state.pop('pending_node')
            state['history'].append({'node': next(node for node in scope['nodes'] if node['id'] == identifier),
                                     'resolved_arguments': deepcopy(self.resolved_arguments.get(identifier)),
                                     'status': result_status(self.results[identifier]),
                                     'result': deepcopy(self.results[identifier])})
        proposal = await self.propose_recovery(deepcopy(state['node']), deepcopy(state['result']),
                                               deepcopy(state['history']), deepcopy(self.bindings))
        if not isinstance(proposal, dict):
            raise UnhandledSkillResult(state['node']['id'], state['result'])
        action = proposal.get('action')
        nodes, edges = [], []
        if action == 'retry':
            if state['node']['type'] not in {'router', 'end'} and state['node']['skill'] not in self.repeatable_skills:
                raise ValueError('recovery_attempt_not_repeatable:' + state['node']['skill'])
            target = state['node']['id']
            state['history'].append({'retry_node': target, 'previous_result': deepcopy(self.last_result)})
        elif action == 'use_recovery_result':
            latest = state['history'][-1] if state['history'] else {}
            recovered_node = latest.get('node', {})
            original = state['node']
            recovered = latest.get('result', {})
            ignored = self.reusable_result_ignored_arguments.get(original.get('skill'), ())
            if (original['type'] != 'skill' or original['skill'] not in self.repeatable_skills
                    or original['skill'] not in self.reusable_result_skills
                    or recovered_node.get('id') != proposal.get('source_node_id')
                    or recovered_node.get('skill') != original['skill']
                    or not same_result_arguments(original['arguments'], recovered_node.get('arguments'), ignored)
                    or not same_result_arguments(self.bind(original['arguments']), latest.get('resolved_arguments'), ignored)
                    or latest.get('status') != 'success' or result_status(recovered) != 'success'):
                raise ValueError('recovery_result_not_reusable')
            success = next((edge for edge in scope['edges']
                            if edge['source'] == original['id'] and edge['on'] == 'success'), None)
            if success is None:
                raise ValueError('graph_success_edge_missing:' + original['id'])
            for name, field in self.result_bindings.get(original['skill'], {}).items():
                if recovered.get(field) is None or self.bindings.get(name) != recovered[field]:
                    raise ValueError('recovery_result_not_reusable:binding_changed')
            target = success['target']
            self.results[original['id']] = deepcopy(recovered)
            self.event('policy_recovery_result_reused', 'Decisions 采用已成功的恢复结果', {
                'failed_node_id': original['id'], 'source_node_id': recovered_node['id'],
                'original_success_edge_id': success['id'], 'evidence': proposal.get('evidence', {})})
        elif action == 'skill':
            target = self._new_id('recovery_action_')
            nodes.append({'type': 'skill', 'id': target, 'skill': proposal['skill'],
                          'arguments': deepcopy(proposal['arguments'])})
            state['pending_node'] = target
            for status in ('success', 'failure', 'unknown'):
                edges.append({'id': target + '_' + status, 'source': target, 'target': router['id'], 'on': status})
        else:
            target = self._new_id('recovery_end_')
            state['terminal_node'] = target
            nodes.append({'type': 'end', 'id': target, 'status': 'unknown',
                          'reason': proposal.get('reason', 'decisions_recovery_no_progress')})
        edge = {'id': self._new_id('decisions_edge_'), 'source': router['id'], 'target': target, 'on': 'selected'}
        edges.append(edge)
        self._patch(scope, nodes, edges, source='decisions', evidence=proposal.get('evidence', {}))
        return edge

    async def _scope(self, scope):
        identifier = scope['entry']
        while True:
            if self.check_cancelled is not None:
                self.check_cancelled()
            self.steps += 1
            if self.steps > self.step_limit:
                raise ValueError('policy_execution_budget_exceeded')
            node = next(node for node in scope['nodes'] if node['id'] == identifier)
            self.event('policy_node_started', 'node 开始', {'subgraph_id': scope['id'], 'node_id': identifier,
                'input_id': scope.get('input_id'), 'author': self.node_authors.get(identifier, 'responses')})
            selected = None
            kind = node['type']
            if kind == 'skill':
                if identifier in self.completed_actions:
                    raise ValueError('completed_action_replay_refused:' + identifier)
                arguments = self.bind(node['arguments'])
                self.resolved_arguments[identifier] = deepcopy(arguments)
                outputs = self.result_bindings.get(node['skill'], {})
                for name in outputs:
                    self.bindings.pop(name, None)
                result = await self.invoke(node['skill'], arguments)
                status = result_status(result)
                if status != 'success':
                    self.last_failure = {'node_id': identifier, **deepcopy(result)}
                if status == 'success':
                    for name, field in outputs.items():
                        if result.get(field) is None:
                            raise ValueError('declared_result_binding_missing:' + identifier + ':' + field)
                        self.bindings[name] = deepcopy(result[field])
                    if node['skill'] not in self.repeatable_skills:
                        self.completed_actions.add(identifier)
            elif kind == 'subgraph':
                child = next(child for child in self.graph['subgraphs'] if child['id'] == node['subgraph_id'])
                result = await self._subgraph(child)
                status = result_status(result)
            elif kind == 'router':
                verdict = await self.judge(node['condition'], dict(self.bindings)) if self.judge else None
                status = 'yes' if verdict is True else 'no' if verdict is False else 'unknown'
                result = {'status': status, 'success': verdict is not None}
            elif kind == 'recovery':
                selected = await self._recovery_choice(scope, node)
                status, result = 'selected', {'status': 'success', 'selected_edge': selected['id']}
            else:
                result = {'status': node['status'], 'success': node['status'] == 'success', 'reason': node['reason']}
                stopped = next((state for state in self.recoveries.values()
                                if state.get('terminal_node') == identifier), None)
                if stopped is not None:
                    result = {**deepcopy(stopped['result']), 'status': node['status'], 'success': False,
                              'recovery_stop_reason': node['reason']}
                status = node['status']
                if status == 'success' and scope.get('input_id') is not None and self.after_target:
                    checked = await self.after_target(scope['input_id'])
                    if isinstance(checked, dict) and result_status(checked) != 'success':
                        result = {**checked, 'status': 'unknown', 'success': False}
                        status = 'unknown'
            self.results[identifier] = deepcopy(result)
            for recovery in self.recoveries.values():
                if recovery['node']['id'] == identifier and status != 'success':
                    recovery['result'] = deepcopy(result)
            self.last_result = deepcopy(result)
            self.event('policy_node', 'node 完成', {'subgraph_id': scope['id'], 'node_id': identifier,
                'input_id': scope.get('input_id'), 'status': status, 'reason': result.get('reason')})
            if kind == 'end' and status == node['status']:
                return result
            if selected is None:
                selected = next((edge for edge in scope['edges'] if edge['source'] == identifier and edge['on'] == status), None)
            if selected is None:
                if status == 'success':
                    raise ValueError('graph_success_edge_missing:' + identifier)
                selected = self._recovery_entry(scope, node, result)
            self.event('policy_edge', 'edge 执行', {'subgraph_id': scope['id'], 'edge_id': selected['id'],
                'source': identifier, 'target': selected['target'], 'on': selected['on']})
            identifier = selected['target']

    async def _subgraph(self, scope):
        task = scope['task']
        if task is None:
            return await self._scope(scope)
        identifier = scope['input_id']
        targets = list(self.resolve_target_ids()) if self.resolve_target_ids else self.target_ids
        pending = self.is_target_pending is None or self.is_target_pending(identifier)
        if identifier not in targets or not pending:
            return {'status': 'success', 'success': True, 'skipped': True, 'input_id': identifier}
        outer_bindings = deepcopy(self.bindings)
        self.bindings = {key: value for key, value in self.bindings.items() if key not in self.output_names}
        self.bindings.update({'$target': identifier, '$source': task['source_ref'],
            '$destination': task['destination_ref'], '$work_area': task.get('work_area_ref')})
        try:
            if self.object_task_context:
                await self.object_task_context(task)
            if self.before_target:
                await self.before_target(identifier, targets.index(identifier))
            self.event('object_task_started', 'subgraph 开始', {'subgraph_id': scope['id'], 'input_id': identifier, 'task_spec': task})
            result = await self._scope(scope)
            if result_status(result) == 'success':
                self.event('object_task_completed', 'subgraph 完成', {'subgraph_id': scope['id'], 'input_id': identifier})
            else:
                self.failure_bindings = deepcopy(self.bindings)
            return result
        except Exception:
            self.failure_bindings = deepcopy(self.bindings)
            raise
        finally:
            self.bindings = outer_bindings
            if self.object_task_context:
                await self.object_task_context(None)
