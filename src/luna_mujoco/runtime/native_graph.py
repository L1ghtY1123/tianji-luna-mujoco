"""Execute explicit edges and objects-owned scopes, without changing model-authored topology."""
from copy import deepcopy

from ..core.decisions import GraphDecisionRequest, GraphDecisionResult
from ..core.graph import Graph
from ..core.outcomes import GraphSkillResult
from .diagnostics import result_details
from .execution_state import ExecutionState, PolicyStopped, UnhandledSkillResult, result_status


class NativeGraphExecutor(ExecutionState):
    def __init__(self, *args, repeatable_skills=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.repeatable_skills = set(repeatable_skills)
        self.graph = None
        self.nonrepeatable_invocations = set()
        self.call_path = []
        self.last_failure = None
        self.unresolved_failures = {}

    def event(self, kind, title, data):
        if self.emit is not None:
            self.emit(kind, title, {'graph_id': self.graph['id'], **deepcopy(data)})

    async def run(self, graph):
        self.graph = Graph.model_validate(graph.model_dump() if hasattr(graph, 'model_dump') else graph).model_dump()
        result = await self._scope(self.graph)
        if result_status(result) != 'success':
            raise PolicyStopped(result.get('reason', 'graph_end_failure'), self.last_failure or result)
        return deepcopy(self.results)

    async def _scope(self, scope):
        identifier = scope['entry']
        traversals = {}
        # All failures inside a successfully recovered subgraph are resolved together.
        scope_key = (*self.call_path, scope['id'], scope.get('input_id'))
        while True:
            if self.check_cancelled is not None:
                self.check_cancelled()
            self.steps += 1
            if self.steps > self.step_limit:
                raise ValueError('policy_execution_budget_exceeded')
            node = next(node for node in scope['nodes'] if node['id'] == identifier)
            self.event('policy_node_started', 'node 开始', {'subgraph_id': scope['id'], 'node_id': identifier,
                'input_id': scope.get('input_id'),
                'author': 'program' if scope['id'] == 'terminal_failure_return' else 'responses'})
            kind = node['type']
            if kind == 'skill':
                # Distinct authored calls may reuse a physical operation. A loop
                # back through the same call path must not replay that operation.
                invocation = (self.bindings.get('$current', self.bindings.get('$selected', self.bindings.get('$target'))),
                              tuple(self.call_path), identifier)
                if invocation in self.nonrepeatable_invocations:
                    raise ValueError('completed_action_replay_refused:' + identifier)
                arguments = self.bind(node['arguments'])
                if node['skill'] not in self.repeatable_skills:
                    self.nonrepeatable_invocations.add(invocation)
                outputs = self.result_bindings.get(node['skill'], {})
                cleared = self.output_names if '$selected' in outputs else outputs
                for name in cleared:
                    self.bindings.pop(name, None)
                outcome = await self.invoke(node['skill'], arguments)
                if not isinstance(outcome, GraphSkillResult):
                    raise TypeError('typed_graph_skill_result_required')
                result = outcome.payload()
                status = outcome.status
                if status in {'failure', 'unknown'}:
                    self.last_failure = {**deepcopy(result), 'node_id': identifier,
                                         'skill': node['skill'], 'arguments': deepcopy(arguments)}
                    key = (scope_key, node['skill'], repr(sorted(arguments.items())))
                    self.unresolved_failures.setdefault(key, deepcopy(self.last_failure))
                if status == 'success':
                    self.unresolved_failures.pop((scope_key, node['skill'], repr(sorted(arguments.items()))), None)
                    if (self.last_failure and self.last_failure.get('skill') == node['skill']
                            and self.last_failure.get('arguments') == arguments):
                        self.last_failure = None
                    for name, field in outputs.items():
                        if result.get(field) is None:
                            raise ValueError('declared_result_binding_missing:' + identifier + ':' + field)
                        self.bindings[name] = deepcopy(result[field])
            elif kind == 'subgraph':
                child = next(child for child in self.graph['subgraphs'] if child['id'] == node['subgraph_id'])
                self.call_path.append(identifier)
                try:
                    result = await self._subgraph(child)
                finally:
                    self.call_path.pop()
                status = result_status(result)
            elif kind == 'router':
                if self.judge is None:
                    raise ValueError('policy_judge_missing')
                choices = tuple(node.get('choices') or ())
                branches = []
                if choices:
                    branches = [{'outcome': edge['on'], 'target': next(item for item in scope['nodes']
                                if item['id'] == edge['target']), 'remaining_traversals': max(0,
                                edge['max_traversals'] - traversals.get(edge['id'], 0))}
                                for edge in scope['edges'] if edge['source'] == identifier]
                request = GraphDecisionRequest(condition=node['condition'], rule=node.get('rule', 'visual'), bindings=deepcopy(self.bindings),
                    choices=choices, branches=deepcopy(branches), last_result=deepcopy(self.last_result),
                    last_failure=deepcopy(self.last_failure))
                judgment = await self.judge(request)
                if not isinstance(judgment, GraphDecisionResult):
                    raise TypeError('typed_graph_decision_result_required')
                verdict = judgment.payload()
                if choices:
                    if verdict.get('status') not in {'proposed', 'unknown'}:
                        raise ValueError('invalid_policy_judgment:' + identifier)
                    status = verdict.get('value') if verdict['status'] == 'proposed' else 'unknown'
                    if status not in choices:
                        raise ValueError('undeclared_recovery_choice:' + identifier)
                else:
                    if verdict.get('status') not in {'yes', 'no', 'unknown'}:
                        raise ValueError('invalid_policy_judgment:' + identifier)
                    status = verdict['status']
                result = {**deepcopy(verdict), 'success': status != 'unknown'}
                if status == 'unknown' and self.last_failure is None:
                    self.last_failure = {'node_id': identifier, **deepcopy(result)}
            else:
                result = {'status': node['status'], 'success': node['status'] == 'success', 'reason': node['reason']}
                status = node['status']
                if status == 'success' and scope.get('input_id') is not None and self.after_target:
                    checked = await self.after_target(self.bind(scope['input_id']))
                    if isinstance(checked, dict) and result_status(checked) != 'success':
                        result = {**checked, 'status': 'unknown', 'success': False}
                        status = 'unknown'
            self.results[identifier] = deepcopy(result)
            self.last_result = deepcopy(result)
            outcome = {}
            if status in {'failure', 'unknown'}:
                outcome = {'producer': 'skill:' + node['skill'] if kind == 'skill' else
                    'decisions_router' if kind == 'router' else 'subgraph_return' if kind == 'subgraph' else
                    'owned_completion_receipt_check' if status != node['status'] else 'responses_declared_end',
                    'reason': result.get('reason'), 'details': result_details(result)}
                if result.get('required_evidence'):
                    outcome['required_evidence'] = result['required_evidence']
            self.event('policy_node', 'node 完成', {'subgraph_id': scope['id'], 'node_id': identifier,
                'input_id': scope.get('input_id'), 'status': status, 'reason': result.get('reason'),
                **({'outcome_evidence': outcome} if outcome else {}),
                **({'judgment': deepcopy(result)} if kind == 'router' else {})})
            if kind == 'end' and status == node['status']:
                if status == 'success':
                    for key in tuple(self.unresolved_failures):
                        if key[0] == scope_key:
                            self.unresolved_failures.pop(key)
                return result
            selected = next((edge for edge in scope['edges']
                             if edge['source'] == identifier and edge['on'] == status), None)
            if selected is None:
                raise UnhandledSkillResult(identifier, result)
            count = traversals.get(selected['id'], 0)
            if count >= selected['max_traversals']:
                failure = {'status': 'failure', 'success': False,
                    'reason': 'graph_edge_budget_exhausted', 'error_code': 'graph_edge_budget_exhausted',
                    'target_id': self.bindings.get('$target'),
                    'details': {'scope_id': scope['id'], 'edge_id': selected['id'],
                                'traversals': count, 'max_traversals': selected['max_traversals']}}
                self.last_failure = failure
                self.event('policy_scope_exhausted', '恢复分支次数已耗尽', failure)
                return failure
            traversals[selected['id']] = count + 1
            self.event('policy_edge', 'edge 执行', {'subgraph_id': scope['id'], 'edge_id': selected['id'],
                'source': identifier, 'target': selected['target'], 'on': selected['on'],
                'traversal': count + 1, 'max_traversals': selected['max_traversals']})
            identifier = selected['target']

    async def _subgraph(self, scope):
        task = self.bind(scope['task'])
        if task is None:
            return await self._scope(scope)
        identifier = self.bind(scope['input_id'])
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
                await self.object_task_context(deepcopy(task))
            if self.before_target:
                await self.before_target(identifier, targets.index(identifier))
            self.event('object_task_started', 'subgraph 开始', {'subgraph_id': scope['id'], 'input_id': identifier, 'task_spec': task})
            result = await self._scope({**scope, 'input_id': identifier})
            if result_status(result) == 'success':
                self.event('object_task_completed', 'subgraph 完成', {'subgraph_id': scope['id'], 'input_id': identifier})
            return result
        finally:
            self.bindings = {key: self.bind(value) for key, value in outer_bindings.items()}
            if self.object_task_context:
                await self.object_task_context(None)
