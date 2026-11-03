"""Shared execution state, bindings and errors; no policy traversal."""
from copy import deepcopy

from ..skills import validate_result_bindings


class PolicyStopped(ValueError):
    code = 'policy_stopped'

    def __init__(self, reason, last_result):
        cause = last_result.get('reason') if isinstance(last_result, dict) else None
        super().__init__('policy_stopped:' + reason + (':' + str(cause) if cause else ''))
        self.details = {'policy_reason': reason, 'last_result': deepcopy(last_result)}


class UnhandledSkillResult(ValueError):
    code = 'unhandled_skill_result'

    def __init__(self, identifier, result):
        reason = result.get('reason')
        super().__init__('unhandled_skill_result:' + identifier + (':' + str(reason) if reason else ''))
        self.details = {'node_id': identifier, 'result': deepcopy(result)}


def result_status(result):
    if not isinstance(result, dict):
        raise TypeError('invalid_skill_result')
    if result.get('status') == 'unknown':
        return 'unknown'
    if result.get('success') is False or result.get('status') in {'failure', 'failed', 'rejected'}:
        return 'failure'
    return 'success'


class ExecutionState:
    def __init__(self, invoke, *, target_ids, bindings, judge=None, before_target=None,
                 after_target=None, is_target_pending=None, check_cancelled=None, emit=None, step_limit=4096,
                 result_bindings=None, resolve_target_ids=None, object_task_context=None):
        self.invoke = invoke
        self.target_ids = list(target_ids)
        self.resolve_target_ids = resolve_target_ids
        self.object_task_context = object_task_context
        self.failure_bindings = None
        if len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError('duplicate_policy_target')
        self.bindings = dict(bindings)
        self.result_bindings = {name: dict(validate_result_bindings(value))
                                for name, value in (result_bindings or {}).items()}
        self.output_names = {name for outputs in self.result_bindings.values() for name in outputs}
        self.judge = judge
        self.before_target = before_target
        self.after_target = after_target
        self.is_target_pending = is_target_pending
        self.check_cancelled = check_cancelled
        self.emit = emit
        self.step_limit = step_limit
        self.steps = 0
        self.results = {}
        self.unhandled = None
        self.last_result = None

    def bind(self, value):
        if isinstance(value, str) and value.startswith('$'):
            if value not in self.bindings or self.bindings[value] is None:
                raise ValueError('unbound_policy_reference:' + value)
            return deepcopy(self.bindings[value])
        if isinstance(value, dict):
            return {key: self.bind(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.bind(item) for item in value]
        return value
