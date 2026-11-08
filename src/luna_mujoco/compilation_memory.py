"""Persist compiler counterexamples independently of physical object memory."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .core.recovery_coverage import observation_recovery_outcomes

AUTHORING_CONSTRAINTS = {
    'graph_edge_unknown_node': 'Declare every local node before wiring edges. Copy each endpoint from that scope node list.',
    'unknown_subgraph': 'Name required scopes in caller interfaces, then supply every referenced body with the exact same ID.',
    'graph_unreachable_nodes': 'Walk from every scope entry and remove unused nodes when replacing a route.',
    'uncalled_subgraph': 'Every subgraph definition must be reachable through declared calls from the root.',
    'duplicate_graph_node_id': 'Node IDs must be unique across all scopes. Each node belongs to exactly one scope.',
    'duplicate_graph_edge_id': 'Use compact route groups. The compiler assigns edge IDs. Declare each source/outcome pair once.',
    'graph_route_invalid': 'Each route group needs distinct nonempty sources and outcomes, one local target and a positive integer max_traversals.',
    'graph_ambiguous_edge': 'Replace the existing (source,on) edge when repairing a route. Each pair has exactly one target.',
    'graph_edge_invalid_outcome': 'Use the SOURCE node outcome vocabulary. Predicate routers, choice routers and skills differ.',
    'graph_outcome_edge_missing': 'Wire every declared source outcome to an explicit continuation or end in the same scope.',
    'graph_scope_missing_return': 'Every subgraph reaches an explicit end. A final skill does not implicitly return.',
    'unbound_policy_reference': (
        'A result binding must exist on every incoming call path. $product requires a successful product producer. '
        'Selecting a new input clears previous derived bindings. Separate input and product recoveries when their bindings differ.'),
    'planning_precondition_not_satisfied': (
        'Trace requires, success effects and failure effects through every caller and recovery continuation. '
        'Restore the earliest invalidated prerequisite and all downstream dependencies. A router establishes no facts. '
        'Shared recovery must accept every incoming failure state, including failures before movement.'),
    'graph_observation_recovery_missing': (
        'For every normal-path outcome listed in required_observation_recovery, trace the failure edge and caller returns '
        'to declared move_to_observe and observe_scene. A single recovery route needs no Decisions selector. Establish their skill prerequisites '
        'and then restore the failed continuation. Recovery-only retries may stop on a second failure.'),
    'graph_recovery_reuses_exhausted_success_edge': (
        'A recovery returning before its failed checkpoint reuses the successful prefix. Budget those success edges '
        'for the retry only when those operations are repeatable. Share recovery bodies through distinct call nodes '
        'returning directly to their own continuations. A common resume router exposing earlier and later physical '
        'phases admits cross-phase replay paths. Separate those calls instead of increasing budgets on completed '
        'physical effects. Keep recovery-entry edges bounded to one attempt per scope invocation.'),
    'completion_fact_not_established': 'Every root success path must establish task verification and the required final posture.',
    'batch_requires_shared_subgraphs': 'Use shared input/product scopes for all/count tasks and bind each input through select_target.',
    'batch_requires_select_target': 'Select current inputs in a sensing loop. Initial target_refs do not close membership.',
    'batch_target_must_use_runtime_binding': 'Use $selected or the derived current/product binding for collection inputs.',
    'graph_no_possible_completion': 'Keep a legal normal path to verified root success while repairing failures.',
    'graph_revision_conflicting_scope_edits': 'Edit each scope once per revision. Replacement and removal of the same ID conflict.',
    'graph_revision_remove_unknown_scope': 'Remove only a currently defined scope. Repair its callers in the same revision.',
}

SEMANTIC_FIELDS = ('arguments', 'requires', 'provides', 'planning_effects', 'failure_effects',
                   'failure_preserves', 'outcomes', 'outcome_rules', 'argument_rules', 'result_bindings',
                   'invalidates_observation', 'retry_safe', 'processing_effect')


def catalog_fingerprint(catalog):
    def semantics(value):
        if isinstance(value, dict):
            return {key: semantics(item) for key, item in value.items() if key not in {'description', 'title'}}
        if isinstance(value, list):
            return [semantics(item) for item in value]
        return value

    contracts = {name: {key: semantics(contract[key]) for key in SEMANTIC_FIELDS if key in contract}
                 for name, contract in catalog.items()}
    return hashlib.sha256(json.dumps(contracts, sort_keys=True).encode()).hexdigest()


class CompilationMemory:
    """Dated compiler and execution failures for authoring. Writes are atomic and idempotent."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS compilation_failures (
                request_hash TEXT PRIMARY KEY,
                catalog_hash TEXT NOT NULL,
                observation_id TEXT,
                recorded_at TEXT NOT NULL,
                source_run TEXT,
                examples TEXT NOT NULL
            )''')
            connection.execute('''CREATE TABLE IF NOT EXISTS execution_failures (
                run_id TEXT PRIMARY KEY,
                catalog_hash TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                evidence TEXT NOT NULL
            )''')

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record_failure(self, *, catalog, request_hash, observation_id, issues, graph,
                       counterexamples=(), source_run=None):
        graph = graph if isinstance(graph, dict) else {}
        subgraphs = graph.get('subgraphs', [])
        nodes = {}
        for scope in [graph, *(subgraphs if isinstance(subgraphs, list) else [])]:
            if not isinstance(scope, dict) or not isinstance(scope.get('nodes'), list):
                continue
            for node in scope['nodes']:
                if isinstance(node, dict) and isinstance(node.get('id'), str):
                    nodes[node['id']] = node
        paths = {item['error']: item for item in counterexamples}
        examples = []
        for issue in issues:
            message = issue.get('message', '') if isinstance(issue, dict) else issue
            code, _, remainder = message.partition(':')
            code = code.removeprefix('Value error, ')
            node_id = remainder.split(':', 1)[0]
            if code == 'graph_observation_recovery_missing' and ':' in remainder:
                node_id = remainder.split(':')[1]
            node = nodes.get(node_id, {})
            example = {'code': code, 'error': message}
            if node.get('type') == 'skill':
                example.update(skill=node.get('skill'), arguments=node.get('arguments'))
            if message in paths:
                example['counterexample'] = paths[message]
            examples.append(example)
        if not examples:
            return
        with self.connection() as connection:
            connection.execute('INSERT OR IGNORE INTO compilation_failures VALUES (?, ?, ?, ?, ?, ?)',
                (request_hash, catalog_fingerprint(catalog), observation_id, datetime.now(UTC).isoformat(),
                 source_run, json.dumps(examples, ensure_ascii=False)))

    def record_execution_failure(self, *, catalog, run_id, report):
        skill = report.get('skill') or report.get('stage')
        if skill not in catalog:
            return
        if not run_id or not report.get('code'):
            raise ValueError('execution_memory_requires_run_and_failure')
        evidence = {'skill': skill, 'error_code': report['code'].partition(':')[0],
            'summary': report.get('summary'), 'observation_id': report.get('observation_id'),
            'diagnostic_checks': report.get('checks', []),
            'known_facts': report.get('known_facts', []), 'unknowns': report.get('unknowns', [])}
        with self.connection() as connection:
            connection.execute('INSERT OR IGNORE INTO execution_failures VALUES (?, ?, ?, ?)',
                (run_id, catalog_fingerprint(catalog), datetime.now(UTC).isoformat(),
                 json.dumps(evidence, ensure_ascii=False, allow_nan=False)))

    def snapshot(self, catalog):
        fingerprint = catalog_fingerprint(catalog)
        with self.connection() as connection:
            rows = connection.execute('''SELECT request_hash, observation_id, source_run, examples
                FROM compilation_failures WHERE catalog_hash = ? ORDER BY recorded_at DESC, request_hash''',
                (fingerprint,)).fetchall()
            total = connection.execute('SELECT COUNT(*) FROM compilation_failures').fetchone()[0]
            executions = connection.execute('''SELECT run_id, evidence FROM execution_failures
                WHERE catalog_hash=? ORDER BY recorded_at DESC, run_id''', (fingerprint,)).fetchall()
        lessons = {}
        for request_hash, observation_id, source_run, encoded in rows:
            for example in json.loads(encoded):
                code = example['code'].removeprefix('Value error, ')
                if (code == 'graph_observation_recovery_missing' and example.get('skill') in catalog
                        and not observation_recovery_outcomes(catalog[example['skill']])):
                    continue
                lesson = lessons.setdefault(code, {
                    'error_code': code, 'occurrences': 0,
                    'constraint': AUTHORING_CONSTRAINTS.get(code, 'Satisfy the current graph schema and registered contracts.'),
                    'examples': []})
                lesson['occurrences'] += 1
                if len(lesson['examples']) < 2:
                    lesson['examples'].append({**example, 'request_hash': request_hash,
                        'observation_id': observation_id, 'source_run': source_run})
        execution_lessons = {}
        for run_id, encoded in executions:
            evidence = json.loads(encoded)
            key = (evidence['skill'], evidence['error_code'])
            lesson = execution_lessons.setdefault(key, {'skill': key[0], 'error_code': key[1],
                                                       'occurrences': 0, 'examples': []})
            lesson['occurrences'] += 1
            if len(lesson['examples']) < 2:
                lesson['examples'].append({'source_run': run_id, **evidence})
        return {'source': 'persisted_compiler_validation_failures', 'graph_version': 2,
                'catalog_hash': fingerprint, 'matching_failed_requests': len(rows),
                'excluded_other_contract_requests': total - len(rows),
                'execution_failures': {'matching_runs': len(executions),
                    'use': 'Historical terminal skill failures. Inspect relevant prerequisites and authored '
                           'recovery routes. Historical IDs and measurements are not current scene facts. '
                           'A failure does not prove that any alternative recovery succeeds.',
                    'lessons': sorted(execution_lessons.values(),
                                      key=lambda row: (-row['occurrences'], row['skill'], row['error_code']))},
                'use': ('Apply these constraints while composing every scope. Examples are rejected historical '
                        'fragments, not executable plans or current object facts. Current skill contracts are authoritative.'),
                'lessons': sorted(lessons.values(), key=lambda row: (-row['occurrences'], row['error_code']))}
