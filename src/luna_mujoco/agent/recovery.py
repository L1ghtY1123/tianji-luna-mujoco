"""Decisions selects a registered capability and typed arguments, never arbitrary code."""
import json
from copy import deepcopy
from itertools import product

from jsonschema import Draft202012Validator

from ..skills import argument_contract, same_result_arguments
from .decisions import ChoiceRequest
from .visibility_recovery import visibility_allows_motion, visibility_assessment, visibility_requests


def recovery_measurements(context):
    """Separate latest measured geometry from the published collision map."""
    observation = context.get('observation', {})
    scene = context.get('scene', {})
    return {'observation_id': observation.get('observation_id'),
        'capture_id': context.get('latest_capture', {}).get('observation_id'),
        'measured_entities': [{'id': item['id'], 'kind': item.get('kind'), 'position': item.get('position'),
                              'placement_geometry_issue': item.get('placement_geometry_issue')}
                              for item in observation.get('objects', [])],
        'missing_geometry_ids': observation.get('visual_inventory', {}).get('missing_geometry_ids', []),
        'geometry_failures': [{key: row[key] for key in ('instance_id', 'observation_id', 'reason')}
                              for row in observation.get('geometry_failures', [])],
        'map_observation_id': scene.get('observation_id'),
        'map_coverage': scene.get('coverage', {}),
        'map_entities': scene.get('inventory_map', {}).get('entities', {}),
        'map_mode': scene.get('exclusion_state'),
        'map_refresh_required': context.get('scene_refresh_required', False),
        'mapping_lifecycle': context.get('mapping_lifecycle', {}),
        'pending_transfer': context.get('pending_transfer')}


def parameter_values(schema, *, bindings, entity_ids):
    """Enumerate permitted values, not a developer-authored recovery sequence."""
    if 'enum' in schema:
        values = list(schema['enum'])
    elif 'const' in schema:
        values = [schema['const']]
    elif 'anyOf' in schema:
        values = [value for branch in schema['anyOf'] for value in parameter_values(
            branch, bindings=bindings, entity_ids=entity_ids)]
    elif schema.get('type') == 'null':
        values = [None]
    elif schema.get('type') == 'boolean':
        values = [True, False]
    elif schema.get('type') == 'string' and 'default' not in schema:
        values = [key for key, value in bindings.items() if value is not None] + list(entity_ids)
    elif schema.get('type') == 'array' and schema.get('items', {}).get('type') == 'string':
        values = [[], *[[identifier] for identifier in entity_ids], list(entity_ids)]
    elif 'default' in schema:
        values = [schema['default']]
    else:
        values = []  # No invented free-form metric values.
    unique = {json.dumps(value, sort_keys=True): value for value in values
              if Draft202012Validator(schema).is_valid(value)}
    if 'default' in schema and Draft202012Validator(schema).is_valid(schema['default']):
        unique.setdefault(json.dumps(schema['default'], sort_keys=True), schema['default'])
    return list(unique.values())


def capability_domains(catalog, facts, bindings, entity_ids):
    result = {}
    for name, contract in catalog.items():
        if any(facts.get(key) != expected for key, expected in contract.get('requires', {}).items()):
            continue
        schema = contract['arguments']
        domains = {}
        for field, definition in schema.get('properties', {}).items():
            # Inline ordinary Pydantic definitions without changing the SDK.
            if '$ref' in definition:
                definition = schema.get('$defs', {}).get(definition['$ref'].split('/')[-1], definition)
            values = parameter_values(definition, bindings=bindings, entity_ids=entity_ids)
            if not values:
                if field in schema.get('required', []):
                    break
                continue
            domains[field] = values
        else:
            result[name] = domains
    return result


class RecoveryPlanner:
    """Build one graph extension per model decision, then reconsider with new evidence."""
    def __init__(self, judge, catalog, validate_arguments=None):
        self.judge = judge
        self.catalog = catalog
        self.validate_arguments = validate_arguments

    async def propose(self, node, result, history, bindings, *, facts, inventory, context):
        from .completion import required_observation_return

        required_return = required_observation_return(
            context.get('completion_requirements', {}), result.get('reason'))
        domains = capability_domains(self.catalog, facts, bindings, inventory)
        if required_return and 'viewpoint' in domains.get('move_to_observe', {}):
            domains['move_to_observe']['viewpoint'] = ['configured']
        original_skill = node.get('skill')
        retry = node.get('type') in {'router', 'end'} or bool(self.catalog.get(original_skill, {}).get('recovery_retry_safe'))
        original_contract = argument_contract(self.catalog.get(original_skill, {}), node.get('arguments', {}))
        retry_conflicts = {key: {'required': expected, 'actual': facts.get(key)}
            for key, expected in original_contract.get('requires', {}).items() if facts.get(key) != expected}
        retry = retry and not retry_conflicts
        original_domains = domains.get(original_skill, {})
        alternative_arguments = any(len(values) > 1 for values in original_domains.values())
        if not retry or not alternative_arguments:
            domains.pop(original_skill, None)  # Cannot disguise an unsafe retry as a new node.
        combinations = {}
        rejected = context.get('rejected_proposals', [])
        for name, fields in domains.items():
            blocked = [row.get('arguments') for row in rejected if row.get('skill') == name]
            combinations[name] = []
            if None in blocked:
                continue  # A rejected capability without arguments applies to all its variants.
            for values in product(*fields.values()):
                arguments = dict(zip(fields, values))
                if self.validate_arguments is not None:
                    try:
                        self.validate_arguments(name, arguments)
                    except ValueError:
                        continue
                contract = argument_contract(self.catalog[name], arguments)
                if arguments not in blocked and all(facts.get(key) == expected
                        for key, expected in contract.get('requires', {}).items()):
                    combinations[name].append(arguments)
        domains = {name: fields for name, fields in domains.items() if combinations[name]}
        retry = retry and not any(row.get('reason') == 'identical_evidence_and_operation_no_progress'
                                 and not row.get('skill') for row in rejected)
        choices = (*domains, *(('retry_failed_node',) if retry else ()), 'stop', 'unknown')
        state = {**context, 'verification_scope': 'graph_recovery', 'failed_node': node,
                 'failed_result': result, 'recovery_history': history, 'bindings': bindings,
                 'retry_precondition_conflicts': retry_conflicts,
                 'facts': facts, 'available_capabilities': {name: self.catalog[name] for name in domains},
                 'parameter_domains': domains}
        evidence = {}
        latest = history[-1] if history else {}
        recovered = latest.get('result', {})
        reusable = context.get('reusable_recovery_result', {})
        reuse = (retry and node.get('type') == 'skill'
                 and self.catalog.get(original_skill, {}).get('recovery_result_reusable')
                 and latest.get('node', {}).get('skill') == original_skill
                 and same_result_arguments(node.get('arguments'), latest.get('node', {}).get('arguments'),
                     self.catalog.get(original_skill, {}).get('recovery_result_ignored_arguments', ()))
                 and reusable.get('node_id') == latest.get('node', {}).get('id')
                 and reusable.get('observation_id') is not None
                 and reusable.get('observation_id') == recovered.get('observation_id')
                 and reusable.get('generation') == recovered.get('generation'))
        if retry and (latest.get('status') == 'success' or recovered.get('status') == 'success'
                      or recovered.get('success') is True):
            continuation = ChoiceRequest(name='recovery_continuation', selection_mode='proposal',
                choices=(*(('use_recovery_result',) if reuse else ()),
                         'retry_failed_node', 'select_recovery_capability', 'stop', 'unknown'),
                instructions=(f'The original node is {node.get("id")} ({original_skill or node.get("type")}); '
                    f'its recorded failure is {result.get("reason", result.get("status"))}. '
                    'The latest recovery capability succeeded. Decide whether to resume the '
                    'original graph. When offered, use_recovery_result adopts the already successful '
                    'result of the SAME skill with contract-equivalent result arguments and current observation evidence, '
                    'Acquisition-only capture_mode may differ when the contract explicitly permits it; '
                    'the already successful current observation is unchanged. '
                    'then follows the original success edge without executing that skill again. '
                    'Its original failure and the successful recovery remain separately archived. '
                    'Choose it when that completed operation already satisfies the failed node. '
                    'Otherwise re-evaluate the failed node against the CURRENT context. '
                    'retry_failed_node re-runs only this repeat-safe skill, router question or end-node '
                    'completion check, never the whole subgraph or earlier physical actions, with all '
                    'normal checks; its success follows the original success edge. The historical '
                    'failed_result is not a fresh failure. Choose retry_failed_node if the latest '
                    'result repaired its prerequisites or supplied the missing evidence. Choose '
                    'select_recovery_capability if another repair is needed, stop if no useful '
                    'progress is possible, or unknown to abstain. This does not certify success '
                    'or bypass any numerical or identity check.'))
            decision = await self.judge([continuation], {**context,
                'verification_scope': 'graph_recovery_continuation', 'failed_node': node,
                'historical_failed_result': result, 'latest_recovery': latest, 'facts': facts,
                'bindings': bindings})
            evidence['continuation'] = decision
            answer = decision['judgments']['recovery_continuation']
            choice = answer['value'] if answer['status'] == 'proposed' else 'unknown'
            if choice == 'use_recovery_result' and reuse:
                return {'action': 'use_recovery_result', 'source_node_id': reusable['node_id'],
                        'evidence': evidence}
            if choice == 'retry_failed_node':
                return {'action': 'retry', 'evidence': evidence}
            if choice != 'select_recovery_capability':
                return {'action': 'stop', 'reason': 'decisions_recovery_' + choice, 'evidence': evidence}
        request = ChoiceRequest(name='recovery_capability', selection_mode='proposal', choices=choices,
            instructions=(f'The failed node is {node.get("id")} ({original_skill or node.get("type")}); '
                f'its recorded failure is {result.get("reason", result.get("status"))}. '
                'Choose a capability that repairs this node, not an unrelated later task step. '
                'Choose the next registered capability to recover this failed/unknown node, or '
                'retry_failed_node ONLY after evidence/prerequisites changed enough to warrant retry. '
                'You may construct a new multi-step recovery by choosing one capability now, seeing its '
                'result, then choosing the next. The initial graph need not contain this branch. '
                'failed_result records the ORIGINAL failed evaluation, not a new evaluation after recovery. '
                'After successful evidence acquisition or prerequisite repair, consider retry_failed_node '
                'to evaluate the repaired situation; do not treat that historical failure as a fresh failure. '
                'current_measurements separates the latest measured entities from published map layers. '
                'A missing map layer does not imply missing current geometry. New observation does not '
                'publish a new map; compare both observation IDs and entity sets before choosing. '
                'Use inventory memory, current geometry, robot pose and prior attempts. Visibility unknown '
                'is not object absence. retry_precondition_conflicts lists the failed node requirements '
                'still missing; select a capability whose declared effects can repair those requirements. '
                'Rebuilding the same measured scene cannot repair missing placement support geometry or '
                'cad_support_not_horizontal: that accepted pose lacks a horizontal declared floor. '
                'If a new view could resolve the task, choose a legal viewpoint movement after the '
                'observation map is available, then observe. A map/prerequisite failure does not itself '
                'mean a new view is needed. '
                'An occluded/offscreen subject needs a different legal viewpoint; '
                'unchanged-view photography is not such a movement. Do not repeat completed transfers, '
                'chase a favorable confidence by repeating the same question, or relax original goals. '
                'Choose stop if no offered capability can make useful safe progress, unknown to abstain.'))
        requests = [request]
        if required_return:
            state['return_obligation'] = (
                'The user requires a configured observation-pose return and inventory observation after '
                'this verified delivery, before processing. This is an unmet action-order obligation, '
                'not a visibility problem. Choose the legal required return/observation capability or '
                'its missing map prerequisite; if the configured pose is already reached, observe there. '
                'Do not claim the receipt or resume processing until it has been recorded.')
        if ('move_to_observe' in domains and not required_return) or original_skill == 'verify_place':
            requests[0:0] = visibility_requests()
            state['viewpoint_rule'] = ('move_to_observe is eligible only when recovery_visibility confirms '
                'not_visible, cropped or occluded. CAD/identity ambiguity, service and planning failures '
                'alone do not justify moving. Other capabilities remain independently selectable.')
        if original_skill == 'verify_place':
            state['release_verification_scope'] = (
                'verify_place judges current RGB, approximate target proximity and gripper retention. '
                'It does not consume SAM3, OBB, CAD pose or a newly published map. Reanalyzing the same '
                'RGBD with observe_scene cannot improve this visual answer. If visibility is sufficient '
                'but the visual answer is uncertain, do not request geometry or repeat that unchanged '
                'visual question. If an actual visual limitation requires moving, current geometry may '
                'be needed to plan that movement; choose legal prerequisite repairs before the movement. '
                'Never replace a visual verdict with a geometry result.')
        decision = await self.judge(requests, state)
        answer = decision['judgments']['recovery_capability']
        name = answer['value'] if answer['status'] == 'proposed' else 'unknown'
        evidence['selection'] = decision
        visibility = visibility_assessment(decision.get('judgments', {}))
        if (original_skill == 'verify_place' and name == 'observe_scene'
                and not visibility_allows_motion(visibility)):
            return {'action': 'rejected', 'skill': name, 'reason': 'rgb_release_does_not_require_geometry',
                    'evidence': evidence}
        if name == 'move_to_observe' and not required_return and not visibility_allows_motion(visibility):
            return {'action': 'rejected', 'skill': name, 'reason': 'viewpoint_requires_visual_limitation',
                    'evidence': evidence}
        if name not in choices:
            return {'action': 'stop', 'reason': 'recovery_selection_outside_domain', 'evidence': evidence}
        if name == 'retry_failed_node':
            return {'action': 'retry', 'evidence': evidence}
        if name not in domains:
            return {'action': 'stop', 'reason': 'decisions_recovery_' + name, 'evidence': evidence}
        arguments = {}
        remaining = combinations[name]
        ambiguous_fields = sum(len({json.dumps(row[field], sort_keys=True) for row in remaining}) > 1
                               for field in domains[name])
        if ambiguous_fields > 1 and 1 < len(remaining) <= 63:
            # One bounded choice preserves cross-field constraints and avoids
            # a separate network round trip for every independent parameter.
            options = {f'arguments_{index}': row for index, row in enumerate(remaining)}
            question = ChoiceRequest(name='recovery_arguments', choices=(*options, 'unknown'),
                selection_mode='proposal', instructions=(f'Choose one complete legal argument set for {name}. '
                    'argument_options contains typed values checked against its schema and current prerequisites. '
                    'Use the original task and current evidence to choose; no option certifies geometry safety. '
                    'Choose unknown when none is appropriate.'))
            selected = await self.judge([question], {**state, 'chosen_capability': name,
                                                    'argument_options': options})
            evidence['parameters'] = [selected]
            answer = selected['judgments'][question.name]
            if answer['status'] != 'proposed' or answer['value'] not in options:
                return {'action': 'stop', 'reason': 'recovery_parameter_unknown:arguments', 'evidence': evidence}
            remaining = [deepcopy(options[answer['value']])]
        for index, field in enumerate(domains[name]):
            values = []
            for candidate in remaining:
                if candidate[field] not in values:
                    values.append(candidate[field])
            if len(values) == 1:
                arguments[field] = deepcopy(values[0])
                continue
            options = {f'value_{number}': value for number, value in enumerate(values)}
            if len(values) > 63:
                return {'action': 'stop', 'reason': 'recovery_parameter_domain_too_large', 'evidence': evidence}
            question = ChoiceRequest(name='argument_' + str(index), choices=(*options, 'unknown'),
                selection_mode='proposal', instructions=(f'For chosen recovery capability {name}, select parameter {field}. '
                'Use its declared meaning and current facts. Options: ' + json.dumps(options, ensure_ascii=False) +
                '. Do not infer hidden geometry or waive preconditions. Parameter schema: ' +
                json.dumps(self.catalog[name]['arguments']['properties'][field], ensure_ascii=False)))
            selected = await self.judge([question], {**state, 'chosen_capability': name,
                                                    'selected_arguments': deepcopy(arguments)})
            evidence.setdefault('parameters', []).append(selected)
            answer = selected['judgments'][question.name]
            if answer['status'] != 'proposed' or answer['value'] not in options:
                return {'action': 'stop', 'reason': 'recovery_parameter_unknown:' + field, 'evidence': evidence}
            arguments[field] = deepcopy(options[answer['value']])
            remaining = [candidate for candidate in remaining if candidate[field] == arguments[field]]
        Draft202012Validator(self.catalog[name]['arguments']).validate(arguments)
        contract = argument_contract(self.catalog[name], arguments)
        conflicts = [key for key, expected in contract.get('requires', {}).items() if facts.get(key) != expected]
        if conflicts:
            return {'action': 'rejected', 'reason': 'recovery_precondition_not_satisfied:' + ','.join(conflicts),
                    'evidence': evidence, 'skill': name, 'arguments': arguments}
        return {'action': 'skill', 'skill': name, 'arguments': arguments, 'evidence': evidence}
