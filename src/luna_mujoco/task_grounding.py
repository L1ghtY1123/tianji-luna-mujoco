"""Resolve natural-language task roles over permanently registered, currently measured IDs."""
import hashlib
import json

from .core.grounding import relation_refs
from .decision_conditions import parse_answers
from .raw_visual_input import capture_content
from .visual_api import VisualAPI, VisualDecisionError
from .visual_identity import choice


def current_subjects(grounding, group_ids, entities):
    subjects, masks, unresolved = {}, {}, []
    for group in grounding.groups:
        identifier = group_ids.get(group.group_id)
        if group.geometry_evidence.get('physical_status') == 'invalid':
            continue
        rows = [grounding.observation.candidates[i] for i in group.candidate_indices]
        if identifier is None:
            label = 'U' + str(len(unresolved) + 1)
            unresolved.append(label)
        else:
            label = 'ID' + identifier
            entity = entities[identifier]
            quotes = {r['scene_quote'] for r in rows if r.get('scene_quote')}
            subjects[identifier] = {key: entity[key] for key in (
                'kind', 'color', 'description', 'observed_support', 'processing_history',
                'last_motion_receipt', 'current_region', 'input_entity_id', 'output_entity_id') if key in entity}
            subjects[identifier].update(label=label, scene_quotes=sorted(quotes),
                center_world_m=group.geometry_evidence.get('visible_surface_center_world_m'),
                visible_bounds_world_m=group.geometry_evidence.get('full_visible_bounds_world_m'))
        for row in rows:
            masks.setdefault(row['camera_id'], {})[label] = row['silhouette_mask']
    return subjects, masks, unresolved


def choose_role(client, capture, selector, offered, subjects, images, state, records):
    rule = selector.selection
    collection = rule.binding_mode == 'each_selection'
    if len(offered) > 253:  # WireDecisionQuestion allows at most 255 choices including done and unknown.
        return None, 'too_many_registered_candidates_for_decisions'
    selected = []
    # A collection is enumerated with the same finite choice protocol as an individual.
    # Each accepted ID disappears from the next question. An explicit end checks coverage.
    while True:
        remaining = [i for i in offered if i not in selected]
        name = 'bind_' + hashlib.sha256(selector.ref.encode()).hexdigest()[:12] + '_' + str(len(selected))
        request_state = {**state, 'ref': selector.ref, 'role': rule.role,
            'user_phrase': selector.instruction_quote, 'declared_subject': selector.scene_quote,
            'description': selector.requested_attributes, 'appearance_constraints': list(selector.appearance_constraints),
            'selection': rule.model_dump(mode='json'), 'already_selected_ids': list(selected),
            'offered_ids_this_round': remaining}
        options = [{'value': i, 'description': 'CURRENT ID' + i + ': ' + (subjects[i].get('description') or subjects[i]['kind'])}
                   for i in remaining]
        options.append({'value': 'done', 'description': 'The comparison set is complete and no remaining offered ID qualifies.'})
        question = choice(name,
            f'本题只绑定角色 {rule.role}（{selector.ref}）：{selector.requested_attributes}。'
            f'对应的用户原话为「{selector.instruction_quote or selector.scene_quote}」。'
            '根据用户原话、当前标注图、已绑定的参照物及基座坐标，选择符合当前角色的实例 ID。'
            '只回答当前角色。来源和目的地可以是同类中的不同个体。图片标签 ID 后的数字仅为编号。'
            '机器人左侧对应 +Y，右侧对应 -Y。序数须考虑可见的完整比较集合。'
            '优先遵循用户的限定，包括容纳、支撑、相对位置、顺序、历史和外观。当前实测范围仅覆盖可见表面。'
            + ('这是集合。依次选择尚未选过的合格 ID，数目足够执行也要列出完整的当前合格集合，最后选 done。'
               if collection else '这是单个或第 m 个对象。有多个无法区分的合法答案时选 unknown。')
            + '确知没有剩余合格对象才选 done；遮挡、未登记候选或不确定性可能改变回答时选 unknown。',
            [*options, 'unknown'])
        if collection and not remaining:
            question = choice(name,
                f'集合「{selector.instruction_quote or selector.requested_attributes}」是否已列全？'
                f'同类已登记 ID 为 {offered}，前面已逐件确认 {selected}。当前没有剩余可选 ID。'
                '用场景段声明的数量、当前标注图和待定候选核对完整性。所有符合原话的物体均已列出选 done；'
                '确有可能漏掉符合条件的物体时选 unknown。背景及无关类别的未解释深度不影响此集合。',
                [{'value': 'done', 'description': 'All subjects of this specific user request are accounted for.'}, 'unknown'])
        content = [{'type': 'input_text', 'text': json.dumps(request_state, ensure_ascii=False)}, *images]
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': [question]}
        try:
            body, evidence = VisualAPI.request(client, payload, capture, 'task_role_binding',
                lambda body, evidence: (body, evidence))
            value = parse_answers(body, [question])[0][name]
        except VisualDecisionError as error:
            records.append({**error.evidence, 'ref': selector.ref, 'error': str(error)})
            return None, 'role_decision_unavailable'
        records.append({**evidence, 'ref': selector.ref, 'offered_ids': remaining,
                        'previously_selected': list(selected), 'selected_value': value, 'answers': body['answers']})
        if value == 'done':
            return (selected, None) if collection else (None, 'requested_subject_absent')
        if value not in remaining:
            return None, 'role_identity_unknown'
        selected.append(value)
        if not collection:
            return value, None


def bind_task_roles(client, capture, grounding, group_ids, entities, selectors, *, instruction,
                    bound_refs, initial, inventory=(), proposal_observation_id=None):
    if grounding.observation.observation_id != capture['observation_id'] or grounding.observation.generation != capture['generation']:
        raise ValueError('role_binding_capture_mismatch')
    subjects, masks, unresolved = current_subjects(grounding, group_ids, entities)
    state = {'user_instruction': instruction, 'observation_id': capture['observation_id'],
             'frame': 'Link_Base in metres: +X forward, +Y robot-left, +Z up',
             'registered_current_instances': subjects, 'unregistered_labels': unresolved,
             'coverage': grounding.observation.coverage}
    images = None
    identifiers, records, missing, membership = {}, [], [], {}
    pending = list(selectors)
    while pending:
        ready = [s for s in pending if relation_refs(s.selection.predicate) <= identifiers.keys()]
        if not ready:
            missing.extend({'proposal': {'ref': s.ref}, 'reason': 'unresolved_reference'} for s in pending)
            break
        for selector in ready:
            pending.remove(selector)
            rule = selector.selection
            previous = bound_refs.get(selector.ref)
            error = None
            if rule.binding_mode == 'once' and previous is not None:
                # Identity already selected by Decisions stays bound even after its position changes.
                value = previous if previous in subjects and subjects[previous]['kind'] == selector.kind else None
                if value is None:
                    error = 'bound_instance_not_observed'
            else:
                if images is None:
                    images = capture_content(capture, masks=masks, camera_order=('3', '2', '1'), modalities=('rgb',))
                hints = [p.model_dump(mode='json') for p in inventory if p.proposal_id in selector.proposal_refs
                         and proposal_observation_id == capture['observation_id']]
                role_state = {**state, 'bound_roles': dict(identifiers), 'unverified_responses_hints': hints}
                offered = [i for i, row in subjects.items() if row['kind'] == selector.kind]
                value, error = choose_role(client, capture, selector, offered, subjects, images, role_state, records)
                if not error and initial and rule.quantifier == 'count' and len(value) < rule.quantity:
                    error = 'requested_count_not_observed'
            if error:
                missing.append({'proposal': {'ref': selector.ref}, 'reason': error})
                continue
            identifiers[selector.ref] = value
            if rule.role == 'target':
                membership.update(dict.fromkeys(value if isinstance(value, list) else [value], 'required'))
    if not missing and any(s.selection.role == 'target' for s in selectors):
        for i in subjects:
            membership.setdefault(i, 'excluded')
    return {'identifiers': identifiers, 'membership': membership, 'unresolved_proposals': missing,
            'decision_records': records}
