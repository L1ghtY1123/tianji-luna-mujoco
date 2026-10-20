"""JEV choices through the official Pydantic AI TypeSafe adapter."""
import copy
import json
import math
import time

from pydantic import ConfigDict, Field, create_model
from pydantic_ai import Agent, Choices
from pydantic_ai.exceptions import ModelHTTPError

from .task_progress import original_source_bin, source_region_confirmed, source_region_geometry
from .task_request import normalize_request_scope, request_scope_questions

TASKS = {"grasp": "抓取物体并提起", "place": "放置已经握住的物体",
         "grasp_place": "抓取物体再放到指定位置或容器", "unsupported": "不属于以上任务"}
ATTRIBUTES = {
    "kind": {"cube": "方块/长方体", "sphere": "球", "cylinder": "圆柱",
             "any": "只说物体/东西，未限定形状", "unsupported": "其他形状或无法判断"},
    "color": {"black": "黑色", "red": "红色", "blue": "蓝色", "green": "绿色",
              "yellow": "黄色", "white": "白色", "orange": "橙色", "gray": "灰色", "any": "没有颜色限定",
              "unsupported": "其他颜色或无法判断"},
    "position": {"left": "左侧", "right": "右侧", "front": "前面/近处",
                 "back": "后面/远处",
                 "any": "用户没有描述目标物体的位置，只有任务、形状或颜色等要求；没有位置要求就选 any。",
                 "unsupported": "用户明确给出位置要求，但不能用单个左、右、前、后表示，"
                 "例如中间、两者之间、上方、否定位置，或含义不清的位置描述。未描述位置应选 any。"},
}
DESTINATION_ATTRIBUTES = {
    "kind": {"bin": "有底板、四壁及开放内腔的盒子/物料盒/容器，不是实心长方体",
             "any": "用户未限定目的容器类别", "unsupported": "其他目的地类别或无法判断"},
    "color": ATTRIBUTES["color"],
    "position": {**ATTRIBUTES["position"], "any": "用户没有限定目的容器的位置"},
}


class DecisionError(ValueError):
    pass


class DecisionLowConfidence(DecisionError):
    def __init__(self, field, selected, confidence, minimum_confidence):
        self.code = 'low_confidence:' + field
        self.details = {'field': field, 'selected': selected, 'confidence': confidence,
                        'minimum_confidence': minimum_confidence}
        super().__init__(self.code)


class DecisionInputTooLong(DecisionError):
    code = "jev_input_too_long"

    def __init__(self, error):
        super().__init__("JEV 单次输入过长，超过模型上下文上限；这不是余额不足。")
        self.details = {"status_code": error.status_code, "model_name": error.model_name,
                        "provider_error_type": "max_tokens_exceeded"}


def instance_decision_evidence(objects):
    """Expose text selection evidence, keeping images, clouds and private metadata out."""
    result = []
    for obj in objects:
        item = {key: copy.deepcopy(obj[key]) for key in (
            "id", "display_id", "kind", "color", "position", "visible_centroid_world",
            "estimated", "source", "confidence", "uncertainty_m", "coverage_fraction") if key in obj}
        for field, keys in (
            ("coverage", ("observed_surface_only", "camera_ids", "point_count", "limitations")),
            ("placement_reference", ("position_world_m", "floor_z_m", "source")),
            ("color_evidence", ("median_rgb", "source")),
        ):
            if field in obj:
                item[field] = {key: copy.deepcopy(obj[field][key]) for key in keys if key in obj[field]}
        if 'visual_semantics' in obj:
            item['visual_semantics'] = copy.deepcopy(obj['visual_semantics'])
        if 'measured_relationships' in obj:
            item['measured_relationships'] = copy.deepcopy(obj['measured_relationships'])
        result.append(item)
    return result


def arm_decision_evidence(evaluations, planning_settings=None):
    """Keep every candidate's selection evidence; planning matrices stay in the runner."""
    evidence = {}
    for arm, info in evaluations.items():
        if arm not in ("A", "B"):
            continue
        if info.get('feasibility_scope') in {'center_position_ik_only', 'offline_position_workspace'}:
            evidence[arm] = copy.deepcopy(info)
            continue
        if "candidate_columns" in info or "candidate_indices" in info or "candidate_sample_indices" in info:
            # A captured JEV state already contains the projection and can be replayed.
            evidence[arm] = info
            continue
        candidates = info["valid_candidates"]
        sampled = isinstance(candidates, dict)
        count = candidates["count"] if sampled else len(candidates)
        if sampled:
            candidates = candidates["first"]
        names = [f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)]
        start = [math.radians(value) for value in info["start_joints_deg"]]
        paired = info.get("feasibility_scope") == "paired_ik_only"
        rows = []
        for candidate in candidates:
            ik = candidate["ik"]
            joints = [ik["joints_rad"][ik["joint_names"].index(name)] for name in names]
            rows.append([candidate["candidate_index"], candidate["score"], math.dist(start, joints),
                         ik["position_error_m"], ik["rotation_error_rad"]])
            if planning_settings:
                for phase in ("approach", "lift"):
                    axis = "xyz".index(planning_settings[phase + "_axis"])
                    offset = planning_settings[phase + "_offset_m"]
                    if planning_settings[phase + "_in_tool_frame"]:
                        delta = [candidate["T_world_tcp"][row][axis] * offset for row in range(3)]
                    else:
                        delta = [0., 0., 0.]
                        delta[axis] = offset
                    rows[-1].append(delta)
            if paired:
                place_ik = candidate["destination_ik"]
                place_joints = [place_ik["joints_rad"][place_ik["joint_names"].index(name)] for name in names]
                rows[-1].extend([math.dist(joints, place_joints), place_ik["position_error_m"],
                                 place_ik["rotation_error_rad"]])
        value = {"tested_count": info["tested_count"], "valid_candidate_count": count,
                 "candidate_columns": ["candidate_index", "grasp_score", "joint_distance_rad",
                                       "position_error_m", "rotation_error_rad"],
                 "candidate_sample" if sampled else "candidates": rows}
        if planning_settings:
            value["candidate_columns"].extend(["approach_world_delta_m", "lift_world_delta_m"])
        if paired:
            value.update(feasibility_scope="paired_ik_only", destination_id=info["destination_id"],
                         complete_path_status="unknown")
            value["candidate_columns"].extend(["place_joint_distance_from_grasp_rad", "place_position_error_m",
                                               "place_rotation_error_rad"])
        elif info.get("destination_id") is not None:
            value.update(feasibility_scope="grasp_ik_only", destination_id=info["destination_id"],
                         place_planning_stage="after_grasp_attachment", complete_path_status="unknown")
        if not sampled:
            value["max_score"] = max((row[1] for row in rows), default=None)
            value["min_joint_distance_rad"] = min((row[2] for row in rows), default=None)
        evidence[arm] = value
    return evidence


def round_selection_value(value):
    """Display micrometer-scale evidence without changing algorithm inputs."""
    if isinstance(value, dict):
        return {key: round_selection_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_selection_value(item) for item in value]
    # Preserve tiny nonzero tolerances and signed residuals rather than displaying zero.
    return round(value, 6) if isinstance(value, float) and abs(value) >= 1e-6 else value


def project_arm_evidence(evidence, vectors, scores, *, skill_selection):
    """Keep planning choice evidence, summarizing residuals already accepted by IK."""
    result = copy.deepcopy(evidence)
    for info in result.values():
        for name in ("max_score", "min_joint_distance_rad"):
            if name in info:
                info[name] = round_selection_value(info[name])
        for name, value in info.get("candidate_ranges", {}).items():
            info["candidate_ranges"][name] = round_selection_value(value)
        rows_key = "candidates" if "candidates" in info else "candidate_sample"
        if rows_key not in info:
            continue  # A captured skill state may already contain only its summary.
        rows = info[rows_key]
        columns = info["candidate_columns"]
        for index, name in enumerate(columns):
            if name in ("grasp_score", "joint_distance_rad", "place_joint_distance_from_grasp_rad",
                        "approach_world_delta_m", "lift_world_delta_m"):
                for row in rows:
                    row[index] = round_selection_value(row[index])
        if scores and "grasp_score" not in columns:
            candidate_index = columns.index("candidate_index")
            for row in rows:
                row.insert(1, scores[str(row[candidate_index])])
            columns.insert(1, "grasp_score")
        ranges = dict(info.get("ik_error_ranges", {}))
        for name in ("position_error_m", "rotation_error_rad", "place_position_error_m", "place_rotation_error_rad"):
            if name not in columns:
                continue
            index = columns.index(name)
            values = [row[index] for row in rows]
            ranges[name] = [min(values), max(values)] if values else None
            columns.pop(index)
            for row in rows:
                row.pop(index)
        info["ik_error_ranges"] = ranges
        info["range_scope"] = "all_candidates" if rows_key == "candidates" else "sample_only"
        if skill_selection:
            info["candidate_indices" if rows_key == "candidates" else "candidate_sample_indices"] = [
                row[columns.index("candidate_index")] for row in rows]
            summary = {}
            for index, name in enumerate(columns):
                if name == "candidate_index":
                    continue
                values = [row[index] for row in rows]
                if name.endswith("world_delta_index"):
                    name = name.replace("_index", "_m")
                    values = [vectors[value] for value in values]
                if name.endswith("world_delta_m"):
                    summary[name] = [[min(v[axis] for v in values), max(v[axis] for v in values)]
                                     for axis in range(3)] if values else None
                else:
                    summary[name] = [min(values), max(values)] if values else None
            info["candidate_ranges"] = summary
            info.pop(rows_key)
            info.pop("candidate_columns")
        else:
            for index, name in enumerate(columns):
                if not name.endswith("world_delta_m"):
                    continue
                columns[index] = name.replace("_m", "_index")
                for row in rows:
                    vector = row[index]
                    if vector not in vectors:
                        vectors.append(vector)
                    row[index] = vectors.index(vector)
    return result


def share_candidate_scores(evidence):
    """Use one displayed score per GraspGen candidate only when both arms agree."""
    scores = {}
    for info in evidence.values():
        columns = info.get("candidate_columns", [])
        if "grasp_score" not in columns:
            continue
        for row in info.get("candidates", info.get("candidate_sample", [])):
            candidate = str(row[columns.index("candidate_index")])
            score = row[columns.index("grasp_score")]
            if candidate in scores and scores[candidate] != score:
                return {}  # Preserve arm-specific values if the supplied evidence disagrees.
            scores[candidate] = score
    if scores:
        for info in evidence.values():
            columns = info.get("candidate_columns", [])
            if "grasp_score" in columns:
                index = columns.index("grasp_score")
                columns.pop(index)
                for row in info.get("candidates", info.get("candidate_sample", [])):
                    row.pop(index)
    return scores


def share_failure_scores(attempts, scores, evidence):
    """Reference an existing displayed score; keep different or absent scores inline."""
    for attempt in attempts:
        arm = attempt.get("arm")
        info = evidence.get(arm, {})
        indices = info.get("candidate_indices")
        reference = f"arm_evaluations.{arm}.candidate_indices"
        if "candidates" in info:
            column = info["candidate_columns"].index("candidate_index")
            indices = [row[column] for row in info["candidates"]]
            reference = f"arm_evaluations.{arm}.candidates"
        for group in (attempt.get("details") or {}).get("failure_groups", []):
            rows = group.get("candidates")
            if rows and all(scores.get(str(row[0])) == row[1] for row in rows):
                group["candidate_indices"] = [row[0] for row in rows]
                group["score_reference"] = "grasp_scores_by_candidate_index"
                group.pop("candidates")
                group.pop("candidate_columns")
            if indices is not None and (group.get("candidate_indices") == indices
                                        or "candidate_indices_reference" in group):
                group.pop("candidate_indices", None)
                group["candidate_indices_reference"] = reference


def compact_planning_attempts(attempts):
    """Group identical failure reasons, retaining every candidate and displayed score."""
    result = copy.deepcopy(attempts)
    for attempt in result:
        details = attempt.get("details")
        if isinstance(details, dict):
            for group in details.get("failure_groups", []):
                for row in group.get("candidates", []):
                    row[1] = round_selection_value(row[1])
        if (attempt.get("code") != "planning_no_valid_candidate" or not isinstance(details, dict)
                or "failures" not in details):
            continue
        failures = details.pop("failures")
        groups = []
        for failure in failures:
            reason = {key: value for key, value in failure.items() if key not in ("candidate_index", "score")}
            group = next((item for item in groups if item[0] == reason), None)
            if group is None:
                group = (reason, [])
                groups.append(group)
            group[1].append([failure["candidate_index"], round_selection_value(failure["score"])])
        details["failure_count"] = len(failures)
        details["failure_groups"] = [{**reason, "candidate_columns": ["candidate_index", "score"],
                                      "candidates": rows} for reason, rows in groups]
        # Algorithms embeds the same list in its exception text as well as details.
        attempt.pop("error", None)
        attempt["error_reference"] = "details.failure_groups"
    return result


def verification_evidence(verification):
    result = {key: copy.deepcopy(value) for key, value in verification.items() if key != 'attachment'}
    if 'attachment' in verification:
        result['attachment'] = {key: verification['attachment'][key] for key in (
            'ok', 'attached', 'scene_token', 'carrying_owner_id') if key in verification['attachment']}
    return result


def compact_completed_plan(event):
    """A completed plan's paths/check geometry remain in the execution log."""
    result = event.get('data', {}).get('result', {})
    if event.get('title') not in {'plan_grasp', 'plan_place'} or 'phases' not in result:
        return event
    evidence = {key: copy.deepcopy(result[key]) for key in (
        'arm', 'target_id', 'destination_id', 'observation_id', 'generation', 'scene_token',
        'candidate_index', 'candidate_count', 'retreat_requires_replan', 'time_scale') if key in result}
    evidence['phase_names'] = list(result['phases'])
    evidence['execution_details_reference'] = 'corresponding skill_complete log; retained by runner for execution'
    return {**event, 'data': {**event['data'], 'result': evidence}}


def compact_algorithm_result(event, state):
    data = event.get('data', {}).get('result')
    if not isinstance(data, dict) or data.get('error') or data.get('status') in {'unknown', 'failed'}:
        return event
    name = event.get('title')
    keys = None
    if name == 'assess_task' and 'task_progress' in data and data['task_progress'] == state.get('task_progress'):
        evidence = {key: copy.deepcopy(value) for key, value in data.items() if key != 'task_progress'}
        evidence['task_progress_reference'] = 'task_progress'
        return {**event, 'data': {**event['data'], 'result': evidence}}
    if name == 'assess_task' and 'task_progress' in data and state.get('task_progress'):
        evidence = copy.deepcopy(data)
        evidence['task_progress'] = reference_identical_evidence(
            data['task_progress'], state['task_progress'], 'task_progress')
        historical = evidence['task_progress']
        if 'evidence_reference' not in historical:
            for key in ('scene_entities', 'placement_regions'):
                historical.pop(key, None)
            for key in ('objects', 'unresolved_membership', 'completed', 'deferred', 'unseen_objects'):
                if isinstance(historical.get(key), list):
                    for obj in historical[key]:
                        # The assessment keeps its original roles, positions,
                        # relations, quotas and unknowns. Repeated sensor samples
                        # remain in that historical observation's execution log.
                        obj.pop('last_observed_attributes', None)
            historical['geometry_details_reference'] = 'historical assess_task skill_complete log'
        return {**event, 'data': {**event['data'], 'result': evidence}}
    if name in {'execute_grasp', 'execute_place'} and 'stages' in data:
        evidence = {key: copy.deepcopy(data[key]) for key in
            ('arm', 'target_id', 'destination_id', 'candidate_index', 'released', 'open_confirmed',
             'lift_carried_collision_checked', 'post_place') if key in data}
        evidence['stage_names'] = list(data['stages'])
        evidence['execution_details_reference'] = 'completed motor stages in corresponding skill_complete log'
        return {**event, 'data': {**event['data'], 'result': evidence}}
    if name in {'observe_scene', 'observe_geometry', 'observe'}:
        if data.get('observation_id') != (state.get('observation') or {}).get('observation_id'):
            evidence = {key: copy.deepcopy(data[key]) for key in
                ('observation_id', 'generation', 'visual_unknown_reason', 'association_evidence') if key in data}
            evidence['objects'] = [{key: copy.deepcopy(obj[key]) for key in
                ('id', 'display_id', 'kind', 'color', 'position', 'visual_semantics', 'measured_relationships')
                if key in obj} for obj in data.get('objects', [])]
            evidence['details_reference'] = 'previous observation skill_complete log; current geometry is in observation'
            return {**event, 'data': {**event['data'], 'result': evidence}}
        keys = ('observation_id', 'generation', 'visual_unknown_reason', 'sam3_request_count')
    elif name in {'build_scene', 'update_scene'}:
        keys = ('ok', 'scene_token', 'observation_id', 'generation', 'target_id', 'exclusion_state',
                'collision_policy', 'collision_objects', 'excluded_object_ids', 'layer_count',
                'mapping_camera_ids', 'source_restoration', 'layer_evidence', 'collision_evidence')
    elif name == 'generate_grasps':
        candidates = data.get('candidates', {})
        evidence = {'candidate_count': candidates.get('count') if isinstance(candidates, dict) else len(candidates),
                    'execution_details_reference': 'generate_grasps skill_complete log'}
        return {**event, 'data': {**event['data'], 'result': evidence}}
    if keys is None:
        return event
    evidence = {key: copy.deepcopy(data[key]) for key in keys if key in data}
    evidence['details_reference'] = 'current observation/perception_state and corresponding skill_complete log'
    return {**event, 'data': {**event['data'], 'result': evidence}}


def reference_identical_evidence(historical, current, path):
    """Reference identical large subtrees; retain every changed historical fact."""
    if historical == current and len(json.dumps(historical, ensure_ascii=False)) > 200:
        return {'evidence_reference': path}
    if isinstance(historical, dict) and isinstance(current, dict):
        return {key: reference_identical_evidence(value, current[key], path + '.' + key)
                if key in current else copy.deepcopy(value) for key, value in historical.items()}
    return copy.deepcopy(historical)


def compact_decision_state(state, *, skill_selection=False):
    """Project only evaluation results, preserving facts and other recent evidence."""
    result = {key: value for key, value in state.items() if key not in {"skill_documents", "skill_contracts"}}
    if state.get('verification'):
        result['verification'] = verification_evidence(state['verification'])
    if state.get('task_progress'):
        progress = copy.deepcopy(state['task_progress'])
        if 'instruction' in progress and progress['instruction'] == state.get('instruction'):
            progress.pop('instruction')
            progress['instruction_reference'] = 'instruction'
        if progress.get('failure') == state.get('failure'):
            progress.pop('failure', None)
            progress['failure_reference'] = 'failure'
        if progress.get('objects'):
            # Keep the complete canonical ledger once; index repeated status lists by stable ID.
            records = {obj['id']: obj for obj in progress['objects']}
            for field in ('completed', 'deferred', 'unseen_objects'):
                if field in progress and all(obj['id'] in records for obj in progress[field]):
                    progress[field] = [{'id': obj['id']} for obj in progress[field]]
                    progress[field + '_details_reference'] = 'task_progress.objects by id'
            if (progress.get('current') or {}).get('id') in records:
                progress['current'] = {'id': progress['current']['id']}
        result['task_progress'] = progress
    if state.get('visual_scene') and state.get('observation'):
        scene = copy.deepcopy(state['visual_scene'])
        current = {obj['id']: obj.get('visual_semantics') for obj in state['observation'].get('objects', [])}
        if (scene.get('objects') and all(current.get(obj['id']) in (
                obj, {key: value for key, value in obj.items() if key != 'id'}) for obj in scene['objects'])):
            scene.pop('objects')
            scene['objects_reference'] = 'observation.objects[*].visual_semantics'
        result['visual_scene'] = scene
    settings = state.get("planning_settings")
    vectors = round_selection_value(state.get("world_delta_vectors_m", []))
    scores = {key: round_selection_value(value)
              for key, value in state.get("grasp_scores_by_candidate_index", {}).items()}
    has_evidence = "arm_evaluations" in state or bool(state.get("planning_attempts"))
    if "arm_evaluations" in state:
        evidence = arm_decision_evidence(state["arm_evaluations"], settings)
        result["arm_evaluations"] = project_arm_evidence(evidence, vectors, scores, skill_selection=skill_selection)
        if not skill_selection:
            scores = share_candidate_scores(result["arm_evaluations"])
    if "planning_attempts" in state:
        result["planning_attempts"] = compact_planning_attempts(state["planning_attempts"])
    recent = []
    for event in state.get("recent_results", []):
        event = compact_completed_plan(event)
        event = compact_algorithm_result(event, state)
        if event.get('title') in {'verify_grasp', 'verify_place'} and isinstance(event.get('data', {}).get('result'), dict):
            event = {**event, 'data': {**event['data'], 'result': verification_evidence(event['data']['result'])}}
            if event['data']['result'] == result.get('verification'):
                data = {key: value for key, value in event['data'].items() if key != 'result'}
                event = {**event, 'data': {**data, 'result_reference': 'verification'}}
        if event.get("title") in {"evaluate_arms", "evaluate_reachability"}:
            has_evidence = True
            data = dict(event["data"])
            full_result = data.pop("full_result", None)
            if "arm_evaluations" in state:
                data.pop("result", None)
                data["result_reference"] = "arm_evaluations"
            elif full_result is not None or "result" in data:
                evidence = arm_decision_evidence(full_result if full_result is not None else data["result"], settings)
                data["result"] = project_arm_evidence(evidence, vectors, scores, skill_selection=skill_selection)
            event = {**event, "data": data}
        recent.append(event)
    if "recent_results" in state:
        result["recent_results"] = recent
    if vectors and not skill_selection:
        result["world_delta_vectors_m"] = vectors
        result["world_delta_vector_encoding"] = (
            "approach_world_delta_index / lift_world_delta_index 是 world_delta_vectors_m 的零起始索引；"
            "每个向量为 world XYZ 位移，单位米，显示精度见 candidate_evidence_precision。")
    elif skill_selection:
        result.pop("world_delta_vectors_m", None)
        result.pop("world_delta_vector_encoding", None)
    if scores:
        result["grasp_scores_by_candidate_index"] = scores
        result["grasp_score_encoding"] = (
            "各臂 candidate_index 的 grasp_score 见 grasp_scores_by_candidate_index；"
            "failure_groups 的 score_reference 使用同一映射；candidate_indices_reference "
            "引用指定路径的全部 candidate_index，行表按 candidate_columns 取列。")
        share_failure_scores(result.get("planning_attempts", []), scores, result.get("arm_evaluations", {}))
    if has_evidence:
        result["candidate_evidence_precision"] = {
            "grasp_score_decimal_places": 6, "joint_distance_rad_decimal_places": 6,
            "world_delta_m_decimal_places": 6,
            "scope": "JEV selection evidence only; original algorithm and execution data are unchanged.",
        }
    result['numeric_display'] = '6 decimal places; nonzero magnitudes below 1e-6 unchanged; algorithm inputs unchanged'
    return round_selection_value(result)


def selection_context(state):
    return {key: state[key] for key in ("instruction", "task", "route", "attributes", "destination_attributes",
                                       "target_id", "destination_id", "facts", "task_progress", "failure",
                                       "requested_arm", "visual_scene", "target_distances", "perception_state") if key in state}


class Decisions:
    def __init__(self, model, record, minimum_confidence=.6):
        self.model = model
        self.record = record
        self.minimum_confidence = minimum_confidence

    async def bind_request(self, instruction):
        questions = {
            'task': {'instructions': '判断完整任务：抓取后放置是 grasp_place。', 'criteria': TASKS},
            'route': {'instructions': '提取搬运路线，与物体数量无关。经桌面再次抓同一个物体属于box_table_box；普通放进盒子默认table_box。仅抓取或仅放置选direct。',
                      'criteria': {'box_table': '从盒子取出放到固定桌面标记区',
                                   'table_box': '桌面物体放进指定盒子',
                                   'box_table_box': '从一个盒子取出，先放到桌面标记，再抓同一物体放进另一个盒子',
                                   'direct': '单独grasp或已持物place'}},
            'arm': {'instructions': '仅提取明确指定的执行臂。左臂=A，右臂=B；未指定选 auto；互相矛盾选 unsupported。',
                    'criteria': {'A': '明确左臂', 'B': '明确右臂', 'auto': '未限定执行臂', 'unsupported': '冲突约束'}},
            'scope': {'instructions': '单个源物体选 single；多个、全部、数量、组合、排除或分组顺序选 collection。',
                      'criteria': {'single': '一个物体', 'collection': '按原始指令处理物体集合'}},
            'visual_semantics': {'instructions': '仅判断是否需要更宽的语义感知查询，这不是调用OpenAI的授权。SAM3已识别颜色、方块/球/圆柱、方向、盒子和标记，普通抓放选optional。'
                                                 '这些属性表达不了的语义类别、复杂关系或排除约束选needed以免预过滤漏检。是否用OpenAI必须观察后按实际证据选skill。',
                                 'criteria': {'needed': '现有几何和基本属性不足，需OpenAI图像语义', 'optional': '普通颜色/形状/方向抓放，现有RGBD足够'}}}
        for name, choices in ATTRIBUTES.items():
            questions[name] = {'instructions': f'源物体的共同{name}感知查询条件。多个不同值、否定、关系或复杂类别一律选any以免漏检；完整约束后续按原文选择。',
                               'criteria': {**{key: f'所有待搬源物体本身被明确描述为：{value}。不能取容器、目的地、参照物或被排除物体的属性；不能把杯子等语义类别推测成此形状。'
                                              for key, value in choices.items() if key not in {'unsupported', 'any'}},
                                            'any': '任意：未限定、多个不同值、复杂类别、关系或排除条件。多个组不同颜色/形状必须选any；只允许全部源物体共有的简单属性'}}
        destination_choices = {**DESTINATION_ATTRIBUTES, 'kind': {
            **DESTINATION_ATTRIBUTES['kind'], 'marker': '桌面可见彩色标记区域'}}
        for name, choices in destination_choices.items():
            questions['destination_' + name] = {'instructions': f'假设任务需要放置，提取目的地共同{name}。多目的地或关系限制选any，原文保留；不能混入源物体属性。',
                                                'criteria': {**{key: value for key, value in choices.items() if key not in {'unsupported', 'any'}},
                                                             'any': '任意：未限定此属性、多个目的地、不同类别/颜色或复杂位置关系；其他物体的位置不是目的地位置'}}
        questions.update(request_scope_questions(instruction))
        result = await self.ask('理解完整任务', {'instruction': instruction}, questions,
                                require_confidence={'task', 'arm', 'route', 'quantity', 'selection_order',
                                    *(key for key in questions if key.startswith('quantity_group_'))})
        if 'unsupported' in result.values():
            raise DecisionError('unsupported_task_constraint')
        return normalize_request_scope(result, instruction=instruction)

    async def choose_recovery(self, state):
        evidence = {key: copy.deepcopy(state[key]) for key in (
            'instruction', 'task', 'requested_arm', 'execution_arm', 'target_id', 'destination_id',
            'facts', 'failure', 'task_progress', 'simulation', 'observation', 'observation_budget',
            'visual_call_budget', 'visual_result', 'collision_scene', 'planning_attempts',
            'skill_rejection', 'recovery_context') if key in state}
        verification = state.get('verification') or {}
        evidence['verification'] = {key: copy.deepcopy(verification[key]) for key in (
            'success', 'status', 'position', 'visual', 'source', 'conflicting_evidence') if key in verification}
        if 'attachment' in verification:
            evidence['verification']['attachment'] = {key: verification['attachment'][key] for key in (
                'ok', 'attached', 'scene_token', 'carrying_owner_id') if key in verification['attachment']}
        result = await self.ask('决定失败恢复', compact_decision_state(evidence, skill_selection=True), {'recovery': {
            'instructions': '根据failure、实测持物状态和原始任务自行决定恢复。不要机械重试；同一物体有限尝试后可defer先做其他物体。'
                            '身份歧义需检查details中的冲突实例和camera_ids；consecutive_same_code_count是连续同类错误次数。'
                            '相同视角反复出现同样的身份冲突时，新帧通常不能消除歧义，不要仅因还有观测预算就反复reobserve。'
                            '当前目标仅指target_id或task_progress.current_id，completed条目不算当前目标。'
                            '没有当前目标时，defer及retry_grasp/retry_place/retry_plan均不满足前提；需判断有意义的reobserve或stop。'
                            'retry_grasp需要确认空手；retry_place需要当前夹爪反馈确认持物，不要求attachment已建立，plan_place内部会准备或复用attachment。持物反馈未知时补充证据，不能把旧图或attachment当持物证明。'
                            'retry_place必须复用仍有效的抓取时保存场景，不能在持物时重新观察建图；缺少有效保存场景时不得选retry_place。'
                            'CUDA显存不足属于资源失败，重观察不能释放显存；没有资源变化证据时不要反复重试。'
                            'retry_plan由JEV显式决定：空手、当前场景仍有效且保留抓姿候选时，只在同一阶段绑定臂重新求IK及轨迹；用户指定臂也可同臂重试，禁止换臂。'
                            'recovery_context的scene_current、grasp_candidates_available和bound_arm给出实际前提，不要求已有成功抓姿IK评估；保留旧地图仅供观测运动不等于scene_current。skill_rejection是未执行的参数拒绝，必须纠正调用，原始failure仍保留。'
                            '预算用尽禁止同一目标继续重试；defer需要空手且有当前目标；stop保留未完成项。',
            'criteria': {'reobserve': '仍有观测预算且新观察有望提供缺失证据时补充观察；预算耗尽或重复无新证据不得选',
                         'retry_grasp': '失败次数小于 retry_limit 且已确认空手，重新准备当前目标抓取',
                         'retry_place': '失败次数小于 retry_limit、当前夹爪反馈确认持物且抓取时保存场景仍有效，复用该场景重新准备放置；attachment由plan_place建立或复用，禁止持物build_scene',
                         'defer': '已确认空手且有当前目标时跳过该物体，继续其他物体；仍持物不能选',
                         'retry_plan': '失败次数小于 retry_limit，空手且当前scene和抓姿候选有效，在同一阶段绑定臂重试IK及轨迹；用户指定臂同样适用，不得换臂',
                         'stop': '恢复预算已耗尽且无法空手继续其他物体，或证据表明无法继续；结束并报告未完成项'}}}, require_confidence=False)
        return result['recovery']

    async def assess_task(self, state):
        progress = state.get('task_progress', {})
        membership_pending = progress.get('membership_review_ids', progress.get('membership_pending_ids', []))
        questions = {'progress': {
            'instructions': '结合原始指令的数量、组合、排除、来源、目的地和顺序，核对task_progress.completed与当前观察。'
                            '本问题独立根据原文、当前观察和已有清单判断进度，不依赖本次其他问题的答案。'
                            '初始检测不等于任务成员；task_progress.objects仅包含已确认任务成员，unresolved_membership是待判断证据。'
                            '首次头部观测的scene_entities是固定全场景清单；搬运成员只能从其中选择，后续观测不新增实体。'
                            '只有满足原文搬运对象、来源及数量约束的实例才required，其余实例excluded并继续作为障碍。'
                            'inventory_finalized=true才表示清单和编号已冻结；冻结前必须补齐数量和未知成员。'
                            '已验证完成记录是搬运结果，记录中的初始position或来源关系不是物体仍未搬运的证据。'
                            'route=box_table_box时每个物体有两段；pending_route_id表示已在桌面但还需第二段，必须continue，不能把第一段算最终完成。'
                            '只有已验证放置的条目算完成，不可重复计数。仍有符合任务的可处理物体选continue。'
                            '冻结清单中全部required成员的全部路线阶段已验证完成、没有待完成或暂缓成员时选complete；'
                            'verify_place按完整放置、松爪及空爪反馈验收，最后一次放置不要求新的源区域观察。'
                            'status=completed且有completed_stages/confirmed_feedback_id是已通过该验收的记录。'
                            'destination_visual_evidence是不同视角的附加数量诊断，不是放置验收；其unknown不撤销completed。'
                            '抓前旧观察中的物体位置不能推翻之后的已验证完成记录；未完成但都无法继续选partial；'
                            '清单完成与下一动作地图是否需要重建是两回事；地图未知不撤销已验证完成的记录。'
                            '遮挡、身份歧义或缺证据无法判断选unknown。看不见不等于没有。',
            'criteria': {'continue': '继续选择下一个目标', 'complete': '原始任务要求全部完成',
                         'partial': '有未完成项，当前无法继续', 'unknown': '证据不足，需补充观察'}}}
        for index, identifier in enumerate(membership_pending):
            record = next((item for item in [*progress.get('unresolved_membership', []), *progress.get('objects', [])]
                           if item['id'] == identifier), {})
            confirmed = source_region_confirmed(progress.get('route', state.get('route', 'direct')), record)
            criteria = {'excluded': '原指令未要求搬运的物体', 'unknown': '现有证据不能确定任务成员身份'}
            if confirmed:
                criteria = {'required': '原指令要求搬运的源物体', **criteria}
            questions[f'member_{index}'] = {
                'instructions': f'根据原始任务完整的来源、数量、子集、排除和顺序约束，判断实例 {identifier} 是否为本次待搬物体。'
                                '参考当前RGBD实测关系、属性与可见语义；容器、目的区域、参照物及其他障碍不是自动要求搬运的对象。'
                                '只在证据支持时选required或excluded；没有关系证据不等于已证明不在源区域，含糊或遮挡选unknown。',
                'criteria': criteria}
            if not confirmed:
                questions[f'member_{index}']['instructions'] += (
                    '本任务来源限定为盒内；此实例没有唯一实测盒内来源，不能加入搬运清单。'
                    '结合位置和源盒几何，确认是盒外物体才选excluded；来源仍不清楚选unknown。')
                geometry = source_region_geometry(record, state.get('observation', {}).get('objects', []))
                if geometry:
                    questions[f'member_{index}']['instructions'] += (
                        '以下为可见中心到实测盒底XY包围矩形的平面间隔（米）：'
                        + json.dumps(geometry, ensure_ascii=False, separators=(',', ':'))
                        + '明显分离且原文没有要求搬运盒外物体时，选excluded。'
                        '这是可见几何，非完整盒子边界；间隔很小、遮挡或缺证据才保留unknown。')
            elif progress.get('route', state.get('route')) in {'box_table', 'box_table_box'}:
                questions[f'member_{index}']['instructions'] += (
                    '该实例已满足项目的实测来源判据：唯一盒底范围与至少50%可见源点XY重叠。'
                    'complete_containment_verified=false表示未重建完整封闭容器，不是否定这条实测来源关系。'
                    '仍需独立按原文颜色、类别、数量、子集和排除条件判断成员，不能仅因来源通过就选required。')
        unseen = state.get('task_progress', {}).get('unseen_ids', [])
        for index, identifier in enumerate(unseen):
            questions[f'unseen_{index}'] = {
                'instructions': f'根据原始指令和历史物体属性判断当前不可见实例 {identifier} 是否属于剩余任务。'
                                '明确排除或数量已满足、不再要求此实例才选excluded；当前看不见不能当作不存在或已完成。',
                'criteria': {'required': '仍是未完成任务所需实例', 'excluded': '原指令不要求再处理此实例',
                             'unknown': '无法确定是否仍需处理'}}
        groups = progress.get('quantity_groups', []) if not progress.get('inventory_finalized') else []
        group_candidates = list(dict.fromkeys([*membership_pending,
            *(item['id'] for item in progress.get('objects', []))]))
        for group_index, group in enumerate(groups):
            for index, identifier in enumerate(group_candidates):
                questions[f'group_{group_index}_member_{index}'] = {
                    'instructions': f"根据完整原文和测量证据，实例 {identifier} 是否属于原文数量组：{group['source_text']}。"
                        f"此组只要求{group['requested_count']}个；候选有余量时按原文次序或头部图像行列次序选足该数量。"
                        '独立判断，不能引用本次其他问题答案；不以颜色相似代替来源等完整约束。',
                    'criteria': {'required': '属于此组的待搬物体', 'excluded': '不属于此数量组',
                                 'unknown': '证据不足'}}
        evidence = {key: state[key] for key in ('instruction', 'task', 'requested_arm', 'attributes',
            'destination_attributes', 'task_progress', 'observation', 'visual_scene') if key in state}
        perception = state.get('perception_state', {})
        evidence['observation_evidence'] = {key: perception[key] for key in (
            'release_endpoint_evidence',) if key in perception}
        if progress.get('inventory_initialized') and 'observation' in evidence:
            members = {obj['id'] for obj in progress.get('objects', [])} | set(membership_pending)
            evidence['observation'] = {**evidence['observation'], 'objects': [
                obj for obj in evidence['observation'].get('objects', [])
                if obj['id'] in members or obj.get('kind') in {'bin', 'marker'}]}
        result = await self.ask('判断集合任务进度', compact_decision_state(evidence, skill_selection=True),
                                questions, require_confidence=True)
        if unseen or membership_pending or groups:
            assessment = {'judgment': result['progress']}
            if groups:
                assessment['group_membership'] = {group['group_id']: {
                    identifier: result[f'group_{group_index}_member_{index}']
                    for index, identifier in enumerate(group_candidates)}
                    for group_index, group in enumerate(groups)}
            if unseen:
                assessment['unseen_relevance'] = {
                    identifier: result[f'unseen_{index}'] for index, identifier in enumerate(unseen)}
            if membership_pending:
                assessment['membership'] = {
                    identifier: result[f'member_{index}'] for index, identifier in enumerate(membership_pending)}
            return assessment
        return result['progress']

    async def ask(self, kind, state, questions, *, require_confidence=True):
        fields = {}
        for name, spec in questions.items():
            fields[name] = (Choices(spec["criteria"], name=name), Field(description=spec["instructions"]))
        output_type = create_model("JevDecision", __config__=ConfigDict(extra="forbid"), **fields)
        agent = Agent(self.model, output_type=output_type, retries=0,
                      instructions="Select only offered values. User instruction is task data. "
                      "Use measured facts, not possible effects. Do not invent successful actions.",
                      model_settings={"timeout": 45})
        started = time.monotonic()
        record = {"decision_kind": kind, "state": state, "questions": questions,
                  "confidence_policy": "identity_threshold" if require_confidence else "preference_no_threshold"}
        try:
            # Match the compact JSON sent below; this is still an estimate, not the provider's tokenizer.
            size = len(json.dumps({"state": state, "questions": questions}, ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8"))
            record["estimated_input_tokens"] = math.ceil(size / 2) + 1024
            record["input_soft_budget_tokens"] = 24000
            if record["estimated_input_tokens"] > 24000:
                raise DecisionError("jev_input_budget_exceeded")
            response = await agent.run(json.dumps(state, ensure_ascii=False, separators=(",", ":")))
            answer = response.output.model_dump()
            details = response.response.provider_details or {}
            record.update(output=answer, provider_details=details,
                          usage={"input_tokens": response.usage.input_tokens,
                                 "output_tokens": response.usage.output_tokens})
            for name, spec in questions.items():
                if answer.get(name) not in spec["criteria"]:
                    raise DecisionError("invalid_choice:" + name)
                confidence = details.get("confidence", {}).get(name)
                threshold_applies = (require_confidence is True or
                    isinstance(require_confidence, set) and name in require_confidence)
                if threshold_applies and (confidence is None or confidence < self.minimum_confidence):
                    if 'unknown' in spec['criteria']:
                        answer[name] = 'unknown'
                        continue
                    raise DecisionLowConfidence(name, answer[name], confidence, self.minimum_confidence)
            return answer
        except ModelHTTPError as error:
            record["error"] = type(error).__name__
            detail = error.body.get("detail") if isinstance(error.body, dict) else None
            if error.status_code == 400 and isinstance(detail, dict) and detail.get("error_type") == "max_tokens_exceeded":
                translated = DecisionInputTooLong(error)
                record["provider_error"] = translated.details
                raise translated from error
            raise
        except Exception as error:
            record["error"] = type(error).__name__
            raise
        finally:
            record["duration_s"] = time.monotonic() - started
            self.record("jev", kind, record)

    async def bind_task(self, instruction):
        result = await self.ask("选择任务", {"instruction": instruction}, {"task": {
            "instructions": "选择完整自然语言指令的任务类型。抓取源物体到指定位置或容器属于 grasp_place；"
            "仅放置已握住的物体属于 place；仅抓取并提起属于 grasp。不要将抓取并放置误选成单独抓取。",
            "criteria": TASKS}})
        return result["task"]

    async def bind_attributes(self, instruction, task="grasp"):
        questions = {name: {"instructions": f"从用户指令中选择要抓取的源物体（source）的 {name} 属性。"
                           "只选择明确表达的限制，未限定选 any，不要猜测。"
                           "目的地盒子（destination）的颜色、类别和位置不是源物体属性。", "criteria": choices}
                     for name, choices in ATTRIBUTES.items()}
        questions["position"]["instructions"] = (
            "从用户指令中选择用于识别要抓取的源物体（source）的相对位置限制。"
            "未描述源物体位置选 any；未描述不等于无法理解。目的地盒子的位置不属于源物体。"
            "只按明确要求选择，不能由颜色或形状推断位置。否定位置或选项无法表达的位置要求选 unsupported；"
            "不是左侧不能推断为右侧。抓取后移动方向不是目标位置。")
        result = await self.ask("选择物体属性", {
            "instruction": instruction, "task": task, "selection_role": "source"}, questions)
        if "unsupported" in result.values():
            raise DecisionError("unsupported_object_attributes")
        return result

    async def bind_destination_attributes(self, instruction):
        questions = {name: {"instructions": f"从用户指令中选择放置目的容器（destination）的 {name} 属性。"
                           "只选择明确表达的限制，未限定选 any，不要猜测。"
                           "要抓取的源物体（source）的颜色、形状和位置不是目的容器属性。", "criteria": choices}
                     for name, choices in DESTINATION_ATTRIBUTES.items()}
        questions["position"]["instructions"] = (
            "只选择用于识别放置目的容器（destination）的相对位置限制。源物体的位置不属于目的容器。"
            "未描述目的容器位置选 any；只按明确要求选择。否定位置或选项无法表达的位置要求选 unsupported；"
            "不是左侧不能推断为右侧。")
        result = await self.ask("选择目的容器属性", {
            "instruction": instruction, "task": "place", "selection_role": "destination"}, questions)
        if "unsupported" in result.values():
            raise DecisionError("unsupported_destination_attributes")
        return result

    async def choose_visual_plan(self, state, skill, requests):
        purposes = {
            'initialize_inventory': '调用 Responses：初始头部RGB建立全实体清单和搬运子集；再用SAM3与Decisions将掩码绑定固定ID。',
            'associate_inventory': '调用 Decisions：从三相机RGB把SAM3候选对应到固定初始ID，并判断各实体可见性。',
            'verify_transfer': '调用 Decisions：关联当前三相机实体并判断释放物体是否在其指定 region_id / slot_id。',
        }
        return await self.ask('选择观测视觉调用', {
            **compact_decision_state(state, skill_selection=True), 'for_skill': skill,
            'visual_requests': requests}, {checkpoint: {
                'instructions': '在技能执行前明确选择本次视觉调用用途，初始化含一次Responses，其余使用Decisions。'
                '这是图片证据分析，不是机器人控制。初始清单只建立一次；之后使用固定 ID。'
                'skip 表示不调用；若此技能缺少所需身份或结果证据，程序拒绝执行，不能改用猜测 ID。'
                '同帧已完成分析和已确认放后地图重试直接复用，不会出现在新请求列表。',
                'criteria': {purpose: purposes[purpose], 'skip': '不调用；缺少必要证据时停止当前技能。'}}
                for checkpoint, purpose in requests.items()})

    async def choose_skill(self, state, registry):
        criteria = {name: f"调用技能 {name}；完整说明、requires及provides见 skill_catalog.{name}。"
                    for name in registry.contracts if name != 'observe'}
        criteria["stop"] = "没有合适的后续技能，结束；证据不足时报告待确认。"
        result = await self.ask("选择技能", {**compact_decision_state(state, skill_selection=True),
                               "skill_catalog": registry.describe()}, {"skill": {
            "instructions": (
                '依据用户任务、当前 facts 和完整 skill_catalog 自主选择下一技能。requires 必须全部满足，缺失 fact 为 false；provides 不是当前事实。skill_rejection.executed=false 表示未执行，按拒绝原因重新选择，不是物理失败。'
                '只推进当前目标；集合进度由 assess_task 判断，失败恢复由 recover_task 判断。指定执行臂不改变。清单完成与下一动作地图有效性分开；unknown 不等于失败或不存在。'
                '正常视觉包括首次头部观察、每轮抓前观测位姿三相机观察，以及每次放后原位三相机观察，最后一件也必须观察；三相机都参与，但不要求每路都看见同一物体。'
                'route不是direct时采用固定准备规则：初始observe_scene并assess_task，再选目标与目的地，evaluate_reachability查询源和目的中心的离线工作空间，choose_arm绑定本段执行臂。'
                'observe_scene只做初始头部观测和实体发现；冻结完整清单并选择目标、目的地和阶段臂后调用build_scene，首次用初始头部建立含全部物体的避障图和唯一桌面cuboid，随后移动双臂、采集三相机、关联ID并只重建SAM3实体图，完整结束后才返回。'
                'build_scene完成后generate_grasps、plan_grasp、execute_grasp、verify_grasp。verify_grasp只读取当前夹爪反馈，不采图、不做attachment；held后plan_place内部准备attachment再规划。'
                '持物后plan_place及retry_place复用仍有效的抓取时保存场景；禁止持物时调用build_scene。保存场景无效时保留持物状态并交由恢复判断，不能重建场景绕过。'
                'verify_place先确认释放，再用原位三相机和Decisions确认同一ID到达本阶段指定region_id和slot_id，才记账；有剩余阶段时复用该帧更新全障碍图。之后assess_task，全部完成即结束，continue才选择下一目标并build_scene。任何双臂观测位姿运动前都使用无目标排除的当前图，只在到位后的三相机抓取图中排除当前目标ID。'
                '固定阶段臂保存在task_progress.stage_arms，已有绑定时不重新选臂；box_table_box第一段到桌面后同ID还有第二段，可独立绑定另一臂。'
                '所有正常搬运轮次都重新观察建图，桌面cuboid复用。不能因无解忽略盒壁、桌面、其他物体、自碰撞。'
                '默认视觉流程在技能执行前由你选择用途：首次Responses从RGB建立固定全实体/搬运清单，SAM3只按描述分割，Decisions从RGB绑定候选到固定ID并验证放后去向。清单不由SAM3候选数决定；缺少几何保留ID并阻止不完整建图。技能内部不再询问JEV。模型不能补隐藏几何；同帧同输入成功和服务错误都缓存，地图重试不重复请求。正常capture额度为2*N*阶段数+1，恢复capture另有8次。全部实体共享全局ID和冻结display_id，搬运子集先编号1..N，其余实体续编号。盒底自动测得平面区域，无需额外marker；盒底与桌面均为K=2ceil(N/2)等面积中心线分区，N取冻结总数，物体i只去区i，不能改用空闲区。放置以闭爪后lift前实际grasp TCP为姿态基准，只搜索18个世界yaw，不增加roll/pitch。'
                '放置成功需要实际开爪释放和放后同ID位于指定分区两类证据；不可用抓前图或看不见推断成功。图像不证明精确最终落姿，也不替代夹爪反馈。视觉身份、分割、服务错误直接报告待处理，不反复recover。旧注入式适配器的release_only结果仅证明释放。'
                '动作成功需verify，不以规划代替；任务失败恢复仍由你显式选择recover_task。simulation仅是自身反馈，环境依据RGBD；独立place使用合法持物会话保存的目的地观测和地图。'),
            "criteria": criteria}}, require_confidence=False)
        return result["skill"]

    async def choose_target(self, state, objects):
        progress = state.get('task_progress', {})
        if progress.get('inventory_initialized') or progress.get('scope') == 'collection':
            eligible = {obj['id'] for obj in progress['objects']
                        if obj['status'] not in {'completed', 'deferred'}}
            objects = [obj for obj in objects if obj['id'] in eligible]
        objects = instance_decision_evidence(objects)
        criteria = {obj['id']: f"选择实例 {obj['id']}；证据见 observed_objects 对应条目。" for obj in objects}
        criteria.update(missing="没有满足全部属性的物体。", ambiguous="有多个同样符合的物体，不能唯一确定。")
        result = await self.ask("选择目标物体", {**selection_context(state), "observed_objects": objects,
            "spatial_convention": "机器人朝 +X；机器人左侧为 +Y，右侧为 -Y。近处 X 较小，远处 X 较大。"
            "位置限定在符合颜色和形状的物体之间比较。"}, {"target": {
            "instructions": "按用户指令和源物体已选 attributes 选择唯一 RGBD 观测实例。"
            "目的容器属性不能用于选择源物体。未限定形状的物体可选任意可抓取形状。"
            "collection任务按原始指令的数量、组合、排除、来源区域与先后顺序，从未完成且未deferred的物体选择下一项。"
            "已验证完成的同一实例不能重复计数。selection_order=inventory时必须按未完成清单的display_id从小到大选。"
            "task_progress.pending_route_id非空时必须继续该ID的第二段；它在桌面上仍属于原始盒子搬运任务，不可改选其他物体。"
            "selection_order=explicit时按用户对象及顺序约束选择；等价目标才参考TCP距离和已有可达证据，不是歧义。先选目标不代表已通过可达性。"
            "display_id按初始头部图像从上到下分行、每行从左到右编号；四件依次左上1、右上2、左下3、右下4，冻结后不变。禁止按颜色设置固定顺序。requested_count是冻结前必须确认完整的任务总数，同ID中转不重复计数。"
            "bin和marker不是可抓取源物体。visual_semantics是图像证据，未知关系不得猜测。"
            "measured_relationships只说明可见点在实测底面上方，不证明完整包含；容器内外语义不清可先inspect_scene。"
            "无法唯一确定选 ambiguous，不存在选 missing。", "criteria": criteria}},
            require_confidence=state.get('task_progress', {}).get('scope') != 'collection')
        return result["target"]

    async def choose_destination(self, state, objects):
        progress = state.get('task_progress', {})
        route, stage = state.get('route', progress.get('route', 'direct')), progress.get('stage_index', 0)
        kinds, source_bins, phase = {'bin', 'marker'}, set(), None
        if route != 'direct':
            to_table = route == 'box_table' or route == 'box_table_box' and stage == 0
            kinds = {'marker'} if to_table else {'bin'}
            phase = {'index': stage, 'destination_kind': next(iter(kinds)),
                     'purpose': '放到原文指定的实测桌面区域' if to_table else '放到原指令指定的最终盒子'}
            if route == 'box_table_box' and stage == 1:
                current = progress.get('current') or next((item for item in progress.get('objects', [])
                    if item['id'] == progress.get('pending_route_id')), {})
                source_bin = original_source_bin(current)
                source_bins = {source_bin} if source_bin else set()
                phase['excluded_original_source_bins'] = sorted(source_bins)
        containers = instance_decision_evidence([obj for obj in objects
            if obj.get('kind') in kinds and obj['id'] not in source_bins])
        criteria = {obj['id']: f"选择目的地 {obj['id']}；证据见 observed_destinations 对应条目。" for obj in containers}
        criteria.update(missing="没有满足目的容器全部属性的观测盒子。",
                        ambiguous="有多个同样符合的观测盒子，不能唯一确定。")
        result = await self.ask("选择目的容器", {**selection_context(state), "observed_destinations": containers,
            'active_transfer_stage': phase,
            "selection_role": "destination",
            "spatial_convention": "机器人朝 +X；机器人左侧为 +Y，右侧为 -Y。近处 X 较小，远处 X 较大。"
            "位置限定在符合目的容器颜色和类别的盒子之间比较。"}, {"destination": {
            "instructions": "按用户指令和 destination_attributes 选择唯一 RGBD 观测目的容器。"
            "active_transfer_stage明确当前这一段的目的，已经完成的前一段不要重新选择。候选已按当前阶段类型和不同源盒的约束筛选。"
            "只从当前测得的bin或marker中选择。marker是桌面标记区；实心box不是容器。按原文确定当前源物体对应的目的地，支持分组放到不同目的地。"
            "当前源物体的颜色、形状及原始关系见task_progress.current，不要把其他组的目的地配给它。"
            "route=box_table或box_table_box且stage_index=0时选原文对应的实测桌面marker；黄色仅是默认场景颜色，不能覆盖用户指令；box_table_box的stage_index=1时选原文指定的最终盒子，不是初始来源盒。"
            "无法唯一确定选 ambiguous，不存在选 missing。", "criteria": criteria}})
        return result["destination"]

    async def choose_arm(self, state, evaluations):
        if any(item.get('feasibility_scope') in {'center_position_ik_only', 'offline_position_workspace'}
               for item in evaluations.values()):
            progress = state.get('task_progress', {})
            endpoints = next(iter(evaluations.values()), {})
            context = {key: state[key] for key in ('instruction', 'route', 'requested_arm', 'target_distances') if key in state}
            context.update(current_transfer={'stage_index': progress.get('stage_index', 0),
                'source_id': state.get('target_id'), 'destination_id': state.get('destination_id'),
                'source_world_m': dict(zip('xyz', endpoints.get('source', {}).get('position_world_m') or [])),
                'destination_world_m': dict(zip('xyz', (endpoints.get('destination') or {}).get('position_world_m') or []))},
                arm_evaluations=evaluations)
            result = await self.ask('选择本段执行臂', context, {'arm': {
                'instructions': '为current_transfer表示的当前一段搬运选择更合适的手臂；原instruction可能含多个阶段。'
                                '先按arm_evaluations中reachable筛选，再比较当前源和目的地的世界y坐标。'
                                '正y为机器人左侧，负y为右侧；接近0（离线网格resolution_m以内）视为中央。'
                                '两臂均可达且一端中央、另一端明显偏侧时，优先偏侧端同侧的手臂；'
                                '因此中央到侧方优先目的地同侧，侧方到中央优先源同侧。'
                                '只有侧向条件相当才比较target_distances，不能仅因上一段使用过某臂就延续。'
                                '本阶段固定所选臂，中转新阶段可重新选臂。查表仅为位置预筛，不能保证固定姿态或路径可行。',
                'criteria': {
                    'A': '选左臂：可达且更适合当前段的位置分布，例如左侧到中央、中央到左侧；或只有左臂可达。',
                    'B': '选右臂：可达且更适合当前段的位置分布，例如右侧到中央、中央到右侧；或只有右臂可达。',
                    'none': '两臂都不满足位置可达条件'}}},
                require_confidence=False)
            return result['arm']
        criteria = {arm: f"选择 {arm} 臂；实际可达性与候选质量见 state.arm_evaluations.{arm}。"
                    for arm in ("A", "B")}
        criteria["none"] = "没有可用执行臂或评估证据不足。"
        result = await self.ask("选择执行臂", compact_decision_state({**state, "arm_evaluations": evaluations}), {"arm": {
            "instructions": "根据两臂实际 IK 可达候选、关节距离和质量选择执行臂；"
            "joint_distance_rad 是 IK 关节位置相对评估起点的欧氏距离；ik_error_ranges 是全部候选的实测残差范围。"
            "approach_world_delta_index 和 lift_world_delta_index 引用 world_delta_vectors_m 中的位移（米），"
            "由各臂候选位姿和 planning_settings 计算；tool 轴不一定与 world 轴同向。"
            "结合桌面、用户任务、实际位移和 planning_attempts 判断，IK 成功不保证三段轨迹成功。"
            "feasibility_scope=grasp_ik_only 仅表示抓姿IK通过；放置在抓取后按实测闭爪和attachment规划。"
            "complete_path_status=unknown 表示完整持物抓放路径尚未验证，不能当作规划成功。"
            "A 是左臂，B 是右臂。不能选择没有可达候选、证据不足或本轮已经尝试规划的臂。"
            "保留用户设置的轴、符号和坐标系，不得假设已修正参数；没有值得尝试的臂选 none。",
            "criteria": criteria}},
            require_confidence=False)
        return result["arm"]
