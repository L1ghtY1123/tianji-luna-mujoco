"""Measured motion operations; task compilation and graph policy live in runtime."""
import copy
import time
from pathlib import Path

import numpy as np

from ..algorithms import AlgorithmError
from ..demo_sampling import (
    MAX_SAMPLING_ROUNDS, grasp_candidate_count,
    empty_pool_filter,
    grasp_input_identity,
    round_evidence,
    validate_planning_exhaustion,
    validate_sampling_evaluation,
)
from ..demo_transfer_verification import (
    build_release_binding,
    freeze_release_execution,
    release_execution_context,
    released_target_map_is_current,
    validate_release_capture,
)
from ..observation_motion import observation_poses, source_view_evidence
from ..placement import observed_object
from ..runner import Runner
from ..skills import Registry
from .container_geometry import container_transport_region
from .processing import ProcessingSkills
from .return_pose import prepare_empty_hand_motion_scene

DESTINATION_ID = "demo-destination"
ROOT = Path(__file__).resolve().parents[3]


def configured_destination(observation, region, *, identifier=DESTINATION_ID, support_id=None):
    """Bind operator XY intent to a currently measured support height."""
    from scipy.spatial import ConvexHull

    from .placement_region import measured_table_region

    center = np.asarray(region["center_world_xy_m"], dtype=float)
    polygon = np.asarray(region["region_polygon_world_xy_m"], dtype=float)
    if (center.shape != (2,) or polygon.shape != (4, 2)
            or not np.isfinite(center).all() or not np.isfinite(polygon).all()):
        raise ValueError("invalid_configured_destination")
    supports = []
    for table_id in observation.get('demo_tables', {}).get('tables', {}):
        if support_id is not None and table_id != support_id:
            continue
        measured = measured_table_region(observation, table_id)['placement_reference']
        hull = ConvexHull(np.asarray(measured['region_polygon_world_xy_m']))
        if np.all(np.vstack((center, polygon)) @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9):
            supports.append((table_id, measured))
    if len(supports) != 1:
        raise ValueError('configured_region_requires_unique_measured_table:' + identifier)
    support_id, measured = supports[0]
    floor = measured['floor_z_m']
    precise = {}
    if region.get('precise_xy'):
        tolerance = region.get('xy_tolerance_m')
        if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('configured_precise_region_requires_xy_tolerance')
        precise = {'precise_xy': True, 'xy_tolerance_m': tolerance}
    constraints = {}
    if region.get('required_robot_configurations'):
        constraints['required_robot_configurations'] = copy.deepcopy(region['required_robot_configurations'])
    return {"id": identifier, "kind": "marker", "color": "unmarked",
        "source": "configured_demo_region", "position_source": "configured_xy_measured_support",
        "position": [*center.tolist(), floor], "visible_centroid_world": [*center.tolist(), floor],
        "roles": ["target_region"], "visual_membership": "excluded",
        "coverage": {"camera_ids": [], "configured_region": True},
        "placement_reference": {**precise, **constraints, "floor_z_m": floor, "position_world_m": [*center.tolist(), floor],
            "region_polygon_world_xy_m": polygon.tolist(), "region_id": "region:" + identifier,
            "support_entity_id": support_id, "observation_id": observation["observation_id"],
            "support_measurement": copy.deepcopy(measured),
            "source": "configured_xy_measured_support"}}


class RobotOperations(ProcessingSkills, Runner):

    def __init__(self, *args, region=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = Registry(ROOT / "skill_catalog")
        if region is None:
            from ..demo_scene import demo_region
            region = demo_region()
        self.region = copy.deepcopy(region)

    def task_budgets(self, context):
        # Resource bounds must not encode the number of legs in a particular task.
        # The bounded graph controls when captures are requested.
        return {"normal": {"used": context.get("normal_capture_count", 0), "limit": 0},
            "recovery": {"used": context.get("recovery_capture_count", 0), "limit": 0},
            "decision_limit": 0}


    def is_observation_pose(self, context, observation=None):
        poses = observation_poses(context["planning_settings"])
        current = observation if observation is not None else self.simulation.joint_feedback(self.state['generation'])
        for arm, goal in poses.items():
            joints = np.asarray(current.get('arms', {}).get(arm, {}).get('joints_rad'), dtype=float)
            if joints.shape != (7,) or not np.allclose(joints, goal, atol=np.deg2rad(1), rtol=0):
                return False
        return True

    def require_observation_pose(self, context, observation=None):
        if not self.is_observation_pose(context, observation):
            raise ValueError("mapping_requires_observation_pose")

    def require_grasp_view(self, context, observation=None):
        capture = observation if observation is not None else context.get('unpacked_grasp_capture', context.get('observation'))
        current = self.simulation.joint_feedback(self.state['generation'])
        if (not capture or not capture.get('observation_id')
                or capture.get('generation') != self.state.get('generation')
                or current.get('generation') != self.state.get('generation')):
            raise ValueError('grasp_view_changed')
        for arm in ('A', 'B'):
            measured = np.asarray(capture.get('arms', {}).get(arm, {}).get('joints_rad'), float)
            live = np.asarray(current.get('arms', {}).get(arm, {}).get('joints_rad'), float)
            if (measured.shape != (7,) or live.shape != (7,) or not np.isfinite(measured).all()
                    or not np.isfinite(live).all() or not np.allclose(live, measured, atol=np.deg2rad(1), rtol=0)):
                raise ValueError('grasp_view_changed')
        product_capture = context.get('unpacked_grasp_capture')
        if product_capture is not None and (
                self.state.get('target_id') != context.get('unpacked_grasp_target_id')
                or product_capture.get('observation_id') != context.get('post_unpack_capture', {}).get('observation_id')
                or capture['observation_id'] != product_capture.get('observation_id')):
            raise ValueError('grasp_view_changed')

    async def plan_grasp(self, snapshot, target, arm, context):
        self.require_grasp_view(context)
        self.require_grasp_view(context, snapshot)
        capture_id = context.get('observation_pose_capture_id')
        if not capture_id or any(frame.get('observation_id') != capture_id
                for frame in (snapshot, context.get('observation', {}), self.spatial_memory.scene)):
            raise ValueError('grasp_requires_current_observation_scene')
        result = await super().plan_grasp(snapshot, target, arm, context)
        return {**result, "sampling_evidence": copy.deepcopy(context["sampling_evidence"])}

    async def generate_bound_grasps(self, context, observation):
        """Each graph generation node requests one pool for the current transfer input."""
        binding = self.motion_plan_binding(context, observation=observation)
        placement = context['grasp_placement_request']
        budget = context.get('grasp_sampling_budget')
        if budget is None or budget['binding'] != binding or budget['placement'] != placement:
            target = observed_object(observation, binding.target_id)
            budget = {'binding': binding, 'placement': copy.deepcopy(placement), 'issued': 0,
                      'limit': 1 if target.get('cad') is not None else MAX_SAMPLING_ROUNDS,
                      'retry_allowed': False, 'rounds': []}
            context['grasp_sampling_budget'] = budget
            context.pop('sampling_evidence', None)
        if budget['issued'] >= budget['limit']:
            raise AlgorithmError('grasp_sampling_budget_exhausted', code='grasp_sampling_budget_exhausted')
        if budget['issued'] and not budget['retry_allowed']:
            raise AlgorithmError('grasp_pool_exhaustion_unconfirmed', code='grasp_pool_exhaustion_unconfirmed')
        budget['issued'] += 1
        budget['retry_allowed'] = False
        try:
            grasps = await self.call_algorithm(self.algorithms.generate_grasps,
                observation, binding.target_id, arm=binding.arm, placement_request=placement,
                planning_settings=context['planning_settings'], sampling_round=budget['issued'], report_progress=True)
        except AlgorithmError as error:
            # A completely filtered TSDF pool can fail during generation, before a planning node.
            detail = error.details or {}
            rejected = detail.get('rejected_candidates') or []
            rejected_indices = {row['index'] for row in rejected if isinstance(row, dict)
                                and type(row.get('index')) is int and row.get('reason')}
            budget['retry_allowed'] = (error.code == 'no_tianji_contact_candidate'
                and detail.get('prepared_count') == 0
                and detail.get('input_count') == len(rejected_indices) == len(rejected) > 0)
            if budget['retry_allowed']:
                budget['rounds'].append({'sampling_round': budget['issued'], 'outcome': 'no_tianji_contact_candidate',
                    'requested_candidate_count': grasp_candidate_count(budget['issued']),
                    'generated_candidate_count': len(rejected), 'prepared_candidate_count': 0,
                    'tested_count': 0, 'valid_count': 0, 'source_ik_success_count': 0,
                    'rejected_candidates': copy.deepcopy(rejected)})
            raise
        self.check_generation()
        binding.require_current(self.motion_plan_binding(context))
        identity = grasp_input_identity(grasps)
        if grasps['sampling_round'] != budget['issued'] or identity != budget.get('input', identity):
            raise AlgorithmError('grasp_sampling_input_changed', code='planning_stale_evaluation')
        budget['input'] = identity
        return grasps

    async def plan_grasp_candidates(self, snapshot, target, arm, context):
        """Evaluate only the pool supplied by the graph's generate_grasps node."""
        budget = context['grasp_sampling_budget']
        grasps = context['grasps']
        scene = self.spatial_memory.scene
        settings = context['planning_settings']
        destination = self.state['destination_id']
        sampling_round = budget['issued']
        def check_current():
            self.check_generation()
            budget['binding'].require_current(self.motion_plan_binding(context))
            if budget['placement'] != context.get('grasp_placement_request'):
                raise AlgorithmError('grasp_sampling_input_changed', code='planning_stale_evaluation')

        check_current()
        if (budget['retry_allowed'] or grasps.get('sampling_round') != sampling_round
                or budget['input'] != grasp_input_identity(grasps)):
            raise AlgorithmError('grasp_pool_changed_or_already_spent', code='planning_stale_evaluation')
        if grasps['candidates']:
            evaluations = await self.call_algorithm(self.algorithms.evaluate_arms, snapshot, target, grasps, scene,
                requested_arm=arm, placement_request=budget['placement'], planning_settings=settings,
                **({'destination_id': destination} if destination else {}), report_progress=True)
            check_current()
            evaluation = evaluations.get(arm, {})
            if not isinstance(evaluation.get('valid_candidates'), list):
                raise AlgorithmError('source IK outcome is unknown', code='invalid_source_ik_evaluation')
            validate_sampling_evaluation(evaluation, snapshot, target, arm, scene, settings, destination, grasps)
        else:
            evaluation = {'valid_candidates': [], 'tested_count': 0}
        record = round_evidence(grasps, evaluation, sampling_round)
        rounds = [*budget['rounds'], record]
        evidence = {'rounds': rounds, 'maximum_sampling_rounds': budget['limit'], 'arm': arm,
                    'target_id': target, 'scene_token': scene['scene_token'],
                    'total_requested_candidate_count': sum(r['requested_candidate_count'] for r in rounds),
                    'total_tested_count': sum(r['tested_count'] for r in rounds),
                    'trajectory_attempt_count': sum(bool(r['valid_count']) for r in rounds)}
        context['sampling_evidence'] = evidence
        failure = None
        if not grasps['candidates']:
            outcome, code, filtered = empty_pool_filter(grasps)
            record.update(outcome=outcome, empty_pool_filter=copy.deepcopy(filtered))
            self.emit('grasp_pool_filtered_empty', '本轮抓姿全部被几何筛选拒绝', evidence)
        elif evaluation['valid_candidates']:
            record['outcome'] = 'planning'
            evaluation['sampling_evidence'] = copy.deepcopy(evidence)
            try:
                result = await self.call_algorithm(self.algorithms.plan_grasp, snapshot, target, arm, evaluations,
                    scene, planning_settings=settings, **({'destination_id': destination} if destination else {}),
                    report_progress=True)
            except AlgorithmError as error:
                check_current()
                if not validate_planning_exhaustion(error, evaluation):
                    raise
                failure = error
                code = error.code
                record.update(outcome='determinate_planner_failure', planner_failure=copy.deepcopy(error.details))
            else:
                check_current()
                record['outcome'] = 'planned'
                return result
        else:
            screen = evaluation.get('destination_screening') or {}
            if screen.get('applied') and screen.get('input_count', 0) > 0:
                code = 'grasp_destination_samples_exhausted'
                record['outcome'] = 'no_sampled_destination_endpoint'
                evidence['whole_destination_infeasible'] = False
            else:
                code = 'grasp_candidate_preconditions_rejected' if record['source_ik_success_count'] else 'no_collision_free_source_ik'
                record['outcome'] = 'no_eligible_source_ik'
        budget.update(retry_allowed=True, rounds=rounds)
        self.state['facts']['grasps_ready'] = False
        details = {**failure.details, 'sampling_evidence': evidence} if failure is not None else evidence
        raise AlgorithmError('no valid grasp in the current bounded candidate pool', code=code, details=details) from failure


    async def capture_scene_observation(self, context, arguments):
        capture = context.get('unpacked_grasp_capture')
        if capture is None:
            capture = context.pop('fine_capture', None)
        if arguments.get('capture_mode') == 'fresh':
            capture = None
        return await self.capture_observation(context, capture=capture)

    async def execute(self, skill, context, arguments=None):
        from .failure_return import (
            build_failure_scene,
            observe_after_failure,
            prepare_failure_return,
            return_after_failure,
        )
        terminal = {'prepare_failure_return': prepare_failure_return,
                    'observe_after_failure': observe_after_failure,
                    'build_failure_scene': build_failure_scene,
                    'return_after_failure': return_after_failure}
        if skill in terminal:
            return await terminal[skill](self, context)
        if skill == 'recover_failed_grasp':
            from .grasp_recovery import recover_failed_grasp
            return await recover_failed_grasp(self, context)
        if skill == 'prepare_unpack':
            return await self._prepare_unpack(context)
        if skill == 'vla_unpack':
            return await self._vla_unpack(context)
        if skill == 'verify_unpack':
            return await self._verify_unpack(context)
        if skill in {"move_to_observe", "return_to_initial_pose"}:
            if skill == 'return_to_initial_pose':
                from .return_pose import return_to_initial_pose
                return await return_to_initial_pose(self, context)
            prepare_empty_hand_motion_scene(context, collision=self.spatial_memory.collision_snapshot())
            from ..observation_motion import observation_viewpoint_poses

            viewpoint = (arguments or {}).get('viewpoint', 'configured')
            if viewpoint == 'configured':
                return {**await self.move_to_observe(context), 'viewpoint': viewpoint}
            goals = observation_viewpoint_poses(context['planning_settings'], viewpoint)
            if viewpoint != 'configured':
                feedback = self.simulation.robot_snapshot()['arms']
                if all(np.max(np.abs(np.asarray(feedback[arm]['joints_rad']) - goals[arm]))
                       <= np.deg2rad(.05) for arm in ('A', 'B')):
                    raise ValueError('observation_viewpoint_unchanged')
            result = await self.move_to_observe(context, joint_goals=goals)
            return {**result, 'viewpoint': viewpoint, 'requested_joints_rad': goals}
        if skill == "observe_scene":
            capture = await self.capture_scene_observation(context, arguments or {})
            if (arguments or {}).get('full_collision_survey'):
                capture = {**capture, 'full_collision_survey': True}
            resources = await self.call_algorithm(self.algorithms.prepare_observation_memory)
            result = await self.observe(context, checkpoint="initial", capture=capture)
            result['observation_gpu_memory'] = resources
            context["observation_pose_capture_id"] = result["observation_id"]
            context["observation_pose_reference"] = {
                key: copy.deepcopy(result[key]) for key in ("observation_id", "time", "arms")}
            self._check_source_view(result)
            return result
        if skill == "build_scene":
            source = observed_object(context["observation"], self.state["target_id"])
            evidence = source_view_evidence(context["observation"], source)
            if not evidence["views_ready"]:
                raise ValueError("current_target_geometry_required")
            context["view_evidence"] = evidence
            self.state["facts"]["views_ready"] = True
            return await self.build_scene(context, "selected")
        if skill == "generate_grasps":
            self.require_grasp_view(context)
            if not self.state["facts"].get("views_ready"):
                raise ValueError("move_to_observe_required")
            observation = context["observation"]
            destination = observed_object(observation, self.state['destination_id'])
            if destination.get('kind') == 'bin':
                context['grasp_placement_request'] = {'object_id': self.state['target_id'],
                    'release_mode': self.task_progress.records[self.state['target_id']]['release_mode'],
                    'transport_region': container_transport_region(destination, self.spatial_memory.scene, observation)}
            else:
                context['grasp_placement_request'] = copy.deepcopy(self._bind_region(observation))
            context["grasps"] = await self.generate_bound_grasps(context, observation)
            context["grasps_observation_id"] = observation["observation_id"]
            return context["grasps"]
        if skill == "execute_place":
            result = await super().execute(skill, context, arguments)
            context["demo_place_execution"] = copy.deepcopy(result)
            context['release_execution_evidence'] = freeze_release_execution(
                self.state, context, self.run_id, scene=self.spatial_memory.scene)
            self.holding_state = None
            context.pop('holding', None)
            return result
        if skill == "verify_place":
            target = self.state["target_id"]
            declared_mode = (arguments or {}).get('release_mode', 'bound')
            actual_mode = context.get('release_execution_evidence', {}).get('release_reference', {}).get('release_mode')
            if declared_mode != 'bound' and declared_mode != actual_mode:
                raise ValueError('release_verification_mode_mismatch')
            completed = context.get('place_verification', {})
            receipt = completed.get('container_entry', {})
            if completed.get('success') is True and receipt.get('target_id') == target:
                release_execution_context(self.state, context, self.run_id)
                if receipt.get('destination_id') != self.state['destination_id']:
                    raise ValueError('container_entry_receipt_binding_changed')
                return copy.deepcopy(completed)
            previous = context.get('release_verification_attempt')
            if previous and previous['target_id'] == target:
                carried = copy.deepcopy(previous['carried'])
                result = copy.deepcopy(previous['physical_release'])
            else:
                carried = copy.deepcopy(release_execution_context(self.state, context, self.run_id).get('holding'))
                result = await self.verify_place(context)
                if not result["success"]:
                    result['verification_steps'] = {'holding_presence': 'not_checked', 'destination': 'not_checked'}
                    return result
                result["verified_perf_counter_ns"] = time.perf_counter_ns()
                self.task_progress.record_release(target, feedback_id=result["feedback_id"])
                context['release_verification_attempt'] = {
                    'target_id': target, 'carried': copy.deepcopy(carried),
                    'physical_release': copy.deepcopy(result)}
            # Rechecks retain the original physical release. A view change
            # cannot create a new release or erase the held/attachment proof.
            context['holding'] = carried
            binding = build_release_binding(self.state, context,
                self.task_progress.release_snapshot(target, self.state['destination_id']),
                self.run_id, result, scene=self.spatial_memory.scene)
            self.emit("release_verification_binding", "放置结果核验的执行链与目标区域", binding)
            capture = await self.sample_gripper(context, binding['arm'], binding['release_feedback_time'], 'verify_place')
            validate_release_capture(capture, binding)
            context['post_place_capture'] = capture
            after = await self.call_algorithm(self.algorithms.verify_release, capture, binding,
                context["identity_reference"], self.state["instruction"])
            if after.get('observation_assessment', {}).get('needs_observation'):
                self.emit('visual_transfer_result', '释放后的夹爪状态需要补充观测', after)
                context['use_retained_scene_for_return'] = True
                after['observation_assessment']['retry_owner'] = 'policy_graph'
            self.check_generation()
            current = build_release_binding(self.state, {**context, "holding": carried},
                self.task_progress.release_snapshot(target, self.state['destination_id']),
                self.run_id, result, scene=self.spatial_memory.scene)
            if current != binding or after.get("binding_id") != binding["binding_id"]:
                raise ValueError("release_verification_binding_changed")
            evidence = after["visual_evidence"]
            result['verification_steps'] = copy.deepcopy(evidence['verification_steps'])
            from ..surface_placement import placement_evidence_id
            if (after.get("observation_id") != capture["observation_id"]
                    or evidence.get("observation_id") != capture["observation_id"]
                    or evidence.get("binding_id") != binding["binding_id"]
                    or evidence.get("release_feedback_id") != result["feedback_id"]
                    or evidence.get("target_id") != target
                    or evidence.get("destination_id") != binding["destination_id"]
                    or evidence.get("layout_revision") != binding["layout_revision"]
                    or evidence.get("identity_source") != "execution_chain"
                    or evidence.get("source") != "bound_release_execution"
                    or not placement_evidence_id(evidence)):
                raise ValueError("visual_identity_progress_frame_mismatch")
            confidence = evidence.get("confidence")
            if confidence is not None and (type(confidence) not in (int, float) or not 0. <= confidence <= 1.):
                raise ValueError("release_verification_result_confidence")
            if (evidence.get('status') == 'at_destination'
                    and evidence.get('placement_outcome') != 'execution_assumed'):
                raise ValueError('release_verification_placement_outcome')
            context["post_place_result"] = after
            released = evidence.get('judgments', {}).get('target_released', 'unknown')
            if released != 'yes':
                context['holding'] = {**carried, 'state': 'release_unconfirmed',
                    'verified': False, 'attached': False, 'release_visual_status': released}
                self.holding_state = copy.deepcopy(context['holding'])
            else:
                context.pop('holding', None)
                self.holding_state = None
            context["visual_result"] = copy.deepcopy(evidence)
            self.state["visual_result"] = copy.deepcopy(evidence)
            context["post_place_verification_input"] = {
                "observation_id": capture["observation_id"],
                "visual_observation_id": evidence["observation_id"], "binding_id": binding["binding_id"]}
            self.sync_result_evidence(context, "post_place")
            self.emit("visual_transfer_result", "放置结果核验", after)
            zone = binding.get('placement_zone', {})
            if (zone.get('intent', {}).get('release_mode') == 'container_release'
                    and zone.get('semantic_scope') == 'entire_region'
                    and evidence.get('status') == 'at_destination'):
                from ..demo_transfer_verification import container_entry_receipt

                reference = context['release_execution_evidence']['release_reference']
                receipt = container_entry_receipt(binding, reference, capture, evidence)
                context['container_entry_receipt'] = copy.deepcopy(receipt)
                previous_scene = self.spatial_memory.scene
                if not released_target_map_is_current(context, target, collision=self.spatial_memory.collision_snapshot()):
                    updated_scene = await self.call_algorithm(
                        self.algorithms.publish_container_occupancy, previous_scene, receipt)
                    self.check_generation()
                    self.spatial_memory.accept_container_occupancy(updated_scene, receipt,
                        previous_scene_token=previous_scene['scene_token'])
                confirmed = self.task_progress.confirm_container_transfer(target, receipt)
                if not confirmed:
                    raise ValueError('container_entry_receipt_binding_changed')
                result.update(success=True, status='confirmed', verification_scope='bound_container_release', visual_evidence=evidence,
                              container_entry=receipt, placement_outcome=evidence['placement_outcome'], reason=None)
                context['pending_transfer']['verified'] = True
                context['place_verification'] = result
                context['verified_release_reference'] = copy.deepcopy(reference)
                self.state['facts']['table_release_verified'] = False
                self.state['facts']['scene_available'] = not self.spatial_memory.refresh_required
                self.emit('container_entry_confirmed', '已在目标盒释放，等待更新碰撞几何', receipt)
                self.sync_memory(context)
                return result
            confirmed = self.task_progress.confirm_visual_transfer(target, evidence)
            rejected = evidence.get('status') == 'rejected'
            result.update(success=confirmed, status="confirmed" if confirmed else "failure" if rejected else "unknown",
                placement_outcome=evidence['placement_outcome'],
                verification_scope="bound_release_and_empty_gripper", visual_evidence=evidence,
                reason=None if confirmed else "object_still_held_after_release" if rejected
                else "release_presence_unknown")
            result['observation_assessment'] = copy.deepcopy(after.get('observation_assessment', {}))
            if 'pending_transfer' in context:
                context['pending_transfer']['verified'] = confirmed
            context["place_verification"] = result
            context["demo_objects_check"] = after.get("demo_objects_check", {})
            if confirmed:
                context['verified_release_reference'] = copy.deepcopy(
                    context['release_execution_evidence']['release_reference']
                    if 'release_execution_evidence' in context else context['place_plan']['goal']['release_reference'])
                self.state['facts']['table_release_verified'] = (
                    evidence.get('destination_id') == self.task_bindings.get('work_area_id'))
                self.sync_memory(context)
                if not carried:
                    raise ValueError("release_geometry_missing")
                if not released_target_map_is_current(context, target, collision=self.spatial_memory.collision_snapshot()):
                    previous_scene = self.spatial_memory.scene
                    updated_scene = await self.call_algorithm(self.algorithms.invalidate_scene_entity,
                        previous_scene, target, result['feedback_id'], source='verified_release_action')
                    self.spatial_memory.replace_geometry(updated_scene, previous_scene_token=previous_scene['scene_token'])
                    self.spatial_memory.invalidate([target])
                    self.state['facts'].update(scene_available=False, scene_ready=False, observation_map_ready=False)
                    self.emit("retained_scene_invalidated", "释放后几何失效，等待模型选择实测建图", {
                        "scene_token": self.spatial_memory.scene["scene_token"], "target_id": target,
                        "source": "verified_release_action", "rgbd_remapped": False})
            return result
        return await super().execute(skill, context, arguments)
