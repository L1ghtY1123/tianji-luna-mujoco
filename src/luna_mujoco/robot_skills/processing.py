"""Independent configured unpack preparation, effect and result-verification skills."""
import asyncio
import copy
from uuid import uuid4

import numpy as np

from ..vla_configuration import vla_unpack_configuration
from .evidence import unpack_table_support, validate_table_release_reference


def create_unpack_product(input_id, action, result, capture, *, owner_id):
    """Create a distinct identity only from a bound current visual result."""
    visual = result.get('visual_evidence', {})
    view_valid = visual.get('view_scope') == 'fused_current_three_rgb'
    if visual.get('view_scope') == 'independent_current_three_rgb':
        witnesses = visual.get('camera_evidence', [])
        selected = [row for row in witnesses if row.get('camera_id') == visual.get('selected_camera_id')]
        view_valid = (len(witnesses) == 3 and {row.get('camera_id') for row in witnesses} == {'1', '2', '3'}
            and len(selected) == 1 and selected[0].get('request_hash') == result.get('request_hash')
            and selected[0].get('observation_id') == capture.get('observation_id')
            and selected[0].get('required_judgments') == {'unpack_result': 'yes', 'wrapper_removed': 'yes'}
            and selected[0].get('confidence') == visual.get('confidence'))
    if (not owner_id or result.get('success') is not True or result.get('status') != 'confirmed'
            or result.get('target_id') != input_id or not result.get('request_hash')
            or result.get('observation_id') != capture.get('observation_id')
            or result.get('generation') != capture.get('generation')
            or action.get('target_id') != input_id or action.get('owner_id') != owner_id
            or action.get('generation') != capture.get('generation')
            or action.get('completed') is not True or not action.get('process_id')
            or not action.get('action_id') or visual.get('action_id') != action['action_id']
            or visual.get('judgments') != {'unpack_result': 'yes'}
            or not view_valid
            or type(visual.get('confidence')) not in (int, float) or not .8 <= visual['confidence'] <= 1.):
        raise ValueError('product_requires_confirmed_visible_result')
    identifier = 'product-' + uuid4().hex
    evidence = {'source': 'confirmed_unpack_process_output', 'processing_skill': 'vla_unpack',
        'process_id': action['process_id'], 'input_entity_id': input_id, 'output_entity_id': identifier,
        'owner_id': owner_id, 'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'request_hash': result['request_hash'], 'action_id': action['action_id'],
        'relationship': 'processing_input_and_output_are_distinct_entities'}
    return {'id': identifier, 'kind': 'object', 'color': 'black',
        'description': 'separate black unpacked product on the measured work support',
        'process_id': action['process_id'], 'input_entity_id': input_id, 'processing_skill': 'vla_unpack',
        'identity_source': 'confirmed_current_product_creation', 'creation_evidence': evidence,
        'visual_membership': 'required', 'roles': ['product', 'collision'],
        'segmentation_query': {'kind': 'object', 'color': 'black', 'prompt': 'black object'}}


class ProcessingSkills:
    def _verified_processing_release(self, context):
        target = self.state['target_id']
        verification = context.get('place_verification', {})
        release = verification.get('visual_evidence', {})
        pending = context.get('pending_transfer')
        if pending is None:
            from .evidence import table_release_reference
            expected = table_release_reference(context.get('post_place_capture'), release, self.run_id)
            if context.get('post_place_verification_input') != expected:
                raise ValueError('unpack_requires_verified_table_release_and_empty_hands')
            pending = {'target_id': release.get('target_id'),
                       'destination_id': release.get('destination_id'), 'verified': True}
        work_area_id = self.task_inventory.get('work_area_id')
        if (context.get('holding') or verification.get('success') is not True
                or release.get('status') != 'at_destination' or release.get('target_id') != target
                or release.get('destination_id') != work_area_id or not work_area_id
                or pending.get('target_id') != target or pending.get('destination_id') != work_area_id
                or pending.get('verified') is not True or not release.get('release_feedback_id')
                or context.get('post_place_capture', {}).get('generation') != self.state['generation']):
            raise ValueError('unpack_requires_verified_table_release_and_empty_hands')
        requirement = getattr(self, 'environment_assumptions', {}).get(
            'completion_requirements', {}).get('after_each_object', {})
        if requirement:
            proof = self.task_progress.records.get(target, {}).get('post_delivery_observation', {})
            delivery = proof.get('delivery', {})
            if (proof.get('status') != 'verified' or proof.get('generation') != self.state['generation']
                    or delivery.get('entity_id') != target
                    or delivery.get('confirmed_observation_id') != release.get('observation_id')
                    or requirement.get('observe_inventory') and proof.get('inventory_observed') is not True):
                raise ValueError('processing_requires_post_delivery_observation')
        return release

    def _processing_work_area(self, context):
        destination = context.get('destination', {})
        reference = destination.get('placement_reference', {})
        if destination.get('id') != self.task_inventory.get('work_area_id'):
            raise ValueError('unpack_work_area_binding_changed')
        polygon = np.asarray(reference.get('region_polygon_world_xy_m'), float)
        floor = reference.get('floor_z_m')
        if (reference.get('position_source', destination.get('position_source')) == 'simulation_truth'
                or not reference.get('observation_id') or polygon.shape != (4, 2)
                or not np.isfinite(polygon).all() or np.any(np.ptp(polygon, axis=0) <= 0)
                or type(floor) not in (int, float) or not np.isfinite(floor)):
            raise ValueError('unpack_requires_measured_work_area')
        return {'center_world_xy_m': ((polygon.min(0) + polygon.max(0)) / 2).tolist(),
            'region_polygon_world_xy_m': polygon.tolist(), 'floor_z_m': floor,
            'source': 'configured_xy_measured_support', 'source_observation_id': reference['observation_id']}

    def _processing_support(self, context):
        observation = context['observation']
        support_id = (self.task_inventory.get('support_id')
                      or observation.get('demo_tables', {}).get('primary_table_id'))
        # Output is sampled over the supported tabletop, not the input square.
        # This comes only from measured table evidence, never the effect receipt.
        support = unpack_table_support(observation, support_id)
        known = []
        objects = {item['id']: item for item in observation.get('objects', [])}
        for identifier, entity in self.task_progress.scene_entities.items():
            if identifier == self.state['target_id'] or entity.get('kind') != 'object' or entity.get('color') != 'black':
                continue
            measured = objects.get(identifier, {})
            points = np.asarray(measured.get('points_world_m'), float)
            if (measured.get('position_source') == 'simulation_truth' or points.ndim != 2
                    or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all()):
                raise ValueError('unpack_existing_black_identity_requires_measured_geometry:' + identifier)
            known.append({'id': identifier, 'minimum_world_m': points.min(axis=0).tolist(),
                          'maximum_world_m': points.max(axis=0).tolist()})
        support['known_black_objects'] = known
        return support

    def _vla_input_requirements(self, context):
        config = vla_unpack_configuration(context.get('planning_settings'))
        area = self._processing_work_area(context)
        reference = context.get('verified_release_reference',
            context.get('place_plan', {}).get('goal', {}).get('release_reference', {}))
        endpoint = context.get('release_endpoint_evidence', {})
        object_xy = np.asarray(endpoint.get('actual_object_reference_world_xy_m'), float)
        angle = endpoint.get('rotation_error_rad')
        expected = np.asarray(config['input_xy_m'])
        entity = self.task_progress.scene_entities.get(self.state['target_id'], {})
        if (entity.get('color') != 'pink' or reference.get('object_id') != self.state['target_id']
                or object_xy.shape != (2,)
                or not np.isfinite(object_xy).all()
                or np.any(np.abs(object_xy - expected) > config['input_region']['size_m'] / 2)
                or type(angle) not in (int, float) or not np.isfinite(angle)
                or not 0 <= angle <= config['input_orientation_tolerance_rad']
                or not np.allclose(area['region_polygon_world_xy_m'],
                                   config['input_region']['region_polygon_world_xy_m'], atol=1e-9, rtol=0)
                or np.linalg.norm(np.asarray(area['center_world_xy_m']) - expected) > 1e-9
                or context.get('release_endpoint_evidence', {}).get('at_release_goal') is not True):
            raise ValueError('vla_input_requires_verified_package_in_configured_square_with_preserved_orientation')
        return area

    def _require_vla_preparation(self, context):
        config = vla_unpack_configuration(context.get('planning_settings'))
        feedback = self.simulation.robot_snapshot()
        if feedback.get('generation') != self.state['generation']:
            raise ValueError('vla_preparation_feedback_stale')
        for arm, degrees in config['preparation_joints_deg'].items():
            actual = np.asarray(feedback.get('arms', {}).get(arm, {}).get('joints_rad'), float)
            if (actual.shape != (7,) or not np.isfinite(actual).all()
                    or not np.allclose(actual, np.deg2rad(degrees), atol=np.deg2rad(1), rtol=0)):
                raise ValueError('vla_preparation_pose_changed:' + arm)

    def verify_unpacking(self, capture, **kwargs):
        """Keep the visual client outside the algorithm's archived arguments."""
        from .processing_evidence import verify_unpacking

        return verify_unpacking(self.visual_inventory, capture, **kwargs)

    async def _prepare_unpack(self, context):
        release = self._verified_processing_release(context)
        self._vla_input_requirements(context)
        observation = self.measured_state(context)
        goals = vla_unpack_configuration(context['planning_settings'])['preparation_joints_deg']
        scene = context['scene'] | {'return_after_place': True}
        plan = await self.call_algorithm(self.algorithms.plan_dual_vla_motion,
            observation, scene, goals, planning_settings=context['planning_settings'])
        feedback = self.simulation.robot_snapshot()
        paths = {}
        for arm in ('A', 'B'):
            path = np.asarray(plan['paths'][arm], float)
            current = feedback['arms'][arm]['joints_rad']
            if np.max(np.abs(path[0] - current)) > np.deg2rad(1):
                raise ValueError('vla_pose_trajectory_start_changed')
            paths[arm] = np.vstack((current, path)).tolist()
        await self.call_algorithm(self.algorithms.validate_dual_observation,
            self.measured_state(context), scene, paths, planning_settings=context['planning_settings'])
        result = await self.execution_stage(context, 'vla_pose_AB', self.motor,
            self.simulation.execute_dual_trajectory, paths, plan['dt'])
        feedback = self.simulation.robot_snapshot()
        if feedback.get('generation') != self.state['generation']:
            raise ValueError('vla_preparation_feedback_stale')
        for arm, degrees in goals.items():
            actual = np.asarray(feedback['arms'][arm]['joints_rad'], float)
            if (actual.shape != (7,) or not np.isfinite(actual).all()
                    or not np.allclose(actual, np.deg2rad(degrees), atol=np.deg2rad(1), rtol=0)):
                raise ValueError('vla_preparation_goal_unconfirmed:' + arm)
        context['vla_pose_execution'] = {'execution': result, 'target_id': self.state['target_id'],
            'release_feedback_id': release['release_feedback_id'], 'generation': self.state['generation'],
            'goal_joints_deg': goals}
        self.state['facts'].update(observation_pose_ready=False, grasp_view_ready=False, observation_map_ready=False,
            observed=False, views_ready=False, scene_ready=False)
        return {'success': True, 'plan': plan, 'execution': result, 'rgbd_remapped': False,
                'goal_joints_deg': goals}

    async def _vla_unpack(self, context):
        release = self._verified_processing_release(context)
        preparation = context.get('vla_pose_execution', {})
        if (preparation.get('target_id') != self.state['target_id']
                or preparation.get('release_feedback_id') != release['release_feedback_id']
                or preparation.get('generation') != self.state['generation']):
            raise ValueError('mock_unpack_requires_completed_vla_pose')
        if context.get('processing_action'):
            raise ValueError('processing_action_already_dispatched')
        area = self._vla_input_requirements(context)
        self._require_vla_preparation(context)
        polygon = np.asarray(area['region_polygon_world_xy_m'])
        work_area = {**area, 'size_m': float(np.ptp(polygon[:, 0]))}
        self.check_generation()
        step = self.state.get('current_step')
        if step is not None:
            step.setdefault('effect_commands', []).append('vla_unpack')
        receipt = await asyncio.to_thread(self.unpack_effect.execute, work_area,
            generation=self.state['generation'], target_id=self.state['target_id'], owner_id=self.run_id)
        self.check_generation()
        receipt.update(table_release_feedback_id=release['release_feedback_id'],
            process_id='process-' + uuid4().hex, processing_skill='vla_unpack')
        context['processing_action'] = copy.deepcopy(receipt)
        context['scene_refresh_required'] = True
        lifecycle = context.setdefault('mapping_lifecycle', {})
        lifecycle['invalidated_entity_ids'] = sorted(
            set(lifecycle.get('invalidated_entity_ids', ())) | {self.state['target_id']})
        self.state['facts'].update(scene_ready=False, scene_available=False, observation_map_ready=False)
        # The effect invalidates prior object poses. New map publication belongs
        # to the graph; the effect receipt supplies no measured geometry.
        self.emit('mock_vla_completed', '模拟 VLA：粉色包装变为黑色内件', receipt)
        return {'success': True, 'receipt': receipt, 'rgbd_remapped': False}

    async def _verify_unpack(self, context):
        target = self.state['target_id']
        action = context.get('processing_action', {})
        if action.get('target_id') != target or action.get('processing_skill') != 'vla_unpack':
            raise ValueError('unpack_requires_bound_processing_action')
        release = self._verified_processing_release(context)
        context.pop('fine_capture', None)
        before = context.get('post_place_capture')
        reference = context.get('post_place_verification_input')
        work_area = self._processing_support(context)
        capture = await self.capture_observation(context)
        context.update(post_unpack_capture=capture, latest_capture=capture)
        self.record_observation(capture, 'post_unpack')
        validate_table_release_reference(before, reference, capture,
            release, action, self.run_id)
        result = await self.call_algorithm(self.verify_unpacking,
            capture, target_id=target, release_evidence=copy.deepcopy(release),
            action_evidence=copy.deepcopy(action), owner_id=self.run_id, work_area=work_area,
            table_release_capture=before, table_release_reference=copy.deepcopy(reference), active_target_ids=[target])
        self.emit('visual_unpack_result', 'Decisions：拆包结果与观测充分性', result)
        # Verification supplies current visual evidence, not a mapping policy.
        # A graph-selected build_scene must decide scope and publish fresh TSDF/ESDF.
        context['scene_refresh_required'] = True
        self.state['facts'].update(scene_ready=False, scene_available=False, observation_map_ready=False)
        if result['success']:
            try:
                measured = await self.call_algorithm(self.algorithms.measure_local_object,
                    capture, work_area, 'unpacked')
            except ValueError as error:
                if str(error) != 'local_object_depth_missing':
                    raise
                result.update(success=False, status='unknown', reason='unpack_depth_unconfirmed_after_observation')
                result['measurement_diagnostics'] = copy.deepcopy(getattr(error, 'details', {}))
                result['observation_assessment'].update(needs_observation=True,
                    missing_evidence=['current_object_depth'], status='unknown')
            else:
                evaluations = {}
                for arm in ('A', 'B'):
                    evaluations[arm] = await self.call_algorithm(self.algorithms.measured_object_workspace,
                        measured, arm)
                feasible = [arm for arm, evidence in evaluations.items() if evidence['reachable']]
                workspace = {'reachable': bool(feasible), 'arm': None,
                    'arm_selection': 'pending_choose_arm', 'feasible_arms': feasible,
                    'evaluations': evaluations, 'observation_id': measured['observation_id'],
                    'source': 'current_rgbd_object_center_and_offline_workspace',
                    'scope': 'position_workspace_only_grasp_pose_and_path_not_yet_planned'}
                result['workspace_evidence'] = workspace
                result['visual_evidence']['workspace_evidence'] = workspace
                self.emit('unpack_workspace_checked', '实测黑件位置与执行臂工作空间核对', workspace)
        if result['success']:
            product = create_unpack_product(target, action, result, capture, owner_id=self.run_id)
            support_id = work_area.get('table_id') or work_area.get('support_entity_id')
            if support_id:
                product['observed_support'] = {'support_entity_id': support_id,
                    'observation_id': capture['observation_id'], 'request_hash': result['request_hash'],
                    'source': 'openai_decisions'}
            self.task_progress.register_product(product)
            reference = context['visual_inventory_reference']
            reference.setdefault('product_references', {})[product['id']] = {
                'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'verification_capture': copy.deepcopy(result['verification_capture']),
                'entity': copy.deepcopy(product), 'evidence': copy.deepcopy(product['creation_evidence']),
                'views': {camera: {'rgb': view['rgb']} for camera, view in capture['views'].items()}}
            self.state['target_id'] = product['id']
            self.sync_inventory(context)
            observation_pose_ready = self.state['facts'].get('observation_pose_ready', False)
            self.reset_transfer(context)
            self.state['target_id'] = product['id']
            context.update(unpacked_grasp_capture=capture, unpacked_grasp_target_id=product['id'])
            result = {**result, 'input_entity_id': target, 'output_entity_id': product['id'],
                      'product_id': product['id'],
                      'process_id': product['process_id']}
            self.emit('product_created', '已确认拆包产物：创建独立黑件实体', {
                **copy.deepcopy(product['creation_evidence']), 'product_entity': copy.deepcopy(product)})
            self.state['facts'].update(observation_pose_ready=observation_pose_ready, grasp_view_ready=False, observation_map_ready=False,
                observed=False, scene_available=False,
                inventory_finalized=True, task_assessed=True, needs_place=True)
        return result
