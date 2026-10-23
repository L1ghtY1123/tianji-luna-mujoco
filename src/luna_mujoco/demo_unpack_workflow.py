"""Explicit sequential unpacking stages; model decisions never move the simulator."""
import asyncio
import copy
import hashlib
import json
import time

import numpy as np

from .cad_assets import cad_asset
from .demo_bin_geometry import published_bin_shell_layers
from .demo_grasp import transport_tcp_height
from .demo_process import create_product, process_snapshot
from .geometry import front_tcp_offset, front_tcp_pose, rigid_matrix
from .placement import observed_object, region_contains_xy
from .verification import gripper_open_confirmed


def transport_plan_key(plan):
    bound = {key: plan.get(key) for key in ('goal', 'command', 'carrying_binding', 'planning_settings',
        'generation', 'observation_id', 'scene_token', 'arm', 'target_id')}
    return hashlib.sha256(json.dumps(bound, sort_keys=True, allow_nan=False,
        default=lambda value: value.tolist()).encode()).hexdigest()


def right_bin_occupied_bounds(destination, scene, observation):
    """Use the exact enabled bin's complete published cells, never raw mask outliers."""
    header = scene.get('collision_layer_bounds')
    if (not isinstance(header, dict) or not isinstance(header.get('layers'), dict)
            or not isinstance(scene.get('scene_token'), str) or not scene['scene_token']
            or not isinstance(scene.get('observation_id'), str) or not scene['observation_id']
            or type(scene.get('generation')) is not int or scene['generation'] < 0
            or any(header.get(key) != scene.get(key)
                   for key in ('scene_token', 'observation_id', 'generation'))
            or any(observation.get(key) != scene.get(key) for key in ('observation_id', 'generation'))):
        raise ValueError('drop_requires_current_bin_occupancy')
    identifier = destination.get('id')
    if destination.get('kind') != 'bin' or not isinstance(identifier, str) or not identifier:
        raise ValueError('drop_requires_complete_enabled_bin_occupancy')
    name = 'semantic_instance::' + identifier
    layer = header['layers'].get(name, {})
    lower = np.asarray(layer.get('minimum_world_m'), float)
    upper = np.asarray(layer.get('maximum_world_m'), float)
    voxel_size = layer.get('voxel_size_m')
    if (layer.get('kind') != 'voxel' or layer.get('enabled') is not True
            or name in scene.get('disabled_obstacle_names', [])
            or layer.get('source') != 'native_surface_voxel_cell_bounds'
            or layer.get('bounds_source') != 'all_native_surface_cells_before_preview_sampling'
            or type(layer.get('cell_count')) is not int or layer['cell_count'] <= 0
            or type(voxel_size) not in (int, float) or not np.isfinite(voxel_size) or voxel_size <= 0
            or lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower)):
        raise ValueError('drop_requires_complete_enabled_bin_occupancy')
    shell = published_bin_shell_layers(scene, identifier)
    if shell:
        lower = np.minimum(lower, np.min([part['minimum_world_m'] for part in shell.values()], axis=0))
        upper = np.maximum(upper, np.max([part['maximum_world_m'] for part in shell.values()], axis=0))
    return {'source': 'published_inventory_container_occupancy_bounds', 'entity_id': identifier,
        'scene_token': scene['scene_token'], 'observation_id': scene['observation_id'],
        'generation': scene['generation'], 'layer_name': name,
        'bounds_source': ('exact_inventory_native_cells_and_derived_shell_union' if shell else layer['bounds_source']),
        'layer_names': [name, *shell], 'derived_shell_components': copy.deepcopy(shell),
        'cell_count': layer['cell_count'], 'voxel_size_m': voxel_size,
        'minimum_world_m': lower.tolist(), 'maximum_world_m': upper.tolist(),
        'identity_source': destination.get('identity_source')}


def right_bin_transport_region(destination, scene, observation):
    """Bind the complete cavity to the latest measured and published bin pose."""
    evidence = right_bin_occupied_bounds(destination, scene, observation)
    if destination.get('placement_reference', {}).get('observation_id') != observation['observation_id']:
        raise ValueError('drop_requires_current_bin_occupancy')
    polygon = np.asarray(destination['placement_reference']['region_polygon_world_xy_m'], float)
    region_source = 'measured_visible_floor_patch'
    completion = scene.get('mapping_diagnostics', {}).get('bin_shell_completions', {}).get(destination['id'])
    if completion and completion.get('source') == 'known_cad_measured_registration':
        # The visible floor rectangle can be clipped by perspective or the
        # gripper. Use the complete known cavity at its current measured pose.
        # right_bin_occupied_bounds already validated all published CAD parts.
        asset = cad_asset('blue_box')
        if completion.get('asset_id') != asset['id'] or completion.get('cad_sha256') != asset['source']['sha256']:
            raise ValueError('drop_requires_current_bin_cad_registration')
        transform = rigid_matrix(completion['T_world_cad'])
        section = np.asarray(asset['cavity_bottom_section']['polygon_xy_m'], float)
        local = np.column_stack((section, np.full(len(section), asset['cavity_floor_z_m'])))
        polygon = (local @ transform[:3, :3].T + transform[:3, 3])[:, :2]
        region_source = 'registered_cad_cavity_bottom_section'
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError('drop_requires_finite_bin_and_carried_geometry')
    center = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
    # Robot +X points forward. Stay in the near half and move the finger-front
    # target toward the robot midline to avoid stretching the outer arm.
    center[0] -= .25 * np.ptp(polygon[:, 0])
    inset = center.copy()
    inset[1] += np.clip(-center[1], -.32 * np.ptp(polygon[:, 1]), .32 * np.ptp(polygon[:, 1]))
    if bool(region_contains_xy(inset[None], polygon)[0]):
        center = inset
    return {'center_world_xy_m': center.tolist(), 'target_reference': 'finger_front',
        'polygon_world_xy_m': polygon.tolist(), 'source': region_source,
        'occupied_top_z_m': evidence['maximum_world_m'][2],
        'observation_id': observation['observation_id'], 'occupancy': evidence}


def right_bin_transport_goal(lift_tcp, holding, destination, scene, observation):
    """Aim the finger front above the near half; attached collision checks stay in cuRobo."""
    region = right_bin_transport_region(destination, scene, observation)
    evidence = region['occupancy']
    polygon = np.asarray(region['polygon_world_xy_m'])
    goal = rigid_matrix(lift_tcp).copy()
    tool_points = np.asarray(holding['points_tool_m'], float)
    if tool_points.ndim != 2 or tool_points.shape[1] != 3 or not len(tool_points):
        raise ValueError('drop_requires_finite_bin_and_carried_geometry')
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not all(np.isfinite(a).all() for a in (polygon, tool_points))):
        raise ValueError('drop_requires_finite_bin_and_carried_geometry')
    held = tool_points @ goal[:3, :3].T
    arm = holding.get('arm', 'B')
    # SDK goals are TCP_Link (flange); the user's target is the finger front.
    # Convert the fixed front target, never aim the held-cloud centroid.
    front_offset = goal[:3, :3] @ front_tcp_offset(arm)
    goal[:2, 3] = np.asarray(region['center_world_xy_m']) - front_offset[:2]
    top = evidence['maximum_world_m'][2]
    minimum_height = float(top + .10 - held[:, 2].min())
    goal[2, 3] = transport_tcp_height(goal[:3, :3], held[:, 2].min(), top, holding.get('arm', 'B'))
    evidence.update(occupied_top_z_m=top, above_occupancy_allowance_m=.10,
        z_policy='shortest_shoulder_wrist_reach_above_minimum_bin_clearance', minimum_tcp_z_m=minimum_height,
        carried_lowest_offset_world_z_m=float(held[:, 2].min()), goal_tcp_z_m=float(goal[2, 3]),
        xy_policy='finger_front_at_robot_near_bin_half', goal_tcp_xy_m=goal[:2, 3].tolist(),
        goal_front_xy_m=front_tcp_pose(goal, arm)[:2, 3].tolist(),
        goal_front_z_m=float(front_tcp_pose(goal, arm)[2, 3]),
        region_source=region['source'], region_polygon_world_xy_m=polygon.tolist())
    return goal, evidence


class UnpackWorkflow:
    unpack_skills = frozenset({'prepare_unpack', 'mock_unpack', 'verify_unpack', 'plan_transport',
                              'execute_transport', 'drop', 'verify_drop'})

    @property
    def unpack_task(self):
        return bool(self.demo_task and self.demo_task.get('workflow') == 'unpack_transfer')

    @property
    def transfer_routes(self):
        # Demo owns observation-only mapping and explicit symbolic destinations.
        # The ledger still uses the existing two-leg completion accounting.
        return False if self.unpack_task else super().transfer_routes

    def current_lifecycle(self):
        return self.demo_task['item_lifecycle'][self.state['target_id']]

    def verify_unpacking(self, capture, **kwargs):
        """Keep the visual client outside the algorithm's archived arguments."""
        from .demo_lifecycle_perception import verify_unpacking

        return verify_unpacking(self.visual_inventory, capture, **kwargs)

    def verify_right_bin_drop(self, capture, **kwargs):
        """Archive only evidence data while retaining the bound visual client."""
        from .demo_lifecycle_perception import verify_right_bin_drop

        return verify_right_bin_drop(self.visual_inventory, capture, **kwargs)

    def sync_lifecycle(self, context):
        if self.demo_task.get('process_records') is not None:
            self.demo_task.update(process_snapshot(self.demo_task))
            self.task_progress.process_task = self.demo_task
            self.task_progress.sync_process_geometry()
            target = self.state.get('target_id')
            process = next((row for row in self.demo_task['process_records'].values()
                            if target in (row['input_entity_id'], row['output_entity_id'])), None)
            self.state['process_id'] = process['id'] if process else None
        self.state['demo_task'] = copy.deepcopy(self.demo_task)
        reference = context.get('visual_inventory_reference')
        if reference is not None:
            reference['demo_task'] = copy.deepcopy(self.demo_task)
        for identifier, lifecycle in self.demo_task['item_lifecycle'].items():
            if identifier in self.task_progress.records:
                self.task_progress.records[identifier]['lifecycle'] = copy.deepcopy(lifecycle)

    def set_observation_stage(self, observation):
        identifier = self.state.get('target_id') or self.task_progress.pending_route_id()
        lifecycle = self.demo_task['item_lifecycle'].get(identifier, {})
        observation['stage'] = lifecycle.get('appearance', 'wrapped')

    def measured_state(self, context):
        observation = super().measured_state(context)
        if self.unpack_task:
            self.set_observation_stage(observation)
        return observation

    async def supplement_result_observation(self, context, assessment, purpose):
        """One explicit, task-requested view change; never rebuild a global map here."""
        self.emit('supplemental_observation', '证据不足：补充观测后重新核验', {
            'purpose': purpose, 'assessment': copy.deepcopy(assessment),
            'method': 'configured_dual_observation_pose', 'maximum_supplements': 1,
            'scene_token': context['scene']['scene_token'], 'global_remap': False})
        context['use_retained_scene_for_return'] = True
        await self.move_to_observe(context)
        self.state['facts']['observation_pose_ready'] = True
        capture = context.pop('fine_capture')
        if capture['observation_id'] == assessment.get('observation_id'):
            raise ValueError('supplemental_observation_requires_new_capture')
        return capture

    async def execute_unpack(self, skill, context):
        method = getattr(self, '_' + skill)
        return await method(context)

    async def _prepare_unpack(self, context):
        if self.current_lifecycle()['phase'] != 'table_released' or context.get('holding'):
            raise ValueError('unpack_requires_verified_table_release_and_empty_hands')
        observation = self.measured_state(context)
        goals = {'A': [-86., 76., 88., -90., 2., 14., -24.],
                 'B': [106., 61., -113., -78., 27., 36., 37.]}
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
        context['vla_pose_execution'] = result
        self.state['facts'].update(observation_pose_ready=False, grasp_view_ready=False, observation_map_ready=False,
            observed=False, views_ready=False, scene_ready=False)
        return {'success': True, 'plan': plan, 'execution': result, 'rgbd_remapped': False}

    async def _mock_unpack(self, context):
        lifecycle = self.current_lifecycle()
        if lifecycle['phase'] != 'table_released' or not context.get('vla_pose_execution'):
            raise ValueError('mock_unpack_requires_completed_vla_pose')
        floor = context['observation']['table']['surface_height_m']
        polygon = np.asarray(self.region['region_polygon_world_xy_m'])
        work_area = {'center_world_xy_m': self.region['center_world_xy_m'],
                     'size_m': float(np.ptp(polygon[:, 0])), 'floor_z_m': floor}
        self.check_generation()
        receipt = await asyncio.to_thread(self.simulation.mock_unpack, work_area,
            generation=self.state['generation'], target_id=self.state['target_id'], owner_id=self.run_id)
        self.check_generation()
        receipt['table_release_feedback_id'] = lifecycle['table_release_feedback_id']
        lifecycle.update(phase='unpack_pending', mock_action=copy.deepcopy(receipt))
        self.sync_lifecycle(context)
        # Only verify_unpack may add black geometry, after visual confirmation
        # and same-capture RGBD measurement. The mock pose is not a map input.
        self.emit('mock_vla_completed', '模拟 VLA：粉色包装变为黑色内件', receipt)
        return {'success': True, 'receipt': receipt, 'rgbd_remapped': False}

    async def _verify_unpack(self, context):
        target = self.state['target_id']
        lifecycle = self.current_lifecycle()
        active_ids = [identifier for identifier, record in self.demo_task['item_lifecycle'].items()
                      if record.get('phase') not in {'wrapped_source', 'completed', 'consumed'}]
        if active_ids != [target] or lifecycle.get('phase') != 'unpack_pending':
            raise ValueError('unpack_single_active_target_required')
        context.pop('fine_capture', None)
        from .demo_lifecycle_perception import unpack_table_support, validate_table_release_reference

        before = context.get('post_place_capture')
        reference = context.get('post_place_verification_input')
        work_area = unpack_table_support(context['observation'], self.demo_task['destination_table_id'])
        capture = await self.capture_observation(context)
        for attempt in range(2):
            context.update(post_unpack_capture=capture, latest_capture=capture)
            self.record_observation(capture, 'post_unpack')
            validate_table_release_reference(before, reference, capture,
                lifecycle.get('table_visual_evidence', {}), lifecycle['mock_action'], self.run_id)
            result = await self.call_algorithm(self.verify_unpacking,
                capture, target_id=target, lifecycle=copy.deepcopy(lifecycle),
                action_evidence=copy.deepcopy(lifecycle['mock_action']), owner_id=self.run_id, work_area=work_area,
                table_release_capture=before, table_release_reference=copy.deepcopy(reference), active_target_ids=active_ids)
            self.emit('visual_unpack_result', 'Decisions：拆包结果与观测充分性', result)
            if attempt == 1:
                break
            local = None
            visible = result.get('visual_evidence', {}).get('judgments', {}).get('unpack_result') == 'yes'
            if visible or result['success']:
                try:
                    local = await self.call_algorithm(self.algorithms.publish_local_object,
                        context['scene'], capture, target, work_area, 'unpacked')
                except ValueError as error:
                    if str(error) != 'local_object_depth_missing':
                        raise
                    result['observation_assessment'] = {**result.get('observation_assessment', {}),
                        'observation_id': capture['observation_id'], 'needs_observation': True,
                        'missing_evidence': ['current_object_depth'], 'status': 'unknown'}
                if local:
                    context['scene'] = local['scene']
                    self.emit('local_object_obstacle', '黑件可见：加入实测局部障碍后回观测位姿', local['measurement'])
            if local is None:
                context['scene'] = await self.call_algorithm(self.algorithms.omit_unpack_obstacle,
                    context['scene'], target, lifecycle['mock_action']['action_id'])
            self.emit('unpack_observation_return', '拆包后双臂回观测位姿，准备重新建立全局场景',
                {'target_id': target, 'scene_token': context['scene']['scene_token'],
                 'black_obstacle_measured': local is not None})
            assessment = result.get('observation_assessment', {})
            if assessment.get('needs_observation'):
                capture = await self.supplement_result_observation(context, assessment, 'verify_unpack')
            else:
                context['use_retained_scene_for_return'] = True
                await self.move_to_observe(context)
                capture = context.pop('fine_capture')
                self.state['facts']['observation_pose_ready'] = True
        if result['success']:
            try:
                measured = await self.call_algorithm(self.algorithms.measure_local_object,
                    capture, work_area, 'unpacked')
            except ValueError as error:
                if str(error) != 'local_object_depth_missing':
                    raise
                result.update(success=False, status='unknown', reason='unpack_depth_unconfirmed_after_observation')
                result['observation_assessment'].update(needs_observation=True,
                    missing_evidence=['current_object_depth'], status='unknown', supplement_exhausted=True)
            else:
                workspace = await self.call_algorithm(self.algorithms.measured_object_workspace,
                    measured, lifecycle['to_right_bin_arm'])
                result['workspace_evidence'] = workspace
                result['visual_evidence']['workspace_evidence'] = workspace
                self.emit('unpack_workspace_checked', '实测黑件位置与执行臂工作空间核对', workspace)
                if not workspace['reachable']:
                    result.update(success=False, status='failure', reason='unpacked_object_outside_execution_workspace')
        if result.get('observation_assessment', {}).get('needs_observation'):
            result['observation_assessment']['supplement_exhausted'] = True
        if result['success']:
            product = create_product(self.demo_task, target, result, capture, owner_id=self.run_id)
            self.task_progress.register_product(product)
            reference = context['visual_inventory_reference']
            reference.setdefault('product_references', {})[product['id']] = {
                'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'verification_capture': copy.deepcopy(result['verification_capture']),
                'entity': copy.deepcopy(product), 'evidence': copy.deepcopy(product['creation_evidence']),
                'views': {camera: {'rgb': view['rgb']} for camera, view in capture['views'].items()}}
            self.state['target_id'] = product['id']
            self.sync_lifecycle(context)
            self.reset_transfer(context)
            self.state['target_id'] = product['id']
            context.update(unpacked_grasp_capture=capture, unpacked_grasp_target_id=product['id'])
            result = {**result, 'input_entity_id': target, 'output_entity_id': product['id'],
                      'process_id': product['process_id']}
            self.emit('product_created', '已确认拆包产物：创建独立黑件实体', {
                **copy.deepcopy(product['creation_evidence']), 'product_entity': copy.deepcopy(product)})
            self.state['facts'].update(observation_pose_ready=True, grasp_view_ready=True, observation_map_ready=False,
                observed=False, scene_available=True,
                inventory_finalized=True, task_assessed=True, needs_place=True)
        return result

    async def _plan_transport(self, context):
        if self.current_lifecycle()['appearance'] != 'unpacked':
            raise ValueError('transport_requires_unpacked_core')
        context.pop('transport_execution', None)
        context.pop('transport_plan', None)
        self.state['facts'].update(transport_execution_completed=False, drop_completed=False,
                                   transport_plan_ready=False)
        await self.verify_scene(context)
        observation = self.measured_state(context)
        await self.prepare_attachment(context, observation)
        arm, target = self.state['arm'], self.state['target_id']
        destination = observed_object(observation, self.demo_task['right_bin_id'])
        goal, evidence = right_bin_transport_goal(observation['arms'][arm]['tcp'], context['holding'],
            destination, context['scene'], observation)
        self.emit('transport_goal_derived', '从当前右盒完整体素层推导运输高度', evidence)
        plan = await self.call_algorithm(self.algorithms.plan_attached_pose,
            observation, target, arm, context['scene'], context['holding'], goal.tolist(),
            planning_settings=context['planning_settings'])
        plan['transport_goal_evidence'] = evidence
        context['transport_plan'] = plan
        self.current_lifecycle()['phase'] = 'black_held'
        self.sync_lifecycle(context)
        return plan

    async def _execute_transport(self, context):
        plan = context['transport_plan']
        self._require_transport_binding(context)
        plan_id = transport_plan_key(plan)
        await self.verify_scene(context)
        command = self.prepare_command(plan['command'])
        await self.validate_motion(context, 'transport', command, plan['carrying_binding'],
                                   previous_check=plan.get('command_check'))
        result = await self.execution_stage(context, 'transport', self.motor,
            self.simulation.execute_trajectory, self.state['arm'], command['positions_rad'], command['dt'])
        if transport_plan_key(context['transport_plan']) != plan_id:
            raise ValueError('transport_execution_plan_changed')
        context['transport_execution'] = {'result': result, 'plan_id': plan_id}
        return {'success': True, 'execution': result}

    @staticmethod
    def _require_transport_binding(context):
        plan = context['transport_plan']
        if (plan['planning_settings'] != context['planning_settings']
                or not np.array_equal(plan['carrying_binding']['points_tool_m'],
                                      context['holding']['points_tool_m'])):
            raise ValueError('transport_holding_or_settings_changed')

    async def _drop(self, context):
        target, arm = self.state['target_id'], self.state['arm']
        plan = context['transport_plan']
        if not context.get('transport_execution') or self.current_lifecycle()['phase'] != 'black_held':
            raise ValueError('drop_requires_completed_attached_transport')
        if context['transport_execution'].get('plan_id') != transport_plan_key(plan):
            raise ValueError('transport_execution_plan_changed')
        self._require_transport_binding(context)
        snapshot = self.measured_state(context)
        tcp = rigid_matrix(snapshot['arms'][arm]['tcp'])
        goal = rigid_matrix(plan['goal'])
        if np.linalg.norm(tcp[:2, 3] - goal[:2, 3]) > .04 or tcp[2, 3] < goal[2, 3] - .04:
            raise ValueError('drop_endpoint_not_above_bin')
        check = await self.validate_motion(context, 'transport',
            {'positions_rad': [snapshot['arms'][arm]['joints_rad']]}, plan['carrying_binding'],
            measured_observation=snapshot)
        held = copy.deepcopy(context['holding'])
        opening_check = await self.validate_motion(context, 'open',
            {'positions_rad': [snapshot['arms'][arm]['joints_rad']]}, plan['carrying_binding'],
            measured_observation=snapshot, gripper_end_position_rad=0.)
        await self.gripper(context, False, binding=plan['carrying_binding'])
        await self.motor(self.simulation.dwell, max(.8, context['planning_settings']['dwell_release_s']))
        feedback = self.simulation.robot_snapshot()
        if not gripper_open_confirmed(feedback['arms'][arm]):
            raise ValueError('drop_gripper_open_unconfirmed')
        feedback_id = f"feedback-{feedback['generation']}-{feedback['time']}"
        receipt = {'source': 'validated_held_attachment_pose_drop', 'target_id': target,
            'owner_id': self.run_id, 'generation': feedback['generation'], 'action_id': feedback_id,
            'completed': True, 'completed_time': feedback['time'], 'completed_perf_counter_ns': time.perf_counter_ns(),
            'release_feedback_id': feedback_id, 'destination_id': self.demo_task['right_bin_id'],
            'before_observation_id': context['initial_capture']['observation_id'],
            'holding_verified': held.get('verified') is True, 'attachment_verified': held.get('attached') is True,
            'open_confirmed': True, 'collision_check': check, 'opening_collision_check': opening_check,
            'release_tcp': tcp.tolist()}
        self.task_progress.record_release(target, feedback_id=feedback_id)
        lifecycle = self.current_lifecycle()
        lifecycle.update(phase='right_bin_released', drop_action=receipt)
        self.holding_state = None
        context.pop('holding', None)
        context['pending_transfer'] = {'target_id': target, 'destination_id': self.demo_task['right_bin_id'],
            'verified': False, 'scene_token': context['scene']['scene_token']}
        self.state['facts'].update(grasp_held=False, attached=False, released=True)
        self.sync_lifecycle(context)
        self.emit('drop_executed', '打开右夹爪并投放黑色内件', receipt)
        return {'success': True, 'receipt': receipt}

    async def _verify_drop(self, context):
        lifecycle, target = self.current_lifecycle(), self.state['target_id']
        capture = await self.capture_observation(context)
        self.record_observation(capture, 'post_place')
        before = context['initial_capture']
        reference = {**before, 'demo_task': copy.deepcopy(self.demo_task)}
        measured = context['scene_observation']
        if (measured.get('observation_id') != before['observation_id']
                or measured.get('generation') != before['generation']):
            raise ValueError('drop_bin_measurement_frame_mismatch')
        destination = observed_object(measured, self.demo_task['right_bin_id'])
        reference['drop_bin_region'] = right_bin_occupied_bounds(destination, context['scene'], measured)
        result = await self.call_algorithm(self.verify_right_bin_drop,
            capture, target_id=target, lifecycle=copy.deepcopy(lifecycle),
            action_evidence=copy.deepcopy(lifecycle['drop_action']), reference=reference, owner_id=self.run_id)
        self.emit('visual_drop_result', 'Decisions 核验本次黑件进入右盒', result)
        if result.get('observation_assessment', {}).get('needs_observation'):
            occupied = reference['drop_bin_region']
            bounds = {'minimum_world_m': occupied['minimum_world_m'],
                'maximum_world_m': (np.asarray(occupied['maximum_world_m']) + [0, 0, .02]).tolist()}
            context['scene'] = await self.call_algorithm(self.algorithms.publish_action_envelope,
                context['scene'], target, bounds, lifecycle['drop_action']['action_id'],
                source='drop_release_action_bounds')
            capture = await self.supplement_result_observation(context, result['observation_assessment'], 'verify_drop')
            self.record_observation(capture, 'post_place')
            result = await self.call_algorithm(self.verify_right_bin_drop,
                capture, target_id=target, lifecycle=copy.deepcopy(lifecycle),
                action_evidence=copy.deepcopy(lifecycle['drop_action']), reference=reference, owner_id=self.run_id)
            self.emit('visual_drop_result', 'Decisions：补充观测后核验入盒', result)
        if result['success']:
            evidence = result['visual_evidence']
            release = self.task_progress.records[target]['visual_release']
            evidence.update({key: release[key] for key in ('slot_id', 'region_id') if key in release})
            if not self.task_progress.confirm_visual_transfer(target, evidence):
                raise ValueError('drop_confirmation_ledger_binding_changed')
            lifecycle.update(phase='completed', drop_visual_evidence=copy.deepcopy(evidence))
            context['pending_transfer']['verified'] = True
            self.sync_lifecycle(context)
            context.pop('unpacked_grasp_capture', None)
            context.pop('unpacked_grasp_target_id', None)
            # The final item also returns home, so its destination must remain an obstacle.
            occupied = right_bin_occupied_bounds(destination, context['scene'], measured)
            bounds = {'minimum_world_m': occupied['minimum_world_m'],
                'maximum_world_m': (np.asarray(occupied['maximum_world_m']) + [0, 0, .02]).tolist()}
            self.emit('drop_action_envelope_derived', '从同一右盒体素层推导投放后动作包络',
                {'occupancy': occupied, 'upper_z_allowance_m': .02, 'action_bounds': bounds})
            context['scene'] = await self.call_algorithm(self.algorithms.publish_action_envelope,
                context['scene'], target, bounds, lifecycle['drop_action']['action_id'],
                source='verified_drop_action_bounds')
        return result
