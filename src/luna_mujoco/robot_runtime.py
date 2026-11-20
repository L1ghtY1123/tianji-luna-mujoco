"""Luna's robot boundary: all observations and physical commands go through Base."""
import asyncio
from contextlib import asynccontextmanager

import hashlib
import json
import queue
import threading
import time
import uuid
from copy import deepcopy

import numpy as np

from .geometry import front_tcp_pose
from .robot_contracts import CAMERA_ROLES, RGBDCaptureIncomplete, RGBDCaptureTimeout, detached_capture
from .robot_errors import StaleCommandError, TrajectoryExecutionError
from .robot_interface import RobotGateway, RobotGeometry


class RobotRuntime:
    def __init__(self, gateway: RobotGateway, geometry: RobotGeometry):
        self.gateway = gateway
        self.geometry = geometry
        self.url = gateway.endpoint
        self._lock = threading.RLock()
        self._motor_lock = threading.Lock()
        self._fk_lock = threading.Lock()
        self._epoch = 0
        self._command_revision = 0
        self._capture = None
        self._fk_key = None
        self._poses = None
        self._cancelled = threading.Event()
        self._prepared_identity = None

    @property
    def world_id(self):
        """Physical identity scope, independent of task cancellation generations."""
        state = self.gateway.robot_snapshot()
        if (state.get('runtime_mode') not in {'real', 'simulation'}
                or not state.get('robot_ip') or not state.get('session_id')):
            return None
        identity = {'endpoint': self.url, 'mode': state['runtime_mode'], 'robot_ip': state['robot_ip']}
        if state.get('world_revision'):
            identity['world_revision'] = state['world_revision']
        else:
            identity['controller_session'] = state['session_id']
        return 'robot-world-v2:' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def start(self):
        self.gateway.start()

    def close(self):
        self._cancelled.set()
        self.gateway.close()

    def prepare(self):
        self._cancelled.clear()
        result = self.gateway.prepare_observation(cancelled=self._cancelled, progress_records=queue.SimpleQueue())
        self.gateway.refresh()
        state = self.gateway.robot_snapshot()
        self._tool_poses(state['arms'])
        self._prepared_identity = state['geometry_revision'], state['session_id']
        return result

    def _tool_poses(self, arms):
        signature = self.gateway.robot_snapshot().get('kinematics_sha256')
        key = (signature, *(value for arm in ('A', 'B') for value in
                    (*arms[arm]['joints_rad'], arms[arm]['gripper_position_rad'])))
        with self._fk_lock:
            if self._fk_key != key:
                result = self.geometry.query(arms, links=['TCP_Link_L', 'TCP_Link_R'])
                if not signature or result.get('kinematics_sha256') != signature:
                    raise ValueError('base_curobo_kinematic_model_mismatch')
                self._poses = result['link_poses']
                self._fk_key = key
            poses = deepcopy(self._poses)
        for arm, link in (('A', 'TCP_Link_L'), ('B', 'TCP_Link_R')):
            arms[arm]['tcp_mount'] = poses[link]
            arms[arm]['tcp'] = front_tcp_pose(poses[link], arm).tolist()
            arms[arm]['tcp_front'] = arms[arm]['tcp']
            arms[arm]['tcp_frame'] = 'gripper_front_' + arm
            arms[arm]['fk_source'] = 'curobo_v2_fast_0.9.0'
        return arms

    def robot_snapshot(self, *, include_tcp=True, expected_generation=None):
        state = (self.gateway.robot_snapshot() if expected_generation is None
                 else self.joint_feedback(expected_generation))
        if include_tcp and state.get('available') and any(
                'gripper_position_rad' not in state['arms'].get(arm, {}) for arm in ('A', 'B')):
            state = self.gateway.wait_for_gripper_feedback(['A', 'B'], state['generation'])
        prepared = self._prepared_identity == (state.get('geometry_revision'), state.get('session_id'))
        state.update(confirmed=bool(prepared and state['ready'] and all(
            'gripper_position_rad' in state['arms'].get(arm, {}) for arm in ('A', 'B'))),
            manifest_revision=state.get('geometry_revision'), command_revision=self._command_revision,
            physics_error=state.get('error'), source='tianji_base')
        if include_tcp and state.get('available'):
            if any('gripper_position_rad' not in state['arms'].get(arm, {}) for arm in ('A', 'B')):
                raise ValueError('robot_gripper_feedback_unavailable')
            state['arms'] = self._tool_poses(state['arms'])
            current = self.gateway.robot_snapshot()
            if state['generation'] != current['generation']:
                raise StaleCommandError('robot_session_changed_during_fk')
        return state

    def snapshot(self):
        return {**self.robot_snapshot(include_tcp=False), 'geometry_poses': [], 'objects': [],
                'base_url': self.url, 'scene_owner': 'luna'}

    def manifest(self):
        return self.gateway.viewer_manifest()

    def viewer_asset(self, path):
        return self.gateway.viewer_asset(path)

    def recording_snapshot(self, known_revision=None):
        state = self.robot_snapshot(include_tcp=False)
        scene = self.gateway.viewer_scene()
        return {**state, 'geometry_poses': [], 'base_scene': scene}

    def collision_snapshot(self):
        from .grasp_contact import grasp_contact_links
        from .robot_collision import configuration_spheres
        state = self.robot_snapshot(include_tcp=False)
        joints_deg = {arm: np.rad2deg(state['arms'][arm]['joints_rad']).tolist() for arm in ('A', 'B')}
        balls, links = configuration_spheres({'id': 'open_gripper_display',
            'joints_deg': joints_deg,
            'gripper_positions_rad': {'A': 0., 'B': 0.}})
        scene = self.gateway.viewer_configuration(joints_deg, {'A': 0., 'B': 0.})
        self.gateway.check_generation(state['generation'])
        contacts = {arm: grasp_contact_links(arm) for arm in ('A', 'B')}
        spheres = [{'id': f'collision::{i}', 'link': link, 'center_m': sphere[:3].tolist(),
                    'radius_m': float(sphere[3]), 'target_contact_arms': [arm for arm in contacts if link in contacts[arm]]}
                   for i, (sphere, link) in enumerate(zip(balls, links))]
        return {'simulation': {**state, 'geometry_poses': [], 'base_scene': scene}, 'source': 'cuRoboV2-Fast / resources/robot/marvin_gui_dual.yml',
                'gripper_state': 'open', 'radius_policy': 'original_yaml_plus_link_buffer',
                'sphere_count': len(spheres), 'link_count': len(set(links)), 'spheres': spheres}

    def joint_feedback(self, generation):
        """Read the current joint packet for pose checks, including Base's stale-cache recovery."""
        state = self.gateway.check_generation(generation)
        if any(np.shape(state['arms'].get(arm, {}).get('joints_rad')) != (7,)
               or not np.isfinite(state['arms'][arm]['joints_rad']).all() for arm in ('A', 'B')):
            raise ValueError('measured_dual_arm_feedback_unavailable')
        return state

    def gripper_verification_sample(self, arm, generation, after_time):
        """Read an action-ordered jaw measurement. Only a closed jaw needs its own wrist."""
        from .robot_skills.grasp_verification import WRIST_CAMERAS, holding_from_feedback

        camera = WRIST_CAMERAS[arm]
        if not isinstance(after_time, (int, float)) or not np.isfinite(after_time):
            raise ValueError('verification_action_timestamp_required')
        revision = self._command_revision

        def check_idle():
            state = self.gateway.check_generation(generation)
            if (self._cancelled.is_set() or revision != self._command_revision
                    or state.get('active_commands') or not state.get('session_id')):
                raise StaleCommandError('robot_changed_during_gripper_verification')
            return state

        deadline = time.monotonic() + 3.
        while True:
            requested_at = time.perf_counter_ns()
            self.gateway.refresh()
            state = check_idle()
            jaw = state['arms'].get(arm, {})
            stamp = jaw.get('gripper_measured_time_s')
            if stamp is not None and stamp >= after_time and 'gripper_closed_fraction' in jaw:
                break
            if time.monotonic() >= deadline:
                stamp, jaw = state['time'], {}
                break
            self._cancelled.wait(.05)
        packet = {'observation_id': f"gripper-{generation}-{arm}-{state['feedback_id']}-{requested_at}",
            'feedback_id': state['feedback_id'], 'generation': generation, 'session_id': state['session_id'],
            'source': 'gripper_feedback', 'arm': arm, 'command_revision': revision,
            'bridge_instance_id': state['bridge_instance_id'], 'manifest_revision': state['geometry_revision'],
            'active_commands': [], 'captured_monotonic_s': time.monotonic(),
            'time': stamp, 'captured_perf_counter_ns': requested_at, 'arms': {**state['arms'], arm: jaw}}
        if not holding_from_feedback(packet, arm)['requires_visual']:
            return packet
        deadline = time.monotonic() + 3.
        while True:
            try:
                capture = self.gateway.capture_rgbd([camera])
            except RGBDCaptureIncomplete as error:
                check_idle()
                if time.monotonic() < deadline:
                    self._cancelled.wait(.1)
                    continue
                packet['wrist_error'] = str(error)
                return packet
            check_idle()
            if (capture['generation'] != generation or capture.get('session_id') != packet['session_id']
                    or set(capture['views']) != {camera} or capture.get('active_commands')):
                raise StaleCommandError('wrist_verification_binding_changed')
            exposure = capture['views'][camera]
            sample = exposure['arms'].get(arm, {})
            if min(exposure['capture_time'], sample.get('gripper_measured_time_s', -1)) >= after_time:
                capture['arms'] = self._tool_poses(exposure['arms'])
                check_idle()
                capture.update(arm=arm, command_revision=revision, manifest_revision=capture['geometry_revision'],
                               feedback_id=packet['feedback_id'], source='gripper_wrist_rgbd')
                return capture
            if time.monotonic() >= deadline:
                packet['wrist_error'] = 'wrist_measurement_precedes_action'
                return packet
            self._cancelled.wait(.05)

    def capture_rgbd(self):
        expected = self.gateway.robot_snapshot()
        revision = self._command_revision
        deadline = time.monotonic() + 30.
        attempts = 0
        while True:
            if self._cancelled.is_set():
                raise StaleCommandError('capture_cancelled')
            if self.gateway.robot_snapshot()['generation'] != expected['generation']:
                raise StaleCommandError('robot_session_changed_during_capture')
            try:
                attempts += 1
                capture = self.gateway.capture_rgbd(list(CAMERA_ROLES))
                if capture.get('active_commands'):
                    raise RGBDCaptureIncomplete([{'error': 'robot_in_motion_during_capture'}])
                break
            except RGBDCaptureIncomplete as error:
                if (set(error.codes) & {'three_camera_same_frame_unverified', 'synchronized_rgbd_unsupported'}
                        and error.details['synchronization'].get('same_frame_supported') is False):
                    raise
                if time.monotonic() >= deadline:
                    raise RGBDCaptureTimeout(error, attempts) from error
                self._cancelled.wait(.1)
        capture['arms'] = self._tool_poses(capture['arms'])
        for view in capture['views'].values():
            view['arms'] = self._tool_poses(view['arms'])
        geometry = self.geometry.query(capture['arms'], views=capture['views'])
        for name, view in capture['views'].items():
            view['robot_mask'] = geometry['views'][name]['robot_mask']
            view['robot_mask'].setflags(write=False)
            view['robot_mask_source'] = geometry['views'][name]['source']
        if (self.gateway.robot_snapshot()['generation'] != expected['generation']
                or revision != self._command_revision or self._cancelled.is_set()):
            raise StaleCommandError('robot_changed_during_observation')
        capture.update(manifest_revision=capture['geometry_revision'], command_revision=revision)
        with self._lock:
            self._capture = detached_capture(capture)
        return capture

    def latest_rgbd(self):
        with self._lock:
            if self._capture and self._capture['generation'] == self.gateway.robot_snapshot()['generation']:
                return detached_capture(self._capture)
        return None

    def preview_rgb(self):
        return self.gateway.preview_rgb()

    def motion_epoch(self):
        with self._lock:
            return self._epoch

    def run_motor(self, epoch, function, *args, **kwargs):
        with self._motor_lock:
            if epoch != self.motion_epoch():
                raise StaleCommandError('motion_cancelled')
            self._command_revision += 1
            result = function(*args, **kwargs)
            if epoch != self.motion_epoch():
                raise StaleCommandError('motion_cancelled')
            return result

    def stop_motion(self, *, confirm_world=False):
        with self._lock:
            self._epoch += 1
            self._command_revision += 1
            epoch = self._epoch
            self._capture = None
        self._cancelled.set()
        before = self.gateway.robot_snapshot()
        result = self.gateway.stop('luna_cancelled')
        if confirm_world:
            self.gateway.refresh()
        after = self.gateway.robot_snapshot()
        identity_fields = ('session_id', 'bridge_instance_id', 'geometry_revision', 'world_revision')
        identity_changed = (not all(after.get(key) for key in identity_fields[:3])
                            or any(before.get(key) != after.get(key) for key in identity_fields))
        old_stop, new_stop = before.get('stop_epoch'), after.get('stop_epoch')
        stop_advance = (new_stop - old_stop if type(old_stop) is int and type(new_stop) is int else None)
        delta = after['generation'] - before['generation']
        continuity = bool(confirm_world and result.get('ok') and not identity_changed and after.get('available')
                          and after.get('active_commands') == [] and self.motion_epoch() == epoch
                          and ((delta == 1 and stop_advance == 0) or
                               (delta == 2 and stop_advance == 1 and result.get('hardware_command_sent') is True)))
        with self._lock:
            if result.get('ok') is True and self._epoch == epoch:
                self._cancelled.clear()
        return {**result, 'motion_epoch': epoch, 'previous_generation': before['generation'],
                'generation': after['generation'], 'robot_identity_changed': identity_changed,
                'world_continuity_confirmed': continuity,
                'world': {key: after.get(key) for key in identity_fields[:3]}, 'stopped_time_s': after['time']}

    def reset(self):
        result = self.stop_motion()
        with self._lock:
            self._capture = None
        self._cancelled.clear()
        return {'ok': result.get('ok', False), 'scope': 'luna_task', 'robot': self.snapshot()}

    @asynccontextmanager
    async def motion_scope(self, arms, kind, generation, epoch):
        # One Base owner spans the full motion, including its checked start.
        # Shield begin until its acknowledgement so cancellation cannot orphan an enabled arm.
        request_id = uuid.uuid4().hex
        begin = asyncio.create_task(asyncio.to_thread(self.run_motor, epoch, self.gateway.begin_motion,
            kind, generation, .02, mode='position', arms=arms, request_id=request_id))
        receipt = None
        reason = 'failed'
        try:
            try:
                receipt = await asyncio.shield(begin)
            except asyncio.CancelledError:
                receipt = await begin
                raise
            await asyncio.to_thread(self.gateway.refresh)
            yield receipt['motion_id']
            reason = 'completed'
        finally:
            import sys
            primary_error = sys.exc_info()[1]
            if receipt is not None:
                try:
                    result = await asyncio.to_thread(self.gateway.end_motion, reason, expected_id=receipt['motion_id'])
                    if result.get('disable_confirmed') is not True:
                        raise TrajectoryExecutionError('base_motion_disable_unconfirmed', result)
                except Exception as cleanup_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(f'Motion cleanup: {cleanup_error}')
                    if hasattr(primary_error, 'details'):
                        primary_error.details['motion_cleanup_error'] = getattr(cleanup_error, 'details', str(cleanup_error))
            elif self.motion_epoch() == epoch:
                # A lost begin reply can still own the arm. Base reconciles by request ID.
                await asyncio.to_thread(self.gateway.end_motion, 'failed', expected_request=request_id)

    def _motion(self, arms, generation, action, *, motion_id=None):
        if motion_id is not None:
            current = self.gateway.check_generation(generation).get('current_motion') or {}
            if current.get('motion_id') != motion_id or current.get('state') != 'enabled':
                raise StaleCommandError('transfer_motion_owner_changed')
            return action()
        self.gateway.check_generation(generation)
        completed = False
        try:
            self.gateway.begin_motion('observe' if len(arms) == 2 else 'grasp', generation, .02, mode='position', arms=arms)
            result = action()
            completed = True
            return result
        finally:
            cleanup = self.gateway.end_motion('completed' if completed else 'failed')
            if cleanup.get('ok') is not True or (cleanup.get('disable_confirmed') is not True
                    and cleanup.get('hardware_command_sent') is not False):
                raise TrajectoryExecutionError('base_motion_disable_unconfirmed', cleanup)

    def execute_trajectory(self, arm, positions_rad, dt, generation, *, execution_settings=None,
                           times_s=None, gripper_events=None, motion_id=None):
        positions = np.asarray(positions_rad, dtype=float)
        times = np.arange(len(positions)) * dt if times_s is None else np.asarray(times_s, dtype=float)
        if (positions.ndim != 2 or positions.shape[1] != 7 or not len(positions)
                or times.shape != (len(positions),) or not np.isfinite(positions).all()
                or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0)):
            raise ValueError('invalid_base_trajectory')
        events = deepcopy(gripper_events or [])
        if events != sorted(events, key=lambda item: item['time_s']) or any(
                type(item.get('closed')) is not bool or not 0 <= item['time_s'] <= times[-1] for item in events):
            raise ValueError('invalid_base_gripper_events')
        indices = [int(np.argmin(abs(times - event['time_s']))) for event in events]
        if len(set(indices)) != len(indices) or any(abs(times[i] - event['time_s']) > 1e-8 for i, event in zip(indices, events)):
            raise ValueError('gripper_event_must_match_unique_trajectory_sample')
        def execute():
            start, receipts = time.monotonic(), []
            settings = execution_settings or {}
            grippers = None
            if events:
                feedback = self.gateway.wait_for_gripper_feedback([arm], generation)
                echo = feedback['arms'][arm].get('gripper_command_evidence') or {}
                initial = echo.get('closed_fraction')
                if initial not in (0., 1.) or initial == float(events[0]['closed']):
                    raise TrajectoryExecutionError('transfer_initial_gripper_command_unconfirmed', {
                        'arm': arm, 'dispatched': False, 'commanded_closed_fraction': initial})
                # Base sends the first gripper row as a command. A transport path
                # has no gripper rows until measured arrival at the release pose.
                if initial == 1. and indices[0] > 0:
                    raise ValueError('carrying_transport_requires_joint_only_path')
                values = [initial] * len(times)
                for index, event in zip(indices, events):
                    values[index:] = [float(event['closed'])] * (len(times) - index)
                grippers = {arm: values}
            result = self.gateway.execute_trajectory_program({arm: positions.tolist()}, grippers, .02, generation,
                times_s=times.tolist(), tolerance_deg=float(np.rad2deg(settings.get('motion_position_tolerance_rad', .005))),
                arrival_timeout_s=settings.get('motion_settle_timeout_s', 3.))
            measured = {item['source_end_index']: item for item in result.get('gripper_feedback', [])}
            for index, event in zip(indices, events):
                if index not in measured:
                    raise TrajectoryExecutionError('base_gripper_event_feedback_missing', {'source_end_index': index})
                item = measured[index]
                feedback = deepcopy(item['feedback'])
                feedback['arms'] = self._tool_poses(feedback['arms'])
                receipts.append({**event, 'receipt': item['completion'], 'feedback': feedback})
            self.gateway.refresh()
            return {'generation': generation, 'duration_s': time.monotonic() - start,
                    'source': 'tianji_base', 'execution': result, 'gripper_events': receipts,
                    'feedback': self.robot_snapshot()}
        return self._motion([arm], generation, execute, motion_id=motion_id)

    def execute_dual_trajectory(self, paths, dt, generation, *, execution_settings=None, motion_id=None):
        settings = execution_settings or {}
        return self._motion(['A', 'B'], generation, lambda: self.gateway.execute_trajectory_program(paths, None, dt, generation,
            tolerance_deg=float(np.rad2deg(settings.get('motion_position_tolerance_rad', .005))),
            arrival_timeout_s=settings.get('motion_settle_timeout_s', 3.)), motion_id=motion_id)

    def set_gripper(self, arm, closed, generation, *, execution_settings=None, motion_id=None):
        def execute():
            receipt = self.gateway.set_gripper(arm, closed, generation,
                arrival_timeout_s=(execution_settings or {}).get('gripper_timeout_s', 3.))
            self.gateway.refresh()
            return {**self.robot_snapshot()['arms'][arm], 'arm': arm, 'generation': generation,
                    'closed_command': closed, 'source': 'tianji_base', 'receipt': receipt}
        return self._motion([arm], generation, execute, motion_id=motion_id)

    def dwell(self, duration_s, generation):
        return self.gateway.dwell(duration_s, generation)

    def unpack_at_pose(self, work_area, task_id=None, *, generation, target_id=None, owner_id=None):
        intent = {key: work_area[key] for key in ('region_polygon_world_xy_m', 'center_world_xy_m', 'size_m') if key in work_area}
        prompt = '拆开指定工作区域中的粉色包装并取出黑色物品。工作区域约束（世界坐标，米）：' + json.dumps(intent, ensure_ascii=False)
        result = self.gateway.execute_vla(prompt, {}, generation)
        return {'kind': 'unpack', 'source': 'tianji_base_vla', 'provenance': result,
                'is_visual_evidence': False, 'target_id': target_id, 'owner_id': owner_id, 'task_id': task_id,
                'action_id': result.get('run_id'), 'generation': generation, 'completed': result.get('ok') is True,
                'completed_time': time.time(), 'completed_monotonic_ns': time.monotonic_ns(),
                'completed_perf_counter_ns': time.perf_counter_ns()}
