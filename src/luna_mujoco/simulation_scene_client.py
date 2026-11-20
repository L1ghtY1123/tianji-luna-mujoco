"""Luna's scene configuration channel, separate from robot control and RGBD."""
import math
import time
from ipaddress import ip_address
from urllib.parse import urlsplit

import httpx


class SimulationSceneClient:
    def __init__(self, robot_state, *, scene_url=None, transport=None):
        self.robot_state = robot_state
        self.scene_url = (scene_url or '').rstrip('/')
        self.client = httpx.Client(timeout=10., trust_env=False, transport=transport)

    def close(self):
        self.client.close()

    def _origin(self, state):
        if state.get('runtime_mode') != 'simulation' or state.get('error'):
            raise ValueError('手工仿真场景仅可用于身份有效的仿真设备')
        address = ip_address(state.get('robot_ip') or '')
        if not self.scene_url:
            host = f'[{address}]' if address.version == 6 else str(address)
            return f'http://{host}:8767'
        url = urlsplit(self.scene_url)
        if (url.scheme != 'http' or url.username or url.password or url.path or url.query or url.fragment
                or ip_address(url.hostname or '') != address):
            raise ValueError('Simulation 场景地址必须指向当前 Base 发布的仿真机器人 IP')
        return self.scene_url

    @staticmethod
    def _identity(state):
        return tuple(state.get(key) for key in ('runtime_mode', 'robot_ip', 'bridge_instance_id', 'session_id'))

    @staticmethod
    def _require_idle(state):
        preparation = state.get('preparation') or {}
        if (state.get('current_motion') or state.get('vla_run') or state.get('active_execution_id')
                or state.get('active_commands') or preparation.get('worker_pending')
                or preparation.get('state') == 'preparing'):
            raise ValueError('机器人作业尚未结束，不能修改仿真场景')
        if state.get('connected') and state.get('enable_state') != {'A': 0, 'B': 0}:
            raise ValueError('请先停止任务并禁用双臂，再修改场景')

    def _request(self, method, url, **kwargs):
        try:
            response = self.client.request(method, url, **kwargs)
        except httpx.RequestError as error:
            raise ValueError('Simulation 场景服务连接失败') from error
        body = response.json()
        if not response.is_success or body.get('ok') is not True:
            raise ValueError(body.get('message') or body.get('error') or f'Simulation HTTP {response.status_code}')
        return body

    def _scene(self, origin):
        scene = self._request('GET', origin + '/v1/scene')
        if scene.get('provider') != 'tianji-simulation' or scene.get('runtime_mode') != 'simulation':
            raise ValueError('场景服务未提供有效的 Simulation 身份')
        return scene

    def snapshot(self):
        state = self.robot_state()
        if state.get('runtime_mode') != 'simulation' or state.get('error'):
            return dict(ok=True, editable=False, runtime_mode=state.get('runtime_mode') or 'unknown',
                        message='手工场景仅在连接仿真时可用；真机和身份未知时禁用')
        scene = self._scene(self._origin(state))
        if self._identity(self.robot_state()) != self._identity(state):
            raise ValueError('设备身份已变化，请重新读取场景')
        try:
            self._require_idle(state)
        except ValueError as error:
            return {**scene, 'editable': False, 'message': str(error)}
        return {**scene, 'editable': True}

    def _write(self, path, payload, *, expected_identity=None):
        state = self.robot_state()
        if expected_identity is not None and self._identity(state) != expected_identity:
            raise ValueError('设备身份在复位前已变化')
        origin = self._origin(state)
        self._require_idle(state)
        self._scene(origin)
        current = self.robot_state()
        if self._origin(current) != origin or self._identity(current) != self._identity(state):
            raise ValueError('设备身份已变化，请重新读取场景')
        self._require_idle(current)
        return self._request('POST', origin + '/v1/scene/' + path, json=payload)

    def reset(self, expected_revision=None, *, initial_state=None):
        from .demo_scene import demo_observation_poses

        state = self.robot_state() if initial_state is None else initial_state
        origin = self._origin(state)
        identity = self._identity(state)
        initial_scene = self._scene(origin)
        expected = initial_scene['config_revision'] if expected_revision is None else expected_revision
        goals = demo_observation_poses()
        result = self._write('reset', {'expected_revision': expected}, expected_identity=identity)
        revision = result['config_revision']
        deadline = time.monotonic() + 20.
        while time.monotonic() < deadline:
            current = self.robot_state()
            if self._identity(current) != identity:
                raise ValueError('设备身份在确认复位时已变化')
            scene = self._scene(origin)
            if scene.get('instance_id') != initial_scene.get('instance_id') or scene['config_revision'] != revision:
                raise ValueError('等待复位时仿真实例或场景版本已变化')
            if scene.get('apply_error'):
                raise ValueError(str(scene['apply_error']))
            # The simulated camera publishes instance:generation:scene revision:calibration.
            # During a reload Base briefly reports None:None:calibration.
            frame = (current.get('world_revision') or '').rsplit(':', 3)
            rendered = (len(frame) == 4 and frame[0] not in ('', 'None')
                        and frame[1].isdecimal() and frame[2] == str(revision) and bool(frame[3]))
            if (scene.get('ready') and scene.get('applied_revision') == revision and rendered
                    and current['world_revision'] != state.get('world_revision')
                    and current.get('ready') and current.get('enable_state') == {'A': 0, 'B': 0}):
                errors = {}
                for arm in ('A', 'B'):
                    feedback = current.get('arms', {}).get(arm, {})
                    actual = feedback.get('joints_rad', [])
                    opening = feedback.get('gripper_closed_fraction')
                    if (len(actual) == len(goals[arm]) and all(math.isfinite(v) for v in actual)
                            and isinstance(opening, (float, int)) and 0 <= opening <= .02):
                        errors[arm] = max(abs(q - ref) for q, ref in zip(actual, goals[arm], strict=True))
                if len(errors) == 2 and max(errors.values()) <= math.radians(1):
                    self._require_idle(current)
                    return {**scene, 'robot_reset': True, 'joint_error_rad': errors,
                            'reset_world_revision': current['world_revision']}
            time.sleep(.15)
        raise ValueError('未在时限内确认场景、双臂观测位与夹爪复位')


    def register_mesh(self, resource):
        return self._write('meshes', resource)['asset_id']

    def replace(self, configuration, expected_revision):
        return self._write('config', {'configuration': configuration, 'expected_revision': expected_revision})
