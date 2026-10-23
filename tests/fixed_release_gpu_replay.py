"""Replay actual RGBD and held feedback through current 10 mm place planning."""
import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import httpx
import numpy as np


def main(folder, url):
    from luna_mujoco.algorithms import AlgorithmError, Algorithms
    from luna_mujoco.kinematics import robot_tcp
    from luna_mujoco.placement import release_endpoint_evidence
    from luna_mujoco.planning_settings import PlanningSettings

    holding = json.loads((folder / 'stage-1.json').read_text())['holding']
    mapping = json.loads((folder / 'gpu-requests/017-perception-build.json').read_text())
    payload = json.loads((folder / 'gpu-requests/018-carrying-prepare.json').read_text())['body']
    settings = PlanningSettings().model_dump()
    holding['points_tool_m'] = payload['carrying_binding']['points_tool_m']
    holding.update(settings['attachment'])
    holding.pop('attachment_evidence', None)
    observation = deepcopy(holding['observation'])
    arm = holding['arm']
    peer = 'B' if arm == 'A' else 'A'
    for selected, key in ((arm, 'start_joints_deg'), (peer, 'other_joints_deg')):
        observation['arms'][selected]['joints_rad'] = np.deg2rad(payload[key]).tolist()
        observation['arms'][selected]['gripper_position_rad'] = payload['gripper_joint_positions_rad'][selected]
        observation['arms'][selected]['tcp'] = robot_tcp(observation['arms'], selected).tolist()
    destination = next(obj for obj in observation['objects'] if obj['kind'] == 'bin')
    client = httpx.Client(base_url=url, timeout=300.)
    def request(path, body):
        response = client.post(path, json=body)
        result = response.json()
        if response.is_error or result.get('ok') is False:
            error = result.get('error', {})
            raise AlgorithmError(str(error), code=error.get('code', 'algorithm_error') if isinstance(error, dict)
                                 else 'algorithm_error', details=error)
        return result
    algorithms = Algorithms.__new__(Algorithms)
    algorithms.request = request
    started = time.perf_counter()
    report = {'physical_motion': False, 'source': 'recorded_rgbd_and_robot_feedback', 'release_height_m': .01}
    try:
        mapping['body']['planning_settings'] = settings
        scene = request(mapping['path'], mapping['body'])
        observation.update(observation_id=scene['observation_id'], generation=scene['generation'])
        report['mapping_seconds'] = time.perf_counter() - started
        def progress(kind, title, data):
            print(json.dumps({'kind': kind, 'data': data}, ensure_ascii=False), flush=True)
        plan = algorithms.plan_place(observation, holding['instance_id'], arm, scene, holding, destination['id'],
                                     planning_settings=settings, progress=progress)
        release = deepcopy(observation['arms'])
        release[arm]['joints_rad'] = plan['phases']['place']['positions_rad'][-1]
        report['release_endpoint'] = release_endpoint_evidence(plan['goal'], holding, robot_tcp(release, arm))
        report.update(status='planned', selection=plan['selection_evidence'], checks=plan['command_checks'])
        assert report['release_endpoint']['at_release_goal']
    except (AlgorithmError, ValueError, RuntimeError, httpx.HTTPError, AssertionError) as error:
        report.update(status='rejected', error_type=type(error).__name__, code=getattr(error, 'code', None),
                      message=str(error), details=getattr(error, 'details', None))
    finally:
        request('/carrying/clear', {})
        client.close()
    report['total_seconds'] = time.perf_counter() - started
    print('RESULT ' + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--url', required=True, help='An isolated candidate GPU service, not an active task service')
    args = parser.parse_args()
    main(args.folder, args.url)
