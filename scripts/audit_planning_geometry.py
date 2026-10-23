"""Compare live cuRobo sphere geometry with independent MuJoCo FK; no robot motion.

Run --reference in the web Python environment, then --gpu in the cuRobo image.
The container project mount is read-only: use --output /tmp/geometry-audit.json.
Only this project's public adapter/configuration is exercised; the SDK is read-only.
"""
import argparse
import gc
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from luna_mujoco.carrying import CarryingBinding, carrying_spheres, configure_gripper_feedback

ROOT = Path(__file__).resolve().parents[1]


def cases():
    from luna_mujoco.planning_settings import PlanningSettings

    observed = PlanningSettings().observation.model_dump()
    points = np.random.default_rng(7).uniform(-.02, .02, (80, 3)) + [0., 0., .16]
    result = []
    for arm in ('A', 'B'):
        peer = 'B' if arm == 'A' else 'A'
        for phase, closure, attached in (
                ('grasp', 0., False), ('grasp_lift', .38, False),
                ('place_approach', .38, True), ('place', .38, True),
                ('retreat', 0., False), ('transport', .43, True), ('detach', 0., False)):
            joints = {a: np.asarray(observed[a]).copy() for a in ('A', 'B')}
            if phase in ('place', 'transport'):
                joints[arm][2] += .15
                joints[peer][4] -= .12
            result.append({'arm': arm, 'phase': phase,
                'joints': {a: q.tolist() for a, q in joints.items()},
                'feedback': {arm: closure, peer: .21},
                'points': points.tolist() if attached else None})
    return result


def binding_for(case):
    if case['points'] is None:
        return None
    return CarryingBinding(case['arm'], 'geometry-audit', 'capture', 1,
        'audit-scene', 'target', 'tool', 'held', tuple(map(tuple, case['points'])), .005)


def reference(output):
    import mujoco

    from luna_mujoco.kinematics import robot_model

    config = yaml.safe_load((ROOT / 'resources/curobo/marvin_gui_dual.yml').read_text())
    urdf = ROOT / 'assets/marvin_urdf/marvin_with_grippers.urdf'
    mimics = [(j.get('name'), j.find('mimic')) for j in ET.parse(urdf).getroot().findall('joint')
              if j.find('mimic') is not None]
    model = robot_model()
    rows = cases()
    for case in rows:
        data = mujoco.MjData(model)
        kin = configure_gripper_feedback(config, case['feedback'], urdf)['robot_cfg']['kinematics']
        values = dict(kin['lock_joints'])
        for arm, side in (('A', 'L'), ('B', 'R')):
            values.update({f'Joint{i}_{side}': q for i, q in enumerate(case['joints'][arm], 1)})
        for name, mimic in mimics:
            values[name] = values[mimic.get('joint')] * float(mimic.get('multiplier', 1)) + float(mimic.get('offset', 0))
        for name, value in values.items():
            data.qpos[int(model.joint(name).qposadr[0])] = value
        mujoco.mj_fwdPosition(model, data)
        expected = {}
        for link in kin['collision_link_names']:
            body = model.body(link).id
            rotation = data.xmat[body].reshape(3, 3)
            buffer = kin['collision_sphere_buffer'].get(link, 0.)
            expected[link] = [[*(rotation @ np.asarray(s['center']) + data.xpos[body]), s['radius'] + buffer]
                              for s in kin['collision_spheres'][link]]
        binding = binding_for(case)
        if binding is not None:
            body = model.body('TCP_Link_' + ('L' if case['arm'] == 'A' else 'R')).id
            balls = np.asarray(carrying_spheres(binding))
            balls[:, :3] = balls[:, :3] @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            expected['jev_carried_object_' + case['arm']] = balls.tolist()
        case['expected_spheres'] = expected
    output.write_text(json.dumps(rows), encoding='utf-8')
    print(json.dumps({'reference': str(output), 'cases': len(rows), 'source': 'MuJoCo FK'}), flush=True)


def gpu(reference_path, output):
    import torch
    from tianji_curobo.curobo_service import materialize_urdf

    from luna_mujoco.backend_builder import build_geometry_backend
    from luna_mujoco.geometry_service import GeometryRuntime

    urdf = materialize_urdf(Path(os.environ['LUNA_ROBOT_URDF']),
        marvin_meshes=Path(os.environ['LUNA_MARVIN_MESHES']), rag_meshes=Path(os.environ['LUNA_RAG_MESHES']))
    runtime = GeometryRuntime(robot_config_path=ROOT / 'resources/curobo/marvin_gui_dual.yml',
                              urdf_path=urdf, asset_root_path=urdf.parent)
    rows = json.loads(reference_path.read_text())
    report = []
    for arm in ('A', 'B'):
        for capacity, ik_only in ((100, True), (4, False), (16, False), (None, False)):
            backend = build_geometry_backend(runtime, arm, batch_size=capacity, ik_only=ik_only)
            planner = backend.motion_planner
            try:
                for case in rows:
                    if case['arm'] != arm:
                        continue
                    binding = binding_for(case)
                    backend.resident_geometry.update(backend, case['feedback'], binding)
                    peer = 'B' if arm == 'A' else 'A'
                    backend.update_peer_collision_model(None, np.rad2deg(case['joints'][arm]),
                                                        np.rad2deg(case['joints'][peer]))
                    if binding is not None:
                        backend.attach_measured_carrying(binding, np.rad2deg(case['joints'][arm]))
                    state = backend._state(np.rad2deg(case['joints'][arm]))
                    targets = [('planner', planner.kinematics), ('checker', backend._motion_clearance_checker.kinematics)]
                    if not ik_only:
                        targets += [('ik', planner.ik_solver.kinematics), ('graph', planner.graph_planner.kinematics)]
                    errors = {}
                    for label, kin in targets:
                        balls = kin.compute_kinematics(state).robot_spheres.detach().cpu().numpy().reshape(-1, 4)
                        params = kin.config.kinematics_config
                        differences = []
                        for link, expected in case['expected_spheres'].items():
                            indices = params.get_sphere_index_from_link_name(link).detach().cpu().numpy()
                            actual = balls[indices]
                            actual = actual[actual[:, 3] > 0]
                            expected = np.asarray(expected)
                            assert actual.shape == expected.shape, (arm, capacity, case['phase'], label, link)
                            delta = float(np.max(np.abs(actual - expected)))
                            assert delta < 5e-6, (arm, capacity, case['phase'], label, link, delta)
                            differences.append(delta)
                        for owner in ('A', 'B'):
                            indices = params.get_sphere_index_from_link_name('jev_carried_object_' + owner).detach().cpu().numpy()
                            count = int(np.sum(balls[indices, 3] > 0))
                            assert count == (len(carrying_spheres(binding)) if binding and owner == arm else 0)
                            assert count <= 31
                        errors[label] = max(differences)
                    row = {'arm': arm, 'capacity': capacity, 'ik_only': ik_only,
                           'phase': case['phase'], 'max_sphere_error_m': errors, 'attached': binding is not None}
                    report.append(row)
                    print(json.dumps(row), flush=True)
            finally:
                backend.destroy()
                del backend, planner
                gc.collect()
                torch.cuda.empty_cache()
    output.write_text(json.dumps({'motion_executed': False, 'rows': report}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--gpu', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.reference:
        reference(args.output)
    elif args.gpu:
        gpu(args.gpu, args.output)
    else:
        parser.error('choose --reference or --gpu')
