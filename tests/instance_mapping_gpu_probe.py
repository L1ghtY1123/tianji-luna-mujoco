"""Explicit offline GPU acceptance; saved RGBD only, no service or robot calls."""
import argparse
import cProfile
import json
import time
from pathlib import Path

import numpy as np


def recorded_views(directory):
    """Restore masks only from saved SAM3 measured point/pixel provenance."""
    from jev_mujoco.semantic_transport import encode_project_semantic_request
    capture = json.loads((directory / 'capture.json').read_text())
    observed = json.loads((directory / 'observation.json').read_text())
    for camera, view in capture['views'].items():
        with np.load(directory / f'camera-{camera}-rgbd.npz') as stored:
            view.update({key: stored[key] for key in stored.files})
        view['instance_masks'] = {}
    for index, obj in enumerate(observed['objects']):
        with np.load(directory / f'instance-{index}.npz') as stored:
            for camera, view in capture['views'].items():
                pixels = stored['pixels_rc'][stored['camera_ids'] == camera]
                mask = np.zeros(view['depth_m'].shape, dtype=np.uint8)
                mask[pixels[:, 0], pixels[:, 1]] = 1
                view['instance_masks'][obj['id']] = mask
    views = {}
    for camera, source in capture['views'].items():
        height, width = source['depth_m'].shape
        top, left = (480 - height) // 2, (640 - width) // 2
        def pad(value, top=top, left=left, height=height, width=width):
            result = np.zeros((480, 640), dtype=value.dtype)
            result[top:top + height, left:left + width] = value
            return result
        view = source | {'depth_m': pad(source['depth_m']), 'valid_depth': pad(source['valid_depth']),
            'instance_masks': {key: pad(mask) for key, mask in source['instance_masks'].items()}}
        k = source['intrinsics'] | {'width': 640, 'height': 480,
            'cx': source['intrinsics']['cx'] + left, 'cy': source['intrinsics']['cy'] + top}
        intrinsics = [[k['fx'], 0., k['cx']], [0., k['fy'], k['cy']], [0., 0., 1.]]
        views[camera] = {'camera_id': camera, 'stereo_sequence': capture['generation'],
            'depth_left_m': view['depth_m'], 'valid_left_mask': view['valid_depth'],
            'left_intrinsics': intrinsics, 'rgb_intrinsics': intrinsics,
            'T_base_left_camera': view['T_world_camera'], 'robot_joints_at_capture': view['robot_joints_deg'],
            'rgb_size': [k['width'], k['height']], 'left_to_rgb': np.eye(4).tolist(),
            'instance_masks': view['instance_masks']}
    semantic = {'views': views, 'observation_set_id': capture['observation_id'],
        'selection_generation': capture['generation'], 'sam3_generation': capture['generation'],
        'confirmed_instance_id': observed['objects'][0]['id']}
    return encode_project_semantic_request(semantic, exclusion_state='selected') | {
        'exclusion_state': 'selected', 'gripper_joint_positions_rad': {'A': 0., 'B': 0.},
        'planning_settings': {'experimental_semantic_only_mapping': True}, 'full_observation_complete': True}


def collision_oracle(grids, model):
    import torch
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.scene import Scene
    from scipy.spatial import cKDTree
    result = {}
    for grid in grids:
        surface = np.asarray(model['voxel'][grid.name].get('centers_base_m', []))
        if not len(surface):
            continue
        axis = np.linspace(-.18, .18, 25)
        offsets = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
        centers = surface.mean(axis=0) + offsets
        radius = .13234879076480865
        query = torch.tensor(np.column_stack((centers, np.full(len(centers), radius))),
            device='cuda', dtype=torch.float32).reshape(1, 1, -1, 4)
        result[grid.name] = {}
        for clearance in (.01, .03):
            checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                scene_model=Scene(voxel=[grid]), collision_activation_distance=clearance))
            checker.scene_model.data.voxels.load_batch([grid], env_idx=0)
            checker.scene_model.data.voxels.params[..., :3].round_()
            costs = checker.get_collision_distance(query).detach().cpu().numpy().reshape(-1)
            contact = cKDTree(surface).query(centers)[0] < radius + clearance - .001
            missed = int(np.count_nonzero(contact & (costs == 0)))
            result[grid.name][str(clearance)] = {'queries': len(centers), 'missed_surface_contacts': missed}
    return result


def compare_baseline(report, grids, baseline):
    previous = json.loads((baseline / 'native-report.json').read_text())
    fields = np.load(baseline / 'native-esdf.npz')
    for grid in grids:
        name = grid.name
        before = np.asarray(previous['model']['voxel'][name]['centers_base_m'])
        after = np.asarray(report['model']['voxel'][name]['centers_base_m'])
        np.testing.assert_array_equal(before[np.lexsort(before.T[::-1])], after[np.lexsort(after.T[::-1])])
        old, new = fields[name], grid.feature_tensor.cpu().numpy()
        delta = np.array(old.shape) - new.shape
        assert np.all(delta >= 0) and np.all(delta % 2 == 0)
        overlap = old[tuple(slice(int(i), int(i + n)) for i, n in zip(delta // 2, new.shape, strict=True))]
        np.testing.assert_array_equal(overlap, new)
        assert report['collision_oracle'][name] == previous['collision_oracle'][name]
    return {'identical_tsdf_surfaces': True, 'identical_retained_esdf_cells': True,
            'identical_sampled_collision_results': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mode', choices=['native', 'serial', 'parallel', 'compact'], required=True)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--coordinate-frame', choices=['world', 'local'], default='local')
    parser.add_argument('--kernel-cache', type=Path)
    args = parser.parse_args()
    if args.kernel_cache is not None:
        import warp
        warp.config.kernel_cache_dir = str(args.kernel_cache)
    import torch
    from tianji_curobo.curobo_service import materialize_urdf

    from jev_mujoco.geometry_service import ROOT, sdk_identity
    from jev_mujoco.perception_mapping import (
        PerceptionMapper,
        create_feedback_segmenter,
        decode_mapping_payload,
        filter_mapping_views,
        mapping_filter_settings,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = recorded_views(args.input) if args.input.is_dir() else json.loads(args.input.read_text())['body']
    # Mapping replay does not execute release; adapt this historical setting
    # to the current validated planner contract, keeping sensor data intact.
    payload.setdefault('planning_settings', {})['place_release_height_m'] = .015
    decoded, identity = decode_mapping_payload(payload)
    filter_mapping_views(decoded, mapping_filter_settings(payload))
    urdf = materialize_urdf(ROOT / 'assets/marvin_urdf/marvin_with_grippers.urdf',
        marvin_meshes=ROOT / 'assets/marvin_urdf/meshes',
        rag_meshes=ROOT / 'assets/marvin_urdf/rag100_050/meshes', cache_root=args.output / 'urdf')
    segmenter = create_feedback_segmenter(robot_config_path=ROOT / 'resources/curobo/marvin_gui_dual.yml',
        urdf_path=urdf, asset_root_path=urdf.parent, feedback=identity['gripper_joint_positions_rad'])
    mapper = PerceptionMapper(segmenter.sdk_boundary)
    if args.mode == 'native':
        mapper.builder.instance_mapper = None
    else:
        mapper.builder.instance_mapper.max_workers = 1 if args.mode == 'serial' else 4
        mapper.builder.instance_mapper.compact_esdf = args.mode == 'compact'
        mapper.builder.instance_mapper.local_frame = args.coordinate_frame == 'local'
    report = {'mode': args.mode, 'coordinate_frame': args.coordinate_frame,
              'sdk': sdk_identity()[0], 'input': str(args.input),
              'physical_motion': False, 'scope': 'PerceptionMapper.build; excludes capture/segmentation model configuration', 'rows': []}
    for iteration in range(args.repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        profiler = cProfile.Profile()
        if args.profile:
            profiler.enable()
        candidate, grids, model = mapper.build(decoded)
        torch.cuda.synchronize()
        if args.profile:
            profiler.disable()
            profiler.dump_stats(args.output / f'{args.mode}-{iteration}.prof')
        row = {'iteration': iteration, 'wall_s': time.perf_counter() - started,
            'timings_ms': dict(candidate.timings_ms), 'instance_metrics': {k: dict(v) for k, v in candidate.instance_metrics.items()},
            'grid_cells': {g.name: g.feature_tensor.numel() for g in grids},
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mb': torch.cuda.max_memory_reserved() / 2**20}
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
    report['model'] = model
    report['collision_oracle'] = collision_oracle(grids, model)
    if args.baseline is not None:
        report['baseline_comparison'] = compare_baseline(report, grids, args.baseline)
    np.savez_compressed(args.output / (args.mode + '-esdf.npz'),
                        **{g.name: g.feature_tensor.cpu().numpy() for g in grids})
    (args.output / (args.mode + '-report.json')).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
