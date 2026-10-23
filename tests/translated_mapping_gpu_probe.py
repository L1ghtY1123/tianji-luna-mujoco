"""Offline coordinate-translation/cache regression using measured RGBD inputs."""
import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
from instance_mapping_gpu_probe import recorded_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--frame', choices=['world', 'local'], required=True)
    args = parser.parse_args()
    import torch
    import warp
    from curobo.types import Pose
    from tianji_curobo.curobo_service import materialize_urdf

    from luna_mujoco.geometry_service import ROOT
    from luna_mujoco.instance_mapping import InstanceMapper
    from luna_mujoco.perception_mapping import (
        PerceptionMapper,
        create_feedback_segmenter,
        decode_mapping_payload,
        filter_mapping_views,
        mapping_filter_settings,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    warp.config.kernel_cache_dir = str(args.output / 'warp-cache')
    payload = recorded_views(args.input)
    decoded, identity = decode_mapping_payload(payload)
    filter_mapping_views(decoded, mapping_filter_settings(payload))
    urdf = materialize_urdf(ROOT / 'assets/marvin_urdf/marvin_with_grippers.urdf',
        marvin_meshes=ROOT / 'assets/marvin_urdf/meshes',
        rag_meshes=ROOT / 'assets/marvin_urdf/rag100_050/meshes', cache_root=args.output / 'urdf')
    segmenter = create_feedback_segmenter(robot_config_path=ROOT / 'resources/curobo/marvin_gui_dual.yml',
        urdf_path=urdf, asset_root_path=urdf.parent, feedback=identity['gripper_joint_positions_rad'])
    perception = PerceptionMapper(segmenter.sdk_boundary)

    class RecordedMapper(InstanceMapper):
        def build(self, observations, instances, **kwargs):
            self.recorded_inputs = (observations, instances, kwargs)
            return super().build(observations, instances, **kwargs)

    mapper = RecordedMapper(segmenter.sdk_boundary, local_frame=args.frame == 'local')
    perception.builder.instance_mapper = mapper
    perception.build(decoded)
    observations, instances, kwargs = mapper.recorded_inputs
    rows = [{'translation_m': [0, 0, 0], 'wall_ms': mapper.wall_ms,
             'cache_hits': sum(m['cache_hit'] for m in mapper.metrics.values())}]
    # This changes the coordinate origin of the recorded measurement, not the
    # scene or robot. It isolates kernel reuse from new perception and physics.
    for index, offset in enumerate(([.02, 0., 0.], [.04, -.02, 0.], [.06, -.04, 0.])):
        shifted = []
        for view in observations:
            camera = view.camera_observation
            shift = torch.tensor(offset, device=camera.pose.position.device, dtype=camera.pose.position.dtype)
            shifted.append(replace(view, camera_observation=replace(camera,
                pose=Pose(position=camera.pose.position + shift, quaternion=camera.pose.quaternion))))
        points = {name: {cam: p + np.asarray(offset, dtype=p.dtype) for cam, p in views.items()}
                  for name, views in instances.items()}
        options = deepcopy(kwargs)
        options['table_model']['pose'][:3] = (np.asarray(options['table_model']['pose'][:3]) + offset).tolist()
        geometries = mapper.build(shifted, points, **options)
        rows.append({'translation_m': offset, 'wall_ms': mapper.wall_ms,
                     'cache_hits': sum(m['cache_hit'] for m in mapper.metrics.values()),
                     'metrics': deepcopy(mapper.metrics)})
        np.savez_compressed(args.output / f'grids-{index}.npz',
                            **{g.name: g.feature_tensor.cpu().numpy() for g in mapper.grids})
        (args.output / f'geometry-{index}.json').write_text(json.dumps(
            {name: dict(g.model) for name, g in geometries.items()}))
    (args.output / 'report.json').write_text(json.dumps({'frame': args.frame, 'rows': rows}, indent=2))
    print(json.dumps({'frame': args.frame, 'rows': [{k: v for k, v in row.items() if k != 'metrics'}
                                                for row in rows]}), flush=True)


if __name__ == '__main__':
    main()
