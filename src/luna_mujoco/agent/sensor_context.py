"""Measured RGBD metadata supplied with the corresponding current RGB images."""
import numpy as np


def rgbd_context(capture):
    views = {}
    for camera, view in capture['views'].items():
        item = {'rgb_shape': list(view['rgb'].shape), 'observation_id': capture['observation_id']}
        for field in ('K', 'T_world_camera'):
            if field in view:
                item[field] = np.asarray(view[field]).tolist()
        if 'depth_m' in view:
            depth = np.asarray(view['depth_m'])
            valid = np.isfinite(depth) & (depth > 0)
            if 'valid_depth' in view:
                valid &= np.asarray(view['valid_depth'], dtype=bool)
            values = depth[valid]
            item['depth'] = {'unit': 'metres', 'valid_pixels': int(valid.sum()),
                'total_pixels': int(depth.size), 'percentiles_5_50_95':
                    np.quantile(values, [.05, .5, .95]).tolist() if values.size else []}
        views[camera] = item
    return {'observation_id': capture['observation_id'], 'views': views,
            'scope': 'Measured sensor context. Skills consume workspace-filtered RGBD for object geometry and planning.'}
