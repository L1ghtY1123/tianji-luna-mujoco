"""Small real GPU query: public VoxelGrid pose changes reuse the ESDF tensor."""
import json
from dataclasses import replace

import numpy as np


def main():
    import torch
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.scene import Scene, VoxelGrid

    axis = (np.arange(41) - 20) * .01
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1)
    # Pinned native VoxelData SDF: negative inside, positive outside.
    values = torch.as_tensor(np.linalg.norm(points, axis=-1) - .04, dtype=torch.float16, device='cuda:0')
    target = VoxelGrid(name='semantic_instance::target', pose=[0, 0, 0, 1, 0, 0, 0],
                       dims=[.41] * 3, voxel_size=.01, feature_tensor=values, feature_dtype=torch.float16)
    fixed = replace(target, name='semantic_instance::fixed', pose=[0, .5, 0, 1, 0, 0, 0])
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        scene_model=Scene(voxel=[target, fixed]), collision_activation_distance=0.))
    data = checker.scene_model.data.voxels
    data.load_batch([target, fixed], env_idx=0)
    data.params[..., :3].round_()
    # Query points sit inside the small sphere and inside its moved counterpart.
    spheres = torch.tensor([[[[.02, 0, 0, .005], [.52, 0, 0, .005], [.02, .5, 0, .005]]]], device='cuda:0')
    before = checker.get_collision_distance(spheres).detach().cpu().numpy().reshape(-1)
    moved = replace(target, pose=[.5, 0, 0, 1, 0, 0, 0])
    data.load_batch([moved, fixed], env_idx=0)
    data.params[..., :3].round_()
    after = checker.get_collision_distance(spheres).detach().cpu().numpy().reshape(-1)
    assert before[0] > 0 and before[1] == 0 and before[2] > 0, before
    assert after[0] == 0 and after[1] > 0 and after[2] == before[2], after
    assert moved.feature_tensor is target.feature_tensor
    print(json.dumps({'before': before.tolist(), 'after': after.tolist(), 'same_tensor': True,
                     'old_position_clear': True, 'new_position_colliding': True, 'fixed_layer_unchanged': True}))


if __name__ == '__main__':
    main()
