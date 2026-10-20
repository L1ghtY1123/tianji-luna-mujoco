"""Offline native-padding TSDF preservation and measured-surface oracle probe."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    import torch
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.scene import Scene
    from tianji_curobo.curobo_service import materialize_urdf
    from tianji_curobo.robot_segmenter import FfsDepthObservationRequest, SemanticProjectionRequest
    from tianji_curobo.volumetric_mapper import CuroboVolumetricMapperBoundary

    from jev_mujoco.perception_mapping import CaptureSegmenter, create_feedback_segmenter
    from jev_mujoco.semantic_builder import rounded_collision_padding

    parser = argparse.ArgumentParser()
    parser.add_argument("--offset-x", type=float, default=0.)
    parser.add_argument("--depth", type=float, default=.54)
    parser.add_argument("--half-width", type=int, default=10)
    args = parser.parse_args()
    root = Path("/opt/jev")
    urdf = materialize_urdf(root / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=root / "assets/marvin_urdf/meshes", rag_meshes=root / "assets/marvin_urdf/rag100_050/meshes")
    segmenter = create_feedback_segmenter(robot_config_path=root / "resources/curobo/marvin_gui_dual.yml",
        urdf_path=urdf, asset_root_path=urdf.parent, feedback={"A": 0., "B": 0.}).sdk_boundary
    capture = CaptureSegmenter(segmenter)
    depth = np.zeros((64, 64), dtype=np.float32)
    depth[32-args.half_width:32+args.half_width, 32-args.half_width:32+args.half_width] = args.depth
    transform = [[1, 0, 0, .55+args.offset_x], [0, -1, 0, 0], [0, 0, -1, 1.3], [0, 0, 0, 1]]
    intrinsics = [[120, 0, 31.5], [0, 120, 31.5], [0, 0, 1]]
    joints = {"A": [0, 0, 0, -42.5, 0, 0, 0], "B": [0, 0, 0, -42.5, 0, 0, 0]}
    prepared, points = [], {}
    for camera in ("1", "2", "3"):
        view = {"camera_id": camera, "stereo_sequence": 1, "depth_left_m": depth,
            "valid_left_mask": (depth > 0).astype(np.uint8), "left_intrinsics": intrinsics,
            "T_base_left_camera": transform, "robot_joints_at_capture": joints,
            "rgb_size": [64, 64], "rgb_intrinsics": intrinsics, "left_to_rgb": np.eye(4).tolist(),
            "instance_masks": {"object": (depth > 0).astype(np.uint8)}}
        raw = capture.prepare_ffs_observation(FfsDepthObservationRequest.from_mapping(view))
        semantic = capture.remove_semantic_instances(raw, SemanticProjectionRequest.from_mapping(view))
        prepared.append(semantic)
        points[camera] = semantic.instance_points_base_m["object"]
    radius = float(np.max(segmenter.robot_spheres(joints)[:, 3]))
    table_crop = {"pose": [.55, 0, 1.2, 1, 0, 0, 0], "dims": [1, 1, 1]}
    results, grids = {}, {}
    for name, padding in (("legacy", radius + .5 + .02), ("bounded", rounded_collision_padding(radius + .03 + .02))):
        mapper = CuroboVolumetricMapperBoundary(robot_segmenter_boundary=capture,
            extent_meters_xyz=(.3, .3, .3), grid_center_meters=(.55, 0, .76))
        mapper.integrate_multiview(prepared, table_model=table_crop)
        elapsed = []
        for _ in range(4):
            mapper.clear_instance_layers()
            torch.cuda.synchronize()
            started = time.perf_counter()
            geometry = mapper.build_instance_esdf(prepared, "object", points, padding_m=padding,
                                                   table_model=table_crop)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - started) * 1000)
        grid = mapper.scene_voxel_grids[-1]
        grids[name] = grid
        results[name] = {"padding_m": padding, "dims_m": grid.dims,
            "shape": list(grid.feature_tensor.shape), "cells": grid.feature_tensor.numel(),
            "cold_ms": elapsed[0], "warm_ms": elapsed[1:],
            "surface_centers": geometry.model["centers_base_m"]}
    surfaces = {}
    for name in results:
        centers = np.asarray(results[name]["surface_centers"])
        surfaces[name] = centers[np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))]
    np.testing.assert_array_equal(surfaces["legacy"], surfaces["bounded"])
    old, new = grids["legacy"], grids["bounded"]
    old_shape = np.array(old.feature_tensor.shape)
    new_shape = np.array(new.feature_tensor.shape)
    shift = (np.asarray(new.pose[:3]) - old.pose[:3]) / .01 + (old_shape - new_shape) / 2
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree
    cells = np.indices(new_shape).reshape(3, -1).T
    world = np.asarray(new.pose[:3]) + (cells - (new_shape - 1) / 2) * new.voxel_size
    previous = map_coordinates(old.feature_tensor.cpu().numpy().astype(float), (cells + shift).T, order=1)
    new_values = new.feature_tensor.cpu().numpy().reshape(-1).astype(float)
    difference = np.abs(previous-new_values)
    max_difference = float(difference.max())
    exact = cKDTree(surfaces["bounded"]).query(world)[0]
    near = exact <= radius + .03
    results["overlap_probe"] = {"phase_shift_voxels": shift.tolist(),
        "changed_cell_count": int(np.count_nonzero(difference > .001)),
        "near_field_maximum_overestimate_vs_surface_m": float(np.max(new_values[near] - exact[near]))}
    assert results["overlap_probe"]["near_field_maximum_overestimate_vs_surface_m"] <= .0001
    axis = np.linspace(-.25, .25, 31)
    centers = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3) + [.55+args.offset_x, 0, 1.3-args.depth]
    query = torch.tensor(np.column_stack((centers, np.full(len(centers), radius))), device="cuda:0",
                         dtype=torch.float32).reshape(1, 1, -1, 4)
    results["collision_probe"] = {}
    for clearance in (.01, .03):
        costs = {}
        for name, grid in grids.items():
            checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                scene_model=Scene(voxel=[grid]), collision_activation_distance=clearance))
            checker.scene_model.data.voxels.load_batch([grid], env_idx=0)
            checker.scene_model.data.voxels.params[..., :3].round_()
            costs[name] = checker.get_collision_distance(query).detach().cpu().numpy().reshape(-1)
        collision = cKDTree(surfaces["bounded"]).query(centers)[0] < radius + clearance - .001
        results["collision_probe"][str(clearance)] = {
            "maximum_cost_difference": float(np.max(np.abs(costs["legacy"]-costs["bounded"]))),
            "classification_disagreement_count": int(np.count_nonzero((costs["legacy"] > 0) != (costs["bounded"] > 0))),
            "bounded_surface_oracle_false_negatives": int(np.count_nonzero(collision & (costs["bounded"] == 0))),
            "legacy_surface_oracle_false_negatives": int(np.count_nonzero(collision & (costs["legacy"] == 0)))}
        assert np.any(costs["legacy"] > 0) and np.any(costs["legacy"] == 0)
        assert results["collision_probe"][str(clearance)]["bounded_surface_oracle_false_negatives"] == 0
    surface_count = len(results["bounded"].pop("surface_centers"))
    results["legacy"].pop("surface_centers")
    results.update(max_esdf_overlap_difference_m=max_difference, shared_tsdf_surface_count=surface_count,
        collision_queries=len(centers), torch=torch.__version__,
        cell_reduction_fraction=1-results["bounded"]["cells"]/results["legacy"]["cells"], fixture=vars(args))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
