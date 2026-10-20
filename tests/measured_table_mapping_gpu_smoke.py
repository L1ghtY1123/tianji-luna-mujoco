"""Offline table reuse/crop and explicit semantic-only capture probe."""

import json
from pathlib import Path

import numpy as np


def main():
    from tianji_curobo.curobo_service import materialize_urdf

    from jev_mujoco.perception_mapping import PerceptionMapper, create_feedback_segmenter

    root = Path("/opt/jev")
    urdf = materialize_urdf(root / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=root / "assets/marvin_urdf/meshes", rag_meshes=root / "assets/marvin_urdf/rag100_050/meshes")
    segmenter = create_feedback_segmenter(robot_config_path=root / "resources/curobo/marvin_gui_dual.yml",
        urdf_path=urdf, asset_root_path=urdf.parent, feedback={"A": 0., "B": 0.}).sdk_boundary
    mapper = PerceptionMapper(segmenter)
    depth = np.full((96, 96), .6, dtype=np.float32)
    masks = {"selected": np.zeros(depth.shape, dtype=np.uint8), "container": np.zeros(depth.shape, dtype=np.uint8)}
    masks["selected"][42:54, 56:68] = 1
    masks["container"][30:42, 25:37] = 1
    depth[masks["selected"] != 0] = .54
    depth[masks["container"] != 0] = .53
    depth[45, 60] = .62  # Deliberate measured return below fitted support.
    depth[20:24, 42:46] = .57  # Unclassified measured obstacle above support.
    transform = [[1, 0, 0, .55], [0, -1, 0, 0], [0, 0, -1, 1.3], [0, 0, 0, 1]]
    intrinsics = [[120, 0, 47.5], [0, 120, 47.5], [0, 0, 1]]
    views = {camera: {"camera_id": camera, "stereo_sequence": 1, "depth_left_m": depth.copy(),
        "valid_left_mask": np.ones(depth.shape, dtype=np.uint8), "left_intrinsics": intrinsics,
        "T_base_left_camera": transform,
        "robot_joints_at_capture": {"A": [0, 0, 0, -42.5, 0, 0, 0], "B": [0, 0, 0, -42.5, 0, 0, 0]},
        "rgb_size": [96, 96], "rgb_intrinsics": intrinsics, "left_to_rgb": np.eye(4).tolist(),
        "instance_masks": {name: mask.copy() for name, mask in masks.items()}} for camera in ("1", "2", "3")}
    payload = {"views": views, "confirmed_instance_id": "selected", "selection_generation": 1,
        "observation_set_id": "table-1", "exclusion_state": "selected",
        "experimental_semantic_only_mapping": True, "full_observation_complete": False,
        "planning_settings": {"experimental_semantic_only_mapping": True},
        "camera_capture_metadata": {camera: {"calibration_id": "table-fixture-v1"} for camera in views}}
    initial, _, _ = mapper.build(payload)
    initial_diagnostics = dict(mapper.diagnostics)
    assert initial.integration.point_count > 0
    assert not initial_diagnostics["semantic_only_active"]
    full, grids, _ = mapper.build(payload | {"observation_set_id": "table-2", "selection_generation": 2,
                                           "full_observation_complete": True})
    diagnostics = dict(mapper.diagnostics)
    assert diagnostics["semantic_only_active"] and diagnostics["table_cache"]["cache_hit"]
    assert full.table is initial.table
    assert full.integration.point_count == 0
    assert {grid.name for grid in grids} == {"semantic_instance::selected", "semantic_instance::container"}
    assert diagnostics['background_grid_omitted']
    assert diagnostics["removed_unclassified_background_point_count"] > 0
    assert full.table.model["pose"][2] + full.table.model["dims"][2] / 2 == full.table.height_m
    removed = sum(item["removed_table_penetrating_instance_point_count"] for item in full.camera_metrics.values())
    assert removed == 3
    for layer in full.scene_model["voxel"].values():
        assert np.asarray(layer["centers_base_m"])[:, 2].min() >= full.table.height_m - 1e-6
    repeated, repeated_grids, _ = mapper.build(payload | {'observation_set_id': 'table-3',
        'selection_generation': 3, 'full_observation_complete': True})
    assert mapper.diagnostics['background_grid_omitted']
    assert repeated.table is full.table and repeated.integration.point_count == 0
    assert {grid.name for grid in repeated_grids} == {"semantic_instance::selected", "semantic_instance::container"}
    assert all(new is not old for new, old in zip(repeated_grids, grids, strict=True))
    repeated_diagnostics = dict(mapper.diagnostics)
    mapper.reset()
    assert mapper.builder.table_cache.table is None
    assert mapper.mapper.esdf_grid is None
    print(json.dumps({"initial_background_surface_count": initial.integration.point_count,
        "full_background_surface_count": full.integration.point_count,
        "table_cache_reused": True, "reset_invalidated_table": True,
        "empty_background_not_allocated": True,
        "removed_table_penetrating_instance_points": removed,
        "table": dict(full.table.model), "diagnostics": diagnostics,
        "repeated_diagnostics": repeated_diagnostics,
        "timings_ms": {'initial': dict(initial.timings_ms), 'full': dict(full.timings_ms),
                       'repeat': dict(repeated.timings_ms)}}, indent=2))


if __name__ == "__main__":
    main()
