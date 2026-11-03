# Luna SAM6D client

The unified implementation is maintained at `C:/project/tianji-algorithms/services/tianji_sam6d` and `/home/guangyu/projects/tianji-algorithms/services/tianji_sam6d`.
Luna includes only its small client/contracts package under `src/tianji_sam6d`, so existing source mounts and editable installs can import it without replacing the application image.

```python
from luna_mujoco.pose_estimation import CadPoseEstimator

poses = CadPoseEstimator(algorithm="sam6d").segment_and_estimate(
    frozen_view, "blue box", model_id="blue_box"
)
```

The view is the existing camera RGBD record, with rgb, depth_m, valid_depth, intrinsics, T_world_camera, frame_id, camera_id, calibration_id, simulation_generation and capture_time.
For a previously selected mask call `estimate_instance` with its capture/camera IDs.
The caller explicitly binds the correct CAD reference. Multiple SAM3 candidates stay separate.

Default endpoint: `http://127.0.0.1:8364`; override `LUNA_SAM6D_URL` when needed.
Deployment is owned by the unified repository's `deploy/sam6d.compose.yaml`.
The running Luna host-network web container can reach this endpoint.
The existing SAM3 endpoint and model are reused.

T_camera_object and T_world_object describe the canonical CAD frame (AABB-centred, source axes preserved); T_world_source describes the original CAD frame. All translations are metres and optical camera axes are right/down/forward.
Returned scores are not calibrated correctness probabilities.
Only blue_box and the official validation example are initially registered. Pink approximate foam is excluded; additional rigid CADs are registered through the independent service image without restarting Luna.

Verified: real SAM3 -> SAM6D calls from the running Luna container on an archived blue-box RGB-D fixture, with source/camera/world transforms. Contract tests reject capture mismatch and verify rotated source-origin composition.
Not verified: real robot grasp/place execution, live-scene accuracy, automatic CAD identity binding.

This opt-in API does not change existing robot skill contracts, task orchestration, inventory identity, collision maps or placement rules.

SAM6D is retained as a stopped backup as of the FoundationPose integration. Start its compose service explicitly before selecting algorithm="sam6d".
