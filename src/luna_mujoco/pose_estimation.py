"""SAM3 -> FoundationPose pose estimates from frozen registered camera views.

CAD poses are hypotheses only. This module never changes collision geometry,
placement regions, inventory identity, or robot execution.
"""
import numpy as np

from tianji_foundationpose.contracts import PoseError, validate_pose

# FoundationPose accepts metric depth only in [0, 65.535], with zero invalid.
POSE_TRANSPORT_MAX_DEPTH_M = 65.535


def snapshot_view(view):
    required = ("frame_id", "camera_id", "calibration_id", "simulation_generation", "capture_time")
    if any(key not in view for key in required):
        raise PoseError("invalid_capture")
    rgb = np.asarray(view["rgb"]).copy()
    depth = np.asarray(view["depth_m"], dtype=np.float32).copy()
    valid = np.asarray(view["valid_depth"], dtype=bool)
    if rgb.shape[:2] != depth.shape or valid.shape != depth.shape:
        raise PoseError("invalid_capture")
    if np.any(valid & (~np.isfinite(depth) | (depth < 0))):
        raise PoseError("invalid_capture", "valid depth must be finite and nonnegative")
    out_of_range = valid & (depth > POSE_TRANSPORT_MAX_DEPTH_M)
    zero = valid & (depth == 0)
    transport = {"policy": "invalidate_depth_outside_pose_transport_range",
                 "transport_max_depth_m": POSE_TRANSPORT_MAX_DEPTH_M,
                 "source_invalid_depth_pixels": int(np.count_nonzero(~valid)),
                 "input_valid_depth_pixels": int(np.count_nonzero(valid)),
                 "out_of_range_depth_pixels": int(np.count_nonzero(out_of_range)),
                 "zero_depth_pixels": int(np.count_nonzero(zero))}
    valid = valid & ~out_of_range & ~zero
    transport["transport_valid_depth_pixels"] = int(np.count_nonzero(valid))
    depth[~valid] = 0
    return {**view, "rgb":rgb, "depth_m":depth, "valid_depth":valid.copy(),
            "depth_transport":transport,
            "intrinsics":dict(view["intrinsics"]),
            "T_world_camera":validate_pose(view["T_world_camera"]).copy()}


class CadPoseEstimator:
    def __init__(self, pose_client=None, segmentation_client=None):
        if pose_client is None:
            from .foundationpose_tracking import FoundationPoseTrackingClient
            pose_client = FoundationPoseTrackingClient()
        self.pose_client = pose_client
        self.segmentation_client = segmentation_client

    def estimate_instance(self, view, mask, score, *, model_id, mask_capture_id, mask_camera_id, tracking_seed=None):
        view = snapshot_view(view)
        if mask_capture_id != view["frame_id"] or mask_camera_id != view["camera_id"]:
            raise PoseError("capture_mismatch")
        intrinsics = view["intrinsics"]
        K = [[intrinsics["fx"],0,intrinsics["cx"]],
             [0,intrinsics["fy"],intrinsics["cy"]],[0,0,1]]
        options = {'model_id': model_id, 'capture_id': view['frame_id'], 'camera_id': view['camera_id'], 'score': score}
        if tracking_seed is None:
            result = self.pose_client.estimate(view['rgb'], view['depth_m'], mask, K, **options)
        else:
            seed = {key: tracking_seed[key] for key in ('instance_id', 'source_sha256', 'observation_id')}
            seed['T_camera_object'] = (np.linalg.inv(view['T_world_camera']) @ validate_pose(
                tracking_seed['T_world_object'])).tolist()
            result = self.pose_client.track(view['rgb'], view['depth_m'], mask, K, seed=seed, **options)
        if result.get("capture_id") != view["frame_id"] or result.get("camera_id") != view["camera_id"]:
            raise PoseError("provenance_mismatch")
        camera_pose = validate_pose(result["T_camera_object"])
        source_from_canonical = validate_pose(result["T_source_canonical_m"])
        world_pose = view["T_world_camera"] @ camera_pose
        return {**result,"T_world_object":world_pose.tolist(),
                "T_world_source":(world_pose @ np.linalg.inv(source_from_canonical)).tolist(),
                "calibration_id":view["calibration_id"],
                "simulation_generation":view["simulation_generation"],
                "depth_transport":view["depth_transport"],
                "capture_time":view["capture_time"],"usage":"pose_hypothesis_only"}

    def segment_and_estimate(self, view, text, *, model_id):
        """Caller explicitly binds model_id; every SAM3 instance remains separate."""
        # Validate for segmentation, but retain original depths for the per-pose
        # snapshot so transport filtering evidence is not lost on a second pass.
        segmentation_view = snapshot_view(view)
        segmenter = self.segmentation_client
        if segmenter is None:
            from .perception import Sam3Client
            segmenter = Sam3Client()
        detections = segmenter.segment(segmentation_view["rgb"],text,
                                       round(segmentation_view["capture_time"]*1_000_000))
        return [self.estimate_instance(view,detection["mask"],detection["score"],model_id=model_id,
                    mask_capture_id=view["frame_id"],mask_camera_id=view["camera_id"])
                for detection in detections]
