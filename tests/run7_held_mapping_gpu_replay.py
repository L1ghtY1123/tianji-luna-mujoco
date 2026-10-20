"""Replay actual held RGBD through public raw-depth/semantic Mapper adapters."""

import argparse
import json
import time

import numpy as np
from tianji_curobo.curobo_service import materialize_urdf

from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity
from jev_mujoco.perception_mapping import decode_mapping_payload
from jev_mujoco.semantic_transport import IDENTITY_FIELDS, encode_project_semantic_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty-head", action="store_true", help="Exercise genuinely empty head semantic masks with recorded raw depth")
    options = parser.parse_args()
    payload = json.loads((ROOT / "data/visible-cloud-work/run7-replay/017-perception-build.json").read_text())["body"]
    if options.empty_head:
        decoded, _ = decode_mapping_payload(payload)
        request = {key: decoded[key] for key in (*IDENTITY_FIELDS, "views")}
        request["views"]["3"]["instance_masks"] = {}
        metadata = {key: value for key, value in payload.items() if key not in (*IDENTITY_FIELDS, "views")}
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    report = {"sdk": sdk_identity()[0], "physical_motion": False, "empty_head_semantics": options.empty_head, "stages": {}}
    for state in ("held", "released"):
        started = time.perf_counter()
        values = (encode_project_semantic_request(request, exclusion_state=state) | metadata
                  if options.empty_head else payload)
        result = run_operation(runtime, "/perception/build", values | {"exclusion_state": state})
        source = result["target_id"]
        assert result["ok"] and result["instance_metrics"][source]["input_point_count"] > 1000
        assert result["implicit_sensor_filter"] is False and result["implicit_pointcloud_filter"] is False
        assert result["background_includes_selected"] is False
        assert (source in result["excluded_object_ids"]) == (state == "held")
        assert (source in result["collision_objects"]) == (state == "released")
        restoration = result["source_restoration"]
        assert restoration["1"]["unsupported_point_count"] == 18
        assert restoration["1"]["supported_point_count"] == 1271
        assert restoration["2"]["supported_point_count"] == 14784
        assert "3" not in restoration  # head has no source mask; no mask invented.
        assert all(value["optional_rejected_returns_restored"] is False for value in restoration.values())
        guard = runtime.observed_free_guard
        assert guard.source_contact_mask_policy == "current_registered_measured_surface_support"
        source_mask_counts = [int(view[4].get(source, np.zeros((1, 1), dtype=bool)).sum()) for view in guard.views]
        assert source_mask_counts == [1271, 14784, 0]
        balls = guard.capture_spheres
        report["stages"][state] = {"wall_s": time.perf_counter()-started,
            "scene_token": result["scene_token"], "source_restoration": restoration,
            "instance_metrics": result["instance_metrics"], "camera_metrics": result["camera_metrics"],
            "table": result["table"], "robot_sphere_count": len(balls),
            "minimum_robot_sphere_radius_m": float(np.min(balls[:, 3])),
            "guard_source_contact_mask_policy": guard.source_contact_mask_policy,
            "guard_supported_source_pixel_counts": source_mask_counts,
            "layer_count": result["layer_count"], "excluded_object_ids": result["excluded_object_ids"],
            "background_includes_selected": result["background_includes_selected"],
            "collision_objects": result["collision_objects"], "timings_ms": result["timings_ms"]}
        print("STAGE " + state + " PASS", flush=True)
    runtime.clear_carrying()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
