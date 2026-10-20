"""Compare project calibration to public MuJoCo camera frusta without GL."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from jev_mujoco.cameras import load_calibration
from jev_mujoco.kinematics import robot_model

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/visible-cloud-work/mujoco-pixel-center-corrected-audit.json")
options = parser.parse_args()
model = robot_model()
report = {"mujoco": mujoco.__version__, "gpu_used": False, "gl_used": False, "cameras": {}}
for camera_id, definition in load_calibration()["cameras"].items():
    calibration = definition["intrinsics"]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = model.camera("rgbd_" + camera_id).id
    vertical, horizontal, clip = (np.zeros(2, dtype=np.float32) for _ in range(3))
    mujoco.mjv_cameraFrustum(vertical, horizontal, clip, model, camera)
    width, height = calibration["width"], calibration["height"]
    left, right = -horizontal[0], horizontal[1]
    bottom, top = -vertical[1], vertical[0]
    native = {"fx": float(clip[0]*width/(right-left)), "fy": float(clip[0]*height/(top-bottom)),
              "cx": float(-width*left/(right-left)), "cy": float(height*top/(top-bottom))}
    report["cameras"][camera_id] = {"project_K": calibration, "render_edge_coordinate_K": native,
        "render_integer_index_center_K": {**native, "cx": native["cx"]-.5, "cy": native["cy"]-.5},
        "edge_K_minus_project_K": {k: native[k]-calibration[k] for k in native},
        "offsamples": int(model.vis.quality.offsamples)}
    print(camera_id, json.dumps(report["cameras"][camera_id]), flush=True)
report["source"] = {
    "pixel_ray": "https://github.com/google-deepmind/mujoco/blob/3.10.0/src/engine/engine_util_misc.c#L470",
    "frustum": "https://github.com/google-deepmind/mujoco/blob/3.10.0/src/engine/engine_vis_visualize.c#L403",
    "classic_projection": "https://github.com/google-deepmind/mujoco/blob/3.10.0/src/render/classic/render_gl3.c#L657"}
center_errors = [abs(values["render_integer_index_center_K"][key] - values["project_K"][key])
                 for values in report["cameras"].values() for key in ("cx", "cy")]
report["maximum_integer_center_principal_error_px"] = max(center_errors)
report["integer_center_projection_matches_project_calibration"] = max(center_errors) < 1e-4
report["limits"] = ["No calibration, ray, mask, point or product source was changed.",
                    "Multisampling and depth resolve can sample within the pixel footprint; this audit does not prove exact depth-sample position."]
(ROOT / options.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
