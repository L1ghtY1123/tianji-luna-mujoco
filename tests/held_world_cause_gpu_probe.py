"""Read-only localization of the recorded trial12 held world's positive cost."""

import json

import numpy as np
import torch
import yaml
from tianji_curobo.curobo_service import materialize_urdf

from luna_mujoco.carrying import CarryingBinding
from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation

folder = ROOT / "data/visible-cloud-work/trial12-place-diagnostic"
mapping = json.loads((folder / "gpu-requests/017-perception-build.json").read_text())
payload = json.loads((folder / "gpu-requests/018-carrying-prepare.json").read_text())["body"]
urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
    marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
try:
    scene = run_operation(runtime, mapping["path"], mapping["body"])
    run_operation(runtime, "/carrying/prepare", payload)
    binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm="A")
    with runtime.carrying_scope(binding, payload["gripper_joint_positions_rad"]):
        backend = next(iter(runtime._compound_backends.values()))
        planner, checker = backend.motion_planner, backend._motion_clearance_checker
        positions = torch.as_tensor(np.deg2rad(payload["start_joints_deg"])[None, None],
                                    **planner.device_cfg.as_torch_dict())
        state = checker.get_kinematics(positions)
        spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
        world = checker.get_collision_distance(state).detach().cpu().numpy().reshape(-1)
        config = checker.kinematics.kinematics_config
        robot = yaml.safe_load(runtime._robot_config_path.read_bytes())["robot_cfg"]["kinematics"]
        labels = {}
        for link in robot["collision_link_names"]:
            try:
                indices = config.get_sphere_index_from_link_name(link).detach().cpu().numpy().reshape(-1)
            except KeyError:
                continue
            for index in indices:
                labels[int(index)] = link
        result = {"physical_motion": False, "oracle_used": False,
                  "positive": [{"index": int(index), "link": labels.get(int(index)),
                                "sphere_world": spheres[index].tolist(), "cost": float(world[index])}
                               for index in np.flatnonzero(world > 0)],
                  "sphere_count": len(spheres), "scene_token": scene["scene_token"]}
        print("RESULT " + json.dumps(result), flush=True)
finally:
    runtime.clear_carrying()
