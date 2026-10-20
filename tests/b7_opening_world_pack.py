"""CPU-only measured opening sphere pack for one bounded native world query."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / "data/visible-cloud-work/b7_opening_cpu.py"
namespace = {"__file__": str(helper), "__name__": "opening_cpu_definitions"}
# Read the frozen owner's pure definitions; do not execute its report-writing replay.
source = helper.read_text(encoding="utf-8").split("start, end = payload[")[0]
exec(compile(source, str(helper), "exec"), namespace)  # noqa: S102 -- trusted frozen project diagnostic definitions
payload, robot = namespace["payload"], namespace["robot"]
driver_values = np.asarray(json.loads((ROOT / "data/visible-cloud-work/b7-opening-cpu.json").read_text())[
    "driver_samples_rad_in_checker_order"])
groups = [{"kind": "measured_start", "driver_rad": payload["gripper_start_position_rad"]}]
configs = [robot]
for driver in driver_values:
    groups.append({"kind": "actual_driver", "driver_rad": float(driver)})
    configs.append(namespace["configure_gripper_feedback"](robot,
        payload["gripper_joint_positions_rad"] | {"B": float(driver)}, namespace["urdf"]))
for lower, upper in pairwise(driver_values):
    groups.append({"kind": "actual_subinterval", "interval_rad": [float(lower), float(upper)], "sample_count": 3})
    configs.append(namespace["interval_envelope"](robot, lower, upper, sample_count=3))
groups.append({"kind": "old_full_authored_sweep_baseline"})
configs.append(namespace["configure_gripper_sweep_self"](
    namespace["create_gripper_closing_envelope"](robot, "B", namespace["urdf"]), "B"))
values, labels = [], []
for config in configs:
    kin = config["robot_cfg"]["kinematics"]
    transforms = namespace["audit"].forward_kinematics(namespace["document"], kin, namespace["joints"])
    transforms[namespace["carried_link"]] = transforms["TCP_Link_R"]
    spheres, current_labels = [], []
    for link in kin["collision_link_names"]:
        block = namespace["audit"].sphere_modes(kin, link)["world_buffered"]
        pose = transforms[link]
        block[:, :3] = block[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        spheres.extend(block.tolist())
        current_labels.extend({"link": link, "link_sphere_index": i} for i in range(len(block)))
    if labels:
        assert current_labels == labels
    labels = current_labels
    values.append(spheres)
output = ROOT / "data/visible-cloud-work/b7-opening-world-pack.npz"
metadata = {"groups": groups, "labels": labels, "oracle_used": False, "physical_motion": False,
    "original_radii_buffers_preserved": True, "all_original_robot_and_attachment_spheres_retained": True,
    "interval_rule": "frozen CPU helper: endpoint/midpoint plus authored Lipschitz half-step bound"}
np.savez_compressed(output, spheres=np.asarray(values, dtype=np.float64), metadata=np.array(json.dumps(metadata)))
print(json.dumps({"output": str(output), "shape": list(np.asarray(values).shape), "group_count": len(groups)}))
