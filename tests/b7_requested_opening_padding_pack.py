"""Dense CPU witness for actual driver sample center-speed world padding."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / "data/visible-cloud-work/b7_opening_cpu.py"
namespace = {"__file__": str(helper), "__name__": "opening_cpu_definitions"}
source = helper.read_text(encoding="utf-8").split("start, end = payload[")[0]
exec(compile(source, str(helper), "exec"), namespace)  # noqa: S102 -- trusted frozen own-robot diagnostic definitions
old = np.load(ROOT / "data/visible-cloud-work/b7-opening-world-pack.npz", allow_pickle=False)
original = old["spheres"]
metadata = json.loads(str(old["metadata"]))
samples = np.array([group["driver_rad"] for group in metadata["groups"] if group["kind"] == "actual_driver"])
max_step = float(np.diff(samples).max())
dense = np.unique(np.concatenate([np.linspace(a, b, 65) for a, b in pairwise(samples)]))
nearest_angle = np.abs(dense[:, None] - samples[None]).argmin(axis=1)
kin = namespace["robot"]["robot_cfg"]["kinematics"]
parents = {joint.find("child").get("link"): joint for joint in namespace["document"].findall("joint")}
padding = np.zeros(len(metadata["labels"]))
proof = []
for link, sphere_specs in kin["collision_spheres"].items():
    if not link.startswith("gripR_") or link == "gripR_Base":
        continue
    chain, ancestor = [], link
    while ancestor != "gripR_Base":
        joint = parents[ancestor]
        chain.append(joint)
        ancestor = joint.find("parent").get("link")
    chain.reverse()
    origins = [namespace["_origin"](joint) for joint in chain]
    motions = [namespace["_joint_motion"](joint, "gripR_finger_joint") for joint in chain]
    poses = []
    for angle in np.concatenate((samples, dense)):
        pose = np.eye(4)
        for origin, (axis, multiplier, offset) in zip(origins, motions, strict=True):
            rotation = np.eye(4)
            rotation[:3, :3] = namespace["_rotation"](axis, multiplier * angle + offset)
            pose = pose @ origin @ rotation
        poses.append(pose)
    poses = np.asarray(poses)
    for local_index, sphere in enumerate(sphere_specs):
        center = np.asarray(sphere["center"])
        length, speed = float(np.linalg.norm(center)), 0.
        for i in range(len(chain) - 1, -1, -1):
            speed += abs(motions[i][1]) * length
            length += float(np.linalg.norm(origins[i][:3, 3]))
        bound = speed * max_step / 2
        positions = poses[:, :3, :3] @ center + poses[:, :3, 3]
        displacements = np.linalg.norm(positions[len(samples):] - positions[:len(samples)][nearest_angle], axis=1)
        assert float(displacements.max()) <= bound + 1e-12
        index = next(i for i, label in enumerate(metadata["labels"])
            if label == {"link": link, "link_sphere_index": local_index})
        padding[index] = bound
        proof.append({"cpu_index": index, "link": link, "link_sphere_index": local_index,
            "center_speed_bound_m_per_rad": speed, "additional_radius_m": bound,
            "dense_max_nearest_sample_center_displacement_m": float(displacements.max())})
candidate = original[1:27].copy()
candidate[..., 3] += padding
groups = [metadata["groups"][0], *[{**group, "kind": "actual_driver_continuous_padding"}
    for group in metadata["groups"][1:27]], metadata["groups"][-1]]
output = ROOT / "data/visible-cloud-work/b7-requested-opening-padding-pack.npz"
metadata.update(groups=groups, padding_rule="own-URDF center speed L times max actual sample step/2",
    max_actual_step_rad=max_step, dense_driver_pose_count=len(dense), dense_center_coverage_passed=True,
    original_radii_preserved_before_continuous_padding=True, padding_proof=proof)
np.savez_compressed(output, spheres=np.concatenate((original[:1], candidate, original[-1:])),
    metadata=np.array(json.dumps(metadata)))
report = {"gpu_used": False, "oracle_used": False, "product_mutated": False,
    "shape": [28, len(padding), 4], "max_actual_step_rad": max_step, "dense_pose_count": len(dense),
    "maximum_additional_radius_m": float(padding.max()), "all_dense_centers_covered": True, "padding_proof": proof}
(ROOT / "data/visible-cloud-work/b7-requested-opening-padding-cpu.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: value for k, value in report.items() if k != "padding_proof"}))
