"""Compare historical union with all actual post-grasp points; diagnostic only."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from jev_mujoco.carrying import CarryingBinding, carrying_geometry

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "data/visible-cloud-work"
recording = folder / "grasp-place-left-17"
probe = json.loads((folder / "a17-held-state-probe.json").read_text(encoding="utf-8"))
payload = json.loads((recording / "gpu-requests/021-tools-plan_place.json").read_text(encoding="utf-8"))["body"]
stage = json.loads((recording / "stage-1.json").read_text(encoding="utf-8"))
pre = json.loads((recording / "gpu-requests/008-perception-build.json").read_text(encoding="utf-8"))["body"]
post = json.loads((recording / "gpu-requests/018-perception-build.json").read_text(encoding="utf-8"))["body"]
observation = np.load(recording / f"perception-{post['observation_set_id']}.npz", allow_pickle=True)["observation"].item()
grasp = np.asarray(stage["holding"]["grasp_support_reference"]["T_world_tcp_at_grasp"])
post_tcp = np.asarray(observation["arms"]["A"]["tcp"])
old_points = np.asarray(pre["selected_surface_world_m"])
current_points = np.asarray(post["selected_surface_world_m"])
old_local = (old_points-grasp[:3, 3]) @ grasp[:3, :3]
current_local = (current_points-post_tcp[:3, 3]) @ post_tcp[:3, :3]
union = np.asarray(payload["carrying_binding"]["points_tool_m"])
old_match = cKDTree(old_local).query(union)[0] <= 1e-10
current_match = cKDTree(current_local).query(union)[0] <= 1e-10
assert np.all(old_match | current_match)
body = np.asarray(probe["world_spheres"][:267])
padding = np.asarray(probe["public_self_config"]["sphere_padding"][:267])
body[:, 3] += padding
# Recover the exact native tool transform by matching all tool/world union points.
world = np.asarray(probe["measured_world_points"])
centered_tool = union-union.mean(0)
centered_world = world-world.mean(0)
u, _, vt = np.linalg.svd(centered_tool.T @ centered_world)
rotation = u @ vt
translation = world.mean(0)-union.mean(0) @ rotation
residual = np.linalg.norm(union @ rotation+translation-world, axis=1)
assert residual.max() < 1e-6 and np.linalg.det(rotation) > .99999
scene = {"scene_token": payload["scene_token"], "observation_id": payload["carrying_binding"]["observation_id"],
         "generation": payload["carrying_binding"]["generation"], "target_id": payload["carrying_binding"]["instance_id"]}
report = {"gpu_used": False, "oracle_used": False, "native_frame_fit_max_error_m": float(residual.max()),
    "union_point_count": len(union), "pre_point_count": len(old_local), "post_point_count": len(current_local),
    "post_observation_id": post["observation_set_id"], "cases": {}}
for name, points in (("original_union", union), ("current_post_grasp_only", current_local)):
    candidate = deepcopy(payload["carrying_binding"])
    candidate["points_tool_m"] = points.tolist()
    candidate["measurement_provenance"]["point_count"] = len(points)
    binding = CarryingBinding.from_mapping(candidate, scene=scene, arm="A")
    geometry = carrying_geometry(binding)
    spheres = np.asarray(geometry["spheres"])
    cells = np.floor((points-points.min(0))/geometry["effective_occupied_voxel_size_m"]).astype(int)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    spheres[:, :3] = spheres[:, :3] @ rotation+translation
    gaps = np.linalg.norm(body[None, :, :3]-spheres[:, None, :3], axis=-1)-body[None, :, 3]-spheres[:, None, 3]
    pairs = []
    for index, body_index in np.argwhere(gaps < 0):
        link = probe["sphere_labels"].get(str(int(body_index)))
        if link in ("gripL_Link3_L", "gripL_Link3_R"):
            continue
        pair = {"attachment_index": int(index), "body_index": int(body_index), "body_link": link,
                "gap_m": float(gaps[index, body_index]), "contributing_point_count": int(np.sum(inverse == index))}
        if name == "original_union":
            selection = inverse == index
            pair.update(pre_contributor_count=int(np.sum(selection & old_match)),
                        post_contributor_count=int(np.sum(selection & current_match)))
        pairs.append(pair)
    measured_world = points @ rotation+translation
    measured_gaps = np.linalg.norm(measured_world[:, None, :3]-body[None, :, :3], axis=-1)-body[None, :, 3]
    entry = {"point_count": len(points), "geometry": {k:v for k,v in geometry.items() if k != "spheres"},
        "radius_min_m": float(spheres[:, 3].min()), "radius_max_m": float(spheres[:, 3].max()),
        "nonpad_negative_pairs": pairs,
        "closest_measured_point_to_Link4_L_gap_m": float(measured_gaps[:, 232].min()),
        "closest_measured_point_to_Link4_R_gap_m": float(measured_gaps[:, 236].min())}
    report["cases"][name] = entry
    print(name, json.dumps(entry), flush=True)
report["closest_original_point_provenance"] = {str(i): {"pre": bool(old_match[i]), "post": bool(current_match[i])}
                                                  for i in (8067, 675)}
report["limits"] = ["Current-only changes point provenance for diagnosis; no product or GPU binding was changed.",
                    "All actual post-grasp points and voxel/margin/uncertainty/budget are retained; no parameters were tuned."]
(folder / "a17-current-only-cpu-probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
