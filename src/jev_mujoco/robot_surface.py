"""Own visual-surface evidence from measured joints and calibrated RGBD rays.

The isolated model contains only authored robot visual geoms in its ray group.
No live simulation state, environment object geometry, collision spheres or
rendered segmentation identifiers enter this boundary.
"""

import xml.etree.ElementTree as ET
from copy import copy
from functools import lru_cache

import mujoco
import numpy as np

from .kinematics import robot_model
from .model_builder import ARM_JOINTS, ASSET_ROOT, GRIP_DRIVERS

SURFACE_TOLERANCE_M = .003
PIXEL_OFFSETS_RC = ((0., 0.), (-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5))
RAY_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
CONTACT_LINKS = {f"grip{side}_Link3_{finger}" for side in ("L", "R") for finger in ("L", "R")}


@lru_cache(maxsize=1)
def _own_visual_model():
    root = ET.parse(ASSET_ROOT / "marvin_urdf/marvin_with_grippers.urdf").getroot()
    visual_links = {link.get("name") for link in root.findall("link") if link.find("visual") is not None}
    model = copy(robot_model())
    visual = np.array([model.body(int(body)).name in visual_links and model.geom_group[index] != 3
                       for index, body in enumerate(model.geom_bodyid)])
    # The existing isolated FK model also has default table/ground geoms. They
    # and every collision duplicate are explicitly excluded, including static geoms.
    model.geom_group[:] = 5
    model.geom_group[visual] = 0
    mimics = []
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics.append((joint.get("name"), mimic.get("joint"),
                           float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))))
    return model, np.flatnonzero(visual), mimics


def _measured_joint_values(arms):
    values = {}
    for arm, names in ARM_JOINTS.items():
        feedback = arms.get(arm, {})
        joints = np.asarray(feedback.get("joints_rad", []), dtype=float)
        driver = feedback.get("gripper_position_rad")
        if joints.shape != (7,) or driver is None or not np.isfinite(joints).all():
            return None
        if not np.isfinite(driver):
            return None
        values.update(zip(names, joints, strict=True))
        values[GRIP_DRIVERS[arm]] = float(driver)
    return values


def first_surface_matches(model, data, view, pixels_rc, *, visual_indices=None):
    """Match unchanged actual points to first mesh hits within their pixel footprint."""
    pixels = np.asarray(pixels_rc, dtype=int).reshape(-1, 2)
    rows, columns = pixels.T
    depth = np.asarray(view["depth_m"])[rows, columns]
    intrinsics = view["intrinsics"]
    transform = np.asarray(view["T_world_camera"], dtype=float)
    origin = transform[:3, 3]
    local = np.column_stack(((columns - intrinsics["cx"]) * depth / intrinsics["fx"],
                             (rows - intrinsics["cy"]) * depth / intrinsics["fy"], depth))
    measured = local @ transform[:3, :3].T + origin
    candidate = np.isfinite(depth) & (depth > 0) & np.asarray(view["valid_depth"])[rows, columns]
    if visual_indices is not None:
        near = np.zeros(len(pixels), dtype=bool)
        for index in visual_indices:
            # An enclosing visual bound is only a broad phase. The decision
            # below always requires a first triangle hit, never volume inclusion.
            radius = model.geom_rbound[index] + SURFACE_TOLERANCE_M
            near |= np.linalg.norm(measured - data.geom_xpos[index], axis=1) <= radius
        candidate &= near
    best = np.full(len(pixels), np.inf)
    links = np.full(len(pixels), -1, dtype=int)
    offsets = np.zeros((len(pixels), 2))
    contact = np.zeros(len(pixels), dtype=bool)
    contact_geometry = np.array([model.body(int(body)).name in CONTACT_LINKS for body in model.geom_bodyid])
    selected = np.flatnonzero(candidate)
    for row_offset, column_offset in PIXEL_OFFSETS_RC:
        if not len(selected):
            break
        directions = np.column_stack(((columns[selected] + column_offset - intrinsics["cx"]) / intrinsics["fx"],
                                      (rows[selected] + row_offset - intrinsics["cy"]) / intrinsics["fy"],
                                      np.ones(len(selected))))
        directions = directions @ transform[:3, :3].T
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        geometry_ids = np.full(len(selected), -1, dtype=np.int32)
        distances = np.full(len(selected), -1., dtype=float)
        mujoco.mj_multiRay(model, data, origin, directions.reshape(-1), RAY_GROUP, True, -1,
                          geometry_ids, distances, None, len(selected), np.inf)
        hits = origin + directions * distances[:, None]
        error = np.linalg.norm(hits - measured[selected], axis=1)
        valid_hit = (geometry_ids >= 0) & (distances >= 0)
        own_contact = np.zeros(len(selected), dtype=bool)
        own_contact[valid_hit] = contact_geometry[geometry_ids[valid_hit]]
        contact[selected] |= valid_hit & (error <= SURFACE_TOLERANCE_M) & own_contact
        update = valid_hit & (error < best[selected])
        indices = selected[update]
        best[indices] = error[update]
        links[indices] = geometry_ids[update]
        offsets[indices] = [row_offset, column_offset]
    return {"matched": best <= SURFACE_TOLERANCE_M, "distance_m": best,
            "unresolved_near_robot": candidate & (best > SURFACE_TOLERANCE_M),
            "geom_ids": links, "pixel_offsets_rc": offsets, "points_world_m": measured,
            "contact_surface_ambiguous": contact}


def own_robot_surface_pixels(capture, masks):
    """Return candidate ownership only; callers preserve independent object evidence."""
    result = {"status": "measured", "masks": {}, "matches": {}, "unresolved_near_robot_masks": {},
              "source": "own_urdf_visual_first_return_and_measured_joint_feedback",
              "tolerance_m": SURFACE_TOLERANCE_M, "pixel_footprint_offsets_rc": PIXEL_OFFSETS_RC}
    values = _measured_joint_values(capture["arms"])
    if values is None:
        # Missing own feedback cannot justify removing any measured return.
        result["status"] = "robot_feedback_unavailable"
        return result
    model, visual_indices, mimics = _own_visual_model()
    data = mujoco.MjData(model)
    for name, driver, multiplier, offset in mimics:
        values[name] = values[driver] * multiplier + offset
    for name, value in values.items():
        data.qpos[int(model.joint(name).qposadr[0])] = value
    mujoco.mj_forward(model, data)
    for camera, mask in masks.items():
        view = capture["views"][camera]
        valid = mask & view["valid_depth"] & np.isfinite(view["depth_m"]) & (view["depth_m"] > 0)
        pixels = np.column_stack(np.nonzero(valid))
        evidence = first_surface_matches(model, data, view, pixels, visual_indices=visual_indices)
        accepted = np.flatnonzero(evidence["matched"])
        result["masks"][camera] = np.zeros(mask.shape, dtype=bool)
        result["masks"][camera][tuple(pixels[accepted].T)] = True
        unresolved = pixels[evidence['unresolved_near_robot']]
        result['unresolved_near_robot_masks'][camera] = np.zeros(mask.shape, bool)
        result['unresolved_near_robot_masks'][camera][tuple(unresolved.T)] = True
        result["matches"][camera] = [{"pixel_rc": pixels[index].tolist(),
            "distance_m": float(evidence["distance_m"][index]),
            "robot_link": model.body(int(model.geom_bodyid[evidence["geom_ids"][index]])).name,
            "pixel_offset_rc": evidence["pixel_offsets_rc"][index].tolist(),
            "contact_surface_ambiguous": bool(evidence["contact_surface_ambiguous"][index])}
            for index in accepted]
    return result
