"""Offline validation of the user's exact observation pose and its FK mirror.

Simulator geometry/segmentation is used exclusively in this independent QA
script. No optimizer may alter the requested left joints. Runtime still must
plan and validate motion from measured RGBD obstacles.
"""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
from calibrate_observation_poses import clearance, set_joints, sphere_config
from PIL import Image

from luna_mujoco.cameras import CAMERA_NAMES, RgbdRenderer, camera_pose, clone_render_model, load_calibration
from luna_mujoco.demo_scene import demo_scene, load_demo_config
from luna_mujoco.model_builder import ARM_JOINTS, ModelBuilder

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/demo-observation-poses/current"
POSE_PATH = ROOT / "resources/demo_observation_poses.json"


def source_fingerprints():
    paths = ["assets/marvin_urdf/marvin_with_grippers.urdf", "resources/cameras/calibration.json",
             "resources/curobo/marvin_gui_dual.yml", "resources/demo-config.json",
             "src/luna_mujoco/model_builder.py", "src/luna_mujoco/cameras.py", "src/luna_mujoco/demo_scene.py",
             "src/luna_mujoco/cad_assets.py", "src/luna_mujoco/geometry.py", "resources/cad/manifest.json", "scripts/build_cad_assets.py",
             "scripts/calibrate_demo_observation_poses.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def endpoint_clearance(model, data):
    """Check all articulated sphere envelopes against actual scene boxes/cylinders."""
    result = clearance(model, data)
    # The legacy helper's table metric assumes its old reference table. The
    # scene metric below measures every support in this compiled demo instead.
    result.pop("table")
    result["scene"] = [1., None]
    config = sphere_config()
    scene_geoms = [index for index in range(model.ngeom)
                   if (model.geom(index).name.startswith(("table", "support_table_", "object_geom_"))
                       and model.geom_contype[index])]
    for link, spheres in config["collision_spheres"].items():
        if link in ("Link_Base", "Link_Stand", "Base_L", "Base_R"):
            continue
        body = data.body(link)
        centers = np.array([sphere["center"] for sphere in spheres]) @ body.xmat.reshape(3, 3).T + body.xpos
        radii = np.array([sphere["radius"] for sphere in spheres])
        radii += config["collision_sphere_buffer"].get(link, 0.)
        for index in scene_geoms:
            local = (centers - data.geom_xpos[index]) @ data.geom_xmat[index].reshape(3, 3)
            size = model.geom_size[index]
            if model.geom_type[index] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH):
                # Each CAD convex piece uses a conservative local AABB for this
                # offline sphere audit. Physics uses its actual convex mesh.
                offset = np.abs(local) - size
            elif model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                offset = np.column_stack((np.linalg.norm(local[:, :2], axis=1) - size[0],
                                          np.abs(local[:, 2]) - size[1]))
            else:
                raise ValueError("unexpected offline scene collision geometry")
            signed = np.linalg.norm(np.maximum(offset, 0.), axis=1) + np.minimum(offset.max(axis=1), 0.)
            distance = float(np.min(signed - radii))
            if distance < result["scene"][0]:
                result["scene"] = [distance, [link, model.geom(index).name]]
    return result


def is_clear(distances):
    return distances["scene"][0] >= .01 and distances["peer"][0] >= .01 and distances["self"][0] > 0.


def user_observation_poses(model, data, config):
    """Find the unique sign mirror from all corresponding URDF link FK poses."""
    set_joints(model, data, {"A": [0.] * 7, "B": [0.] * 7})
    reflection = np.diag([1., -1., 1.])
    links = [f"Link{index}" for index in range(1, 8)] + ["TCP_Link"]
    local = {link: data.body(link + "_L").xmat.reshape(3, 3).T @ reflection
                   @ data.body(link + "_R").xmat.reshape(3, 3) for link in links}
    left = np.radians(config["observation_left_joints_deg"])
    set_joints(model, data, {"A": left, "B": [0.] * 7})
    expected = {link: (reflection @ data.body(link + "_L").xpos.copy(),
                       reflection @ data.body(link + "_L").xmat.reshape(3, 3).copy() @ local[link])
                for link in links}
    candidates = []
    for signs in product((-1, 1), repeat=7):
        right = left * np.asarray(signs)
        set_joints(model, data, {"A": left, "B": right})
        position_errors, rotation_errors = [], []
        for link in links:
            position, rotation = expected[link]
            position_errors.append(float(np.linalg.norm(data.body(link + "_R").xpos - position)))
            rotation_errors.append(float(np.linalg.norm(data.body(link + "_R").xmat.reshape(3, 3) - rotation)))
        score = sum(error ** 2 for error in position_errors + rotation_errors)
        candidates.append((score, signs, max(position_errors), max(rotation_errors)))
    candidates.sort()
    score, signs, position_error, rotation_error = candidates[0]
    if position_error > .0001 or rotation_error > .0001 or candidates[1][0] < 100 * score:
        raise ValueError("user_pose_has_no_unique_FK_mirror")
    poses = {"A": left.tolist(), "B": (left * np.asarray(signs)).tolist()}
    for arm, joints in poses.items():
        for name, value in zip(ARM_JOINTS[arm], joints, strict=True):
            lower, upper = model.joint(name).range
            if not lower <= value <= upper:
                raise ValueError("user_observation_joint_outside_limit:" + name)
    set_joints(model, data, poses)
    return poses, {"joint_signs_A_to_B": list(signs), "checked_sign_combinations": 128,
                   "links": links, "maximum_position_error_m": position_error,
                   "maximum_rotation_matrix_error": rotation_error,
                   "best_squared_error": score, "next_best_squared_error": candidates[1][0],
                   "scope": "robot_link_mirror_independent_of_scene_layout",
                   "evidence_origin": "computed_current_robot_fk",
                   "robot_urdf_sha256": source_fingerprints()["assets/marvin_urdf/marvin_with_grippers.urdf"]}


def source_bin_coverage(model, data, config=None):
    """Report full-bin clipping for diagnosis; it is not an acceptance gate."""
    config = load_demo_config() if config is None else config
    bin_item = config["source_bin"]
    half = np.asarray(bin_item["size"]) / 2
    local = np.asarray([[x, y, z] for z in (half[2], -half[2])
                        for x, y in ((-half[0], -half[1]), (half[0], -half[1]),
                                     (half[0], half[1]), (-half[0], half[1]))])
    body = data.body("object_" + bin_item["id"])
    world = local @ body.xmat.reshape(3, 3).T + body.xpos
    pose = camera_pose(model, data, "2")
    optical = (world - pose[:3, 3]) @ pose[:3, :3]
    intrinsics = load_calibration()["cameras"]["2"]["intrinsics"]
    if np.any(optical[:, 2] <= 0):
        raise ValueError("source_bin_opening_behind_camera")
    pixels = np.column_stack((intrinsics["fx"] * optical[:, 0] / optical[:, 2] + intrinsics["cx"],
                              intrinsics["fy"] * optical[:, 1] / optical[:, 2] + intrinsics["cy"]))
    edge_distances = np.column_stack((pixels[:, 0], pixels[:, 1],
                                      intrinsics["width"] - 1 - pixels[:, 0],
                                      intrinsics["height"] - 1 - pixels[:, 1]))
    minimum = float(edge_distances.min())
    return {
        "camera_id": "2", "opening_corners_px": pixels[:4].tolist(),
        "envelope_corners_px": pixels.tolist(), "minimum_edge_margin_px": minimum,
        "full_envelope_inside_image": minimum >= 0.,
        "acceptance_gate": False,
        "scope": "offline full-bin frustum diagnostic only; inspect recognizable bin and all objects in rendered RGBD",
    }


def validate(model, data, poses, output, config=None):
    config = load_demo_config() if config is None else config
    set_joints(model, data, poses)
    bin_coverage = source_bin_coverage(model, data, config)
    distances = endpoint_clearance(model, data)
    contacts = []
    source_contacts = []
    for contact in data.contact:
        first = model.geom_group[contact.geom1] == 3 and model.geom_bodyid[contact.geom1] != 0
        second = model.geom_group[contact.geom2] == 3 and model.geom_bodyid[contact.geom2] != 0
        names = [model.geom(contact.geom1).name, model.geom(contact.geom2).name]
        # Cup cylinder proxies also use hidden group 3; only robot-scene pairs count.
        robot = [not name.startswith(("object_", "support_", "table", "ground")) for name in names]
        if (first and robot[0] and not robot[1]) or (second and robot[1] and not robot[0]):
            contacts.append(names)
        if any(name.startswith("object_geom_pink_cylinder_") for name in names):
            source_contacts.append(names)
    renderer = RgbdRenderer(clone_render_model(model))
    evidence = {}
    output.mkdir(parents=True, exist_ok=True)
    try:
        capture = renderer.capture(data, 1, {arm: {"joints_rad": values} for arm, values in poses.items()})
        for camera_id, view in capture["views"].items():
            height, width = view["depth_m"].shape
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0
            with mujoco.Renderer(model, height=height, width=width) as segmentation_renderer:
                segmentation_renderer.enable_segmentation_rendering()
                segmentation_renderer.update_scene(data, camera=CAMERA_NAMES[camera_id], scene_option=options)
                labels = segmentation_renderer.render()
            per_object = {}
            target_geoms = {}
            for index in range(model.ngeom):
                body_name = model.body(int(model.geom_bodyid[index])).name
                if body_name.startswith("object_pink_cylinder_") and model.geom_group[index] != 3:
                    target_geoms.setdefault(body_name.removeprefix("object_"), []).append(index)
            for name, indices in target_geoms.items():
                mask = np.isin(labels[..., 0], indices) & view["valid_depth"]
                pixels = int(mask.sum())
                per_object[name] = pixels
            visible_count = sum(count >= 100 for count in per_object.values())
            bin_ids = [index for index in range(model.ngeom)
                       if model.body(int(model.geom_bodyid[index])).name == "object_blue_source_bin"
                       and model.geom_group[index] != 3]
            bin_mask = np.isin(labels[..., 0], bin_ids) & view["valid_depth"]
            rows, columns = np.nonzero(bin_mask)
            bin_bounds = None
            if len(rows):
                bin_bounds = [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]
            evidence[camera_id] = {
                "visible_source_count": visible_count, "source_pixels_by_object": per_object,
                "bin_visible_pixels": int(bin_mask.sum()), "bin_bounds_px": bin_bounds,
                "valid_depth_pixels": int(view["valid_depth"].sum()),
                "T_world_camera": view["T_world_camera"],
            }
            Image.fromarray(view["rgb"]).save(output / f"camera-{camera_id}.png")
            np.savez_compressed(output / f"camera-{camera_id}-rgbd.npz", rgb=view["rgb"],
                                depth_m=view["depth_m"], valid_depth=view["valid_depth"],
                                T_world_camera=view["T_world_camera"])
    finally:
        renderer.close()
    return {
        "mujoco_version": mujoco.__version__, "physical_left_arm": "A", "source_camera_id": "2",
        "physical_right_arm": "B", "right_camera_id": "1", "endpoint_clearance_m": distances,
        "mujoco_robot_scene_contacts": len(contacts), "robot_scene_contact_pairs": contacts,
        "mujoco_source_object_contacts": len(source_contacts), "source_object_contact_pairs": source_contacts,
        "source_positions_world_m": {
            name: data.body("object_" + name).xpos.tolist() for name in target_geoms
        },
        "rgbd_views": evidence,
        "source_bin_coverage": bin_coverage,
        "endpoint_collision_clear": is_clear(distances) and not contacts,
        "all_source_objects_visible": evidence["2"]["visible_source_count"] == len(target_geoms),
        "source_object_count": len(target_geoms),
        "scene_config": config,
        "environment_truth_in_runtime": False,
        "scope": "offline endpoint geometry and rendered RGBD only; runtime path must be collision planned",
        "collision_geometry": "buffered planner spheres against supports and conservative AABBs of individual CAD convex pieces",
        "minimum_clearance_m": {"scene": .01, "peer": .01, "self": 0.},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--object-count", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    builder = ModelBuilder()
    try:
        config = load_demo_config()
        count = config["object_count"] if args.object_count is None else args.object_count
        model = builder.build(demo_scene(count))
        data = mujoco.MjData(model)
        poses, mirror = user_observation_poses(model, data, config)
        output = args.output if count == config["object_count"] else args.output / f"count-{count}"
        evidence = validate(model, data, poses, output, config)
        evidence["mirror_fk"] = mirror
        count_validation = {}
        counts = (count,) if args.verify and args.object_count is not None else range(1, 5)
        for checked_count in counts:
            checked_output = args.output if checked_count == config["object_count"] else args.output / f"count-{checked_count}"
            if checked_count == count:
                checked = evidence
            else:
                checked_model = builder.build(demo_scene(checked_count, config))
                checked_data = mujoco.MjData(checked_model)
                checked = validate(checked_model, checked_data, poses, checked_output, config)
            evidence_path = checked_output / "evidence.json"
            evidence_path.write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")
            count_validation[str(checked_count)] = {
                "source_object_count": checked["source_object_count"],
                "source_pixels_by_object": checked["rgbd_views"]["2"]["source_pixels_by_object"],
                "all_source_objects_visible": checked["all_source_objects_visible"],
                "mujoco_robot_scene_contacts": checked["mujoco_robot_scene_contacts"],
                "mujoco_source_object_contacts": checked["mujoco_source_object_contacts"],
                "source_positions_world_m": checked["source_positions_world_m"],
                "endpoint_collision_clear": checked["endpoint_collision_clear"],
                "evidence_path": evidence_path.relative_to(ROOT).as_posix(),
                "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            }
        if args.verify:
            record = json.loads(POSE_PATH.read_text(encoding="utf-8"))
            for arm in poses:
                np.testing.assert_allclose(record["joints_rad"][arm], poses[arm], atol=1e-12)
            record["validation"] = evidence
            record["validation_by_object_count"] = count_validation
        else:
            record = {
                "schema_version": 1, "model_validated": evidence["endpoint_collision_clear"],
                "joints_rad": poses, "joints_deg": {arm: np.degrees(values).tolist() for arm, values in poses.items()},
                "validation": evidence, "arm_camera_mapping": load_demo_config()["arm_camera_mapping"],
                "validation_by_object_count": count_validation,
                "source_sha256": source_fingerprints(),
                "startup_joints_rad": poses,
                "provenance": "exact user left joints; unique sign mirror verified from all corresponding link FK",
            }
            POSE_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
    finally:
        builder.close()


if __name__ == "__main__":
    main()
