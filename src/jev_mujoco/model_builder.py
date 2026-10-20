"""Self-contained Marvin URDF adapter using public MuJoCo 3.10 APIs.

The importer adaptations originate in tianji-simulation; see ASSET_PROVENANCE.md.
Only this module knows MuJoCo's model-building types and asset paths.
"""

import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"
ARM_JOINTS = {"A": [f"Joint{i}_L" for i in range(1, 8)], "B": [f"Joint{i}_R" for i in range(1, 8)]}
GRIP_DRIVERS = {"A": "gripL_finger_joint", "B": "gripR_finger_joint"}
TCP_BODIES = {"A": "TCP_Link_L", "B": "TCP_Link_R"}
ARM_COLORS = [0x4C72B0, 0xDD8452, 0x55A868, 0xC44E52, 0x8172B3, 0x937860, 0xDA8BC3]
COLORS = {
    "black": [0.035, 0.040, 0.050, 1.0], "red": [0.8, 0.12, 0.1, 1.0],
    "blue": [0.1, 0.35, 0.85, 1.0], "green": [0.15, 0.62, 0.3, 1.0],
    "yellow": [0.95, 0.72, 0.12, 1.0], "white": [0.9, 0.9, 0.92, 1.0],
    "gray": [0.4, 0.43, 0.48, 1.0], "orange": [0.95, 0.4, 0.1, 1.0],
}
DEFAULT_SCENE = {
    "table": {"position": [0.55, 0, 0.56], "size": [0.7, 1.0, 0.08]},
    "objects": [
        {"id": "black_cube", "kind": "cube", "color": "black", "position": [0.43, 0.15, 0.72],
         "size": [0.04, 0.04, 0.04], "yaw_deg": 0},
        {"id": "red_box", "kind": "box", "color": "red", "position": [0.48, -0.16, 0.72],
         "size": [0.05, 0.04, 0.06], "yaw_deg": 15},
        {"id": "green_cylinder", "kind": "cylinder", "color": "green", "position": [0.65, 0.02, 0.72],
         "size": [0.045, 0.045, 0.07], "yaw_deg": 0},
        {"id": "blue_bin", "kind": "bin", "color": "blue", "position": [0.73, 0.27, 0.70],
         "size": [0.22, 0.24, 0.10], "yaw_deg": 0, "wall_thickness_m": 0.01, "mass_kg": 0.4},
        {"id": "table_transfer", "kind": "marker", "color": "yellow", "position": [0.38, 0., 0.6],
         "size": [0.20, 0.20, 0.001], "yaw_deg": 0},
    ],
}


def _vector(value, label, positive=False):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite numbers")
    if positive and np.any(vector <= 0):
        raise ValueError(f"{label} must be positive")
    return vector.tolist()


def color_rgba(color):
    if color in COLORS:
        return COLORS[color]
    if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)] + [1.0]
    raise ValueError(f"unsupported color {color!r}; use a color name or #RRGGBB")


def normalize_scene(scene):
    """Validate before invalidating the current generation or rebuilding physics."""
    if not isinstance(scene, dict):
        raise ValueError("scene must be an object")  # noqa: TRY004 - JSON validation uses ValueError.
    table = scene.get("table", DEFAULT_SCENE["table"])
    result = {"table": {
        "position": _vector(table.get("position"), "table position"),
        "size": _vector(table.get("size"), "table size", positive=True),
    }, "objects": []}
    objects = scene.get("objects", DEFAULT_SCENE["objects"])
    if not isinstance(objects, list) or len(objects) > 30:
        raise ValueError("objects must be a list with at most 30 entries")
    identifiers = set()
    marker_colors = set()
    for item in objects:
        identifier = item.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier):
            raise ValueError("object id must be 1–64 letters, numbers, underscores or hyphens")
        if identifier in identifiers:
            raise ValueError(f"duplicate object id {identifier!r}")
        identifiers.add(identifier)
        kind = item.get("kind", "box")
        if kind not in ("box", "cube", "sphere", "cylinder", "bin", "marker"):
            raise ValueError(f"unsupported object kind {kind!r}")
        size = _vector(item.get("size", [0.04] * 3), "object size", positive=True)
        if kind in ("cube", "sphere") and not np.allclose(size, size[0]):
            raise ValueError(f"{kind} size must have equal dimensions")
        if kind == "cylinder" and not np.isclose(size[0], size[1]):
            raise ValueError("cylinder size x/y must be equal diameters")
        color = item.get("color", "gray")
        color_rgba(color)
        yaw = float(item.get("yaw_deg", 0))
        if not np.isfinite(yaw):
            raise ValueError("yaw_deg must be finite")
        normalized = {
            "id": identifier, "kind": kind, "color": color,
            "position": _vector(item.get("position", [0.45, 0.15, 0.7]), "object position"),
            "size": size, "yaw_deg": yaw,
        }
        if kind == "marker":
            if color not in {"red", "orange", "yellow", "green", "blue"} or color in marker_colors:
                raise ValueError("marker color must be a unique saturated named color")
            marker_colors.add(color)
            angle = np.radians(yaw)
            extent = np.abs(np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])) @ (np.array(size[:2]) / 2)
            offset = np.abs(np.array(normalized["position"][:2]) - np.array(result["table"]["position"][:2]))
            if np.any(offset + extent > np.array(result["table"]["size"][:2]) / 2):
                raise ValueError("marker region must remain on the tabletop")
            normalized["position"][2] = result["table"]["position"][2] + result["table"]["size"][2] / 2 + .0001
            normalized["size"][2] = .0002
        if kind == "bin":
            thickness = float(item.get("wall_thickness_m", .01))
            mass = float(item.get("mass_kg", .4))
            if not np.isfinite(thickness) or thickness <= 0:
                raise ValueError("bin wall_thickness_m must be finite and positive")
            if size[0] <= 2 * thickness or size[1] <= 2 * thickness or size[2] <= thickness:
                raise ValueError("bin dimensions must leave a nonzero open cavity")
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("bin mass_kg must be finite and positive")
            normalized.update(wall_thickness_m=thickness, mass_kg=mass)
        result["objects"].append(normalized)
    return result


class ModelBuilder:
    """Retain staged meshes for the lifetime of all compiled scene models."""

    def __init__(self):
        self._directory = tempfile.TemporaryDirectory(prefix="jev_mujoco_meshes_")
        self._mesh_directory = Path(self._directory.name)
        self._robot = self._load_robot()
        self._render_meshes = {}

    def close(self):
        self._directory.cleanup()

    def _load_robot(self):
        urdf_path = ASSET_ROOT / "marvin_urdf" / "marvin_with_grippers.urdf"
        root = ET.parse(urdf_path).getroot()
        efforts = {}
        mimics = []
        for joint in root.findall("joint"):
            name = joint.get("name")
            if name in ARM_JOINTS["A"] + ARM_JOINTS["B"]:
                efforts[name] = float(joint.find("limit").get("effort"))
            mimic = joint.find("mimic")
            if mimic is not None:
                mimics.append((name, mimic.get("joint"), float(mimic.get("multiplier", 1)),
                               float(mimic.get("offset", 0))))

        staged = {}
        for link in root.findall("link"):
            for role in ("visual", "collision"):
                for element in link.findall(role):
                    mesh = element.find("geometry/mesh")
                    if mesh is None:
                        continue
                    source = (urdf_path.parent / mesh.get("filename")).resolve()
                    key = (str(source), role)
                    if key not in staged:
                        suffix = "__collision" if role == "collision" else ""
                        target = self._mesh_directory / f"{source.stem}{suffix}{source.suffix}"
                        if source.suffix.lower() == ".obj":
                            # The importer only takes the first OBJ object/group. Preserve all faces.
                            lines = source.read_bytes().splitlines(keepends=True)
                            target.write_bytes(b"".join(line for line in lines if not line.split()
                                                       or line.split(maxsplit=1)[0] not in
                                                       (b"o", b"g", b"usemtl", b"mtllib")))
                        else:
                            shutil.copyfile(source, target)
                        staged[key] = target.as_posix()
                    mesh.set("filename", staged[key])
                    if role == "visual":
                        name = link.get("name", "")
                        match = re.fullmatch(r"Link([1-7])_[LR]", name)
                        color = ARM_COLORS[int(match[1]) - 1] if match else None
                        if "WristCam" in source.name:
                            color = 0xB3E5FC if "Window" in source.name else 0x29B6F6
                        if color is not None:
                            material = element.find("material")
                            if material is None:
                                material = ET.SubElement(element, "material")
                            rgba = material.find("color")
                            if rgba is None:
                                rgba = ET.SubElement(material, "color")
                            rgba.set("rgba", " ".join(str((color >> shift & 255) / 255)
                                                     for shift in (16, 8, 0)) + " 1")
        settings = root.find("mujoco")
        if settings is None:
            settings = ET.SubElement(root, "mujoco")
        compiler = settings.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(settings, "compiler")
        compiler.set("discardvisual", "false")
        compiler.set("strippath", "false")
        compiler.set("fusestatic", "false")
        source_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        canonical = self._mesh_directory / "robot.xml"
        mujoco.mj_saveLastXML(str(canonical), source_model)
        spec = mujoco.MjSpec.from_file(str(canonical))

        arm_names = set(efforts)
        follower_names = {name for name, *_ in mimics}
        for joint in spec.joints:
            if joint.name in arm_names:
                joint.damping = np.array([[20.0], [0.0], [0.0]])
                joint.armature = 0.5
            elif joint.name in GRIP_DRIVERS.values() or joint.name in follower_names:
                joint.damping = np.array([[0.2], [0.0], [0.0]])
                joint.armature = 0.01
        for name, effort in efforts.items():
            actuator = spec.add_actuator()
            actuator.name = "act_" + name
            actuator.target = name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[0] = 1
            actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
            actuator.forcelimited = True
            actuator.forcerange = [-effort, effort]
        for name in GRIP_DRIVERS.values():
            actuator = spec.add_actuator()
            actuator.name = "act_" + name
            actuator.target = name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[0] = 200
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.biasprm[1] = -200
            actuator.biasprm[2] = -10
            actuator.forcelimited = True
            actuator.forcerange = [-200, 200]
        for follower, driver, multiplier, offset in mimics:
            equality = spec.add_equality()
            equality.type = mujoco.mjtEq.mjEQ_JOINT
            equality.name1 = follower
            equality.name2 = driver
            equality.data[0] = offset
            equality.data[1] = multiplier
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.group = 3
                geom.contype, geom.conaffinity = 1, 0
        # The stand/base's convex hull encloses the tabletop workspace. Its
        # authored visual remains, while articulated links retain all contacts.
        for body in spec.bodies:
            if body.name in ("Link_Base", "Link_Stand", "Base_L", "Base_R"):
                for geom in body.geoms:
                    geom.contype, geom.conaffinity = 0, 0
        spec.option.timestep = 0.001
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.option.noslip_iterations = 3
        return spec

    def build(self, scene):
        spec = self._robot.copy()
        # Neutral room illumination keeps the actual RGBD colors readable from
        # all three viewpoints, including surfaces facing away from the camera.
        spec.visual.headlight.ambient = [0.8, 0.8, 0.8]
        spec.visual.headlight.diffuse = [0.2, 0.2, 0.2]
        spec.visual.headlight.specular = [0, 0, 0]
        self._objects_by_id = {item["id"]: deepcopy(item) for item in scene["objects"]}
        table = scene["table"]
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=table["position"], size=np.array(table["size"]) / 2,
                                rgba=[0.50, 0.53, 0.59, 1], friction=[1, 0.005, 0.0001])
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[3, 3, 0.1], rgba=[0.15, 0.18, 0.23, 1])
        for item in scene["objects"]:
            yaw = np.radians(item["yaw_deg"]) / 2
            body = spec.worldbody.add_body(name="object_" + item["id"], pos=item["position"],
                                           quat=[np.cos(yaw), 0, 0, np.sin(yaw)])
            if item["kind"] == "marker":
                # A visible paint-like region is fixed to the table and has no contacts.
                body.add_geom(name="object_geom_" + item["id"], type=mujoco.mjtGeom.mjGEOM_BOX,
                              size=np.array(item["size"]) / 2, rgba=color_rgba(item["color"]),
                              contype=0, conaffinity=0, mass=0)
                continue
            body.add_freejoint(name="free_" + item["id"])
            size = np.array(item["size"]) / 2
            kind = item["kind"]
            if kind == "bin":
                width, depth, height = item["size"]
                thickness = item["wall_thickness_m"]
                wall_height = height - thickness
                components = [
                    ("bottom", [0, 0, (-height + thickness) / 2], [width, depth, thickness]),
                    ("wall_x_minus", [(-width + thickness) / 2, 0, thickness / 2], [thickness, depth, wall_height]),
                    ("wall_x_plus", [(width - thickness) / 2, 0, thickness / 2], [thickness, depth, wall_height]),
                    ("wall_y_minus", [0, (-depth + thickness) / 2, thickness / 2], [width - 2 * thickness, thickness, wall_height]),
                    ("wall_y_plus", [0, (depth - thickness) / 2, thickness / 2], [width - 2 * thickness, thickness, wall_height]),
                ]
                volume = sum(np.prod(dimensions) for _, _, dimensions in components)
                for component, position, dimensions in components:
                    body.add_geom(name="object_geom_" + item["id"] + "__" + component,
                                  type=mujoco.mjtGeom.mjGEOM_BOX, pos=position, size=np.array(dimensions) / 2,
                                  rgba=color_rgba(item["color"]), mass=item["mass_kg"] * np.prod(dimensions) / volume,
                                  condim=4, friction=[1.0, 0.01, 0.001], priority=1, solref=[0.003, 1],
                                  solimp=[0.95, 0.99, 0.001, 0.5, 2])
                continue
            geom_type = mujoco.mjtGeom.mjGEOM_BOX
            if kind == "sphere":
                geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
            elif kind == "cylinder":
                geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
                size = [size[0], size[2], 0]
            body.add_geom(name="object_geom_" + item["id"], type=geom_type, size=size,
                          rgba=color_rgba(item["color"]), mass=0.08, condim=4,
                          friction=[1.0, 0.01, 0.001], priority=1, solref=[0.003, 1],
                          solimp=[0.95, 0.99, 0.001, 0.5, 2])
        from .cameras import install_cameras

        install_cameras(spec)
        return spec.compile()

    def render_geometries(self, model, data):
        geometries = []
        render_directory = ASSET_ROOT / "render_meshes"
        render_directory.mkdir(exist_ok=True)
        for index in range(model.ngeom):
            if model.geom_group[index] == 3:
                continue
            kind = mujoco.mjtGeom(int(model.geom_type[index])).name.removeprefix("mjGEOM_").lower()
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.geom_xmat[index])
            item = {"id": str(index), "name": model.geom(index).name,
                    "type": kind, "position": data.geom_xpos[index].tolist(),
                    "quaternion": quaternion.tolist(), "size": model.geom_size[index].tolist(),
                    "rgba": model.geom_rgba[index].tolist()}
            body_name = model.body(int(model.geom_bodyid[index])).name
            if body_name.startswith("object_"):
                identifier = body_name.removeprefix("object_")
                source = self._objects_by_id[identifier]
                item.update(object_id=identifier, object_kind=source["kind"])
                if source["kind"] == "bin":
                    component = item["name"].removeprefix("object_geom_" + identifier + "__")
                    item.update(component=component, object_size_m=source["size"],
                                wall_thickness_m=source["wall_thickness_m"],
                                local_position_m=model.geom_pos[index].tolist())
            if kind == "mesh":
                mesh_id = int(model.geom_dataid[index])
                name = model.mesh(mesh_id).name
                if name not in self._render_meshes:
                    vertex_start = model.mesh_vertadr[mesh_id]
                    face_start = model.mesh_faceadr[mesh_id]
                    vertices = model.mesh_vert[vertex_start:vertex_start + model.mesh_vertnum[mesh_id]]
                    faces = model.mesh_face[face_start:face_start + model.mesh_facenum[mesh_id]]
                    filename = f"mesh_{mesh_id}.json"
                    payload = {"vertices": np.round(vertices.astype(float), 7).ravel().tolist(),
                               "faces": faces.ravel().tolist()}
                    (render_directory / filename).write_text(json.dumps(payload, separators=(",", ":")),
                                                              encoding="utf-8")
                    self._render_meshes[name] = "/assets/render_meshes/" + filename
                item["mesh_url"] = self._render_meshes[name]
            geometries.append(item)
        return deepcopy(geometries)
