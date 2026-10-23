"""Self-contained Marvin URDF adapter using public MuJoCo 3.10 APIs.

The importer adaptations originate in tianji-simulation; see ASSET_PROVENANCE.md.
Only this module knows MuJoCo's model-building types and asset paths.
"""

import hashlib
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from .cad_assets import _mesh_path, cad_asset
from .geometry import scene_object_rotation

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"
ARM_JOINTS = {"A": [f"Joint{i}_L" for i in range(1, 8)], "B": [f"Joint{i}_R" for i in range(1, 8)]}
GRIP_DRIVERS = {"A": "gripL_finger_joint", "B": "gripR_finger_joint"}
TCP_BODIES = {"A": "TCP_Link_L", "B": "TCP_Link_R"}
# The demo models high-rigidity position servos, not hardware torque-limited
# arms. A frozen loaded closure saturated the authored 18 Nm wrist capacity.
# Keep finite simulator capacity and position-angle ctrl; grippers still use
# their actual URDF effort. No task code commands forces or overwrites qpos.
ARM_POSITION_STIFFNESS = 240000.
ARM_POSITION_DAMPING = 600.
ARM_POSITION_EFFORT_MULTIPLIER = 10.
ARM_COLORS = [0x4C72B0, 0xDD8452, 0x55A868, 0xC44E52, 0x8172B3, 0x937860, 0xDA8BC3]
COLORS = {
    "black": [0.035, 0.040, 0.050, 1.0], "red": [0.8, 0.12, 0.1, 1.0],
    "blue": [0.1, 0.35, 0.85, 1.0], "green": [0.15, 0.62, 0.3, 1.0],
    "yellow": [0.95, 0.72, 0.12, 1.0], "white": [0.9, 0.9, 0.92, 1.0],
    "gray": [0.4, 0.43, 0.48, 1.0], "orange": [0.95, 0.4, 0.1, 1.0],
    "pink": [0.94, 0.37, 0.61, 1.0],
}
DEFAULT_SCENE = {
    "table": {"position": [0.55, 0, 0.56], "size": [0.7, 1.0, 0.08], "color": "white"},
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
    table_color = table.get("color", "white")
    color_rgba(table_color)
    result = {"table": {
        "position": _vector(table.get("position"), "table position"),
        "size": _vector(table.get("size"), "table size", positive=True),
        "color": table_color,
    }, "objects": []}
    if "support_tables" in scene:
        supports = scene["support_tables"]
        if not isinstance(supports, list) or len(supports) > 4:
            raise ValueError("support_tables must be a list with at most four entries")
        result["support_tables"] = []
        support_ids = set()
        for support in supports:
            identifier = support.get("id")
            if (not isinstance(identifier, str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier)
                    or identifier in support_ids):
                raise ValueError("support table id must be unique letters, numbers, underscores or hyphens")
            support_ids.add(identifier)
            result["support_tables"].append({
                "id": identifier,
                "position": _vector(support.get("position"), "support table position"),
                "size": _vector(support.get("size"), "support table size", positive=True),
            })
    objects = scene.get("objects", DEFAULT_SCENE["objects"])
    if not isinstance(objects, list):
        raise ValueError("objects must be a list")  # noqa: TRY004 - JSON validation uses ValueError.
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
        if kind not in ("box", "cube", "sphere", "cylinder", "bin", "marker", "cad"):
            raise ValueError(f"unsupported object kind {kind!r}")
        asset = cad_asset(item.get("asset_id")) if kind == "cad" else None
        size = _vector(item.get("size", asset["size_m"] if asset else [.04]*3), "object size", positive=True)
        if asset and not np.allclose(size, asset["size_m"], rtol=0, atol=1e-9):
            previous = np.ptp(asset['source']['bounds_mm'], axis=0) * .001
            if asset['id'] == 'pink_wrapped' and asset.get('preprocessing') and np.allclose(size, previous, rtol=0, atol=1e-9):
                size = asset['size_m']
            else:
                raise ValueError("CAD size must equal the native asset size")
        if kind in ("cube", "sphere") and not np.allclose(size, size[0]):
            raise ValueError(f"{kind} size must have equal dimensions")
        cross_section = item.get("cross_section")
        if cross_section is not None and (kind != "cylinder" or cross_section != "ellipse"):
            raise ValueError("cross_section ellipse requires a cylinder")
        if kind == "cylinder" and cross_section is None and not np.isclose(size[0], size[1]):
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
        if asset:
            mass = float(item.get("mass_kg", asset["demo_mass_kg"]))
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("CAD mass_kg must be finite and positive")
            normalized.update(asset_id=asset["id"], size=asset["size_m"], mass_kg=mass)
        if 'fixed' in item:
            if type(item['fixed']) is not bool:
                raise ValueError('fixed must be a boolean')
            normalized['fixed'] = item['fixed']
        if 'pitch_deg' in item:
            pitch = float(item['pitch_deg'])
            if kind != 'cad' or not np.isfinite(pitch):
                raise ValueError('pitch_deg requires a CAD object and finite angle')
            normalized['pitch_deg'] = pitch
        if cross_section is not None:
            normalized["cross_section"] = cross_section
        if "visual_style" in item:
            if kind != "cylinder" or item["visual_style"] != "cup":
                raise ValueError("visual_style cup requires a cylinder")
            normalized["visual_style"] = "cup"
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


def _cup_mesh(radius, height, segments=48):
    """Open visual cup; physics deliberately keeps a solid cylinder envelope."""
    thickness = min(.003, radius / 4, height / 4)
    vertices = []
    for ring_radius, z in ((radius, -height / 2), (radius, height / 2),
                           (radius - thickness, height / 2),
                           (radius - thickness, -height / 2 + thickness)):
        for index in range(segments):
            angle = 2 * np.pi * index / segments
            vertices.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle), z])
    vertices.extend([[0, 0, -height / 2], [0, 0, -height / 2 + thickness]])
    faces = []
    for ring in range(3):
        for index in range(segments):
            following = (index + 1) % segments
            a, b = ring * segments + index, ring * segments + following
            c, d = b + segments, a + segments
            faces.extend([[a, b, c], [a, c, d]])
    for index in range(segments):
        following = (index + 1) % segments
        faces.append([4 * segments, following, index])
        faces.append([4 * segments + 1, 3 * segments + index, 3 * segments + following])
    return np.asarray(vertices).ravel().tolist(), np.asarray(faces).ravel().tolist()


def _elliptical_cylinder_mesh(size, segments=64):
    """Closed convex elliptical prism; full size is X/Y diameter and Z height."""
    half = np.asarray(size) / 2
    vertices = []
    for z in (-half[2], half[2]):
        for index in range(segments):
            angle = 2 * np.pi * index / segments
            vertices.append([half[0] * np.cos(angle), half[1] * np.sin(angle), z])
    vertices.extend([[0, 0, -half[2]], [0, 0, half[2]]])
    faces = []
    for index in range(segments):
        following = (index + 1) % segments
        faces.extend([[index, following, following + segments],
                      [index, following + segments, index + segments],
                      [2 * segments, following, index],
                      [2 * segments + 1, index + segments, following + segments]])
    return np.asarray(vertices).ravel().tolist(), np.asarray(faces).ravel().tolist()


class ModelBuilder:
    """Retain staged meshes for the lifetime of all compiled scene models."""

    def __init__(self):
        self._directory = tempfile.TemporaryDirectory(prefix="luna_mujoco_meshes_")
        self._mesh_directory = Path(self._directory.name)
        self._robot = self._load_robot()
        self._render_meshes = {}

    def close(self):
        self._directory.cleanup()

    def _load_robot(self):
        urdf_path = ASSET_ROOT / "marvin_urdf" / "marvin_with_grippers.urdf"
        root = ET.parse(urdf_path).getroot()
        efforts = {}
        gripper_efforts = {}
        mimics = []
        for joint in root.findall("joint"):
            name = joint.get("name")
            if name in ARM_JOINTS["A"] + ARM_JOINTS["B"]:
                efforts[name] = float(joint.find("limit").get("effort"))
            elif name in GRIP_DRIVERS.values():
                gripper_efforts[name] = float(joint.find("limit").get("effort"))
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

        # Native gravity compensation for authored robot bodies only. Scene
        # objects are added later and retain their physical weight. Position
        # servo ctrl remains the requested joint angle.
        for body in spec.bodies:
            if body.name != "world":
                body.gravcomp = 1.
        arm_names = set(efforts)
        follower_names = {name for name, *_ in mimics}
        for joint in spec.joints:
            if joint.name in arm_names:
                joint.damping = np.array([[20.0], [0.0], [0.0]])
                joint.armature = 0.5
                joint.actgravcomp = True
                joint.actfrclimited = True
                capacity = efforts[joint.name] * ARM_POSITION_EFFORT_MULTIPLIER
                joint.actfrcrange = [-capacity, capacity]
            elif joint.name in GRIP_DRIVERS.values() or joint.name in follower_names:
                joint.damping = np.array([[0.2], [0.0], [0.0]])
                joint.armature = 0.01
                if joint.name in GRIP_DRIVERS.values():
                    joint.actgravcomp = True
                    joint.actfrclimited = True
                    effort = gripper_efforts[joint.name]
                    joint.actfrcrange = [-effort, effort]
        for name, effort in efforts.items():
            actuator = spec.add_actuator()
            actuator.name = "act_" + name
            actuator.target = name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            # MuJoCo position servo: ctrl is a joint angle in radians.
            actuator.gainprm[0] = ARM_POSITION_STIFFNESS
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.biasprm[1] = -ARM_POSITION_STIFFNESS
            actuator.biasprm[2] = -ARM_POSITION_DAMPING
            actuator.forcelimited = True
            capacity = effort * ARM_POSITION_EFFORT_MULTIPLIER
            actuator.forcerange = [-capacity, capacity]
        for name in GRIP_DRIVERS.values():
            effort = gripper_efforts[name]
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
            actuator.forcerange = [-effort, effort]
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
        cad_records = {item["asset_id"]: cad_asset(item["asset_id"])
                       for item in scene["objects"] if item["kind"] == "cad"}
        if cad_records:
            # MuJoCo 3.10 native mesh manifolds chatter on the authored CAD
            # bottom facets. The public libccd path retains multiCCD and
            # settles the same geometry without changing contact/servo gains.
            spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
        for asset_id, record in cad_records.items():
            if record.get('simulation_geometry') == 'cad_bounds_box':
                continue
            for label, mesh in [("visual", record["visual"]),
                                *[(str(i), part) for i, part in enumerate(record["collision"])]]:
                spec.add_mesh(name=f"cad_{asset_id}_{label}", file=str(_mesh_path(mesh["path"])),
                              inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
        table = scene["table"]
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=table["position"], size=np.array(table["size"]) / 2,
                                rgba=color_rgba(table.get("color", "white")), friction=[1, 0.005, 0.0001])
        for support in scene.get("support_tables", []):
            spec.worldbody.add_geom(name="support_table_" + support["id"], type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=support["position"], size=np.array(support["size"]) / 2,
                                    rgba=[0.50, 0.53, 0.59, 1], friction=[1, 0.005, 0.0001])
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[3, 3, 0.1], rgba=[0.15, 0.18, 0.23, 1])
        for item in scene["objects"]:
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, scene_object_rotation(item).ravel())
            body = spec.worldbody.add_body(name="object_" + item["id"], pos=item["position"],
                                           quat=quaternion)
            if item["kind"] == "marker":
                # A visible paint-like region is fixed to the table and has no contacts.
                body.add_geom(name="object_geom_" + item["id"], type=mujoco.mjtGeom.mjGEOM_BOX,
                              size=np.array(item["size"]) / 2, rgba=color_rgba(item["color"]),
                              contype=0, conaffinity=0, mass=0)
                continue
            if not item.get('fixed', False):
                body.add_freejoint(name="free_" + item["id"])
            size = np.array(item["size"]) / 2
            kind = item["kind"]
            if kind == "cad":
                record = cad_records[item["asset_id"]]
                if record.get('simulation_geometry') == 'cad_bounds_box':
                    # The CAD is already centered on its local AABB. This is
                    # the same tight box in RGB and physics, with no padding.
                    body.add_geom(name='object_geom_' + item['id'], type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=np.asarray(record['size_m']) / 2, rgba=color_rgba(item['color']),
                        mass=item['mass_kg'], condim=4, friction=[1., .01, .001], priority=1,
                        solref=[.003, 1], solimp=[.95, .99, .001, .5, 2])
                    continue
                body.add_geom(name="object_visual_" + item["id"], type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=f"cad_{item['asset_id']}_visual", rgba=color_rgba(item["color"]),
                              mass=0, contype=0, conaffinity=0)
                volume = sum(part["volume_m3"] for part in record["collision"])
                for index, part in enumerate(record["collision"]):
                    body.add_geom(name=f"object_geom_{item['id']}__cad_{index}",
                                  type=mujoco.mjtGeom.mjGEOM_MESH,
                                  meshname=f"cad_{item['asset_id']}_{index}", group=3,
                                  rgba=color_rgba(item["color"]),
                                  mass=item["mass_kg"] * part["volume_m3"] / volume,
                                  condim=4, friction=[1., .01, .001], priority=1, solref=[.003, 1],
                                  solimp=[.95, .99, .001, .5, 2])
                continue
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
            if kind == "cylinder" and item.get("cross_section") == "ellipse":
                vertices, faces = _elliptical_cylinder_mesh(item["size"])
                mesh_name = "ellipse_" + item["id"]
                spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)
                body.add_geom(name="object_geom_" + item["id"], type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=mesh_name, rgba=color_rgba(item["color"]), mass=.08, condim=4,
                              friction=[1., .01, .001], priority=1, solref=[.003, 1],
                              solimp=[.95, .99, .001, .5, 2])
                continue
            if kind == "sphere":
                geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
            elif kind == "cylinder":
                geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
                size = [size[0], size[2], 0]
            collision = body.add_geom(name="object_geom_" + item["id"], type=geom_type, size=size,
                          rgba=color_rgba(item["color"]), mass=0.08, condim=4,
                          friction=[1.0, 0.01, 0.001], priority=1, solref=[0.003, 1],
                          solimp=[0.95, 0.99, 0.001, 0.5, 2])
            if item.get("visual_style") == "cup":
                collision.group = 3
                vertices, faces = _cup_mesh(item["size"][0] / 2, item["size"][2])
                mesh_name = "cup_visual_" + item["id"]
                spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)
                body.add_geom(name="object_visual_" + item["id"], type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=mesh_name, rgba=color_rgba(item["color"]), mass=0,
                              contype=0, conaffinity=0)
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
                if source["kind"] == "cad":
                    item.update(asset_id=source["asset_id"], object_size_m=source["size"],
                                local_position_m=model.geom_pos[index].tolist(),
                                local_quaternion=model.geom_quat[index].tolist())
            if kind == "mesh":
                mesh_id = int(model.geom_dataid[index])
                name = model.mesh(mesh_id).name
                if name not in self._render_meshes:
                    vertex_start = model.mesh_vertadr[mesh_id]
                    face_start = model.mesh_faceadr[mesh_id]
                    vertices = model.mesh_vert[vertex_start:vertex_start + model.mesh_vertnum[mesh_id]]
                    faces = model.mesh_face[face_start:face_start + model.mesh_facenum[mesh_id]]
                    payload = {"vertices": np.round(vertices.astype(float), 7).ravel().tolist(),
                               "faces": faces.ravel().tolist()}
                    encoded = json.dumps(payload, separators=(",", ":"))
                    # Mesh indices can change after mock replacement or editing.
                    # Content-addressed URLs keep previous experiment revisions immutable.
                    filename = f"mesh_{hashlib.sha256(encoded.encode()).hexdigest()}.json"
                    (render_directory / filename).write_text(encoded, encoding="utf-8")
                    self._render_meshes[name] = "/assets/render_meshes/" + filename
                item["mesh_url"] = self._render_meshes[name]
            geometries.append(item)
        return deepcopy(geometries)
