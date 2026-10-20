"""Convert shared Marvin workcell geometry into cuRobo world primitives."""

from typing import Any, Mapping
import math

from .scene_config import static_collision_boxes
from .volumetric_mapper import (
    DEFAULT_ESDF_VOXEL_SIZE_M,
    DEFAULT_EXTENT_METERS_XYZ,
)


# ========================================
# MODIFICATION START: SDK_CUROBO canonical collision scene adapter
# ========================================
def build_curobo_scene_model(
    scene: str,
    *,
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the static or synchronized workcell as cuRobo cuboids."""

    if snapshot is not None:
        configuration = snapshot.get("configuration")
        planning_pose = snapshot.get("planning_object_pose_wxyz")
        if not isinstance(configuration, Mapping):
            raise ValueError("scene snapshot has no configuration")
        table = configuration.get("table")
        obstacle = configuration.get("object")
        if not isinstance(table, Mapping) or not isinstance(obstacle, Mapping):
            raise ValueError("scene snapshot configuration is incomplete")
        return {
            "cuboid": {
                "scene_tabletop": {
                    "dims": [float(value) for value in table["size_m"]],
                    "pose": [float(value) for value in table["pose_wxyz"]],
                },
                "scene_obstacle": {
                    "dims": [float(value) for value in obstacle["size_m"]],
                    "pose": [float(value) for value in planning_pose],
                },
            }
        }
    boxes = static_collision_boxes(scene)
    if not boxes:
        return None
    return {
        "cuboid": {
            name: {
                "dims": [2.0 * float(value) for value in half_extents],
                "pose": [
                    *[float(value) for value in center],
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
            for name, center, half_extents in boxes
        }
    }


def collision_cache_for_scene(
    scene_model: Mapping[str, Any] | None,
    *additional_scene_models: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Size each geometry type for the worlds that may be loaded or restored."""

    cache: dict[str, Any] = {
        # update_world() requires each geometry type to exist at construction.
        # Keep the five-box built-in workcell, two-box manual world and empty
        # world in the same small allocation. Exact 5 -> 2 -> 1 resizing would
        # destroy every warmed planner on an ordinary scene toggle. Larger
        # semantic worlds grow this allocation; the runtime retains its capacity.
        "mesh": 1,
        "cuboid": 5,
        "voxel": {
            # Background plus three tabletop objects, including at startup.
            # Loading these layers must not destroy already-warmed planners.
            "layers": 4,
            # Mapper rounds extents up; VoxelGrid cache allocation rounds to
            # nearest. Reserve whole cells so the last background slice fits.
            "dims": [math.ceil(d / DEFAULT_ESDF_VOXEL_SIZE_M)
                     * DEFAULT_ESDF_VOXEL_SIZE_M for d in DEFAULT_EXTENT_METERS_XYZ],
            "voxel_size": DEFAULT_ESDF_VOXEL_SIZE_M,
        }
    }
    for model in (scene_model, *additional_scene_models):
        if model is None:
            continue
        for kind in ("mesh", "cuboid"):
            geometry = model.get(kind, {})
            if not isinstance(geometry, Mapping):
                raise ValueError(f"scene_model[{kind!r}] must be a mapping")
            cache[kind] = max(cache[kind], len(geometry))
        voxels = model.get("voxel", {})
        if not isinstance(voxels, Mapping):
            raise ValueError("scene_model['voxel'] must be a mapping")
        if voxels:
            # Metadata lists objects only; the mapper also loads one background.
            # so a one-object count change need not invalidate CUDA graphs.
            layers = 1 << len(voxels).bit_length()
            cache["voxel"]["layers"] = max(cache["voxel"]["layers"], layers)
            required = max(math.prod(math.ceil(float(d) / float(v["voxel_size"]) - 1e-5)
                                     for d in v["dims"]) for v in voxels.values())
            capacity = math.prod(math.ceil(d / cache["voxel"]["voxel_size"])
                                 for d in cache["voxel"]["dims"])
            if required > capacity:
                side = math.ceil(required ** (1 / 3)) + 1
                cache["voxel"]["dims"] = [side * cache["voxel"]["voxel_size"]] * 3
    return cache
# ========================================
# MODIFICATION END: SDK_CUROBO canonical collision scene adapter
# ========================================
