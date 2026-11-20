"""Pure RGBD pixel geometry; images and calibration arrive through Base."""
import numpy as np


def _immutable(array):
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def letterbox_view(view, width=640, height=480):
    """Pad registered RGBD and masks without scaling or inventing depth samples."""
    source_height, source_width = view["depth_m"].shape
    if source_width > width or source_height > height:
        raise ValueError("registered view is larger than the mapper resolution")
    top, left = (height - source_height) // 2, (width - source_width) // 2
    result = dict(view)
    result["intrinsics"] = {**view["intrinsics"], "cx": view["intrinsics"]["cx"] + left,
                            "cy": view["intrinsics"]["cy"] + top, "width": width, "height": height}
    for key in ("rgb", "depth_m", "valid_depth", "robot_mask"):
        if key not in view:
            continue
        source = np.asarray(view[key])
        shape = (height, width) + source.shape[2:]
        padded = np.zeros(shape, dtype=source.dtype)
        padded[top:top + source_height, left:left + source_width] = source
        result[key] = _immutable(padded)
    result["instance_masks"] = {}
    for instance_id, mask in view.get("instance_masks", {}).items():
        padded = np.zeros((height, width), dtype=bool)
        padded[top:top + source_height, left:left + source_width] = mask
        result["instance_masks"][instance_id] = _immutable(padded)
    return result


def deproject_view(view, mask=None):
    """Visible measured points in world metres; never generates hidden surfaces."""
    depth = np.asarray(view["depth_m"])
    valid = np.asarray(view["valid_depth"], dtype=bool) & np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    intrinsics = view["intrinsics"]
    points = np.column_stack(((columns - intrinsics["cx"]) * z / intrinsics["fx"],
                              (rows - intrinsics["cy"]) * z / intrinsics["fy"], z))
    transform = np.asarray(view["T_world_camera"], dtype=float)
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
