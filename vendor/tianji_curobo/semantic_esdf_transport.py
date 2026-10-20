"""Bounded JSON transport for immutable semantic or head-anchored background ESDF."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from copy import deepcopy
import zlib

import numpy as np


CAMERA_IDS = ("1", "2", "3")
MAX_IMAGE_PIXELS = 4_194_304
MAX_INSTANCE_COUNT = 200
MAX_MASK_ELEMENTS = 64 * 1024 * 1024
_ARRAY_FIELDS = {
    "depth_left_wire": (np.dtype("<f4"), 2),
    "valid_left_wire": (np.dtype("u1"), 2),
    "instance_masks_wire": (np.dtype("u1"), 3),
}


def _array(value: object, *, dtype: np.dtype, dimensions: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not a valid array") from error
    if result.ndim != dimensions or result.size < 1:
        raise ValueError(f"{name} has an invalid shape")
    if dimensions == 2 and result.size > MAX_IMAGE_PIXELS:
        raise ValueError(f"{name} exceeds the image size limit")
    if dimensions == 3 and (
        result.shape[0] > MAX_INSTANCE_COUNT
        or result.shape[1] * result.shape[2] > MAX_IMAGE_PIXELS
        or result.size > MAX_MASK_ELEMENTS
    ):
        raise ValueError(f"{name} exceeds the semantic mask limit")
    if dtype == np.dtype("<f4"):
        if not np.isfinite(result).all() or (result < 0.0).any():
            raise ValueError(f"{name} must contain finite non-negative depth")
    elif not np.logical_or(result == 0, result == 1).all():
        raise ValueError(f"{name} must be binary")
    return np.ascontiguousarray(result, dtype=dtype)


def _encode_array(value: object, *, dtype: np.dtype, dimensions: int, name: str) -> dict:
    array = _array(value, dtype=dtype, dimensions=dimensions, name=name)
    compressed = zlib.compress(array.tobytes(order="C"), level=1)
    return {
        "encoding": "zlib+base64",
        "dtype": dtype.str,
        "shape": list(array.shape),
        "data_base64": base64.b64encode(compressed).decode("ascii"),
    }


def _decode_array(value: object, *, dtype: np.dtype, dimensions: int, name: str) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != {
        "encoding",
        "dtype",
        "shape",
        "data_base64",
    }:
        raise ValueError(f"{name} envelope is invalid")
    if value.get("encoding") != "zlib+base64" or value.get("dtype") != dtype.str:
        raise ValueError(f"{name} encoding or dtype is invalid")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != dimensions
        or any(type(item) is not int or item < 1 for item in shape)
    ):
        raise ValueError(f"{name} shape is invalid")
    element_count = 1
    for item in shape:
        element_count *= item
    if (
        (dimensions == 2 and element_count > MAX_IMAGE_PIXELS)
        or (
            dimensions == 3
            and (
                shape[0] > MAX_INSTANCE_COUNT
                or shape[1] * shape[2] > MAX_IMAGE_PIXELS
                or element_count > MAX_MASK_ELEMENTS
            )
        )
    ):
        raise ValueError(f"{name} exceeds the decoded size limit")
    expected_bytes = element_count * dtype.itemsize
    try:
        compressed = base64.b64decode(value["data_base64"], validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_bytes + 1)
    except (TypeError, ValueError, zlib.error) as error:
        raise ValueError(f"{name} data is invalid") from error
    if (
        len(raw) != expected_bytes
        or decompressor.unconsumed_tail
        or decompressor.unused_data
        or not decompressor.eof
    ):
        raise ValueError(f"{name} decoded byte count is invalid")
    result = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return _array(result, dtype=dtype, dimensions=dimensions, name=name)


def _identity(payload: object) -> tuple[dict, Mapping]:
    required = {
        "observation_set_id",
        "sam3_generation",
        "confirmed_instance_id",
        "selection_generation",
        "views",
    }
    if not isinstance(payload, Mapping) or set(payload) not in (required, required | {"mapping_mode"}):
        raise ValueError("semantic ESDF request schema is invalid")
    mode = payload.get("mapping_mode", "semantic")
    if mode not in ("semantic", "background"):
        raise ValueError("unknown ESDF mapping mode")
    if mode == "background" and (
        payload.get("confirmed_instance_id") != "" or payload.get("sam3_generation") != 0
    ):
        raise ValueError("background mapping cannot exclude a selected instance")
    views = payload.get("views")
    allowed_views = (("3",), ("1", "3"), ("2", "3"), CAMERA_IDS) if mode == "background" else (CAMERA_IDS,)
    if not isinstance(views, Mapping) or tuple(views) not in allowed_views:
        raise ValueError("semantic ESDF requires ordered 1/2/3; background requires an ordered subset containing 3")
    identity = {
        key: deepcopy(payload[key])
        for key in (
            "observation_set_id",
            "sam3_generation",
            "confirmed_instance_id",
            "selection_generation",
        )
    }
    if "mapping_mode" in payload:
        identity["mapping_mode"] = mode
    return identity, views


def encode_semantic_esdf_request(payload: object) -> dict:
    """Encode only dense arrays; all remaining fields retain their public shape."""

    identity, views = _identity(payload)
    encoded_views = {}
    for camera_id in views:
        view = views[camera_id]
        if not isinstance(view, Mapping):
            raise ValueError(f"semantic ESDF camera {camera_id} view is invalid")
        masks = view.get("instance_masks")
        background = identity.get("mapping_mode") == "background"
        if (not isinstance(masks, Mapping) or len(masks) > MAX_INSTANCE_COUNT
                or (background and masks)
                or (not background and camera_id == "3" and not masks)):
            raise ValueError(f"semantic ESDF camera {camera_id} masks are invalid")
        instance_ids = tuple(masks)
        if any(not isinstance(item, str) or not item for item in instance_ids):
            raise ValueError(f"semantic ESDF camera {camera_id} instance id is invalid")
        valid = _array(
            view.get("valid_left_mask"),
            dtype=np.dtype("u1"),
            dimensions=2,
            name=f"camera {camera_id} valid_left_mask",
        )
        depth_name = f"camera {camera_id} depth_left_m"
        try:
            raw_depth = np.asarray(view.get("depth_left_m"), dtype=np.dtype("<f4"))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{depth_name} is not a valid array") from error
        if (
            raw_depth.ndim != 2
            or raw_depth.size < 1
            or raw_depth.size > MAX_IMAGE_PIXELS
        ):
            raise ValueError(f"{depth_name} has an invalid shape")
        if valid.shape != raw_depth.shape:
            raise ValueError(f"semantic ESDF camera {camera_id} left arrays disagree")
        normalized_depth = np.array(raw_depth, dtype=np.dtype("<f4"), copy=True)
        normalized_depth[valid == 0] = 0.0
        depth = _array(
            normalized_depth,
            dtype=np.dtype("<f4"),
            dimensions=2,
            name=depth_name,
        )
        stacked_masks = np.stack(
            [
                _array(
                    masks[instance_id],
                    dtype=np.dtype("u1"),
                    dimensions=2,
                    name=f"camera {camera_id} instance mask",
                )
                for instance_id in instance_ids
            ],
            axis=0,
        ) if masks else None
        rgb_size = view.get("rgb_size")
        if (
            not isinstance(rgb_size, (list, tuple))
            or len(rgb_size) != 2
            or any(type(item) is not int or item < 1 for item in rgb_size)
            or rgb_size[0] * rgb_size[1] > MAX_IMAGE_PIXELS
            or (stacked_masks is not None and stacked_masks.shape[1:] != (rgb_size[1], rgb_size[0]))
        ):
            raise ValueError(f"semantic ESDF camera {camera_id} RGB mask shape is invalid")
        copied = {
            key: deepcopy(value)
            for key, value in view.items()
            if key not in {"depth_left_m", "valid_left_mask", "instance_masks"}
        }
        copied["depth_left_wire"] = _encode_array(
            depth,
            dtype=np.dtype("<f4"),
            dimensions=2,
            name=f"camera {camera_id} depth_left_m",
        )
        copied["valid_left_wire"] = _encode_array(
            valid,
            dtype=np.dtype("u1"),
            dimensions=2,
            name=f"camera {camera_id} valid_left_mask",
        )
        copied["instance_ids"] = list(instance_ids)
        copied["instance_masks_wire"] = _encode_array(
            stacked_masks,
            dtype=np.dtype("u1"),
            dimensions=3,
            name=f"camera {camera_id} instance_masks",
        ) if stacked_masks is not None else None
        encoded_views[camera_id] = copied
    return {**identity, "views": encoded_views}


def decode_semantic_esdf_request(payload: object) -> dict:
    """Decode the bounded wire request into the provider-neutral NumPy contract."""

    identity, views = _identity(payload)
    decoded_views = {}
    for camera_id in views:
        view = views[camera_id]
        if not isinstance(view, Mapping):
            raise ValueError(f"semantic ESDF camera {camera_id} view is invalid")
        instance_ids = view.get("instance_ids")
        background = identity.get("mapping_mode") == "background"
        if (
            not isinstance(instance_ids, list)
            or (background and instance_ids)
            or (not background and camera_id == "3" and not instance_ids)
            or len(instance_ids) > MAX_INSTANCE_COUNT
            or len(set(instance_ids)) != len(instance_ids)
            or any(not isinstance(item, str) or not item for item in instance_ids)
        ):
            raise ValueError(f"semantic ESDF camera {camera_id} instance ids are invalid")
        depth = _decode_array(
            view.get("depth_left_wire"),
            dtype=np.dtype("<f4"),
            dimensions=2,
            name=f"camera {camera_id} depth_left_m",
        )
        valid = _decode_array(
            view.get("valid_left_wire"),
            dtype=np.dtype("u1"),
            dimensions=2,
            name=f"camera {camera_id} valid_left_mask",
        )
        if not instance_ids and view.get("instance_masks_wire") is not None:
            raise ValueError("an empty view must not carry instance masks")
        masks = _decode_array(
            view.get("instance_masks_wire"),
            dtype=np.dtype("u1"),
            dimensions=3,
            name=f"camera {camera_id} instance_masks",
        ) if instance_ids else None
        if depth.shape != valid.shape or (masks is not None and masks.shape[0] != len(instance_ids)):
            raise ValueError(f"semantic ESDF camera {camera_id} array shapes disagree")
        copied = {
            key: deepcopy(value)
            for key, value in view.items()
            if key
            not in {
                "depth_left_wire",
                "valid_left_wire",
                "instance_ids",
                "instance_masks_wire",
            }
        }
        copied["depth_left_m"] = depth
        copied["valid_left_mask"] = valid
        copied["instance_masks"] = {
            instance_id: masks[index]
            for index, instance_id in enumerate(instance_ids)
        }
        decoded_views[camera_id] = copied
    return {**identity, "views": decoded_views}


__all__ = ["decode_semantic_esdf_request", "encode_semantic_esdf_request"]
