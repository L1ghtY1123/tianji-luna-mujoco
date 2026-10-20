"""GraspGen computation entry points, independent of HTTP and model lifecycle."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any
import math
import numpy
from .contracts import GraspCandidate, InvalidOutput, NoCandidates
from .wire import MAX_CANDIDATES, MAX_RESPONSE_BYTES

def _shape_and_bytes(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            shape = numpy.asarray(value).shape
        except Exception as error:
            raise InvalidOutput("official shape unavailable") from error
    try:
        dimensions = tuple(int(item) for item in shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidOutput("invalid official shape") from error
    if any(item < 0 for item in dimensions):
        raise InvalidOutput("invalid official shape")
    item_count = math.prod(dimensions)
    element_size = getattr(value, "element_size", None)
    if callable(element_size):
        try:
            element_bytes = int(element_size())
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidOutput("invalid official item size") from error
    else:
        dtype = getattr(value, "dtype", numpy.dtype("float64"))
        try:
            element_bytes = numpy.dtype(dtype).itemsize
        except TypeError:
            element_bytes = 8
    return dimensions, item_count * element_bytes


def _as_numpy(value):
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return numpy.asarray(value)


def _validate_rigid(transform):
    rotation = transform[:3, :3]
    return (
        numpy.allclose(transform[3], [0, 0, 0, 1], atol=1e-5)
        and numpy.allclose(rotation.T @ rotation, numpy.eye(3), atol=1e-4)
        and numpy.isclose(numpy.linalg.det(rotation), 1.0, atol=1e-4)
    )


def generate(
    sampler: Any, object_points: numpy.ndarray, run_graspmoe: Callable,
) -> list[GraspCandidate]:
    """Return camera-frame grasps from finite float32 [N,3] points in metres.

    The service supplies the loaded sampler and official callable. One official
    call owns outlier removal; centroid restoration and stable ordering happen here.
    """
    centroid = object_points.mean(axis=0, dtype=numpy.float64).astype(numpy.float32)
    centered = numpy.ascontiguousarray(object_points - centroid, dtype=numpy.float32)
    try:
        official = run_graspmoe(
            centered,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=200,
            topk_num_grasps=100,
        )
    except (InvalidOutput, NoCandidates):
        raise
    except Exception as error:
        raise InvalidOutput("official inference failed") from error
    if type(official) is not dict or not {
        "grasps_diff", "grasps_obb", "scores_diff", "scores_obb"
    }.issubset(official):
        raise InvalidOutput("invalid official result")
    values = [
        official["grasps_diff"],
        official["grasps_obb"],
        official["scores_diff"],
        official["scores_obb"],
    ]
    preflight = [_shape_and_bytes(value) for value in values]
    pose_shapes = [preflight[0][0], preflight[1][0]]
    score_shapes = [preflight[2][0], preflight[3][0]]
    if any(len(shape) != 3 or shape[1:] != (4, 4) for shape in pose_shapes):
        raise InvalidOutput("invalid official pose shape")
    if any(len(shape) != 1 for shape in score_shapes):
        raise InvalidOutput("invalid official score shape")
    counts = [pose_shapes[0][0], pose_shapes[1][0], score_shapes[0][0], score_shapes[1][0]]
    if counts[0] != counts[2] or counts[1] != counts[3]:
        raise InvalidOutput("official count mismatch")
    total_count = counts[0] + counts[1]
    if total_count == 0:
        raise NoCandidates("no candidates")
    if total_count > MAX_CANDIDATES or sum(item[1] for item in preflight) > MAX_RESPONSE_BYTES:
        raise InvalidOutput("official result exceeds limit")
    arrays = [_as_numpy(value) for value in values]
    try:
        poses = numpy.concatenate(arrays[:2], axis=0).astype(numpy.float32, copy=True)
        scores = numpy.concatenate(arrays[2:], axis=0).astype(numpy.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidOutput("invalid official result") from error
    if poses.shape != (total_count, 4, 4) or scores.shape != (total_count,):
        raise InvalidOutput("invalid official result shape")
    if not numpy.isfinite(poses).all() or not numpy.isfinite(scores).all():
        raise InvalidOutput("non-finite official result")
    if not all(_validate_rigid(transform) for transform in poses):
        raise InvalidOutput("nonrigid official pose")
    poses[:, :3, 3] += centroid
    order = sorted(range(total_count), key=lambda index: (-float(scores[index]), index))
    return [
        {
            "index": index,
            "T_camera_grasp": poses[index].tolist(),
            "score": float(scores[index]),
        }
        for index in order
    ]
