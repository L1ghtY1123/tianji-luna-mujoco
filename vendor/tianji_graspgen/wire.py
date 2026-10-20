import math


MAX_POINTS = 4_194_304
MAX_ARRAY_ITEMS = MAX_POINTS * 3
MAX_ARRAY_BYTES = MAX_ARRAY_ITEMS * 4
MAX_REQUEST_BYTES = 104 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_CANDIDATES = 200
MAX_VISUALIZATION_CANDIDATES = 2 * MAX_CANDIDATES

MODEL_IDENTITY = {
    "name": "graspgen",
    "source_sha": "2dd8852e1be60f5f9d277fafcc621835cdf59110",
    "models_source_sha": "ec1ccbb5eec0680db669246ac312a3636f16ee43",
    "config": {
        "name": "graspgen_robotiq_2f_140.yml",
        "size": 4870,
        "sha256": "93804f423ec02def08add164d83379212204ddc33779cd273ffffdacf94f67f3",
    },
    "generator": {
        "size": 471436414,
        "sha256": "fe8497108e39d8fc50be06cd7df22a2f680e0495e713d24c3104616f291e00dd",
    },
    "discriminator": {
        "size": 39427928,
        "sha256": "52f68d3cd3decb71499e084fdb88808d174419a72cd9ef38f821083a4409895d",
    },
}


class WireError(ValueError):
    pass


def pack(value):
    import msgpack

    return msgpack.packb(value, use_bin_type=True)


def unpack(data, *, limit=MAX_RESPONSE_BYTES):
    import msgpack

    if type(data) is not bytes or type(limit) is not int or not 0 <= len(data) <= limit:
        raise WireError("message exceeds limit")
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=True)
    except Exception as error:
        raise WireError("invalid MessagePack") from error


def encode_array(array, *, unit, frame):
    import numpy

    if type(array) is not numpy.ndarray or array.ndim != 2 or array.shape[1:] != (3,):
        raise WireError("invalid ndarray")
    if array.dtype not in (numpy.dtype("float32"), numpy.dtype("uint8")):
        raise WireError("unsupported ndarray dtype")
    item_count = math.prod(array.shape)
    if array.shape[0] > MAX_POINTS or item_count > MAX_ARRAY_ITEMS:
        raise WireError("ndarray exceeds item limit")
    byte_count = item_count * array.itemsize
    if byte_count > MAX_ARRAY_BYTES or array.nbytes != byte_count:
        raise WireError("ndarray exceeds byte limit")
    contiguous = numpy.ascontiguousarray(array)
    if contiguous.nbytes != byte_count:
        raise WireError("invalid ndarray byte count")
    data = contiguous.tobytes(order="C")
    if len(data) != byte_count:
        raise WireError("invalid ndarray byte count")
    return {
        "dtype": str(array.dtype),
        "rank": 2,
        "shape": list(contiguous.shape),
        "unit": unit,
        "frame": frame,
        "data": data,
    }


def decode_array(value, *, dtype, rank, shape_suffix, unit, frame):
    import numpy

    if type(value) is not dict or set(value) != {
        "dtype", "rank", "shape", "unit", "frame", "data"
    }:
        raise WireError("invalid ndarray keys")
    if (
        value["dtype"] != dtype
        or value["rank"] != rank
        or value["unit"] != unit
        or value["frame"] != frame
    ):
        raise WireError("invalid ndarray metadata")
    dimensions = value["shape"]
    if type(dimensions) is not list or len(dimensions) != rank:
        raise WireError("invalid ndarray rank")
    if any(type(item) is not int or item < 0 for item in dimensions):
        raise WireError("invalid ndarray shape")
    if tuple(dimensions[-len(shape_suffix):]) != tuple(shape_suffix):
        raise WireError("unexpected ndarray shape")
    item_count = math.prod(dimensions)
    if dimensions[0] > MAX_POINTS or item_count > MAX_ARRAY_ITEMS:
        raise WireError("ndarray exceeds item limit")
    data = value["data"]
    if type(data) is not bytes:
        raise WireError("invalid ndarray data")
    numpy_dtype = numpy.dtype(dtype)
    byte_count = item_count * numpy_dtype.itemsize
    if byte_count > MAX_ARRAY_BYTES or len(data) != byte_count:
        raise WireError("invalid ndarray byte count")
    return numpy.frombuffer(data, dtype=numpy_dtype).reshape(dimensions).copy()
