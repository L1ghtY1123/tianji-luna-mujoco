"""Bounded, read-only NumPy previews for the experiment evidence viewer."""
from zipfile import BadZipFile, ZipFile

import numpy as np


def read_array_preview(store, identifier, relative, key, max_points=30000):
    if not relative.endswith(".npz") or not key or "/" in key or "\\" in key:
        raise ValueError("invalid_archive_array_reference")
    with store.read_file(identifier, relative) as path:
        try:
            with ZipFile(path) as archive:
                try:
                    info = archive.getinfo(key + ".npy")
                except KeyError as error:
                    raise FileNotFoundError("archive_array_not_found") from error
                if info.file_size > 128 * 1024 * 1024:
                    raise ValueError("archive_array_too_large")
            with np.load(path, allow_pickle=False) as archive:
                array = archive[key]
        except BadZipFile as error:
            raise ValueError("invalid_archive_array_file") from error
    if array.dtype.kind not in "buif" or array.ndim > 4:
        raise ValueError("unsupported_archive_array_type")
    shape = list(array.shape)
    indices = None
    if array.ndim == 2 and array.shape[1] == 3 and array.dtype.kind != "b" and len(array) > max_points:
        indices = np.linspace(0, len(array) - 1, max_points, dtype=np.int64)
        array = array[indices]
    if array.size > 4_000_000:
        raise ValueError("archive_array_preview_too_large")
    data = array.astype(object)
    if array.dtype.kind == "f":
        data[~np.isfinite(array)] = None
    return {
        "path": relative, "key": key, "dtype": str(array.dtype), "shape": shape,
        "sampled": indices is not None, "indices": indices.tolist() if indices is not None else None,
        "sampling": "uniform_original_indices" if indices is not None else "all",
        "data": data.tolist(),
    }
