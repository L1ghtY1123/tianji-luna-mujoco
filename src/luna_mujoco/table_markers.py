"""Measure colored tabletop placement regions from registered RGBD only.

No scene definition, authored dimensions or simulator identifiers enter this
module. One visible connected region per color provides a stable visual label.
Hidden pixels stay absent; the measured hull is a region, not a free-space claim.
"""

import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .cameras import deproject_view


def _readonly(values):
    result = np.ascontiguousarray(values).copy()
    result.flags.writeable = False
    return result


def _color_masks(rgb):
    pixels = np.asarray(rgb, dtype=float) / 255
    maximum = pixels.max(axis=2)
    delta = maximum - pixels.min(axis=2)
    saturated = (maximum > .08) & (delta > .3 * maximum)
    denominator = np.maximum(delta, 1e-9)
    red, green, blue = np.moveaxis(pixels, 2, 0)
    hue = np.zeros(maximum.shape)
    hue[maximum == red] = ((green - blue) / denominator)[maximum == red] % 6
    hue[maximum == green] = ((blue - red) / denominator + 2)[maximum == green]
    hue[maximum == blue] = ((red - green) / denominator + 4)[maximum == blue]
    hue /= 6
    return {"red": saturated & ((hue < .04) | (hue >= .94)),
            "orange": saturated & (hue >= .04) & (hue < .11),
            "yellow": saturated & (hue >= .11) & (hue < .19),
            "green": saturated & (hue >= .19) & (hue < .46),
            "blue": saturated & (hue >= .46) & (hue < .75)}


def _view_regions(camera_id, view, table):
    valid = view["valid_depth"] & np.isfinite(view["depth_m"]) & (view["depth_m"] > 0)
    pixels = np.column_stack(np.nonzero(valid))
    points = deproject_view(view, valid)
    tolerance = max(.001, min(.004, float(table.get("uncertainty_m", .003))))
    on_table = np.abs(points[:, 2] - table["surface_height_m"]) <= tolerance
    plane = np.zeros(valid.shape, bool)
    plane[tuple(pixels[on_table].T)] = True
    regions = []
    for color, color_mask in _color_masks(view["rgb"]).items():
        candidate_mask = color_mask & plane
        if np.count_nonzero(candidate_mask) < 30:
            continue
        # Height alone admits the bottom strip of a vertical colored wall.
        # Measure normals in the full visible colored surface before clipping
        # to tabletop height, so that strip cannot become a horizontal region.
        colored_points = deproject_view(view, color_mask)
        candidates = deproject_view(view, candidate_mask)
        distances, indices = cKDTree(colored_points).query(candidates, k=min(16, len(colored_points)))
        neighbours = colored_points[indices]
        centered = neighbours - neighbours.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbours.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        horizontal = ((np.abs(eigenvectors[:, 2, 0]) >= np.cos(np.deg2rad(15.)))
                      & (eigenvalues[:, 1] > 1e-8)
                      & (eigenvalues[:, 0] <= .15 * eigenvalues[:, 1])
                      & (distances[:, -1] <= .03))
        rows, columns = np.nonzero(candidate_mask)
        candidate_mask[rows[~horizontal], columns[~horizontal]] = False
        components, count = label(candidate_mask)
        for component in range(1, count + 1):
            mask = components == component
            if np.count_nonzero(mask) < 30:
                continue
            measured = deproject_view(view, mask)
            if np.any(np.ptp(measured[:, :2], axis=0) < .025):
                continue
            try:
                hull = ConvexHull(measured[:, :2])
            except QhullError:
                continue
            if hull.volume < .0009:
                continue
            regions.append({"color": color, "camera_id": camera_id, "mask": mask,
                            "points": measured, "pixels_rc": np.column_stack(np.nonzero(mask)),
                            "rgb": view["rgb"][mask]})
    return regions


def _overlapping_regions(first, second):
    if first["color"] != second["color"] or first["camera_id"] == second["camera_id"]:
        return False
    a, b = first["points"], second["points"]
    if len(a) < len(b):
        a, b = b, a
    return np.mean(cKDTree(a).query(b)[0] <= .006) >= .1


def _measured_marker(group, capture):
    raw = np.vstack([entry["points"] for entry in group])
    _, indices = np.unique(np.floor(raw / .001).astype(np.int64), axis=0, return_index=True)
    indices = np.sort(indices)
    points = raw[indices].astype(np.float32)
    cameras = np.concatenate([np.full(len(entry["points"]), entry["camera_id"]) for entry in group])[indices]
    pixels = np.vstack([entry["pixels_rc"] for entry in group])[indices]
    centroid = points.mean(axis=0, dtype=np.float64)
    reference_index = int(np.argmin(np.linalg.norm(points[:, :2] - centroid[:2], axis=1)))
    reference = points[reference_index].tolist()
    color = group[0]["color"]
    median = np.median(np.vstack([entry["rgb"] for entry in group]), axis=0) / 255
    polygon = points[ConvexHull(points[:, :2]).vertices, :2].tolist()
    camera_ids = sorted({entry["camera_id"] for entry in group})
    masks = {camera: np.zeros(capture["views"][camera]["depth_m"].shape, bool) for camera in camera_ids}
    for entry in group:
        masks[entry["camera_id"]] |= entry["mask"]
    return {"id": "marker-" + color, "kind": "marker", "color": color, "graspable": False,
            "position": centroid.tolist(), "visible_centroid_world": centroid.tolist(),
            "estimated": True, "uncertainty_m": .003,
            "source": "colored_table_region_registered_rgbd",
            "color_evidence": {"median_rgb": median.tolist(), "source": "measured_table_rgb"},
            "points_world_m": _readonly(points),
            "point_sources": {"camera_ids": _readonly(cameras), "pixels_rc": _readonly(pixels),
                              "observation_id": capture["observation_id"]},
            "masks_by_camera": {camera: _readonly(mask) for camera, mask in masks.items()},
            "observation_provenance": {"observation_id": capture["observation_id"],
                                       "generation": capture["generation"], "capture_time": capture["time"]},
            "coverage": {"observed_surface_only": True, "camera_ids": camera_ids, "sam3_camera_ids": [],
                         "point_count": len(points), "limitations": ["visible_colored_table_surface_only",
                          "region_hull_does_not_establish_unoccupied_space"]},
            "placement_reference": {"position_world_m": reference, "floor_z_m": float(np.median(points[:, 2])),
                "region_polygon_world_xy_m": polygon, "source": "measured_colored_table_surface",
                "measurement_sources": {"point_count": len(points), "camera_ids": camera_ids,
                    "point_indices": list(range(len(points))), "reference_camera_id": str(cameras[reference_index]),
                    "reference_pixel_rc": pixels[reference_index].tolist()}}}


def detect_table_markers(capture, table):
    """Return measured marker objects; ambiguous same-color regions raise ValueError.

    ``capture`` has registered RGB, depth and calibration under ``views``.
    ``table`` supplies the RGBD-estimated horizontal surface height/uncertainty.
    Callers install ``masks_by_camera`` as instance masks and exclude them from
    graspable source detections. This function never changes input sensor arrays.
    """
    regions = []
    for camera_id, view in capture["views"].items():
        regions.extend(_view_regions(camera_id, view, table))
    groups = []
    for region in regions:
        matches = [index for index, group in enumerate(groups)
                   if any(_overlapping_regions(region, entry) for entry in group)]
        if not matches:
            groups.append([region])
            continue
        combined = [region]
        for index in reversed(matches):
            combined.extend(groups.pop(index))
        groups.append(combined)
    colors = set()
    for group in groups:
        color = group[0]["color"]
        cameras = [entry["camera_id"] for entry in group]
        if color in colors or len(set(cameras)) != len(cameras):
            raise ValueError("table_marker_color_ambiguous:" + color)
        colors.add(color)
    return [_measured_marker(group, capture) for group in sorted(groups, key=lambda item: item[0]["color"])]
