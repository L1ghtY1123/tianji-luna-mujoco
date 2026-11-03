"""Current RGBD surface correspondences reduce duplicate cross-camera ID questions."""

import hashlib
import json
from copy import deepcopy

import numpy as np

from .geometry import rigid_matrix
from .visual_api import VisualInventoryError

DEPTH_TOLERANCE_M = 0.02
MINIMUM_PIXELS = 512
MINIMUM_OVERLAP = 0.95


def _stamp(array):
    array = np.asarray(array)
    return hashlib.sha256(
        str(array.dtype).encode() + str(array.shape).encode() + np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _current_view(capture, camera):
    view = capture.get("views", {}).get(camera, {})
    if (
        not capture.get("observation_id")
        or capture.get("generation") is None
        or view.get("frame_id") != capture["observation_id"]
        or view.get("simulation_generation") != capture["generation"]
        or view.get("camera_id") != camera
        or view.get("registered_to") != "rgb"
        or not view.get("calibration_id")
    ):
        return None
    depth, valid = view.get("depth_m"), view.get("valid_depth")
    if (
        not isinstance(depth, np.ndarray)
        or depth.ndim != 2
        or not isinstance(valid, np.ndarray)
        or valid.shape != depth.shape
        or valid.dtype != np.bool_
    ):
        return None
    intrinsics = view.get("intrinsics", {})
    values = [intrinsics.get(key, float("nan")) for key in ("fx", "fy", "cx", "cy")]
    if not np.isfinite(values).all() or min(values[:2]) <= 0:
        return None
    try:
        transform = rigid_matrix(view["T_world_camera"])
    except (KeyError, ValueError, TypeError):
        return None
    return {**view, "T_world_camera": transform}


def _visible_pixels(points, view):
    matrix = np.linalg.inv(view["T_world_camera"])
    camera = points @ matrix[:3, :3].T + matrix[:3, 3]
    camera = camera[camera[:, 2] > 1e-6]
    intrinsics = view["intrinsics"]
    projected = np.rint(
        camera[:, :2] / camera[:, 2, None] * [intrinsics["fx"], intrinsics["fy"]]
        + [intrinsics["cx"], intrinsics["cy"]]
    )
    height, width = view["depth_m"].shape
    inside = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    uv, camera = projected[inside].astype(int), camera[inside]
    depth = view["depth_m"][uv[:, 1], uv[:, 0]]
    visible = (
        view["valid_depth"][uv[:, 1], uv[:, 0]]
        & np.isfinite(depth)
        & (depth > 0)
        & (np.abs(depth - camera[:, 2]) <= DEPTH_TOLERANCE_M)
    )
    return np.unique(uv[visible], axis=0)


def measured_view_groups(capture, detections):
    """Require mutual visible overlap; reject ambiguous or non-clique components."""
    views = {
        camera: view
        for camera in capture.get("views", {})
        if (view := _current_view(capture, camera)) is not None
    }
    measured = {}
    for index, detection in enumerate(detections):
        camera = detection["camera_id"]
        if camera not in views:
            continue
        points = np.asarray(detection.get("points", []), float)
        mask = detection.get("silhouette_mask")
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < MINIMUM_PIXELS
            or not np.isfinite(points).all()
            or not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool_
            or mask.shape != views[camera]["depth_m"].shape
        ):
            continue
        measured[index] = {"camera": camera, "points": points, "mask": mask}
    pairs = {}
    for i, source in measured.items():
        for camera, view in views.items():
            if camera == source["camera"]:
                continue
            pixels = _visible_pixels(source["points"], view)
            if len(pixels) < MINIMUM_PIXELS:
                continue
            for j, target in measured.items():
                if target["camera"] != camera:
                    continue
                fraction = float(np.mean(target["mask"][pixels[:, 1], pixels[:, 0]]))
                pairs[i, j] = {
                    "candidate": i,
                    "camera": source["camera"],
                    "compared_candidate": j,
                    "compared_camera": camera,
                    "visible_pixels_20mm": len(pixels),
                    "inside_other_mask_fraction": fraction,
                }
    neighbors = {i: set() for i in measured}
    for (i, j), row in pairs.items():
        reverse = pairs.get((j, i))
        if (
            reverse
            and min(row["inside_other_mask_fraction"], reverse["inside_other_mask_fraction"])
            >= MINIMUM_OVERLAP
        ):
            neighbors[i].add(j)
    groups, visited = [], set()
    for start in sorted(neighbors):
        if start in visited:
            continue
        component, pending = set(), [start]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(neighbors[current] - component)
        visited.update(component)
        if (
            len(component) < 2
            or len({measured[i]["camera"] for i in component}) != len(component)
            or any(component - {i} != neighbors[i] for i in component)
        ):
            continue
        members = sorted(component)
        groups.append(
            {
                "representative": "candidate-" + str(members[0]),
                "members": ["candidate-" + str(i) for i in members],
                "mutual_projection_checks": [pairs[i, j] for i in members for j in members if i != j],
            }
        )
    if not groups:
        return {}
    used = {int(name.removeprefix("candidate-")) for group in groups for name in group["members"]}
    cameras = {measured[i]["camera"] for i in used}
    proof = {
        "observation_id": capture["observation_id"],
        "generation": capture["generation"],
        "source": "mutual_current_registered_depth_and_silhouette_overlap",
        "groups": groups,
        "depth_tolerance_m": DEPTH_TOLERANCE_M,
        "minimum_visible_pixels": MINIMUM_PIXELS,
        "minimum_overlap": MINIMUM_OVERLAP,
        "candidate_stamps": {
            str(i): {
                "points_sha256": _stamp(measured[i]["points"]),
                "silhouette_sha256": _stamp(measured[i]["mask"]),
            }
            for i in sorted(used)
        },
        "view_stamps": {
            camera: {
                "calibration_id": views[camera]["calibration_id"],
                "depth_sha256": _stamp(views[camera]["depth_m"]),
                "valid_sha256": _stamp(views[camera]["valid_depth"]),
                "intrinsics": views[camera]["intrinsics"],
                "T_world_camera": views[camera]["T_world_camera"].tolist(),
            }
            for camera in sorted(cameras)
        },
    }
    return {**proof, "binding_hash": _digest(proof)}


def validate_group_proof(proof, capture):
    if not proof:
        return
    content = {key: value for key, value in proof.items() if key != "binding_hash"}
    if (
        proof.get("observation_id") != capture.get("observation_id")
        or proof.get("generation") != capture.get("generation")
        or proof.get("binding_hash") != _digest(content)
    ):
        raise VisualInventoryError("multiview_identity_proof_invalid")


def reduce_identity_questions(questions, proof, *, preferred_questions=(), visible_pixels=None):
    by_name = {question["name"]: question for question in questions}
    aliases = {}
    for group in proof.get("groups", []):
        names = ["identity_" + member.removeprefix("candidate-") for member in group["members"]]
        if any(name not in by_name or by_name[name]["type"] != "choice" for name in names):
            continue
        options = [{row["value"] for row in by_name[name]["choices"]} for name in names]
        if any(values != options[0] for values in options[1:]):
            continue
        source = min(names, key=lambda name: (name not in preferred_questions,
                     -(visible_pixels or {}).get(name, 0)))
        aliases.update({name: source for name in names if name != source})
    return [deepcopy(q) for q in questions if q["name"] not in aliases], aliases


def grouped_candidates(context, candidate):
    """Return all views of an already validated current surface correspondence."""
    for group in context.get('current_measured_view_groups', {}).get('groups', []):
        if candidate in group['members']:
            active = context.get('identity_request_candidate_ids', group['members'])
            return [member for member in group['members'] if member in active]
    return [candidate]


def grouped_view_instruction(context, candidate):
    members = grouped_candidates(context, candidate)
    if len(members) < 2:
        return ''
    labels = ', '.join('Candidate ' + member.removeprefix('candidate-') for member in members)
    return (f' Use all CURRENT views of this measured surface group: {labels}. '
            'Use the clearest member view to compare the subject with its reference; '
            'the representative camera is not the only evidence. '
            'The RGBD correspondence links current views, not a permanent identity.')


def expand_identity_answers(body, questions, aliases, proof):
    answers = {answer["name"]: deepcopy(answer) for answer in body["answers"]}
    derivations = {}
    for name, source in aliases.items():
        if source not in answers or name in answers:
            raise VisualInventoryError("multiview_identity_answer_binding_invalid")
        answers[name] = {**deepcopy(answers[source]), "name": name}
        derivations[name] = {
            "source": "current_measured_view_correspondence",
            "source_question": source,
            "binding_hash": proof["binding_hash"],
            "observation_id": proof["observation_id"],
            "generation": proof["generation"],
        }
    if set(answers) != {q["name"] for q in questions}:
        raise VisualInventoryError("multiview_identity_answer_binding_invalid")
    return {**body, "answers": [answers[q["name"]] for q in questions]}, derivations
