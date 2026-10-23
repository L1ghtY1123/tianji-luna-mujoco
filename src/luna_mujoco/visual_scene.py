"""Image-grounded semantic evidence, with no motion or metric-pose outputs."""
import base64
import hashlib
import json
from io import BytesIO
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field


class VisualObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(max_length=128)
    label: str = Field(max_length=100)
    kind: str = Field(max_length=40)
    color: str = Field(max_length=40)
    visibility: Literal["visible", "partial", "occluded", "unknown"]
    ambiguous: bool
    on_table: bool | None
    inside_id: str | None = Field(max_length=128)
    on_marker_id: str | None = Field(max_length=128)
    evidence: str = Field(max_length=240)


class VisualRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_id: str = Field(max_length=128)
    relation: Literal["inside", "on_marker", "beside", "touching", "occludes"]
    object_id: str = Field(max_length=128)


class VisualScene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objects: list[VisualObject] = Field(max_length=32)
    relationships: list[VisualRelationship] = Field(max_length=64)
    evidence: str = Field(max_length=600)


def capture_fingerprint(capture, *, mask_ids=None):
    """Hash actual RGB/masks too: reused IDs cannot certify an unchanged snapshot."""
    digest = hashlib.sha256(json.dumps([capture["generation"], capture["observation_id"]]).encode())
    for camera in ("3", "2", "1"):
        view = capture["views"][camera]
        rgb = np.asarray(view["rgb"])
        digest.update(json.dumps([camera, rgb.shape, str(rgb.dtype)]).encode())
        digest.update(rgb.tobytes())
        masks = view.get("instance_masks", {})
        identifiers = masks if mask_ids is None else mask_ids.get(camera, [])
        for identifier in sorted(identifiers):
            values = masks.get(identifier)
            if values is None:
                digest.update(json.dumps([identifier, 'missing']).encode())
                continue
            mask = np.asarray(values, dtype=bool)
            digest.update(json.dumps([identifier, mask.shape]).encode())
            digest.update(mask.tobytes())
    return digest.hexdigest()


def scene_legend(capture, objects):
    if len(objects) > 32:
        raise ValueError("visual_input_budget_exceeded")
    legend, identifiers = [], set()
    for number, obj in enumerate(objects, 1):
        identifier = obj["id"]
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128 or identifier in identifiers:
            raise ValueError("invalid_visual_object_ids")
        identifiers.add(identifier)
        cameras = []
        for camera in ("3", "2", "1"):
            view = capture["views"][camera]
            mask = view.get("instance_masks", {}).get(identifier)
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != np.asarray(view["rgb"]).shape[:2]:
                raise ValueError("invalid_visual_instance_mask")
            if mask.any():
                cameras.append(camera)
        legend.append({"id": identifier, "overlay_number": number, "camera_ids": cameras})
    return legend


def image_content(capture, reference=None, *, legend=()):
    selected = [(capture, camera, "current") for camera in ("3", "2", "1")]
    if reference is not None:
        selected.append((reference, "3", "reference"))
    content = []
    palette = ((255, 200, 0), (0, 220, 255), (255, 110, 200), (140, 255, 110))
    for item, camera, role in selected:
        view = item["views"][camera]
        picture = Image.fromarray(np.asarray(view["rgb"])).convert("RGB")
        width, height = picture.size
        labels = []
        for entry in legend:
            mask = view.get("instance_masks", {}).get(entry["id"])
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (height, width) or not mask.any():
                continue
            color = palette[(entry["overlay_number"] - 1) % len(palette)]
            tint = Image.new("RGB", picture.size, color)
            picture = Image.composite(tint, picture, Image.fromarray(mask.astype(np.uint8) * 45))
            rows, columns = np.nonzero(mask)
            labels.append((float(columns.mean()), float(rows.mean()), str(entry["overlay_number"]), color))
        picture.thumbnail((640, 640))
        draw = ImageDraw.Draw(picture)
        for x, y, label, color in labels:
            xy = (min(x * picture.width / width, max(0, picture.width - 20)),
                  min(y * picture.height / height, max(0, picture.height - 14)))
            draw.rectangle(draw.textbbox(xy, label), fill=(0, 0, 0))
            draw.text(xy, label, fill=color)
        buffer = BytesIO()
        picture.save(buffer, format="JPEG", quality=85)
        content.extend([{"type": "input_text", "text": f"{role} capture={item['observation_id']}, camera={camera}"},
                        {"type": "input_image", "detail": "auto",
                         "image_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}])
    return content, len(selected)


def grounded_scene(scene, legend):
    """Refuse invented identities; unavailable or ambiguous evidence remains unknown."""
    allowed = {entry["id"] for entry in legend}
    grounded = {entry["id"] for entry in legend if entry["camera_ids"]}
    returned = {}
    for obj in scene.objects:
        if obj.id not in grounded or obj.id in returned:
            raise ValueError("ungrounded_visual_identity")
        if any(identifier is not None and (identifier not in allowed or identifier == obj.id)
               for identifier in (obj.inside_id, obj.on_marker_id)):
            raise ValueError("ungrounded_visual_membership")
        returned[obj.id] = obj.model_dump()
    visible = {obj.id for obj in scene.objects if obj.visibility in {"visible", "partial"} and not obj.ambiguous}
    objects = []
    for entry in legend:
        obj = returned.get(entry["id"], {"id": entry["id"], "visibility": "unknown", "ambiguous": True,
                                         "evidence": "No unambiguous current visual evidence."})
        if entry["id"] not in visible:
            obj.update(label="", kind="", color="", on_table=None, inside_id=None, on_marker_id=None)
        for field in ("inside_id", "on_marker_id"):
            if obj[field] not in visible:
                obj[field] = None
        objects.append(obj)
    relationships = []
    for relation in scene.relationships:
        if relation.subject_id not in allowed or relation.object_id not in allowed:
            raise ValueError("ungrounded_visual_relationship")
        if relation.subject_id in visible and relation.object_id in visible:
            relationships.append(relation.model_dump())
    return {"objects": objects, "relationships": relationships, "evidence": scene.evidence}
