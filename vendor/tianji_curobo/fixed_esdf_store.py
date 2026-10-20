"""Double buffer for candidate and published cuRobo volumetric mappers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any
from uuid import uuid4

from .contracts import CuroboToolError
from .volumetric_mapper import EsdfIntegrationResult, EsdfStatusResult


class FixedEsdfStore:
    """Keep the published mapper alive while a replacement is constructed."""

    def __init__(
        self,
        *,
        mapper_factory: Callable[[], Any],
        required_frame_count: int = 20,
        selection_validator: Callable[[Any, Mapping, Mapping, str], bool] | None = None,
    ) -> None:
        if not callable(mapper_factory):
            raise TypeError("mapper_factory must be callable")
        if isinstance(required_frame_count, bool) or int(required_frame_count) < 1:
            raise ValueError("required_frame_count must be a positive integer")
        self._mapper_factory = mapper_factory
        if selection_validator is not None and not callable(selection_validator):
            raise TypeError("selection_validator must be callable")
        self._selection_validator = selection_validator
        self._required_frame_count = int(required_frame_count)
        self._spare_mapper = None
        self._candidate_build_id: str | None = None
        self._candidate_frame_count = 0
        self._candidate_expected_frame_count = self._required_frame_count
        self._candidate_mapper = None
        self._candidate_result: EsdfIntegrationResult | None = None
        self._candidate_scene_model: dict | None = None
        self._candidate_metadata: dict | None = None
        self._published_mapper = None
        self._published_result: EsdfIntegrationResult | None = None
        self._published_scene_model: dict | None = None
        self._published_metadata: dict | None = None
        self._published_build_id: str | None = None
        self._cleared_publication: tuple[str, str] | None = None
        self._instance_id = uuid4().hex
        self._revision = 0

    @staticmethod
    def _validated_build_id(build_id: object) -> str:
        value = str(build_id) if isinstance(build_id, str) else ""
        if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("fixed ESDF build_id must be 32 lowercase hexadecimal characters")
        return value

    def _require_candidate(self, build_id: object) -> str:
        value = self._validated_build_id(build_id)
        if self._candidate_mapper is None or self._candidate_build_id != value:
            raise RuntimeError("fixed ESDF candidate transaction does not match build_id")
        return value

    @property
    def published_mapper(self):
        return self._published_mapper

    @property
    def candidate_mapper(self):
        return self._candidate_mapper

    def candidate_mapper_for(self, build_id: str):
        self._require_candidate(build_id)
        return self._candidate_mapper

    @property
    def candidate_scene_model(self):
        return deepcopy(self._candidate_scene_model)

    def candidate_scene_model_for(self, build_id: str):
        self._require_candidate(build_id)
        return deepcopy(self._candidate_scene_model)

    def candidate_metadata_for(self, build_id: str):
        self._require_candidate(build_id)
        return deepcopy(self._candidate_metadata)

    @property
    def published_scene_model(self):
        return deepcopy(self._published_scene_model)

    @property
    def published_selection(self):
        metadata = self._published_metadata or {}
        return {
            "selected_instance_id": metadata.get("selected_instance_id"),
            "selection_generation": metadata.get("selection_generation"),
        }

    def begin_candidate(self, build_id: str) -> None:
        value = self._validated_build_id(build_id)
        if self._candidate_mapper is not None and self._candidate_build_id == value:
            return
        spare = self._spare_mapper
        reset = getattr(getattr(spare, "sdk_boundary", None), "reset_capture", None)
        if callable(reset):
            reset()
            self._candidate_mapper = spare
        else:
            self._candidate_mapper = self._mapper_factory()
        self._spare_mapper = None
        self._candidate_build_id = value
        self._candidate_frame_count = 0
        self._candidate_expected_frame_count = self._required_frame_count
        self._candidate_result = None
        self._candidate_scene_model = None
        self._candidate_metadata = None

    def integrate_candidate(
        self,
        build_id: str,
        request: Any,
        *,
        finalize: bool,
    ) -> EsdfIntegrationResult:
        self._require_candidate(build_id)
        if not isinstance(finalize, bool):
            raise TypeError("finalize must be a bool")
        next_frame_count = self._candidate_frame_count + 1
        if next_frame_count > self._candidate_expected_frame_count:
            raise RuntimeError("fixed ESDF candidate already contains all required frames")
        expected_finalize = next_frame_count == self._candidate_expected_frame_count
        if finalize is not expected_finalize:
            raise RuntimeError(
                "fixed ESDF finalize must be true only on required frame "
                f"{self._candidate_expected_frame_count}"
            )
        result = self._candidate_mapper.integrate_esdf(
            request,
            finalize=finalize,
        )
        if not isinstance(result, EsdfIntegrationResult):
            raise TypeError("fixed ESDF candidate returned an invalid result")
        if result.integrated_frame_count != next_frame_count:
            raise RuntimeError("fixed ESDF mapper returned an invalid frame count")
        if result.ready is not expected_finalize:
            raise RuntimeError("fixed ESDF mapper returned an invalid ready state")
        self._candidate_frame_count = next_frame_count
        self._candidate_result = result
        return result

    def stage_semantic_candidate(
        self,
        build_id: str,
        result: EsdfIntegrationResult,
        scene_model: Mapping,
        metadata: Mapping,
    ) -> None:
        """Stage a fused result with the exact camera count, without publishing."""

        self._require_candidate(build_id)
        if self._candidate_frame_count != 0 or self._candidate_result is not None:
            raise RuntimeError("fixed ESDF candidate already contains observations")
        camera_ids = tuple(metadata.get("camera_metrics", {}))
        mode = metadata.get("mapping_mode", "semantic")
        if mode == "background":
            if camera_ids not in (("3",), ("1", "3"), ("2", "3"), ("1", "2", "3")):
                raise ValueError("background ESDF requires ordered camera metrics containing head camera 3")
            expected_count = len(camera_ids)
        elif mode == "semantic":
            if "camera_metrics" in metadata and camera_ids != ("1", "2", "3"):
                raise ValueError("semantic ESDF requires all three ordered camera metrics")
            expected_count = 3
        else:
            raise ValueError("invalid ESDF mapping mode")
        if (
            not isinstance(result, EsdfIntegrationResult)
            or result.ready is not True
            or result.integrated_frame_count != expected_count
        ):
            raise TypeError("ESDF candidate frame count does not match its mapping mode")
        self.set_candidate_scene(build_id, scene_model, metadata)
        self._candidate_expected_frame_count = expected_count
        self._candidate_frame_count = expected_count
        self._candidate_result = result

    def set_candidate_scene(
        self,
        build_id: str,
        scene_model: Mapping,
        metadata: Mapping,
    ) -> None:
        self._require_candidate(build_id)
        cuboids = scene_model.get("cuboid") if isinstance(scene_model, Mapping) else None
        if not isinstance(cuboids, Mapping) or not cuboids:
            raise ValueError("fixed ESDF semantic scene requires at least one Cuboid")
        if not isinstance(metadata, Mapping):
            raise TypeError("fixed ESDF semantic metadata must be a mapping")
        self._candidate_scene_model = deepcopy(dict(scene_model))
        self._candidate_metadata = deepcopy(dict(metadata))

    def publish_candidate(self, build_id: str) -> EsdfIntegrationResult:
        value = self._validated_build_id(build_id)
        if self._published_build_id == value and self._published_result is not None:
            return self._published_result
        self._require_candidate(value)
        result = self._candidate_result
        if self._candidate_frame_count != self._candidate_expected_frame_count:
            raise RuntimeError(
                "fixed ESDF publish requires exactly "
                f"{self._candidate_expected_frame_count} frames"
            )
        if self._candidate_mapper is None or result is None or not result.ready:
            raise RuntimeError("fixed ESDF publish requires a ready candidate")
        if self._candidate_scene_model is None:
            raise RuntimeError("fixed ESDF publish requires a semantic scene")
        self._revision += 1
        published = replace(
            result,
            revision=self._revision,
            scene_token=(
                f"fixed-esdf:{self._instance_id}:{self._revision}"
            ),
        )
        self._spare_mapper = self._published_mapper
        self._published_mapper = self._candidate_mapper
        self._published_result = published
        self._published_scene_model = self._candidate_scene_model
        self._published_metadata = self._candidate_metadata
        self._published_build_id = value
        self._candidate_mapper = None
        self._candidate_build_id = None
        self._candidate_frame_count = 0
        self._candidate_expected_frame_count = self._required_frame_count
        self._candidate_result = None
        self._candidate_scene_model = None
        self._candidate_metadata = None
        return published

    def clear_published(self, build_id: str, expected_scene_token: str) -> bool:
        """Forget only the owned result; recycle its allocation on next capture."""
        value = self._validated_build_id(build_id)
        if self._published_result is None and self._cleared_publication == (value, expected_scene_token):
            return True
        if (self._published_build_id != value or self._published_result is None
                or self._published_result.scene_token != expected_scene_token):
            return False
        self._cleared_publication = (value, expected_scene_token)
        self._spare_mapper = self._published_mapper
        self._published_mapper = None
        self._published_result = None
        self._published_scene_model = None
        self._published_metadata = None
        self._published_build_id = None
        return True

    def discard_candidate(self, build_id: str) -> None:
        value = self._validated_build_id(build_id)
        if self._published_build_id == value:
            return
        if self._candidate_build_id != value:
            return
        self._spare_mapper = self._candidate_mapper
        self._candidate_mapper = None
        self._candidate_build_id = None
        self._candidate_frame_count = 0
        self._candidate_expected_frame_count = self._required_frame_count
        self._candidate_result = None
        self._candidate_scene_model = None
        self._candidate_metadata = None

    def reconcile_candidate(self, build_id: str) -> EsdfIntegrationResult | None:
        value = self._validated_build_id(build_id)
        if self._published_build_id == value:
            return self._published_result
        return None

    @staticmethod
    def _semantic_obstacles(scene_model: Mapping) -> dict[str, str]:
        result = {}
        for geometry_type in ("cuboid", "mesh", "voxel"):
            values = scene_model.get(geometry_type)
            if not isinstance(values, Mapping):
                continue
            for name in values:
                if isinstance(name, str) and name.startswith("semantic_instance::"):
                    result[name.removeprefix("semantic_instance::")] = name
        return result

    def select_instance(
        self,
        *,
        expected_revision: int,
        selected_instance_id: str,
        selection_generation: int,
    ) -> EsdfIntegrationResult:
        """Atomically change only the disabled named semantic obstacle."""

        published = self._published_result
        mapper = self._published_mapper
        scene_model = self._published_scene_model
        metadata = self._published_metadata
        if published is None or mapper is None or scene_model is None or metadata is None:
            raise CuroboToolError(
                "fixed semantic ESDF is not published",
                stage="scene",
                code="CUROBO_ESDF_NOT_READY",
                status=409,
            )
        if type(expected_revision) is not int or expected_revision != published.revision:
            raise CuroboToolError(
                "fixed ESDF scene revision changed",
                stage="scene",
                code="CUROBO_SCENE_CHANGED",
                status=409,
                details={
                    "expected_revision": expected_revision,
                    "actual_revision": published.revision,
                },
            )
        current_generation = metadata.get("selection_generation", 0)
        if (
            type(selection_generation) is not int
            or selection_generation <= int(current_generation or 0)
        ):
            raise CuroboToolError(
                "semantic object selection is stale",
                stage="scene",
                code="SCENE_OBJECT_UNSTABLE",
                status=409,
            )
        instances = self._semantic_obstacles(scene_model)
        disabled_name = instances.get(str(selected_instance_id))
        if disabled_name is None:
            raise CuroboToolError(
                "semantic object is not part of the published scene",
                stage="scene",
                code="SCENE_OBJECT_UNSTABLE",
                status=409,
            )
        validator = self._selection_validator
        if validator is None:
            raise CuroboToolError(
                "semantic scene selection validation is unavailable",
                stage="configuration",
                code="CUROBO_INVALID_CONFIG",
                status=500,
            )
        if (
            validator(
                mapper,
                deepcopy(scene_model),
                deepcopy(metadata),
                disabled_name,
            )
            is not True
        ):
            raise CuroboToolError(
                "semantic target selection collides at the captured pose",
                stage="scene",
                code="CUROBO_SCENE_SELECTION_COLLISION",
                status=409,
            )
        self._revision += 1
        next_result = replace(
            published,
            revision=self._revision,
            scene_token=f"fixed-esdf:{self._instance_id}:{self._revision}",
        )
        next_metadata = deepcopy(metadata)
        next_metadata["selected_instance_id"] = str(selected_instance_id)
        next_metadata["selection_generation"] = selection_generation
        self._published_result = next_result
        self._published_metadata = next_metadata
        return next_result

    def published_status(self) -> EsdfStatusResult:
        result = self._published_result
        if result is None:
            return EsdfStatusResult(
                ready=False,
                revision=0,
                integrated_frame_count=0,
                point_count=0,
                scene_token=None,
            )
        return EsdfStatusResult(
            ready=True,
            revision=result.revision,
            integrated_frame_count=result.integrated_frame_count,
            point_count=result.point_count,
            scene_token=result.scene_token,
        )


__all__ = ["FixedEsdfStore"]
