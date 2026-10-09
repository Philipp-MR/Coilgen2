from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..config import CoilGenConfig
from ..config_validation import validate_config
from ..object_registry import is_in_memory_mesh_key, is_in_memory_target_key
from .storage import json_value


@dataclass(frozen=True, slots=True)
class ResearchRequest:
    """Serializable headless design using the native CoilGenConfig, in SI units.

    Controls are never duplicated in a preset/UI schema. ``design_margin`` only
    defines the final independent-validation budget; it does not silently change
    the optimization tolerance. Set the tighter design tolerance in ``config``.
    """

    name: str
    config: Mapping[str, Any]
    fabricate: bool = False
    kernel_backend: str = "auto"
    validation_resolution: int = 12
    design_margin: float = 1.0
    pipeline_stage: str = "evaluated"
    resume_checkpoint: str | None = None
    project_fingerprint: str = ""
    project_model_fingerprint: str = ""
    project_id: str = ""
    project_revision: int | None = None
    engineering: Mapping[str, Any] | None = None
    requirements: Mapping[str, Any] | None = None
    scene: Mapping[str, Any] | None = None
    dynamic_analysis: Mapping[str, Any] | None = None
    array_study: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 120:
            raise ValueError("Design name must contain 1–120 characters.")
        if not isinstance(self.config, Mapping):
            raise ValueError("config must contain native CoilGenConfig settings.")
        unknown = set(self.config) - {f.name for f in fields(CoilGenConfig)}
        if unknown:
            raise ValueError(f"Unknown CoilGenConfig settings: {', '.join(sorted(unknown))}")
        # Reject opaque objects and nonfinite values, not convert them to null.
        json.dumps(dict(self.config), allow_nan=False)
        if not isinstance(self.fabricate, bool):
            raise ValueError("fabricate must be a boolean.")
        if self.kernel_backend not in {"auto", "reference", "numpy", "numba"}:
            raise ValueError("kernel_backend must be auto, reference, numpy or numba.")
        if (
            isinstance(self.validation_resolution, bool)
            or not isinstance(self.validation_resolution, int)
            or self.validation_resolution < 2
        ):
            raise ValueError("validation_resolution must be an integer >= 2.")
        if (
            isinstance(self.design_margin, bool)
            or not math.isfinite(self.design_margin)
            or not 0 < self.design_margin <= 1
        ):
            raise ValueError("design_margin must lie in (0, 1].")
        object.__setattr__(self, "config", deepcopy(dict(self.config)))
        if self.pipeline_stage not in {"prepared", "sheet", "contours", "routed", "evaluated"}:
            raise ValueError("pipeline_stage must be prepared, sheet, contours, routed, or evaluated.")
        if self.resume_checkpoint is not None and not isinstance(self.resume_checkpoint, str):
            raise ValueError("resume_checkpoint must be a path string or null.")
        if not isinstance(self.project_fingerprint, str):
            raise ValueError("project_fingerprint must be a string.")
        if not isinstance(self.project_model_fingerprint, str):
            raise ValueError("project_model_fingerprint must be a string.")
        if not isinstance(self.project_id, str):
            raise ValueError("project_id must be a string.")
        if self.project_revision is not None and (
            isinstance(self.project_revision, bool)
            or not isinstance(self.project_revision, int)
            or self.project_revision < 1
        ):
            raise ValueError("project_revision must be a positive integer or null.")
        if self.fabricate and self.pipeline_stage != "evaluated":
            raise ValueError("Fabrication review requires an evaluated pipeline result.")
        if self.engineering is not None:
            from ..gradient_engineering import GradientEngineeringSpec

            spec = GradientEngineeringSpec.from_dict(self.engineering)
            if not self.fabricate or self.pipeline_stage != "evaluated":
                raise ValueError("Gradient engineering review requires evaluated fabrication review.")
            object.__setattr__(self, "engineering", spec.to_dict())
        for key, module, name in (
            ("requirements", "research_requirements", "ResearchRequirements"),
            ("scene", "spatial_scene", "SpatialScene"),
            ("dynamic_analysis", "dynamic_studies", "DynamicStudySpec"),
            ("array_study", "array_studies", "ArrayStudySpec"),
        ):
            value = getattr(self, key)
            if value is not None:
                from importlib import import_module

                spec_type = getattr(import_module(f"coilgen.application.{module}"), name)
                object.__setattr__(self, key, spec_type.from_dict(dict(value)).to_dict())

    @classmethod
    def from_config(cls, config: CoilGenConfig, *, name: str = "Research design", **controls) -> ResearchRequest:
        values = asdict(config)

        # Configuration defaults include no unavailable results. Preserve NaN
        # rejection even though result serialization intentionally maps it to null.
        def arrays(value):
            if isinstance(value, dict):
                return {k: arrays(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [arrays(v) for v in value]
            return value.tolist() if hasattr(value, "tolist") else str(value) if isinstance(value, Path) else value

        return cls(name=name, config=arrays(values), **controls)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchRequest:
        if not isinstance(value, Mapping):
            raise ValueError("A research request must be a JSON object.")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown request fields: {', '.join(sorted(unknown))}")
        return cls(**{"name": "Research design", "config": {}, **dict(value)})

    def to_dict(self) -> dict:
        value = json_value(asdict(self))
        for key in ("engineering", "requirements", "scene", "dynamic_analysis", "array_study"):
            if getattr(self, key) is None:
                value.pop(key)
        return value

    def to_config(self, *, threads: int = 1) -> CoilGenConfig:
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer.")
        for name, value in self.config.items():
            if is_in_memory_mesh_key(value) or is_in_memory_target_key(value):
                raise ValueError(f"{name}: process-local registry objects cannot cross workers; use an input file.")
        config = CoilGenConfig(**deepcopy(dict(self.config)))
        geometry_root = config.resolved_geometry_source_path(Path.cwd()).resolve()
        paths = {}
        for key in ("coil_mesh_file", "target_mesh_file", "secondary_target_mesh_file"):
            value = getattr(config, key)
            if str(value).lower().endswith((".stl", ".npz")):
                paths[key] = str((geometry_root / Path(value).expanduser()).resolve())
        if config.target_field_definition_file != "none":
            paths["target_field_definition_file"] = str(
                Path(config.target_field_definition_file).expanduser().resolve()
            )
        for key in ("preoptimized_stream_function_file", "sf_source_file"):
            value = str(getattr(config, key)).strip()
            if value.lower() not in {"", "none"}:
                from ..preoptimization import _resolve_preoptimized_path

                paths[key] = str(_resolve_preoptimized_path(value, Path.cwd()).resolve())
        config = replace(
            config,
            **paths,
            parallel_worker_count=threads,
            constrained_optimization_threads=threads,
            plot_flag=False,
            save_stl_flag=False,
        )
        validation_config = config
        if self.requirements is not None:
            from ..application.research_requirements import requirements_validation_config

            validation_config = requirements_validation_config(config)
        validate_config(validation_config).raise_for_errors()
        if config.research_physics is not None:
            from .convex import ResearchPhysicsSpec

            ResearchPhysicsSpec.from_dict(config.research_physics)
        return config
