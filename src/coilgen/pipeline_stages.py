"""Detached, restartable scientific stage artifacts shared by scripts and GUIs.

Artifacts contain strict JSON and numerical NPZ data, never pickle or imported
class names. Unknown settings invalidate preparation rather than guessing that
they are safe to reuse. A checkpoint is evidence for a completed stage, not an
engineering acceptance certificate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import io
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import uuid
import zipfile

import numpy as np

from .config import CoilGenConfig
from .models import CoilGenResult, CoilPart, Mesh, TargetField
from .persistence import file_digest, numerical_source_digest


class PipelineStage(str, Enum):
    PREPARED = "prepared"
    SHEET = "sheet"
    CONTOURS = "contours"
    ROUTED = "routed"
    EVALUATED = "evaluated"

    @property
    def index(self) -> int:
        return tuple(type(self)).index(self)


# These are explicit exceptions to conservative preparation invalidation.
# In particular, wire area, sheet material, current and spacing constraints are
# deliberately not classified as routing-only settings.
_DISPLAY_FIELDS = frozenset(
    {
        "progress_flag",
        "progress_min_interval_seconds",
        "progress_percent_step",
        "plot_flag",
        "plot_pointcloud_max_points",
        "plot_pointcloud_random_seed",
        "field_display_units",
        "output_directory",
        "save_stl_flag",
        "save_wire_trace_csv",
        "wire_trace_csv_format",
        "wire_trace_csv_length_unit",
        "wire_trace_csv_decimal_places",
        "chunked_output_writing",
    }
)
_CONTOUR_FIELDS = frozenset(
    {
        "levels",
        "pot_offset_factor",
        "contour_part_phase_offset_deltas",
        "level_set_method",
        "zero_crossing_level_strategy",
        "min_point_loop_number",
        "min_loop_signifcance",
        "smooth_flag",
        "smooth_factor",
        "skip_calculation_min_winding_distance",
        "skip_unconnected_contour_wire_distance",
        "wire_distance_method",
        "wire_distance_chunk_size",
        "wire_distance_spatial_min_segments",
        "area_perimeter_deletion_ratio",
        "max_allowed_angle_within_coil_track",
        "min_allowed_angle_within_coil_track",
        "tiny_segment_length_percentage",
    }
)
_ROUTING_FIELDS = frozenset(
    {
        "interconnection_cut_width",
        "within_group_cut_width",
        "inter_group_cut_width",
        "interconnection_method",
        "group_interconnection_method",
        "normal_shift_length",
        "normal_shift_smooth_factors",
        "normal_sweep_smooth_factor",
        "force_cut_selection",
        "within_group_cut_candidate_count",
        "min_opening_cut_span",
        "fail_on_route_quality_issues",
        "skip_postprocessing",
        "skip_normal_shift",
        "skip_sweep",
        "make_cylndrical_pcb",
        "contour_only_sweep_mode",
        "pcb_interconnection_method",
        "pcb_spiral_end_shift_factor",
        "track_width_factor",
        "min_inter_group_connector_clearance",
        "suspicious_inter_group_connector_ratio",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "skip_inductance_calculation",
        "inductance_source",
        "low_memory_mode",
        "retain_heavy_fields",
    }
)
_SHEET_FIELDS = frozenset(
    {
        "sf_opt_method",
        "stream_function_cost_function",
        "tikonov_reg_factor",
        "regularization_selection_method",
        "conductor_thickness",
        "specific_conductivity_conductor",
        "conductor_cross_section_width",
        "conductor_cross_section_height",
        "cross_sectional_points",
        "max_current_A",
        "max_wire_current_A",
        "min_wire_distance_m",
        "min_turn_distance",
        "surface_current_density_limit_A_per_m",
        "engineering_contour_step_safety_factor",
        "engineering_contour_max_levels",
        "gradient_hardware_infeasible_policy",
    }
)


_LEVEL_FIELDS = frozenset({"levels", "pot_offset_factor", "contour_level_grid_mode", "contour_level_range_percentile"})


def config_field_stage(name: str, config: CoilGenConfig | None = None) -> PipelineStage | None:
    """Earliest affected stage; unknown/new inputs invalidate all operators."""
    if name in _DISPLAY_FIELDS:
        return None
    if name in _LEVEL_FIELDS:
        return PipelineStage.SHEET if config is None or _level_settings_affect_sheet(config) else PipelineStage.CONTOURS
    if name in {"skip_postprocessing", "contour_only_sweep_mode", "skip_sweep"}:
        # These modes decide whether contour resampling is deferred and whether
        # sparsification/pruning checks candidate solids before route creation.
        return PipelineStage.CONTOURS
    if name in _CONTOUR_FIELDS or name.startswith(("contour_path_", "contour_resample_", "contour_smoothing_")):
        return PipelineStage.CONTOURS
    if name in _ROUTING_FIELDS or name.startswith(("joint_group_routing", "routing_", "inter_group_", "sweep_")):
        if config is None or _contours_check_manufacturing(config):
            return PipelineStage.CONTOURS
        return PipelineStage.ROUTED
    if name in _EVALUATION_FIELDS or name.startswith("gradient_slice_map"):
        return PipelineStage.EVALUATED
    if name in _SHEET_FIELDS or name.startswith(
        (
            "constrained_",
            "regularization_",
            "l_curve_",
            "stream_function_",
            "optimization_solver",
            "turn_spacing_",
            "target_error_",
            "suppression_error_",
        )
    ):
        return PipelineStage.SHEET
    return PipelineStage.PREPARED


def changed_config_fields(previous: CoilGenConfig, current: CoilGenConfig) -> tuple[str, ...]:
    return tuple(
        item.name
        for item in fields(CoilGenConfig)
        if _field_identity(item.name, getattr(previous, item.name))
        != _field_identity(item.name, getattr(current, item.name))
    )


def invalidated_stage(previous: CoilGenConfig, current: CoilGenConfig) -> PipelineStage | None:
    stages = [config_field_stage(name, current) for name in changed_config_fields(previous, current)]
    stages += [config_field_stage(name, previous) for name in changed_config_fields(previous, current)]
    return min((stage for stage in stages if stage is not None), key=lambda stage: stage.index, default=None)


@dataclass
class PipelineState:
    parts: list[CoilPart] = field(default_factory=list)
    target_field: TargetField | None = None
    prepared_diagnostics: dict[str, Any] = field(default_factory=dict)
    combined_mesh: Mesh | None = None
    sf_b_field: np.ndarray | None = None
    regularization_diagnostics: dict[str, Any] = field(default_factory=dict)
    primary_surface_index: int = 0
    result: CoilGenResult | None = None


class PipelineCheckpoint:
    """Owned stage snapshot. Public data access returns detached copies.

    Loading old artifacts is permitted for inspection. Resuming them verifies
    source, numerical settings, and input-content hashes before execution.
    """

    SCHEMA = "coilgen.pipeline-checkpoint/2"

    def __init__(
        self,
        stage: PipelineStage | str,
        state: PipelineState,
        config: CoilGenConfig,
        project_root: Path,
        *,
        fingerprints: dict[str, str] | None = None,
    ):
        self._stage = PipelineStage(stage)
        self._state = deepcopy(state)
        self._config = deepcopy(config)
        self._project_root = Path(project_root).resolve()
        self._fingerprints = dict(fingerprints or stage_fingerprints(config, self._project_root))

    @property
    def stage(self) -> PipelineStage:
        return self._stage

    @property
    def state(self) -> PipelineState:
        """Detached scientific state for inspection of incomplete workflows."""
        return deepcopy(self._state)

    @property
    def parts(self) -> list[CoilPart]:
        return deepcopy(self._state.parts)

    @property
    def target_field(self) -> TargetField | None:
        return deepcopy(self._state.target_field)

    @property
    def sf_b_field(self) -> np.ndarray | None:
        return deepcopy(self._state.sf_b_field)

    @property
    def result(self) -> CoilGenResult | None:
        return deepcopy(self._state.result)

    @property
    def config(self) -> CoilGenConfig:
        return deepcopy(self._config)

    def scientific_arrays(self, *, max_bytes: int | None = None) -> dict[str, np.ndarray]:
        """Detach only display data, without copying sensitivities or solver state."""
        from .scientific_arrays import scientific_arrays

        state = self._state
        arrays = scientific_arrays(state.parts, state.target_field, state.combined_mesh, state.sf_b_field)
        if max_bytes is not None and sum(np.asarray(value).nbytes for value in arrays.values()) > max_bytes:
            raise ValueError("Stage preview arrays exceed the permitted size.")
        return {key: np.array(value, copy=True) for key, value in arrays.items()}

    @property
    def diagnostics(self) -> dict[str, Any]:
        return deepcopy(self._state.regularization_diagnostics or self._state.prepared_diagnostics)

    def assert_compatible(self, config: CoilGenConfig, project_root: Path, *, fingerprints=None) -> None:
        self._validate_state()
        current = stage_fingerprints(config, project_root) if fingerprints is None else fingerprints
        key = self.stage.value
        if self._fingerprints.get(key) != current[key]:
            try:
                changed = changed_config_fields(self._config, config)
            except KeyError:
                # A same-process registry may not exist after reopening. The
                # fingerprint mismatch still prohibits reuse; its explanatory
                # text must not depend on looking up that old registry.
                changed = ()
            detail = ", ".join(changed) or "source code or input asset contents"
            raise ValueError(f"Cannot reuse {key} checkpoint: changed {detail}; recompute affected stages.")

    def save(self, path: str | Path) -> Path:
        """Atomically save one portable archive containing JSON and NPZ only."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        encoded = _encode({"state": self._state, "config": self._config}, arrays)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        numerical = buffer.getvalue()
        manifest = {
            "schema": self.SCHEMA,
            "stage": self.stage.value,
            "project_root": str(self._project_root),
            "fingerprints": self._fingerprints,
            "payload": encoded,
            "arrays_sha256": sha256(numerical).hexdigest(),
        }
        manifest["payload_sha256"] = sha256(
            json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as output:
                with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False, separators=(",", ":")))
                    archive.writestr("arrays.npz", numerical)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path, *, max_bytes: int = 2 * 1024**3) -> "PipelineCheckpoint":
        """Read allowlisted data without executing constructors selected by files."""
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if sorted(names) != ["arrays.npz", "manifest.json"]:
                raise ValueError("Checkpoint archive requires exactly manifest.json and arrays.npz.")
            if archive.getinfo("manifest.json").file_size > 64 * 1024**2:
                raise ValueError("Checkpoint manifest exceeds the permitted size.")
            if sum(item.file_size for item in archive.infolist()) > max_bytes:
                raise ValueError("Checkpoint archive exceeds the permitted size.")
            manifest = json.loads(archive.read("manifest.json"))
            numerical = archive.read("arrays.npz")
        if manifest.get("schema") != cls.SCHEMA:
            raise ValueError("Unsupported pipeline checkpoint schema.")
        # Stage and inputs are scientific evidence too. Numerical-array hashes
        # alone would leave a changed stage/config manifest undetected.
        claimed = manifest.pop("payload_sha256", None)
        actual = sha256(
            json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if claimed != actual:
            raise ValueError("Checkpoint manifest failed its integrity check.")
        if sha256(numerical).hexdigest() != manifest.get("arrays_sha256"):
            raise ValueError("Checkpoint numerical data failed its integrity check.")
        _validate_npz(numerical, max_bytes=max_bytes)
        with np.load(io.BytesIO(numerical), allow_pickle=False) as archive:
            payload = _decode(manifest["payload"], archive)
        if not isinstance(payload, dict) or not isinstance(payload.get("state"), PipelineState):
            raise ValueError("Checkpoint requires a typed pipeline state.")
        if not isinstance(payload.get("config"), CoilGenConfig):
            raise ValueError("Checkpoint requires a typed CoilGenConfig.")
        checkpoint = cls(
            manifest["stage"],
            payload["state"],
            payload["config"],
            Path(manifest["project_root"]),
            fingerprints=manifest["fingerprints"],
        )
        checkpoint._validate_state()
        return checkpoint

    def _validate_state(self) -> None:
        state = self._state
        if (
            not isinstance(state.parts, list)
            or not state.parts
            or not all(isinstance(part, CoilPart) for part in state.parts)
        ):
            raise ValueError("Checkpoint is missing typed prepared geometry.")
        if not isinstance(state.target_field, TargetField):
            raise ValueError("Checkpoint is missing typed target data.")
        if self.stage.index >= PipelineStage.SHEET.index:
            if not isinstance(state.combined_mesh, Mesh) or state.sf_b_field is None:
                raise ValueError("Solved checkpoint is missing its combined mesh or stream-function field.")
            if np.asarray(state.sf_b_field).shape != np.asarray(state.target_field.b).shape:
                raise ValueError("Checkpoint sheet field does not match its target sample shape.")
            if any(
                part.stream_function is None or len(part.stream_function) != part.mesh.n_vertices
                for part in state.parts
            ):
                raise ValueError("Checkpoint stream functions do not match prepared mesh vertices.")
        if self.stage is PipelineStage.EVALUATED and not isinstance(state.result, CoilGenResult):
            raise ValueError("Evaluated checkpoint is missing its typed result.")


def stage_fingerprints(config: CoilGenConfig, project_root: Path) -> dict[str, str]:
    """Numerical-source, solver-version and input-content changes prevent reuse."""
    source = numerical_source_digest(Path(__file__).parent)
    assets = _asset_fingerprints(config, Path(project_root))
    environment = {"python": list(sys.version_info[:3]), "byteorder": sys.byteorder}
    for name in ("numpy", "scipy", "cvxpy", "clarabel", "scs", "ecos", "osqp", "numba", "llvmlite"):
        try:
            environment[name] = version(name)
        except PackageNotFoundError:
            environment[name] = None
    output = {}
    for stage in PipelineStage:
        inputs = {
            item.name: assets.get(item.name, getattr(config, item.name))
            for item in fields(CoilGenConfig)
            if item.name != "geometry_source_path"
            and not (item.name == "target_region_geometry" and config.target_region_geometry is None)
            and (dependency := config_field_stage(item.name, config)) is not None
            and dependency.index <= stage.index
        }
        output[stage.value] = _identity(
            {"source": source, "environment": environment, "inputs": inputs, "assets": assets}
        )
    return output


def _asset_fingerprints(config: CoilGenConfig, project_root: Path) -> dict[str, str]:
    from .object_registry import (
        get_in_memory_mesh,
        get_in_memory_target_field,
        is_in_memory_mesh_key,
        is_in_memory_target_key,
    )

    assets = {}
    for item in fields(CoilGenConfig):
        if not item.name.endswith("_file"):
            continue
        value = str(getattr(config, item.name))
        if is_in_memory_mesh_key(value):
            assets[item.name] = _identity(get_in_memory_mesh(value))
        elif is_in_memory_target_key(value):
            assets[item.name] = _identity(get_in_memory_target_field(value))
        elif value.strip().lower() not in {"", "none"}:
            path = Path(value)
            candidates = (
                (config.resolved_geometry_source_path(project_root) / path, project_root / path, path)
                if "mesh" in item.name
                else (path, project_root / path)
            )
            for candidate in candidates:
                if candidate.is_file():
                    assets[item.name] = file_digest(candidate)
                    break
    return assets


def _field_identity(name: str, value: Any) -> str:
    if name.endswith("_file"):
        from .object_registry import (
            get_in_memory_mesh,
            get_in_memory_target_field,
            is_in_memory_mesh_key,
            is_in_memory_target_key,
        )

        if is_in_memory_mesh_key(value):
            return _identity(get_in_memory_mesh(value))
        if is_in_memory_target_key(value):
            return _identity(get_in_memory_target_field(value))
        if Path(str(value)).is_file():
            return file_digest(Path(str(value)))
    return _identity(value)


def _level_settings_affect_sheet(config: CoilGenConfig) -> bool:
    # Coupling includes preoptimized-sheet hardware rescaling and automatic
    # factor selection. Reusing a sheet is permitted only when every linked
    # current/spacing/contour-step guard is disabled.
    controls = (
        "max_current_A",
        "max_wire_current_A",
        "min_wire_distance_m",
        "min_turn_distance",
        "surface_current_density_limit_A_per_m",
        "turn_spacing_regularization_factor",
        "regularization_auto_range_max_contour_step_A",
    )
    return any(float(getattr(config, name, 0.0)) > 0.0 for name in controls)


def _contours_check_manufacturing(config: CoilGenConfig) -> bool:
    return bool(
        config.contour_only_sweep_mode
        and not config.skip_sweep
        and any(
            str(getattr(config, name, "off")).strip().lower() != "off"
            for name in (
                "contour_sparsification_mode",
                "contour_sparsification_pruning_mode",
                "contour_sparsification_post_pruning_merge_mode",
            )
        )
    )


def _identity(value: Any) -> str:
    arrays: dict[str, np.ndarray] = {}
    encoded = _encode(value, arrays)
    digest = sha256(json.dumps(encoded, allow_nan=False, sort_keys=True, separators=(",", ":")).encode())
    for key, array in arrays.items():
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _allowed_types() -> dict[str, type]:
    from . import models
    from .design_context import CoilApplication, CoilDesignContext, EvaluationFeature
    from .surface_topology import FaceAdjacency, FaceCut, VirtualCutGraph, BoundaryComponent, LoopTopologySignature

    types = [
        PipelineState,
        CoilGenConfig,
        CoilApplication,
        CoilDesignContext,
        EvaluationFeature,
        FaceAdjacency,
        FaceCut,
        VirtualCutGraph,
        BoundaryComponent,
        LoopTopologySignature,
    ]
    types += [
        value
        for value in vars(models).values()
        if isinstance(value, type)
        and value.__module__ == models.__name__
        and (is_dataclass(value) or issubclass(value, Enum))
    ]
    return {value.__name__: value for value in types}


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "value": _encode(value.value, arrays)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)
    if isinstance(value, float):
        return value if math.isfinite(value) else {"$float": str(value)}
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufcSU" or value.ndim > 8:
            raise TypeError("Checkpoint arrays require plain numerical/string dtypes with at most eight axes.")
        key = f"a{len(arrays):07d}"
        arrays[key] = value
        return {"$array": key, "dtype": value.dtype.str, "shape": list(value.shape)}
    if is_dataclass(value) and not isinstance(value, type):
        if type(value).__name__ not in _allowed_types() or _allowed_types()[type(value).__name__] is not type(value):
            raise TypeError(f"Unsupported checkpoint data class {type(value).__name__}.")
        return {
            "$class": type(value).__name__,
            "fields": {item.name: _encode(getattr(value, item.name), arrays) for item in fields(value)},
        }
    if isinstance(value, dict):
        return {"$dict": [[_encode(key, arrays), _encode(item, arrays)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"$tuple" if isinstance(value, tuple) else "$list": [_encode(item, arrays) for item in value]}
    raise TypeError(f"Unsupported checkpoint value {type(value).__name__}.")


def _decode(value: Any, arrays: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, dict):
        raise ValueError("Malformed checkpoint value.")
    if "$array" in value:
        array = arrays[value["$array"]]
        if array.dtype.str != value["dtype"] or list(array.shape) != value["shape"]:
            raise ValueError("Checkpoint array shape/dtype disagrees with its manifest.")
        return array.copy()
    if "$class" in value:
        allowed = _allowed_types()
        cls = allowed.get(value["$class"])
        if cls is None or not is_dataclass(cls):
            raise ValueError("Checkpoint data class is not allowlisted.")
        attributes = value["fields"]
        if set(attributes) != {item.name for item in fields(cls)}:
            raise ValueError("Checkpoint data-class fields do not match this schema.")
        return cls(**{name: _decode(item, arrays) for name, item in attributes.items()})
    if "$enum" in value:
        cls = _allowed_types().get(value["$enum"])
        if cls is None or not issubclass(cls, Enum):
            raise ValueError("Checkpoint enum is not allowlisted.")
        return cls(_decode(value["value"], arrays))
    if "$dict" in value:
        return {_decode(key, arrays): _decode(item, arrays) for key, item in value["$dict"]}
    if "$list" in value:
        return [_decode(item, arrays) for item in value["$list"]]
    if "$tuple" in value:
        return tuple(_decode(item, arrays) for item in value["$tuple"])
    if "$path" in value:
        return Path(value["$path"])
    if "$float" in value and value["$float"] in {"nan", "inf", "-inf"}:
        return float(value["$float"])
    raise ValueError("Unknown checkpoint value tag.")


def _validate_npz(data: bytes, *, max_bytes: int) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(item.file_size for item in archive.infolist()) > max_bytes:
            raise ValueError("Checkpoint arrays exceed the permitted expanded size.")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Checkpoint contains duplicate arrays.")
        for item in archive.infolist():
            if not item.filename.endswith(".npy") or "/" in item.filename or "\\" in item.filename:
                raise ValueError("Checkpoint contains an invalid array name.")
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported checkpoint array format.")
                if dtype.kind not in "biufcSU" or len(shape) > 8:
                    raise ValueError("Checkpoint requires plain data arrays; object arrays are prohibited.")
                count = math.prod(shape)
                if count * dtype.itemsize > max_bytes or count * dtype.itemsize > item.file_size:
                    raise ValueError("Checkpoint array header exceeds its permitted size.")
