"""Trace-level contour-dictionary strategy for multichannel cohort shims.

This module is intentionally independent of the continuous stream-function POD
and adaptive surface-sectorization implementations.  It treats previously
designed *physical source contours* as an actuator dictionary, evaluates their
fields per ampere, and assigns complete contour families to independently
driven channels.  Consequently no contour is shared between channels.

The assignment is an electromagnetic topology result.  It does not claim that
inter-family leads, terminals, and crossovers have already been routed on one
layer; that remains a separate hardware-compilation validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import json
from math import factorial
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cohort_shim import (
    ChannelCurrentLimits,
    CohortFitOptions,
    CohortFitResult,
    PopulationShimDataset,
    RigidTransform,
    fit_cohort_currents,
)
from .models import CoilGenResult, CoilPart
from .physics import biot_savart_wire_path
from .progress import ProgressReporter, progress_scope


ProgressCallback = Callable[[str], None]
_SCHEMA_VERSION = 1


def _emit(callback: ProgressCallback | None, message: str) -> None:
    ProgressReporter(label="Contour cohort").report(None, message)
    if callback is not None:
        callback(message)


def _readonly(values: Any, dtype: Any = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.flags.writeable:
        array = array.copy()
        array.setflags(write=False)
    return array


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class PhysicalContour:
    """One oriented physical closed-loop centerline in metres."""

    points_m: np.ndarray
    source_part_index: int = -1
    source_contour_index: int = -1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _readonly(self.points_m)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("A physical contour must contain at least three 3-D points.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Physical contour points must be finite.")
        scale = max(float(np.ptp(points, axis=0).max()), 1.0)
        if not np.allclose(points[0], points[-1], atol=1.0e-10 * scale, rtol=0.0):
            raise ValueError("A physical contour must be explicitly closed (first point == last point).")
        if float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) <= 0.0:
            raise ValueError("A physical contour must have positive length.")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "source_part_index", int(self.source_part_index))
        object.__setattr__(self, "source_contour_index", int(self.source_contour_index))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.points_m, axis=0), axis=1).sum())


@dataclass(frozen=True, slots=True)
class ContourFamily:
    """A source-derived set of contours kept together during assignment."""

    family_id: str
    contours: tuple[PhysicalContour, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family_id = str(self.family_id).strip()
        contours = tuple(self.contours)
        if not family_id:
            raise ValueError("Contour-family IDs must be non-empty.")
        if not contours:
            raise ValueError(f"Contour family {family_id!r} must contain at least one contour.")
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "contours", contours)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def length_m(self) -> float:
        return float(sum(contour.length_m for contour in self.contours))


@dataclass(frozen=True, slots=True)
class ContourDictionary:
    """Oriented source contours in either mask-CAD or scanner coordinates."""

    families: tuple[ContourFamily, ...]
    coordinate_frame: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        families = tuple(self.families)
        frame = str(self.coordinate_frame).strip().lower()
        if not families:
            raise ValueError("A contour dictionary must contain at least one family.")
        if len({family.family_id for family in families}) != len(families):
            raise ValueError("Contour-family IDs must be unique.")
        if frame not in {"mask_cad", "scanner"}:
            raise ValueError("coordinate_frame must be either 'mask_cad' or 'scanner'.")
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def family_count(self) -> int:
        return len(self.families)

    @property
    def contour_count(self) -> int:
        return sum(len(family.contours) for family in self.families)


@dataclass(frozen=True, slots=True)
class ContourChannelOptions:
    """Controls exact family assignment and bounded cohort fitting."""

    channel_count: int = 4
    current_limit_A: float = 2.0
    fit_options: CohortFitOptions = field(default_factory=CohortFitOptions)
    scenario_weighting: str = "subject_balanced"
    scenario_weights: Mapping[str, float] | None = None
    optimize_family_polarity: bool = True
    maximum_candidate_count: int = 250_000
    maximum_normalized_field_condition_number: float = float("inf")
    channel_resistance_ohm: tuple[float, ...] | None = None
    total_power_limit_W: float | None = None
    biot_savart_chunk_size: int = 512
    worker_count: int = 1
    parallel_min_work_size: int = 50_000
    parallel_max_chunk_memory_mb: float = 64.0

    def __post_init__(self) -> None:
        if self.channel_count < 1:
            raise ValueError("channel_count must be positive.")
        if not np.isfinite(self.current_limit_A) or self.current_limit_A <= 0.0:
            raise ValueError("current_limit_A must be finite and positive.")
        weighting = str(self.scenario_weighting).strip().lower().replace("-", "_")
        if weighting not in {"subject_balanced", "uniform_scenario"}:
            raise ValueError("scenario_weighting must be 'subject_balanced' or 'uniform_scenario'.")
        explicit = None
        if self.scenario_weights is not None:
            explicit = {str(key): float(value) for key, value in self.scenario_weights.items()}
            if any(not np.isfinite(value) or value <= 0.0 for value in explicit.values()):
                raise ValueError("Explicit scenario weights must be finite and positive.")
        if self.maximum_candidate_count < 1:
            raise ValueError("maximum_candidate_count must be positive.")
        condition = float(self.maximum_normalized_field_condition_number)
        if condition <= 1.0 or np.isnan(condition):
            raise ValueError("maximum_normalized_field_condition_number must exceed one.")
        if self.biot_savart_chunk_size < 1 or self.worker_count < 1:
            raise ValueError("Biot-Savart chunk size and worker count must be positive.")
        if self.parallel_min_work_size < 0 or self.parallel_max_chunk_memory_mb <= 0.0:
            raise ValueError("Biot-Savart parallel thresholds must be non-negative/positive.")
        resistance = self.channel_resistance_ohm
        if resistance is not None:
            resistance = tuple(float(value) for value in resistance)
            if len(resistance) != self.channel_count or any(
                not np.isfinite(value) or value <= 0.0 for value in resistance
            ):
                raise ValueError("channel_resistance_ohm must contain one positive value per channel.")
        if self.total_power_limit_W is not None:
            if resistance is None:
                raise ValueError("A total power limit requires channel_resistance_ohm.")
            if not np.isfinite(self.total_power_limit_W) or self.total_power_limit_W <= 0.0:
                raise ValueError("total_power_limit_W must be finite and positive.")
        object.__setattr__(self, "scenario_weighting", weighting)
        object.__setattr__(self, "scenario_weights", explicit)
        object.__setattr__(self, "channel_resistance_ohm", resistance)


@dataclass(frozen=True, slots=True)
class ContourControllabilityDiagnostics:
    family_singular_values: np.ndarray
    family_numerical_rank: int
    family_normalized_field_condition_number: float
    family_maximum_normalized_field_correlation: float
    channel_singular_values: np.ndarray
    channel_numerical_rank: int
    channel_normalized_field_condition_number: float
    channel_maximum_normalized_field_correlation: float
    channel_field_rms_T_per_A: np.ndarray


@dataclass(frozen=True, slots=True)
class ContourChannelDesignResult:
    """Best exact family partition found under the configured current bounds."""

    dictionary: ContourDictionary
    family_to_channel: np.ndarray
    family_polarity: np.ndarray
    channel_fields_by_scenario: Mapping[str, np.ndarray]
    fit: CohortFitResult
    current_limits: ChannelCurrentLimits
    scenario_weights: np.ndarray
    diagnostics: ContourControllabilityDiagnostics
    partition_count: int
    candidate_count: int
    feasible_candidate_count: int
    objective_T2: float
    routing_validated: bool = False

    def __post_init__(self) -> None:
        assignment = _readonly(self.family_to_channel, np.int64).reshape(-1)
        polarity = _readonly(self.family_polarity, np.int8).reshape(-1)
        if len(assignment) != self.dictionary.family_count or len(polarity) != len(assignment):
            raise ValueError("Assignment and polarity must match the contour-family count.")
        if set(np.unique(assignment)) != set(range(self.current_limits.channel_count)):
            raise ValueError("Every channel must own at least one contour family.")
        if not np.all(np.isin(polarity, (-1, 1))):
            raise ValueError("Family polarity entries must be -1 or +1.")
        object.__setattr__(self, "family_to_channel", assignment)
        object.__setattr__(self, "family_polarity", polarity)
        object.__setattr__(self, "scenario_weights", _readonly(self.scenario_weights))
        object.__setattr__(
            self,
            "channel_fields_by_scenario",
            {key: _readonly(value) for key, value in self.channel_fields_by_scenario.items()},
        )

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": "contour_dictionary",
            "family_count": self.dictionary.family_count,
            "physical_source_contour_count": self.dictionary.contour_count,
            "channel_count": self.current_limits.channel_count,
            "partition_count": self.partition_count,
            "candidate_count": self.candidate_count,
            "feasible_candidate_count": self.feasible_candidate_count,
            "objective_T2": self.objective_T2,
            "family_to_channel": (self.family_to_channel + 1).tolist(),
            "family_polarity": self.family_polarity.tolist(),
            "channel_numerical_rank": self.diagnostics.channel_numerical_rank,
            "channel_normalized_field_condition_number": (self.diagnostics.channel_normalized_field_condition_number),
            "channel_maximum_normalized_field_correlation": (
                self.diagnostics.channel_maximum_normalized_field_correlation
            ),
            "channel_field_rms_T_per_A": self.diagnostics.channel_field_rms_T_per_A.tolist(),
            "routing_validated": self.routing_validated,
            "single_layer_realization_claimed": False,
            "fit": self.fit.summary(),
        }


def contour_dictionary_from_coilgen_result(
    result: CoilGenResult,
    *,
    coordinate_frame: str,
    coordinate_units: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContourDictionary:
    """Build families from CoilGen source contours, never weighted representatives.

    When contour sparsification was used, each final weighted geometry defines a
    family but the returned family contains its original separated source
    contours.  With no sparsification, every contour is its own family.
    """

    units = coordinate_units or str(result.combined_mesh.metadata.get("coordinate_units", ""))
    units = units.strip().lower()
    scale = {"m": 1.0, "mm": 1.0e-3}.get(units)
    if scale is None:
        raise ValueError("coordinate_units must explicitly be 'm' or 'mm'.")
    families: list[ContourFamily] = []
    for part_index, part in enumerate(result.coil_parts):
        families.extend(_families_from_part(part, part_index, scale))
    dictionary_metadata = dict(metadata or {})
    dictionary_metadata.update(
        {
            "source": "CoilGenResult physical source contours",
            "source_coordinate_units": units,
            "weighted_contour_representatives_used": False,
            "physical_routing_validated": False,
        }
    )
    return ContourDictionary(tuple(families), coordinate_frame, dictionary_metadata)


def _families_from_part(part: CoilPart, part_index: int, scale: float) -> list[ContourFamily]:
    provenance = part.contour_sparsification
    if provenance is not None:
        source = tuple(provenance.source_contours)
        groups = tuple(provenance.final_source_contour_indices)
        if not source or not groups:
            raise ValueError("Contour sparsification provenance is incomplete.")
    else:
        source = tuple(part.contour_lines)
        groups = tuple((index,) for index in range(len(source)))
    families: list[ContourFamily] = []
    used: set[int] = set()
    for local_family_index, source_indices in enumerate(groups):
        indices = tuple(int(index) for index in source_indices)
        if not indices or any(index < 0 or index >= len(source) for index in indices):
            raise ValueError("Contour-family provenance references an invalid source contour.")
        if used.intersection(indices):
            raise ValueError("A source contour appears in more than one family.")
        used.update(indices)
        contours = tuple(
            PhysicalContour(
                points_m=np.asarray(source[index].v, dtype=float) * scale,
                source_part_index=part_index,
                source_contour_index=index,
                metadata={"potential": float(source[index].potential)},
            )
            for index in indices
        )
        families.append(
            ContourFamily(
                family_id=f"part_{part_index + 1:02d}_family_{local_family_index + 1:03d}",
                contours=contours,
                metadata={"source_contour_indices": indices},
            )
        )
    if used != set(range(len(source))):
        missing = sorted(set(range(len(source))) - used)
        raise ValueError(f"Contour-family provenance omits source contours {missing}.")
    return families


def save_contour_dictionary(path: str | Path, dictionary: ContourDictionary) -> Path:
    """Save a ragged contour dictionary without Python pickle objects."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points: list[np.ndarray] = []
    offsets = [0]
    family_indices: list[int] = []
    source_parts: list[int] = []
    source_contours: list[int] = []
    contour_metadata_json: list[str] = []
    for family_index, family in enumerate(dictionary.families):
        for contour in family.contours:
            points.append(np.asarray(contour.points_m, dtype=float))
            offsets.append(offsets[-1] + len(contour.points_m))
            family_indices.append(family_index)
            source_parts.append(contour.source_part_index)
            source_contours.append(contour.source_contour_index)
            contour_metadata_json.append(json.dumps(_json_ready(contour.metadata), sort_keys=True))
    np.savez_compressed(
        output,
        schema_version=np.asarray(_SCHEMA_VERSION, dtype=np.int64),
        coordinate_frame=np.asarray(dictionary.coordinate_frame),
        coordinate_units=np.asarray("m"),
        family_ids=np.asarray([family.family_id for family in dictionary.families], dtype=str),
        family_metadata_json=np.asarray(
            [json.dumps(_json_ready(family.metadata), sort_keys=True) for family in dictionary.families],
            dtype=str,
        ),
        contour_family_indices=np.asarray(family_indices, dtype=np.int64),
        contour_point_offsets=np.asarray(offsets, dtype=np.int64),
        contour_points_m=np.vstack(points),
        source_part_indices=np.asarray(source_parts, dtype=np.int64),
        source_contour_indices=np.asarray(source_contours, dtype=np.int64),
        contour_metadata_json=np.asarray(contour_metadata_json, dtype=str),
        metadata_json=np.asarray(json.dumps(_json_ready(dictionary.metadata), sort_keys=True)),
    )
    return output


def load_contour_dictionary(path: str | Path) -> ContourDictionary:
    """Load and validate a contour dictionary written by this module."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        version = int(np.asarray(payload["schema_version"]).reshape(()))
        if version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported contour-dictionary schema version {version}.")
        units = str(np.asarray(payload["coordinate_units"]).reshape(()))
        if units != "m":
            raise ValueError("Contour-dictionary coordinates must be stored in metres.")
        frame = str(np.asarray(payload["coordinate_frame"]).reshape(()))
        family_ids = np.asarray(payload["family_ids"], dtype=str).reshape(-1)
        family_metadata_json = np.asarray(payload["family_metadata_json"], dtype=str).reshape(-1)
        family_indices = np.asarray(payload["contour_family_indices"], dtype=np.int64).reshape(-1)
        offsets = np.asarray(payload["contour_point_offsets"], dtype=np.int64).reshape(-1)
        points = np.asarray(payload["contour_points_m"], dtype=float)
        source_parts = np.asarray(payload["source_part_indices"], dtype=np.int64).reshape(-1)
        source_contours = np.asarray(payload["source_contour_indices"], dtype=np.int64).reshape(-1)
        contour_metadata_json = np.asarray(payload["contour_metadata_json"], dtype=str).reshape(-1)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(())))
    contour_count = len(family_indices)
    if len(offsets) != contour_count + 1 or offsets[0] != 0 or offsets[-1] != len(points):
        raise ValueError("Contour dictionary has inconsistent ragged point offsets.")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("Contour dictionary point offsets must be monotonic.")
    if (
        len(source_parts) != contour_count
        or len(source_contours) != contour_count
        or len(contour_metadata_json) != contour_count
    ):
        raise ValueError("Contour dictionary provenance arrays have inconsistent lengths.")
    if len(family_metadata_json) != len(family_ids):
        raise ValueError("Contour dictionary family metadata has inconsistent length.")
    grouped: list[list[PhysicalContour]] = [[] for _ in family_ids]
    for contour_index, family_index in enumerate(family_indices):
        if family_index < 0 or family_index >= len(family_ids):
            raise ValueError("Contour dictionary references an invalid family index.")
        grouped[int(family_index)].append(
            PhysicalContour(
                points[offsets[contour_index] : offsets[contour_index + 1]],
                int(source_parts[contour_index]),
                int(source_contours[contour_index]),
                json.loads(contour_metadata_json[contour_index]),
            )
        )
    families = tuple(
        ContourFamily(
            family_id,
            tuple(contours),
            json.loads(family_metadata_json[index]),
        )
        for index, (family_id, contours) in enumerate(zip(family_ids, grouped, strict=True))
    )
    return ContourDictionary(families, frame, metadata)


def evaluate_contour_family_fields(
    dictionary: ContourDictionary,
    dataset: PopulationShimDataset,
    *,
    scanner_from_source: RigidTransform | None = None,
    chunk_size: int = 512,
    worker_count: int = 1,
    min_work_size: int = 50_000,
    max_chunk_memory_mb: float = 64.0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray]:
    """Return scalar B0 fields per ampere for every family and scenario."""

    if dictionary.coordinate_frame == "mask_cad":
        if scanner_from_source is None:
            raise ValueError("A mask-CAD contour dictionary requires scanner_from_source.")
        if scanner_from_source.source_frame != "mask_cad" or scanner_from_source.target_frame != "scanner":
            raise ValueError("scanner_from_source must map 'mask_cad' to 'scanner'.")
        for scenario in dataset.scenarios:
            registration = scenario.scanner_from_mask
            if (
                registration.source_frame != scanner_from_source.source_frame
                or registration.target_frame != scanner_from_source.target_frame
                or not np.allclose(
                    registration.matrix_m,
                    scanner_from_source.matrix_m,
                    atol=1.0e-10,
                    rtol=0.0,
                )
            ):
                raise ValueError(
                    "A single mask-CAD contour dictionary evaluation requires all scenarios "
                    "to use the supplied mask-to-scanner registration."
                )
        transform = scanner_from_source
    else:
        if scanner_from_source is not None and (
            scanner_from_source.source_frame != "scanner"
            or scanner_from_source.target_frame != "scanner"
            or not np.allclose(scanner_from_source.matrix_m, np.eye(4), atol=1.0e-12, rtol=0.0)
        ):
            raise ValueError("Scanner-frame contours may only use an identity scanner transform.")
        transform = None

    contour_points = tuple(
        tuple(
            transform.apply_points(contour.points_m) if transform is not None else contour.points_m
            for contour in family.contours
        )
        for family in dictionary.families
    )
    fields: dict[str, np.ndarray] = {}
    cache: dict[tuple[bytes, tuple[float, float, float]], np.ndarray] = {}
    total = len(dataset.scenarios) * dictionary.contour_count
    completed = 0
    reporter = ProgressReporter(label="Contour dictionary fields")
    started = perf_counter()
    _emit(
        progress_callback,
        f"Contour field cache: {dictionary.family_count} families, "
        f"{dictionary.contour_count} physical source contours, "
        f"{len(dataset.scenarios)} scenarios ({total} contour/scenario references).",
    )
    for scenario_index, scenario in enumerate(dataset.scenarios):
        coords = np.ascontiguousarray(scenario.coords_scanner_m, dtype=float)
        direction = tuple(float(value) for value in scenario.b0_direction_scanner)
        cache_key = (coords.tobytes(), direction)
        if cache_key in cache:
            fields[scenario.scenario_id] = cache[cache_key]
            completed += dictionary.contour_count
            _emit(
                progress_callback,
                f"Contour fields {scenario_index + 1}/{len(dataset.scenarios)}: "
                f"{scenario.scenario_id} reused identical sampling cache.",
            )
            reporter.report(100 * completed / total, f"Reused verified sampling cache: {completed}/{total} contours")
            continue
        family_matrix = np.zeros((scenario.sample_count, dictionary.family_count), dtype=float)
        b0 = np.asarray(direction, dtype=float)
        for family_index, family_paths in enumerate(contour_points):
            vector_field = np.zeros((scenario.sample_count, 3), dtype=float)
            for path in family_paths:
                with progress_scope(100 * completed / total, 100 * (completed + 1) / total):
                    vector_field += biot_savart_wire_path(
                        path,
                        coords,
                        chunk_size=chunk_size,
                        worker_count=worker_count,
                        min_work_size=min_work_size,
                        max_chunk_memory_mb=max_chunk_memory_mb,
                    )
                completed += 1
            family_matrix[:, family_index] = vector_field @ b0
        immutable = _readonly(family_matrix)
        cache[cache_key] = immutable
        fields[scenario.scenario_id] = immutable
        _emit(
            progress_callback,
            f"Contour fields {scenario_index + 1}/{len(dataset.scenarios)} complete: "
            f"{scenario.scenario_id}; {completed}/{total} references in {perf_counter() - started:.2f} s.",
        )
        reporter.report(100 * completed / total, f"Contour/scenario references evaluated: {completed}/{total}")
    return fields


def design_contour_dictionary_channels(
    dataset: PopulationShimDataset,
    dictionary: ContourDictionary,
    options: ContourChannelOptions | None = None,
    *,
    scanner_from_source: RigidTransform | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ContourChannelDesignResult:
    """Assign complete contour families to channels by exact canonical search."""

    settings = options or ContourChannelOptions()
    if dictionary.family_count < settings.channel_count:
        raise ValueError("The contour dictionary needs at least one family per channel.")
    # Count before evaluating fields or enumerating partitions: even a moderate
    # dictionary can have an astronomically large exact assignment space.
    partition_count = _stirling_partition_count(dictionary.family_count, settings.channel_count)
    polarity_count = 2 ** (dictionary.family_count - settings.channel_count) if settings.optimize_family_polarity else 1
    asymmetric_power_channels = bool(
        settings.total_power_limit_W is not None
        and settings.channel_resistance_ohm is not None
        and not np.allclose(
            settings.channel_resistance_ohm,
            settings.channel_resistance_ohm[0],
            atol=0.0,
            rtol=1.0e-12,
        )
    )
    labeling_count = factorial(settings.channel_count) if asymmetric_power_channels else 1
    candidate_count = partition_count * polarity_count * labeling_count
    if candidate_count > settings.maximum_candidate_count:
        raise ValueError(
            f"Exact contour assignment would evaluate {candidate_count} candidates, exceeding "
            f"maximum_candidate_count={settings.maximum_candidate_count}. Reduce the family count, "
            "pre-cluster families, or raise the explicit limit."
        )
    scenario_weights = _design_scenario_weights(dataset, settings)
    with progress_scope(0, 50):
        family_fields = evaluate_contour_family_fields(
            dictionary,
            dataset,
            scanner_from_source=scanner_from_source,
            chunk_size=settings.biot_savart_chunk_size,
            worker_count=settings.worker_count,
            min_work_size=settings.parallel_min_work_size,
            max_chunk_memory_mb=settings.parallel_max_chunk_memory_mb,
            progress_callback=progress_callback,
        )
    family_stack = _weighted_field_stack(dataset, family_fields, scenario_weights, center=True)
    family_metrics = _field_metrics(family_stack)
    if family_metrics[1] < settings.channel_count:
        raise ValueError(
            "The centered contour-family field dictionary has numerical rank "
            f"{family_metrics[1]}, below the requested {settings.channel_count} channels."
        )

    _emit(
        progress_callback,
        f"Contour assignment: exact search over {partition_count} unlabeled partitions and "
        f"{polarity_count} relative-polarity states"
        + (f", including {labeling_count} electrical-channel labelings" if labeling_count > 1 else "")
        + f" ({candidate_count} candidates).",
    )
    limits = ChannelCurrentLimits.symmetric(
        settings.channel_count,
        settings.current_limit_A,
        resistance_ohm=settings.channel_resistance_ohm,
        total_power_limit_W=settings.total_power_limit_W,
    )
    best: tuple[tuple[float, float, float, tuple[int, ...], tuple[int, ...]], Any] | None = None
    feasible_count = 0
    completed = 0
    reporter = ProgressReporter(label="Contour dictionary assignment")
    report_interval = max(candidate_count // 20, 1)
    for canonical_partition in _canonical_surjections(dictionary.family_count, settings.channel_count):
        for polarity in _relative_polarities(
            canonical_partition,
            settings.optimize_family_polarity,
        ):
            channel_labelings = (
                permutations(range(settings.channel_count))
                if asymmetric_power_channels
                else (tuple(range(settings.channel_count)),)
            )
            for labeling in channel_labelings:
                partition = tuple(labeling[channel] for channel in canonical_partition)
                completed += 1
                channel_fields = _combine_family_fields(
                    family_fields,
                    np.asarray(partition, dtype=np.int64),
                    np.asarray(polarity, dtype=float),
                    settings.channel_count,
                )
                channel_stack = _weighted_field_stack(
                    dataset,
                    channel_fields,
                    scenario_weights,
                    center=True,
                )
                channel_metrics = _field_metrics(channel_stack)
                condition = channel_metrics[2]
                if channel_metrics[1] < settings.channel_count or (
                    np.isfinite(settings.maximum_normalized_field_condition_number)
                    and condition > settings.maximum_normalized_field_condition_number
                ):
                    if completed % report_interval == 0 or completed == candidate_count:
                        _emit(
                            progress_callback,
                            f"Contour assignment progress: {completed}/{candidate_count} candidates.",
                        )
                    reporter.report(
                        50 + 50 * completed / candidate_count, f"Candidates checked: {completed}/{candidate_count}"
                    )
                    continue
                feasible_count += 1
                with progress_scope(50 + 50 * (completed - 1) / candidate_count, 50 + 50 * completed / candidate_count):
                    fit = fit_cohort_currents(dataset, channel_fields, limits, options=settings.fit_options)
                objective = float(scenario_weights @ np.square(fit.rms_after_T))
                score = (
                    objective,
                    float(np.max(fit.rms_after_T)),
                    condition,
                    tuple(partition),
                    tuple(polarity),
                )
                if best is None or score < best[0]:
                    best = (score, (partition, polarity, channel_fields, channel_metrics, fit))
                    _emit(
                        progress_callback,
                        f"Contour assignment best at {completed}/{candidate_count}: "
                        f"weighted RMS {np.sqrt(objective):.6g} T, condition {condition:.3f}.",
                    )
                elif completed % report_interval == 0 or completed == candidate_count:
                    _emit(
                        progress_callback,
                        f"Contour assignment progress: {completed}/{candidate_count} candidates.",
                    )
                reporter.report(
                    50 + 50 * completed / candidate_count, f"Candidates checked: {completed}/{candidate_count}"
                )
    if best is None:
        raise RuntimeError("No full-rank contour assignment satisfied the configured normalized field-condition limit.")
    score, payload = best
    partition, polarity, channel_fields, channel_metrics, fit = payload
    diagnostics = ContourControllabilityDiagnostics(
        family_singular_values=_readonly(family_metrics[0]),
        family_numerical_rank=family_metrics[1],
        family_normalized_field_condition_number=family_metrics[2],
        family_maximum_normalized_field_correlation=family_metrics[3],
        channel_singular_values=_readonly(channel_metrics[0]),
        channel_numerical_rank=channel_metrics[1],
        channel_normalized_field_condition_number=channel_metrics[2],
        channel_maximum_normalized_field_correlation=channel_metrics[3],
        channel_field_rms_T_per_A=_readonly(channel_metrics[4]),
    )
    _emit(
        progress_callback,
        f"Contour assignment complete: {feasible_count}/{candidate_count} candidates feasible; "
        f"median fitted RMS {np.median(fit.rms_after_T):.6g} T.",
    )
    return ContourChannelDesignResult(
        dictionary=dictionary,
        family_to_channel=np.asarray(partition, dtype=np.int64),
        family_polarity=np.asarray(polarity, dtype=np.int8),
        channel_fields_by_scenario=channel_fields,
        fit=fit,
        current_limits=limits,
        scenario_weights=scenario_weights,
        diagnostics=diagnostics,
        partition_count=partition_count,
        candidate_count=candidate_count,
        feasible_candidate_count=feasible_count,
        objective_T2=float(score[0]),
    )


def save_contour_channel_design(
    output_directory: str | Path,
    result: ContourChannelDesignResult,
    *,
    scanner_from_source: RigidTransform | None = None,
    prefix: str = "contour_channel",
) -> tuple[Path, ...]:
    """Export assigned source traces and compact numerical audit arrays."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if result.dictionary.coordinate_frame == "mask_cad":
        if scanner_from_source is None:
            raise ValueError("Exporting mask-CAD contours requires scanner_from_source.")
        if scanner_from_source.source_frame != "mask_cad" or scanner_from_source.target_frame != "scanner":
            raise ValueError("scanner_from_source must map 'mask_cad' to 'scanner'.")
    written: list[Path] = []
    for channel in range(result.current_limits.channel_count):
        selected = np.flatnonzero(result.family_to_channel == channel)
        points: list[np.ndarray] = []
        scanner_points: list[np.ndarray] = []
        offsets = [0]
        family_indices: list[int] = []
        polarity: list[int] = []
        for family_index in selected:
            family = result.dictionary.families[int(family_index)]
            for contour in family.contours:
                source_points = np.asarray(contour.points_m, dtype=float)
                points.append(source_points)
                scanner_points.append(
                    scanner_from_source.apply_points(source_points)
                    if result.dictionary.coordinate_frame == "mask_cad"
                    else source_points
                )
                offsets.append(offsets[-1] + len(source_points))
                family_indices.append(int(family_index))
                polarity.append(int(result.family_polarity[family_index]))
        path = output / f"{prefix}_{channel + 1:02d}.npz"
        np.savez_compressed(
            path,
            coordinate_frame=np.asarray(result.dictionary.coordinate_frame),
            coordinate_units=np.asarray("m"),
            contour_points_m=np.vstack(points),
            contour_points_scanner_m=np.vstack(scanner_points),
            contour_point_offsets=np.asarray(offsets, dtype=np.int64),
            contour_family_indices=np.asarray(family_indices, dtype=np.int64),
            contour_polarity=np.asarray(polarity, dtype=np.int8),
            assigned_family_indices=selected.astype(np.int64),
            assigned_family_ids=np.asarray(
                [result.dictionary.families[index].family_id for index in selected], dtype=str
            ),
            current_lower_A=np.asarray(result.current_limits.lower_A[channel]),
            current_upper_A=np.asarray(result.current_limits.upper_A[channel]),
            routing_validated=np.asarray(False),
        )
        written.append(path)
    summary_path = output / "contour_channel_design_summary.npz"
    np.savez_compressed(
        summary_path,
        family_ids=np.asarray([family.family_id for family in result.dictionary.families], dtype=str),
        family_to_channel=result.family_to_channel,
        family_polarity=result.family_polarity,
        scenario_ids=np.asarray([item.scenario_id for item in result.fit.scenario_results], dtype=str),
        scenario_weights=result.scenario_weights,
        channel_currents_A=result.fit.current_matrix_A,
        rms_before_T=result.fit.rms_before_T,
        rms_after_T=result.fit.rms_after_T,
        family_singular_values=result.diagnostics.family_singular_values,
        channel_singular_values=result.diagnostics.channel_singular_values,
        channel_field_rms_T_per_A=result.diagnostics.channel_field_rms_T_per_A,
        routing_validated=np.asarray(False),
        single_layer_realization_claimed=np.asarray(False),
    )
    written.append(summary_path)
    summary_json = output / "contour_channel_design_summary.json"
    summary_json.write_text(
        json.dumps(_json_ready(result.summary()), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(summary_json)
    return tuple(written)


def _stirling_partition_count(item_count: int, group_count: int) -> int:
    """Count unlabeled nonempty partitions exactly with O(group_count) storage."""

    if group_count < 0 or item_count < group_count:
        return 0
    counts = [1] + [0] * group_count
    for size in range(1, item_count + 1):
        for groups in range(min(size, group_count), 0, -1):
            counts[groups] = groups * counts[groups] + counts[groups - 1]
        counts[0] = 0
    return counts[group_count]


def _canonical_surjections(item_count: int, group_count: int):
    """Yield restricted-growth strings: each unlabeled partition exactly once."""

    if group_count < 1 or item_count < group_count:
        return
    values = [0] * item_count

    def visit(index: int, maximum: int):
        if index == item_count:
            if maximum == group_count - 1:
                yield tuple(values)
            return
        upper = min(maximum + 1, group_count - 1)
        remaining_after = item_count - index - 1
        for value in range(upper + 1):
            new_maximum = max(maximum, value)
            missing_groups = group_count - 1 - new_maximum
            if missing_groups > remaining_after:
                continue
            values[index] = value
            yield from visit(index + 1, new_maximum)

    values[0] = 0
    yield from visit(1, 0)


def _relative_polarities(partition: Sequence[int], optimize: bool):
    family_count = len(partition)
    if not optimize:
        yield tuple(1 for _ in range(family_count))
        return
    first_by_channel: dict[int, int] = {}
    free: list[int] = []
    for index, channel in enumerate(partition):
        if channel not in first_by_channel:
            first_by_channel[int(channel)] = index
        else:
            free.append(index)
    for state in range(1 << len(free)):
        polarity = [1] * family_count
        for bit, index in enumerate(free):
            polarity[index] = -1 if state & (1 << bit) else 1
        yield tuple(polarity)


def _combine_family_fields(
    family_fields: Mapping[str, np.ndarray],
    assignment: np.ndarray,
    polarity: np.ndarray,
    channel_count: int,
) -> dict[str, np.ndarray]:
    selector = np.zeros((len(assignment), channel_count), dtype=float)
    selector[np.arange(len(assignment)), assignment] = polarity
    return {scenario_id: np.asarray(fields) @ selector for scenario_id, fields in family_fields.items()}


def _weighted_field_stack(
    dataset: PopulationShimDataset,
    fields_by_scenario: Mapping[str, np.ndarray],
    scenario_weights: np.ndarray,
    *,
    center: bool,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for cohort_weight, scenario in zip(scenario_weights, dataset.scenarios, strict=True):
        fields = np.asarray(fields_by_scenario[scenario.scenario_id], dtype=float)
        spatial = (
            np.ones(scenario.sample_count, dtype=float)
            if scenario.weights is None
            else np.asarray(scenario.weights, dtype=float)
        )
        spatial = spatial / np.sum(spatial)
        if center:
            fields = fields - spatial @ fields
        blocks.append(fields * np.sqrt(float(cohort_weight) * spatial)[:, None])
    return np.vstack(blocks)


def _field_metrics(matrix: np.ndarray) -> tuple[np.ndarray, int, float, float, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    singular = np.linalg.svd(values, compute_uv=False)
    tolerance = max(values.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.0)
    rank = int(np.sum(singular > tolerance))
    norms = np.linalg.norm(values, axis=0)
    if np.any(norms <= tolerance):
        return singular, rank, float("inf"), 1.0, norms
    normalized = values / norms
    normalized_singular = np.linalg.svd(normalized, compute_uv=False)
    condition = (
        float(normalized_singular[0] / normalized_singular[-1]) if normalized_singular[-1] > tolerance else float("inf")
    )
    correlation = np.abs(normalized.T @ normalized)
    np.fill_diagonal(correlation, 0.0)
    maximum_correlation = float(np.max(correlation)) if correlation.size else 0.0
    return singular, rank, condition, maximum_correlation, norms


def _design_scenario_weights(
    dataset: PopulationShimDataset,
    options: ContourChannelOptions,
) -> np.ndarray:
    scenario_ids = tuple(scenario.scenario_id for scenario in dataset.scenarios)
    if options.scenario_weights is not None:
        supplied = set(options.scenario_weights)
        expected = set(scenario_ids)
        if supplied != expected:
            raise ValueError(
                f"Explicit scenario weights have missing IDs {sorted(expected - supplied)} "
                f"and extra IDs {sorted(supplied - expected)}."
            )
        weights = np.asarray([options.scenario_weights[item] for item in scenario_ids], dtype=float)
    elif options.scenario_weighting == "uniform_scenario":
        weights = np.ones(len(dataset.scenarios), dtype=float)
    else:
        counts: dict[str, int] = {}
        for scenario in dataset.scenarios:
            counts[scenario.subject_id] = counts.get(scenario.subject_id, 0) + 1
        weights = np.asarray([1.0 / counts[scenario.subject_id] for scenario in dataset.scenarios])
    return _readonly(weights / np.sum(weights))
