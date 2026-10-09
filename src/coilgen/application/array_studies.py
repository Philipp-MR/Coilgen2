"""Opt-in array studies over verified winding geometry and registered cohorts.

Each channel is an independently driven electrical combination of saved bodies.
This layer calls the existing bounded fit and cohort solvers; it never modifies
the independently solved channels or promotes a sheet basis to routed hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from hashlib import sha256
from io import StringIO
import json
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..cohort_shim import ChannelCurrentLimits, CohortFitOptions, RigidTransform, ShimScenario, fit_scenario_currents
from ..persistence import atomic_json, json_value
from ..progress import ProgressReporter, progress_scope
from .gradient_studies import RigidTransform as MapTransform, _saved_winding
from .service import DesignService, RunRecord


ARRAY_SCHEMA = "coilgen-array-study-v1"


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f"{name} must be a nonempty string of at most 2048 characters.")
    return value.strip()


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive." if positive else "."))
    return float(value)


def _array(value, name, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric data.")
    result = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must be finite and have shape {shape}.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ArrayCase:
    name: str
    coordinates_m: np.ndarray
    desired_field_T: np.ndarray
    membership: str = "training"
    direction: tuple = (0, 0, 1)
    weights: np.ndarray | None = None
    subject: str = "research"
    rms_limit_T: float | None = None
    peak_limit_T: float | None = None
    source: str = ""
    registration: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "name", _text(self.name, "Case name"))
        coordinates = _array(self.coordinates_m, "Coordinates")
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not 1 <= len(coordinates) <= 100_000:
            raise ValueError("A case requires 1–100,000 XYZ samples in world metres.")
        object.__setattr__(self, "coordinates_m", coordinates)
        object.__setattr__(
            self, "desired_field_T", _array(self.desired_field_T, "Desired scalar field", (len(coordinates),))
        )
        direction = _array(self.direction, "Projection direction", (3,))
        if not np.any(direction):
            raise ValueError("Projection direction must be nonzero.")
        direction = direction / np.max(np.abs(direction))
        object.__setattr__(self, "direction", tuple(direction / np.linalg.norm(direction)))
        if self.membership not in {"training", "held_out"}:
            raise ValueError("Case membership must be training or held_out.")
        object.__setattr__(self, "subject", _text(self.subject, "Subject"))
        if self.weights is not None:
            weights = _array(self.weights, "Weights", (len(coordinates),))
            if np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError("Weights must be nonnegative with positive total weight.")
            object.__setattr__(self, "weights", weights)
        for key in ("rms_limit_T", "peak_limit_T"):
            value = getattr(self, key)
            if value is not None and _number(value, key) < 0:
                raise ValueError("Field limits must be nonnegative tesla values.")
        if not isinstance(self.source, str) or not isinstance(self.registration, Mapping):
            raise ValueError("Case source and registration must have explicit string/object types.")
        from .project import _freeze

        object.__setattr__(self, "registration", _freeze(json_value(self.registration, nonfinite="raise")))

    def scenario(self):
        return ShimScenario(
            self.name,
            self.subject,
            self.coordinates_m,
            -self.desired_field_T,
            RigidTransform("world", "scanner", np.eye(4)),
            self.weights,
            self.direction,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "coordinates_m": self.coordinates_m.tolist(),
            "desired_field_T": self.desired_field_T.tolist(),
            "membership": self.membership,
            "direction": list(self.direction),
            "weights": None if self.weights is None else self.weights.tolist(),
            "subject": self.subject,
            "rms_limit_T": self.rms_limit_T,
            "peak_limit_T": self.peak_limit_T,
            "source": self.source,
            "registration": json_value(self.registration),
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Array case must be an object.")
        return cls(**value)

    @classmethod
    def from_csv(
        cls,
        path,
        *,
        name=None,
        membership="training",
        length_unit="m",
        field_unit="T",
        transform=MapTransform(),
        direction=(0, 0, 1),
        disturbance=True,
        subject="research",
    ):
        """Import scalar x,y,z,B or vector x,y,z,Bx,By,Bz tables.

        Numeric headerless rows use the selected units. Named SI headers are
        x_m,y_m,z_m,B_T or x_m,y_m,z_m,Bx_T,By_T,Bz_T. Scalar fields are
        invariant under registration; their declared projection axis rotates.
        """
        length_scales, field_scales = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}, {"T": 1.0, "mT": 1e-3, "uT": 1e-6}
        if length_unit not in length_scales or field_unit not in field_scales:
            raise ValueError("Choose explicit supported coordinate and field units.")
        path = Path(path).expanduser().resolve()
        data = path.read_bytes()
        lines = [line.strip() for line in data.decode("utf-8-sig").splitlines() if line.strip()]
        if not lines:
            raise ValueError("Field map is empty.")
        header = lines[0].replace(" ", "")
        if header in {"x_m,y_m,z_m,B_T", "x_m,y_m,z_m,Bx_T,By_T,Bz_T"}:
            if length_unit != "m" or field_unit != "T":
                raise ValueError("Named SI columns require metre and tesla units.")
            lines = lines[1:]
        rows = np.loadtxt(StringIO("\n".join(line.replace(",", " ") for line in lines)), ndmin=2)
        if rows.shape[1] not in {4, 6} or not np.all(np.isfinite(rows)):
            raise ValueError("Field map requires finite x y z B or x y z Bx By Bz rows.")
        rotation = np.asarray(transform.rotation)
        axis = _array(direction, "Projection direction", (3,))
        if not np.any(axis):
            raise ValueError("Projection direction must be nonzero.")
        axis = axis / np.max(np.abs(axis))
        axis = axis / np.linalg.norm(axis)
        world_axis = rotation @ axis
        points = rows[:, :3] * length_scales[length_unit] @ rotation.T + transform.translation_m
        field_values = rows[:, 3:] * field_scales[field_unit]
        scalar = field_values[:, 0] if rows.shape[1] == 4 else (field_values @ rotation.T) @ world_axis
        return cls(
            name or path.stem,
            points,
            -scalar if disturbance else scalar,
            membership,
            tuple(world_axis),
            subject=subject,
            source=str(path),
            registration={
                "source_sha256": sha256(data).hexdigest(),
                "source_length_unit": length_unit,
                "source_field_unit": field_unit,
                "map_kind": "disturbance" if disturbance else "desired_field",
                **transform.to_dict(),
            },
        )


@dataclass(frozen=True, slots=True)
class ArrayChannel:
    id: str
    name: str
    run_directory: str
    request_sha256: str
    body_current_ratios: dict
    path_sha256: dict
    lower_A: float = -2.0
    upper_A: float = 2.0
    resistance_ohm: float | None = None

    def __post_init__(self):
        for key in ("id", "name", "run_directory"):
            object.__setattr__(self, key, _text(getattr(self, key), key))
        if (
            not isinstance(self.request_sha256, str)
            or len(self.request_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.request_sha256)
        ):
            raise ValueError("Array source requires a SHA-256 request identity.")
        if (
            not isinstance(self.body_current_ratios, Mapping)
            or not self.body_current_ratios
            or set(self.body_current_ratios) != set(self.path_sha256)
        ):
            raise ValueError("A channel must explicitly declare a current ratio and path hash for every saved body.")
        ratios = {
            _text(key, "Body identity"): _number(value, "Current ratio")
            for key, value in self.body_current_ratios.items()
        }
        if not any(ratios.values()):
            raise ValueError("A channel must drive at least one winding body.")
        for value in self.path_sha256.values():
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Saved path hashes must be SHA-256 identities.")
        object.__setattr__(self, "body_current_ratios", MappingProxyType(ratios))
        object.__setattr__(self, "path_sha256", MappingProxyType(dict(self.path_sha256)))
        if _number(self.lower_A, "Lower current") > _number(self.upper_A, "Upper current"):
            raise ValueError("Current bounds must satisfy lower <= upper.")
        if self.resistance_ohm is not None:
            _number(self.resistance_ohm, "Channel resistance", positive=True)

    @classmethod
    def from_run(cls, id, name, record, *, body_current_ratios, lower_A=-2.0, upper_A=2.0, resistance_ohm=None):
        from ..gradient_engineering import winding_path_sha256

        record = DesignService.load_run(record.directory if isinstance(record, RunRecord) else record)
        paths, _, _, _ = _saved_winding(record, "original")
        return cls(
            id,
            name,
            str(record.directory),
            record.metadata["request_sha256"],
            dict(body_current_ratios),
            {key: winding_path_sha256(path) for key, path in paths.items()},
            lower_A,
            upper_A,
            resistance_ohm,
        )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "run_directory": self.run_directory,
            "request_sha256": self.request_sha256,
            "body_current_ratios": dict(self.body_current_ratios),
            "path_sha256": dict(self.path_sha256),
            "lower_A": self.lower_A,
            "upper_A": self.upper_A,
            "resistance_ohm": self.resistance_ohm,
        }


@dataclass(frozen=True, slots=True)
class CohortControls:
    strategy: str = "pod"
    channel_count: int = 2
    current_limit_A: float = 2.0
    resistance_ohm: tuple | None = None
    total_power_limit_W: float | None = None
    contour_dictionary: dict | None = None

    def __post_init__(self):
        if self.strategy not in {"pod", "sectorization", "contour_dictionary"}:
            raise ValueError("Unknown cohort strategy.")
        if (
            isinstance(self.channel_count, bool)
            or not isinstance(self.channel_count, int)
            or not 1 <= self.channel_count <= 64
        ):
            raise ValueError("Cohort channel count must be an integer from 1 to 64.")
        _number(self.current_limit_A, "Cohort current limit", positive=True)
        if self.resistance_ohm is not None:
            values = tuple(_number(value, "Cohort channel resistance", positive=True) for value in self.resistance_ohm)
            if len(values) != self.channel_count:
                raise ValueError("Declare one resistance for each synthesized channel.")
            object.__setattr__(self, "resistance_ohm", values)
        if self.total_power_limit_W is not None:
            _number(self.total_power_limit_W, "Cohort simultaneous power", positive=True)
            if self.resistance_ohm is None:
                raise ValueError("Cohort power limits require resistances for the newly synthesized channels.")
        if self.contour_dictionary is not None:
            if self.dictionary().coordinate_frame != "scanner":
                raise ValueError("The cohort contour dictionary must use world/scanner coordinates in metres.")
            from .project import _freeze

            object.__setattr__(
                self, "contour_dictionary", _freeze(json_value(self.contour_dictionary, nonfinite="raise"))
            )

    @staticmethod
    def encode_dictionary(dictionary):
        return {
            "coordinate_frame": dictionary.coordinate_frame,
            "metadata": json_value(dictionary.metadata),
            "families": [
                {
                    "name": family.family_id,
                    "metadata": json_value(family.metadata),
                    "contours": [
                        {
                            "points_m": contour.points_m.tolist(),
                            "source_part_index": contour.source_part_index,
                            "source_contour_index": contour.source_contour_index,
                            "metadata": json_value(contour.metadata),
                        }
                        for contour in family.contours
                    ],
                }
                for family in dictionary.families
            ],
        }

    def dictionary(self):
        from ..cohort_contours import ContourDictionary, ContourFamily, PhysicalContour

        if not isinstance(self.contour_dictionary, Mapping) or set(self.contour_dictionary) != {
            "coordinate_frame",
            "metadata",
            "families",
        }:
            raise ValueError("Malformed owned cohort contour dictionary.")
        value = self.contour_dictionary
        families = tuple(
            ContourFamily(
                item["name"], tuple(PhysicalContour(**contour) for contour in item["contours"]), item["metadata"]
            )
            for item in value["families"]
        )
        return ContourDictionary(families, value["coordinate_frame"], value["metadata"])

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "channel_count": self.channel_count,
            "current_limit_A": self.current_limit_A,
            "resistance_ohm": None if self.resistance_ohm is None else list(self.resistance_ohm),
            "total_power_limit_W": self.total_power_limit_W,
            "contour_dictionary": None if self.contour_dictionary is None else json_value(self.contour_dictionary),
        }


@dataclass(frozen=True, slots=True)
class ArrayStudySpec:
    cases: tuple[ArrayCase, ...] = ()
    channels: tuple[ArrayChannel, ...] = ()
    total_power_limit_W: float | None = None
    center_frequency: bool = False
    current_regularization: float = 0.0
    cohort_controls: CohortControls | None = None

    def __post_init__(self):
        for name, kind, maximum in (("cases", ArrayCase, 256), ("channels", ArrayChannel, 64)):
            values = tuple(getattr(self, name))
            if len(values) > maximum or any(not isinstance(item, kind) for item in values):
                raise ValueError(f"Invalid {name}; maximum {maximum}.")
            ids = [item.name if name == "cases" else item.id for item in values]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Repeated array {name} identities.")
            object.__setattr__(self, name, values)
        occupied = set()
        for channel in self.channels:
            driven = {channel.path_sha256[body] for body, ratio in channel.body_current_ratios.items() if ratio != 0}
            if occupied & driven:
                raise ValueError("Two independent array channels cannot drive the same saved physical conductor path.")
            occupied.update(driven)
        if not isinstance(self.center_frequency, bool):
            raise ValueError("Frequency centering must be a boolean.")
        if _number(self.current_regularization, "Current regularization") < 0:
            raise ValueError("Current regularization must be nonnegative.")
        if self.total_power_limit_W is not None:
            _number(self.total_power_limit_W, "Simultaneous power limit", positive=True)
            if self.channels and any(item.resistance_ohm is None for item in self.channels):
                raise ValueError("A simultaneous power bound requires resistance for every channel.")
        if self.cohort_controls is not None and not isinstance(self.cohort_controls, CohortControls):
            raise ValueError("Cohort controls require a typed CohortControls value.")

    def to_dict(self):
        value = {
            "schema": ARRAY_SCHEMA,
            "cases": [item.to_dict() for item in self.cases],
            "channels": [item.to_dict() for item in self.channels],
            "total_power_limit_W": self.total_power_limit_W,
            "center_frequency": self.center_frequency,
            "current_regularization": self.current_regularization,
        }
        if self.cohort_controls is not None:
            value["cohort_controls"] = self.cohort_controls.to_dict()
        return value

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or value.get("schema") != ARRAY_SCHEMA
            or set(value) - {"cohort_controls"}
            != {"schema", "cases", "channels", "total_power_limit_W", "center_frequency", "current_regularization"}
        ):
            raise ValueError("Malformed array study specification.")
        return cls(
            tuple(ArrayCase.from_dict(item) for item in value["cases"]),
            tuple(ArrayChannel(**item) for item in value["channels"]),
            value["total_power_limit_W"],
            value["center_frequency"],
            value["current_regularization"],
            None if value.get("cohort_controls") is None else CohortControls(**value["cohort_controls"]),
        )

    def save(self, path):
        atomic_json(Path(path), self.to_dict())


def _diagnostics(matrix, weights, center):
    weights = np.ones(len(matrix)) if weights is None else weights
    value = np.asarray(matrix, dtype=float).copy()
    if center:
        value -= (weights @ value / np.sum(weights))[None, :]
    value *= np.sqrt(weights / np.sum(weights))[:, None]
    scales = np.linalg.norm(value, axis=0)
    normalized = np.divide(value, scales, out=np.zeros_like(value), where=scales > 0)
    singular = np.linalg.svd(normalized, compute_uv=False)
    tolerance = (singular[0] if len(singular) else 0) * max(normalized.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if rank == matrix.shape[1] else None
    return {
        "rank": rank,
        "channel_count": matrix.shape[1],
        "normalized_condition_number": condition,
        "singular_values": singular.tolist(),
        "field_rms_T_per_A": scales.tolist(),
        "frequency_centered": center,
    }


def fit_array_fields(spec, channel_fields_by_case, *, provenance=(), progress=None):
    """Pure numerical adapter also usable with independently calibrated field maps."""
    if not isinstance(spec, ArrayStudySpec) or not spec.channels or not spec.cases:
        raise ValueError("An array fit requires channels and cases.")
    resistances = [item.resistance_ohm for item in spec.channels]
    limits = ChannelCurrentLimits(
        [item.lower_A for item in spec.channels],
        [item.upper_A for item in spec.channels],
        None if any(value is None for value in resistances) else resistances,
        spec.total_power_limit_W,
    )
    options = CohortFitOptions(
        center_frequency=spec.center_frequency, current_regularization=spec.current_regularization
    )
    rows = []
    reporter = ProgressReporter(label="Array fitting")
    for index, case in enumerate(spec.cases):
        if progress:
            progress(f"Fitting {case.name} ({index + 1}/{len(spec.cases)})")
        matrix = _array(
            channel_fields_by_case[case.name], "Channel field matrix", (len(case.coordinates_m), len(spec.channels))
        )
        with progress_scope(100 * index / len(spec.cases), 100 * (index + 1) / len(spec.cases), overall=True):
            reporter.report(None, f"Solving bounded signed currents: {case.name}")
            result = fit_scenario_currents(case.scenario(), matrix, limits, options=options)
        error = result.residual_T  # actual minus desired, including the explicitly requested frequency offset
        peak = float(np.max(np.abs(error)))
        checks = [
            value <= limit
            for value, limit in ((result.rms_after_T, case.rms_limit_T), (peak, case.peak_limit_T))
            if limit is not None
        ]
        rows.append(
            {
                "name": case.name,
                "membership": case.membership,
                "subject": case.subject,
                "currents_A": result.channel_currents_A.tolist(),
                "frequency_offset_T": result.frequency_offset_T,
                "rms_error_T": result.rms_after_T,
                "peak_error_T": peak,
                "residual_T": error.tolist(),
                "combined_field_T": (matrix @ result.channel_currents_A + result.frequency_offset_T).tolist(),
                "power_W": result.power_W,
                "saturated_channels": result.saturated_channels.tolist(),
                "solver": result.solver,
                "status": result.status,
                "passed": all(checks) if checks else None,
                "rms_limit_T": case.rms_limit_T,
                "peak_limit_T": case.peak_limit_T,
                "diagnostics": _diagnostics(matrix, case.weights, spec.center_frequency),
                "channel_fields_T_per_A": matrix.tolist(),
            }
        )
        with progress_scope(100 * index / len(spec.cases), 100 * (index + 1) / len(spec.cases), overall=True):
            reporter.report(100, f"Completed array fit {index + 1}/{len(spec.cases)}")
    return {
        "schema": "coilgen-array-results-v1",
        "spec": spec.to_dict(),
        "cases": rows,
        "provenance": json_value(provenance),
        "scope": "bounded scalar field fitting at supplied samples",
        "current_policy": "joint signed driver current fit; explicit body current ratios",
        "thermal_and_mutual_coupling_assessed": False,
        "worst_case": max(rows, key=lambda row: row["rms_error_T"])["name"],
    }


def evaluate_array(spec, *, run_resolver=None, progress=None):
    """Verify the exact saved geometry, evaluate every per-ampere channel, then fit.

    A resolver may locate an owned archive after relocation. Identity and path
    hashes are checked again, including when the resolver returns a RunRecord.
    """
    from ..gradient_engineering import winding_path_sha256
    from ..physics import biot_savart_wire_path
    from ..research.validation import _samples_on_wire
    from .evidence import verified_run_metadata

    fields = {case.name: np.zeros((len(case.coordinates_m), len(spec.channels))) for case in spec.cases}
    sources = []
    reporter = ProgressReporter(label="Array field evaluation")
    for index, channel in enumerate(spec.channels):
        if progress:
            progress(f"Verifying/evaluating {channel.name} ({index + 1}/{len(spec.channels)})")
        start, span = 80 * index / len(spec.channels), 80 / len(spec.channels)
        with progress_scope(start, start + 0.1 * span, overall=True):
            reporter.report(None, f"Verifying immutable saved winding: {channel.name}")
            location = run_resolver(channel) if run_resolver is not None else channel.run_directory
            record = DesignService.load_run(location.directory if isinstance(location, RunRecord) else location)
            _, _, health = verified_run_metadata(record.directory)
        if not health.metadata_checked or record.metadata.get("request_sha256") != channel.request_sha256:
            raise ValueError(f"Channel {channel.name}: saved request identity changed or is unverified.")
        paths, _, source, complete = _saved_winding(record, "original")
        if set(paths) != set(channel.body_current_ratios) or {
            key: winding_path_sha256(path) for key, path in paths.items()
        } != dict(channel.path_sha256):
            raise ValueError(f"Channel {channel.name}: saved electrical geometry changed.")
        total_samples = sum(len(case.coordinates_m) for case in spec.cases) * len(paths)
        completed_samples = 0
        for case in spec.cases:
            for body, path in paths.items():
                if _samples_on_wire(path, case.coordinates_m):
                    raise ValueError(f"Case {case.name}: a sample intersects the thin-wire centreline.")
                with progress_scope(
                    start + span * (0.1 + 0.9 * completed_samples / total_samples),
                    start + span * (0.1 + 0.9 * (completed_samples + len(case.coordinates_m)) / total_samples),
                    overall=True,
                ):
                    reporter.report(0, f"Evaluating {channel.name} · {case.name} · {body}")
                    field_values = biot_savart_wire_path(path, case.coordinates_m, worker_count=1)
                fields[case.name][:, index] += channel.body_current_ratios[body] * (field_values @ case.direction)
                completed_samples += len(case.coordinates_m)
        sources.append(
            {
                "channel_id": channel.id,
                "request_sha256": channel.request_sha256,
                "path_sha256": dict(channel.path_sha256),
                "source": source,
                "complete_circuit": complete,
                "field_model": "thin_wire_exact_segments",
            }
        )
    with progress_scope(80, 100, overall=True):
        return fit_array_fields(spec, fields, provenance=sources, progress=progress)


def synthesize_cohort(project, spec, *, channel_count=None, current_limit_A=None, strategy=None, progress=None):
    """Design an opt-in common-topology sheet basis using training cases only.

    Existing POD/adaptive sectorization solvers enforce their usual rank and
    current-density conditions. Held-out cases only evaluate the finished basis.
    Exact ordered mesh and coordinate-matched stream functions are embedded in
    returned projects; subsequent winding review establishes physical geometry.
    """
    from copy import deepcopy
    from ..basis import calculate_basis_functions, calculate_one_ring_by_mesh
    from ..cohort_shim import (
        CohortChannelDesignOptions,
        RegisteredSurfaceProblem,
        SurfaceShimScenario,
        design_cohort_channels,
    )
    from ..cohort_sectorization import AdaptiveSectorizationOptions
    from ..geometry import load_mesh_from_config, refine_mesh, split_disconnected_mesh
    from ..models import CoilPart, TargetField
    from ..optimization import StreamFunctionOptimizationProblem
    from ..physics import calculate_resistance_matrix, calculate_sensitivity_matrix
    from .assets import InputAsset, mesh_asset, target_asset
    from .project import ProjectDocument

    controls = spec.cohort_controls or CohortControls()
    channel_count = controls.channel_count if channel_count is None else channel_count
    current_limit_A = controls.current_limit_A if current_limit_A is None else current_limit_A
    strategy = controls.strategy if strategy is None else strategy
    training = tuple(case for case in spec.cases if case.membership == "training")
    _check_holdout(spec)
    if strategy not in {"pod", "sectorization"}:
        raise ValueError("Sheet synthesis strategy must be pod or sectorization.")
    if (
        isinstance(channel_count, bool)
        or not isinstance(channel_count, int)
        or not 1 <= channel_count <= 64
        or len(training) < channel_count
    ):
        raise ValueError("Supply at least one independent training case per requested channel (maximum 64).")
    direction = training[0].direction
    if any(not np.allclose(case.direction, direction, rtol=0, atol=1e-10) for case in spec.cases):
        raise ValueError("Common-topology cohort synthesis requires the same world projection direction in every case.")
    config = deepcopy(project.config)
    config.target_field_component, config.b_0_direction = "BdotB0", direction
    config.symmetry_reduction = "none"
    reporter = ProgressReporter.from_config(config, label="Cohort synthesis")
    with progress_scope(0, 5, overall=True):
        reporter.report(None, "Loading and refining the common support surface")
        meshes = [
            refine_mesh(mesh, config.iteration_num_mesh_refinement)
            for mesh in split_disconnected_mesh(load_mesh_from_config(config, project.project_root))
        ]
    with progress_scope(5, 10, overall=True):
        parts = calculate_basis_functions(calculate_one_ring_by_mesh([CoilPart(mesh=mesh) for mesh in meshes]))
    with progress_scope(10, 15, overall=True):
        parts = calculate_resistance_matrix(parts, config)
    surface_cases = []
    identity = RigidTransform("world", "scanner", np.eye(4))
    total_samples = sum(len(case.coordinates_m) for case in spec.cases)
    completed_samples = 0
    for case in spec.cases:
        if progress:
            progress(f"Assembling registered surface operator: {case.name}")
        target = TargetField(
            coords=case.coordinates_m,
            b=np.outer(case.desired_field_T, direction),
            weights=np.ones(len(case.coordinates_m)) if case.weights is None else case.weights,
            suppressed=np.zeros(len(case.coordinates_m), dtype=bool),
            target_field_group_inds=np.ones(len(case.coordinates_m), dtype=np.int64),
        )
        with progress_scope(
            15 + 35 * completed_samples / total_samples,
            15 + 35 * (completed_samples + len(case.coordinates_m)) / total_samples,
            overall=True,
        ):
            reporter.report(0, f"Assembling registered surface operator: {case.name}")
            calculated = calculate_sensitivity_matrix(deepcopy(parts), target, config)
            problem = StreamFunctionOptimizationProblem.from_parts(calculated, target, config)
        completed_samples += len(case.coordinates_m)
        surface_cases.append(SurfaceShimScenario(case.scenario(), RegisteredSurfaceProblem(problem, identity)))
    options = CohortChannelDesignOptions(
        channel_count=channel_count,
        current_limit_A=current_limit_A,
        fit_options=CohortFitOptions(spec.center_frequency, spec.current_regularization),
        sectorization=AdaptiveSectorizationOptions(enabled=strategy == "sectorization"),
        channel_resistance_ohm=None if spec.cohort_controls is None else spec.cohort_controls.resistance_ohm,
        total_power_limit_W=None if spec.cohort_controls is None else spec.cohort_controls.total_power_limit_W,
    )
    with progress_scope(50, 85, overall=True):
        reporter.report(None, "Designing a common cohort basis; solver work depends on convergence")
        result = design_cohort_channels(
            tuple(item for item, case in zip(surface_cases, spec.cases) if case.membership == "training"),
            config,
            options,
            progress_callback=progress,
        )
    mesh = surface_cases[0].problem.combined_mesh
    asset = mesh_asset(mesh)
    initial = training[0]
    target = target_asset(
        TargetField(
            coords=initial.coordinates_m,
            b=np.outer(initial.desired_field_T, direction),
            weights=np.ones(len(initial.coordinates_m)),
            suppressed=np.zeros(len(initial.coordinates_m), dtype=bool),
            target_field_group_inds=np.ones(len(initial.coordinates_m), dtype=np.int64),
        )
    )
    generated, fields = [], {}
    for case, surface in zip(spec.cases, surface_cases):
        fields[case.name] = surface.problem.unweighted_reduced_sensitivity @ result.channel_basis_per_A
    for index in range(channel_count):
        with progress_scope(85, 90, overall=True):
            reporter.report(100 * index / channel_count, f"Creating portable channel {index + 1}/{channel_count}")
        output = StringIO()
        part_indices = np.concatenate(
            [np.full(part.mesh.n_vertices, part_index) for part_index, part in enumerate(parts)]
        )
        np.savetxt(
            output,
            np.column_stack((part_indices, mesh.vertices, result.full_channel_basis_per_A[:, index])),
            delimiter=",",
            header="part,x,y,z,stream_function",
            comments="",
            fmt="%.17g",
        )
        sf_asset = InputAsset(
            output.getvalue().encode("utf-8"),
            ".csv",
            "stream_function",
            "cohort basis with part and coordinate binding",
        )
        values = project.values
        values.update(
            coil_mesh_file="asset:" + asset.id,
            target_field_definition_file="asset:" + target.id,
            preoptimized_stream_function_file="asset:" + sf_asset.id,
            sf_source_file="none",
            target_field_component="BdotB0",
            b_0_direction=list(direction),
            symmetry_reduction="none",
            target_gradient_normalization="none",
            iteration_num_mesh_refinement=0,
            preoptimized_stream_function_field_name="stream_function",
        )
        # Preserve other native options, while dropping study-only source bindings.
        from uuid import uuid4

        all_assets = {**project._assets, asset.id: asset, target.id: target, sf_asset.id: sf_asset}
        required_assets = {
            value[6:] for value in values.values() if isinstance(value, str) and value.startswith("asset:")
        }
        generated.append(
            ProjectDocument(
                uuid4().hex,
                f"Cohort channel {index + 1}",
                1,
                values,
                {key: value for key, value in all_assets.items() if key in required_assets},
                project.project_root,
                _session={
                    "cohort_provenance": {
                        "strategy": strategy,
                        "training_cases": [case.name for case in training],
                        "training_data_sha256": sha256(
                            json.dumps([case.to_dict() for case in training], sort_keys=True, allow_nan=False).encode(
                                "utf-8"
                            )
                        ).hexdigest(),
                        "source_project_fingerprint": project.fingerprint,
                        "routing_validated": False,
                        "basis_unit": "stream_function_per_A",
                    }
                },
            )
        )
    fitted = []
    for index, case in enumerate(spec.cases):
        with progress_scope(90 + 10 * index / len(spec.cases), 90 + 10 * (index + 1) / len(spec.cases), overall=True):
            reporter.report(None, f"Evaluating signed channel currents: {case.name}")
            fitted_case = fit_scenario_currents(
                case.scenario(), fields[case.name], result.current_limits, options=options.fit_options
            )
        peak = float(np.max(np.abs(fitted_case.residual_T)))
        checks = [
            error <= limit
            for error, limit in ((fitted_case.rms_after_T, case.rms_limit_T), (peak, case.peak_limit_T))
            if limit is not None
        ]
        fitted.append(
            {
                "name": case.name,
                "membership": case.membership,
                "currents_A": fitted_case.channel_currents_A.tolist(),
                "rms_error_T": fitted_case.rms_after_T,
                "peak_error_T": peak,
                "passed": all(checks) if checks else None,
                "residual_T": fitted_case.residual_T.tolist(),
                "power_W": fitted_case.power_W,
                "saturated_channels": fitted_case.saturated_channels.tolist(),
                "diagnostics": _diagnostics(fields[case.name], case.weights, options.fit_options.center_frequency),
            }
        )
        with progress_scope(90 + 10 * index / len(spec.cases), 90 + 10 * (index + 1) / len(spec.cases), overall=True):
            reporter.report(100, f"Cohort cases evaluated: {index + 1}/{len(spec.cases)}")
    return {
        "schema": "coilgen-cohort-synthesis-v1",
        "projects": tuple(generated),
        "cases": fitted,
        "fields": fields,
        "training_cases": [case.name for case in training],
        "held_out_cases": [case.name for case in spec.cases if case.membership == "held_out"],
        "singular_values": result.pod.singular_values.tolist(),
        "strategy": strategy,
        "synthesis_controls": {
            **controls.to_dict(),
            "strategy": strategy,
            "channel_count": channel_count,
            "current_limit_A": current_limit_A,
        },
        "scope": "continuous current sheet basis; contouring and electrical routing remain to be evaluated",
        "routing_validated": False,
    }


def _check_holdout(spec):
    def identity(case):
        return sha256(
            case.coordinates_m.tobytes() + case.desired_field_T.tobytes() + np.asarray(case.direction).tobytes()
        ).digest()

    training = {identity(case) for case in spec.cases if case.membership == "training"}
    if any(identity(case) in training for case in spec.cases if case.membership == "held_out"):
        raise ValueError("A held-out case duplicates training coordinates, direction and field values.")


def synthesize_contour_cohort(spec, dictionary=None, *, channel_count=None, current_limit_A=None, progress=None):
    """Expose existing physical contour-family assignment with held-out review.

    The imported dictionary must already use world/scanner coordinates. Every
    contour belongs to one channel; connecting leads are not fabricated here.
    """
    from ..cohort_contours import (
        ContourChannelOptions,
        design_contour_dictionary_channels,
        evaluate_contour_family_fields,
    )
    from ..cohort_shim import PopulationShimDataset

    controls = spec.cohort_controls or CohortControls(strategy="contour_dictionary")
    channel_count = controls.channel_count if channel_count is None else channel_count
    current_limit_A = controls.current_limit_A if current_limit_A is None else current_limit_A
    dictionary = controls.dictionary() if dictionary is None else dictionary
    _check_holdout(spec)
    training = tuple(case for case in spec.cases if case.membership == "training")
    if not training:
        raise ValueError("Contour-family synthesis requires a training case.")
    if dictionary.coordinate_frame != "scanner":
        raise ValueError("Register the contour dictionary into the scanner/world frame before array synthesis.")
    options = ContourChannelOptions(
        channel_count=channel_count,
        current_limit_A=current_limit_A,
        fit_options=CohortFitOptions(spec.center_frequency, spec.current_regularization),
        channel_resistance_ohm=None if spec.cohort_controls is None else spec.cohort_controls.resistance_ohm,
        total_power_limit_W=None if spec.cohort_controls is None else spec.cohort_controls.total_power_limit_W,
    )
    reporter = ProgressReporter(label="Contour cohort synthesis")
    with progress_scope(0, 65, overall=True):
        result = design_contour_dictionary_channels(
            PopulationShimDataset(tuple(case.scenario() for case in training)),
            dictionary,
            options,
            progress_callback=progress,
        )
    with progress_scope(65, 90, overall=True):
        all_fields = evaluate_contour_family_fields(
            dictionary, PopulationShimDataset(tuple(case.scenario() for case in spec.cases)), progress_callback=progress
        )
    rows, channels = [], []
    for index in range(channel_count):
        channels.append(
            {
                "index": index,
                "families": [
                    {
                        "name": family.family_id,
                        "polarity": int(result.family_polarity[family_index]),
                        "contours_m": [contour.points_m.tolist() for contour in family.contours],
                    }
                    for family_index, family in enumerate(dictionary.families)
                    if result.family_to_channel[family_index] == index
                ],
            }
        )
    for case_index, case in enumerate(spec.cases):
        fields = np.column_stack(
            [
                sum(
                    (
                        result.family_polarity[index] * all_fields[case.name][:, index]
                        for index in range(dictionary.family_count)
                        if result.family_to_channel[index] == channel
                    ),
                    np.zeros(len(case.coordinates_m)),
                )
                for channel in range(channel_count)
            ]
        )
        with progress_scope(
            90 + 10 * case_index / len(spec.cases), 90 + 10 * (case_index + 1) / len(spec.cases), overall=True
        ):
            reporter.report(None, f"Evaluating assigned physical channel currents: {case.name}")
            fit = fit_scenario_currents(case.scenario(), fields, result.current_limits, options=options.fit_options)
        peak = float(np.max(np.abs(fit.residual_T)))
        checks = [
            error <= limit
            for error, limit in ((fit.rms_after_T, case.rms_limit_T), (peak, case.peak_limit_T))
            if limit is not None
        ]
        rows.append(
            {
                "name": case.name,
                "membership": case.membership,
                "currents_A": fit.channel_currents_A.tolist(),
                "rms_error_T": fit.rms_after_T,
                "peak_error_T": peak,
                "passed": all(checks) if checks else None,
                "residual_T": fit.residual_T.tolist(),
                "power_W": fit.power_W,
                "saturated_channels": fit.saturated_channels.tolist(),
                "diagnostics": _diagnostics(fields, case.weights, spec.center_frequency),
            }
        )
    return {
        "schema": "coilgen-contour-cohort-v1",
        "spec": spec.to_dict(),
        "cases": rows,
        "summary": result.summary(),
        "synthesis_controls": {
            **controls.to_dict(),
            "strategy": "contour_dictionary",
            "channel_count": channel_count,
            "current_limit_A": current_limit_A,
        },
        "channels": channels,
        "routing_validated": False,
        "scope": "assigned physical source contours; inter-family feeds and complete circuits require routing review",
    }
