"""Bounded numerical studies and fixed-current measured-coil comparisons."""

from __future__ import annotations

from copy import deepcopy
import base64
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping

import numpy as np

from ..geometry import load_mesh_from_config
from .comparison import compare_runs
from .evidence import verified_run_metadata
from .project import ProjectDocument, _freeze
from .service import DesignService, RunRecord


STUDY_SCHEMA = "coilgen-convergence-study-v1"


@dataclass(frozen=True, slots=True)
class StudyCandidate:
    project: ProjectDocument
    axis: str
    value: int | None
    source_fingerprint: str

    def __post_init__(self):
        if not isinstance(self.project, ProjectDocument) or self.axis not in {
            "baseline",
            "mesh",
            "quadrature",
            "independent_sampling",
        }:
            raise ValueError("Study candidate requires a project and a supported numerical axis.")
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint:
            raise ValueError("Study candidate requires its baseline input fingerprint.")
        if self.axis == "baseline":
            if self.value is not None:
                raise ValueError("The baseline has no refinement-axis value.")
        else:
            _levels(
                (self.value,),
                "Candidate level",
                *({"mesh": (0, 3), "quadrature": (1, 16), "independent_sampling": (2, 128)}[self.axis]),
            )

    @property
    def label(self):
        return "Baseline" if self.axis == "baseline" else f"{self.axis.replace('_', ' ').capitalize()} = {self.value}"

    def to_dict(self):
        return {
            "schema": STUDY_SCHEMA,
            "project": self.project.to_dict(),
            "axis": self.axis,
            "value": self.value,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def from_dict(cls, value, *, project_root=None):
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "project", "axis", "value", "source_fingerprint"}
            or value["schema"] != STUDY_SCHEMA
        ):
            raise ValueError("Malformed convergence candidate.")
        return cls(
            ProjectDocument.from_dict(value["project"], project_root=project_root),
            value["axis"],
            value["value"],
            value["source_fingerprint"],
        )


def _levels(values, name, minimum, maximum):
    levels = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum
        for value in levels
    ) or len(set(levels)) != len(levels):
        raise ValueError(f"{name} require distinct integers from {minimum} to {maximum}.")
    return tuple(int(value) for value in levels)


def convergence_candidates(
    project: ProjectDocument,
    *,
    mesh_refinements=(),
    quadrature_orders=(),
    independent_resolutions=(),
    max_candidates=12,
    max_mesh_faces=200_000,
):
    """Change one numerical axis at a time while preserving all requirements.

    Mesh levels use the existing midpoint-subdivision setting: each pass has
    four times as many faces and retains the input surface's native geometry.
    Quadrature changes the field operator. Independent resolution changes only
    the validation samples; it cannot establish mesh or integration convergence.
    Baseline-equivalent levels are omitted. Every candidate owns an independent
    project identity, so queue provenance cannot confuse it with the editor.
    """
    maximum = _levels((max_candidates,), "Candidate limit", 2, 32)[0]
    face_limit = _levels((max_mesh_faces,), "Mesh face limit", 1, 2_000_000)[0]
    axes = (
        ("mesh", _levels(mesh_refinements, "Mesh refinements", 0, 3), "iteration_num_mesh_refinement", False),
        ("quadrature", _levels(quadrature_orders, "Quadrature orders", 1, 16), "gauss_order", False),
        (
            "independent_sampling",
            _levels(independent_resolutions, "Independent resolutions", 2, 128),
            "validation_resolution",
            True,
        ),
    )
    values, study = project.values, project.study
    requested = [
        (axis, level, key, is_study)
        for axis, levels, key, is_study in axes
        for level in levels
        if level != (study if is_study else values)[key]
    ]
    if not requested:
        raise ValueError("Choose at least one numerical level different from the baseline.")
    if len(requested) + 1 > maximum:
        raise ValueError(f"The study exceeds its explicit limit of {maximum} candidates.")
    if any(axis == "mesh" for axis, *_ in requested):
        mesh = load_mesh_from_config(project.config, project.project_root)
        levels = [values["iteration_num_mesh_refinement"], *(level for axis, level, *_ in requested if axis == "mesh")]
        if any(mesh.n_faces * 4**level > face_limit for level in levels):
            raise ValueError(f"A mesh candidate exceeds the explicit {face_limit:,}-face study budget.")
    source = project.fingerprint
    baseline = project.duplicate((project.name[:105] + " · baseline")[:120])
    candidates = [StudyCandidate(baseline, "baseline", None, source)]
    for axis, level, key, is_study in requested:
        candidate = project.duplicate((project.name[:90] + f" · {axis} {level}")[:120])
        candidate = candidate.update_inputs(
            study_changes={key: level} if is_study else None, config_changes=None if is_study else {key: level}
        )
        candidates.append(StudyCandidate(candidate, axis, level, source))
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class SubmittedCandidate:
    candidate: StudyCandidate
    job_id: str
    state: dict


def _check_candidates(candidates):
    if not 2 <= len(candidates) <= 32 or not all(isinstance(item, StudyCandidate) for item in candidates):
        raise ValueError("A study requires 2–32 explicit study candidates.")
    baselines = [item for item in candidates if item.axis == "baseline"]
    if len(baselines) != 1 or len({item.project.id for item in candidates}) != len(candidates):
        raise ValueError("A study requires one baseline and distinct candidate project identities.")
    baseline = baselines[0]
    if any(item.source_fingerprint != baseline.project.fingerprint for item in candidates):
        raise ValueError("Candidates do not share the same baseline input identity.")
    levels = set()
    for item in candidates:
        if item.axis == "baseline":
            continue
        if (item.axis, item.value) in levels:
            raise ValueError("A numerical axis cannot repeat the same candidate level.")
        levels.add((item.axis, item.value))
        config, study = baseline.project.values, baseline.project.study
        key = {
            "mesh": "iteration_num_mesh_refinement",
            "quadrature": "gauss_order",
            "independent_sampling": "validation_resolution",
        }[item.axis]
        changed = study if item.axis == "independent_sampling" else config
        if changed[key] == item.value:
            raise ValueError("A refinement candidate must differ from the baseline level.")
        changed[key] = item.value
        if item.project.values != config or item.project.study != study:
            raise ValueError("A convergence candidate changed physical requirements or more than one numerical axis.")
    return baseline


def submit_convergence(candidates, jobs, *, service=DesignService):
    """Use the existing bounded queue; cancel prior submissions on failure."""
    candidates = tuple(candidates)
    _check_candidates(candidates)
    for candidate in candidates:
        service.preflight(candidate.project).raise_for_errors()
    submitted = []
    try:
        for candidate in candidates:
            state = service.submit(candidate.project, jobs)
            submitted.append(SubmittedCandidate(candidate, state["id"], deepcopy(state)))
    except Exception:
        for item in submitted:
            jobs.cancel(item.job_id)
        raise
    return tuple(submitted)


@dataclass(frozen=True, slots=True)
class ConvergenceCriterion:
    metric: str
    relative_tolerance: float = 0.01
    absolute_tolerance: float = 0.0

    def __post_init__(self):
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("A convergence criterion requires a metric name.")
        for name in ("relative_tolerance", "absolute_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError("Change tolerances must be finite and nonnegative.")


DEFAULT_CONVERGENCE_CRITERIA = (
    ConvergenceCriterion("Sheet power", 0.01),
    ConvergenceCriterion("Sheet design relative error", 0, 0.1),  # percentage points
)


@dataclass(frozen=True, slots=True)
class ConvergenceChange:
    axis: str
    earlier: str
    later: str
    metric: str
    unit: str
    left: float | None
    right: float | None
    absolute_change: float | None
    relative_change: float | None
    comparable: bool
    within_tolerance: bool | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ConvergenceSummary:
    changes: tuple[ConvergenceChange, ...]
    stable_axes: tuple[str, ...]
    criteria: tuple[ConvergenceCriterion, ...]
    notes: tuple[str, ...]

    def to_dict(self):
        return {"schema": STUDY_SCHEMA, **asdict(self)}


def summarize_convergence(
    candidates, records: Mapping[str, RunRecord | str | Path], *, criteria=DEFAULT_CONVERGENCE_CRITERIA
):
    """Compare reverified common-scope metrics against explicit change limits.

    ``records`` maps candidate project IDs to completed run directories. Stable
    means the last two levels of a series with at least three distinct levels
    meet all requested change criteria. It is observed numerical stability,
    not a feasibility, manufacturing or mathematical convergence certificate.
    Different independent samples retain their own scopes and never yield a
    mesh/quadrature stability claim.
    """
    candidates, criteria = tuple(candidates), tuple(criteria)
    baseline = _check_candidates(candidates)
    if not criteria or not all(isinstance(item, ConvergenceCriterion) for item in criteria):
        raise ValueError("A study needs one baseline and explicit convergence criteria.")
    if len({item.metric for item in criteria}) != len(criteria):
        raise ValueError("Each metric requires one unambiguous change criterion.")
    verified = {}
    for candidate in candidates:
        source = records.get(candidate.project.id)
        if source is None:
            raise ValueError(f"Completed evidence is missing for {candidate.label}.")
        record = DesignService.load_run(source.directory if isinstance(source, RunRecord) else source)
        metadata = record.metadata
        if "metadata_sha256" not in metadata:
            raise ValueError("Convergence metrics require checksum-protected result metadata.")
        for key, expected in (
            ("project_id", candidate.project.id),
            ("project_revision", candidate.project.revision),
            ("project_fingerprint", candidate.project.fingerprint),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"Completed evidence does not belong to {candidate.label}: {key} differs.")
        verified[candidate.project.id] = record
    changes, stable = [], []
    notes = [
        "Independent sample refinement does not check mesh or quadrature convergence.",
        "Stability reports metric changes, separate from engineering acceptance and construction release.",
    ]
    baseline_values = {
        "mesh": baseline.project.values["iteration_num_mesh_refinement"],
        "quadrature": baseline.project.values["gauss_order"],
        "independent_sampling": baseline.project.study["validation_resolution"],
    }
    for axis in ("mesh", "quadrature", "independent_sampling"):
        levels = [(item.value, item) for item in candidates if item.axis == axis]
        if not levels:
            continue
        levels = sorted([(baseline_values[axis], baseline), *levels], key=lambda item: item[0])
        last_checks = []
        for (_, left), (_, right) in zip(levels, levels[1:]):
            comparison = compare_runs(verified[left.project.id], verified[right.project.id])
            metrics = {item.name: item for item in comparison.metrics}
            last_checks = []
            for criterion in criteria:
                metric = metrics.get(criterion.metric)
                if metric is None:
                    raise ValueError(f"Unknown convergence metric: {criterion.metric}.")
                delta = abs(metric.right - metric.left) if metric.comparable else None
                scale = max(abs(metric.left), abs(metric.right)) if metric.comparable else None
                relative = delta / scale if scale else 0.0 if delta == 0 else None
                passed = (
                    delta <= max(criterion.absolute_tolerance, criterion.relative_tolerance * scale)
                    if (metric.comparable)
                    else None
                )
                last_checks.append(passed)
                changes.append(
                    ConvergenceChange(
                        axis,
                        left.label,
                        right.label,
                        metric.name,
                        metric.unit,
                        metric.left,
                        metric.right,
                        delta,
                        relative,
                        metric.comparable,
                        passed,
                        metric.reason,
                    )
                )
        if axis != "independent_sampling" and len(levels) >= 3 and all(value is True for value in last_checks):
            stable.append(axis)
        if len(levels) < 3:
            notes.append(f"{axis}: fewer than three distinct levels; changes are descriptive only.")
    return ConvergenceSummary(tuple(changes), tuple(stable), criteria, tuple(notes))


MEASUREMENT_SCHEMA = "coilgen-measured-field-v1"
_LENGTH_UNITS = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}
_FIELD_UNITS = {"T": 1.0, "mT": 1e-3, "uT": 1e-6, "µT": 1e-6}


def _xyz(value, name, minimum=1):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} finite real XYZ rows.")
    array = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class RigidTransform:
    rotation: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    translation_m: tuple = (0, 0, 0)

    def __post_init__(self):
        raw_rotation, raw_translation = np.asarray(self.rotation), np.asarray(self.translation_m)
        if raw_rotation.dtype.kind not in "iuf" or raw_translation.dtype.kind not in "iuf":
            raise ValueError("Rigid registration requires real numeric rotation and translation values.")
        rotation = np.asarray(raw_rotation, dtype=float)
        translation = np.asarray(raw_translation, dtype=float)
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(translation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-10)
        ):
            raise ValueError("Registration requires a finite proper rigid rotation and translation in metres.")
        object.__setattr__(self, "rotation", tuple(tuple(float(value) for value in row) for row in rotation))
        object.__setattr__(self, "translation_m", tuple(float(value) for value in translation))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeasuredFieldMap:
    coordinates: np.ndarray
    field: np.ndarray
    length_unit: str = "m"
    field_unit: str = "T"
    independent: bool = False
    source: str = ""
    source_bytes: bytes = b""

    def __post_init__(self):
        if self.length_unit not in _LENGTH_UNITS or self.field_unit not in _FIELD_UNITS:
            raise ValueError("Measured maps require explicit supported length and field units.")
        if (
            not isinstance(self.independent, bool)
            or not isinstance(self.source, str)
            or not isinstance(self.source_bytes, bytes)
        ):
            raise TypeError("Measured provenance requires boolean independence, a source label and owned bytes.")
        coordinates, field = _xyz(self.coordinates, "Measured coordinates"), _xyz(self.field, "Measured vector field")
        if coordinates.shape != field.shape or len(coordinates) > 100_000:
            raise ValueError("Measured coordinates and fields must match, with at most 100,000 samples.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "field", field)

    @property
    def source_sha256(self):
        return sha256(self.source_bytes).hexdigest() if self.source_bytes else None

    @property
    def data_sha256(self):
        digest = sha256(self.coordinates.tobytes() + self.field.tobytes())
        digest.update(f"{self.length_unit}/{self.field_unit}".encode())
        return digest.hexdigest()

    def world_values(self, transform=RigidTransform()):
        if not isinstance(transform, RigidTransform):
            raise TypeError("Measured registration requires a RigidTransform.")
        rotation = np.asarray(transform.rotation)
        points = (self.coordinates * _LENGTH_UNITS[self.length_unit]) @ rotation.T + transform.translation_m
        field = (self.field * _FIELD_UNITS[self.field_unit]) @ rotation.T
        return points, field

    def to_dict(self):
        return {
            "schema": MEASUREMENT_SCHEMA,
            "coordinates": self.coordinates.tolist(),
            "field": self.field.tolist(),
            "length_unit": self.length_unit,
            "field_unit": self.field_unit,
            "independent": self.independent,
            "source": self.source,
            "source_bytes": base64.b64encode(self.source_bytes).decode("ascii"),
            "source_sha256": self.source_sha256,
            "data_sha256": self.data_sha256,
        }

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema",
                "coordinates",
                "field",
                "length_unit",
                "field_unit",
                "independent",
                "source",
                "source_bytes",
                "source_sha256",
                "data_sha256",
            }
            or value["schema"] != MEASUREMENT_SCHEMA
        ):
            raise ValueError("Malformed measured-map record.")
        measured = cls(
            value["coordinates"],
            value["field"],
            value["length_unit"],
            value["field_unit"],
            value["independent"],
            value["source"],
            base64.b64decode(value["source_bytes"], validate=True),
        )
        if measured.data_sha256 != value["data_sha256"] or measured.source_sha256 != value["source_sha256"]:
            raise ValueError("Measured-map data/provenance integrity check failed.")
        return measured

    @classmethod
    def from_file(cls, path, *, independent=False):
        """CSV columns: x_m,y_m,z_m,Bx_T,By_T,Bz_T; portable JSON/NPZ also supported.

        Independence is an explicit declaration, then checked against the run's
        fitted inputs. Original file bytes survive changes or relocation.
        """
        path = Path(path).expanduser().resolve()
        data = path.read_bytes()
        if path.suffix.lower() == ".csv":
            rows = np.genfromtxt(StringIO(data.decode("utf-8-sig")), delimiter=",", names=True, ndmin=1)
            names = ("x_m", "y_m", "z_m", "Bx_T", "By_T", "Bz_T")
            if not rows.dtype.names or not set(names) <= set(rows.dtype.names):
                raise ValueError("Measured CSV requires x_m,y_m,z_m,Bx_T,By_T,Bz_T columns.")
            return cls(
                np.column_stack([rows[name] for name in names[:3]]),
                np.column_stack([rows[name] for name in names[3:]]),
                independent=independent,
                source=str(path),
                source_bytes=data,
            )
        if path.suffix.lower() == ".json":
            measured = cls.from_dict(json.loads(data))
        elif path.suffix.lower() == ".npz":
            with np.load(BytesIO(data), allow_pickle=False) as arrays:
                measured = cls.from_dict(json.loads(str(arrays["measurement_json"])))
        else:
            raise ValueError("Measured data must be CSV or a portable measured-map JSON/NPZ record.")
        return measured

    def save(self, path):
        from ..persistence import atomic_json

        path = Path(path).expanduser().resolve()
        if path.suffix.lower() == ".json":
            atomic_json(path, self.to_dict())
        elif path.suffix.lower() == ".npz":
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, measurement_json=json.dumps(self.to_dict(), allow_nan=False))
        else:
            raise ValueError("Save a portable measured-map .json or .npz file.")
        return path


@dataclass(frozen=True, slots=True)
class MeasuredComparison:
    measurement: MeasuredFieldMap
    transform: RigidTransform
    points_m: np.ndarray
    measured_T: np.ndarray
    predicted_T: np.ndarray
    residual_T: np.ndarray
    provenance: dict
    rms_error_T: float
    peak_error_T: float
    relative_rms_error: float | None
    rms_limit_T: float | None
    peak_limit_T: float | None
    passed: bool | None

    def __post_init__(self):
        for name in ("points_m", "measured_T", "predicted_T", "residual_T"):
            object.__setattr__(self, name, _xyz(getattr(self, name), name))
        object.__setattr__(self, "provenance", _freeze(deepcopy(self.provenance)))

    def to_dict(self):
        from ..persistence import json_value

        return {
            "schema": "coilgen-measured-comparison-v1",
            "measurement": self.measurement.to_dict(),
            "transform": self.transform.to_dict(),
            **{name: getattr(self, name).tolist() for name in ("points_m", "measured_T", "predicted_T", "residual_T")},
            "provenance": json_value(self.provenance),
            **{
                name: getattr(self, name)
                for name in (
                    "rms_error_T",
                    "peak_error_T",
                    "relative_rms_error",
                    "rms_limit_T",
                    "peak_limit_T",
                    "passed",
                )
            },
        }


def saved_winding_currents(record, *, current_policy="original"):
    """List the body identities/current contract required by measured comparison."""
    record = DesignService.load_run(record.directory if isinstance(record, RunRecord) else record)
    paths, currents, source, complete = _saved_winding(record, current_policy)
    return {"source": source, "body_currents_A": currents, "complete_circuit": complete, "path_count": len(paths)}


def _saved_winding(record, current_policy):
    if current_policy not in {"original", "recorded"}:
        raise ValueError("Current policy must be original or recorded; no fitted current is computed.")
    metadata, arrays = record.metadata, record.select_arrays(copy=False)
    plans = metadata.get("winding_steps") or []
    bodies = ((metadata.get("fabrication") or {}).get("electrical_metrics") or {}).get("bodies") or []
    if plans and all(plan.get("circuit_complete") is True for plan in plans):
        by_id = {body.get("body_id"): body for body in bodies}
        ids = [plan.get("body_id") for plan in plans]
        if (
            not all(isinstance(id_, str) and id_ for id_ in ids)
            or len(set(ids)) != len(ids)
            or len(by_id) != len(bodies)
            or set(ids) != set(by_id)
        ):
            raise ValueError("Fabricated electrical paths and recorded currents do not cover the same bodies.")
        paths, currents = {}, {}
        for index, plan in enumerate(plans):
            id_ = plan["body_id"]
            key = f"review_electrical_{index}_points_m"
            if key not in arrays:
                raise ValueError("The complete fabricated electrical path was not saved.")
            paths[id_] = _xyz(arrays[key], f"Saved electrical body {id_}", 2)
            if not np.any(np.linalg.norm(np.diff(paths[id_], axis=0), axis=1) > 1e-15):
                raise ValueError("A saved winding body has no finite-length electrical segments.")
            currents[id_] = by_id[id_].get("current_A")
            if not np.allclose(paths[id_][0], paths[id_][-1], rtol=0, atol=1e-10):
                raise ValueError("A fabricated body claims a complete circuit but its electrical path is open.")
        return paths, _currents(currents), "fabricated_complete_electrical_paths", True
    wire = (metadata.get("wire_fields") or {}).get("routed") or {}
    policy = wire.get("at_stream_current") or {} if current_policy == "original" else wire
    names = sorted(name for name in arrays if name.startswith("evaluated_winding_") and name.endswith("_points_m"))
    if wire.get("available") is not True or policy.get("available") is not True or not names:
        raise ValueError("No saved complete fabricated circuit or evaluated winding/current contract is available.")
    paths = {name.removesuffix("_points_m"): _xyz(arrays[name], name, 2) for name in names}
    currents = _currents({id_: policy.get("current_A") for id_ in paths})
    return paths, currents, "saved_evaluated_winding", False


def _currents(values):
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values.values()
    ):
        raise ValueError("Every winding body requires its finite recorded or calibrated signed current in amperes.")
    return {key: float(value) for key, value in values.items()}


def compare_measured_winding(
    record,
    measurement: MeasuredFieldMap,
    *,
    transform=RigidTransform(),
    body_currents_A=None,
    current_policy="original",
    rms_limit_T=None,
    peak_limit_T=None,
):
    """Evaluate saved electrical geometry directly at independent measured points.

    No interpolation, extrapolated map, current fit, background subtraction or
    registration fit is performed. Measured vectors rotate with their coordinate
    frame. Explicit calibrated currents must cover every body; signed recorded
    currents are retained separately. Samples on thin-wire paths are rejected.
    A partial winding comparison stays descriptive even with specified limits.
    """
    from ..physics import biot_savart_wire_path
    from ..research.validation import _samples_on_wire

    if not isinstance(measurement, MeasuredFieldMap) or not measurement.independent:
        raise ValueError("Measured comparison requires an explicitly independent validation dataset.")
    record = DesignService.load_run(record.directory if isinstance(record, RunRecord) else record)
    metadata, manifest, health = verified_run_metadata(record.directory)
    if not health.metadata_checked:
        raise ValueError("Measured comparison requires checksum-protected winding/current provenance.")
    if measurement.source_sha256 and any(
        item.get("sha256") == measurement.source_sha256 for item in manifest.get("input_files", {}).values()
    ):
        raise ValueError("Measured validation data was used as a design input; supply independent measurements.")
    paths, original_currents, source, complete = _saved_winding(record, current_policy)
    currents = original_currents if body_currents_A is None else _currents(dict(body_currents_A))
    if set(currents) != set(paths):
        raise ValueError("Calibrated signed currents must explicitly cover every saved winding body.")
    points, measured = measurement.world_values(transform)
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(measured)):
        raise ValueError("Registered measured values overflow their SI coordinate or field representation.")
    field = np.zeros_like(points)
    for id_, path in paths.items():
        if _samples_on_wire(path, points):
            raise ValueError("Measured samples intersect a thin-wire centerline; use a finite-cross-section model.")
        field += currents[id_] * biot_savart_wire_path(path, points, worker_count=1)
    residual = field - measured
    rms, peak = _vector_errors(residual)
    reference, _ = _vector_errors(measured)
    for value in (rms_limit_T, peak_limit_T):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0
        ):
            raise ValueError("Measured error limits must be finite nonnegative values in tesla.")
    checks = [error <= limit for error, limit in ((rms, rms_limit_T), (peak, peak_limit_T)) if limit is not None]
    provenance = {
        "request_sha256": metadata["request_sha256"],
        "project_id": metadata.get("project_id"),
        "project_revision": metadata.get("project_revision"),
        "source": source,
        "field_model": "thin_wire_exact_segments",
        "sampling": "measured_locations_only",
        "independent": True,
        "independence_basis": "explicit declaration; excluded from fitted input hashes",
        "recorded_currents_A": original_currents,
        "evaluation_currents_A": currents,
        "current_source": "explicit_calibration" if body_currents_A is not None else current_policy,
        "current_refitted": False,
        "complete_circuit": complete,
        "construction_released": False,
        "measurement_sha256": measurement.data_sha256,
    }
    return MeasuredComparison(
        measurement,
        transform,
        points,
        measured,
        field,
        residual,
        provenance,
        rms,
        peak,
        rms / reference if reference else None,
        rms_limit_T,
        peak_limit_T,
        all(checks) if checks and complete else None,
    )


def _vector_errors(values):
    scale = float(np.max(np.abs(values)))
    if not math.isfinite(scale):
        raise ValueError("Measured field comparison produced nonfinite values.")
    if not scale:
        return 0.0, 0.0
    norms = np.sum((values / scale) ** 2, axis=1)
    rms, peak = scale * float(np.sqrt(np.mean(norms))), scale * float(np.sqrt(np.max(norms)))
    if not math.isfinite(rms) or not math.isfinite(peak):
        raise ValueError("Measured vector errors exceed the finite numerical range.")
    return rms, peak
