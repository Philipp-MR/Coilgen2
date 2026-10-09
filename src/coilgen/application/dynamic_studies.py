"""Portable, geometry-bound circuit references and external coupled-physics evidence.

Import verifies numerical contracts, source integrity and geometry identity. It
does not certify the accuracy of an external solver or replace an EM/FE solver.
Complex frequency responses remain separate from real magnetostatic datasets.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

from ..gradient_engineering import (
    InductanceMatrixSpec,
    _body_ids,
    _hash,
    _numeric,
    _text,
)

SCHEMA = "coilgen.dynamic-study/1"
REFERENCE_SCHEMA = "coilgen.circuit-reference/1"
ANALYSIS_SCHEMA = "coilgen.external-analysis/1"
_MAX_VALUES = 1000000
_QUANTITIES = {
    "frequency_response": {"field_response": ("T/A", 3), "induced_current": ("A/A", 1), "ac_loss": ("W/A^2", 1)},
    "mechanical": {"force": ("N", 3), "displacement": ("m", 3), "stress": ("Pa", 1)},
    "spatial_thermal": {"temperature": ("K", 1), "heat_flux": ("W/m^2", 3)},
}


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _checked(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Unknown or malformed {name} fields.")
    return dict(value)


def _tuple(array):
    return tuple(_tuple(item) if isinstance(item, list) else item for item in np.asarray(array).tolist())


@dataclass(frozen=True, slots=True)
class CircuitReference:
    kind: str
    body_ids: tuple
    path_sha256: tuple
    values: tuple
    source: str
    source_sha256: str
    method: str
    field_component: str | None = None
    gradient_direction_world: tuple | None = None

    def __post_init__(self):
        if self.kind not in {"inductance", "efficiency"}:
            raise ValueError("Circuit reference kind must be inductance or efficiency.")
        ids = _body_ids(self.body_ids)
        if len(self.path_sha256) != len(ids):
            raise ValueError("Circuit references require one electrical path digest per ordered body.")
        for digest in self.path_sha256:
            _hash(digest, "Electrical path digest")
        for name in ("source", "method"):
            _text(getattr(self, name), name)
        _hash(self.source_sha256, "Circuit source file digest")
        values = _numeric(self.values, "Circuit reference values", dimensions=2 if self.kind == "inductance" else 1)
        if self.kind == "inductance":
            InductanceMatrixSpec(ids, _tuple(values), tuple(self.path_sha256), self.source)
        elif values.shape != (len(ids),) or np.any(values == 0):
            raise ValueError("Gradient efficiency requires one finite nonzero signed T/m/A value per body.")
        if self.kind == "efficiency":
            direction = _numeric(self.gradient_direction_world, "Efficiency gradient direction", dimensions=1)
            if direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1, rtol=1e-10, atol=1e-12):
                raise ValueError("Gradient efficiencies require an explicit unit direction in world coordinates.")
            if self.field_component not in {"Bx", "By", "Bz"}:
                raise ValueError("Gradient efficiencies require the explicit Bx, By or Bz field component.")
            object.__setattr__(self, "gradient_direction_world", _tuple(direction))
        elif self.field_component is not None or self.gradient_direction_world is not None:
            raise ValueError("Field component/direction belong to gradient efficiency references.")
        object.__setattr__(self, "body_ids", ids)
        object.__setattr__(self, "path_sha256", tuple(self.path_sha256))
        object.__setattr__(self, "values", _tuple(values))

    @property
    def unit(self):
        return "H" if self.kind == "inductance" else "T/m/A"

    def to_dict(self):
        value = {
            "schema": REFERENCE_SCHEMA,
            "kind": self.kind,
            "body_ids": list(self.body_ids),
            "path_sha256": list(self.path_sha256),
            "values": np.asarray(self.values).tolist(),
            "unit": self.unit,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "method": self.method,
            "field_component": self.field_component,
            "gradient_direction_world": self.gradient_direction_world,
        }
        value["content_sha256"] = _digest(value)
        return value

    @classmethod
    def from_dict(cls, value):
        fields = {
            "schema",
            "kind",
            "body_ids",
            "path_sha256",
            "values",
            "unit",
            "source",
            "source_sha256",
            "method",
            "content_sha256",
            "field_component",
            "gradient_direction_world",
        }
        value = _checked(value, fields, "CircuitReference")
        digest = value.pop("content_sha256", None)
        if digest != _digest(value):
            raise ValueError("Circuit reference content digest does not match its portable payload.")
        if value.pop("schema", None) != REFERENCE_SCHEMA:
            raise ValueError("Unsupported circuit reference schema.")
        unit = value.pop("unit", None)
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ValueError(f"Incomplete circuit reference: {exc}") from exc
        if unit != result.unit:
            raise ValueError("Persisted circuit references must use canonical SI units.")
        return result


def import_circuit_reference(filename):
    """Import an explicitly ordered JSON reference; compute the actual file digest."""
    raw = Path(filename).read_bytes()
    if len(raw) > 2000000:
        raise ValueError("Circuit reference file is too large.")
    value = json.loads(raw)
    if "content_sha256" in value:
        reference = CircuitReference.from_dict(value)
        return CircuitReference(
            reference.kind,
            reference.body_ids,
            reference.path_sha256,
            reference.values,
            reference.source,
            sha256(raw).hexdigest(),
            reference.method,
            reference.field_component,
            reference.gradient_direction_world,
        )
    value = _checked(
        value,
        {
            "schema",
            "kind",
            "body_ids",
            "path_sha256",
            "values",
            "unit",
            "source",
            "source_sha256",
            "method",
            "field_component",
            "gradient_direction_world",
        },
        "circuit import",
    )
    if value.pop("schema", REFERENCE_SCHEMA) != REFERENCE_SCHEMA:
        raise ValueError("Unsupported circuit reference schema.")
    kind, unit = value.get("kind"), value.pop("unit", None)
    factors = {"inductance": {"H": 1, "mH": 1e-3, "uH": 1e-6}, "efficiency": {"T/m/A": 1, "mT/m/A": 1e-3}}
    if unit not in factors.get(kind, {}):
        raise ValueError("Declare inductance units H/mH/uH or gradient efficiency units T/m/A or mT/m/A.")
    values = _numeric(value.pop("values", None), "Reference values", dimensions=2 if kind == "inductance" else 1)
    value["values"] = _tuple(values * factors[kind][unit])
    value["source_sha256"] = sha256(raw).hexdigest()
    try:
        return CircuitReference(**value)
    except TypeError as exc:
        raise ValueError(f"Incomplete circuit import: {exc}") from exc


def import_waveform_csv(filename, body_ids, *, time_unit="s", current_unit="A"):
    """Read headered time/current columns in exact declared body order, without sorting."""
    ids = _body_ids(body_ids)
    time_factors, current_factors = {"s": 1, "ms": 1e-3, "us": 1e-6}, {"A": 1, "mA": 1e-3}
    if time_unit not in time_factors or current_unit not in current_factors:
        raise ValueError("Unsupported waveform time or current units.")
    raw = Path(filename).read_bytes()
    if len(raw) > 8000000:
        raise ValueError("Waveform CSV is too large.")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if not rows or rows[0] != [f"time_{time_unit}", *[f"{body}_{current_unit}" for body in ids]]:
        raise ValueError(
            f"Waveform header must be exactly: time_{time_unit}, " + ", ".join(f"{body}_{current_unit}" for body in ids)
        )
    try:
        values = np.asarray(rows[1:], dtype=float)
    except ValueError as exc:
        raise ValueError("Waveform CSV requires complete finite numeric rows.") from exc
    if (
        values.ndim != 2
        or values.shape[1] != len(ids) + 1
        or not 2 <= len(values) <= 10000
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Waveform CSV requires 2–10000 finite time/current rows in declared body order.")
    times = values[:, 0] * time_factors[time_unit]
    currents = values[:, 1:] * current_factors[current_unit]
    if np.any(np.diff(times) <= 0) or not np.all(np.isfinite(currents)):
        raise ValueError("Waveform times must strictly increase; rows are never silently sorted.")
    return {
        "time_s": times.tolist(),
        "currents_A": currents.tolist(),
        "source": f"{Path(filename).name}; SHA-256 {sha256(raw).hexdigest()}",
    }


@dataclass(frozen=True, slots=True)
class ExternalAnalysis:
    """One external solver result in a declared world/SI sampling and material contract."""

    name: str
    kind: str
    body_ids: tuple
    path_sha256: tuple
    material_sha256: str
    source: str
    source_sha256: str
    solver: str
    solver_version: str
    points_m: tuple
    axis: tuple
    axis_unit: str
    quantity: str
    unit: str
    real: tuple
    environment_sha256: str
    excitation_body_id: str | None = None
    imaginary: tuple | None = None
    waveform_sha256: str | None = None
    limit: float | None = None
    coordinate_frame: str = "world"

    def __post_init__(self):
        for name in ("name", "source", "solver", "solver_version"):
            _text(getattr(self, name), name)
        if self.kind not in _QUANTITIES or self.quantity not in _QUANTITIES[self.kind]:
            raise ValueError("Unsupported external analysis kind/quantity contract.")
        if self.coordinate_frame != "world":
            raise ValueError("External analyses must be explicitly registered to world coordinates in metres.")
        ids = _body_ids(self.body_ids)
        if len(self.path_sha256) != len(ids):
            raise ValueError("External analysis requires one electrical geometry digest per ordered body.")
        for value in (*self.path_sha256, self.material_sha256, self.environment_sha256, self.source_sha256):
            _hash(value, "External analysis provenance")
        if self.waveform_sha256 is not None:
            _hash(self.waveform_sha256, "External analysis waveform")
        if self.kind != "frequency_response" and self.waveform_sha256 is None:
            raise ValueError("Mechanical and spatial thermal evidence require a bound drive waveform digest.")
        if self.kind == "frequency_response" and self.excitation_body_id not in ids:
            raise ValueError(
                "Frequency response requires its explicit unit-current excitation body; other drives are zero."
            )
        if self.kind != "frequency_response" and self.excitation_body_id is not None:
            raise ValueError(
                "Time/mechanical/thermal analyses use the bound waveform, not a unit-current excitation body."
            )
        points = _numeric(self.points_m, "External sample points", dimensions=2)
        axis = _numeric(self.axis, "External sampling axis", dimensions=1)
        if (
            points.shape[1:] != (3,)
            or not 1 <= len(points) <= 100000
            or not 1 <= len(axis) <= 10000
            or np.any(np.diff(axis) <= 0)
        ):
            raise ValueError("Analysis points require Nx3 coordinates and a strictly increasing sampling axis.")
        if self.axis_unit not in {"Hz", "s"} or self.kind == "frequency_response" and self.axis_unit != "Hz":
            raise ValueError("Frequency responses require Hz; other analyses require an explicit Hz or s axis.")
        if self.axis_unit == "Hz" and np.any(axis <= 0) or self.axis_unit == "s" and np.any(axis < 0):
            raise ValueError("Frequencies must be positive; time samples must be nonnegative.")
        unit, components = _QUANTITIES[self.kind][self.quantity]
        if self.unit != unit:
            raise ValueError(f"{self.quantity} requires declared SI unit {unit}.")
        expected_shape = (len(axis), len(points), components)
        real = _numeric(self.real, "External values", dimensions=3)
        if real.shape != expected_shape or real.size > _MAX_VALUES:
            raise ValueError(
                f"External values require axis × samples × {components} components and at most {_MAX_VALUES} entries."
            )
        imaginary = None
        if self.imaginary is not None:
            if self.axis_unit != "Hz" or self.kind == "spatial_thermal" or self.quantity in {"ac_loss", "stress"}:
                raise ValueError(
                    "This quantity uses real values; complex responses require a compatible frequency contract."
                )
            imaginary = _numeric(self.imaginary, "Imaginary external values", dimensions=3)
            if imaginary.shape != expected_shape:
                raise ValueError("Real and imaginary response arrays must have identical shapes.")
        if self.quantity in {"ac_loss", "stress", "temperature"} and np.any(real < 0):
            raise ValueError("Loss, stress and absolute temperature samples must be nonnegative.")
        if self.limit is not None:
            limit = _numeric([self.limit], "External acceptance limit", dimensions=1)[0]
            if limit < 0:
                raise ValueError("External acceptance upper limit must be nonnegative.")
            object.__setattr__(self, "limit", float(limit))
        object.__setattr__(self, "body_ids", ids)
        object.__setattr__(self, "path_sha256", tuple(self.path_sha256))
        for key, value in (("points_m", points), ("axis", axis), ("real", real), ("imaginary", imaginary)):
            object.__setattr__(self, key, None if value is None else _tuple(value))

    def to_dict(self):
        from dataclasses import asdict

        value = json.loads(json.dumps(asdict(self), allow_nan=False))
        value["schema"] = ANALYSIS_SCHEMA
        value["content_sha256"] = _digest(value)
        return value

    @classmethod
    def from_dict(cls, value):
        from dataclasses import fields

        value = _checked(
            value, {field.name for field in fields(cls)} | {"schema", "content_sha256"}, "ExternalAnalysis"
        )
        checksum = value.pop("content_sha256", None)
        if checksum != _digest(value):
            raise ValueError("External analysis content digest does not match its payload.")
        if value.pop("schema", None) != ANALYSIS_SCHEMA:
            raise ValueError("Unsupported external analysis schema.")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f"Incomplete external analysis: {exc}") from exc


def import_external_analysis(filename):
    raw = Path(filename).read_bytes()
    if len(raw) > 32000000:
        raise ValueError("External analysis file exceeds the 32 MB portable import limit.")
    value = json.loads(raw)
    if "content_sha256" in value:
        from dataclasses import replace

        return replace(ExternalAnalysis.from_dict(value), source_sha256=sha256(raw).hexdigest())
    value = dict(value)
    if value.pop("schema", None) != ANALYSIS_SCHEMA:
        raise ValueError("Unsupported external analysis schema.")
    value["source_sha256"] = sha256(raw).hexdigest()
    from dataclasses import fields

    value = _checked(value, {field.name for field in fields(ExternalAnalysis)}, "ExternalAnalysis")
    try:
        return ExternalAnalysis(**value)
    except TypeError as exc:
        raise ValueError(f"Incomplete external analysis: {exc}") from exc


@dataclass(frozen=True, slots=True)
class DynamicStudySpec:
    inductance_reference: CircuitReference | None = None
    efficiency_reference: CircuitReference | None = None
    analyses: tuple = ()
    material_sha256: str | None = None
    environment_sha256: str | None = None
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ValueError("Unsupported dynamic study schema.")
        for name, kind in (("inductance_reference", "inductance"), ("efficiency_reference", "efficiency")):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = CircuitReference.from_dict(value)
            if value is not None and (not isinstance(value, CircuitReference) or value.kind != kind):
                raise ValueError(f"Invalid {kind} reference.")
            object.__setattr__(self, name, value)
        if not isinstance(self.analyses, (tuple, list)) or len(self.analyses) > 32:
            raise ValueError("A dynamic study supports at most 32 external analyses.")
        analyses = tuple(ExternalAnalysis.from_dict(item) if isinstance(item, dict) else item for item in self.analyses)
        if any(not isinstance(item, ExternalAnalysis) for item in analyses) or len(
            {item.name for item in analyses}
        ) != len(analyses):
            raise ValueError("External analyses require unique names.")
        if sum(np.asarray(item.real).size for item in analyses) > _MAX_VALUES:
            raise ValueError("Combined external evidence exceeds the portable sample limit.")
        if self.material_sha256 is not None:
            _hash(self.material_sha256, "Current assembly material contract")
        if self.environment_sha256 is not None:
            _hash(self.environment_sha256, "Current external environment geometry contract")
        object.__setattr__(self, "analyses", analyses)

    @classmethod
    def from_dict(cls, value):
        value = _checked(
            value,
            {
                "inductance_reference",
                "efficiency_reference",
                "analyses",
                "material_sha256",
                "environment_sha256",
                "schema",
            },
            "DynamicStudySpec",
        )
        return cls(**value)

    def to_dict(self):
        return {
            "schema": self.schema,
            "inductance_reference": self.inductance_reference.to_dict() if self.inductance_reference else None,
            "efficiency_reference": self.efficiency_reference.to_dict() if self.efficiency_reference else None,
            "analyses": [item.to_dict() for item in self.analyses],
            "material_sha256": self.material_sha256,
            "environment_sha256": self.environment_sha256,
        }


def verified_circuit_references(spec, fabrication, engineering):
    """Produce evaluator references only after rechecking exact saved complete paths."""
    spec = DynamicStudySpec.from_dict(spec) if isinstance(spec, dict) else spec
    drive = engineering.drive
    if drive is None:
        return {}
    plans = {item["body_id"]: item for item in fabrication.winding_steps}
    path_hashes = tuple(plans.get(body, {}).get("electrical_path_sha256") for body in drive.body_ids)
    output = {}
    for reference in (spec.inductance_reference, spec.efficiency_reference):
        if reference is None or reference.body_ids != drive.body_ids or reference.path_sha256 != path_hashes:
            continue
        if any(not plans.get(body, {}).get("drive_feed_designed") for body in drive.body_ids):
            continue
        if reference.kind == "inductance":
            output["verified_inductance_reference"] = {
                "body_ids": list(reference.body_ids),
                "path_sha256": list(reference.path_sha256),
                "values_H": np.asarray(reference.values).tolist(),
                "source": reference.source,
                "source_sha256": reference.source_sha256,
            }
        else:
            output["verified_efficiency_reference"] = {
                "body_ids": list(reference.body_ids),
                "path_sha256": list(reference.path_sha256),
                "efficiencies_T_per_m_per_A": list(reference.values),
                "source_sha256": reference.source_sha256,
                "field_component": reference.field_component,
                "gradient_direction_world": list(reference.gradient_direction_world),
            }
    return output


def waveform_digest(drive):
    """Bind external finite-time analyses to body order, signed current and repetitions."""
    if drive is None or not drive.time_s:
        return None
    return _digest(
        {
            "body_ids": list(drive.body_ids),
            "time_s": list(drive.time_s),
            "currents_A": [list(row) for row in drive.currents_A],
            "repetitions": drive.repetitions,
            "rest_s": drive.rest_s,
        }
    )


def review_dynamic_analyses(spec, fabrication, engineering=None, engineering_review=None):
    """Report scoped sampled limits; missing or stale bindings yield unverified evidence."""
    spec = DynamicStudySpec.from_dict(spec) if isinstance(spec, dict) else spec
    plans = {item["body_id"]: item for item in fabrication.winding_steps} if fabrication is not None else {}
    current_waveform = waveform_digest(engineering.drive if engineering else None)
    records, checks = [], []
    for analysis in spec.analyses:
        reasons = []
        if (
            tuple(plans.get(body, {}).get("electrical_path_sha256") for body in analysis.body_ids)
            != analysis.path_sha256
        ):
            reasons.append("Electrical path geometry/body identity is missing or stale.")
        if spec.material_sha256 != analysis.material_sha256:
            reasons.append("Material contract is missing or differs from the external analysis.")
        if spec.environment_sha256 != analysis.environment_sha256:
            reasons.append("External environment geometry contract is missing or differs from the analysis.")
        if analysis.waveform_sha256 is not None and analysis.waveform_sha256 != current_waveform:
            reasons.append("Signed body waveform/repetition contract is missing or stale.")
        values = np.asarray(analysis.real) + (
            1j * np.asarray(analysis.imaginary) if analysis.imaginary is not None else 0
        )
        magnitude = np.linalg.norm(values, axis=2) if values.shape[2] == 3 else np.abs(values[:, :, 0])
        witness = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        peak = float(magnitude[witness])
        witness_components = values[:, witness[1], :]
        phases = np.angle(witness_components, deg=True)
        phase_values = [
            [float(value) if abs(witness_components[row, column]) > 0 else None for column, value in enumerate(phase)]
            for row, phase in enumerate(phases)
        ]
        available = not reasons
        passed = bool(peak <= analysis.limit) if available and analysis.limit is not None else None
        records.append(
            {
                "name": analysis.name,
                "kind": analysis.kind,
                "quantity": analysis.quantity,
                "available": available,
                "binding_verified": available,
                "solver_accuracy_verified": False,
                "source": analysis.source,
                "source_sha256": analysis.source_sha256,
                "solver": analysis.solver,
                "solver_version": analysis.solver_version,
                "material_sha256": analysis.material_sha256,
                "environment_sha256": analysis.environment_sha256,
                "waveform_sha256": analysis.waveform_sha256,
                "excitation_body_id": analysis.excitation_body_id,
                "peak": peak,
                "unit": analysis.unit,
                "limit": analysis.limit,
                "passed": passed,
                "axis": list(analysis.axis),
                "axis_unit": analysis.axis_unit,
                "peak_by_axis": np.max(magnitude, axis=1).tolist(),
                "limiting_axis_value": analysis.axis[witness[0]],
                "limiting_point_m": list(analysis.points_m[witness[1]]),
                "complex_response": analysis.imaginary is not None,
                "reasons": reasons,
                "witness_component_amplitudes": np.abs(witness_components).tolist(),
                "witness_component_phase_deg": phase_values if analysis.imaginary is not None else None,
                "scope": "Supplied external solver samples and declared material/geometry/drive contract; no continuous-domain qualification.",
                "qualification_claim": False,
            }
        )
        checks.append(
            {
                "name": f"External {analysis.kind}: {analysis.name}",
                "value": peak if available else None,
                "limit": analysis.limit,
                "unit": analysis.unit,
                "passed": passed,
                "verified": available,
                "scope": f"external_sampled_{analysis.kind}",
                "reason": "; ".join(reasons)
                or (
                    "No acceptance upper limit was supplied."
                    if analysis.limit is None
                    else "External solver accuracy is separately assessed."
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "qualification_claim": False,
        "analyses": records,
        "checks": checks,
        "engineering_review": engineering_review,
        "unassessed_physics": sorted(set(_QUANTITIES) - {item.kind for item in spec.analyses}),
    }
