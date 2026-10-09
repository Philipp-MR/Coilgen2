"""Explicit gradient circuits and bounded, quasistatic drive estimates in SI.

Authoring a feed or a waveform does not qualify a manufactured coil. Geometry
is bound to exact saved winding paths; no missing conductor is inferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math
from numbers import Real
import re

import numpy as np

from .engineering_metrics import DriveMetrics


SCHEMA = "coilgen.gradient-engineering/1"
_MAX_POINTS = 100000
_MAX_SAMPLES = 10000
_MAX_BODIES = 64
_ENDPOINT_TOLERANCE_M = 1e-9


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f"{name} requires a nonempty source/name string.")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} requires a lowercase SHA-256 digest.")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive in SI units.")
    return float(value)


def _numeric(value, name, *, dimensions):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or array.ndim != dimensions or not np.all(np.isfinite(array)):
            raise ValueError
        return np.asarray(array, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} requires finite real numerical values.") from None


def _polyline(value, name):
    points = _numeric(value, name, dimensions=2)
    if points.shape[1:] != (3,) or not 2 <= len(points) <= _MAX_POINTS:
        raise ValueError(f"{name} requires 2–{_MAX_POINTS} XYZ points in metres.")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 1e-12):
        raise ValueError(f"{name} contains a zero-length or nonfinite interval.")
    return tuple(tuple(float(v) for v in row) for row in points)


def _body_ids(value):
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= _MAX_BODIES:
        raise ValueError(f"body_ids requires 1–{_MAX_BODIES} explicit fabrication bodies.")
    if any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", item) for item in value):
        raise ValueError("Fabrication body IDs must be portable names.")
    if len(set(value)) != len(value):
        raise ValueError("Fabrication body IDs must be unique.")
    return tuple(value)


def _from_dict(cls, value):
    if not isinstance(value, dict) or set(value) - {item.name for item in fields(cls)}:
        raise ValueError(f"Unknown or malformed {cls.__name__} fields.")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"Incomplete {cls.__name__}: {exc}") from exc


def winding_path_sha256(points) -> str:
    """Digest the exact ordered physical-centroid points, including closure."""
    points = np.asarray(_polyline(points, "winding path"), dtype="<f8")
    return sha256(str(points.shape).encode("ascii") + points.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class CircuitSpec:
    body_id: str
    expected_path_sha256: str
    supply_points_m: tuple
    return_points_m: tuple
    source: str
    positive_terminal: str = "drive+"
    negative_terminal: str = "drive-"
    feed_break_segment: int | None = None
    polarity: int = 1
    drive_closure_points_m: tuple | None = None
    coordinate_frame: str = "world"
    length_unit: str = "m"

    def __post_init__(self):
        _body_ids([self.body_id])
        _hash(self.expected_path_sha256, "expected_path_sha256")
        _text(self.source, "Circuit source")
        _text(self.positive_terminal, "Positive terminal")
        _text(self.negative_terminal, "Negative terminal")
        if self.positive_terminal == self.negative_terminal:
            raise ValueError("Positive and negative terminals require distinct names.")
        if self.coordinate_frame != "world" or self.length_unit != "m":
            raise ValueError("Circuit geometry must use world coordinates in metres.")
        if isinstance(self.polarity, bool) or not isinstance(self.polarity, int) or self.polarity not in {-1, 1}:
            raise ValueError("Circuit polarity must explicitly be +1 or -1.")
        if self.feed_break_segment is not None and (
            isinstance(self.feed_break_segment, bool)
            or not isinstance(self.feed_break_segment, int)
            or self.feed_break_segment < 0
        ):
            raise ValueError("feed_break_segment must be a nonnegative integer or null.")
        for name in ("supply_points_m", "return_points_m", "drive_closure_points_m"):
            value = getattr(self, name)
            if value is not None or name != "drive_closure_points_m":
                object.__setattr__(self, name, _polyline(value, name))

    @classmethod
    def from_dict(cls, value):
        return _from_dict(cls, value)

    def to_dict(self):
        return asdict(self)

    def construct_path(self, points, *, closed):
        """Retain saved winding intervals and append only supplied conductor."""
        points = np.asarray(_polyline(points, "recorded winding"), dtype=float)
        if winding_path_sha256(points) != self.expected_path_sha256:
            raise ValueError(f"{self.body_id}: circuit references stale winding geometry.")
        if closed:
            if np.linalg.norm(points[-1] - points[0]) > _ENDPOINT_TOLERANCE_M:
                raise ValueError("A closed winding must explicitly contain its closing point.")
            count = len(points) - 1
            cut = self.feed_break_segment
            if cut is None or cut >= count or count < 3:
                raise ValueError("A closed winding requires a valid explicit feed-break interval.")
            indices = [(cut + 1 + index) % count for index in range(count)]
            winding = points[indices].copy()
            source_intervals = indices[:-1]
        else:
            if self.feed_break_segment is not None:
                raise ValueError("An open winding cannot have a closed-turn feed break.")
            winding = points.copy()
            source_intervals = list(range(len(winding) - 1))
        if self.polarity == -1:
            winding = winding[::-1].copy()
            source_intervals = source_intervals[::-1]
        supply, returned = np.asarray(self.supply_points_m).copy(), np.asarray(self.return_points_m).copy()
        if np.linalg.norm(supply[-1] - winding[0]) > _ENDPOINT_TOLERANCE_M:
            raise ValueError("Supply geometry must terminate at the retained winding start.")
        if np.linalg.norm(returned[0] - winding[-1]) > _ENDPOINT_TOLERANCE_M:
            raise ValueError("Return geometry must start at the retained winding finish.")
        supply[-1], returned[0] = winding[0], winding[-1]
        pieces = [supply, winding[1:], returned[1:]]
        roles = ["supply"] * (len(supply) - 1) + ["winding"] * (len(winding) - 1) + ["return"] * (len(returned) - 1)
        interval_sources = [None] * (len(supply) - 1) + source_intervals + [None] * (len(returned) - 1)
        if self.drive_closure_points_m is None:
            if np.linalg.norm(returned[-1] - supply[0]) > _ENDPOINT_TOLERANCE_M:
                raise ValueError("Distinct drive terminal positions require explicit drive_closure_points_m.")
            pieces[-1][-1] = supply[0]
        else:
            closure = np.asarray(self.drive_closure_points_m).copy()
            if (
                np.linalg.norm(closure[0] - returned[-1]) > _ENDPOINT_TOLERANCE_M
                or np.linalg.norm(closure[-1] - supply[0]) > _ENDPOINT_TOLERANCE_M
            ):
                raise ValueError("Drive closure must run from the negative terminal to the positive terminal.")
            closure[0], closure[-1] = returned[-1], supply[0]
            pieces.append(closure[1:])
            roles.extend(["drive_closure"] * (len(closure) - 1))
            interval_sources.extend([None] * (len(closure) - 1))
        electrical = np.vstack(pieces)
        _polyline(electrical, "complete electrical path")
        return {
            "winding_points_m": winding,
            "electrical_points_m": electrical,
            "segment_roles": roles,
            "source_intervals": interval_sources,
            "electrical_path_sha256": winding_path_sha256(electrical),
            "positive_terminal_xyz_m": supply[0].tolist(),
            "negative_terminal_xyz_m": returned[-1].tolist(),
        }


@dataclass(frozen=True, slots=True)
class InductanceMatrixSpec:
    body_ids: tuple
    values_H: tuple
    expected_electrical_path_sha256: tuple
    source: str
    verified: bool = False

    def __post_init__(self):
        ids = _body_ids(self.body_ids)
        values = _numeric(self.values_H, "Inductance matrix", dimensions=2)
        if values.shape != (len(ids), len(ids)):
            raise ValueError("Inductance matrix dimensions must match its body IDs.")
        scale = max(float(np.max(np.abs(values))), 1e-15)
        if (
            not np.allclose(values, values.T, rtol=1e-10, atol=scale * 1e-12)
            or np.min(np.linalg.eigvalsh(values)) < -scale * 1e-10
        ):
            raise ValueError("Inductance matrix must be symmetric and positive semidefinite.")
        if len(self.expected_electrical_path_sha256) != len(ids):
            raise ValueError("Inductance matrix requires an electrical geometry digest for each body.")
        for value in self.expected_electrical_path_sha256:
            _hash(value, "Inductance matrix geometry")
        _text(self.source, "Inductance matrix source")
        if not isinstance(self.verified, bool):
            raise ValueError(
                "Matrix verified must be a boolean; user declarations do not establish source verification."
            )
        object.__setattr__(self, "body_ids", ids)
        object.__setattr__(self, "values_H", tuple(tuple(float(v) for v in row) for row in values))
        object.__setattr__(self, "expected_electrical_path_sha256", tuple(self.expected_electrical_path_sha256))

    @classmethod
    def from_dict(cls, value):
        return _from_dict(cls, value)


@dataclass(frozen=True, slots=True)
class ThermalSpec:
    thermal_resistance_K_per_W: float
    heat_capacity_J_per_K: float
    ambient_temperature_K: float
    initial_temperature_K: float
    max_temperature_K: float
    source: str
    measured: bool = False

    def __post_init__(self):
        for name in (
            "thermal_resistance_K_per_W",
            "heat_capacity_J_per_K",
            "ambient_temperature_K",
            "initial_temperature_K",
            "max_temperature_K",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        _text(self.source, "Thermal parameter source")
        if self.measured is not True:
            raise ValueError("A thermal model requires explicitly measured lumped parameters.")

    @classmethod
    def from_dict(cls, value):
        return _from_dict(cls, value)


@dataclass(frozen=True, slots=True)
class DriveSpec:
    body_ids: tuple
    current_limits_A: tuple
    voltage_limits_V: tuple
    source: str
    time_s: tuple = ()
    currents_A: tuple = ()
    target_gradient_T_per_m: float | None = None
    target_slew_T_per_m_per_s: float | None = None
    gradient_efficiencies_T_per_m_per_A: tuple | None = None
    gradient_efficiency_path_sha256: tuple | None = None
    gradient_efficiency_source_sha256: str | None = None
    gradient_efficiency_source: str | None = None
    inductance_matrix: InductanceMatrixSpec | None = None
    thermal: ThermalSpec | None = None
    repetitions: int = 1
    rest_s: float = 0.0

    def __post_init__(self):
        ids = _body_ids(self.body_ids)
        object.__setattr__(self, "body_ids", ids)
        _text(self.source, "Drive source")
        for name in ("current_limits_A", "voltage_limits_V"):
            values = _numeric(getattr(self, name), name, dimensions=1)
            if len(values) != len(ids) or np.any(values <= 0):
                raise ValueError(f"{name} requires one positive limit per body.")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or not 1 <= self.repetitions <= 1000
        ):
            raise ValueError("Finite waveform repetitions must be an integer from 1 to 1000.")
        if (
            isinstance(self.rest_s, bool)
            or not isinstance(self.rest_s, Real)
            or not math.isfinite(self.rest_s)
            or self.rest_s < 0
        ):
            raise ValueError("Inter-repetition rest_s must be finite and nonnegative.")
        object.__setattr__(self, "rest_s", float(self.rest_s))
        waveform = len(self.time_s) > 0 or len(self.currents_A) > 0
        target = self.target_gradient_T_per_m is not None or self.target_slew_T_per_m_per_s is not None
        if waveform == target:
            raise ValueError("Drive requires exactly one explicit waveform or target-gradient/slew mode.")
        if waveform:
            times = _numeric(self.time_s, "Waveform time_s", dimensions=1)
            currents = _numeric(self.currents_A, "Waveform currents_A", dimensions=2)
            if not 2 <= len(times) <= _MAX_SAMPLES or currents.shape != (len(times), len(ids)):
                raise ValueError(f"Waveform requires 2–{_MAX_SAMPLES} times and an NxBodies current matrix.")
            if np.any(np.diff(times) <= 0) or not np.isfinite(times[-1] - times[0]):
                raise ValueError("Waveform times must strictly increase over a finite duration.")
            if any(
                value is not None
                for value in (
                    self.gradient_efficiencies_T_per_m_per_A,
                    self.gradient_efficiency_path_sha256,
                    self.gradient_efficiency_source_sha256,
                    self.gradient_efficiency_source,
                )
            ):
                raise ValueError("Waveform mode does not use an inferred gradient efficiency.")
            object.__setattr__(self, "time_s", tuple(float(value) for value in times))
            object.__setattr__(self, "currents_A", tuple(tuple(float(v) for v in row) for row in currents))
            expanded_count = (len(times) - 1) * self.repetitions + 1 + (self.repetitions - 1 if self.rest_s > 0 else 0)
            if expanded_count > _MAX_SAMPLES:
                raise ValueError("Finite waveform repetitions must contain at most 10000 expanded samples.")
            if self.repetitions > 1 and (np.any(currents[0] != 0) or np.any(currents[-1] != 0)):
                raise ValueError(
                    "Repeated pulses require zero current at both endpoints; no instantaneous current jump is inferred."
                )
            if self.repetitions > 1:
                expand_drive_waveform(self)
        else:
            if self.repetitions != 1 or self.rest_s != 0:
                raise ValueError("Repetition and rest require an explicit current waveform.")
            object.__setattr__(self, "time_s", ())
            object.__setattr__(self, "currents_A", ())
            for name in ("target_gradient_T_per_m", "target_slew_T_per_m_per_s"):
                object.__setattr__(self, name, _positive(getattr(self, name), name))
            efficiencies = _numeric(self.gradient_efficiencies_T_per_m_per_A, "Gradient efficiencies", dimensions=1)
            if len(efficiencies) != len(ids) or np.any(efficiencies == 0):
                raise ValueError("Target mode requires one finite nonzero explicit gradient efficiency per body.")
            if self.gradient_efficiency_path_sha256 is None or len(self.gradient_efficiency_path_sha256) != len(ids):
                raise ValueError("Target mode requires exact electrical geometry references for its efficiencies.")
            for value in self.gradient_efficiency_path_sha256:
                _hash(value, "Gradient efficiency geometry")
            _hash(self.gradient_efficiency_source_sha256, "Gradient efficiency evidence")
            _text(self.gradient_efficiency_source, "Gradient efficiency source")
            object.__setattr__(
                self, "gradient_efficiencies_T_per_m_per_A", tuple(float(value) for value in efficiencies)
            )
            object.__setattr__(self, "gradient_efficiency_path_sha256", tuple(self.gradient_efficiency_path_sha256))
        for name, cls in (("inductance_matrix", InductanceMatrixSpec), ("thermal", ThermalSpec)):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = cls.from_dict(value)
            if value is not None and not isinstance(value, cls):
                raise ValueError(f"{name} has an invalid specification.")
            object.__setattr__(self, name, value)
        if self.inductance_matrix is not None and self.inductance_matrix.body_ids != ids:
            raise ValueError("Inductance matrix body order must exactly match the drive body order.")

    @classmethod
    def from_dict(cls, value):
        return _from_dict(cls, value)

    def to_dict(self):
        value = asdict(self)
        if self.repetitions == 1:
            value.pop("repetitions")
        if self.rest_s == 0:
            value.pop("rest_s")
        return value


@dataclass(frozen=True, slots=True)
class GradientEngineeringSpec:
    circuits: tuple = ()
    drive: DriveSpec | None = None
    independent_winding_validation: bool = True
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ValueError("Unsupported gradient engineering schema.")
        if not isinstance(self.independent_winding_validation, bool):
            raise ValueError("independent_winding_validation must be a boolean.")
        if not isinstance(self.circuits, (tuple, list)) or len(self.circuits) > _MAX_BODIES:
            raise ValueError("circuits requires a bounded list of explicit fabrication bodies.")
        circuits = tuple(CircuitSpec.from_dict(value) if isinstance(value, dict) else value for value in self.circuits)
        if any(not isinstance(value, CircuitSpec) for value in circuits) or len(
            {value.body_id for value in circuits}
        ) != len(circuits):
            raise ValueError("Circuit specifications require unique explicit body IDs.")
        object.__setattr__(self, "circuits", circuits)
        drive = DriveSpec.from_dict(self.drive) if isinstance(self.drive, dict) else self.drive
        if drive is not None and not isinstance(drive, DriveSpec):
            raise ValueError("Invalid gradient drive specification.")
        if drive is not None and set(drive.body_ids) - set(self.circuit_map):
            raise ValueError("Drive bodies require explicit complete circuit specifications.")
        object.__setattr__(self, "drive", drive)

    @property
    def circuit_map(self):
        return {value.body_id: value for value in self.circuits}

    @classmethod
    def from_dict(cls, value):
        return _from_dict(cls, value)

    def to_dict(self):
        value = asdict(self)
        if self.drive is not None:
            value["drive"] = self.drive.to_dict()
        return json.loads(json.dumps(value, allow_nan=False))


def expand_drive_waveform(drive):
    """Expand the explicitly finite pulse train, retaining zero-current rest intervals."""
    times, currents = np.asarray(drive.time_s, dtype=float), np.asarray(drive.currents_A, dtype=float)
    if drive.repetitions == 1:
        return times, currents
    duration = times[-1] - times[0]
    all_times, all_currents = [times], [currents]
    for index in range(1, drive.repetitions):
        shifted = times + index * (duration + drive.rest_s)
        # A positive rest contributes a zero-current segment before the next pulse.
        start = 0 if drive.rest_s > 0 else 1
        all_times.append(shifted[start:])
        all_currents.append(currents[start:])
    expanded_times, expanded_currents = np.concatenate(all_times), np.vstack(all_currents)
    if not np.all(np.isfinite(expanded_times)) or np.any(np.diff(expanded_times) <= 0):
        raise ValueError("Expanded pulse train is outside finite, strictly increasing SI time range.")
    return expanded_times, expanded_currents


def _thermal_waveform(times, currents, resistances, thermal):
    """Exact first-order lumped response to piecewise-linear supplied current."""
    from scipy.optimize import brentq

    tau = thermal.thermal_resistance_K_per_W * thermal.heat_capacity_J_per_K
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Measured thermal time constant must be finite and positive.")
    temperature = thermal.initial_temperature_K - thermal.ambient_temperature_K
    peak = temperature
    samples = [thermal.initial_temperature_K]
    for index, duration in enumerate(np.diff(times)):
        initial = currents[index]
        slope = (currents[index + 1] - initial) / duration
        q0 = float(resistances @ (initial * initial))
        q1 = float(2 * resistances @ (initial * slope))
        q2 = float(resistances @ (slope * slope))
        start_temperature = temperature

        def at(t):
            x = t / tau
            if x < 0.1:
                # Normalized convolution moments, evaluated without cancellation.
                moments = []
                for degree in range(3):
                    term = 1 / (degree + 1)
                    total = term
                    for order in range(1, 30):
                        term *= -x / (degree + order + 1)
                        total += term
                        if abs(term) < 1e-16 * abs(total):
                            break
                    moments.append(t ** (degree + 1) * total)
            else:
                tail = -math.expm1(-x)
                moments = [tau * tail, tau * t - tau**2 * tail, tau * t**2 - 2 * tau**2 * t + 2 * tau**3 * tail]
            return (
                math.exp(-x) * start_temperature
                + (q0 * moments[0] + q1 * moments[1] + q2 * moments[2]) / thermal.heat_capacity_J_per_K
            )

        def derivative(t):
            power = q0 + q1 * t + q2 * t * t
            return (power - at(t) / thermal.thermal_resistance_K_per_W) / thermal.heat_capacity_J_per_K

        boundaries = [0.0, float(duration)]
        if q2 > 0:
            minimum = -q1 / (2 * q2)
            if 0 < minimum < duration:
                boundaries.insert(1, minimum)
        candidates = list(boundaries)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if derivative(left) * derivative(right) < 0:
                candidates.append(brentq(derivative, left, right, xtol=max(1e-15, duration * 1e-12)))
        peak = max(peak, *(at(value) for value in candidates))
        temperature = at(duration)
        samples.append(thermal.ambient_temperature_K + temperature)
    peak += thermal.ambient_temperature_K
    if not np.all(np.isfinite([peak, *samples])):
        raise ValueError("Thermal response exceeds finite SI reporting range.")
    return {
        "available": True,
        "model": "measured_shared_lumped_RC_constant_electrical_resistance",
        "source": thermal.source,
        "time_constant_s": tau,
        "peak_temperature_K": peak,
        "temperature_limit_K": thermal.max_temperature_K,
        "temperature_margin_K": thermal.max_temperature_K - peak,
        "within_model_limit": peak <= thermal.max_temperature_K,
        "sample_temperatures_K": samples,
        "qualification_claim": False,
    }


def evaluate_gradient_engineering(
    spec, fabrication_report, *, verified_efficiency_reference=None, verified_inductance_reference=None
):
    """Estimate explicit circuits/drives without modifying the optimized design.

    Trusted references are supplied by pipeline evidence verification, never by
    authoring a ``verified`` flag in the serialized engineering specification.
    """
    spec = GradientEngineeringSpec.from_dict(spec) if isinstance(spec, dict) else spec
    if not isinstance(spec, GradientEngineeringSpec):
        raise TypeError("spec must be a GradientEngineeringSpec.")
    output = {
        "schema": SCHEMA,
        "qualification_claim": False,
        "independent_winding_validation_requested": spec.independent_winding_validation,
        "drive": {"available": False, "reason": "No drive specification was supplied."},
        "thermal": {
            "available": False,
            "reason": "Thermal response was not assessed; measured lumped parameters are absent.",
        },
    }
    drive = spec.drive
    if drive is None:
        return output
    metrics = {row["body_id"]: row for row in fabrication_report.electrical_metrics.get("bodies", ())}
    plans = {row["body_id"]: row for row in fabrication_report.winding_steps}
    if any(
        body not in metrics
        or body not in plans
        or not plans[body].get("drive_feed_designed")
        or plans[body].get("expected_winding_path_sha256") != spec.circuit_map[body].expected_path_sha256
        for body in drive.body_ids
    ):
        output["drive"]["reason"] = "Drive estimates require every referenced explicit complete electrical path."
        return output
    selected = [metrics[body] for body in drive.body_ids]
    resistance = np.asarray([row["resistance_ohm"] for row in selected])
    inductance = np.asarray([row["inductance_H"] for row in selected])
    if not np.all(np.isfinite([resistance, inductance])) or np.any(resistance < 0) or np.any(inductance < 0):
        output["drive"]["reason"] = "Circuit R/L evidence is unavailable, negative or nonfinite."
        return output
    path_hashes = [plans[body]["electrical_path_sha256"] for body in drive.body_ids]
    matrix = np.diag(inductance)
    mutual = False
    matrix_note = "Independent self-inductances only; mutual terms are excluded."
    supplied = drive.inductance_matrix
    if supplied is not None:
        reference = verified_inductance_reference or {}
        matches = (
            tuple(path_hashes) == supplied.expected_electrical_path_sha256
            and reference.get("body_ids") == list(drive.body_ids)
            and reference.get("path_sha256") == path_hashes
            and reference.get("values_H") == [list(row) for row in supplied.values_H]
            and reference.get("source") == supplied.source
        )
        if matches:
            matrix = np.asarray(supplied.values_H)
            mutual = True
            matrix_note = "Explicit verified geometry-matched inductance matrix, including mutual terms."
        else:
            matrix_note += " The supplied matrix has no matching verified source/geometry reference."
    if drive.time_s:
        times, currents = expand_drive_waveform(drive)
        duration = times[-1] - times[0]
        slopes = np.diff(currents, axis=0) / np.diff(times)[:, None]
        inductive = slopes @ matrix.T
        voltages = np.stack((currents[:-1] * resistance + inductive, currents[1:] * resistance + inductive))
        peak_current = np.max(np.abs(currents), axis=0)
        peak_voltage = np.max(np.abs(voltages), axis=(0, 1))
        scale = np.max(np.abs(currents), axis=0)
        normalized = currents / np.where(scale > 0, scale, 1)
        mean_square = (
            np.sum(
                np.diff(times)[:, None]
                * (normalized[:-1] ** 2 + normalized[:-1] * normalized[1:] + normalized[1:] ** 2)
                / 3,
                axis=0,
            )
            / duration
        )
        rms = scale * np.sqrt(mean_square)
        power = (np.sqrt(resistance) * rms) ** 2
        scope = "Finite piecewise-linear supplied waveform; voltage extrema include both sides of every slope change."
        target_metrics = None
        if drive.thermal is not None:
            try:
                output["thermal"] = _thermal_waveform(times, currents, resistance, drive.thermal)
            except (ValueError, OverflowError, FloatingPointError) as exc:
                output["thermal"] = {
                    "available": False,
                    "reason": str(exc) or "Thermal response is outside finite SI reporting range.",
                }
        output["waveform"] = {
            "body_ids": list(drive.body_ids),
            "time_s": times.tolist(),
            "currents_A": currents.tolist(),
            "interval_voltage_start_V": voltages[0].tolist(),
            "interval_voltage_end_V": voltages[1].tolist(),
            "repetitions": drive.repetitions,
            "inter_repetition_rest_s": drive.rest_s,
            "duration_s": float(duration),
            "scope": "Constant circuit R and quasistatic L; no eddy-current, skin-effect or spatial temperature prediction.",
        }
    else:
        reference = verified_efficiency_reference or {}
        matches = (
            tuple(path_hashes) == drive.gradient_efficiency_path_sha256
            and reference.get("body_ids") == list(drive.body_ids)
            and reference.get("path_sha256") == path_hashes
            and reference.get("efficiencies_T_per_m_per_A") == list(drive.gradient_efficiencies_T_per_m_per_A)
            and reference.get("source_sha256") == drive.gradient_efficiency_source_sha256
        )
        if not matches:
            output["drive"]["reason"] = (
                "Target drive requires matching pipeline-verified gradient efficiencies for the exact complete electrical paths."
            )
            return output
        currents = drive.target_gradient_T_per_m / np.asarray(drive.gradient_efficiencies_T_per_m_per_A)
        target_metrics = [
            DriveMetrics.from_circuit(
                current_A=float(current),
                resistance_ohm=float(r),
                inductance_H=float(self_inductance),
                target_gradient_T_per_m=drive.target_gradient_T_per_m,
                target_slew_rate_T_per_m_per_s=drive.target_slew_T_per_m_per_s,
            ).as_dict()
            for current, r, self_inductance in zip(currents, resistance, inductance)
        ]
        slopes = currents * drive.target_slew_T_per_m_per_s / drive.target_gradient_T_per_m
        peak_current = np.abs(currents)
        peak_voltage = np.abs(currents * resistance) + np.abs(matrix @ slopes)
        rms = power = None
        scope = "Specified gradient/slew from verified path efficiency; conservative coincident ramp voltage; duty cycle is unspecified."
        if drive.thermal is not None:
            output["thermal"] = {
                "available": False,
                "reason": "Transient thermal response requires an explicit duty waveform; target gradient/slew alone does not define heating.",
            }
    finite = [*peak_current, *peak_voltage, *((*rms, *power) if rms is not None else ())]
    if not np.all(np.isfinite(finite)):
        output["drive"]["reason"] = "Drive response exceeds finite SI reporting range."
        return output
    output["drive"] = {
        "available": True,
        "scope": scope,
        "source": drive.source,
        "mutual_inductance_included": mutual,
        "inductance_scope": matrix_note,
        "qualification_claim": False,
        "bodies": [
            {
                "body_id": body,
                "electrical_path_sha256": path_hashes[index],
                "peak_current_A": float(peak_current[index]),
                "current_limit_A": drive.current_limits_A[index],
                "current_margin_A": float(drive.current_limits_A[index] - peak_current[index]),
                "peak_voltage_V": float(peak_voltage[index]),
                "voltage_limit_V": drive.voltage_limits_V[index],
                "voltage_margin_V": float(drive.voltage_limits_V[index] - peak_voltage[index]),
                "within_current_limit": bool(peak_current[index] <= drive.current_limits_A[index]),
                "within_voltage_limit": bool(peak_voltage[index] <= drive.voltage_limits_V[index]),
                "rms_current_A": float(rms[index]) if rms is not None else None,
                "average_joule_loss_W": float(power[index]) if power is not None else None,
                "target_metrics": target_metrics[index] if target_metrics is not None else None,
            }
            for index, body in enumerate(drive.body_ids)
        ],
    }
    return output
