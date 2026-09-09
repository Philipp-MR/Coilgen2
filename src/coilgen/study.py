from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from .models import CoilGenResult
from .reporting import result_quality_status, result_summary_dict


@dataclass(frozen=True, slots=True)
class StudyMetric:
    """One scalar study metric with explicit SI/scientific units."""

    name: str
    value: float
    unit: str = "-"
    description: str = ""

    def as_value(self) -> float:
        return float(self.value)


@dataclass(frozen=True, slots=True)
class ParameterAxis:
    """One varied parameter in a design study."""

    name: str
    values: tuple[Any, ...]
    unit: str = "-"
    label: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParameterAxis.name must be non-empty.")
        if len(self.values) == 0:
            raise ValueError(f"ParameterAxis {self.name!r} must contain at least one value.")

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Sequence[Any],
        *,
        unit: str = "-",
        label: str = "",
        description: str = "",
    ) -> "ParameterAxis":
        return cls(name=name, values=tuple(values), unit=unit, label=label, description=description)


@dataclass(frozen=True, slots=True)
class DesignStudyCase:
    """One parameter point in a design study."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParameterStudySpec:
    """Definition of a parameter sweep before running the designs."""

    name: str
    axes: tuple[ParameterAxis, ...]
    fixed_parameters: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParameterStudySpec.name must be non-empty.")
        if not self.axes:
            raise ValueError("ParameterStudySpec requires at least one ParameterAxis.")
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError("ParameterStudySpec axes must have unique names.")

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def point_count(self) -> int:
        count = 1
        for axis in self.axes:
            count *= len(axis.values)
        return count

    def cases(self) -> tuple[DesignStudyCase, ...]:
        cases: list[DesignStudyCase] = []
        axis_names = [axis.name for axis in self.axes]
        for index, values in enumerate(product(*[axis.values for axis in self.axes]), start=1):
            parameters = dict(self.fixed_parameters)
            parameters.update(dict(zip(axis_names, values)))
            suffix = "_".join(f"{name}={value}" for name, value in zip(axis_names, values))
            cases.append(DesignStudyCase(name=f"{self.name}_{index:03d}_{suffix}", parameters=parameters))
        return tuple(cases)


@dataclass(frozen=True, slots=True)
class DesignStudyResult:
    """Typed result row for parameter sweeps and paper-study plots."""

    case: DesignStudyCase
    metrics: Mapping[str, StudyMetric] = field(default_factory=dict)
    status: str = "UNKNOWN"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def flat_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"case": self.case.name, "status": self.status}
        row.update({f"parameter.{key}": value for key, value in self.case.parameters.items()})
        row.update({key: metric.value for key, metric in self.metrics.items()})
        row.update({f"{key}.unit": metric.unit for key, metric in self.metrics.items()})
        if self.notes:
            row["notes"] = "; ".join(self.notes)
        return row


@dataclass(frozen=True, slots=True)
class DesignStudyTable:
    """Collection of typed study results with simple selection helpers."""

    results: tuple[DesignStudyResult, ...]

    @classmethod
    def from_results(cls, results: Sequence[DesignStudyResult]) -> "DesignStudyTable":
        return cls(tuple(results))

    def rows(self) -> list[dict[str, Any]]:
        return [result.flat_row() for result in self.results]

    def best_by(
        self,
        metric_name: str,
        *,
        maximize: bool = True,
        allowed_statuses: Sequence[str] | None = ("PASS", "WARNING"),
    ) -> DesignStudyResult | None:
        """Rank evaluated candidates, excluding failed/unknown designs by default.

        Use ``allowed_statuses=("PASS",)`` for strict quality screening or
        ``None`` to inspect all finite results, including infeasible designs.
        A WARNING candidate still requires inspection of its notes.
        """

        statuses = None if allowed_statuses is None else {status.upper() for status in allowed_statuses}
        valid = [
            result
            for result in self.results
            if (statuses is None or result.status.upper() in statuses)
            and metric_name in result.metrics
            and np.isfinite(result.metrics[metric_name].value)
        ]
        if not valid:
            return None
        return (
            max(valid, key=lambda result: result.metrics[metric_name].value)
            if maximize
            else min(
                valid,
                key=lambda result: result.metrics[metric_name].value,
            )
        )


def study_result_from_coilgen_result(
    name: str,
    parameters: Mapping[str, Any],
    result: CoilGenResult,
) -> DesignStudyResult:
    """Build a typed study result from a finished CoilGen run."""

    summary = result_summary_dict(result)
    quality = result_quality_status(result)
    metric_specs = {
        "layout_gradient_efficiency_mT_per_m_per_A": ("mT m^-1 A^-1", "fitted DSV target-mode efficiency"),
        "layout_mean_rel_error_percent": ("%", "mean relative DSV target-mode error"),
        "layout_max_rel_error_percent": ("%", "peak relative DSV target-mode error"),
        "layout_suppressed_rms_Bz_T_at_fitted_current": ("T", "RMS Bz on suppression points at fitted current"),
        "layout_suppressed_peak_abs_Bz_T_at_fitted_current": (
            "T",
            "peak absolute Bz on suppression points at fitted current",
        ),
        "concomitant_rms_T": ("T", "RMS concomitant field estimate over the DSV"),
        "concomitant_max_abs_T": ("T", "peak absolute concomitant field estimate over the DSV"),
        "concomitant_rms_ppm_of_B0": ("ppm", "RMS concomitant field relative to configured B0"),
        "concomitant_max_abs_ppm_of_B0": ("ppm", "peak concomitant field relative to configured B0"),
        "coil_length_total_m": ("m", "total centerline length"),
        "coil_resistance_total_ohm": ("Ohm", "same-current total resistance"),
        "coil_inductance_same_current_H": ("H", "same-current inductance"),
        "min_contour_distance_m": ("m", "minimum extracted contour spacing"),
        "unconnected_contours_min_wire_distance_m": ("m", "minimum unconnected-contour centerline spacing"),
        "min_wire_wire_distance_m": ("m", "minimum routed wire spacing"),
        "regularization_factor": ("-", "selected Tikhonov regularization factor"),
        "target_error_constraint_relative_residual_percent": (
            "%",
            "achieved weighted target residual for fixed-error optimization",
        ),
    }
    metrics = {
        key: StudyMetric(key, float(summary.get(key, np.nan)), unit, description)
        for key, (unit, description) in metric_specs.items()
    }
    notes = tuple(quality.failures) + tuple(quality.warnings)
    return DesignStudyResult(
        case=DesignStudyCase(name=name, parameters=dict(parameters)),
        metrics=metrics,
        status=quality.status,
        notes=notes,
    )
