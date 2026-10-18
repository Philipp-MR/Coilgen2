from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


MetricFormatter = Callable[[float], str]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Description of one scalar coil-performance metric."""

    key: str
    label: str
    unit: str = "-"
    description: str = ""
    higher_is_better: bool | None = None
    domain: str = "coil"
    formatter: MetricFormatter | None = None

    def format_value(self, value: float) -> str:
        if self.formatter is not None:
            return self.formatter(float(value))
        return f"{float(value):.4g}"

    def label_with_unit(self) -> str:
        return self.label if self.unit in {"", "-"} else f"{self.label} [{self.unit}]"


class MetricRegistry:
    """Small registry for paper-study, plot-table, and report metrics."""

    def __init__(self, definitions: Iterable[MetricDefinition] = ()) -> None:
        self._definitions: dict[str, MetricDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: MetricDefinition) -> None:
        if not definition.key:
            raise ValueError("Metric keys must be non-empty.")
        self._definitions[definition.key] = definition

    def get(self, key: str) -> MetricDefinition:
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise KeyError(f"Unknown metric definition: {key}") from exc

    def labels(self) -> dict[str, str]:
        return {key: definition.label for key, definition in self._definitions.items()}

    def units(self) -> dict[str, str]:
        return {key: definition.unit for key, definition in self._definitions.items()}

    def as_mapping(self) -> Mapping[str, MetricDefinition]:
        return dict(self._definitions)


DEFAULT_METRIC_DEFINITIONS = (
    MetricDefinition(
        "layout_field_efficiency_mT_per_A",
        "field efficiency",
        "mT A^-1",
        "Absolute mean DSV Bz per ampere for a uniform nonzero target.",
        higher_is_better=True,
        domain="dsv",
    ),
    MetricDefinition(
        "layout_gradient_efficiency_mT_per_m_per_A",
        "gradient efficiency",
        "mT m^-1 A^-1",
        "Fitted target-mode efficiency of the connected layout per ampere.",
        higher_is_better=True,
        domain="dsv",
    ),
    MetricDefinition(
        "layout_mean_rel_error_percent",
        "mean internal Bz error",
        "%",
        "Mean relative error on DSV target points only.",
        higher_is_better=False,
        domain="dsv",
    ),
    MetricDefinition(
        "layout_max_rel_error_percent",
        "peak internal Bz error",
        "%",
        "Maximum relative error on DSV target points only.",
        higher_is_better=False,
        domain="dsv",
    ),
    MetricDefinition(
        "layout_suppressed_rms_Bz_T_at_fitted_current",
        "suppression RMS Bz",
        "T",
        "RMS Bz on suppression points at the fitted design current.",
        higher_is_better=False,
        domain="suppression",
    ),
    MetricDefinition(
        "layout_suppressed_peak_abs_Bz_T_at_fitted_current",
        "suppression peak |Bz|",
        "T",
        "Peak absolute Bz on suppression points at the fitted design current.",
        higher_is_better=False,
        domain="suppression",
    ),
    MetricDefinition(
        "coil_inductance_same_current_H",
        "same-current inductance",
        "H",
        "Inductance when all active coil parts are driven with the same current amplitude.",
        higher_is_better=False,
        domain="coil",
    ),
    MetricDefinition(
        "coil_resistance_total_ohm",
        "total resistance",
        "Ohm",
        "DC resistance from total wire length and conductor cross section.",
        higher_is_better=False,
        domain="coil",
    ),
    MetricDefinition(
        "coil_length_total_m",
        "wire length",
        "m",
        "Total connected centerline length.",
        higher_is_better=False,
        domain="layout",
    ),
    MetricDefinition(
        "unconnected_contours_min_wire_distance_m",
        "minimum unconnected-contour clearance",
        "m",
        "Minimum Euclidean centerline distance between distinct extracted contour loops before routing.",
        higher_is_better=True,
        domain="layout",
    ),
    MetricDefinition(
        "min_wire_wire_distance_m",
        "minimum wire clearance",
        "m",
        "Minimum centerline distance between distinct routed wire segments.",
        higher_is_better=True,
        domain="layout",
    ),
    MetricDefinition(
        "regularization_factor",
        "regularization factor",
        "-",
        "Selected Tikhonov regularization factor.",
        higher_is_better=None,
        domain="optimization",
    ),
)


DEFAULT_METRIC_REGISTRY = MetricRegistry(DEFAULT_METRIC_DEFINITIONS)
