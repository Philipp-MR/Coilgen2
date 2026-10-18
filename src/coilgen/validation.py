from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from .config import CoilGenConfig
from .design_context import EvaluationFeature, normalize_design_context
from .evaluation import calculate_layout_gradient, evaluate_field_errors
from .inductance import (
    MU0,
    calculate_inductance_by_coil_layout,
    equivalent_round_wire_radius,
)
from .layout import cleanup_wire_paths
from .models import CoilGenResult, ContourLine, WirePath
from .physics import biot_savart_wire_path


@dataclass(frozen=True, slots=True)
class SolenoidValidationResult:
    turn_count: int
    cylinder_height_m: float
    cylinder_radius_m: float
    effective_height_m: float
    effective_radius_m: float
    wire_radius_m: float
    needed_current_a: float
    requested_center_bz_t: float
    numerical_center_bz_per_a: float
    numerical_center_bz_t: float
    circular_loop_stack_center_bz_per_a: float
    circular_loop_stack_center_bz_t: float
    finite_solenoid_center_bz_per_a: float
    finite_solenoid_center_bz_t: float
    long_solenoid_center_bz_per_a: float
    center_vs_circular_loop_stack_error_percent: float
    center_vs_requested_error_percent: float
    numerical_inductance_h: float
    circular_loop_stack_inductance_h: float
    long_solenoid_inductance_h: float
    inductance_vs_circular_loop_stack_error_percent: float

    def as_dict(self) -> dict[str, float | int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class CircularLoopValidationResult:
    radius_m: float
    z_offset_m: float
    segment_count: int
    numerical_bz_per_a: float
    analytical_bz_per_a: float
    relative_error_percent: float

    def as_dict(self) -> dict[str, float | int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class BiplanarGradientScalingResult:
    radius_m: float
    gap_values_m: tuple[float, ...]
    gradient_values_T_per_m_per_A: tuple[float, ...]
    maximum_gap_m: float
    maximum_gradient_T_per_m_per_A: float
    expected_maximum_gap_m: float
    maximum_gap_error_percent: float

    def as_dict(self) -> dict[str, float | tuple[float, ...]]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class ActiveShieldBenchmarkResult:
    sample_count: int
    unshielded_rms: float
    shielded_rms: float
    unshielded_peak: float
    shielded_peak: float
    rms_reduction_percent: float
    peak_reduction_percent: float
    rms_reduced: bool
    peak_reduced: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def apply_ordered_solenoid_wire_path(
    result: CoilGenResult,
    config: CoilGenConfig,
    *,
    cut_angle_rad: float = 0.0,
) -> CoilGenResult:
    """Replace contour concatenation with a deterministic ordered solenoid path."""

    for part in result.coil_parts:
        ordered = ordered_solenoid_path_from_contours(part.contour_lines, cut_angle_rad=cut_angle_rad)
        if len(ordered) >= 2:
            part.wire_path = WirePath(v=ordered)
    result.coil_parts = cleanup_wire_paths(result.coil_parts, config=config)
    (
        result.coil_parts,
        result.inductance_matrix,
        result.resistance,
        result.coil_lengths,
        result.coil_cross_sections,
    ) = calculate_inductance_by_coil_layout(result.coil_parts, config)
    (
        result.field_by_layout,
        result.field_by_unconnected_loops,
        result.field_layout_per1Amp,
        result.field_loops_per1Amp,
        result.error_vals,
        result.needed_current_layout,
        result.b_field_opt_sf_1A,
        result.target_field_1A,
    ) = evaluate_field_errors(result.coil_parts, config, result.target_field, result.b_field_opt_sf)
    design_context = normalize_design_context(getattr(config, "design_context", "mr_gradient"))
    result.layout_gradient = (
        calculate_layout_gradient(result.field_layout_per1Amp, result.target_field)
        if design_context.enabled(EvaluationFeature.GRADIENT_FIT)
        else None
    )
    return result


def validate_solenoid_against_analytical_formulas(
    result: CoilGenResult,
    config: CoilGenConfig,
    *,
    center_point_m: Sequence[float] = (0.0, 0.0, 0.0),
    elliptic_quadrature_order: int = 96,
) -> SolenoidValidationResult:
    """Compare a generated solenoid-like result against analytical formulas."""

    cylinder_height_m, cylinder_radius_m = solenoid_height_and_radius(config.cylinder_mesh_parameter_list)
    center = np.asarray(center_point_m, dtype=float).reshape(1, 3)
    turn_geometry = turn_geometry_from_contours(result)
    turn_count = int(turn_geometry["turn_count"])
    effective_radius_m = (
        float(turn_geometry["mean_radius_m"])
        if math.isfinite(float(turn_geometry["mean_radius_m"]))
        else cylinder_radius_m
    )
    effective_height_m = (
        float(turn_geometry["z_span_m"]) if math.isfinite(float(turn_geometry["z_span_m"])) else cylinder_height_m
    )
    z_centers_m = np.asarray(turn_geometry["z_centers_m"], dtype=float)
    wire_radius_m = equivalent_round_wire_radius(config)
    needed_current_a = float(result.needed_current_layout)
    center_field_per_a = wire_center_bz_per_amp(result, center)
    center_field_t = center_field_per_a * needed_current_a
    requested_center_bz_t = float(
        result.target_field.b[
            np.argmin(np.linalg.norm(result.target_field.coords - center, axis=1)),
            2,
        ]
    )

    loop_stack_center_per_a = circular_loop_stack_center_bz_per_amp(
        effective_radius_m,
        z_centers_m,
        center_z_m=float(center[0, 2]),
    )
    loop_stack_center_t = loop_stack_center_per_a * needed_current_a
    loop_stack_inductance_h = circular_loop_stack_inductance(
        effective_radius_m,
        z_centers_m,
        wire_radius_m,
        quadrature_order=elliptic_quadrature_order,
    )
    finite_solenoid_center_per_a = finite_solenoid_center_bz_per_amp(
        turn_count,
        effective_radius_m,
        effective_height_m,
    )
    finite_solenoid_center_t = finite_solenoid_center_per_a * needed_current_a
    long_solenoid_center_per_a = MU0 * turn_count / effective_height_m
    long_solenoid_inductance_h = long_solenoid_inductance(turn_count, effective_radius_m, effective_height_m)
    numerical_inductance_h = float(result.inductance_matrix[0, 0]) if result.inductance_matrix is not None else math.nan

    return SolenoidValidationResult(
        turn_count=turn_count,
        cylinder_height_m=cylinder_height_m,
        cylinder_radius_m=cylinder_radius_m,
        effective_height_m=effective_height_m,
        effective_radius_m=effective_radius_m,
        wire_radius_m=wire_radius_m,
        needed_current_a=needed_current_a,
        requested_center_bz_t=requested_center_bz_t,
        numerical_center_bz_per_a=center_field_per_a,
        numerical_center_bz_t=center_field_t,
        circular_loop_stack_center_bz_per_a=loop_stack_center_per_a,
        circular_loop_stack_center_bz_t=loop_stack_center_t,
        finite_solenoid_center_bz_per_a=finite_solenoid_center_per_a,
        finite_solenoid_center_bz_t=finite_solenoid_center_t,
        long_solenoid_center_bz_per_a=long_solenoid_center_per_a,
        center_vs_circular_loop_stack_error_percent=relative_error_percent(center_field_t, loop_stack_center_t),
        center_vs_requested_error_percent=relative_error_percent(center_field_t, requested_center_bz_t),
        numerical_inductance_h=numerical_inductance_h,
        circular_loop_stack_inductance_h=loop_stack_inductance_h,
        long_solenoid_inductance_h=long_solenoid_inductance_h,
        inductance_vs_circular_loop_stack_error_percent=relative_error_percent(
            numerical_inductance_h,
            loop_stack_inductance_h,
        ),
    )


def ordered_solenoid_path_from_contours(contours: Sequence[ContourLine], *, cut_angle_rad: float = 0.0) -> np.ndarray:
    if not contours:
        return np.zeros((0, 3), dtype=float)
    ordered_contours = sorted(contours, key=lambda contour: float(np.mean(contour.v[:, 2])) if len(contour.v) else 0.0)
    path: list[np.ndarray] = []
    for contour in ordered_contours:
        loop = open_contour_at_angle(contour.v, cut_angle_rad=cut_angle_rad)
        if loop.size == 0:
            continue
        if path and np.linalg.norm(path[-1] - loop[0]) <= 1e-12:
            path.extend(loop[1:])
        else:
            path.extend(loop)
    return np.asarray(path, dtype=float) if path else np.zeros((0, 3), dtype=float)


def open_contour_at_angle(points: np.ndarray, *, cut_angle_rad: float = 0.0) -> np.ndarray:
    loop = np.asarray(points, dtype=float)
    if len(loop) < 2:
        return loop.copy()
    if np.linalg.norm(loop[0] - loop[-1]) <= 1e-10:
        loop = loop[:-1]
    if len(loop) < 2:
        return loop.copy()
    angles = np.arctan2(loop[:, 1], loop[:, 0])
    angle_distance = np.abs(np.angle(np.exp(1j * (angles - float(cut_angle_rad)))))
    start_ind = int(np.argmin(angle_distance))
    return np.concatenate([loop[start_ind:], loop[:start_ind]], axis=0)


def solenoid_height_and_radius(cylinder_mesh_parameter_list: Sequence[float]) -> tuple[float, float]:
    values = tuple(float(value) for value in cylinder_mesh_parameter_list)
    if len(values) < 2:
        raise ValueError("cylinder_mesh_parameter_list must contain at least height and radius.")
    return values[0], values[1]


def turn_geometry_from_contours(result: CoilGenResult) -> dict[str, np.ndarray | float | int]:
    z_centers: list[float] = []
    radii: list[float] = []
    for part in result.coil_parts:
        for contour in part.contour_lines:
            if len(contour.v):
                z_center = float(np.mean(contour.v[:, 2]))
                radius = float(np.mean(np.linalg.norm(contour.v[:, :2], axis=1)))
                z_centers.extend([z_center] * int(contour.turn_multiplicity))
                radii.extend([radius] * int(contour.turn_multiplicity))
    if not z_centers:
        return {"turn_count": 0, "z_centers_m": np.zeros(0), "z_span_m": math.nan, "mean_radius_m": math.nan}
    z = np.asarray(z_centers, dtype=float)
    return {
        "turn_count": int(len(z)),
        "z_centers_m": z,
        "z_span_m": float(np.max(z) - np.min(z)),
        "mean_radius_m": float(np.mean(radii)) if radii else math.nan,
    }


def wire_center_bz_per_amp(result: CoilGenResult, center: np.ndarray) -> float:
    fields = []
    for part in result.coil_parts:
        if part.wire_path is not None and len(part.wire_path.v) >= 2:
            fields.append(biot_savart_wire_path(part.wire_path.v, center)[0])
    if not fields:
        return math.nan
    return float(np.sum(fields, axis=0)[2])


def circular_loop_polyline(radius_m: float, *, segment_count: int = 720, z_m: float = 0.0) -> np.ndarray:
    """Return a closed circular-loop polyline in the xy plane."""

    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")
    if segment_count < 8:
        raise ValueError("segment_count must be at least 8.")
    angles = np.linspace(0.0, 2.0 * math.pi, int(segment_count), endpoint=False)
    points = np.column_stack(
        (
            float(radius_m) * np.cos(angles),
            float(radius_m) * np.sin(angles),
            np.full(len(angles), float(z_m)),
        )
    )
    return np.vstack((points, points[0]))


def circular_loop_bz_on_axis_per_amp(radius_m: float, axial_offset_m: float = 0.0) -> float:
    """Analytical axial Bz of one circular loop carrying 1 A."""

    if radius_m <= 0.0:
        return math.nan
    radius2 = float(radius_m) ** 2
    z2 = float(axial_offset_m) ** 2
    return MU0 * radius2 / (2.0 * (radius2 + z2) ** 1.5)


def validate_circular_loop_biot_savart(
    *,
    radius_m: float = 0.1,
    z_offset_m: float = 0.0,
    segment_count: int = 720,
) -> CircularLoopValidationResult:
    """Compare finite-segment Biot-Savart evaluation to the circular-loop formula."""

    loop = circular_loop_polyline(radius_m, segment_count=segment_count, z_m=0.0)
    target = np.asarray([[0.0, 0.0, float(z_offset_m)]], dtype=float)
    numerical = float(biot_savart_wire_path(loop, target)[0, 2])
    analytical = circular_loop_bz_on_axis_per_amp(radius_m, axial_offset_m=z_offset_m)
    return CircularLoopValidationResult(
        radius_m=float(radius_m),
        z_offset_m=float(z_offset_m),
        segment_count=int(segment_count),
        numerical_bz_per_a=numerical,
        analytical_bz_per_a=analytical,
        relative_error_percent=relative_error_percent(numerical, analytical),
    )


def finite_solenoid_center_bz_per_amp(turn_count: int, radius_m: float, height_m: float) -> float:
    """Finite-solenoid axial center field for N turns carrying 1 A."""

    if turn_count <= 0:
        return math.nan
    return MU0 * float(turn_count) / (2.0 * math.sqrt(radius_m * radius_m + (height_m / 2.0) ** 2))


def long_solenoid_inductance(turn_count: int, radius_m: float, height_m: float) -> float:
    """Long air-core solenoid approximation L = mu0 N^2 A / l."""

    if turn_count <= 0:
        return math.nan
    area_m2 = math.pi * radius_m * radius_m
    return MU0 * float(turn_count * turn_count) * area_m2 / height_m


def anti_helmholtz_biplanar_gradient_per_amp(radius_m: float, gap_m: float) -> float:
    """Analytical center dBz/dz for two opposite-current circular loops.

    This is the clean analytical benchmark for biplanar gradient scaling. The
    magnitude peaks at gap ~= radius for fixed loop radius.
    """

    if radius_m <= 0.0 or gap_m <= 0.0:
        return math.nan
    half_gap = 0.5 * float(gap_m)
    radius2 = float(radius_m) ** 2
    return 3.0 * MU0 * radius2 * half_gap / (radius2 + half_gap * half_gap) ** 2.5


def validate_biplanar_gradient_scaling(
    *,
    radius_m: float = 0.3,
    gap_values_m: Sequence[float] = (0.15, 0.22, 0.30, 0.38, 0.45),
) -> BiplanarGradientScalingResult:
    """Check anti-Helmholtz biplanar gradient scaling against its expected maximum."""

    gaps = tuple(float(value) for value in gap_values_m)
    if len(gaps) < 3:
        raise ValueError("At least three gap values are required for a scaling benchmark.")
    values = tuple(float(anti_helmholtz_biplanar_gradient_per_amp(radius_m, gap)) for gap in gaps)
    max_index = int(np.nanargmax(values))
    expected = float(radius_m)
    return BiplanarGradientScalingResult(
        radius_m=float(radius_m),
        gap_values_m=gaps,
        gradient_values_T_per_m_per_A=values,
        maximum_gap_m=gaps[max_index],
        maximum_gradient_T_per_m_per_A=values[max_index],
        expected_maximum_gap_m=expected,
        maximum_gap_error_percent=relative_error_percent(gaps[max_index], expected),
    )


def circular_loop_stack_center_bz_per_amp(
    radius_m: float, z_centers_m: np.ndarray, *, center_z_m: float = 0.0
) -> float:
    """Analytical center Bz of coaxial circular turns carrying 1 A."""

    if radius_m <= 0.0 or len(z_centers_m) == 0:
        return math.nan
    dz = np.asarray(z_centers_m, dtype=float) - float(center_z_m)
    numerator = MU0 * radius_m * radius_m
    denominator = 2.0 * np.power(radius_m * radius_m + dz * dz, 1.5)
    return float(np.sum(numerator / denominator))


def circular_loop_stack_inductance(
    radius_m: float,
    z_centers_m: np.ndarray,
    wire_radius_m: float,
    *,
    quadrature_order: int = 96,
) -> float:
    """Analytical thin-wire inductance of coaxial circular turns."""

    if radius_m <= 0.0 or len(z_centers_m) == 0:
        return math.nan
    wire_radius_m = max(float(wire_radius_m), 1e-12)
    z = np.asarray(z_centers_m, dtype=float)
    loop_self = MU0 * radius_m * (math.log(8.0 * radius_m / wire_radius_m) - 2.0)
    total = float(len(z)) * loop_self
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            total += 2.0 * coaxial_circular_loop_mutual_inductance(
                radius_m,
                abs(float(z[i] - z[j])),
                quadrature_order=quadrature_order,
            )
    return total


def coaxial_circular_loop_mutual_inductance(
    radius_m: float,
    separation_m: float,
    *,
    quadrature_order: int = 96,
) -> float:
    """Exact mutual inductance of two equal-radius coaxial circular loops."""

    if radius_m <= 0.0:
        return math.nan
    separation_m = abs(float(separation_m))
    denominator = 4.0 * radius_m * radius_m + separation_m * separation_m
    if denominator <= 0.0:
        return math.nan
    m = min(4.0 * radius_m * radius_m / denominator, 1.0 - 1e-15)
    k = math.sqrt(m)
    complete_k, complete_e = complete_elliptic_integrals(m, order=quadrature_order)
    return MU0 * radius_m * ((2.0 / k - k) * complete_k - (2.0 / k) * complete_e)


def complete_elliptic_integrals(m: float, *, order: int = 96) -> tuple[float, float]:
    """Complete elliptic integrals K(m), E(m) by Gauss-Legendre quadrature."""

    m = min(max(float(m), 0.0), 1.0 - 1e-15)
    nodes, weights = np.polynomial.legendre.leggauss(max(int(order), 16))
    theta = 0.25 * math.pi * (nodes + 1.0)
    theta_weights = 0.25 * math.pi * weights
    value = np.maximum(1.0 - m * np.sin(theta) ** 2, 1e-30)
    root = np.sqrt(value)
    complete_k = float(np.sum(theta_weights / root))
    complete_e = float(np.sum(theta_weights * root))
    return complete_k, complete_e


def validate_active_shield_suppression(
    shielded_field: np.ndarray,
    unshielded_field: np.ndarray,
    *,
    component: str = "Bz",
) -> ActiveShieldBenchmarkResult:
    """Compare shielded/unshielded fields on the same ordered suppression samples."""

    shielded_values = _component_values(np.asarray(shielded_field, dtype=float), component)
    unshielded_values = _component_values(np.asarray(unshielded_field, dtype=float), component)
    if len(shielded_values) != len(unshielded_values):
        raise ValueError("Shielded and unshielded fields must have the same sample count on the same ordered grid.")
    count = int(len(shielded_values))
    if count == 0:
        return ActiveShieldBenchmarkResult(
            sample_count=0,
            unshielded_rms=math.nan,
            shielded_rms=math.nan,
            unshielded_peak=math.nan,
            shielded_peak=math.nan,
            rms_reduction_percent=math.nan,
            peak_reduction_percent=math.nan,
            rms_reduced=False,
            peak_reduced=False,
        )
    unshielded_rms = float(np.sqrt(np.mean(unshielded_values**2)))
    shielded_rms = float(np.sqrt(np.mean(shielded_values**2)))
    unshielded_peak = float(np.max(np.abs(unshielded_values)))
    shielded_peak = float(np.max(np.abs(shielded_values)))
    return ActiveShieldBenchmarkResult(
        sample_count=count,
        unshielded_rms=unshielded_rms,
        shielded_rms=shielded_rms,
        unshielded_peak=unshielded_peak,
        shielded_peak=shielded_peak,
        rms_reduction_percent=_reduction_percent(unshielded_rms, shielded_rms),
        peak_reduction_percent=_reduction_percent(unshielded_peak, shielded_peak),
        rms_reduced=shielded_rms < unshielded_rms,
        peak_reduced=shielded_peak < unshielded_peak,
    )


def relative_error_percent(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference) or abs(reference) < 1e-30:
        return math.nan
    return 100.0 * abs(value - reference) / abs(reference)


def _component_values(field: np.ndarray, component: str) -> np.ndarray:
    if field.ndim == 1:
        return field.reshape(-1)
    if field.ndim != 2 or field.shape[1] < 3:
        return np.zeros(0, dtype=float)
    key = str(component).strip().lower()
    if key == "bx":
        return field[:, 0]
    if key == "by":
        return field[:, 1]
    if key == "bz":
        return field[:, 2]
    if key in {"magnitude", "mag", "|b|"}:
        return np.linalg.norm(field[:, :3], axis=1)
    raise ValueError("component must be 'Bx', 'By', 'Bz', or 'magnitude'.")


def _reduction_percent(reference: float, value: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(value) or abs(reference) < 1e-30:
        return math.nan
    return 100.0 * (1.0 - value / reference)
