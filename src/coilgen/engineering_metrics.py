"""Engineering-facing metrics for MR gradient-coil studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import dsv_target_mask


def _as_field_array(field_T: Any) -> np.ndarray:
    field = np.asarray(field_T, dtype=float)
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError("field_T must have shape (N, 3)")
    return field


def _finite_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def _finite_p95_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 95.0))


def _finite_max_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _safe_reduction_percent(active: float, unshielded: float) -> float:
    ratio = _safe_ratio(active, unshielded)
    if not np.isfinite(ratio):
        return float("nan")
    return float((1.0 - ratio) * 100.0)


def _unit_vector(value: Any, *, fallback: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        vector = np.asarray(fallback, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        vector = np.asarray(fallback, dtype=float)
        norm = float(np.linalg.norm(vector))
    return vector / norm


def _orthonormal_frame(z_axis: Any) -> np.ndarray:
    """Return rows containing x, y, z unit vectors for a B0-aligned frame."""

    z_hat = _unit_vector(z_axis)
    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(helper, z_hat))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    x_hat = helper - float(np.dot(helper, z_hat)) * z_hat
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    y_hat /= np.linalg.norm(y_hat)
    return np.vstack((x_hat, y_hat, z_hat))


def estimate_suppression_normals(
    coords_m: Any,
    *,
    center_m: Any | None = None,
    model: str = "box",
) -> np.ndarray:
    """Estimate outward normals for suppression samples.

    The default ``box`` model is deliberately simple and deterministic: each
    sample normal points along the coordinate axis where that sample is farthest
    from the sample cloud center. This works well for rectangular and split
    support surfaces, and it keeps the suppression
    metric interpretable as the local field component normal to the nearest
    suppression face.
    """

    coords = np.asarray(coords_m, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_m must have shape (N, 3)")
    if coords.size == 0:
        return np.zeros_like(coords)

    model_key = model.lower().strip()
    center = np.asarray(center_m, dtype=float) if center_m is not None else np.nanmean(coords, axis=0)
    centered = coords - center

    if model_key in {"radial", "cylindrical"}:
        normals = centered.copy()
        norms = np.linalg.norm(normals, axis=1)
        safe = norms > 0.0
        normals[safe] /= norms[safe, None]
        normals[~safe] = np.array([0.0, 0.0, 1.0])
        return normals

    half_span = np.nanmax(np.abs(centered), axis=0)
    half_span[~np.isfinite(half_span) | (half_span <= 0.0)] = 1.0
    scaled_distance = np.abs(centered) / half_span[None, :]
    normal_axis = np.nanargmax(scaled_distance, axis=1)

    normals = np.zeros_like(coords)
    signs = np.sign(centered[np.arange(coords.shape[0]), normal_axis])
    signs[signs == 0.0] = 1.0
    normals[np.arange(coords.shape[0]), normal_axis] = signs
    return normals


@dataclass(frozen=True, slots=True)
class SuppressionFieldMetrics:
    """Suppression-domain field metrics in MR-engineering units."""

    sample_count: int
    rms_Bz_T: float
    p95_abs_Bz_T: float
    max_abs_Bz_T: float
    rms_Bmag_T: float
    p95_Bmag_T: float
    max_Bmag_T: float
    rms_Bnormal_T: float
    p95_abs_Bnormal_T: float
    max_abs_Bnormal_T: float

    @classmethod
    def nan(cls) -> "SuppressionFieldMetrics":
        nan = float("nan")
        return cls(
            sample_count=0,
            rms_Bz_T=nan,
            p95_abs_Bz_T=nan,
            max_abs_Bz_T=nan,
            rms_Bmag_T=nan,
            p95_Bmag_T=nan,
            max_Bmag_T=nan,
            rms_Bnormal_T=nan,
            p95_abs_Bnormal_T=nan,
            max_abs_Bnormal_T=nan,
        )

    @classmethod
    def from_field(
        cls,
        field_T: Any,
        *,
        normals: Any | None = None,
    ) -> "SuppressionFieldMetrics":
        field = _as_field_array(field_T)
        if field.shape[0] == 0:
            return cls.nan()

        bz = field[:, 2]
        bmag = np.linalg.norm(field, axis=1)
        if normals is None:
            bnormal = np.full(field.shape[0], np.nan, dtype=float)
        else:
            normal_arr = np.asarray(normals, dtype=float)
            if normal_arr.shape != field.shape:
                raise ValueError("normals must have the same shape as field_T")
            normal_norm = np.linalg.norm(normal_arr, axis=1)
            safe = normal_norm > 0.0
            unit_normals = np.zeros_like(normal_arr)
            unit_normals[safe] = normal_arr[safe] / normal_norm[safe, None]
            bnormal = np.einsum("ij,ij->i", field, unit_normals)

        return cls(
            sample_count=int(field.shape[0]),
            rms_Bz_T=_finite_rms(bz),
            p95_abs_Bz_T=_finite_p95_abs(bz),
            max_abs_Bz_T=_finite_max_abs(bz),
            rms_Bmag_T=_finite_rms(bmag),
            p95_Bmag_T=_finite_p95_abs(bmag),
            max_Bmag_T=_finite_max_abs(bmag),
            rms_Bnormal_T=_finite_rms(bnormal),
            p95_abs_Bnormal_T=_finite_p95_abs(bnormal),
            max_abs_Bnormal_T=_finite_max_abs(bnormal),
        )

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        stem = f"{prefix}_" if prefix else ""
        return {
            f"{stem}suppression_sample_count": self.sample_count,
            f"{stem}rms_Bz_T": self.rms_Bz_T,
            f"{stem}p95_abs_Bz_T": self.p95_abs_Bz_T,
            f"{stem}max_abs_Bz_T": self.max_abs_Bz_T,
            f"{stem}rms_Bmag_T": self.rms_Bmag_T,
            f"{stem}p95_Bmag_T": self.p95_Bmag_T,
            f"{stem}max_Bmag_T": self.max_Bmag_T,
            f"{stem}rms_Bnormal_T": self.rms_Bnormal_T,
            f"{stem}p95_abs_Bnormal_T": self.p95_abs_Bnormal_T,
            f"{stem}max_abs_Bnormal_T": self.max_abs_Bnormal_T,
        }


@dataclass(frozen=True, slots=True)
class SuppressionComparison:
    """Compare active-shielded and unshielded fringe fields."""

    active: SuppressionFieldMetrics
    unshielded: SuppressionFieldMetrics
    normal_model: str

    @classmethod
    def from_fields(
        cls,
        active_field_T: Any,
        unshielded_field_T: Any,
        *,
        active_normals: Any | None = None,
        unshielded_normals: Any | None = None,
        normal_model: str = "box",
    ) -> "SuppressionComparison":
        return cls(
            active=SuppressionFieldMetrics.from_field(active_field_T, normals=active_normals),
            unshielded=SuppressionFieldMetrics.from_field(unshielded_field_T, normals=unshielded_normals),
            normal_model=normal_model,
        )

    def as_dict(self) -> dict[str, float | int | str]:
        data: dict[str, float | int | str] = {
            "suppression_normal_model": self.normal_model,
            **self.unshielded.as_dict("unshielded"),
            **self.active.as_dict("active"),
        }
        for component in ("Bz", "Bmag", "Bnormal"):
            active_rms = getattr(self.active, f"rms_{component}_T")
            unshielded_rms = getattr(self.unshielded, f"rms_{component}_T")
            data[f"shielding_rms_{component}_reduction_percent"] = _safe_reduction_percent(active_rms, unshielded_rms)
            data[f"shielding_rms_{component}_suppression_factor"] = _safe_ratio(unshielded_rms, active_rms)
        return data


@dataclass(frozen=True, slots=True)
class DriveMetrics:
    """Single-axis gradient-drive estimates at a target slew rate."""

    target_gradient_T_per_m: float
    target_slew_rate_T_per_m_per_s: float
    ramp_time_s: float
    current_A: float
    di_dt_A_per_s: float
    inductance_H: float
    resistance_ohm: float
    inductive_voltage_V: float
    resistive_voltage_V: float
    drive_voltage_V: float
    resistive_power_W: float
    stored_energy_J: float

    @classmethod
    def from_circuit(
        cls,
        *,
        current_A: float,
        resistance_ohm: float,
        inductance_H: float,
        target_gradient_T_per_m: float,
        target_slew_rate_T_per_m_per_s: float,
    ) -> "DriveMetrics":
        if target_slew_rate_T_per_m_per_s > 0.0 and target_gradient_T_per_m > 0.0:
            ramp_time_s = float(target_gradient_T_per_m / target_slew_rate_T_per_m_per_s)
        else:
            ramp_time_s = float("nan")
        di_dt = float(current_A / ramp_time_s) if np.isfinite(ramp_time_s) and ramp_time_s > 0.0 else float("nan")
        inductive_voltage = float(inductance_H * di_dt) if np.isfinite(di_dt) else float("nan")
        resistive_voltage = float(resistance_ohm * current_A)
        drive_voltage = (
            float(abs(inductive_voltage) + abs(resistive_voltage))
            if np.isfinite(inductive_voltage) and np.isfinite(resistive_voltage)
            else float("nan")
        )
        return cls(
            target_gradient_T_per_m=float(target_gradient_T_per_m),
            target_slew_rate_T_per_m_per_s=float(target_slew_rate_T_per_m_per_s),
            ramp_time_s=ramp_time_s,
            current_A=float(current_A),
            di_dt_A_per_s=di_dt,
            inductance_H=float(inductance_H),
            resistance_ohm=float(resistance_ohm),
            inductive_voltage_V=inductive_voltage,
            resistive_voltage_V=resistive_voltage,
            drive_voltage_V=drive_voltage,
            resistive_power_W=float(resistance_ohm * current_A * current_A),
            stored_energy_J=float(0.5 * inductance_H * current_A * current_A),
        )

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        stem = f"{prefix}_" if prefix else ""
        return {
            f"{stem}target_gradient_T_per_m": self.target_gradient_T_per_m,
            f"{stem}target_slew_rate_T_per_m_per_s": self.target_slew_rate_T_per_m_per_s,
            f"{stem}drive_ramp_time_s": self.ramp_time_s,
            f"{stem}drive_di_dt_A_per_s": self.di_dt_A_per_s,
            f"{stem}drive_inductive_voltage_V": self.inductive_voltage_V,
            f"{stem}drive_resistive_voltage_V": self.resistive_voltage_V,
            f"{stem}drive_voltage_target_slew_V": self.drive_voltage_V,
            f"{stem}resistive_power_target_gradient_W": self.resistive_power_W,
            f"{stem}stored_energy_target_gradient_J": self.stored_energy_J,
        }


@dataclass(frozen=True, slots=True)
class CurrentDensityReport:
    """Conductor current-density screen for a fitted operating current."""

    current_A: float
    conductor_width_m: float
    conductor_height_m: float
    conductor_area_m2: float
    current_density_A_per_m2: float
    current_density_A_per_mm2: float
    current_density_class: str
    assumption: str = "rectangular conductor cross section; no thermal duty-cycle model"

    @classmethod
    def nan(cls) -> "CurrentDensityReport":
        nan = float("nan")
        return cls(
            current_A=nan,
            conductor_width_m=nan,
            conductor_height_m=nan,
            conductor_area_m2=nan,
            current_density_A_per_m2=nan,
            current_density_A_per_mm2=nan,
            current_density_class="not_evaluated",
        )

    @classmethod
    def from_current_and_conductor(
        cls,
        *,
        current_A: float,
        conductor_width_m: float,
        conductor_height_m: float,
        conductor_area_m2: float | None = None,
    ) -> "CurrentDensityReport":
        width = float(conductor_width_m)
        height = float(conductor_height_m)
        current = float(current_A)
        area = width * height if conductor_area_m2 is None else float(conductor_area_m2)
        if not np.isfinite(current) or not np.isfinite(area) or area <= 0.0:
            return cls.nan()
        density_m2 = abs(current) / area
        density_mm2 = density_m2 / 1.0e6
        if density_mm2 <= 5.0:
            density_class = "continuous_candidate"
        elif density_mm2 <= 30.0:
            density_class = "pulsed_review_required"
        else:
            density_class = "severe_pulsed_only"
        return cls(
            current_A=current,
            conductor_width_m=width,
            conductor_height_m=height,
            conductor_area_m2=area,
            current_density_A_per_m2=float(density_m2),
            current_density_A_per_mm2=float(density_mm2),
            current_density_class=density_class,
            assumption=(
                "rectangular conductor cross section; no thermal duty-cycle model"
                if conductor_area_m2 is None
                else "configured conductor cross-section area; no thermal duty-cycle model"
            ),
        )

    @classmethod
    def from_result(cls, result: Any) -> "CurrentDensityReport":
        config = getattr(result, "input_data", None)
        if config is None:
            return cls.nan()
        area = None
        if np.asarray(getattr(config, "cross_sectional_points", [])).size >= 6:
            # Use the same polygon area as the resistance calculation, including
            # trapezoids and other profiles whose bounding rectangle is larger.
            from .inductance import conductor_cross_section_area

            area = conductor_cross_section_area(config)
        return cls.from_current_and_conductor(
            current_A=float(getattr(result, "needed_current_layout", np.nan)),
            conductor_width_m=float(getattr(config, "conductor_cross_section_width", np.nan)),
            conductor_height_m=float(getattr(config, "conductor_cross_section_height", np.nan)),
            conductor_area_m2=area,
        )

    def as_dict(self, prefix: str = "current_density") -> dict[str, float | str]:
        stem = f"{prefix}_" if prefix else ""
        return {
            f"{stem}current_A": self.current_A,
            f"{stem}conductor_width_m": self.conductor_width_m,
            f"{stem}conductor_height_m": self.conductor_height_m,
            f"{stem}conductor_area_m2": self.conductor_area_m2,
            f"{stem}A_per_m2": self.current_density_A_per_m2,
            f"{stem}A_per_mm2": self.current_density_A_per_mm2,
            f"{stem}class": self.current_density_class,
            f"{stem}assumption": self.assumption,
        }


@dataclass(frozen=True, slots=True)
class ThermalScreeningReport:
    """Placeholder for thermal screening status in design reports."""

    status: str = "not_evaluated"
    included: bool = False
    assumption: str = "no duty cycle, cooling, temperature-rise, or conductor insulation model evaluated"

    @classmethod
    def not_evaluated(cls) -> "ThermalScreeningReport":
        return cls()

    def as_dict(self, prefix: str = "thermal_screening") -> dict[str, bool | str]:
        stem = f"{prefix}_" if prefix else ""
        return {
            f"{stem}status": self.status,
            f"{stem}included": self.included,
            f"{stem}assumption": self.assumption,
        }


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Explicitly separates magnetostatic feasibility from engineering readiness."""

    engineering_readiness_status: str
    finished_engineering_design_ready: bool
    magnetostatic_status: str
    layout_status: str
    missing_checks: tuple[str, ...]

    @classmethod
    def from_status(
        cls,
        *,
        magnetostatic_status: str,
        layout_status: str,
        missing_checks: tuple[str, ...] | None = None,
    ) -> "ReadinessReport":
        missing = missing_checks or (
            "thermal",
            "gradient-driver voltage/current margin",
            "slew-rate validation",
            "PNS",
            "acoustic response",
            "Lorentz force and torque",
            "eddy-current coupling",
            "cryostat or nearby-conductor heating",
            "mechanical stress",
            "terminal routing validation",
        )
        ready = not missing and magnetostatic_status == "PASS" and layout_status == "PASS"
        status = "ready" if ready else "screening_only"
        return cls(
            engineering_readiness_status=status,
            finished_engineering_design_ready=ready,
            magnetostatic_status=magnetostatic_status,
            layout_status=layout_status,
            missing_checks=tuple(missing),
        )

    def as_dict(self, prefix: str = "") -> dict[str, bool | str]:
        stem = f"{prefix}_" if prefix else ""
        return {
            f"{stem}engineering_readiness_status": self.engineering_readiness_status,
            f"{stem}finished_engineering_design_ready": self.finished_engineering_design_ready,
            f"{stem}magnetostatic_status": self.magnetostatic_status,
            f"{stem}layout_status": self.layout_status,
            f"{stem}engineering_checks_not_included": "; ".join(self.missing_checks),
        }


def drive_metrics_from_inductance(
    *,
    current_A: float,
    resistance_ohm: float,
    inductance_H: float,
    target_gradient_T_per_m: float,
    target_slew_rate_T_per_m_per_s: float,
) -> DriveMetrics:
    return DriveMetrics.from_circuit(
        current_A=current_A,
        resistance_ohm=resistance_ohm,
        inductance_H=inductance_H,
        target_gradient_T_per_m=target_gradient_T_per_m,
        target_slew_rate_T_per_m_per_s=target_slew_rate_T_per_m_per_s,
    )


@dataclass(frozen=True, slots=True)
class ConcomitantFieldReport:
    """Maxwell concomitant-field estimate for a linear gradient in a main field.

    The model assumes the useful target component is parallel to the main field.
    Coordinates and the fitted gradient are rotated into a local frame whose
    z-axis is ``b0_direction``. For a local linear gradient ``G = (Gx, Gy, Gz)``,
    the Maxwell-consistent transverse components are
    ``Bx = Gx*z - 0.5*Gz*x`` and ``By = Gy*z - 0.5*Gz*y``. The reported scalar
    field error is the small-field magnitude term ``(Bx^2 + By^2) / (2*B0)``.
    """

    sample_count: int
    main_field_T: float
    b0_direction_x: float
    b0_direction_y: float
    b0_direction_z: float
    gradient_x_T_per_m: float
    gradient_y_T_per_m: float
    gradient_z_T_per_m: float
    rms_concomitant_T: float
    p95_abs_concomitant_T: float
    max_abs_concomitant_T: float
    rms_concomitant_ppm_of_B0: float
    p95_abs_concomitant_ppm_of_B0: float
    max_abs_concomitant_ppm_of_B0: float
    rms_concomitant_percent_of_target_peak: float
    max_abs_concomitant_percent_of_target_peak: float
    model: str = "linear_gradient_small_field_magnitude"
    assumption: str = "main field defines local z; fitted scalar gradient is B0-parallel"

    @classmethod
    def nan(cls) -> "ConcomitantFieldReport":
        nan = float("nan")
        return cls(
            sample_count=0,
            main_field_T=nan,
            b0_direction_x=nan,
            b0_direction_y=nan,
            b0_direction_z=nan,
            gradient_x_T_per_m=nan,
            gradient_y_T_per_m=nan,
            gradient_z_T_per_m=nan,
            rms_concomitant_T=nan,
            p95_abs_concomitant_T=nan,
            max_abs_concomitant_T=nan,
            rms_concomitant_ppm_of_B0=nan,
            p95_abs_concomitant_ppm_of_B0=nan,
            max_abs_concomitant_ppm_of_B0=nan,
            rms_concomitant_percent_of_target_peak=nan,
            max_abs_concomitant_percent_of_target_peak=nan,
        )

    @classmethod
    def from_linear_gradient(
        cls,
        coords_m: Any,
        gradient_T_per_m: Any,
        *,
        main_field_T: float,
        b0_direction: Any = (0.0, 0.0, 1.0),
        sample_mask: Any | None = None,
        target_peak_T: float | None = None,
    ) -> "ConcomitantFieldReport":
        coords = np.asarray(coords_m, dtype=float)
        gradient = np.asarray(gradient_T_per_m, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords_m must have shape (N, 3)")
        if gradient.shape != (3,):
            raise ValueError("gradient_T_per_m must contain exactly three values")
        if not np.isfinite(main_field_T) or float(main_field_T) <= 0.0:
            return cls.nan()

        mask = np.all(np.isfinite(coords), axis=1)
        if sample_mask is not None:
            sample_mask_array = np.asarray(sample_mask, dtype=bool).reshape(-1)
            if sample_mask_array.shape[0] != coords.shape[0]:
                raise ValueError("sample_mask length must match coords_m")
            mask &= sample_mask_array
        if not np.any(mask) or not np.all(np.isfinite(gradient)):
            return cls.nan()

        frame = _orthonormal_frame(b0_direction)
        local_coords = coords[mask] @ frame.T
        local_gradient = frame @ gradient
        concomitant = cls.field_values(
            local_coords,
            local_gradient,
            main_field_T=float(main_field_T),
            coordinates_are_b0_aligned=True,
        )
        b0_hat = frame[2]
        target_peak = float(target_peak_T) if target_peak_T is not None else float("nan")
        rms = _finite_rms(concomitant)
        p95 = _finite_p95_abs(concomitant)
        peak = _finite_max_abs(concomitant)
        return cls(
            sample_count=int(np.count_nonzero(mask)),
            main_field_T=float(main_field_T),
            b0_direction_x=float(b0_hat[0]),
            b0_direction_y=float(b0_hat[1]),
            b0_direction_z=float(b0_hat[2]),
            gradient_x_T_per_m=float(gradient[0]),
            gradient_y_T_per_m=float(gradient[1]),
            gradient_z_T_per_m=float(gradient[2]),
            rms_concomitant_T=rms,
            p95_abs_concomitant_T=p95,
            max_abs_concomitant_T=peak,
            rms_concomitant_ppm_of_B0=float(1.0e6 * rms / float(main_field_T)),
            p95_abs_concomitant_ppm_of_B0=float(1.0e6 * p95 / float(main_field_T)),
            max_abs_concomitant_ppm_of_B0=float(1.0e6 * peak / float(main_field_T)),
            rms_concomitant_percent_of_target_peak=(
                float(100.0 * rms / target_peak) if np.isfinite(target_peak) and target_peak > 0.0 else float("nan")
            ),
            max_abs_concomitant_percent_of_target_peak=(
                float(100.0 * peak / target_peak) if np.isfinite(target_peak) and target_peak > 0.0 else float("nan")
            ),
        )

    @staticmethod
    def field_values(
        coords_m: Any,
        gradient_T_per_m: Any,
        *,
        main_field_T: float,
        b0_direction: Any = (0.0, 0.0, 1.0),
        coordinates_are_b0_aligned: bool = False,
    ) -> np.ndarray:
        coords = np.asarray(coords_m, dtype=float)
        gradient = np.asarray(gradient_T_per_m, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords_m must have shape (N, 3)")
        if gradient.shape != (3,):
            raise ValueError("gradient_T_per_m must contain exactly three values")
        if not np.isfinite(main_field_T) or float(main_field_T) <= 0.0:
            return np.full(coords.shape[0], np.nan, dtype=float)
        if coordinates_are_b0_aligned:
            local_coords = coords
            local_gradient = gradient
        else:
            frame = _orthonormal_frame(b0_direction)
            local_coords = coords @ frame.T
            local_gradient = frame @ gradient
        x = local_coords[:, 0]
        y = local_coords[:, 1]
        z = local_coords[:, 2]
        gx, gy, gz = local_gradient
        bx = gx * z - 0.5 * gz * x
        by = gy * z - 0.5 * gz * y
        return (bx * bx + by * by) / (2.0 * float(main_field_T))

    def as_dict(self, prefix: str = "concomitant") -> dict[str, float | int | str]:
        stem = f"{prefix}_" if prefix else ""
        return {
            f"{stem}sample_count": self.sample_count,
            f"{stem}main_field_T": self.main_field_T,
            f"{stem}b0_direction_x": self.b0_direction_x,
            f"{stem}b0_direction_y": self.b0_direction_y,
            f"{stem}b0_direction_z": self.b0_direction_z,
            f"{stem}gradient_x_T_per_m": self.gradient_x_T_per_m,
            f"{stem}gradient_y_T_per_m": self.gradient_y_T_per_m,
            f"{stem}gradient_z_T_per_m": self.gradient_z_T_per_m,
            f"{stem}rms_T": self.rms_concomitant_T,
            f"{stem}p95_abs_T": self.p95_abs_concomitant_T,
            f"{stem}max_abs_T": self.max_abs_concomitant_T,
            f"{stem}rms_ppm_of_B0": self.rms_concomitant_ppm_of_B0,
            f"{stem}p95_abs_ppm_of_B0": self.p95_abs_concomitant_ppm_of_B0,
            f"{stem}max_abs_ppm_of_B0": self.max_abs_concomitant_ppm_of_B0,
            f"{stem}rms_percent_of_target_peak": self.rms_concomitant_percent_of_target_peak,
            f"{stem}max_abs_percent_of_target_peak": self.max_abs_concomitant_percent_of_target_peak,
            f"{stem}model": self.model,
            f"{stem}assumption": self.assumption,
        }


@dataclass(frozen=True, slots=True)
class FieldQualityFit:
    """Linear and second-order Bz fit over the DSV."""

    sample_count: int
    offset_T: float
    dbz_dx_T_per_m: float
    dbz_dy_T_per_m: float
    dbz_dz_T_per_m: float
    desired_dbz_dx_T_per_m: float
    desired_dbz_dy_T_per_m: float
    desired_dbz_dz_T_per_m: float
    desired_gradient_axis: str
    desired_gradient_T_per_m: float
    achieved_gradient_T_per_m: float
    cross_gradient_T_per_m: float
    cross_gradient_percent_of_desired: float
    gradient_alignment_cosine: float
    peak_target_T: float
    linear_rms_residual_T: float
    linear_p95_abs_residual_T: float
    linear_max_abs_residual_T: float
    linear_rms_residual_percent_of_peak: float
    quadratic_rms_residual_T: float
    quadratic_rms_residual_percent_of_peak: float
    qxx_T_per_m2: float
    qyy_T_per_m2: float
    qzz_T_per_m2: float
    qxy_T_per_m2: float
    qxz_T_per_m2: float
    qyz_T_per_m2: float
    second_order_rms_T_per_m2: float

    @classmethod
    def nan(cls) -> "FieldQualityFit":
        nan = float("nan")
        return cls(
            sample_count=0,
            offset_T=nan,
            dbz_dx_T_per_m=nan,
            dbz_dy_T_per_m=nan,
            dbz_dz_T_per_m=nan,
            desired_dbz_dx_T_per_m=nan,
            desired_dbz_dy_T_per_m=nan,
            desired_dbz_dz_T_per_m=nan,
            desired_gradient_axis="unknown",
            desired_gradient_T_per_m=nan,
            achieved_gradient_T_per_m=nan,
            cross_gradient_T_per_m=nan,
            cross_gradient_percent_of_desired=nan,
            gradient_alignment_cosine=nan,
            peak_target_T=nan,
            linear_rms_residual_T=nan,
            linear_p95_abs_residual_T=nan,
            linear_max_abs_residual_T=nan,
            linear_rms_residual_percent_of_peak=nan,
            quadratic_rms_residual_T=nan,
            quadratic_rms_residual_percent_of_peak=nan,
            qxx_T_per_m2=nan,
            qyy_T_per_m2=nan,
            qzz_T_per_m2=nan,
            qxy_T_per_m2=nan,
            qxz_T_per_m2=nan,
            qyz_T_per_m2=nan,
            second_order_rms_T_per_m2=nan,
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "sample_count": self.sample_count,
            "offset_T": self.offset_T,
            "dbz_dx_T_per_m": self.dbz_dx_T_per_m,
            "dbz_dy_T_per_m": self.dbz_dy_T_per_m,
            "dbz_dz_T_per_m": self.dbz_dz_T_per_m,
            "desired_dbz_dx_T_per_m": self.desired_dbz_dx_T_per_m,
            "desired_dbz_dy_T_per_m": self.desired_dbz_dy_T_per_m,
            "desired_dbz_dz_T_per_m": self.desired_dbz_dz_T_per_m,
            "desired_gradient_axis": self.desired_gradient_axis,
            "desired_gradient_T_per_m": self.desired_gradient_T_per_m,
            "achieved_gradient_T_per_m": self.achieved_gradient_T_per_m,
            "cross_gradient_T_per_m": self.cross_gradient_T_per_m,
            "cross_gradient_percent_of_desired": self.cross_gradient_percent_of_desired,
            "gradient_alignment_cosine": self.gradient_alignment_cosine,
            "peak_target_T": self.peak_target_T,
            "linear_rms_residual_T": self.linear_rms_residual_T,
            "linear_p95_abs_residual_T": self.linear_p95_abs_residual_T,
            "linear_max_abs_residual_T": self.linear_max_abs_residual_T,
            "linear_rms_residual_percent_of_peak": self.linear_rms_residual_percent_of_peak,
            "quadratic_rms_residual_T": self.quadratic_rms_residual_T,
            "quadratic_rms_residual_percent_of_peak": self.quadratic_rms_residual_percent_of_peak,
            "qxx_T_per_m2": self.qxx_T_per_m2,
            "qyy_T_per_m2": self.qyy_T_per_m2,
            "qzz_T_per_m2": self.qzz_T_per_m2,
            "qxy_T_per_m2": self.qxy_T_per_m2,
            "qxz_T_per_m2": self.qxz_T_per_m2,
            "qyz_T_per_m2": self.qyz_T_per_m2,
            "second_order_rms_T_per_m2": self.second_order_rms_T_per_m2,
        }


def _linear_fit(coords: np.ndarray, values: np.ndarray, *, fit_offset: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if fit_offset:
        matrix = np.column_stack([np.ones(coords.shape[0]), coords])
        coeff, *_ = np.linalg.lstsq(matrix, values, rcond=None)
        return coeff, matrix @ coeff
    matrix = np.asarray(coords, dtype=float)
    gradient, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    coeff = np.concatenate(([0.0], np.asarray(gradient, dtype=float).reshape(3)))
    return coeff, matrix @ gradient


def _quadratic_fit(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    matrix = np.column_stack(
        [
            np.ones(coords.shape[0]),
            x,
            y,
            z,
            x * x,
            y * y,
            z * z,
            x * y,
            x * z,
            y * z,
        ]
    )
    coeff, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    return coeff, matrix @ coeff


def _axis_label(vector: np.ndarray) -> str:
    labels = ("x", "y", "z")
    if vector.size != 3 or not np.any(np.isfinite(vector)):
        return "unknown"
    return labels[int(np.nanargmax(np.abs(vector)))]


def fit_bz_field_quality(
    field_T: Any,
    target_field: Any,
    *,
    sample_mask: Any | None = None,
    fit_offset: bool = True,
) -> FieldQualityFit:
    """Fit Bz over the DSV and report desired/cross-gradient terms."""

    field = _as_field_array(field_T)
    target_b = _as_field_array(target_field.b)
    coords = np.asarray(target_field.coords, dtype=float)
    if coords.shape != field.shape or target_b.shape != field.shape:
        raise ValueError("field_T, target_field.b and target_field.coords must share shape (N, 3)")

    mask = np.asarray(sample_mask, dtype=bool) if sample_mask is not None else dsv_target_mask(target_field)
    if mask.shape[0] != field.shape[0]:
        raise ValueError("sample_mask length must match field_T")
    mask &= np.all(np.isfinite(coords), axis=1) & np.isfinite(field[:, 2]) & np.isfinite(target_b[:, 2])
    if int(np.count_nonzero(mask)) < 4:
        return FieldQualityFit.nan()

    sample_coords = coords[mask]
    bz = field[mask, 2]
    target_bz = target_b[mask, 2]

    linear_coeff, linear_prediction = _linear_fit(sample_coords, bz, fit_offset=fit_offset)
    target_coeff, _ = _linear_fit(sample_coords, target_bz, fit_offset=fit_offset)
    if sample_coords.shape[0] >= 10:
        quadratic_coeff, quadratic_prediction = _quadratic_fit(sample_coords, bz)
    else:
        quadratic_coeff = np.full(10, np.nan, dtype=float)
        quadratic_prediction = np.full_like(bz, np.nan, dtype=float)

    gradient = np.asarray(linear_coeff[1:4], dtype=float)
    desired_gradient = np.asarray(target_coeff[1:4], dtype=float)
    desired_norm = float(np.linalg.norm(desired_gradient))
    gradient_norm = float(np.linalg.norm(gradient))
    if desired_norm > 0.0:
        desired_unit = desired_gradient / desired_norm
        achieved = float(np.dot(gradient, desired_unit))
        cross = float(np.linalg.norm(gradient - achieved * desired_unit))
        alignment = float(np.dot(gradient, desired_gradient) / (max(gradient_norm, 1e-30) * desired_norm))
        cross_percent = float(100.0 * cross / desired_norm)
    else:
        achieved = gradient_norm
        cross = float("nan")
        alignment = float("nan")
        cross_percent = float("nan")

    linear_residual = bz - linear_prediction
    quadratic_residual = bz - quadratic_prediction
    peak_target = _finite_max_abs(target_bz)
    linear_percent = float(100.0 * _finite_rms(linear_residual) / peak_target) if peak_target > 0.0 else float("nan")
    quadratic_percent = (
        float(100.0 * _finite_rms(quadratic_residual) / peak_target) if peak_target > 0.0 else float("nan")
    )
    second_order = np.asarray(quadratic_coeff[4:10], dtype=float)

    return FieldQualityFit(
        sample_count=int(np.count_nonzero(mask)),
        offset_T=float(linear_coeff[0]),
        dbz_dx_T_per_m=float(linear_coeff[1]),
        dbz_dy_T_per_m=float(linear_coeff[2]),
        dbz_dz_T_per_m=float(linear_coeff[3]),
        desired_dbz_dx_T_per_m=float(target_coeff[1]),
        desired_dbz_dy_T_per_m=float(target_coeff[2]),
        desired_dbz_dz_T_per_m=float(target_coeff[3]),
        desired_gradient_axis=_axis_label(desired_gradient),
        desired_gradient_T_per_m=desired_norm,
        achieved_gradient_T_per_m=achieved,
        cross_gradient_T_per_m=cross,
        cross_gradient_percent_of_desired=cross_percent,
        gradient_alignment_cosine=alignment,
        peak_target_T=peak_target,
        linear_rms_residual_T=_finite_rms(linear_residual),
        linear_p95_abs_residual_T=_finite_p95_abs(linear_residual),
        linear_max_abs_residual_T=_finite_max_abs(linear_residual),
        linear_rms_residual_percent_of_peak=linear_percent,
        quadratic_rms_residual_T=_finite_rms(quadratic_residual),
        quadratic_rms_residual_percent_of_peak=quadratic_percent,
        qxx_T_per_m2=float(quadratic_coeff[4]),
        qyy_T_per_m2=float(quadratic_coeff[5]),
        qzz_T_per_m2=float(quadratic_coeff[6]),
        qxy_T_per_m2=float(quadratic_coeff[7]),
        qxz_T_per_m2=float(quadratic_coeff[8]),
        qyz_T_per_m2=float(quadratic_coeff[9]),
        second_order_rms_T_per_m2=_finite_rms(second_order),
    )
