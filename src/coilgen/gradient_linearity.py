from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .engineering_metrics import fit_bz_field_quality
from .metrics import dsv_target_mask, suppression_target_mask


@dataclass(frozen=True, slots=True)
class GradientLinearitySpec:
    """User-facing MR-gradient linearity contract."""

    region: str = "dsv"
    reference_radius_m: float = 0.0
    rms_tolerance_percent: float = 0.0
    peak_tolerance_percent: float = 0.0
    cross_gradient_tolerance_percent: float = 0.0
    gradient_strength_tolerance_percent: float = 0.0
    normalization: str = "requested_gradient_times_radius"
    fit_offset: bool = True

    def to_config_overrides(self) -> dict[str, Any]:
        return {
            "linearity_region": self.region,
            "linearity_reference_radius_m": float(self.reference_radius_m),
            "linearity_rms_tolerance_percent": float(self.rms_tolerance_percent),
            "linearity_peak_tolerance_percent": float(self.peak_tolerance_percent),
            "linearity_cross_gradient_tolerance_percent": float(self.cross_gradient_tolerance_percent),
            "gradient_strength_tolerance_percent": float(self.gradient_strength_tolerance_percent),
            "linearity_normalization": self.normalization,
            "linearity_fit_offset": bool(self.fit_offset),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gradient_linearity_spec_from_value(value: Any) -> GradientLinearitySpec:
    if isinstance(value, GradientLinearitySpec):
        return value
    if isinstance(value, Mapping):
        return GradientLinearitySpec(**dict(value))
    raise TypeError("linearity must be a GradientLinearitySpec or mapping.")


def gradient_linearity_enabled(config: Any) -> bool:
    return any(
        _positive_config_float(config, name) > 0.0
        for name in (
            "linearity_rms_tolerance_percent",
            "linearity_peak_tolerance_percent",
            "linearity_cross_gradient_tolerance_percent",
            "gradient_strength_tolerance_percent",
        )
    )


def gradient_linearity_region_mask(target_field: Any, config: Any) -> np.ndarray:
    region = _normalise_linearity_region(getattr(config, "linearity_region", "dsv"))
    if region == "dsv":
        return dsv_target_mask(target_field)
    if region == "suppression":
        return suppression_target_mask(target_field)
    if region == "all":
        return np.ones(len(target_field.coords), dtype=bool)
    raise ValueError(f"Unsupported gradient linearity region: {region!r}")


def gradient_linearity_reference_radius_m(target_field: Any, config: Any, sample_mask: np.ndarray) -> float:
    configured = _positive_config_float(config, "linearity_reference_radius_m")
    if configured > 0.0:
        return configured
    coords = np.asarray(target_field.coords, dtype=float)
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
    finite = mask & np.all(np.isfinite(coords), axis=1)
    if not np.any(finite):
        return float("nan")
    selected = coords[finite]
    center = np.mean(selected, axis=0)
    return float(np.max(np.linalg.norm(selected - center, axis=1)))


def gradient_linearity_reference_scale_T(target_field: Any, config: Any, sample_mask: np.ndarray) -> float:
    normalization = _normalise_linearity_normalization(
        getattr(config, "linearity_normalization", "requested_gradient_times_radius")
    )
    target_bz = np.asarray(target_field.b, dtype=float)[:, 2]
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1) & np.isfinite(target_bz)
    if normalization == "target_peak":
        if not np.any(mask):
            return float("nan")
        return float(np.max(np.abs(target_bz[mask])))
    if normalization == "requested_gradient_times_radius":
        radius = gradient_linearity_reference_radius_m(target_field, config, sample_mask)
        requested = _requested_gradient_strength_T_per_m(config, target_field, sample_mask)
        return float(abs(requested) * radius)
    raise ValueError(f"Unsupported gradient linearity normalization: {normalization!r}")


def gradient_linearity_summary(field_T: Any, target_field: Any, config: Any) -> dict[str, Any]:
    """Evaluate the configured MR-gradient linearity contract on a field map."""

    enabled = gradient_linearity_enabled(config)
    sample_mask = gradient_linearity_region_mask(target_field, config)
    reference_radius = gradient_linearity_reference_radius_m(target_field, config, sample_mask)
    reference_scale = gradient_linearity_reference_scale_T(target_field, config, sample_mask)
    fit = fit_bz_field_quality(
        field_T,
        target_field,
        sample_mask=sample_mask,
        fit_offset=bool(getattr(config, "linearity_fit_offset", True)),
    )
    data = fit.as_dict()

    requested = _requested_gradient_strength_T_per_m(config, target_field, sample_mask)
    desired = float(data["desired_gradient_T_per_m"])
    achieved = float(data["achieved_gradient_T_per_m"])
    reference_gradient = abs(requested) if abs(requested) > 0.0 else abs(desired)
    gradient_strength_error_percent = (
        100.0 * abs(achieved - reference_gradient) / max(reference_gradient, 1.0e-30)
        if np.isfinite(achieved) and np.isfinite(reference_gradient) and reference_gradient > 0.0
        else float("nan")
    )
    rms_residual = float(data["linear_rms_residual_T"])
    peak_residual = float(data["linear_max_abs_residual_T"])
    p95_residual = float(data["linear_p95_abs_residual_T"])
    rms_percent = _percent_of_reference(rms_residual, reference_scale)
    peak_percent = _percent_of_reference(peak_residual, reference_scale)
    p95_percent = _percent_of_reference(p95_residual, reference_scale)

    gradient_tol = _positive_config_float(config, "gradient_strength_tolerance_percent")
    rms_tol = _positive_config_float(config, "linearity_rms_tolerance_percent")
    peak_tol = _positive_config_float(config, "linearity_peak_tolerance_percent")
    cross_tol = _positive_config_float(config, "linearity_cross_gradient_tolerance_percent")
    cross_percent = float(data["cross_gradient_percent_of_desired"])
    # A constant-field target acquires a tiny affine gradient from least-squares
    # roundoff. Relative gradient metrics would divide by that numerical noise.
    # Detect applicability in field units without changing the solve or its
    # reference-scale construction. Keep the absolute field-fit diagnostics.
    target_bz = np.asarray(target_field.b, dtype=float)[:, 2]
    finite_target = target_bz[sample_mask & np.isfinite(target_bz)]
    target_scale = float(np.max(np.abs(finite_target))) if len(finite_target) else float("nan")
    gradient_scale = abs(desired) * reference_radius
    applicable = bool(
        np.isfinite(gradient_scale)
        and gradient_scale > 128.0 * np.finfo(float).eps * max(target_scale, np.finfo(float).tiny)
    )
    if not applicable:
        gradient_strength_error_percent = cross_percent = rms_percent = peak_percent = p95_percent = float("nan")

    checks = {
        "gradient_strength_met": _tolerance_met(gradient_strength_error_percent, gradient_tol),
        "rms_met": _tolerance_met(rms_percent, rms_tol),
        "peak_met": _tolerance_met(peak_percent, peak_tol),
        "cross_gradient_met": _tolerance_met(cross_percent, cross_tol),
    }
    active_checks = [
        value for key, value in checks.items() if _check_enabled(key, gradient_tol, rms_tol, peak_tol, cross_tol)
    ]
    met = bool(active_checks and all(active_checks)) if enabled else True
    reasons = _linearity_failure_reasons(
        gradient_strength_error_percent,
        gradient_tol,
        rms_percent,
        rms_tol,
        peak_percent,
        peak_tol,
        cross_percent,
        cross_tol,
    )
    if enabled and not applicable:
        reasons = ["Target has no resolved affine Bz gradient; relative gradient requirements cannot be assessed."]

    def check_status(key: str) -> str:
        if not applicable:
            return "not_applicable"
        if not _check_enabled(key, gradient_tol, rms_tol, peak_tol, cross_tol):
            return "not_checked"
        return "PASS" if checks[key] else "FAIL"

    return {
        "linearity_enabled": bool(enabled),
        "linearity_applicable": applicable,
        "linearity_status": ("PASS" if met else "FAIL")
        if enabled
        else "not_checked"
        if applicable
        else "not_applicable",
        "linearity_met": bool(met),
        "linearity_failure_reasons": "; ".join(reasons),
        "linearity_region": _normalise_linearity_region(getattr(config, "linearity_region", "dsv")),
        "linearity_normalization": _normalise_linearity_normalization(
            getattr(config, "linearity_normalization", "requested_gradient_times_radius")
        ),
        "linearity_fit_offset": bool(getattr(config, "linearity_fit_offset", True)),
        "linearity_reference_radius_m": float(reference_radius),
        "linearity_reference_scale_T": float(reference_scale),
        "linearity_requested_gradient_T_per_m": float(requested),
        "linearity_desired_gradient_T_per_m": float(desired),
        "linearity_achieved_gradient_T_per_m": float(achieved),
        "linearity_gradient_strength_error_percent": float(gradient_strength_error_percent),
        "linearity_gradient_strength_tolerance_percent": float(gradient_tol),
        "linearity_gradient_strength_met": bool(checks["gradient_strength_met"]),
        "linearity_gradient_strength_status": check_status("gradient_strength_met"),
        "linearity_cross_gradient_T_per_m": float(data["cross_gradient_T_per_m"]),
        "linearity_cross_gradient_percent": float(cross_percent),
        "linearity_cross_gradient_tolerance_percent": float(cross_tol),
        "linearity_cross_gradient_met": bool(checks["cross_gradient_met"]),
        "linearity_cross_gradient_status": check_status("cross_gradient_met"),
        "linearity_rms_residual_T": float(rms_residual),
        "linearity_p95_abs_residual_T": float(p95_residual),
        "linearity_peak_abs_residual_T": float(peak_residual),
        "linearity_rms_percent": float(rms_percent),
        "linearity_p95_percent": float(p95_percent),
        "linearity_peak_percent": float(peak_percent),
        "linearity_rms_tolerance_percent": float(rms_tol),
        "linearity_peak_tolerance_percent": float(peak_tol),
        "linearity_rms_met": bool(checks["rms_met"]),
        "linearity_peak_met": bool(checks["peak_met"]),
        "linearity_rms_status": check_status("rms_met"),
        "linearity_peak_status": check_status("peak_met"),
        "linearity_sample_count": int(data["sample_count"]),
    }


def _normalise_linearity_region(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "dsv",
        "fov": "dsv",
        "roi": "dsv",
        "target": "dsv",
        "primary": "dsv",
        "internal": "dsv",
        "dsv": "dsv",
        "all": "all",
        "any": "all",
        "suppression": "suppression",
        "secondary": "suppression",
        "shield": "suppression",
    }
    return aliases.get(raw, raw)


def _normalise_linearity_normalization(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "requested_gradient_times_radius",
        "requested": "requested_gradient_times_radius",
        "gradient_radius": "requested_gradient_times_radius",
        "requested_gradient_radius": "requested_gradient_times_radius",
        "requested_gradient_times_radius": "requested_gradient_times_radius",
        "target_peak": "target_peak",
        "peak_target": "target_peak",
        "target": "target_peak",
    }
    return aliases.get(raw, raw)


def _requested_gradient_strength_T_per_m(config: Any, target_field: Any, sample_mask: np.ndarray) -> float:
    requested = _positive_config_float(config, "target_gradient_strength")
    if requested > 0.0:
        return requested
    target_bz = np.asarray(target_field.b, dtype=float)[:, 2]
    coords = np.asarray(target_field.coords, dtype=float)
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
    finite = mask & np.isfinite(target_bz) & np.all(np.isfinite(coords), axis=1)
    if np.count_nonzero(finite) < 4:
        return float("nan")
    matrix = np.column_stack([np.ones(np.count_nonzero(finite)), coords[finite]])
    coeff, *_ = np.linalg.lstsq(matrix, target_bz[finite], rcond=None)
    return float(np.linalg.norm(coeff[1:4]))


def _positive_config_float(config: Any, name: str) -> float:
    try:
        value = float(getattr(config, name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return float(value) if np.isfinite(value) and value > 0.0 else 0.0


def _percent_of_reference(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) <= 1.0e-30:
        return float("nan")
    return float(100.0 * abs(value) / abs(reference))


def _tolerance_met(value_percent: float, tolerance_percent: float) -> bool:
    if tolerance_percent <= 0.0:
        return True
    return bool(np.isfinite(value_percent) and value_percent <= tolerance_percent * (1.0 + 1.0e-10))


def _check_enabled(key: str, gradient_tol: float, rms_tol: float, peak_tol: float, cross_tol: float) -> bool:
    return {
        "gradient_strength_met": gradient_tol,
        "rms_met": rms_tol,
        "peak_met": peak_tol,
        "cross_gradient_met": cross_tol,
    }[key] > 0.0


def _linearity_failure_reasons(
    gradient_strength_error_percent: float,
    gradient_tol: float,
    rms_percent: float,
    rms_tol: float,
    peak_percent: float,
    peak_tol: float,
    cross_percent: float,
    cross_tol: float,
) -> list[str]:
    reasons: list[str] = []
    if not _tolerance_met(gradient_strength_error_percent, gradient_tol):
        reasons.append(f"gradient strength error {gradient_strength_error_percent:.4g}% exceeds {gradient_tol:.4g}%")
    if not _tolerance_met(rms_percent, rms_tol):
        reasons.append(f"linearity RMS {rms_percent:.4g}% exceeds {rms_tol:.4g}%")
    if not _tolerance_met(peak_percent, peak_tol):
        reasons.append(f"linearity peak {peak_percent:.4g}% exceeds {peak_tol:.4g}%")
    if not _tolerance_met(cross_percent, cross_tol):
        reasons.append(f"cross-gradient {cross_percent:.4g}% exceeds {cross_tol:.4g}%")
    return reasons
