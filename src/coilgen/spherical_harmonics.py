from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


def _fit_activity(progress, cancel_check, percent, message):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Spherical harmonic analysis cancelled.")
    if progress is not None:
        progress(percent, message)


@dataclass(frozen=True, slots=True)
class SphericalHarmonicTerm:
    """One real solid spherical-harmonic basis term."""

    degree: int
    order: int
    label: str
    alias: str = ""

    @property
    def display_label(self) -> str:
        return f"{self.label} ({self.alias})" if self.alias else self.label


@dataclass(frozen=True, slots=True)
class SphericalHarmonicFit:
    """Least-squares decomposition of scalar field samples into SH terms."""

    terms: tuple[SphericalHarmonicTerm, ...]
    coefficients: np.ndarray
    fitted_values: np.ndarray
    residual_values: np.ndarray
    center: np.ndarray
    radius: float
    sample_count: int
    unit: str = ""
    basis_rms: np.ndarray | None = None
    field_rms: float = float("nan")
    residual_rms: float = float("nan")
    explained_rms_fraction: float = float("nan")
    rank: int = 0
    singular_values: np.ndarray | None = None


def solid_spherical_harmonic_terms(max_order: int) -> tuple[SphericalHarmonicTerm, ...]:
    """Return real spherical-harmonic term metadata up to ``max_order``."""

    degree_limit = _validate_max_order(max_order)
    terms: list[SphericalHarmonicTerm] = []
    for degree in range(degree_limit + 1):
        for order in range(-degree, degree + 1):
            terms.append(
                SphericalHarmonicTerm(
                    degree=degree,
                    order=order,
                    label=f"l={degree}, m={order:+d}",
                    alias=_term_alias(degree, order),
                )
            )
    return tuple(terms)


def solid_spherical_harmonic_basis(
    coords: np.ndarray,
    max_order: int = 4,
    *,
    center: np.ndarray | None = None,
    radius: float | None = None,
    weights: np.ndarray | None = None,
    progress=None,
    cancel_check=None,
) -> tuple[np.ndarray, tuple[SphericalHarmonicTerm, ...], np.ndarray, float]:
    """Build a normalized real solid spherical-harmonic basis for sample points.

    Coordinates are shifted by ``center`` and scaled by ``radius`` before fitting:
    each term is ``(r / radius)**l * real_Y_lm(theta, phi)``. This keeps the
    coefficient units equal to the scalar field units and makes coefficients
    comparable between before/after fits on the same target cloud.
    """

    _fit_activity(progress, cancel_check, 0, "Prepare harmonic basis coordinates")
    coords = _as_coordinate_array(coords)
    finite = np.all(np.isfinite(coords), axis=1)
    if not np.any(finite):
        raise ValueError("coords must contain at least one finite 3-D sample.")
    fit_weights = _validated_weights(weights, len(coords))
    center_mask = finite.copy()
    if fit_weights is not None:
        weighted_mask = np.isfinite(fit_weights) & (fit_weights > 0.0)
        if np.any(center_mask & weighted_mask):
            center_mask &= weighted_mask
    if center is None:
        center_arr = _weighted_center(coords[center_mask], None if fit_weights is None else fit_weights[center_mask])
    else:
        center_arr = np.asarray(center, dtype=float).reshape(3)
        if not np.all(np.isfinite(center_arr)):
            raise ValueError("center must contain finite values.")
    shifted = coords - center_arr[None, :]
    radii = np.linalg.norm(shifted, axis=1)
    if radius is None:
        finite_radii = radii[finite & np.isfinite(radii)]
        radius_value = float(np.max(finite_radii)) if len(finite_radii) else 1.0
    else:
        radius_value = float(radius)
    if not np.isfinite(radius_value) or radius_value <= 0.0:
        radius_value = 1.0

    rho = radii / radius_value
    cos_theta = np.ones(len(coords), dtype=float)
    nonzero = radii > 0.0
    cos_theta[nonzero] = shifted[nonzero, 2] / radii[nonzero]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    phi = np.arctan2(shifted[:, 1], shifted[:, 0])

    terms = solid_spherical_harmonic_terms(max_order)
    basis = np.empty((len(coords), len(terms)), dtype=float)
    for column, term in enumerate(terms):
        _fit_activity(
            progress, cancel_check, 100 * column / len(terms), f"Construct harmonic basis {column}/{len(terms)}"
        )
        degree = int(term.degree)
        order = int(term.order)
        basis[:, column] = np.power(rho, degree) * _real_spherical_harmonic(degree, order, cos_theta, phi)
    _fit_activity(progress, cancel_check, 100, f"Harmonic basis {len(terms)}/{len(terms)} complete")
    return basis, terms, center_arr, radius_value


def fit_spherical_harmonics(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    max_order: int = 4,
    weights: np.ndarray | None = None,
    center: np.ndarray | None = None,
    radius: float | None = None,
    rcond: float | None = None,
    unit: str = "",
) -> SphericalHarmonicFit:
    """Fit scalar samples with real solid spherical harmonics."""

    return fit_spherical_harmonics_many(
        coords,
        {"field": values},
        max_order=max_order,
        weights=weights,
        center=center,
        radius=radius,
        rcond=rcond,
        unit=unit,
    )["field"]


def fit_spherical_harmonics_many(
    coords: np.ndarray,
    values_by_source: Mapping[str, np.ndarray],
    *,
    max_order: int = 4,
    weights: np.ndarray | None = None,
    center: np.ndarray | None = None,
    radius: float | None = None,
    rcond: float | None = None,
    unit: str = "",
    progress=None,
    cancel_check=None,
) -> dict[str, SphericalHarmonicFit]:
    """Fit fields on one common support with a single basis/factorization."""

    _fit_activity(progress, cancel_check, None, "Validate common harmonic fit samples")
    if not values_by_source:
        return {}
    coords = _as_coordinate_array(coords)
    source_keys = tuple(values_by_source)
    columns = [np.asarray(values_by_source[key], dtype=float).reshape(-1) for key in source_keys]
    if any(len(column) != len(coords) for column in columns):
        raise ValueError("values must match coordinate count.")
    values = np.column_stack(columns)
    fit_weights = _validated_weights(weights, len(coords))
    finite = np.all(np.isfinite(values), axis=1) & np.all(np.isfinite(coords), axis=1)
    if fit_weights is not None:
        finite &= np.isfinite(fit_weights) & (fit_weights > 0.0)
    if not np.any(finite):
        raise ValueError("fit_spherical_harmonics has no finite samples.")

    fit_coords = coords[finite]
    fit_values = values[finite]
    fit_weights = None if fit_weights is None else fit_weights[finite]
    basis, terms, center_arr, radius_value = solid_spherical_harmonic_basis(
        fit_coords,
        max_order,
        center=center,
        radius=radius,
        weights=fit_weights,
        progress=lambda percent, message: _fit_activity(progress, cancel_check, None, message),
        cancel_check=cancel_check,
    )
    if fit_weights is None:
        weighted_basis = basis
        weighted_values = fit_values
    else:
        root_weights = np.sqrt(fit_weights).reshape(-1, 1)
        weighted_basis = basis * root_weights
        weighted_values = fit_values * root_weights

    _fit_activity(
        progress, cancel_check, None, "Solve shared harmonic least squares (factorization progress unavailable)"
    )
    coefficients, _, rank, singular_values = np.linalg.lstsq(weighted_basis, weighted_values, rcond=rcond)
    _fit_activity(progress, cancel_check, None, "Reconstruct harmonic fields and residuals")
    fitted = basis @ coefficients
    residual = fit_values - fitted
    basis_rms = _weighted_rms_columns(basis, fit_weights)
    fits = {}
    for index, source in enumerate(source_keys):
        _fit_activity(
            progress, cancel_check, None, f"Summarize harmonic field {index + 1}/{len(source_keys)} · {source}"
        )
        field_rms = _weighted_rms(fit_values[:, index], fit_weights)
        residual_rms = _weighted_rms(residual[:, index], fit_weights)
        fits[source] = SphericalHarmonicFit(
            terms=terms,
            coefficients=np.asarray(coefficients[:, index], dtype=float),
            fitted_values=np.asarray(fitted[:, index], dtype=float),
            residual_values=np.asarray(residual[:, index], dtype=float),
            center=np.asarray(center_arr, dtype=float),
            radius=float(radius_value),
            sample_count=int(len(fit_values)),
            unit=str(unit),
            basis_rms=np.asarray(basis_rms, dtype=float),
            field_rms=float(field_rms),
            residual_rms=float(residual_rms),
            explained_rms_fraction=float(1.0 - residual_rms / max(field_rms, 1.0e-30)),
            rank=int(rank),
            singular_values=np.asarray(singular_values, dtype=float).copy(),
        )
    _fit_activity(progress, cancel_check, 100, "Common harmonic fit complete")
    return fits


def spherical_harmonic_coefficient_rows(
    fit: SphericalHarmonicFit,
    *,
    min_contribution_fraction: float = 0.01,
    max_terms: int | None = None,
) -> list[dict[str, Any]]:
    """Return significant SH coefficient rows for reports or plots."""

    coefficients = np.asarray(fit.coefficients, dtype=float).reshape(-1)
    basis_rms = _fit_basis_rms(fit)
    rows = [
        _coefficient_row(fit, term, index, coefficients[index], basis_rms[index])
        for index, term in enumerate(fit.terms)
    ]
    rows.sort(key=lambda row: float(row["contribution_rms"]), reverse=True)
    threshold = max(float(min_contribution_fraction), 0.0)
    significant = [row for row in rows if float(row["contribution_fraction"]) >= threshold]
    if not significant:
        significant = rows
    if max_terms is not None:
        significant = significant[: max(1, int(max_terms))]
    return significant


def compare_spherical_harmonic_fits(
    before: SphericalHarmonicFit,
    after: SphericalHarmonicFit,
    *,
    min_contribution_fraction: float = 0.01,
    max_terms: int | None = None,
) -> list[dict[str, Any]]:
    """Return before/after rows ordered by the larger SH contribution."""

    if len(before.terms) != len(after.terms) or any(
        (a.degree, a.order) != (b.degree, b.order) for a, b in zip(before.terms, after.terms)
    ):
        raise ValueError("before and after fits must use the same SH terms.")
    before_coefficients = np.asarray(before.coefficients, dtype=float).reshape(-1)
    after_coefficients = np.asarray(after.coefficients, dtype=float).reshape(-1)
    before_basis_rms = _fit_basis_rms(before)
    after_basis_rms = _fit_basis_rms(after)
    rows: list[dict[str, Any]] = []
    for index, term in enumerate(before.terms):
        before_contribution = abs(float(before_coefficients[index])) * float(before_basis_rms[index])
        after_contribution = abs(float(after_coefficients[index])) * float(after_basis_rms[index])
        contribution_delta = before_contribution - after_contribution
        row = {
            "term": term,
            "term_label": term.display_label,
            "degree": int(term.degree),
            "order": int(term.order),
            "unit": before.unit or after.unit,
            "before_coefficient": float(before_coefficients[index]),
            "after_coefficient": float(after_coefficients[index]),
            "before_contribution_rms": float(before_contribution),
            "after_contribution_rms": float(after_contribution),
            "contribution_rms_delta": float(contribution_delta),
            "contribution_fraction_before": float(before_contribution / max(before.field_rms, 1.0e-30)),
            "contribution_fraction_after": float(after_contribution / max(after.field_rms, 1.0e-30)),
            "coefficient_abs_delta": float(
                abs(float(before_coefficients[index])) - abs(float(after_coefficients[index]))
            ),
            "contribution_percent_reduction": float(100.0 * contribution_delta / before_contribution)
            if before_contribution > 1.0e-30
            else float("nan"),
            "reduced": bool(contribution_delta >= 0.0),
        }
        row["max_contribution_fraction"] = max(
            float(row["contribution_fraction_before"]), float(row["contribution_fraction_after"])
        )
        row["max_contribution_rms"] = max(before_contribution, after_contribution)
        rows.append(row)

    rows.sort(key=lambda row: float(row["max_contribution_rms"]), reverse=True)
    threshold = max(float(min_contribution_fraction), 0.0)
    significant = [row for row in rows if float(row["max_contribution_fraction"]) >= threshold]
    if not significant:
        significant = rows
    if max_terms is not None:
        significant = significant[: max(1, int(max_terms))]
    return significant


def _coefficient_row(
    fit: SphericalHarmonicFit,
    term: SphericalHarmonicTerm,
    index: int,
    coefficient: float,
    basis_rms: float,
) -> dict[str, Any]:
    contribution = abs(float(coefficient)) * float(basis_rms)
    return {
        "term": term,
        "term_label": term.display_label,
        "degree": int(term.degree),
        "order": int(term.order),
        "unit": fit.unit,
        "coefficient": float(coefficient),
        "contribution_rms": float(contribution),
        "contribution_fraction": float(contribution / max(fit.field_rms, 1.0e-30)),
        "index": int(index),
    }


def _real_spherical_harmonic(degree: int, order: int, cos_theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    abs_order = abs(int(order))
    legendre = _associated_legendre(int(degree), abs_order, cos_theta)
    norm = _spherical_harmonic_normalization(int(degree), abs_order)
    if order < 0:
        return math.sqrt(2.0) * norm * legendre * np.sin(abs_order * phi)
    if order > 0:
        return math.sqrt(2.0) * norm * legendre * np.cos(order * phi)
    return norm * legendre


def _associated_legendre(degree: int, order: int, x: np.ndarray) -> np.ndarray:
    if order < 0 or order > degree:
        raise ValueError("associated Legendre order must satisfy 0 <= order <= degree.")
    x = np.asarray(x, dtype=float)
    p_mm = np.ones_like(x, dtype=float)
    if order > 0:
        somx2 = np.sqrt(np.maximum(0.0, (1.0 - x) * (1.0 + x)))
        factor = 1.0
        for _ in range(order):
            p_mm *= -factor * somx2
            factor += 2.0
    if degree == order:
        return p_mm
    p_mmp1 = x * (2 * order + 1) * p_mm
    if degree == order + 1:
        return p_mmp1
    p_lm_minus2 = p_mm
    p_lm_minus1 = p_mmp1
    for ell in range(order + 2, degree + 1):
        p_lm = ((2 * ell - 1) * x * p_lm_minus1 - (ell + order - 1) * p_lm_minus2) / (ell - order)
        p_lm_minus2 = p_lm_minus1
        p_lm_minus1 = p_lm
    return p_lm_minus1


def _spherical_harmonic_normalization(degree: int, order: int) -> float:
    return math.sqrt(
        (2 * degree + 1) / (4.0 * math.pi) * math.factorial(degree - order) / math.factorial(degree + order)
    )


def _fit_basis_rms(fit: SphericalHarmonicFit) -> np.ndarray:
    if fit.basis_rms is not None:
        return np.asarray(fit.basis_rms, dtype=float).reshape(-1)
    return np.ones(len(fit.coefficients), dtype=float)


def _weighted_rms(values: np.ndarray, weights: np.ndarray | None) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if weights is None:
        return float(np.sqrt(np.mean(values * values))) if len(values) else float("nan")
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(weights * values * values) / weight_sum))


def _weighted_rms_columns(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if weights is None:
        return np.sqrt(np.mean(values * values, axis=0))
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return np.full(values.shape[1], float("nan"), dtype=float)
    return np.sqrt((weights[:, None] * values * values).sum(axis=0) / weight_sum)


def _weighted_center(coords: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.mean(coords, axis=0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return np.mean(coords, axis=0)
    return np.sum(coords * weights[:, None], axis=0) / weight_sum


def _as_coordinate_array(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")
    if len(coords) == 0:
        raise ValueError("coords must contain at least one sample.")
    return coords


def _validated_weights(weights: np.ndarray | None, count: int) -> np.ndarray | None:
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if len(weights) != int(count):
        raise ValueError("weights must match coordinate count.")
    return weights


def _validate_max_order(max_order: int) -> int:
    degree_limit = int(max_order)
    if degree_limit < 0:
        raise ValueError("max_order must be non-negative.")
    return degree_limit


def _term_alias(degree: int, order: int) -> str:
    aliases = {
        (0, 0): "B0 offset / static",
        (1, -1): "Gy linear gradient",
        (1, 0): "Gz linear gradient",
        (1, 1): "Gx linear gradient",
        (2, -2): "XY second-order",
        (2, -1): "YZ second-order",
        (2, 0): "Z2 second-order",
        (2, 1): "XZ second-order",
        (2, 2): "X2-Y2 second-order",
        (3, -3): "Y3-like third-order",
        (3, -2): "XYZ-like third-order",
        (3, -1): "YZ2-like third-order",
        (3, 0): "Z3 third-order",
        (3, 1): "XZ2-like third-order",
        (3, 2): "Z(X2-Y2)-like third-order",
        (3, 3): "X3-like third-order",
    }
    return aliases.get((int(degree), int(order)), "")
