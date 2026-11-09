from __future__ import annotations

from typing import Any

import numpy as np


def current_scale_fit(actual: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    """Weighted best single current scale for a field, and the shape error left.

    A winding is driven by a current, so the useful field error is the one left
    after choosing that current: ``scale`` minimizes the weighted RMS of
    ``scale * actual - target``. ``relative_rms_error`` is that residual RMS
    over the target RMS. Rows may be scalars or vectors (magnitude RMS).
    Empty dict when undefined (no field, no target, invalid weights, or reversed
    polarity). Zero-weight rows do not contribute to the fit.
    """
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    if actual.ndim == 0 or target.ndim == 0 or actual.size == 0 or target.size == 0:
        return {}
    actual = actual.reshape(len(actual), -1)
    target = target.reshape(len(target), -1)
    if actual.shape != target.shape:
        return {}
    w = np.ones(len(target)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != len(target) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        return {}
    active = w > 0
    actual, target, w = actual[active], target[active], w[active]
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(target)):
        return {}
    actual_scale = float(np.max(np.abs(actual)))
    target_scale = float(np.max(np.abs(target)))
    if actual_scale == 0 or target_scale == 0:
        return {}
    # Normalize before squaring or summing to retain finite fits across SI scales.
    actual, target = actual / actual_scale, target / target_scale
    w = w / np.max(w)
    w = w / np.sum(w)
    ff = float(w @ np.sum(actual * actual, axis=1))
    fb = float(w @ np.sum(actual * target, axis=1))
    bb = float(w @ np.sum(target * target, axis=1))
    if not (np.isfinite(ff) and np.isfinite(fb) and np.isfinite(bb)) or ff <= 0.0 or bb <= 0.0 or fb <= 0.0:
        return {}
    relative_scale = fb / ff
    scale = relative_scale * (target_scale / actual_scale)
    if not np.isfinite(scale) or scale <= 0:
        return {}
    residual = float(w @ np.sum((relative_scale * actual - target) ** 2, axis=1))
    return {
        "best_fit_current_scale": scale,
        "best_fit_relative_rms_error": float(np.sqrt(max(residual, 0.0) / bb)),
    }


def dsv_target_mask(target_field: Any) -> np.ndarray:
    """Return the internal DSV samples used for gradient-performance metrics."""

    coords = np.asarray(getattr(target_field, "coords", []), dtype=float)
    point_count = len(coords)
    suppressed = np.asarray(
        getattr(target_field, "suppressed", np.zeros(point_count, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if len(suppressed) != point_count:
        return np.ones(point_count, dtype=bool)
    mask = ~suppressed
    if not np.any(mask):
        return np.ones(point_count, dtype=bool)
    return mask


def suppression_target_mask(target_field: Any) -> np.ndarray:
    """Return secondary suppression samples used for fringe-field metrics."""

    coords = np.asarray(getattr(target_field, "coords", []), dtype=float)
    point_count = len(coords)
    suppressed = np.asarray(
        getattr(target_field, "suppressed", np.zeros(point_count, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if len(suppressed) != point_count:
        return np.zeros(point_count, dtype=bool)
    return suppressed


def target_mode_fit_performance(
    field_per_amp: Any,
    target_bz: np.ndarray,
    target_scale: float,
    *,
    sample_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit the requested Bz target mode in field-per-amp data.

    The reported gradient efficiency is the amplitude of the requested target
    mode per ampere. Orthogonal Bz components do not increase the efficiency;
    they remain in the relative error. For Bz = G*y, the target mode is y [m]
    and the fitted coefficient beta has units T/(m A). A uniform nonzero target
    has no gradient efficiency: its field efficiency is the absolute mean
    selected Bz per ampere [mT/A], independent of target-scale normalization.

    This vector-only helper has no spatial coordinates. Its nonuniform-target
    gradient metric assumes target_scale is an SI gradient [T/m] and that
    target_bz/target_scale is a mode in meters with unit reference gradient.
    Result reporting instead resolves the physical target gradient from the
    DSV coordinates, including arbitrary target normalizations and Hz inputs.
    """

    prepared = _prepare_bz_vectors(field_per_amp, target_bz, sample_mask=sample_mask)
    if prepared is None:
        return nan_fit_performance()
    field_vec, target_vec = prepared
    scale_value = float(target_scale)
    if abs(scale_value) <= 1e-30 or not np.isfinite(scale_value):
        return nan_fit_performance()

    target_mode = target_vec / scale_value
    beta = target_mode_slope(field_vec, target_mode)
    if not np.isfinite(beta) or abs(beta) <= 1e-30:
        return nan_fit_performance()

    current = float(scale_value / beta)
    fitted = field_vec * current
    peak_target = max(float(np.max(np.abs(target_vec))), 1e-30)
    rel = np.abs((fitted - target_vec) / peak_target) * 100.0
    uniform_field = bool(
        np.max(np.abs(target_vec)) > 0
        and np.ptp(target_vec) <= 128.0 * np.finfo(float).eps * np.max(np.abs(target_vec))
    )
    return {
        "needed_current_A": current,
        "gradient_efficiency_mT_per_m_per_A": float("nan") if uniform_field else 1000.0 * abs(float(beta)),
        "field_efficiency_mT_per_A": 1000.0 * abs(float(np.mean(field_vec))) if uniform_field else float("nan"),
        "mean_rel_error_percent": float(np.mean(rel)),
        "max_rel_error_percent": float(np.max(rel)),
        "sample_count": int(len(field_vec)),
    }


def fit_target_mode_current(
    field_per_amp: Any,
    target_bz: np.ndarray,
    target_scale: float,
    *,
    sample_mask: np.ndarray | None = None,
) -> float:
    """Return the current that best realizes the target Bz mode."""

    return float(
        target_mode_fit_performance(
            field_per_amp,
            target_bz,
            target_scale,
            sample_mask=sample_mask,
        )["needed_current_A"]
    )


def nan_fit_performance() -> dict[str, float]:
    return {
        "needed_current_A": float("nan"),
        "gradient_efficiency_mT_per_m_per_A": float("nan"),
        "field_efficiency_mT_per_A": float("nan"),
        "mean_rel_error_percent": float("nan"),
        "max_rel_error_percent": float("nan"),
        "sample_count": 0,
    }


def target_mode_slope(field_vec: np.ndarray, target_mode: np.ndarray) -> float:
    """Fit field_vec = beta * target_mode with offset removed for gradients."""

    field_vec = np.asarray(field_vec, dtype=float).reshape(-1)
    target_mode = np.asarray(target_mode, dtype=float).reshape(-1)
    if len(field_vec) != len(target_mode) or len(field_vec) == 0:
        return float("nan")

    field_centered = field_vec - float(np.mean(field_vec))
    mode_centered = target_mode - float(np.mean(target_mode))
    denom = float(mode_centered @ mode_centered)
    if denom > 1e-30:
        return float((mode_centered @ field_centered) / denom)

    denom = float(target_mode @ target_mode)
    if denom <= 1e-30:
        return float("nan")
    return float((target_mode @ field_vec) / denom)


def _prepare_bz_vectors(
    field_per_amp: Any,
    target_bz: np.ndarray,
    *,
    sample_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if field_per_amp is None:
        return None
    field = np.asarray(field_per_amp, dtype=float)
    if field.ndim == 2:
        if field.shape[1] < 3:
            return None
        field_bz = field[:, 2]
    else:
        field_bz = field.reshape(-1)
    target = np.asarray(target_bz, dtype=float).reshape(-1)
    if field_bz.shape[0] != target.shape[0]:
        return None

    finite = np.isfinite(field_bz) & np.isfinite(target)
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
        if len(mask) != len(finite):
            return None
        finite &= mask
    if not np.any(finite):
        return None
    return field_bz[finite], target[finite]
