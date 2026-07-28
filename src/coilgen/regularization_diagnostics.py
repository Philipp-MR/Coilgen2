from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .models import CoilGenResult


@dataclass(frozen=True, slots=True)
class RegularizationSelectionReport:
    """User-readable summary of the L-curve/fixed regularization choice."""

    selection_method: str
    candidate_solver: str
    selected_factor: float
    selected_lambda: float
    selected_index: int
    candidate_count: int
    valid_candidate_count: int
    fallback: bool
    fallback_reason: str
    knee_type: str
    residual_log_span: float
    regularization_log_span: float
    selected_residual_position: float
    selected_regularization_position: float
    selected_curvature: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def regularization_scan_arrays(result: CoilGenResult) -> dict[str, Any]:
    diagnostics = getattr(result, "regularization_diagnostics", None) or {}
    factors = np.asarray(diagnostics.get("candidate_factors", ()), dtype=float)
    residuals = np.asarray(diagnostics.get("residual_norms", ()), dtype=float)
    regularization_norms = np.asarray(diagnostics.get("regularization_norms", ()), dtype=float)
    curvature = np.asarray(diagnostics.get("curvature", ()), dtype=float)
    if len(factors) == 0 or len(residuals) != len(factors) or len(regularization_norms) != len(factors):
        raise ValueError("Result has no L-curve regularization scan diagnostics.")
    if len(curvature) != len(factors):
        curvature = np.full(len(factors), np.nan, dtype=float)
    selected_index = int(diagnostics.get("selected_index", -1))
    if selected_index < 0 or selected_index >= len(factors):
        raise ValueError("L-curve diagnostics have no valid selected regularization index.")
    valid = (
        np.isfinite(factors)
        & np.isfinite(residuals)
        & np.isfinite(regularization_norms)
        & (factors > 0.0)
        & (residuals > 0.0)
        & (regularization_norms > 0.0)
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError("L-curve regularization scan has fewer than three valid points.")
    if not bool(valid[selected_index]):
        raise ValueError("Selected L-curve regularization point is not finite and positive.")
    return {
        "factors": factors,
        "residuals": residuals,
        "regularization_norms": regularization_norms,
        "curvature": curvature,
        "selected_index": selected_index,
        "valid": valid,
    }


def regularization_selection_report(result: CoilGenResult) -> RegularizationSelectionReport:
    """Return a robust interpretation of the regularization selection.

    The raw L-curve plot is still the primary visual diagnostic. This helper
    adds machine-checkable labels that are useful in reports and examples:
    boundary selections, underregularized plateau picks, and late current-norm
    collapse picks are explicitly flagged.
    """

    diagnostics = getattr(result, "regularization_diagnostics", None) or {}
    method = str(diagnostics.get("selection_method", "unknown"))
    if method != "l_curve":
        return _fixed_regularization_report(result, diagnostics, method)

    scan = regularization_scan_arrays(result)
    factors = scan["factors"]
    residuals = scan["residuals"]
    regularization_norms = scan["regularization_norms"]
    curvature = scan["curvature"]
    valid = scan["valid"]
    selected_index = int(scan["selected_index"])
    valid_count = int(np.count_nonzero(valid))

    residual_position = normalized_log_trace(residuals)
    regularization_position = 1.0 - normalized_log_trace(regularization_norms)
    selected_residual_position = _safe_index_float(residual_position, selected_index)
    selected_regularization_position = _safe_index_float(regularization_position, selected_index)
    selected_curvature = _safe_index_float(curvature, selected_index)
    residual_log_span = _positive_log_span(residuals[valid])
    regularization_log_span = _positive_log_span(regularization_norms[valid])
    fallback = bool(diagnostics.get("fallback", False))
    fallback_reason = str(diagnostics.get("fallback_reason", ""))
    knee_type = _classify_l_curve_knee(
        selected_index,
        len(factors),
        selected_residual_position,
        selected_regularization_position,
        fallback=fallback,
    )
    warnings = _regularization_selection_warnings(
        knee_type=knee_type,
        selected_index=selected_index,
        candidate_count=len(factors),
        valid_candidate_count=valid_count,
        residual_log_span=residual_log_span,
        regularization_log_span=regularization_log_span,
        fallback=fallback,
        fallback_reason=fallback_reason,
    )
    return RegularizationSelectionReport(
        selection_method=method,
        candidate_solver=str(diagnostics.get("candidate_solver", "")),
        selected_factor=float(diagnostics.get("selected_factor", result.regularization_factor)),
        selected_lambda=float(diagnostics.get("selected_lambda", result.regularization_lambda)),
        selected_index=selected_index,
        candidate_count=int(len(factors)),
        valid_candidate_count=valid_count,
        fallback=fallback,
        fallback_reason=fallback_reason,
        knee_type=knee_type,
        residual_log_span=residual_log_span,
        regularization_log_span=regularization_log_span,
        selected_residual_position=selected_residual_position,
        selected_regularization_position=selected_regularization_position,
        selected_curvature=selected_curvature,
        warnings=warnings,
    )


def regularization_candidate_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    """Return per-candidate L-curve diagnostics for CSV/Markdown tables."""

    scan = regularization_scan_arrays(result)
    factors = scan["factors"]
    residuals = scan["residuals"]
    regularization_norms = scan["regularization_norms"]
    curvature = scan["curvature"]
    valid = scan["valid"]
    selected_index = int(scan["selected_index"])
    residual_position = normalized_log_trace(residuals)
    regularization_position = 1.0 - normalized_log_trace(regularization_norms)
    rows: list[dict[str, Any]] = []
    for index in range(len(factors)):
        rows.append(
            {
                "index": int(index),
                "regularization_factor": float(factors[index]),
                "residual_norm_T_per_A": float(residuals[index]),
                "resistance_seminorm_ohm_A": float(regularization_norms[index]),
                "curvature": float(curvature[index]) if np.isfinite(curvature[index]) else float("nan"),
                "residual_position": float(residual_position[index]),
                "regularization_position": float(regularization_position[index]),
                "valid": bool(valid[index]),
                "selected": index == selected_index,
            }
        )
    return rows


def normalized_log_trace(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if len(finite) == 0:
        return np.zeros_like(values)
    logs = np.log10(np.maximum(values, np.min(finite)))
    return normalized_linear_trace(logs)


def normalized_linear_trace(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high, rtol=1e-12, atol=1e-15):
        return np.full_like(values, 0.5, dtype=float)
    return (values - low) / (high - low)


def _fixed_regularization_report(
    result: CoilGenResult,
    diagnostics: dict[str, Any],
    method: str,
) -> RegularizationSelectionReport:
    return RegularizationSelectionReport(
        selection_method=method,
        candidate_solver=str(diagnostics.get("candidate_solver", "")),
        selected_factor=float(result.regularization_factor),
        selected_lambda=float(result.regularization_lambda),
        selected_index=int(diagnostics.get("selected_index", -1)),
        candidate_count=0,
        valid_candidate_count=0,
        fallback=bool(diagnostics.get("fallback", False)),
        fallback_reason=str(diagnostics.get("fallback_reason", "")),
        knee_type="fixed",
        residual_log_span=float("nan"),
        regularization_log_span=float("nan"),
        selected_residual_position=float("nan"),
        selected_regularization_position=float("nan"),
        selected_curvature=float("nan"),
        warnings=tuple(),
    )


def _classify_l_curve_knee(
    selected_index: int,
    candidate_count: int,
    selected_residual_position: float,
    selected_regularization_position: float,
    *,
    fallback: bool,
) -> str:
    if fallback:
        return "fallback"
    if candidate_count <= 0 or selected_index < 0:
        return "unknown"
    edge_guard = max(1, int(np.ceil(0.05 * candidate_count)))
    if selected_index < edge_guard:
        return "underregularized_boundary"
    if selected_index >= candidate_count - edge_guard:
        return "overregularized_boundary"
    if selected_residual_position < 0.02 and selected_regularization_position < 0.02:
        return "underregularized_plateau"
    if selected_residual_position > 0.85 or selected_regularization_position > 0.85:
        return "overregularized_collapse"
    if selected_residual_position < 0.25 and selected_regularization_position < 0.35:
        return "first_tradeoff_knee"
    return "tradeoff_knee"


def _regularization_selection_warnings(
    *,
    knee_type: str,
    selected_index: int,
    candidate_count: int,
    valid_candidate_count: int,
    residual_log_span: float,
    regularization_log_span: float,
    fallback: bool,
    fallback_reason: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if fallback:
        warnings.append(f"Regularization fallback was used: {fallback_reason or 'no reliable reason recorded'}.")
    if valid_candidate_count < max(3, candidate_count // 2):
        warnings.append("Many L-curve candidates were invalid; check scan range and solver conditioning.")
    if np.isfinite(residual_log_span) and residual_log_span < 1.0e-3:
        warnings.append("Field-residual dynamic range is very small; L-curve knee may be weak.")
    if np.isfinite(regularization_log_span) and regularization_log_span < 1.0e-3:
        warnings.append("Resistance-seminorm dynamic range is very small; L-curve knee may be weak.")
    if knee_type in {"underregularized_boundary", "overregularized_boundary"}:
        warnings.append(f"L-curve selected a scan-boundary point ({selected_index + 1}/{candidate_count}).")
    if knee_type == "underregularized_plateau":
        warnings.append(
            "L-curve selection is on the underregularized plateau; increase lower regularization range or smoothness."
        )
    if knee_type == "overregularized_collapse":
        warnings.append(
            "L-curve selection is on the late current-collapse arm; inspect whether the first trade-off knee is preferable."
        )
    return tuple(dict.fromkeys(warnings))


def _positive_log_span(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return float("nan")
    return float(np.ptp(np.log(finite)))


def _safe_index_float(values: np.ndarray, index: int) -> float:
    values = np.asarray(values, dtype=float)
    if index < 0 or index >= len(values):
        return float("nan")
    return float(values[index])
