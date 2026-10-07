"""Fixed-current field checks for proposed group routes; no current fitting."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .config import CoilGenConfig
from .engineering_metrics import estimate_suppression_normals
from .field_components import field_component_values, normalize_field_component
from .models import TargetField
from .gradient_linearity import (
    _normalise_linearity_region,
    gradient_linearity_enabled,
    gradient_linearity_region_mask,
    gradient_linearity_summary,
)


@dataclass(frozen=True)
class GroupRouteFieldAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | str | bool]


def _key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _suppression_component(config: CoilGenConfig) -> str:
    raw = _key(config.suppression_field_component)
    if raw in {"", "target", "same", "useful", "target_component"}:
        return normalize_field_component(config.target_field_component)
    if raw in {"normal", "bnormal", "b_normal", "normal_field"}:
        return "Bnormal"
    if raw in {"components", "component", "componentwise", "component_wise", "cartesian_components", "xyz", "bx_by_bz"}:
        return "components"
    return normalize_field_component(raw, allow_bmag=True)


def _rms(values: np.ndarray, weights: np.ndarray) -> float:
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return 0.0
    weights = weights / np.max(weights)
    return scale * float(np.sqrt(np.dot(weights, (values / scale) ** 2) / np.sum(weights)))


def _std(values: np.ndarray, weights: np.ndarray) -> float:
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return 0.0
    normalized = values / scale
    weights = weights / np.max(weights)
    mean = float(np.dot(weights, normalized) / np.sum(weights))
    return scale * _rms(normalized - mean, weights)


def _roundoff(*values: float) -> float:
    return 64 * np.finfo(float).eps * max(*(abs(value) for value in values), 1e-300)


def _maximum_longitudinal_delta(delta: np.ndarray, b0_direction) -> float:
    """Normalize only the optional hard-budget path with overflow-safe scaling."""
    direction = np.asarray(b0_direction, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("B0 direction must contain three finite values.")
    scale = float(np.max(np.abs(direction)))
    if scale == 0:
        raise ValueError("B0 direction must be nonzero.")
    direction = direction / scale
    norm = float(np.linalg.norm(direction))
    if scale <= 1e-15 / norm:
        raise ValueError("B0 direction must be nonzero.")
    direction /= norm
    return float(np.max(np.abs(delta @ direction)))


def _suppression_normals(target: TargetField, mask: np.ndarray, config: CoilGenConfig) -> np.ndarray:
    source = _key(config.suppression_normal_source)
    if source in {"mesh", "target", "metadata"}:
        normals = np.asarray(target.suppression_normals, dtype=float)
        if normals.shape != target.coords.shape:
            raise ValueError("Normal suppression requires the declared per-point normals.")
        normals = normals[mask]
    elif source in {"radial", "cylindrical", "box"}:
        normals = estimate_suppression_normals(target.coords[mask], model=source)
    else:
        raise ValueError("Unknown suppression normal source.")
    lengths = np.hypot.reduce(normals, axis=1)
    if not np.all(np.isfinite(normals)) or np.any(~np.isfinite(lengths) | (lengths <= 0.0)):
        raise ValueError("Active suppression normals must be finite and nonzero.")
    return normals / lengths[:, None]


def _suppression_residuals(
    residual: np.ndarray, target: TargetField, mask: np.ndarray, config: CoilGenConfig, component: str
) -> tuple[np.ndarray, np.ndarray]:
    if component == "Bnormal":
        values = np.einsum("ij,ij->i", residual, _suppression_normals(target, mask, config))
        return np.abs(values), np.abs(values)
    if component in {"vector", "Bmag", "components"}:
        # The optimizer's Bmag constraint is the norm of vector residuals,
        # not the difference of field magnitudes (which could cancel).
        norms = np.hypot.reduce(residual, axis=1)
        peaks = np.abs(residual).reshape(-1) if component == "components" else norms
        return norms, peaks
    values = field_component_values(residual, component, b0_direction=config.b_0_direction)
    return np.abs(values), np.abs(values)


def _constraint_violation(
    reason, before, after, before_met, after_met, policy, absolute, inherited, *, roundoff=None, relaxed=None
):
    """Record absolute failures; policy may permit a measured scalar violation."""
    if after_met:
        return ()
    absolute.append(reason)
    if relaxed is not None:
        if before_met:
            relaxed["newly_failed"].append(reason)
        else:
            inherited.append(reason)
            tolerance = _roundoff(before, after) if roundoff is None else roundoff
            if np.isfinite(before) and np.isfinite(after) and after > before + tolerance:
                relaxed["regressed_inherited"].append(reason)
    if policy == "budget_only" and np.isfinite(before) and np.isfinite(after):
        relaxed["waived"].append(reason)
        return ()
    if (
        policy == "non_regression"
        and not before_met
        and np.isfinite(before)
        and np.isfinite(after)
        and after <= before + (_roundoff(before, after) if roundoff is None else roundoff)
    ):
        inherited.append(reason)
        return ()
    return (reason,)


def _linearity_checks(
    baseline, candidate, target, config, current, active, metrics, policy, absolute, inherited, relaxed=None
):
    """Apply the existing Bz gradient contract at the unchanged drive current."""
    if not gradient_linearity_enabled(config):
        return ()
    mask = gradient_linearity_region_mask(target, config) & active
    # The generic reporting helper falls back to all samples for an empty DSV;
    # an acceptance gate must not substitute suppression points for that domain.
    if _normalise_linearity_region(config.linearity_region) == "dsv":
        mask &= ~np.asarray(target.suppressed, dtype=bool)
    coords = np.asarray(target.coords)[mask]
    if len(coords) < 4:
        absolute.append("linearity_domain_not_certified")
        return ("linearity_domain_not_certified",)
    centered = coords - np.mean(coords, axis=0)
    scale = float(np.max(np.abs(centered)))
    if scale == 0 or np.linalg.matrix_rank(np.column_stack((np.ones(len(coords)), centered / scale))) < 4:
        absolute.append("linearity_domain_rank_deficient")
        return ("linearity_domain_rank_deficient",)
    selected_target = SimpleNamespace(
        coords=coords, b=np.asarray(target.b)[mask], suppressed=np.asarray(target.suppressed)[mask]
    )
    summaries = {}
    for name, values in (("baseline", baseline), ("candidate", candidate)):
        summary = gradient_linearity_summary(values[mask] * current, selected_target, config)
        summaries[name] = summary
        metrics.update({f"{name}_{key}": value for key, value in summary.items()})
    reasons = []
    contracts = (
        ("gradient_strength", "gradient_strength_error_percent", "gradient_strength_tolerance_percent"),
        ("rms", "rms_percent", "linearity_rms_tolerance_percent"),
        ("peak", "peak_percent", "linearity_peak_tolerance_percent"),
        ("cross_gradient", "cross_gradient_percent", "linearity_cross_gradient_tolerance_percent"),
    )
    for name, value_key, tolerance_key in contracts:
        if float(getattr(config, tolerance_key)) <= 0:
            continue
        before, after = (summaries[stage][f"linearity_{value_key}"] for stage in ("baseline", "candidate"))
        before_met, after_met = (summaries[stage][f"linearity_{name}_met"] for stage in ("baseline", "candidate"))
        reason = "gradient_strength_contract" if name == "gradient_strength" else f"linearity_{name}_contract"
        rejected = _constraint_violation(
            reason, before, after, before_met, after_met, policy, absolute, inherited, relaxed=relaxed
        )
        if rejected:
            reasons.append("gradient_linearity_contract")
    return tuple(dict.fromkeys(reasons))


def assess_group_route_field(
    baseline_per_amp: np.ndarray,
    candidate_per_amp: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
    current_a: float,
    maximum_delta_t_per_a: float,
    *,
    maximum_longitudinal_field_delta_t_per_a: float | None = None,
) -> GroupRouteFieldAssessment:
    """Check full fields on the same target grid at one specified nonzero current.

    The explicit budget bounds every active sample's Euclidean vector change
    per ampere, including transverse components and suppression points. Enabled
    layout RMS/std limits apply at ``current_a``. Absolute qualification is the
    default. Explicit ``non_regression`` preserves each passing field limit and
    permits inherited violations only when they do not worsen beyond roundoff;
    absolute failures are still reported. Explicit ``budget_only`` permits
    measured gradient/layout scalar failures within the hard delta budgets,
    while distinguishing new and inherited failures. An optional pointwise
    longitudinal budget uses the normalized configured B0 direction on the same
    active samples. Suppression residual RMS and peak
    always remain nonregression constraints. Optimization percentages grant no
    implicit permission, and no policy relaxes the vector-change budget.

    Zero/nonfinite current, absent context, or an empty active domain cannot
    authorize a route. Zero/negative/nonfinite-weight samples are inactive;
    nonfinite data at an active point are rejected. No input is mutated.
    """
    metrics: dict[str, float | int | str | bool] = {"checked": False, "current_refitted": False}
    reasons: list[str] = []
    absolute: list[str] = []
    inherited: list[str] = []
    relaxed = None
    try:
        if target_field is None or config is None:
            raise ValueError("TargetField and configuration are required.")
        policy = _key(getattr(config, "joint_group_routing_field_acceptance", "absolute"))
        if policy not in {"absolute", "non_regression", "budget_only"}:
            raise ValueError("Unknown joint routing field acceptance policy.")
        metrics["field_acceptance_policy"] = policy
        if policy == "budget_only":
            relaxed = {"waived": [], "newly_failed": [], "regressed_inherited": []}
        current, budget = float(current_a), float(maximum_delta_t_per_a)
        if not np.isfinite(current) or current == 0.0:
            return GroupRouteFieldAssessment(False, ("invalid_fixed_current",), metrics)
        if not np.isfinite(budget) or budget < 0.0:
            return GroupRouteFieldAssessment(False, ("invalid_field_delta_budget",), metrics)
        longitudinal_budget = maximum_longitudinal_field_delta_t_per_a
        if longitudinal_budget is None:
            if getattr(config, "joint_group_routing_max_longitudinal_field_change_rel", None) is not None:
                return GroupRouteFieldAssessment(False, ("missing_longitudinal_field_delta_budget",), metrics)
        else:
            if isinstance(longitudinal_budget, (bool, np.bool_)):
                return GroupRouteFieldAssessment(False, ("invalid_longitudinal_field_delta_budget",), metrics)
            longitudinal_budget = float(longitudinal_budget)
            if not np.isfinite(longitudinal_budget) or longitudinal_budget < 0:
                return GroupRouteFieldAssessment(False, ("invalid_longitudinal_field_delta_budget",), metrics)
        baseline, candidate = np.asarray(baseline_per_amp, dtype=float), np.asarray(candidate_per_amp, dtype=float)
        coords, target = np.asarray(target_field.coords, dtype=float), np.asarray(target_field.b, dtype=float)
        weights = np.asarray(target_field.weights, dtype=float).reshape(-1)
        suppressed = np.asarray(target_field.suppressed, dtype=bool).reshape(-1)
        if coords.ndim != 2 or coords.shape[1:] != (3,):
            raise ValueError("Target coordinates must have shape (N, 3).")
        if any(array.shape != coords.shape for array in (baseline, candidate, target)):
            raise ValueError("Both fields and target values must match target coordinates.")
        if weights.shape != (len(coords),) or suppressed.shape != weights.shape:
            raise ValueError("Weights and suppression membership must match target points.")
        active = np.isfinite(weights) & (weights > 0.0)
        dsv, suppression = active & ~suppressed, active & suppressed
        metrics.update(
            current_a=current,
            maximum_delta_t_per_a=budget,
            active_sample_count=int(np.count_nonzero(active)),
            dsv_sample_count=int(np.count_nonzero(dsv)),
            suppression_sample_count=int(np.count_nonzero(suppression)),
        )
        if not np.any(active):
            return GroupRouteFieldAssessment(False, ("no_active_field_samples",), metrics)
        if not all(np.all(np.isfinite(array[active])) for array in (baseline, candidate, target, coords)):
            return GroupRouteFieldAssessment(False, ("nonfinite_active_field_context",), metrics)
        names = (
            "layout_target_error_std_limit_T",
            "layout_target_error_rms_limit_T",
            "layout_target_error_acceptance_absolute_T",
            "suppression_field_rms_limit_T",
            "suppression_field_peak_limit_T",
            "linearity_rms_tolerance_percent",
            "linearity_peak_tolerance_percent",
            "linearity_cross_gradient_tolerance_percent",
            "gradient_strength_tolerance_percent",
        )
        limits = {name: float(getattr(config, name)) for name in names}
        if any(not np.isfinite(value) or value < 0 for value in limits.values()):
            raise ValueError("Explicit field limits and acceptance tolerances must be finite and nonnegative.")
        component = normalize_field_component(config.target_field_component)
        with np.errstate(over="raise", invalid="raise"):
            maximum_delta = float(np.max(np.hypot.reduce(candidate[active] - baseline[active], axis=1)))
            metrics["maximum_pointwise_vector_delta_t_per_a"] = maximum_delta
            if maximum_delta > budget:
                reasons.append("pointwise_field_delta_budget")
            if longitudinal_budget is not None:
                maximum_longitudinal_delta = _maximum_longitudinal_delta(
                    candidate[active] - baseline[active], config.b_0_direction
                )
                metrics["maximum_longitudinal_field_delta_t_per_a"] = longitudinal_budget
                metrics["maximum_pointwise_longitudinal_delta_t_per_a"] = maximum_longitudinal_delta
                if not np.isfinite(maximum_longitudinal_delta) or maximum_longitudinal_delta > longitudinal_budget:
                    reasons.append("pointwise_longitudinal_field_delta_budget")
            reasons.extend(
                _linearity_checks(
                    baseline,
                    candidate,
                    target_field,
                    config,
                    current,
                    active,
                    metrics,
                    policy,
                    absolute,
                    inherited,
                    relaxed=relaxed,
                )
            )
            if np.any(dsv):
                row_weights = np.repeat(weights[dsv], 3) if component == "vector" else weights[dsv]
                for name, field in (("baseline", baseline), ("candidate", candidate)):
                    residual = field_component_values(
                        field[dsv] * current - target[dsv], component, b0_direction=config.b_0_direction
                    )
                    metrics[f"{name}_dsv_residual_rms_T"] = _rms(residual, row_weights)
                    metrics[f"{name}_dsv_residual_std_T"] = _std(residual, row_weights)
                for metric in ("std", "rms"):
                    limit = limits[f"layout_target_error_{metric}_limit_T"]
                    effective = limit + limits["layout_target_error_acceptance_absolute_T"]
                    if limit > 0:
                        before, after = (
                            metrics[f"{stage}_dsv_residual_{metric}_T"] for stage in ("baseline", "candidate")
                        )
                        reasons.extend(
                            _constraint_violation(
                                f"layout_target_{metric}_limit",
                                before,
                                after,
                                before <= effective,
                                after <= effective,
                                policy,
                                absolute,
                                inherited,
                                relaxed=relaxed,
                            )
                        )
            elif limits["layout_target_error_std_limit_T"] > 0 or limits["layout_target_error_rms_limit_T"] > 0:
                reasons.append("layout_target_unchecked")
                absolute.append("layout_target_unchecked")
            if np.any(suppression):
                suppression_component = _suppression_component(config)
                weighting = _key(config.suppression_surface_weighting)
                point_weights = np.ones(int(np.count_nonzero(suppression)))
                if weighting in {"area", "mesh_area", "surface_area", "vertex_area"}:
                    areas = np.asarray(target_field.suppression_area_weights, dtype=float)
                    if areas.shape != weights.shape or not np.all(
                        np.isfinite(areas[suppression]) & (areas[suppression] > 0)
                    ):
                        raise ValueError("Area-weighted suppression requires finite positive active area weights.")
                    point_weights = areas[suppression]
                elif weighting not in {"", "none", "uniform", "point", "points"}:
                    raise ValueError("Unknown suppression surface weighting.")
                metrics["suppression_component"] = suppression_component
                scale = float(np.max(np.abs(target[suppression])))
                for name, field in (("baseline", baseline), ("candidate", candidate)):
                    at_current = field[suppression] * current
                    scale = max(scale, float(np.max(np.abs(at_current))))
                    values, peaks = _suppression_residuals(
                        at_current - target[suppression], target_field, suppression, config, suppression_component
                    )
                    metrics[f"{name}_suppression_residual_rms_T"] = _rms(values, point_weights)
                    metrics[f"{name}_suppression_residual_peak_T"] = float(np.max(peaks))
                tolerance = _roundoff(scale)
                metrics["suppression_roundoff_tolerance_T"] = tolerance
                for metric in ("rms", "peak"):
                    before, after = (
                        float(metrics[f"{name}_suppression_residual_{metric}_T"]) for name in ("baseline", "candidate")
                    )
                    if after > before + tolerance:
                        reasons.append(f"suppression_{metric}_regression")
                    limit = limits[f"suppression_field_{metric}_limit_T"]
                    if limit > 0:
                        reasons.extend(
                            _constraint_violation(
                                f"suppression_{metric}_limit",
                                before,
                                after,
                                before <= limit + tolerance,
                                after <= limit + tolerance,
                                "non_regression" if policy == "budget_only" else policy,
                                absolute,
                                inherited,
                                roundoff=tolerance,
                                relaxed=relaxed,
                            )
                        )
            elif limits["suppression_field_rms_limit_T"] > 0 or limits["suppression_field_peak_limit_T"] > 0:
                reasons.append("suppression_limit_unchecked")
                absolute.append("suppression_limit_unchecked")
        metrics["checked"] = True
    except (AttributeError, TypeError, ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        metrics["context_error"] = str(error)
        reasons.append("invalid_field_context")
    metrics.update(
        absolute_field_contract_met=bool(metrics["checked"] and not absolute),
        absolute_field_contract_reasons="; ".join(dict.fromkeys(absolute)),
        inherited_field_contract_violations="; ".join(dict.fromkeys(inherited)),
    )
    if metrics.get("field_acceptance_policy") == "budget_only":
        metrics.update(
            budget_waived_field_contract_violations="; ".join(dict.fromkeys(relaxed["waived"])),
            newly_failed_field_contract_violations="; ".join(dict.fromkeys(relaxed["newly_failed"])),
            regressed_inherited_field_contract_violations="; ".join(dict.fromkeys(relaxed["regressed_inherited"])),
        )
    return GroupRouteFieldAssessment(not reasons, tuple(dict.fromkeys(reasons)), metrics)
