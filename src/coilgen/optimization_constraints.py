"""Physical residual constraints and independent post-solve acceptance diagnostics."""

from __future__ import annotations
from .optimization_problem import StreamFunctionOptimizationProblem
from dataclasses import dataclass
from typing import Any
import numpy as np
from .config import CoilGenConfig
from .config_semantics import normalize_field_budget_metric
from .engineering_limits import (
    engineering_contour_max_levels,
    engineering_contouring_enabled,
    engineering_max_contour_step_A,
    engineering_surface_current_density_limit_A_per_m,
)
from .field_components import field_component_values, sensitivity_component_matrix
from .gradient_linearity import (
    gradient_linearity_enabled,
    gradient_linearity_reference_scale_T,
    gradient_linearity_region_mask,
)


@dataclass(frozen=True)
class ResidualConstraintBlock:
    """One physically meaningful residual constraint in the stream-function solve."""

    name: str
    matrix: np.ndarray
    target: np.ndarray
    allowed_residual: float
    reference_norm: float
    relative_tolerance_percent: float
    absolute_tolerance: float
    rms_field_limit_T: float | None = None
    metric: str = "norm"
    field_limit_T: float | None = None
    field_metric_denominator: float | None = None
    component: str = ""
    weighting: str = "uniform"
    group_size: int = 1
    # Reporting metadata only; matrices, budgets and feasibility remain in the
    # original solver row units. STD rows have already lost their DC offset.
    row_weights: np.ndarray | None = None


def _constrained_solution_contract_diagnostics(
    problem: StreamFunctionOptimizationProblem,
    reduced_stream_function: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, Any]:
    """Recompute every constrained-solver bound from the returned physical solution."""

    reduced = np.asarray(reduced_stream_function, dtype=float).reshape(-1)
    full = np.asarray(problem.expand(reduced), dtype=float).reshape(-1)
    finite_solution = bool(np.all(np.isfinite(reduced)) and np.all(np.isfinite(full)))
    blocks = _target_error_constraint_blocks(problem, config)
    diagnostics = _residual_constraint_diagnostics(problem, reduced, blocks, config)
    rows: list[dict[str, Any]] = []
    for block in blocks:
        denominator = _constraint_field_metric_denominator(block) if block.metric in {"std", "rms"} else 1.0
        rows.append(
            {
                "name": block.name,
                "value": float(diagnostics[f"{block.name}_constraint_residual_norm_T"]) / denominator,
                "limit": float(block.allowed_residual) / denominator,
                "effective_limit": float(
                    diagnostics[f"{block.name}_constraint_allowed_with_acceptance_residual_norm_T"]
                )
                / denominator,
                "unit": "T/m" if block.name in {"gradient_strength", "cross_gradient"} else "T",
                "met": bool(diagnostics[f"{block.name}_constraint_met"]),
                "available": bool(
                    np.isfinite(diagnostics[f"{block.name}_constraint_residual_norm_T"])
                    and np.isfinite(block.allowed_residual)
                ),
            }
        )
    requested_residual_blocks = {
        "linearity_rms": "linearity_rms_tolerance_percent",
        "linearity_peak": "linearity_peak_tolerance_percent",
        "gradient_strength": "gradient_strength_tolerance_percent",
        "cross_gradient": "linearity_cross_gradient_tolerance_percent",
        "suppression": "suppression_field_rms_limit_T",
        "suppression_peak": "suppression_field_peak_limit_T",
    }
    built_blocks = {block.name for block in blocks}
    for name, parameter in requested_residual_blocks.items():
        if _positive_config_float(config, parameter) <= 0.0 or name in built_blocks:
            continue
        diagnostics[f"{name}_constraint_met"] = False
        rows.append(
            {
                "name": name,
                "value": float("nan"),
                "limit": float("nan"),
                "effective_limit": float("nan"),
                "unit": "T/m" if name in {"gradient_strength", "cross_gradient"} else "T",
                "met": False,
                "available": False,
                "reason": f"Requested {parameter}, but no corresponding residual constraint could be constructed.",
            }
        )

    relative_tolerance = _constraint_acceptance_relative_tolerance(config)
    density_limit = engineering_surface_current_density_limit_A_per_m(config)
    if density_limit > 0.0:
        peak, face_count, reason = _contract_surface_current_density(problem, full)
        available = bool(np.isfinite(peak) and face_count > 0)
        effective_limit = density_limit * (1.0 + relative_tolerance)
        met = bool(available and peak <= effective_limit)
        diagnostics.update(
            {
                "surface_current_density_constraint_enabled": True,
                "surface_current_density_constraint_limit_A_per_m": float(density_limit),
                "surface_current_density_constraint_max_A_per_m": float(peak),
                "surface_current_density_constraint_effective_limit_A_per_m": float(effective_limit),
                "surface_current_density_constraint_count": face_count,
                "surface_current_density_constraint_met": met,
                "surface_current_density_constraint_available": available,
                "surface_current_density_constraint_reason": reason,
            }
        )
        rows.append(
            {
                "name": "surface_current_density",
                "value": peak,
                "limit": density_limit,
                "effective_limit": effective_limit,
                "unit": "A/m",
                "met": met,
                "available": available,
                "reason": reason,
            }
        )

    contour = _constrained_contour_step_limit(config)
    diagnostics.update(contour)
    if contour["contour_step_constraint_enabled"]:
        stream_range = float(np.ptp(full)) if full.size and finite_solution else float("nan")
        limit = float(contour["contour_step_constraint_max_range_A"])
        effective_limit = limit * (1.0 + relative_tolerance)
        available = bool(np.isfinite(stream_range))
        met = bool(available and stream_range <= effective_limit)
        diagnostics.update(
            {
                "contour_step_constraint_range_A": stream_range,
                "contour_step_constraint_effective_max_range_A": effective_limit,
                "contour_step_constraint_met": met,
                "contour_step_constraint_actual_step_A": stream_range
                / float(contour["contour_step_constraint_denominator"]),
            }
        )
        rows.append(
            {
                "name": "contour_step",
                "value": stream_range,
                "limit": limit,
                "effective_limit": effective_limit,
                "unit": "A",
                "met": met,
                "available": available,
            }
        )

    shield = _shield_gradient_opposition_constraint_data(problem, config)
    shield_diagnostics = _shield_gradient_opposition_constraint_diagnostics(shield, reduced)
    diagnostics.update(shield_diagnostics)
    if shield["requested"]:
        value = float(shield_diagnostics["shield_gradient_opposition_value_T_per_m"])
        limit = float(shield["limit_T_per_m"])
        available = bool(shield["enabled"] and np.isfinite(value) and np.isfinite(limit))
        tolerance = relative_tolerance * abs(float(shield["scale_T_per_m"]))
        effective_limit = limit + tolerance
        met = bool(available and value <= effective_limit)
        diagnostics["shield_gradient_opposition_constraint_met"] = met
        diagnostics["shield_gradient_opposition_effective_limit_T_per_m"] = effective_limit
        rows.append(
            {
                "name": "shield_gradient_opposition",
                "value": value,
                "limit": limit,
                "effective_limit": effective_limit,
                "unit": "T/m",
                "met": met,
                "available": available,
                "reason": shield["reason"],
            }
        )

    failed = [str(row["name"]) for row in rows if not row["met"]]
    if not finite_solution:
        failed.append("finite_stream_function")
    diagnostics.update(
        {
            "constrained_solver_physical_constraints_met": not failed,
            "constrained_solver_constraint_data_available": all(row["available"] for row in rows),
            "constrained_solver_solution_finite": finite_solution,
            "constrained_solver_failed_constraints": failed,
            "constrained_solver_contract_constraint_diagnostics": rows,
            "constrained_solver_hardware_acceptance_relative_tolerance": relative_tolerance,
        }
    )
    return diagnostics


def _contract_surface_current_density(
    problem: StreamFunctionOptimizationProblem,
    full_stream_function: np.ndarray,
) -> tuple[float, int, str]:
    counts = problem.part_vertex_counts
    records = (problem.part_current_density_mats, problem.part_faces, problem.part_face_basis_currents)
    if not counts or any(len(record) != len(counts) for record in records) or sum(counts) != len(full_stream_function):
        return float("nan"), 0, "Current-density operators do not cover every stream-function vertex."
    peak = 0.0
    face_count = 0
    offset = 0
    for count, dense, faces, local in zip(counts, *records):
        values = full_stream_function[offset : offset + count]
        offset += count
        faces = np.asarray(faces, dtype=np.int64)
        if dense is not None:
            dense = np.asarray(dense, dtype=float)
            if dense.shape != (count, len(faces), 3) or not np.all(np.isfinite(dense)):
                return float("nan"), face_count, "A dense current-density operator is incomplete or nonfinite."
            components = np.einsum("vfa,v->fa", dense, values, optimize=True)
        else:
            local = np.asarray(local, dtype=float)
            if faces.shape != (len(faces), 3) or local.shape != (len(faces), 3, 3):
                return float("nan"), face_count, "A local current-density operator has inconsistent dimensions."
            if np.any(faces < 0) or np.any(faces >= count) or not np.all(np.isfinite(local)):
                return float("nan"), face_count, "A local current-density operator is invalid or nonfinite."
            components = np.einsum("fva,fv->fa", local, values[faces], optimize=True)
        if not len(faces) or not np.all(np.isfinite(components)):
            return float("nan"), face_count, "A coil part has no finite face-current-density values."
        magnitudes = np.linalg.norm(components, axis=1)
        if not np.all(np.isfinite(magnitudes)):
            return float("nan"), face_count, "Surface-current-density evaluation overflowed."
        peak = max(peak, float(np.max(magnitudes)))
        face_count += len(faces)
    return peak, face_count, "Every coil face was checked."


def _diagnostic_float(diagnostics: dict[str, Any], field_name: str, default: float = float("nan")) -> float:
    try:
        value = float(diagnostics.get(field_name, default))
    except (TypeError, ValueError):
        return float(default)
    return float(value) if np.isfinite(value) else float(default)


def _target_error_residual_budget(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> tuple[float, float, float, float]:
    target_norm = float(np.linalg.norm(problem.target_z))
    relative_tolerance_percent = float(getattr(config, "target_error_tolerance_percent", 5.0))
    absolute_tolerance = max(float(getattr(config, "target_error_tolerance_absolute_T", 0.0)), 0.0)
    allowed_residual = target_norm * relative_tolerance_percent / 100.0 + absolute_tolerance
    if not np.isfinite(allowed_residual) or allowed_residual <= 0.0:
        raise ValueError(
            "target_error_tolerance_percent and target_error_tolerance_absolute_T define a zero error budget. "
            "Use a positive tolerance for fixed-error stream-function optimization."
        )
    return target_norm, relative_tolerance_percent, absolute_tolerance, float(allowed_residual)


def _normalise_target_error_constraint_mode(config: CoilGenConfig) -> str:
    raw = str(getattr(config, "target_error_constraint_mode", "separate")).strip().lower().replace("-", "_")
    aliases = {
        "combined": "combined",
        "weighted": "combined",
        "weighted_combined": "combined",
        "legacy": "combined",
        "separate": "separate",
        "split": "separate",
        "dsv_and_suppression": "separate",
        "separate_dsv_suppression": "separate",
    }
    if raw not in aliases:
        raise ValueError(
            "target_error_constraint_mode must be 'separate' or 'combined'. "
            "'separate' constrains DSV accuracy and suppression-field residual independently."
        )
    return aliases[raw]


def _normalise_target_error_constraint_metric(config: CoilGenConfig) -> str:
    key = normalize_field_budget_metric(getattr(config, "target_error_constraint_metric", "norm"))
    if key not in {"norm", "rms", "std"}:
        raise ValueError("target_error_constraint_metric must be 'norm', 'rms', or 'std'.")
    return key


def _unweighted_constraint_system(problem: StreamFunctionOptimizationProblem) -> tuple[np.ndarray, np.ndarray]:
    a_matrix = problem.unweighted_reduced_sensitivity
    target = problem.unweighted_target_z
    if a_matrix is None or target is None:
        return np.asarray(problem.reduced_sensitivity, dtype=float), np.asarray(problem.target_z, dtype=float)
    return np.asarray(a_matrix, dtype=float), np.asarray(target, dtype=float)


def _weighted_constraint_system(
    a_matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    center: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(a_matrix, dtype=float)
    values = np.asarray(target, dtype=float)
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    if len(weight_values) != len(values):
        raise ValueError("DSV constraint weights must match the target rows.")
    valid = np.isfinite(weight_values) & (weight_values > 0.0)
    if not np.all(valid):
        raise ValueError("Active DSV constraint weights must be finite and positive.")
    weight_sum = float(np.sum(weight_values))
    if center:
        matrix = matrix - (weight_values @ matrix / weight_sum)[None, :]
        values = values - float(np.dot(weight_values, values) / weight_sum)
    sqrt_weights = np.sqrt(weight_values)
    return matrix * sqrt_weights[:, None], values * sqrt_weights, weight_sum


def _dsv_constraint_block(
    a_matrix: np.ndarray,
    target: np.ndarray,
    config: CoilGenConfig,
    *,
    weights: np.ndarray | None = None,
) -> ResidualConstraintBlock:
    metric = _normalise_target_error_constraint_metric(config)
    row_count = int(target.shape[0])
    weight_values = np.ones(row_count, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    relative_tolerance_percent = float(getattr(config, "target_error_tolerance_percent", 5.0))
    absolute_tolerance = max(float(getattr(config, "target_error_tolerance_absolute_T", 0.0)), 0.0)

    if metric == "std":
        constraint_matrix, constraint_target, weight_sum = _weighted_constraint_system(
            a_matrix,
            target,
            weight_values,
            center=True,
        )
        root_count = np.sqrt(max(weight_sum, 1.0e-30))
        reference_norm = float(np.linalg.norm(constraint_target))
        explicit_limit = max(float(getattr(config, "target_error_std_limit_T", 0.0)), 0.0)
        if explicit_limit > 0.0:
            allowed = root_count * explicit_limit
            field_limit = explicit_limit
        else:
            allowed = reference_norm * relative_tolerance_percent / 100.0 + root_count * absolute_tolerance
            field_limit = allowed / root_count
        zero_budget_message = (
            "target_error_constraint_metric='std' requires a positive target_error_std_limit_T, "
            "target_error_tolerance_percent, or target_error_tolerance_absolute_T."
        )
    elif metric == "rms":
        constraint_matrix, constraint_target, weight_sum = _weighted_constraint_system(
            a_matrix,
            target,
            weight_values,
            center=False,
        )
        root_count = np.sqrt(max(weight_sum, 1.0e-30))
        reference_norm = float(np.linalg.norm(constraint_target))
        explicit_limit = max(float(getattr(config, "target_error_rms_limit_T", 0.0)), 0.0)
        if explicit_limit > 0.0:
            allowed = root_count * explicit_limit
            field_limit = explicit_limit
        else:
            allowed = reference_norm * relative_tolerance_percent / 100.0 + root_count * absolute_tolerance
            field_limit = allowed / root_count
        zero_budget_message = (
            "target_error_constraint_metric='rms' requires a positive target_error_rms_limit_T, "
            "target_error_tolerance_percent, or target_error_tolerance_absolute_T."
        )
    else:
        constraint_matrix, constraint_target, _weight_sum = _weighted_constraint_system(
            a_matrix,
            target,
            weight_values,
            center=False,
        )
        reference_norm = float(np.linalg.norm(constraint_target))
        allowed = reference_norm * relative_tolerance_percent / 100.0 + absolute_tolerance
        field_limit = None
        zero_budget_message = (
            "target_error_tolerance_percent and target_error_tolerance_absolute_T define a zero DSV error budget. "
            "Use a positive fixed-error tolerance."
        )

    if not np.isfinite(allowed) or allowed <= 0.0:
        raise ValueError(zero_budget_message)

    return ResidualConstraintBlock(
        name="dsv",
        matrix=constraint_matrix,
        target=constraint_target,
        allowed_residual=float(allowed),
        reference_norm=reference_norm,
        relative_tolerance_percent=relative_tolerance_percent,
        absolute_tolerance=absolute_tolerance,
        metric=metric,
        field_limit_T=float(field_limit) if field_limit is not None else None,
        field_metric_denominator=float(root_count) if metric in {"std", "rms"} else 1.0,
        weighting=("uniform" if not row_count or np.all(weight_values == weight_values[0]) else "target_weights"),
        row_weights=weight_values.copy(),
    )


def _target_error_constraint_blocks(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> tuple[ResidualConstraintBlock, ...]:
    mode = _normalise_target_error_constraint_mode(config)
    metric = _normalise_target_error_constraint_metric(config)
    if mode == "combined":
        if metric != "norm":
            raise ValueError(
                "target_error_constraint_metric='std' or 'rms' requires "
                "target_error_constraint_mode='separate' so the DSV metric is not mixed with suppression rows."
            )
        target_norm, relative_tolerance_percent, absolute_tolerance, allowed_residual = _target_error_residual_budget(
            problem,
            config,
        )
        return (
            ResidualConstraintBlock(
                name="combined_weighted",
                matrix=np.asarray(problem.reduced_sensitivity, dtype=float),
                target=np.asarray(problem.target_z, dtype=float),
                allowed_residual=allowed_residual,
                reference_norm=target_norm,
                relative_tolerance_percent=relative_tolerance_percent,
                absolute_tolerance=absolute_tolerance,
                metric="norm",
                weighting=(
                    "target_weights"
                    if problem.target_weights is not None and np.unique(np.asarray(problem.target_weights)).size > 1
                    else "uniform"
                ),
            ),
        )

    a_matrix, target = _unweighted_constraint_system(problem)
    row_count = target.shape[0]
    weights = problem.target_weights
    if weights is None:
        active = np.ones(row_count, dtype=bool)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != row_count:
            raise ValueError("Target weights do not match the target vector length.")
        active = np.isfinite(weights) & (weights > 0.0)
    suppressed = problem.target_suppressed
    if suppressed is None:
        suppressed_mask = np.zeros(row_count, dtype=bool)
    else:
        suppressed_mask = np.asarray(suppressed, dtype=bool).reshape(-1)
        if len(suppressed_mask) != row_count:
            raise ValueError("Suppression mask does not match the target vector length.")

    dsv_mask = active & ~suppressed_mask
    if not np.any(dsv_mask):
        dsv_mask = active
    if not np.any(dsv_mask):
        raise ValueError("No active DSV target rows are available for fixed-error optimization.")

    dsv_target = target[dsv_mask]
    dsv_reference_norm = float(np.linalg.norm(dsv_target))
    dsv_weights = np.ones(len(dsv_target), dtype=float) if weights is None else weights[dsv_mask]
    blocks: list[ResidualConstraintBlock] = [
        _dsv_constraint_block(a_matrix[dsv_mask], dsv_target, config, weights=dsv_weights)
    ]
    blocks.extend(_gradient_linearity_constraint_blocks(problem, a_matrix, target, dsv_mask, config))

    suppression_point_mask = _suppression_point_mask(problem)
    if suppression_point_mask is not None:
        active_point_mask = _active_point_mask(problem)
        suppression_point_mask = suppression_point_mask & active_point_mask
        if np.any(suppression_point_mask):
            blocks.extend(
                _suppression_constraint_blocks(
                    problem,
                    suppression_point_mask,
                    config,
                    dsv_reference_norm=dsv_reference_norm,
                    default_relative_tolerance_percent=blocks[0].relative_tolerance_percent,
                )
            )
    return tuple(blocks)


def _suppression_constraint_blocks(
    problem: StreamFunctionOptimizationProblem,
    point_mask: np.ndarray,
    config: CoilGenConfig,
    *,
    dsv_reference_norm: float,
    default_relative_tolerance_percent: float,
) -> list[ResidualConstraintBlock]:
    matrix, target, component, base_weighting, denominator = _suppression_constraint_system(problem, point_mask, config)
    row_count = int(matrix.shape[0])
    if row_count == 0:
        return []
    field_rms_limit = max(float(getattr(config, "suppression_field_rms_limit_T", 0.0)), 0.0)
    peak_limit = max(float(getattr(config, "suppression_field_peak_limit_T", 0.0)), 0.0)
    relative_tolerance_percent = float(
        getattr(config, "suppression_error_tolerance_percent", default_relative_tolerance_percent)
    )
    absolute_tolerance = max(float(getattr(config, "suppression_error_tolerance_absolute_T", 0.0)), 0.0)
    blocks: list[ResidualConstraintBlock] = []
    if field_rms_limit > 0.0:
        allowed = float(denominator) * field_rms_limit
        blocks.append(
            ResidualConstraintBlock(
                name="suppression",
                matrix=matrix,
                target=target,
                allowed_residual=allowed,
                reference_norm=dsv_reference_norm,
                relative_tolerance_percent=100.0 * allowed / max(dsv_reference_norm, 1.0e-30),
                absolute_tolerance=allowed,
                rms_field_limit_T=field_rms_limit,
                metric="rms",
                field_limit_T=field_rms_limit,
                field_metric_denominator=float(denominator),
                component=component,
                weighting=base_weighting,
            )
        )
    elif peak_limit <= 0.0:
        allowed = dsv_reference_norm * relative_tolerance_percent / 100.0 + absolute_tolerance
        if not np.isfinite(allowed) or allowed <= 0.0:
            raise ValueError(
                "suppression_error_tolerance_percent and suppression_error_tolerance_absolute_T define a zero "
                "suppression error budget. Set a positive suppression_field_rms_limit_T for a physical field limit "
                "or use a positive active-shield suppression tolerance."
            )
        unweighted_matrix, unweighted_target, _, _, unweighted_denominator = _suppression_constraint_system(
            problem,
            point_mask,
            config,
            force_uniform=True,
        )
        blocks.append(
            ResidualConstraintBlock(
                name="suppression",
                matrix=unweighted_matrix,
                target=unweighted_target,
                allowed_residual=float(allowed),
                reference_norm=dsv_reference_norm,
                relative_tolerance_percent=relative_tolerance_percent,
                absolute_tolerance=absolute_tolerance,
                metric="norm",
                field_metric_denominator=float(unweighted_denominator),
                component=component,
                weighting="uniform",
            )
        )
    if peak_limit > 0.0:
        peak_matrix, peak_target, _, _, _ = _suppression_constraint_system(
            problem, point_mask, config, force_uniform=True
        )
        peak_group_size = 3 if component in {"vector", "Bmag"} else 1
        blocks.append(
            ResidualConstraintBlock(
                name="suppression_peak",
                matrix=peak_matrix,
                target=peak_target,
                allowed_residual=peak_limit,
                reference_norm=dsv_reference_norm,
                relative_tolerance_percent=100.0 * peak_limit / max(dsv_reference_norm, 1.0e-30),
                absolute_tolerance=peak_limit,
                metric="linf",
                field_limit_T=peak_limit,
                component=component,
                weighting="uniform",
                group_size=peak_group_size,
            )
        )
    return blocks


def _suppression_constraint_system(
    problem: StreamFunctionOptimizationProblem,
    point_mask: np.ndarray,
    config: CoilGenConfig,
    *,
    force_uniform: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, str, float]:
    if problem.target_coords is None:
        raise ValueError("Suppression constraints require target coordinates.")
    point_mask = np.asarray(point_mask, dtype=bool).reshape(-1)
    if len(point_mask) != len(problem.target_coords):
        raise ValueError("Suppression point mask does not match target coordinates.")
    point_indices = np.flatnonzero(point_mask)
    component = _suppression_component(config)
    full_matrix, full_target = _suppression_component_system(problem, point_indices, component, config)
    if full_matrix.shape[0] == 0:
        return full_matrix, full_target, component, "uniform", 1.0
    point_weights = _suppression_point_weights(problem, point_indices, config, force_uniform=force_uniform)
    if component in {"vector", "Bmag", "components"}:
        row_weights = np.repeat(point_weights, 3)
    else:
        row_weights = point_weights
    if len(row_weights) != full_matrix.shape[0]:
        raise ValueError("Suppression row weights do not match suppression constraint rows.")
    sqrt_weights = np.sqrt(row_weights)
    matrix = full_matrix * sqrt_weights[:, None]
    target = full_target * sqrt_weights
    denominator = float(np.sqrt(np.sum(point_weights)))
    weighting = "uniform" if force_uniform else _normalise_suppression_surface_weighting(config)
    return matrix, target, component, weighting, max(denominator, 1.0e-300)


def _suppression_component_system(
    problem: StreamFunctionOptimizationProblem,
    point_indices: np.ndarray,
    component: str,
    config: CoilGenConfig,
) -> tuple[np.ndarray, np.ndarray]:
    sensitivity = np.asarray(problem.sensitivity_matrix, dtype=float)[:, point_indices, :]
    target_b = _suppression_target_b(problem, point_indices)
    if component == "Bnormal":
        normals = _suppression_normals(problem, point_indices, config)
        full_matrix = (
            sensitivity[0] * normals[:, 0, None]
            + sensitivity[1] * normals[:, 1, None]
            + sensitivity[2] * normals[:, 2, None]
        )
        full_target = np.einsum("ij,ij->i", target_b, normals)
    elif component in {"vector", "Bmag", "components"}:
        full_matrix = np.moveaxis(sensitivity, 0, 1).reshape(3 * len(point_indices), sensitivity.shape[2])
        full_target = target_b.reshape(-1)
    else:
        b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
        full_matrix = sensitivity_component_matrix(sensitivity, component, b0_direction=b0_direction)
        full_target = field_component_values(target_b, component, b0_direction=b0_direction)
    return problem.reducer.reduce_columns(full_matrix), np.asarray(full_target, dtype=float).reshape(-1)


def _suppression_target_b(problem: StreamFunctionOptimizationProblem, point_indices: np.ndarray) -> np.ndarray:
    if problem.target_b is None:
        return np.zeros((len(point_indices), 3), dtype=float)
    target_b = np.asarray(problem.target_b, dtype=float)
    if target_b.ndim != 2 or target_b.shape[1] != 3:
        return np.zeros((len(point_indices), 3), dtype=float)
    return target_b[point_indices]


def _suppression_component(config: CoilGenConfig) -> str:
    raw = (
        str(getattr(config, "suppression_field_component", "target"))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if raw in {"", "target", "same", "useful", "target_component"}:
        raw = str(getattr(config, "target_field_component", "Bz")).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "x": "Bx",
        "bx": "Bx",
        "b_x": "Bx",
        "y": "By",
        "by": "By",
        "b_y": "By",
        "z": "Bz",
        "bz": "Bz",
        "b_z": "Bz",
        "b0": "BdotB0",
        "bdotb0": "BdotB0",
        "b_dot_b0": "BdotB0",
        "b_parallel": "BdotB0",
        "parallel_b0": "BdotB0",
        "normal": "Bnormal",
        "bnormal": "Bnormal",
        "b_normal": "Bnormal",
        "normal_field": "Bnormal",
        "vector": "vector",
        "full_vector": "vector",
        "bvector": "vector",
        "components": "components",
        "component": "components",
        "componentwise": "components",
        "component_wise": "components",
        "cartesian_components": "components",
        "xyz": "components",
        "bx_by_bz": "components",
        "bmag": "Bmag",
        "b_mag": "Bmag",
        "magnitude": "Bmag",
    }
    if raw not in aliases:
        raise ValueError(
            "suppression_field_component must be target, Bx, By, Bz, BdotB0, Bnormal, vector, components, or Bmag."
        )
    return aliases[raw]


def _suppression_point_mask(problem: StreamFunctionOptimizationProblem) -> np.ndarray | None:
    if problem.target_coords is None or problem.target_suppressed is None:
        return None
    point_count = len(problem.target_coords)
    suppressed = np.asarray(problem.target_suppressed, dtype=bool).reshape(-1)
    if len(suppressed) == point_count:
        return suppressed
    if len(suppressed) == 3 * point_count:
        return np.any(suppressed.reshape(point_count, 3), axis=1)
    return None


def _active_point_mask(problem: StreamFunctionOptimizationProblem) -> np.ndarray:
    if problem.target_coords is None:
        return np.zeros(0, dtype=bool)
    point_count = len(problem.target_coords)
    weights = problem.target_weights
    if weights is None:
        return np.ones(point_count, dtype=bool)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) == point_count:
        return np.isfinite(values) & (values > 0.0)
    if len(values) == 3 * point_count:
        reshaped = values.reshape(point_count, 3)
        return np.any(np.isfinite(reshaped) & (reshaped > 0.0), axis=1)
    return np.ones(point_count, dtype=bool)


def _suppression_point_weights(
    problem: StreamFunctionOptimizationProblem,
    point_indices: np.ndarray,
    config: CoilGenConfig,
    *,
    force_uniform: bool,
) -> np.ndarray:
    if force_uniform or _normalise_suppression_surface_weighting(config) == "uniform":
        return np.ones(len(point_indices), dtype=float)
    area_weights = problem.target_suppression_area_weights
    if area_weights is None:
        return np.ones(len(point_indices), dtype=float)
    values = np.asarray(area_weights, dtype=float).reshape(-1)
    if problem.target_coords is None or len(values) != len(problem.target_coords):
        return np.ones(len(point_indices), dtype=float)
    selected = values[point_indices]
    finite_positive = selected[np.isfinite(selected) & (selected > 0.0)]
    if finite_positive.size == 0:
        return np.ones(len(point_indices), dtype=float)
    fallback = float(np.median(finite_positive))
    selected = np.where(np.isfinite(selected) & (selected > 0.0), selected, fallback)
    return selected


def _normalise_suppression_surface_weighting(config: CoilGenConfig) -> str:
    raw = (
        str(getattr(config, "suppression_surface_weighting", "uniform"))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if raw in {"", "none", "uniform", "point", "points"}:
        return "uniform"
    if raw in {"area", "mesh_area", "surface_area", "vertex_area"}:
        return "area"
    raise ValueError("suppression_surface_weighting must be 'uniform' or 'area'.")


def _suppression_normals(
    problem: StreamFunctionOptimizationProblem,
    point_indices: np.ndarray,
    config: CoilGenConfig,
) -> np.ndarray:
    source = (
        str(getattr(config, "suppression_normal_source", "mesh")).strip().lower().replace("-", "_").replace(" ", "_")
    )
    coords = np.asarray(problem.target_coords, dtype=float)[point_indices]
    normals = None
    if source in {"mesh", "target", "metadata"} and problem.target_suppression_normals is not None:
        candidate = np.asarray(problem.target_suppression_normals, dtype=float)
        if candidate.shape == np.asarray(problem.target_coords).shape:
            normals = candidate[point_indices]
    if normals is None:
        model = "radial" if source in {"radial", "cylindrical"} else "box"
        normals = _estimate_suppression_normals(coords, model=model)
    return _normalise_normal_rows(normals, coords)


def _estimate_suppression_normals(coords: np.ndarray, *, model: str) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=float)
    center = np.nanmean(coords, axis=0)
    centered = coords - center[None, :]
    if model in {"radial", "cylindrical"}:
        normals = centered
    else:
        half_span = np.nanmax(np.abs(centered), axis=0)
        half_span[~np.isfinite(half_span) | (half_span <= 0.0)] = 1.0
        axis = np.nanargmax(np.abs(centered) / half_span[None, :], axis=1)
        normals = np.zeros_like(coords)
        signs = np.sign(centered[np.arange(coords.shape[0]), axis])
        signs[signs == 0.0] = 1.0
        normals[np.arange(coords.shape[0]), axis] = signs
    return normals


def _normalise_normal_rows(normals: np.ndarray, coords: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=float)
    lengths = np.linalg.norm(values, axis=1)
    safe = np.isfinite(lengths) & (lengths > 1.0e-15)
    out = np.zeros_like(values)
    out[safe] = values[safe] / lengths[safe, None]
    if np.any(~safe):
        fallback = _estimate_suppression_normals(coords[~safe], model="box")
        fallback_lengths = np.linalg.norm(fallback, axis=1)
        fallback_safe = np.isfinite(fallback_lengths) & (fallback_lengths > 1.0e-15)
        out[np.flatnonzero(~safe)[fallback_safe]] = fallback[fallback_safe] / fallback_lengths[fallback_safe, None]
    return out


def _shield_gradient_opposition_constraint_data(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> dict[str, Any]:
    requested = bool(getattr(config, "shield_gradient_opposition_enabled", False))
    disabled = {
        "requested": requested,
        "enabled": False,
        "reason": "disabled" if not requested else "unavailable",
        "matrix": None,
        "limit_T_per_m": float("nan"),
        "scale_T_per_m": float("nan"),
        "shield_part_indices": (),
        "desired_gradient_T_per_m": float("nan"),
        "max_fraction_of_target": float(getattr(config, "shield_gradient_max_fraction_of_target", 0.0)),
    }
    if not requested:
        return disabled
    if problem.target_coords is None or problem.unweighted_target_z is None:
        return {**disabled, "reason": "target data unavailable"}
    target = np.asarray(problem.unweighted_target_z, dtype=float).reshape(-1)
    coords = np.asarray(problem.target_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] != target.shape[0]:
        return {**disabled, "reason": "target coordinate rows do not match scalar target rows"}

    dsv_mask = _active_dsv_point_mask(problem, target.shape[0])
    sample_mask = _linearity_sample_mask(problem, target, dsv_mask, config)
    if np.count_nonzero(sample_mask) < 4:
        return {**disabled, "reason": "not enough DSV points for shield-gradient fit"}

    shield_part_indices = _outer_half_part_indices(problem)
    if not shield_part_indices:
        return {**disabled, "reason": "no distinct outer shield parts identified"}
    shield_columns = _part_column_indices(problem.part_vertex_counts, shield_part_indices)
    if shield_columns.size == 0:
        return {**disabled, "reason": "shield parts have no stream-function vertices"}

    fit_offset = bool(getattr(config, "linearity_fit_offset", True))
    basis = _linearity_affine_basis(coords[sample_mask], fit_offset=fit_offset)
    if basis.shape[0] <= basis.shape[1]:
        return {**disabled, "reason": "not enough DSV points for affine gradient fit"}
    pseudo_inverse = np.linalg.pinv(basis)
    target_coeff = _linearity_coefficients_from_pinv(pseudo_inverse, target[sample_mask], fit_offset=fit_offset)
    desired_gradient = np.asarray(target_coeff[1:4], dtype=float)
    desired_norm = float(np.linalg.norm(desired_gradient))
    requested_gradient = float(getattr(config, "target_gradient_strength", desired_norm))
    reference_gradient = (
        abs(requested_gradient) if np.isfinite(requested_gradient) and abs(requested_gradient) > 0.0 else desired_norm
    )
    if not np.isfinite(desired_norm) or desired_norm <= 1.0e-30 or reference_gradient <= 0.0:
        return {**disabled, "reason": "desired target gradient is zero or unavailable"}

    b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
    component = getattr(config, "target_field_component", "Bz")
    full_component = sensitivity_component_matrix(
        np.asarray(problem.sensitivity_matrix, dtype=float),
        component,
        b0_direction=b0_direction,
    )
    expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
    shield_operator = full_component[:, shield_columns] @ expansion[shield_columns, :]
    shield_selected = shield_operator[sample_mask]
    shield_coeff = _linearity_coefficients_from_pinv(pseudo_inverse, shield_selected, fit_offset=fit_offset)
    shield_gradient_matrix = np.asarray(shield_coeff[1:4], dtype=float)
    desired_unit = desired_gradient / desired_norm
    row = np.asarray(desired_unit.reshape(1, 3) @ shield_gradient_matrix, dtype=float).reshape(-1)
    if row.shape[0] != expansion.shape[1] or not np.all(np.isfinite(row)):
        return {**disabled, "reason": "shield-gradient constraint row is invalid"}

    max_fraction = float(getattr(config, "shield_gradient_max_fraction_of_target", 0.0))
    if not np.isfinite(max_fraction):
        return {**disabled, "reason": "shield-gradient max fraction is not finite"}
    return {
        "requested": requested,
        "enabled": True,
        "reason": "outer shield contribution constrained along desired gradient axis",
        "matrix": row,
        "limit_T_per_m": float(max_fraction * reference_gradient),
        "scale_T_per_m": float(reference_gradient),
        "shield_part_indices": tuple(int(index) for index in shield_part_indices),
        "desired_gradient_T_per_m": float(reference_gradient),
        "max_fraction_of_target": float(max_fraction),
    }


def _active_dsv_point_mask(problem: StreamFunctionOptimizationProblem, row_count: int) -> np.ndarray:
    if problem.target_weights is None:
        active = np.ones(row_count, dtype=bool)
    else:
        weights = np.asarray(problem.target_weights, dtype=float).reshape(-1)
        active = np.isfinite(weights) & (weights > 0.0) if len(weights) == row_count else np.ones(row_count, dtype=bool)
    if problem.target_suppressed is None:
        suppressed = np.zeros(row_count, dtype=bool)
    else:
        values = np.asarray(problem.target_suppressed, dtype=bool).reshape(-1)
        suppressed = values if len(values) == row_count else np.zeros(row_count, dtype=bool)
    dsv_mask = active & ~suppressed
    return dsv_mask if np.any(dsv_mask) else active


def _linearity_sample_mask(
    problem: StreamFunctionOptimizationProblem,
    target: np.ndarray,
    dsv_mask: np.ndarray,
    config: CoilGenConfig,
) -> np.ndarray:
    target_field = _linearity_target_field_view(problem, target)
    try:
        sample_mask = np.asarray(gradient_linearity_region_mask(target_field, config), dtype=bool).reshape(-1)
    except ValueError:
        sample_mask = np.asarray(dsv_mask, dtype=bool).reshape(-1)
    region_key = str(getattr(config, "linearity_region", "dsv")).strip().lower().replace("-", "_")
    if region_key in {"", "dsv", "fov", "roi", "target", "primary", "internal"}:
        sample_mask &= np.asarray(dsv_mask, dtype=bool).reshape(-1)
    sample_mask &= np.all(np.isfinite(np.asarray(problem.target_coords, dtype=float)), axis=1)
    return sample_mask


def _outer_half_part_indices(problem: StreamFunctionOptimizationProblem) -> tuple[int, ...]:
    counts = tuple(int(count) for count in problem.part_vertex_counts)
    if len(counts) < 4 or len(counts) % 2:
        return ()
    scores = tuple(float(value) for value in getattr(problem, "part_outward_scores", ()) or ())
    if len(scores) != len(counts) or not all(np.isfinite(score) for score in scores):
        return ()
    ranked = sorted(range(len(scores)), key=lambda index: scores[index])
    midpoint = len(ranked) // 2
    inner_max = max(scores[index] for index in ranked[:midpoint])
    outer_min = min(scores[index] for index in ranked[midpoint:])
    scale = max(max(abs(score) for score in scores), 1.0)
    if outer_min <= inner_max + 1.0e-6 * scale:
        return ()
    return tuple(sorted(ranked[midpoint:]))


def _part_column_indices(part_vertex_counts: tuple[int, ...], part_indices: tuple[int, ...]) -> np.ndarray:
    offsets = np.concatenate(([0], np.cumsum(np.asarray(part_vertex_counts, dtype=np.int64))))
    columns: list[np.ndarray] = []
    for index in part_indices:
        if 0 <= int(index) < len(part_vertex_counts):
            columns.append(np.arange(offsets[index], offsets[index + 1], dtype=np.int64))
    return np.concatenate(columns) if columns else np.empty(0, dtype=np.int64)


def _shield_gradient_opposition_constraint_diagnostics(
    constraint: dict[str, Any],
    reduced_stream_function: np.ndarray,
) -> dict[str, Any]:
    enabled = bool(constraint.get("enabled", False))
    diagnostics: dict[str, Any] = {
        "shield_gradient_opposition_requested": bool(constraint.get("requested", False)),
        "shield_gradient_opposition_enabled": enabled,
        "shield_gradient_opposition_reason": str(constraint.get("reason", "")),
        "shield_gradient_opposition_limit_T_per_m": float(constraint.get("limit_T_per_m", np.nan)),
        "shield_gradient_opposition_target_T_per_m": float(constraint.get("desired_gradient_T_per_m", np.nan)),
        "shield_gradient_opposition_max_fraction_of_target": float(constraint.get("max_fraction_of_target", np.nan)),
        "shield_gradient_opposition_part_indices": ",".join(
            str(int(index) + 1) for index in constraint.get("shield_part_indices", ())
        ),
        "shield_gradient_opposition_value_T_per_m": float("nan"),
        "shield_gradient_opposition_margin_T_per_m": float("nan"),
        "shield_gradient_opposition_constraint_met": not bool(constraint.get("requested", False)),
    }
    if not enabled:
        return diagnostics
    row = np.asarray(constraint.get("matrix"), dtype=float).reshape(-1)
    value = float(row @ np.asarray(reduced_stream_function, dtype=float).reshape(-1))
    limit = float(constraint.get("limit_T_per_m", np.nan))
    scale = max(abs(float(constraint.get("scale_T_per_m", 0.0))), 1.0e-30)
    tolerance = max(1.0e-12, 1.0e-9 * scale)
    diagnostics.update(
        {
            "shield_gradient_opposition_value_T_per_m": value,
            "shield_gradient_opposition_margin_T_per_m": float(limit - value),
            "shield_gradient_opposition_constraint_met": bool(value <= limit + tolerance),
        }
    )
    return diagnostics


def _gradient_linearity_constraint_blocks(
    problem: StreamFunctionOptimizationProblem,
    a_matrix: np.ndarray,
    target: np.ndarray,
    dsv_mask: np.ndarray,
    config: CoilGenConfig,
) -> list[ResidualConstraintBlock]:
    if not gradient_linearity_enabled(config):
        return []
    if problem.target_coords is None:
        return []
    target_field = _linearity_target_field_view(problem, target)
    try:
        region_mask = gradient_linearity_region_mask(target_field, config)
        sample_mask = np.asarray(region_mask, dtype=bool).reshape(-1)
    except ValueError:
        sample_mask = np.asarray(dsv_mask, dtype=bool).reshape(-1)
    region_key = str(getattr(config, "linearity_region", "dsv")).strip().lower().replace("-", "_")
    if region_key in {"", "dsv", "fov", "roi", "target", "primary", "internal"}:
        sample_mask &= np.asarray(dsv_mask, dtype=bool).reshape(-1)
    sample_mask &= np.all(np.isfinite(np.asarray(problem.target_coords, dtype=float)), axis=1)
    if np.count_nonzero(sample_mask) < 4:
        return []

    coords = np.asarray(problem.target_coords, dtype=float)[sample_mask]
    a_selected = np.asarray(a_matrix, dtype=float)[sample_mask]
    target_selected = np.asarray(target, dtype=float)[sample_mask]
    fit_offset = bool(getattr(config, "linearity_fit_offset", True))
    basis = _linearity_affine_basis(coords, fit_offset=fit_offset)
    if basis.shape[0] <= basis.shape[1]:
        return []
    pseudo_inverse = np.linalg.pinv(basis)
    residual_projection = np.eye(basis.shape[0]) - basis @ pseudo_inverse
    reference_scale = gradient_linearity_reference_scale_T(target_field, config, sample_mask)
    if not np.isfinite(reference_scale) or reference_scale <= 0.0:
        return []

    target_coeff = _linearity_coefficients_from_pinv(pseudo_inverse, target_selected, fit_offset=fit_offset)
    desired_gradient = np.asarray(target_coeff[1:4], dtype=float)
    desired_norm = float(np.linalg.norm(desired_gradient))
    requested_gradient = float(getattr(config, "target_gradient_strength", desired_norm))
    reference_gradient = (
        abs(requested_gradient) if np.isfinite(requested_gradient) and abs(requested_gradient) > 0.0 else desired_norm
    )
    blocks: list[ResidualConstraintBlock] = []

    rms_tolerance = _positive_config_float(config, "linearity_rms_tolerance_percent")
    peak_tolerance = _positive_config_float(config, "linearity_peak_tolerance_percent")
    cross_tolerance = _positive_config_float(config, "linearity_cross_gradient_tolerance_percent")
    gradient_tolerance = _positive_config_float(config, "gradient_strength_tolerance_percent")
    row_count = int(np.count_nonzero(sample_mask))
    if rms_tolerance > 0.0:
        field_limit = reference_scale * rms_tolerance / 100.0
        blocks.append(
            ResidualConstraintBlock(
                name="linearity_rms",
                matrix=residual_projection @ a_selected,
                target=np.zeros(row_count, dtype=float),
                allowed_residual=float(np.sqrt(float(row_count)) * field_limit),
                reference_norm=float(np.sqrt(float(row_count)) * reference_scale),
                relative_tolerance_percent=float(rms_tolerance),
                absolute_tolerance=0.0,
                metric="rms",
                field_limit_T=float(field_limit),
            )
        )
    if peak_tolerance > 0.0:
        field_limit = reference_scale * peak_tolerance / 100.0
        blocks.append(
            ResidualConstraintBlock(
                name="linearity_peak",
                matrix=residual_projection @ a_selected,
                target=np.zeros(row_count, dtype=float),
                allowed_residual=float(field_limit),
                reference_norm=float(reference_scale),
                relative_tolerance_percent=float(peak_tolerance),
                absolute_tolerance=0.0,
                metric="linf",
                field_limit_T=float(field_limit),
            )
        )
    if desired_norm > 0.0 and reference_gradient > 0.0:
        coeff_matrix = _linearity_coefficients_from_pinv(pseudo_inverse, a_selected, fit_offset=fit_offset)
        gradient_matrix = coeff_matrix[1:4]
        desired_unit = desired_gradient / desired_norm
        if gradient_tolerance > 0.0:
            field_limit = reference_gradient * gradient_tolerance / 100.0
            blocks.append(
                ResidualConstraintBlock(
                    name="gradient_strength",
                    matrix=(desired_unit.reshape(1, 3) @ gradient_matrix),
                    target=np.asarray([reference_gradient], dtype=float),
                    allowed_residual=float(field_limit),
                    reference_norm=float(reference_gradient),
                    relative_tolerance_percent=float(gradient_tolerance),
                    absolute_tolerance=0.0,
                    metric="linf",
                    field_limit_T=float(field_limit),
                )
            )
        if cross_tolerance > 0.0:
            field_limit = reference_gradient * cross_tolerance / 100.0
            orthogonal = np.eye(3) - np.outer(desired_unit, desired_unit)
            blocks.append(
                ResidualConstraintBlock(
                    name="cross_gradient",
                    matrix=orthogonal @ gradient_matrix,
                    target=np.zeros(3, dtype=float),
                    allowed_residual=float(field_limit),
                    reference_norm=float(reference_gradient),
                    relative_tolerance_percent=float(cross_tolerance),
                    absolute_tolerance=0.0,
                    metric="norm",
                    field_limit_T=float(field_limit),
                )
            )
    return blocks


def _linearity_target_field_view(problem: StreamFunctionOptimizationProblem, target: np.ndarray) -> Any:
    class _TargetView:
        pass

    view = _TargetView()
    view.coords = np.asarray(problem.target_coords, dtype=float)
    values = np.asarray(target, dtype=float).reshape(-1)
    b = np.zeros((len(values), 3), dtype=float)
    b[:, 2] = values
    view.b = b
    if problem.target_suppressed is None:
        view.suppressed = np.zeros(len(values), dtype=bool)
    else:
        view.suppressed = np.asarray(problem.target_suppressed, dtype=bool).reshape(-1)
    return view


def _linearity_affine_basis(coords: np.ndarray, *, fit_offset: bool) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if fit_offset:
        return np.column_stack([np.ones(coords.shape[0]), coords])
    return coords


def _linearity_coefficients_from_pinv(
    pseudo_inverse: np.ndarray, values: np.ndarray, *, fit_offset: bool
) -> np.ndarray:
    coeff = np.asarray(pseudo_inverse @ np.asarray(values, dtype=float), dtype=float)
    if fit_offset:
        return coeff.reshape(4, -1) if coeff.ndim == 2 else coeff.reshape(4)
    if coeff.ndim == 2:
        return np.vstack([np.zeros((1, coeff.shape[1]), dtype=float), coeff])
    return np.concatenate(([0.0], coeff.reshape(3)))


def _residual_constraint_diagnostics(
    problem: StreamFunctionOptimizationProblem,
    reduced_stream_function: np.ndarray,
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    config: CoilGenConfig,
) -> dict[str, Any]:
    tolerance = _constraint_acceptance_relative_tolerance(config)
    absolute_acceptance = _constraint_acceptance_absolute_tolerance(config)
    diagnostics: dict[str, Any] = {
        "target_error_constraint_mode": _normalise_target_error_constraint_mode(config),
        "target_error_constraint_metric": _normalise_target_error_constraint_metric(config),
        "target_error_constraint_block_count": int(len(constraint_blocks)),
        "target_error_constraint_acceptance_relative_tolerance": float(tolerance),
        "target_error_constraint_acceptance_absolute_T": float(absolute_acceptance),
        "suppression_constraint_enabled": any(str(block.name).startswith("suppression") for block in constraint_blocks),
    }
    all_met = True
    primary_block = constraint_blocks[0]
    primary_residual = float("nan")
    primary_relative_percent = float("nan")
    primary_metric_value = float("nan")
    primary_allowed_metric_value = float("nan")
    for block in constraint_blocks:
        residual_values = np.asarray(block.matrix @ reduced_stream_function - block.target, dtype=float).reshape(-1)
        norm_residual = float(np.linalg.norm(residual_values))
        residual = _constraint_residual_measure(block, reduced_stream_function)
        field_denominator = _constraint_field_metric_denominator(block)
        rms_residual = norm_residual / field_denominator
        linf_residual = _constraint_linf_residual(block, residual_values)
        # STD blocks already contain sqrt(weight)-scaled, weighted-centered
        # physical residuals. Their L2 norm / sqrt(sum(weights)) is the physical
        # population STD; np.std of those transformed rows changes its units
        # with an arbitrary rescaling of the weights.
        std_residual = (
            norm_residual / field_denominator
            if block.metric == "std"
            else float(np.std(residual_values))
            if len(residual_values)
            else float("nan")
        )
        mean_residual = float(np.mean(residual_values)) if len(residual_values) else float("nan")
        statistics_scope = "constraint_rows"
        if block.row_weights is not None and len(residual_values):
            weights = np.asarray(block.row_weights, dtype=float).reshape(-1)
            # Undo only the reporting transform. In a STD block these values
            # remain weighted-centered; the unconstrained DC error is absent.
            physical_values = residual_values / np.sqrt(weights)
            weight_sum = float(np.sum(weights))
            mean_residual = float(np.dot(weights, physical_values) / weight_sum)
            std_residual = float(np.sqrt(np.dot(weights, (physical_values - mean_residual) ** 2) / weight_sum))
            rms_residual = float(norm_residual / np.sqrt(weight_sum))
            linf_residual = float(np.max(np.abs(physical_values)))
            statistics_scope = (
                "weighted_centered_physical_field" if block.metric == "std" else "weighted_physical_field"
            )
        reference = max(float(block.reference_norm), 1.0e-30)
        relative_percent = 100.0 * residual / reference
        # Gradient residuals are T/m; a field-valued T allowance cannot be added
        # to them. Keep relative numerical acceptance for these fitted gradients.
        residual_unit = "T/m" if block.name in {"gradient_strength", "cross_gradient"} else "T"
        block_absolute_acceptance = absolute_acceptance if residual_unit == "T" else 0.0
        allowed_with_acceptance = float(block.allowed_residual) * (1.0 + tolerance) + block_absolute_acceptance
        met = residual <= allowed_with_acceptance
        all_met = all_met and bool(met)
        prefix = block.name
        metric = str(block.metric)
        field_metric = (
            std_residual
            if metric == "std"
            else rms_residual
            if metric == "rms"
            else linf_residual
            if metric == "linf"
            else residual
        )
        allowed_field_metric = (
            float(block.field_limit_T)
            if block.field_limit_T is not None
            else float(block.allowed_residual) / field_denominator
            if metric in {"std", "rms"}
            else float(block.allowed_residual)
        )
        diagnostics.update(
            {
                f"{prefix}_constraint_met": bool(met),
                f"{prefix}_constraint_residual_unit": residual_unit,
                f"{prefix}_constraint_metric": metric,
                f"{prefix}_constraint_component": str(block.component),
                f"{prefix}_constraint_weighting": str(block.weighting),
                f"{prefix}_constraint_residual_statistics_scope": statistics_scope,
                f"{prefix}_constraint_residual_norm_T": residual,
                f"{prefix}_constraint_l2_residual_norm_T": norm_residual,
                f"{prefix}_constraint_allowed_residual_norm_T": float(block.allowed_residual),
                f"{prefix}_constraint_allowed_with_acceptance_residual_norm_T": float(allowed_with_acceptance),
                f"{prefix}_constraint_reference_norm_T": float(block.reference_norm),
                f"{prefix}_constraint_relative_residual_percent": float(relative_percent),
                f"{prefix}_constraint_rms_residual_T": float(rms_residual),
                f"{prefix}_constraint_linf_residual_T": float(linf_residual),
                f"{prefix}_constraint_std_T": float(std_residual),
                f"{prefix}_constraint_mean_T": float(mean_residual),
                f"{prefix}_constraint_field_metric_T": float(field_metric),
                f"{prefix}_constraint_allowed_field_metric_T": float(allowed_field_metric),
                f"{prefix}_constraint_allowed_with_acceptance_field_metric_T": (
                    float(allowed_with_acceptance) / field_denominator
                    if metric in {"std", "rms"}
                    else float(allowed_with_acceptance)
                ),
                f"{prefix}_constraint_tolerance_percent": float(block.relative_tolerance_percent),
                f"{prefix}_constraint_absolute_tolerance_T": float(block.absolute_tolerance),
            }
        )
        if metric == "std":
            diagnostics[f"{prefix}_constraint_allowed_std_T"] = float(allowed_field_metric)
        if metric == "rms":
            diagnostics[f"{prefix}_constraint_allowed_rms_T"] = float(allowed_field_metric)
        if metric == "linf":
            diagnostics[f"{prefix}_constraint_allowed_linf_T"] = float(allowed_field_metric)
        if block.rms_field_limit_T is not None:
            diagnostics[f"{prefix}_constraint_rms_field_limit_T"] = float(block.rms_field_limit_T)
        if block.name == "suppression":
            diagnostics["suppression_field_rms_T"] = float(rms_residual)
            diagnostics["suppression_field_rms_limit_T"] = (
                float(block.rms_field_limit_T) if block.rms_field_limit_T is not None else float("nan")
            )
            diagnostics["suppression_field_component"] = str(block.component)
            diagnostics["suppression_surface_weighting"] = str(block.weighting)
        if block.name == "suppression_peak":
            diagnostics["suppression_field_peak_T"] = float(field_metric)
            diagnostics["suppression_field_peak_limit_T"] = float(allowed_field_metric)
            diagnostics.setdefault("suppression_field_component", str(block.component))
            diagnostics.setdefault("suppression_surface_weighting", str(block.weighting))
        if block.name == "dsv":
            primary_block = block
            primary_residual = residual
            primary_relative_percent = relative_percent
            primary_metric_value = float(field_metric)
            primary_allowed_metric_value = float(allowed_field_metric)
    if not np.isfinite(primary_residual):
        primary_values = np.asarray(primary_block.matrix @ reduced_stream_function - primary_block.target, dtype=float)
        primary_residual = _constraint_residual_measure(primary_block, reduced_stream_function)
        primary_relative_percent = 100.0 * primary_residual / max(float(primary_block.reference_norm), 1.0e-30)
        primary_metric = str(primary_block.metric)
        if primary_metric in {"std", "rms"}:
            primary_metric_value = float(np.linalg.norm(primary_values)) / _constraint_field_metric_denominator(
                primary_block
            )
        elif primary_metric == "linf":
            primary_metric_value = _constraint_linf_residual(primary_block, primary_values)
        else:
            primary_metric_value = primary_residual
        primary_allowed_metric_value = (
            float(primary_block.field_limit_T)
            if primary_block.field_limit_T is not None
            else float(primary_block.allowed_residual) / _constraint_field_metric_denominator(primary_block)
            if primary_metric in {"std", "rms"}
            else float(primary_block.allowed_residual)
        )

    diagnostics.update(
        {
            "target_error_constraint_met": bool(all_met),
            "target_error_constraint_metric": str(primary_block.metric),
            "target_error_tolerance_percent": float(primary_block.relative_tolerance_percent),
            "target_error_tolerance_absolute_T": float(primary_block.absolute_tolerance),
            "target_error_allowed_residual_norm_T": float(primary_block.allowed_residual),
            "target_error_constraint_field_metric_T": float(primary_metric_value),
            "target_error_allowed_field_metric_T": float(primary_allowed_metric_value),
            "target_error_target_norm_T": float(primary_block.reference_norm),
            "target_error_constraint_relative_residual_percent": float(primary_relative_percent),
            "target_error_constraint_residual_norm_T": float(primary_residual),
            "combined_weighted_residual_norm_T": problem.residual_norm(reduced_stream_function),
        }
    )
    block_diagnostics = _constraint_block_diagnostic_rows(constraint_blocks, diagnostics)
    suppression_rows = [row for row in block_diagnostics if str(row["name"]).startswith("suppression")]
    failed_blocks = [row["name"] for row in block_diagnostics if not bool(row["met"])]
    finite_ratios = [
        float(row["violation_ratio"]) for row in block_diagnostics if np.isfinite(float(row["violation_ratio"]))
    ]
    diagnostics.update(
        {
            "target_error_constraint_block_diagnostics": block_diagnostics,
            "target_error_failed_constraint_blocks": failed_blocks,
            "suppression_constraint_enabled": bool(suppression_rows),
            "suppression_constraint_met": bool(suppression_rows) and all(bool(row["met"]) for row in suppression_rows),
            "target_error_constraint_max_violation_ratio": (
                float(max(finite_ratios)) if finite_ratios else float("nan")
            ),
        }
    )
    return diagnostics


def _constraint_acceptance_relative_tolerance(config: CoilGenConfig) -> float:
    explicit = getattr(config, "target_error_constraint_acceptance_tolerance", 0.0)
    try:
        explicit_value = float(explicit)
    except (TypeError, ValueError):
        explicit_value = 0.0
    if np.isfinite(explicit_value) and explicit_value > 0.0:
        return float(explicit_value)

    try:
        solver_tolerance = float(getattr(config, "constrained_optimization_tolerance", 1.0e-7))
    except (TypeError, ValueError):
        solver_tolerance = 1.0e-7
    if not np.isfinite(solver_tolerance) or solver_tolerance <= 0.0:
        solver_tolerance = 1.0e-7
    return float(max(1.0e-8, 10.0 * solver_tolerance))


def _constraint_acceptance_absolute_tolerance(config: CoilGenConfig) -> float:
    explicit = getattr(config, "target_error_constraint_acceptance_absolute_T", 1.0e-7)
    try:
        explicit_value = float(explicit)
    except (TypeError, ValueError):
        explicit_value = 1.0e-7
    if np.isfinite(explicit_value) and explicit_value >= 0.0:
        return float(explicit_value)
    return 1.0e-7


def _constraint_block_diagnostic_rows(
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in constraint_blocks:
        prefix = block.name
        residual = _diagnostic_float(diagnostics, f"{prefix}_constraint_residual_norm_T")
        allowed = _diagnostic_float(diagnostics, f"{prefix}_constraint_allowed_residual_norm_T")
        field_metric = _diagnostic_float(diagnostics, f"{prefix}_constraint_field_metric_T")
        allowed_field_metric = _diagnostic_float(diagnostics, f"{prefix}_constraint_allowed_field_metric_T")
        ratio = residual / allowed if np.isfinite(residual) and np.isfinite(allowed) and allowed > 0.0 else float("nan")
        rows.append(
            {
                "name": prefix,
                "metric": str(diagnostics.get(f"{prefix}_constraint_metric", block.metric)),
                "component": str(diagnostics.get(f"{prefix}_constraint_component", block.component)),
                "weighting": str(diagnostics.get(f"{prefix}_constraint_weighting", block.weighting)),
                "met": bool(diagnostics.get(f"{prefix}_constraint_met", False)),
                "residual_norm_T": float(residual),
                "allowed_residual_norm_T": float(allowed),
                "field_metric_T": float(field_metric),
                "allowed_field_metric_T": float(allowed_field_metric),
                "violation_ratio": float(ratio),
            }
        )
    return rows


def _constraint_field_metric_denominator(block: ResidualConstraintBlock) -> float:
    if block.field_metric_denominator is not None:
        try:
            value = float(block.field_metric_denominator)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0.0:
            return value
    return float(np.sqrt(max(int(block.matrix.shape[0]), 1)))


def _constraint_residual_measure(block: ResidualConstraintBlock, reduced_stream_function: np.ndarray) -> float:
    residual_values = np.asarray(block.matrix @ reduced_stream_function - block.target, dtype=float).reshape(-1)
    if str(block.metric).lower() == "linf":
        value = _constraint_linf_residual(block, residual_values)
        return float(value) if np.isfinite(value) else float("inf")
    return float(np.linalg.norm(residual_values))


def _constraint_linf_residual(block: ResidualConstraintBlock, residual_values: np.ndarray) -> float:
    values = np.asarray(residual_values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    group_size = max(int(getattr(block, "group_size", 1) or 1), 1)
    if group_size <= 1:
        return float(np.max(np.abs(values)))
    if values.size % group_size != 0:
        raise ValueError(
            f"Constraint block {block.name!r} has {values.size} residual rows, "
            f"which is not divisible by group_size={group_size}."
        )
    grouped = values.reshape((-1, group_size))
    return float(np.max(np.linalg.norm(grouped, axis=1)))


def _positive_config_float(config: CoilGenConfig, name: str) -> float:
    try:
        value = float(getattr(config, name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return float(value) if np.isfinite(value) and value > 0.0 else 0.0


def _constrained_contour_step_limit(config: CoilGenConfig) -> dict[str, float | int | bool | str]:
    configured_step = _nonnegative_config_value(config, "regularization_auto_range_max_contour_step_A", 0.0)
    engineering_step = engineering_max_contour_step_A(config)
    step_candidates: list[tuple[str, float]] = []
    if configured_step > 0.0 and np.isfinite(configured_step):
        step_candidates.append(("regularization_auto_range", float(configured_step)))
    if engineering_step > 0.0 and np.isfinite(engineering_step):
        step_candidates.append(("max_wire_current", float(engineering_step)))

    if step_candidates:
        min_step = min(step for _source, step in step_candidates)
        tied_sources = [
            source for source, step in step_candidates if abs(step - min_step) <= 1.0e-12 * max(abs(min_step), 1.0)
        ]
        source = "_and_".join(tied_sources)
        max_step = float(min_step)
    else:
        source = "disabled"
        max_step = 0.0

    level_count = _contour_step_constraint_level_count(
        config,
        prefer_engineering_max_levels="max_wire_current" in source,
    )
    denominator = _contour_step_denominator(level_count, config)
    enabled = bool(max_step > 0.0 and np.isfinite(max_step) and denominator > 0.0)
    max_range = float(max_step * denominator) if enabled else float("nan")
    return {
        "contour_step_constraint_enabled": enabled,
        "contour_step_constraint_max_step_A": float(max_step),
        "contour_step_constraint_regularization_max_step_A": float(configured_step),
        "contour_step_constraint_engineering_max_step_A": float(engineering_step),
        "contour_step_constraint_source": source,
        "contour_step_constraint_level_count": int(level_count),
        "contour_step_constraint_denominator": float(denominator),
        "contour_step_constraint_max_range_A": max_range,
    }


def _contour_step_constraint_level_count(
    config: CoilGenConfig,
    *,
    prefer_engineering_max_levels: bool = False,
) -> int:
    if prefer_engineering_max_levels and engineering_contouring_enabled(config):
        max_levels = engineering_contour_max_levels(config)
        if max_levels > 0:
            return int(max_levels)
    try:
        level_count = max(int(getattr(config, "levels", 1)), 1)
    except (TypeError, ValueError):
        level_count = 1
    return int(level_count)


def _contour_step_denominator(level_count: int, config: CoilGenConfig) -> float:
    grid_mode = str(getattr(config, "contour_level_grid_mode", "centered_margin")).strip().lower().replace("-", "_")
    if grid_mode in {"fixed_step_phase", "fixed_phase", "quantizer_phase"}:
        return max(float(max(int(level_count), 1)), np.finfo(float).eps)
    try:
        level_offset = float(getattr(config, "pot_offset_factor", 0.5))
    except (TypeError, ValueError):
        level_offset = 0.5
    return max(float(max(int(level_count), 1) - 1) + 2.0 * level_offset, np.finfo(float).eps)


def _nonnegative_config_value(config: CoilGenConfig, name: str, default: float) -> float:
    try:
        value = float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value) or value < 0.0:
        return float(default)
    return value
