from __future__ import annotations

from .optimization_cvxpy import (
    _cvxpy_conic_solver_order as _cvxpy_conic_solver_order,
    _cvxpy_matrix_scale as _cvxpy_matrix_scale,
    _cvxpy_positive_scale as _cvxpy_positive_scale,
    _cvxpy_requested_thread_count as _cvxpy_requested_thread_count,
    _cvxpy_residual_constraint as _cvxpy_residual_constraint,
    _cvxpy_solver_options as _cvxpy_solver_options,
    _cvxpy_solver_thread_option as _cvxpy_solver_thread_option,
    _cvxpy_surface_current_density_constraints as _cvxpy_surface_current_density_constraints,
    _reduced_surface_current_density_operators as _reduced_surface_current_density_operators,
    _solve_constrained_quadratic_cvxpy as _solve_constrained_quadratic_cvxpy,
)

from .optimization_constraints import (
    ResidualConstraintBlock as ResidualConstraintBlock,
    _active_dsv_point_mask as _active_dsv_point_mask,
    _active_point_mask as _active_point_mask,
    _constrained_contour_step_limit as _constrained_contour_step_limit,
    _constrained_solution_contract_diagnostics as _constrained_solution_contract_diagnostics,
    _constraint_acceptance_absolute_tolerance as _constraint_acceptance_absolute_tolerance,
    _constraint_acceptance_relative_tolerance as _constraint_acceptance_relative_tolerance,
    _constraint_block_diagnostic_rows as _constraint_block_diagnostic_rows,
    _constraint_field_metric_denominator as _constraint_field_metric_denominator,
    _constraint_linf_residual as _constraint_linf_residual,
    _constraint_residual_measure as _constraint_residual_measure,
    _contour_step_constraint_level_count as _contour_step_constraint_level_count,
    _contour_step_denominator as _contour_step_denominator,
    _contract_surface_current_density as _contract_surface_current_density,
    _diagnostic_float as _diagnostic_float,
    _dsv_constraint_block as _dsv_constraint_block,
    _estimate_suppression_normals as _estimate_suppression_normals,
    _gradient_linearity_constraint_blocks as _gradient_linearity_constraint_blocks,
    _linearity_affine_basis as _linearity_affine_basis,
    _linearity_coefficients_from_pinv as _linearity_coefficients_from_pinv,
    _linearity_sample_mask as _linearity_sample_mask,
    _linearity_target_field_view as _linearity_target_field_view,
    _nonnegative_config_value as _nonnegative_config_value,
    _normalise_normal_rows as _normalise_normal_rows,
    _normalise_suppression_surface_weighting as _normalise_suppression_surface_weighting,
    _normalise_target_error_constraint_metric as _normalise_target_error_constraint_metric,
    _normalise_target_error_constraint_mode as _normalise_target_error_constraint_mode,
    _outer_half_part_indices as _outer_half_part_indices,
    _part_column_indices as _part_column_indices,
    _positive_config_float as _positive_config_float,
    _residual_constraint_diagnostics as _residual_constraint_diagnostics,
    _shield_gradient_opposition_constraint_data as _shield_gradient_opposition_constraint_data,
    _shield_gradient_opposition_constraint_diagnostics as _shield_gradient_opposition_constraint_diagnostics,
    _suppression_component as _suppression_component,
    _suppression_component_system as _suppression_component_system,
    _suppression_constraint_blocks as _suppression_constraint_blocks,
    _suppression_constraint_system as _suppression_constraint_system,
    _suppression_normals as _suppression_normals,
    _suppression_point_mask as _suppression_point_mask,
    _suppression_point_weights as _suppression_point_weights,
    _suppression_target_b as _suppression_target_b,
    _target_error_constraint_blocks as _target_error_constraint_blocks,
    _target_error_residual_budget as _target_error_residual_budget,
    _unweighted_constraint_system as _unweighted_constraint_system,
    _weighted_constraint_system as _weighted_constraint_system,
)

from .optimization_problem import (
    BoundaryReducer as BoundaryReducer,
    GaugeReducer as GaugeReducer,
    SpectralReducer as SpectralReducer,
    StreamFunctionOptimizationProblem as StreamFunctionOptimizationProblem,
    SymmetryReducer as SymmetryReducer,
    _block_diag as _block_diag,
    _component_mean_gauge_constraints as _component_mean_gauge_constraints,
    _configured_mirror_plane as _configured_mirror_plane,
    _constraint_nullspace_parameterization as _constraint_nullspace_parameterization,
    _face_local_basis_currents as _face_local_basis_currents,
    _full_to_boundary_reduced_column as _full_to_boundary_reduced_column,
    _generated_candidate_symmetric_part_pairs as _generated_candidate_symmetric_part_pairs,
    _geometry_scale as _geometry_scale,
    _geometry_tolerance as _geometry_tolerance,
    _median_mesh_edge_length as _median_mesh_edge_length,
    _mesh_face_areas as _mesh_face_areas,
    _mesh_vertex_component_labels as _mesh_vertex_component_labels,
    _mirror_vertex_mapping_for_pair as _mirror_vertex_mapping_for_pair,
    _mirror_vertex_mapping_for_single_mesh as _mirror_vertex_mapping_for_single_mesh,
    _nearest_indices as _nearest_indices,
    _normalise_constrained_quadratic_objective_mode as _normalise_constrained_quadratic_objective_mode,
    _normalise_stream_function_basis as _normalise_stream_function_basis,
    _normalise_stream_function_method as _normalise_stream_function_method,
    _normalise_symmetry_reduction_mode as _normalise_symmetry_reduction_mode,
    _normalise_symmetry_reduction_source as _normalise_symmetry_reduction_source,
    _part_outward_score as _part_outward_score,
    _part_vertex_offsets as _part_vertex_offsets,
    _quadratic_norm as _quadratic_norm,
    _reduced_surface_current_density_metric as _reduced_surface_current_density_metric,
    _reflect_points as _reflect_points,
    _safe_positive_float as _safe_positive_float,
    _select_configured_target_parity as _select_configured_target_parity,
    _select_single_mesh_stream_sign as _select_single_mesh_stream_sign,
    _select_stream_function_reducer as _select_stream_function_reducer,
    _select_symmetry_pair_stream_sign as _select_symmetry_pair_stream_sign,
    _sensitivity_parity_residual as _sensitivity_parity_residual,
    _signed_constraint_expansion as _signed_constraint_expansion,
    _solution_max_surface_current_density as _solution_max_surface_current_density,
    _spectral_stream_function_basis_requested as _spectral_stream_function_basis_requested,
    _stream_function_basis_diagnostics as _stream_function_basis_diagnostics,
    _stream_function_smoothness_requested as _stream_function_smoothness_requested,
    _surface_biharmonic_metric as _surface_biharmonic_metric,
    _symmetric_quadratic_matrix as _symmetric_quadratic_matrix,
    _target_parity_residual as _target_parity_residual,
    _target_parity_tolerance as _target_parity_tolerance,
    _target_symmetry_mapping as _target_symmetry_mapping,
    _target_tolerance as _target_tolerance,
    _turn_spacing_metric_requested as _turn_spacing_metric_requested,
    _turn_spacing_penalty_requested as _turn_spacing_penalty_requested,
    _turn_spacing_regularization_factor as _turn_spacing_regularization_factor,
    _turn_spacing_threshold as _turn_spacing_threshold,
)

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .basis import current_density_from_basis
from .config import CoilGenConfig
from .engineering_limits import (
    engineering_contouring_enabled,
    engineering_limit_diagnostics,
    engineering_max_contour_step_A,
    engineering_surface_current_density_limit_A_per_m,
)
from .models import CoilPart, Mesh, TargetField
from .progress import DesignCancelled, ProgressReporter, progress_scope


STREAM_FUNCTION_COST_FORMULAS = {
    "field_error_plus_resistance": (
        "min_x ||A x - b||_2^2 + lambda^2 x^T R x"
        " + eta x^T C M^-1 C x when stream_function_smoothness_factor > 0"
        " + mu x^T S x when min_turn_distance/min_wire_distance_m > 0"
    ),
    "minimum_resistance_subject_to_field_error": (
        "Search the Tikhonov family for the lowest x^T R x meeting the configured "
        "residual budgets. This restricted scalar search is not a general "
        "multiple-constraint optimum."
    ),
    "minimum_resistance_subject_to_field_error_socp": (
        "min_x x^T Q x, Q = weighted R + configured quadratic penalties, subject to "
        "DSV_error(x) <= fixed_limit_dsv using the configured DSV metric and, "
        "when suppression targets are present, "
        "suppression_field_rms(x) <= fixed_limit_sup, optionally "
        "stream_function_range(x) <= max_wire_current_A * contour_denominator, and optionally "
        "||K_face(x)||_2 <= surface_current_density_limit_A_per_m, and optionally "
        "g_shield(x) dot g_target_unit <= configured_fraction * |g_target|. "
        "Solved directly as convex second-order cone constraints when CVXPY is available"
    ),
    "minimum_peak_current_density_subject_to_field_error_socp": (
        "min_{x,t} t subject to ||K_face(x)||_2 <= t on every face, "
        "the configured field-error budgets, and all requested engineering bounds. "
        "t is the peak surface current density in A/m."
    ),
}


@dataclass
class StreamFunctionSolution:
    reduced_stream_function: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AdaptiveRegularizationRange:
    factors: np.ndarray
    singular_value_count: int
    retained_singular_value_count: int
    singular_value_max: float
    singular_value_min_retained: float
    singular_value_floor: float
    target_projection_floor: float
    factor_min: float
    factor_max: float
    regularization_rank: int = 0
    regularization_nullity: int = 0
    regularization_eigenvalue_floor: float = 0.0
    regularization_nullspace_relative_field_norm: float = 0.0


@dataclass(frozen=True)
class RegularizationWhitenedSystem:
    transformed_sensitivity: np.ndarray
    regularization_rank: int
    regularization_nullity: int
    eigenvalue_floor: float
    nullspace_relative_field_norm: float


def stream_function_optimization(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
) -> tuple[list[CoilPart], Mesh, np.ndarray, dict[str, Any]]:
    problem = StreamFunctionOptimizationProblem.from_parts(parts, target_field, config)
    solution = solve_stream_function_problem(problem, config)
    opt_stream_func = problem.expand(solution.reduced_stream_function)

    problem.combined_mesh.stream_function = opt_stream_func
    b_field_opt_sf = problem.evaluate_full_b_field(opt_stream_func)

    offset = 0
    for part in parts:
        count = part.mesh.n_vertices
        part.stream_function = opt_stream_func[offset : offset + count]
        if part.current_density_mat is not None:
            part.current_density = np.tensordot(part.stream_function, part.current_density_mat, axes=(0, 0))
        else:
            part.current_density = current_density_from_basis(part)
        offset += count
    if solution.diagnostics.get("research_physics_enabled", False):
        # Physical constraints certify this amplitude; post-solve rescaling
        # would invalidate arbitrary field equalities and scenarios.
        density_limit_diagnostics = {
            "surface_current_density_scaling_applied": False,
            "surface_current_density_scaling_factor": 1.0,
        }
    else:
        parts, problem.combined_mesh, b_field_opt_sf, density_limit_diagnostics = (
            enforce_stream_function_current_density_limit(
                parts,
                problem.combined_mesh,
                b_field_opt_sf,
                config,
            )
        )
    solution.diagnostics.update(density_limit_diagnostics)
    if solution.diagnostics.get("stream_function_scheme") == "constrained_quadratic" and bool(
        density_limit_diagnostics.get("surface_current_density_scaling_applied", False)
    ):
        solution.reduced_stream_function *= float(density_limit_diagnostics["surface_current_density_scaling_factor"])
        if solution.diagnostics.get("minimax_current_density_enabled", False):
            peak, _face_count, _reason = _contract_surface_current_density(
                problem, problem.expand(solution.reduced_stream_function)
            )
            solution.diagnostics["minimax_current_density_pre_scale_peak_A_per_m"] = solution.diagnostics.get(
                "minimax_current_density_peak_A_per_m", float("nan")
            )
            solution.diagnostics["minimax_current_density_peak_A_per_m"] = peak
            solution.diagnostics["minimax_current_density_epigraph_describes_pre_scale_solution"] = True
            solution.diagnostics["constrained_solver_objective_value"] = peak
        solution.diagnostics.update(
            _constrained_solution_contract_diagnostics(problem, solution.reduced_stream_function, config)
        )
        solution.diagnostics["constrained_solver_contract_met"] = bool(
            solution.diagnostics.get("constrained_solver_hard_constraints_enforced", False)
            and solution.diagnostics["constrained_solver_physical_constraints_met"]
        )
        if not solution.diagnostics["constrained_solver_contract_met"]:
            reason = _constrained_contract_failure_summary(solution.diagnostics)
            if _normalise_constrained_quadratic_fallback_policy(config) == "strict":
                raise RuntimeError(
                    f"Post-solve current-density scaling violates the constrained-solver contract. {reason}"
                )
            solution.diagnostics["constrained_solver_diagnostic_result"] = True
            solution.diagnostics["constrained_solver_diagnostic_reason"] = reason
    solution.diagnostics.update(stream_function_current_density_diagnostics(parts, config))
    return parts, problem.combined_mesh, b_field_opt_sf, solution.diagnostics


def enforce_stream_function_current_density_limit(
    parts: list[CoilPart],
    combined_mesh: Mesh,
    b_field_opt_sf: np.ndarray,
    config: CoilGenConfig,
) -> tuple[list[CoilPart], Mesh, np.ndarray, dict[str, Any]]:
    limit = engineering_surface_current_density_limit_A_per_m(config)
    diagnostics: dict[str, Any] = {
        "surface_current_density_scaling_applied": False,
        "surface_current_density_scaling_factor": 1.0,
        "surface_current_density_pre_scale_max_A_per_m": float("nan"),
        "gradient_hardware_infeasible_policy": _gradient_hardware_infeasible_policy(config),
        "gradient_hardware_feasible": True,
        "gradient_hardware_scale_factor": 1.0,
        "gradient_hardware_effective_target_gradient_strength": float(getattr(config, "target_gradient_strength", 1.0)),
    }
    if limit <= 0.0:
        return parts, combined_mesh, b_field_opt_sf, diagnostics
    max_density = _parts_max_surface_current_density(parts)
    diagnostics["surface_current_density_pre_scale_max_A_per_m"] = float(max_density)
    if not np.isfinite(max_density) or max_density <= limit * (1.0 + 1.0e-8):
        return parts, combined_mesh, b_field_opt_sf, diagnostics

    scale = float(limit / max(max_density, 1.0e-300))
    policy = _gradient_hardware_infeasible_policy(config)
    diagnostics.update(
        {
            "gradient_hardware_feasible": False,
            "gradient_hardware_scale_factor": scale,
            "gradient_hardware_effective_target_gradient_strength": float(
                getattr(config, "target_gradient_strength", 1.0)
            )
            * scale,
        }
    )
    if policy == "raise":
        raise ValueError(
            "Requested gradient exceeds the configured surface-current-density hardware limit. "
            f"Maximum solved surface current density is {max_density:.6g} A/m, "
            f"effective limit is {limit:.6g} A/m, feasible scale is {scale:.6g}."
        )
    if policy == "report_only":
        return parts, combined_mesh, b_field_opt_sf, diagnostics

    for part in parts:
        if part.stream_function is not None:
            part.stream_function = np.asarray(part.stream_function, dtype=float) * scale
        if part.current_density is not None:
            part.current_density = np.asarray(part.current_density, dtype=float) * scale
        elif part.current_density_mat is not None and part.stream_function is not None:
            part.current_density = np.tensordot(part.stream_function, part.current_density_mat, axes=(0, 0))
        else:
            part.current_density = current_density_from_basis(part)
    if combined_mesh.stream_function is not None:
        combined_mesh.stream_function = np.asarray(combined_mesh.stream_function, dtype=float) * scale
    diagnostics.update(
        {
            "surface_current_density_scaling_applied": True,
            "surface_current_density_scaling_factor": scale,
        }
    )
    return parts, combined_mesh, np.asarray(b_field_opt_sf, dtype=float) * scale, diagnostics


def _gradient_hardware_infeasible_policy(config: CoilGenConfig) -> str:
    raw = (
        str(getattr(config, "gradient_hardware_infeasible_policy", "scale_to_limit")).strip().lower().replace("-", "_")
    )
    if raw in {"raise", "error"}:
        return "raise"
    if raw in {"report", "report_only", "diagnose"}:
        return "report_only"
    return "scale_to_limit"


def _parts_max_surface_current_density(parts: list[CoilPart]) -> float:
    values: list[float] = []
    for part in parts:
        density = part.current_density if part.current_density is not None else current_density_from_basis(part)
        if density is None:
            continue
        norm = np.linalg.norm(np.asarray(density, dtype=float), axis=1)
        finite = norm[np.isfinite(norm)]
        if finite.size:
            values.append(float(np.max(finite)))
    return max(values) if values else float("nan")


def stream_function_current_density_diagnostics(
    parts: list[CoilPart],
    config: CoilGenConfig,
) -> dict[str, Any]:
    """Report solved surface-current density against engineering limits."""

    density_norms: list[np.ndarray] = []
    for part in parts:
        if part.current_density is not None:
            density = np.asarray(part.current_density, dtype=float)
        else:
            density = current_density_from_basis(part)
        if density is None:
            continue
        norm = np.linalg.norm(density, axis=1)
        finite = norm[np.isfinite(norm)]
        if finite.size:
            density_norms.append(finite)
    if density_norms:
        values = np.concatenate(density_norms)
        max_density = float(np.max(values))
        p95_density = float(np.percentile(values, 95.0))
        p99_density = float(np.percentile(values, 99.0))
        rms_density = float(np.sqrt(np.mean(values * values)))
    else:
        max_density = p95_density = p99_density = rms_density = float("nan")

    limit = engineering_surface_current_density_limit_A_per_m(config)
    limit_enabled = bool(limit > 0.0)
    met = bool(not limit_enabled or (np.isfinite(max_density) and max_density <= limit * (1.0 + 1.0e-8)))
    diagnostics: dict[str, Any] = {
        **engineering_limit_diagnostics(config),
        "surface_current_density_limit_enabled": limit_enabled,
        "surface_current_density_limit_met": met,
        "surface_current_density_limit_A_per_m_effective": float(limit),
        "surface_current_density_max_A_per_m": float(max_density),
        "surface_current_density_p99_A_per_m": float(p99_density),
        "surface_current_density_p95_A_per_m": float(p95_density),
        "surface_current_density_rms_A_per_m": float(rms_density),
    }
    if limit_enabled and np.isfinite(max_density):
        diagnostics["surface_current_density_limit_ratio"] = float(max_density / max(limit, 1.0e-30))
    else:
        diagnostics["surface_current_density_limit_ratio"] = float("nan")
    return diagnostics


def solve_stream_function_problem(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> StreamFunctionSolution:
    """Dispatch the selected stream-function optimization scheme.

    Add a new scheme by implementing a function with signature
    ``new_solver(problem, config) -> StreamFunctionSolution`` and adding one
    branch here. Keep the mathematical objective in that solver's docstring.
    """

    if config.research_physics is not None:
        from .research.convex import solve_research_problem

        return solve_research_problem(problem, config)
    scheme = _normalise_stream_function_method(config.sf_opt_method)
    if scheme == "tikhonov":
        solution = solve_tikhonov_stream_function(problem, config)
    elif scheme == "target_error_constrained_efficiency":
        solution = solve_target_error_constrained_efficiency_stream_function(problem, config)
    elif scheme == "constrained_quadratic":
        solution = solve_constrained_quadratic_stream_function(problem, config)
    else:
        raise NotImplementedError(
            f"Stream-function optimization scheme '{config.sf_opt_method}' is not implemented. "
            "Add a branch in solve_stream_function_problem()."
        )
    solution.diagnostics.update(_suppression_target_diagnostics(problem, solution.reduced_stream_function, config))
    return solution


def solve_target_error_constrained_efficiency_stream_function(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> StreamFunctionSolution:
    """Search the Tikhonov family for low sheet power within field-error budgets.

    A scalar multiplier parameterizes the constrained minimum-power solution
    for one matching squared-L2 residual constraint without additional objective
    penalties. Multiple independent bounds, mean-free metrics, and fixed extra
    penalties generally need multiple multipliers or a different normal system.
    This search is a restricted family in those cases; use the direct conic
    method to solve the full constrained problem.
    """

    solution = _select_target_error_constrained_efficiency(problem, config)
    solution.diagnostics.update(
        {
            "stream_function_scheme": "target_error_constrained_efficiency",
            "stream_function_cost_function": "minimum_resistance_subject_to_field_error",
            "cost_function_formula": STREAM_FUNCTION_COST_FORMULAS["minimum_resistance_subject_to_field_error"],
            "unknown_count": int(problem.reduced_sensitivity.shape[1]),
            "target_sample_count": int(problem.reduced_sensitivity.shape[0]),
        }
    )
    if problem.symmetry_diagnostics is not None:
        solution.diagnostics.update(problem.symmetry_diagnostics)
    return solution


def solve_constrained_quadratic_stream_function(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> StreamFunctionSolution:
    """Solve a fixed-field-error stream-function problem by direct constraints.

    Mathematical problem:
        min_x x^T Q x, or min_x max_f ||K_f x||_2 in minimax-current-density mode
        subject to DSV_error_metric(A_dsv x - b_dsv) <= configured budget
                   ||A_sup x||_2 / sqrt(N_sup) <= B_sup,rms,max
        when suppression rows are present and target_error_constraint_mode is
        "separate" and suppression_field_rms_limit_T > 0. If no physical
        suppression limit is configured, the legacy suppression residual budget
        is used for compatibility. The legacy weighted-combined residual is
        available with target_error_constraint_mode="combined".

    Q contains the requested quadratic penalties; K_f maps reduced stream
    function to the vector surface current density on face f. CVXPY solves the
    convex problem. Every returned bound is independently checked in physical
    units. A residual-only problem may fall back to a Tikhonov-multiplier scan,
    reported without a full constrained-solver contract. Requested density,
    current-step, shield-sign, or minimax requirements need a direct solver;
    only explicit diagnostic mode permits their unsupported scan fallback.
    """

    fallback_policy = _normalise_constrained_quadratic_fallback_policy(config)
    try:
        solution = _select_constrained_quadratic(problem, config)
    except DesignCancelled:
        raise
    except Exception as exc:
        if fallback_policy == "strict":
            raise RuntimeError(
                f"Direct constrained-quadratic solve failed and constrained_quadratic_fallback_policy='strict'. {exc}"
            ) from exc
        solution = _constrained_scan_fallback(problem, config, reason=str(exc), backend="tikhonov_scan_fallback")
    else:
        solution.diagnostics.update(
            _constrained_solution_contract_diagnostics(problem, solution.reduced_stream_function, config)
        )
        direct_met = bool(solution.diagnostics["constrained_solver_physical_constraints_met"])
        solution.diagnostics["constrained_solver_hard_constraints_enforced"] = bool(
            solution.diagnostics["constrained_solver_constraint_data_available"]
        )
        solution.diagnostics["constrained_solver_contract_met"] = direct_met
        if not direct_met and fallback_policy == "strict":
            direct_diagnostics = dict(solution.diagnostics)
            failure_summary = _constrained_contract_failure_summary(direct_diagnostics)
            raise RuntimeError(
                "Direct constrained-quadratic solve returned a solution that does not satisfy "
                "all physical constraints and constrained_quadratic_fallback_policy='strict'. "
                f"{failure_summary} "
                f"Minimum residual slack: {direct_diagnostics.get('constrained_solver_residual_slack_T', float('nan'))}."
            )
        if not direct_met and fallback_policy == "diagnostic":
            direct_diagnostics = dict(solution.diagnostics)
            failure_summary = _constrained_contract_failure_summary(direct_diagnostics)
            if bool(getattr(config, "progress_flag", False)):
                print(
                    "Constrained quadratic diagnostic: keeping direct CVXPY stream function "
                    f"for reporting despite unmet constraints. {failure_summary}"
                )
            solution.diagnostics.update(
                {
                    "constrained_solver_diagnostic_result": True,
                    "constrained_solver_diagnostic_reason": failure_summary,
                    "constrained_solver_contract_met": False,
                }
            )
        elif (
            not direct_met
            and _target_error_auto_retry_enabled(config)
            and _target_error_auto_retry_max_reruns(config) > 0
        ):
            direct_diagnostics = dict(solution.diagnostics)
            solution = _constrained_scan_fallback(
                problem,
                config,
                reason="Direct solve failed its physical post-check. "
                + _constrained_contract_failure_summary(direct_diagnostics),
                backend="cvxpy_tikhonov_retry_fallback",
            )
            solution.diagnostics.update(
                {
                    "constrained_solver_initial_status": direct_diagnostics.get("constrained_solver_status", ""),
                    "constrained_solver_initial_residual_met": bool(
                        direct_diagnostics.get("target_error_constraint_met", False)
                    ),
                    "constrained_solver_initial_field_metric_T": direct_diagnostics.get(
                        "target_error_constraint_field_metric_T",
                        float("nan"),
                    ),
                    "constrained_solver_initial_allowed_field_metric_T": direct_diagnostics.get(
                        "target_error_allowed_field_metric_T",
                        float("nan"),
                    ),
                    "constrained_solver_hard_constraints_enforced": False,
                    "constrained_solver_contract_met": False,
                }
            )
        else:
            if not direct_met:
                solution.diagnostics["constrained_solver_diagnostic_result"] = True
                solution.diagnostics["constrained_solver_diagnostic_reason"] = _constrained_contract_failure_summary(
                    solution.diagnostics
                )

    cost_function = (
        "minimum_peak_current_density_subject_to_field_error_socp"
        if _normalise_constrained_quadratic_objective_mode(config) == "minimax_current_density"
        else "minimum_resistance_subject_to_field_error_socp"
    )
    solution.diagnostics.update(
        {
            "stream_function_scheme": "constrained_quadratic",
            "stream_function_cost_function": cost_function,
            "cost_function_formula": STREAM_FUNCTION_COST_FORMULAS[cost_function],
            "unknown_count": int(problem.reduced_sensitivity.shape[1]),
            "target_sample_count": int(problem.reduced_sensitivity.shape[0]),
            "constrained_quadratic_fallback_policy": fallback_policy,
            "constrained_solver_fallback_allowed": fallback_policy != "strict",
        }
    )
    if problem.symmetry_diagnostics is not None:
        solution.diagnostics.update(problem.symmetry_diagnostics)
    return solution


def _constrained_scan_fallback(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
    *,
    reason: str,
    backend: str,
) -> StreamFunctionSolution:
    unsupported: list[str] = []
    if _normalise_constrained_quadratic_objective_mode(config) == "minimax_current_density":
        unsupported.append("minimax_current_density_objective")
    if engineering_surface_current_density_limit_A_per_m(config) > 0.0:
        unsupported.append("surface_current_density")
    if _constrained_contour_step_limit(config)["contour_step_constraint_enabled"]:
        unsupported.append("contour_step")
    if bool(getattr(config, "shield_gradient_opposition_enabled", False)):
        unsupported.append("shield_gradient_opposition")
    if unsupported and _normalise_constrained_quadratic_fallback_policy(config) != "diagnostic":
        raise RuntimeError(
            "The Tikhonov scan fallback cannot enforce the requested hard constraints: "
            + ", ".join(unsupported)
            + ". Use a supported direct conic solver or explicit diagnostic mode. Direct solver: "
            + reason
        )
    solution = _select_target_error_constrained_efficiency(problem, config)
    solution.diagnostics.update(
        _constrained_solution_contract_diagnostics(problem, solution.reduced_stream_function, config)
    )
    solution.diagnostics.update(
        {
            "constrained_solver_backend": backend,
            "constrained_solver_status": "fallback",
            "constrained_solver_fallback_reason": reason,
            "constrained_solver_fallback_unsupported_constraints": unsupported,
            "constrained_solver_hard_constraints_enforced": False,
            "constrained_solver_contract_met": False,
            "constrained_solver_diagnostic_result": True,
            "constrained_solver_diagnostic_reason": (
                "The fallback is a restricted regularization scan; it does not certify the full constrained optimum."
                + (" Unsupported hard constraints: " + ", ".join(unsupported) + "." if unsupported else "")
            ),
        }
    )
    return solution


def _constrained_contract_failure_summary(diagnostics: dict[str, Any]) -> str:
    failed = [
        row for row in diagnostics.get("constrained_solver_contract_constraint_diagnostics", ()) if not row["met"]
    ]
    details = [
        f"{row['name']}: value={row['value']:.6g} {row['unit']}, allowed={row['effective_limit']:.6g} {row['unit']}"
        for row in failed
    ]
    if not diagnostics.get("constrained_solver_solution_finite", True):
        details.append("stream function contains nonfinite values")
    return "Failed physical constraint(s): " + "; ".join(details) + "."


def minimum_residual_optimality_certificate(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
    *,
    solve_engineering_constrained_floor: bool = True,
    stable_rank_relative_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Return rigorous continuous-current residual floors for one reduced basis.

    The projection floors solve the linear least-squares problem exactly in the
    selected symmetry/spectral space. The optional constrained floor minimizes
    the same DSV metric with the configured contour-range, surface-current-
    density, suppression, and shield-gradient constraints. It is a convex SOCP,
    so an optimal conic-solver status is a global certificate for this discrete
    continuous-current model.
    """

    blocks = _target_error_constraint_blocks(problem, config)
    dsv = blocks[0]
    matrix = np.asarray(dsv.matrix, dtype=float)
    target = np.asarray(dsv.target, dtype=float).reshape(-1)
    denominator = float(dsv.field_metric_denominator or 1.0)
    algebraic = _linear_projection_floor(matrix, target, denominator, rcond=None)
    stable_tolerance = float(stable_rank_relative_tolerance)
    if not np.isfinite(stable_tolerance) or stable_tolerance <= 0.0:
        raise ValueError("stable_rank_relative_tolerance must be finite and positive.")
    stable = _linear_projection_floor(matrix, target, denominator, rcond=stable_tolerance)
    report: dict[str, Any] = {
        "metric": str(dsv.metric),
        "field_metric_denominator": denominator,
        "target_row_count": int(matrix.shape[0]),
        "unknown_count": int(matrix.shape[1]),
        "algebraic_projection": algebraic,
        "stable_projection": stable,
        "stable_rank_relative_tolerance": stable_tolerance,
        "engineering_constrained": {
            "requested": bool(solve_engineering_constrained_floor),
            "available": False,
        },
    }
    if not solve_engineering_constrained_floor:
        return report
    try:
        report["engineering_constrained"] = _minimum_residual_engineering_floor_cvxpy(
            problem,
            config,
            dsv,
            blocks[1:],
        )
    except Exception as exc:
        report["engineering_constrained"] = {
            "requested": True,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return report


def _linear_projection_floor(
    matrix: np.ndarray,
    target: np.ndarray,
    denominator: float,
    *,
    rcond: float | None,
) -> dict[str, Any]:
    solution, _residuals, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=rcond)
    residual_norm = float(np.linalg.norm(matrix @ solution - target))
    singular_values = np.asarray(singular_values, dtype=float)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    smallest = float(singular_values[int(rank) - 1]) if int(rank) > 0 else 0.0
    return {
        "field_metric_T": residual_norm / max(float(denominator), 1.0e-300),
        "residual_norm_T": residual_norm,
        "rank": int(rank),
        "nullity": int(matrix.shape[1] - int(rank)),
        "singular_value_max": largest,
        "singular_value_min_retained": smallest,
        "retained_condition_number": largest / smallest if smallest > 0.0 else float("inf"),
        "solution_norm": float(np.linalg.norm(solution)),
        "rcond": None if rcond is None else float(rcond),
    }


def _minimum_residual_engineering_floor_cvxpy(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
    dsv: ResidualConstraintBlock,
    auxiliary_blocks: Sequence[ResidualConstraintBlock],
) -> dict[str, Any]:
    import cvxpy as cp  # type: ignore

    matrix = np.asarray(dsv.matrix, dtype=float)
    target = np.asarray(dsv.target, dtype=float).reshape(-1)
    n_unknowns = int(matrix.shape[1])
    variable = cp.Variable(n_unknowns)
    scale = max(float(np.linalg.norm(target)), float(np.linalg.norm(matrix, ord="fro")), 1.0e-30)
    objective = cp.Minimize(cp.norm((matrix @ variable - target) / scale, 2))
    constraints = [
        _cvxpy_residual_constraint(cp, block, variable, scale=block.allowed_residual) for block in auxiliary_blocks
    ]

    contour_step = _constrained_contour_step_limit(config)
    if bool(contour_step["contour_step_constraint_enabled"]):
        expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
        range_limit = float(contour_step["contour_step_constraint_max_range_A"])
        full_stream_function = expansion @ variable
        stream_min = cp.Variable()
        stream_max = cp.Variable()
        constraints.extend(
            [
                full_stream_function <= stream_max,
                full_stream_function >= stream_min,
                stream_max - stream_min <= range_limit,
            ]
        )

    engineering = engineering_limit_diagnostics(config)
    density_limit = float(engineering["engineering_surface_current_density_limit_A_per_m"])
    density_count = 0
    if density_limit > 0.0:
        density_constraints, density_count = _cvxpy_surface_current_density_constraints(
            cp,
            problem,
            variable,
            density_limit,
        )
        constraints.extend(density_constraints)

    shield = _shield_gradient_opposition_constraint_data(problem, config)
    if bool(shield["enabled"]):
        shield_scale = _cvxpy_positive_scale(shield["scale_T_per_m"], "shield-gradient opposition scale")
        constraints.append(
            (np.asarray(shield["matrix"], dtype=float).reshape(1, -1) / shield_scale) @ variable
            <= float(shield["limit_T_per_m"]) / shield_scale
        )

    cvx_problem = cp.Problem(objective, constraints)
    errors: list[str] = []
    for solver_name in _cvxpy_conic_solver_order(cp, config):
        try:
            cvx_problem.solve(solver=solver_name, verbose=False, **_cvxpy_solver_options(solver_name, config))
        except Exception as exc:
            errors.append(f"{solver_name}: {exc}")
            continue
        if cvx_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or variable.value is None:
            errors.append(f"{solver_name}: status {cvx_problem.status}")
            continue
        solution = np.asarray(variable.value, dtype=float).reshape(-1)
        residual_norm = float(np.linalg.norm(matrix @ solution - target))
        denominator = max(float(dsv.field_metric_denominator or 1.0), 1.0e-300)
        full_stream_function = np.asarray(problem.reducer.expansion_matrix, dtype=float) @ solution
        stream_function_range = float(np.ptp(full_stream_function)) if full_stream_function.size else 0.0
        max_surface_current_density = _solution_max_surface_current_density(problem, full_stream_function)
        return {
            "requested": True,
            "available": True,
            "globally_convex": True,
            "solver": str(solver_name),
            "status": str(cvx_problem.status),
            "field_metric_T": residual_norm / denominator,
            "residual_norm_T": residual_norm,
            "solution_norm": float(np.linalg.norm(solution)),
            "surface_current_density_limit_A_per_m": density_limit,
            "surface_current_density_constraint_count": int(density_count),
            "surface_current_density_max_A_per_m": max_surface_current_density,
            "surface_current_density_limit_ratio": (
                max_surface_current_density / density_limit if density_limit > 0.0 else float("nan")
            ),
            "contour_step_constraint_enabled": bool(contour_step["contour_step_constraint_enabled"]),
            "contour_step_constraint_max_range_A": float(contour_step["contour_step_constraint_max_range_A"]),
            "stream_function_range_A": stream_function_range,
            "stream_function_range_limit_ratio": (
                stream_function_range / float(contour_step["contour_step_constraint_max_range_A"])
                if bool(contour_step["contour_step_constraint_enabled"])
                else float("nan")
            ),
            "auxiliary_constraint_block_count": int(len(auxiliary_blocks)),
            "constraint_expression_count": int(len(constraints)),
        }
    details = "; ".join(errors) if errors else "no installed conic solver found"
    raise RuntimeError(f"Minimum-residual engineering floor solve failed ({details}).")


def solve_tikhonov_stream_function(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> StreamFunctionSolution:
    """Solve the default CoilGen stream-function inverse problem.

    Cost function:
        min_x ||A x - b||_2^2 + lambda^2 x^T R x

    A is the Bz sensitivity matrix in the target region, b is the target Bz
    field, R is the reduced resistance/current-energy metric, and lambda is selected
    either by fixed value or by the L-curve knee.
    """

    cost_function = _normalise_stream_function_cost_function(config.stream_function_cost_function)
    solution = _select_tikhonov_regularization(problem, config)
    solution.diagnostics.update(
        {
            "stream_function_scheme": "tikhonov",
            "stream_function_cost_function": cost_function,
            "cost_function_formula": STREAM_FUNCTION_COST_FORMULAS[cost_function],
            "unknown_count": int(problem.reduced_sensitivity.shape[1]),
            "target_sample_count": int(problem.reduced_sensitivity.shape[0]),
        }
    )
    if problem.symmetry_diagnostics is not None:
        solution.diagnostics.update(problem.symmetry_diagnostics)
    return solution


def _select_tikhonov_regularization(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> StreamFunctionSolution:
    method = str(getattr(config, "regularization_selection_method", "l_curve")).strip().lower().replace("-", "_")
    if method == "auto":
        method = "l_curve"
    if method not in {"l_curve", "fixed"}:
        raise ValueError("regularization_selection_method must be 'l_curve', 'auto', or 'fixed'.")

    scale = problem.regularization_scale
    ata, rtr, rhs = problem.normal_equation_matrices()
    spacing = _turn_spacing_penalty_context(problem, ata, config)
    smoothness = _stream_function_smoothness_context(problem, config)
    max_attempts = spacing["max_attempts"] if spacing["enabled"] else 1
    penalty_weight = spacing["initial_weight"]
    attempts: list[dict[str, float | bool]] = []
    solution: StreamFunctionSolution | None = None
    progress = ProgressReporter.from_config(config, label="Regularization")

    for attempt_ind in range(max_attempts):
        fixed_quadratic = smoothness["quadratic"]
        if spacing["enabled"]:
            spacing_quadratic = float(penalty_weight) * spacing["scaled_metric"]
            fixed_quadratic = spacing_quadratic if fixed_quadratic is None else fixed_quadratic + spacing_quadratic
        with progress_scope(
            95 * attempt_ind / max_attempts,
            95 * (attempt_ind + 1) / max_attempts,
            f"Regularization attempt {attempt_ind + 1} of at most {max_attempts}",
            label=progress.label,
        ):
            if method == "fixed":
                solution = _fixed_regularization_choice(
                    ata,
                    rtr,
                    rhs,
                    scale,
                    config,
                    selection_method="fixed",
                    fixed_quadratic=fixed_quadratic,
                )
            else:
                solution = _l_curve_regularization_choice(
                    problem,
                    ata,
                    rtr,
                    rhs,
                    scale,
                    config,
                    fixed_quadratic=fixed_quadratic,
                )

        progress.report(None, "checking the selected stream-function turn spacing")
        estimated_spacing = _estimate_min_turn_distance(problem, solution.reduced_stream_function, config)
        threshold_met = float(spacing["threshold"]) <= 0.0 or estimated_spacing >= float(spacing["threshold"])
        attempts.append(
            {
                "penalty_weight": float(penalty_weight) if spacing["enabled"] else 0.0,
                "estimated_min_turn_distance": float(estimated_spacing),
                "threshold_met": bool(threshold_met),
            }
        )
        if threshold_met:
            break
        penalty_weight *= float(spacing["growth"])

    if solution is None:
        raise RuntimeError("Stream-function regularization did not produce a solution.")
    solution.diagnostics.update(_stream_function_smoothness_diagnostics(smoothness))
    solution.diagnostics.update(_turn_spacing_diagnostics(spacing, attempts))
    progress.report(100.0, "regularization and turn-spacing review completed")
    return solution


def _select_target_error_constrained_efficiency(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> StreamFunctionSolution:
    scale = problem.regularization_scale
    ata, rtr, rhs = problem.normal_equation_matrices()
    spacing = _turn_spacing_penalty_context(problem, ata, config)
    smoothness = _stream_function_smoothness_context(problem, config)
    max_attempts = spacing["max_attempts"] if spacing["enabled"] else 1
    penalty_weight = spacing["initial_weight"]
    attempts: list[dict[str, float | bool]] = []
    solution: StreamFunctionSolution | None = None
    progress = ProgressReporter.from_config(config, label="Target-error efficiency")

    for attempt_ind in range(max_attempts):
        fixed_quadratic = smoothness["quadratic"]
        if spacing["enabled"]:
            spacing_quadratic = float(penalty_weight) * spacing["scaled_metric"]
            fixed_quadratic = spacing_quadratic if fixed_quadratic is None else fixed_quadratic + spacing_quadratic

        with progress_scope(
            95 * attempt_ind / max_attempts,
            95 * (attempt_ind + 1) / max_attempts,
            f"Target-error attempt {attempt_ind + 1} of at most {max_attempts}",
            label=progress.label,
        ):
            solution = _target_error_constrained_efficiency_choice_with_retries(
                problem,
                ata,
                rtr,
                rhs,
                scale,
                config,
                fixed_quadratic=fixed_quadratic,
            )

        progress.report(None, "checking the selected stream-function turn spacing")
        estimated_spacing = _estimate_min_turn_distance(problem, solution.reduced_stream_function, config)
        threshold_met = float(spacing["threshold"]) <= 0.0 or estimated_spacing >= float(spacing["threshold"])
        attempts.append(
            {
                "penalty_weight": float(penalty_weight) if spacing["enabled"] else 0.0,
                "estimated_min_turn_distance": float(estimated_spacing),
                "threshold_met": bool(threshold_met),
            }
        )
        if threshold_met:
            break
        penalty_weight *= float(spacing["growth"])

    if solution is None:
        raise RuntimeError("Target-error constrained stream-function optimization did not produce a solution.")
    solution.diagnostics.update(_stream_function_smoothness_diagnostics(smoothness))
    solution.diagnostics.update(_turn_spacing_diagnostics(spacing, attempts))
    progress.report(100.0, "target-error search and turn-spacing review completed")
    return solution


def _select_constrained_quadratic(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> StreamFunctionSolution:
    if _normalise_constrained_quadratic_objective_mode(config) == "minimax_current_density":
        if _stream_function_smoothness_requested(config) or _positive_config_float(
            config, "turn_spacing_regularization_factor"
        ):
            raise ValueError(
                "Pure minimax_current_density requires zero smoothness and turn-spacing penalty factors. "
                "Field, current-density, and contour-step hard bounds remain available."
            )
    ata, rtr, _rhs = problem.normal_equation_matrices()
    spacing = _turn_spacing_penalty_context(problem, ata, config)
    smoothness = _stream_function_smoothness_context(problem, config)
    max_attempts = spacing["max_attempts"] if spacing["enabled"] else 1
    penalty_weight = spacing["initial_weight"]
    attempts: list[dict[str, float | bool]] = []
    solution: StreamFunctionSolution | None = None
    progress = ProgressReporter.from_config(config, label="Constrained quadratic")

    for attempt_ind in range(max_attempts):
        fixed_quadratic = smoothness["quadratic"]
        if spacing["enabled"]:
            spacing_quadratic = float(penalty_weight) * spacing["scaled_metric"]
            fixed_quadratic = spacing_quadratic if fixed_quadratic is None else fixed_quadratic + spacing_quadratic

        with progress_scope(
            95 * attempt_ind / max_attempts,
            95 * (attempt_ind + 1) / max_attempts,
            f"Constrained attempt {attempt_ind + 1} of at most {max_attempts}",
            label=progress.label,
        ):
            solution = _constrained_quadratic_choice(problem, rtr, config, fixed_quadratic=fixed_quadratic)

        progress.report(None, "checking the selected stream-function turn spacing")
        estimated_spacing = _estimate_min_turn_distance(problem, solution.reduced_stream_function, config)
        threshold_met = float(spacing["threshold"]) <= 0.0 or estimated_spacing >= float(spacing["threshold"])
        attempts.append(
            {
                "penalty_weight": float(penalty_weight) if spacing["enabled"] else 0.0,
                "estimated_min_turn_distance": float(estimated_spacing),
                "threshold_met": bool(threshold_met),
            }
        )
        if threshold_met:
            break
        penalty_weight *= float(spacing["growth"])

    if solution is None:
        raise RuntimeError("Constrained quadratic stream-function optimization did not produce a solution.")
    solution.diagnostics.update(_stream_function_smoothness_diagnostics(smoothness))
    solution.diagnostics.update(_turn_spacing_diagnostics(spacing, attempts))
    progress.report(100.0, "constrained solve and turn-spacing review completed")
    return solution


def _target_error_constrained_efficiency_choice_with_retries(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    scale: float,
    config: CoilGenConfig,
    *,
    fixed_quadratic: np.ndarray | None = None,
) -> StreamFunctionSolution:
    """Run the fixed-error search and retry with broader search policy if needed."""

    solutions: list[StreamFunctionSolution] = []
    attempts: list[dict[str, Any]] = []
    solution_attempt_indices: list[int] = []
    retry_enabled = _target_error_auto_retry_enabled(config)
    max_reruns = _target_error_auto_retry_max_reruns(config)
    maximum_attempts = 1 + (max_reruns if retry_enabled else 0)
    progress = ProgressReporter.from_config(config, label="Target-error efficiency")

    with progress_scope(0, 95 / maximum_attempts, "Initial target-error search", label=progress.label):
        initial_solution = _target_error_constrained_efficiency_choice(
            problem,
            ata,
            rtr,
            rhs,
            scale,
            config,
            fixed_quadratic=fixed_quadratic,
        )
    solutions.append(initial_solution)
    attempts.append(_target_error_auto_retry_attempt_summary(initial_solution, config, rerun=0))
    solution_attempt_indices.append(0)

    selected_index = 0
    selected_violation = _target_error_solution_violation_ratio(initial_solution)

    if retry_enabled and max_reruns > 0 and not _target_error_solution_meets_constraint(initial_solution):
        for rerun in range(1, max_reruns + 1):
            retry_config = _target_error_retry_config(config, rerun)
            try:
                with progress_scope(
                    95 * rerun / maximum_attempts,
                    95 * (rerun + 1) / maximum_attempts,
                    f"Target-error retry {rerun}/{max_reruns}",
                    label=progress.label,
                ):
                    retry_solution = _target_error_constrained_efficiency_choice(
                        problem,
                        ata,
                        rtr,
                        rhs,
                        scale,
                        retry_config,
                        fixed_quadratic=fixed_quadratic,
                    )
            except DesignCancelled:
                raise
            except Exception as exc:  # pragma: no cover - diagnostic fallback for rare numerical failures
                attempts.append(_target_error_auto_retry_error_summary(retry_config, rerun, exc))
                continue

            solutions.append(retry_solution)
            attempts.append(_target_error_auto_retry_attempt_summary(retry_solution, retry_config, rerun=rerun))
            solution_attempt_indices.append(len(attempts) - 1)
            retry_violation = _target_error_solution_violation_ratio(retry_solution)
            if _target_error_retry_solution_is_better(
                retry_solution,
                retry_violation,
                solutions[selected_index],
                selected_violation,
            ):
                selected_index = len(solutions) - 1
                selected_violation = retry_violation
            if _target_error_solution_meets_constraint(retry_solution):
                selected_index = len(solutions) - 1
                break

    selected_solution = solutions[selected_index]
    selected_solution.diagnostics.update(
        _target_error_auto_retry_diagnostics(
            config,
            attempts,
            selected_index=solution_attempt_indices[selected_index],
            retry_enabled=retry_enabled,
            max_reruns=max_reruns,
        )
    )
    progress.report(100.0, "target-error search and retry comparison completed")
    return selected_solution


def _target_error_constrained_efficiency_choice(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    scale: float,
    config: CoilGenConfig,
    *,
    fixed_quadratic: np.ndarray | None = None,
) -> StreamFunctionSolution:
    constraint_blocks = _target_error_constraint_blocks(problem, config)

    factors = _target_error_constraint_candidate_factors(
        config,
        problem=problem,
        scale=scale,
        fixed_quadratic=fixed_quadratic,
    )
    lambdas = factors * scale
    with progress_scope(0, 80, "Scanning target-error regularization candidates", label="Target-error efficiency"):
        scan = _scan_l_curve_by_normal_equations(
            problem,
            ata,
            rtr,
            rhs,
            lambdas,
            config,
            fixed_quadratic,
            progress_label="Target-error efficiency",
        )
    residuals, regularization_norms, candidate_solver = scan
    valid = np.isfinite(residuals) & np.isfinite(regularization_norms)
    if not np.any(valid):
        raise RuntimeError("Target-error constrained efficiency scan produced no finite candidate solutions.")

    selected_index, selected_lambda, selection_reason, constraint_met = _select_error_constrained_lambda(
        problem,
        ata,
        rtr,
        rhs,
        lambdas,
        residuals,
        regularization_norms,
        constraint_blocks,
        config,
        fixed_quadratic,
        minimum_allowed_lambda=_regularization_auto_min_lambda(config, scale),
    )
    target_error_lambda = float(selected_lambda)
    target_error_selection_met = bool(constraint_met)
    guard_notes: list[str] = []
    contour_step_guard_relaxed = False
    selected_lambda, guard_notes = _apply_auto_range_contour_step_lambda_guard(
        problem,
        ata,
        rtr,
        rhs,
        lambdas,
        selected_lambda,
        config,
        fixed_quadratic,
    )
    if guard_notes and target_error_selection_met:
        try:
            guarded_candidate = _solve_for_regularization_lambda(
                ata,
                rtr,
                rhs,
                float(selected_lambda),
                config,
                fixed_quadratic,
            )
        except np.linalg.LinAlgError:
            guarded_candidate = None
        if guarded_candidate is None or not _constraint_blocks_met(
            constraint_blocks,
            guarded_candidate,
            tolerance=0.0,
        ):
            guard_notes.append(
                "auto-range contour-step guard not enforced because it would violate the target-error constraint"
            )
            selected_lambda = target_error_lambda
            contour_step_guard_relaxed = True
    if guard_notes:
        selection_reason = f"{selection_reason}; {'; '.join(guard_notes)}"
    selected_factor = float(selected_lambda / scale) if scale > 0.0 else float(selected_lambda)
    reduced_sf = _solve_for_regularization_lambda(ata, rtr, rhs, selected_lambda, config, fixed_quadratic)
    constraint_diagnostics = _residual_constraint_diagnostics(problem, reduced_sf, constraint_blocks, config)
    constraint_met = bool(constraint_diagnostics["target_error_constraint_met"])
    selected_residual = float(constraint_diagnostics["target_error_constraint_residual_norm_T"])
    selected_norm = problem.resistance_norm(reduced_sf)
    estimated_contour_step = _estimated_contour_step_current_A(problem, reduced_sf, config)
    factors_out, lambdas_out, residuals_out, regularization_norms_out, selected_index_out = (
        _augment_regularization_scan_with_selected_point(
            factors,
            lambdas,
            residuals,
            regularization_norms,
            selected_factor,
            float(selected_lambda),
            float(selected_residual),
            float(selected_norm),
        )
    )
    diagnostics = {
        "selection_method": "target_error_constraint",
        "candidate_solver": candidate_solver,
        "regularization_auto_range": _regularization_auto_range_enabled(config),
        "fallback": not bool(constraint_met),
        "fallback_reason": "" if constraint_met else selection_reason,
        "target_error_constraint_selection_reason": selection_reason,
        "candidate_notes": guard_notes,
        "target_error_contour_step_guard_relaxed_for_constraint": bool(contour_step_guard_relaxed),
        "estimated_selected_contour_step_A": estimated_contour_step,
        "regularization_auto_range_max_contour_step_A": _nonnegative_config_value(
            config,
            "regularization_auto_range_max_contour_step_A",
            0.0,
        ),
        "regularization_auto_range_min_lambda": _regularization_auto_min_lambda(config, scale),
        **constraint_diagnostics,
        "target_error_constraint_resistance_norm": float(selected_norm),
        "factor_scale": scale,
        "selected_index": int(selected_index_out),
        "selected_factor": selected_factor,
        "selected_lambda": float(selected_lambda),
        "candidate_factors": _float_list(factors_out),
        "candidate_lambdas": _float_list(lambdas_out),
        "residual_norms": _float_list(residuals_out),
        "regularization_norms": _float_list(regularization_norms_out),
        "curvature": [],
    }
    return StreamFunctionSolution(reduced_sf, diagnostics)


def _constrained_quadratic_choice(
    problem: StreamFunctionOptimizationProblem,
    rtr: np.ndarray,
    config: CoilGenConfig,
    *,
    fixed_quadratic: np.ndarray | None = None,
) -> StreamFunctionSolution:
    constraint_blocks = _target_error_constraint_blocks(problem, config)
    objective_mode = _normalise_constrained_quadratic_objective_mode(config)
    if objective_mode == "minimax_current_density" and fixed_quadratic is not None and np.any(fixed_quadratic):
        raise ValueError(
            "Pure minimax_current_density does not combine quadratic smoothness/spacing penalties. "
            "Disable these penalties, or use balanced mode with a hard surface-current-density bound."
        )
    resistance_weight = _constrained_quadratic_resistance_weight(config, objective_mode)
    objective_quadratic = _quadratic_objective_matrix(resistance_weight * rtr, fixed_quadratic)
    reduced_sf, solver_diagnostics = _solve_constrained_quadratic_cvxpy(
        problem,
        objective_quadratic,
        constraint_blocks,
        config,
    )
    constraint_diagnostics = _residual_constraint_diagnostics(problem, reduced_sf, constraint_blocks, config)
    selected_residual = float(constraint_diagnostics["target_error_constraint_residual_norm_T"])
    selected_norm = problem.resistance_norm(reduced_sf)
    contour_step_limit = _constrained_contour_step_limit(config)
    estimated_contour_step = _estimated_contour_step_current_A(problem, reduced_sf, config)
    guarded_level_count = int(contour_step_limit["contour_step_constraint_level_count"])
    estimated_guarded_step = _estimated_contour_step_current_for_level_count_A(
        problem,
        reduced_sf,
        config,
        guarded_level_count,
    )
    diagnostics = {
        "selection_method": "constrained_quadratic",
        "candidate_solver": solver_diagnostics["solver"],
        "fallback": False,
        "fallback_reason": "",
        "target_error_constraint_selection_reason": "direct constrained quadratic solve",
        **constraint_diagnostics,
        "target_error_constraint_resistance_norm": float(selected_norm),
        "estimated_selected_contour_step_A": float(estimated_contour_step),
        "estimated_guarded_contour_step_A": float(estimated_guarded_step),
        "regularization_auto_range_max_contour_step_A": _nonnegative_config_value(
            config,
            "regularization_auto_range_max_contour_step_A",
            0.0,
        ),
        "constrained_solver_backend": "cvxpy",
        "constrained_quadratic_objective_mode": objective_mode,
        "constrained_quadratic_resistance_weight": float(resistance_weight),
        "constrained_solver_status": solver_diagnostics["status"],
        "constrained_solver_name": solver_diagnostics["solver"],
        "constrained_solver_objective_value": solver_diagnostics["objective_value"],
        "constrained_solver_objective_unit": solver_diagnostics.get("objective_unit", "weighted_quadratic"),
        "minimax_current_density_enabled": bool(solver_diagnostics.get("minimax_current_density_enabled", False)),
        "minimax_current_density_peak_A_per_m": float(
            solver_diagnostics.get("minimax_current_density_peak_A_per_m", np.nan)
        ),
        "minimax_current_density_epigraph_A_per_m": float(
            solver_diagnostics.get("minimax_current_density_epigraph_A_per_m", np.nan)
        ),
        "minimax_current_density_face_count": int(solver_diagnostics.get("minimax_current_density_face_count", 0)),
        "constrained_solver_problem_unknown_count": int(solver_diagnostics.get("problem_unknown_count", 0)),
        "constrained_solver_residual_constraint_block_count": int(
            solver_diagnostics.get("residual_constraint_block_count", 0)
        ),
        "constrained_solver_residual_constraint_row_count": int(
            solver_diagnostics.get("residual_constraint_row_count", 0)
        ),
        "constrained_solver_cvxpy_constraint_expression_count": int(
            solver_diagnostics.get("cvxpy_constraint_expression_count", 0)
        ),
        "constrained_solver_objective_normalization_scale": float(
            solver_diagnostics.get("objective_normalization_scale", np.nan)
        ),
        "constrained_solver_scaled_objective_value": float(
            solver_diagnostics.get("solver_scaled_objective_value", np.nan)
        ),
        "constrained_solver_residual_normalization_min": float(
            solver_diagnostics.get("residual_constraint_normalization_min", np.nan)
        ),
        "constrained_solver_residual_normalization_max": float(
            solver_diagnostics.get("residual_constraint_normalization_max", np.nan)
        ),
        "constrained_solver_contour_step_normalization_A": float(
            solver_diagnostics.get("contour_step_constraint_normalization_A", np.nan)
        ),
        "constrained_solver_verbose": bool(solver_diagnostics.get("solver_verbose", False)),
        "constrained_solver_thread_count_requested": int(solver_diagnostics.get("solver_thread_count_requested", 0)),
        "constrained_solver_thread_count_applied": int(solver_diagnostics.get("solver_thread_count_applied", 0)),
        "constrained_solver_thread_option": str(solver_diagnostics.get("solver_thread_option", "")),
        "constrained_solver_elapsed_seconds": float(solver_diagnostics.get("solver_elapsed_seconds", np.nan)),
        "constrained_solver_error_count": int(solver_diagnostics.get("solver_error_count", 0)),
        "constrained_solver_residual_slack_T": _minimum_residual_slack(constraint_blocks, reduced_sf),
        "surface_current_density_constraint_enabled": bool(
            solver_diagnostics.get("surface_current_density_constraint_enabled", False)
        ),
        "surface_current_density_constraint_limit_A_per_m": float(
            solver_diagnostics.get("surface_current_density_constraint_limit_A_per_m", np.nan)
        ),
        "surface_current_density_constraint_count": int(
            solver_diagnostics.get("surface_current_density_constraint_count", 0)
        ),
        "shield_gradient_opposition_requested": bool(
            solver_diagnostics.get("shield_gradient_opposition_requested", False)
        ),
        "shield_gradient_opposition_enabled": bool(solver_diagnostics.get("shield_gradient_opposition_enabled", False)),
        "shield_gradient_opposition_constraint_met": bool(
            solver_diagnostics.get("shield_gradient_opposition_constraint_met", True)
        ),
        "shield_gradient_opposition_reason": str(solver_diagnostics.get("shield_gradient_opposition_reason", "")),
        "shield_gradient_opposition_value_T_per_m": float(
            solver_diagnostics.get("shield_gradient_opposition_value_T_per_m", np.nan)
        ),
        "shield_gradient_opposition_limit_T_per_m": float(
            solver_diagnostics.get("shield_gradient_opposition_limit_T_per_m", np.nan)
        ),
        "shield_gradient_opposition_margin_T_per_m": float(
            solver_diagnostics.get("shield_gradient_opposition_margin_T_per_m", np.nan)
        ),
        "shield_gradient_opposition_target_T_per_m": float(
            solver_diagnostics.get("shield_gradient_opposition_target_T_per_m", np.nan)
        ),
        "shield_gradient_opposition_max_fraction_of_target": float(
            solver_diagnostics.get("shield_gradient_opposition_max_fraction_of_target", np.nan)
        ),
        "shield_gradient_opposition_part_indices": str(
            solver_diagnostics.get("shield_gradient_opposition_part_indices", "")
        ),
        **contour_step_limit,
        "factor_scale": problem.regularization_scale,
        "selected_index": 0,
        "selected_factor": 0.0,
        "selected_lambda": 0.0,
        "candidate_factors": [1.0],
        "candidate_lambdas": [0.0],
        "residual_norms": [float(selected_residual)],
        "regularization_norms": [float(selected_norm)],
        "curvature": [float("nan")],
    }
    if bool(getattr(config, "progress_flag", False)):
        _print_constraint_status("Constrained quadratic post-check", diagnostics)
    return StreamFunctionSolution(reduced_sf, diagnostics)


def _target_error_solution_meets_constraint(solution: StreamFunctionSolution) -> bool:
    return bool(solution.diagnostics.get("target_error_constraint_met", False))


def _normalise_constrained_quadratic_fallback_policy(config: CoilGenConfig) -> str:
    raw_value = str(getattr(config, "constrained_quadratic_fallback_policy", "target_error_scan"))
    key = raw_value.strip().lower().replace("-", "_")
    aliases = {
        "target_error_scan": "target_error_scan",
        "tikhonov_scan": "target_error_scan",
        "diagnostic": "diagnostic",
        "strict": "strict",
        "fail": "strict",
        "error": "strict",
    }
    return aliases.get(key, "target_error_scan")


def _target_error_retry_config(config: CoilGenConfig, rerun: int) -> CoilGenConfig:
    retry = copy.copy(config)
    rerun = max(int(rerun), 1)

    base_min_multiplier = _finite_positive_config_attr(
        config,
        "target_error_constraint_factor_min_multiplier",
        1.0e-10,
    )
    shrink = _target_error_auto_retry_factor_shrink(config)
    retry.target_error_constraint_factor_min_multiplier = max(base_min_multiplier * (shrink**rerun), 1.0e-300)

    base_count = max(int(getattr(config, "target_error_constraint_factor_count", 61)), 3)
    max_count = max(_target_error_auto_retry_max_candidate_count(config), base_count)
    growth = _target_error_auto_retry_candidate_growth(config)
    grown_count = int(np.ceil(base_count * (growth**rerun)))
    retry.target_error_constraint_factor_count = min(max_count, max(base_count, grown_count))

    # Retry search policy: never relax the physical residual specification or
    # current/contour-step guard. Only remove artificial lower-lambda floors and
    # use the lowest achievable residual if the specification is infeasible.
    retry.regularization_auto_range_min_factor = 0.0
    retry.target_error_infeasible_fallback = "minimum_residual"
    retry.target_error_infeasible_residual_margin_percent = 0.0
    return retry


def _target_error_auto_retry_enabled(config: CoilGenConfig) -> bool:
    value = getattr(config, "target_error_auto_retry_enabled", True)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "auto"}
    return bool(value)


def _target_error_auto_retry_max_reruns(config: CoilGenConfig) -> int:
    try:
        return max(int(getattr(config, "target_error_auto_retry_max_reruns", 10)), 0)
    except (TypeError, ValueError):
        return 10


def _target_error_auto_retry_factor_shrink(config: CoilGenConfig) -> float:
    value = _finite_positive_config_attr(config, "target_error_auto_retry_factor_shrink", 0.1)
    if value >= 1.0:
        return 0.1
    return max(value, 1.0e-12)


def _target_error_auto_retry_candidate_growth(config: CoilGenConfig) -> float:
    value = _finite_positive_config_attr(config, "target_error_auto_retry_candidate_growth", 1.2)
    return max(value, 1.0)


def _target_error_auto_retry_max_candidate_count(config: CoilGenConfig) -> int:
    try:
        return max(int(getattr(config, "target_error_auto_retry_max_candidate_count", 201)), 3)
    except (TypeError, ValueError):
        return 201


def _finite_positive_config_attr(config: CoilGenConfig, field_name: str, default: float) -> float:
    try:
        value = float(getattr(config, field_name, default))
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value) or value <= 0.0:
        return float(default)
    return float(value)


def _finite_config_attr(config: CoilGenConfig, field_name: str, default: float) -> float:
    try:
        value = float(getattr(config, field_name, default))
    except (TypeError, ValueError):
        return float(default)
    return float(value) if np.isfinite(value) else float(default)


def _target_error_solution_violation_ratio(solution: StreamFunctionSolution) -> float:
    diagnostics = solution.diagnostics
    max_ratio = _diagnostic_float(diagnostics, "target_error_constraint_max_violation_ratio")
    if np.isfinite(max_ratio):
        return float(max_ratio)
    ratios: list[float] = []
    for prefix in (
        "combined_weighted",
        "dsv",
        "linearity_rms",
        "linearity_peak",
        "gradient_strength",
        "cross_gradient",
        "suppression",
    ):
        if f"{prefix}_constraint_met" not in diagnostics:
            continue
        value = _diagnostic_float(diagnostics, f"{prefix}_constraint_field_metric_T")
        allowed = _diagnostic_float(diagnostics, f"{prefix}_constraint_allowed_field_metric_T")
        if np.isfinite(value) and np.isfinite(allowed) and allowed > 0.0:
            ratios.append(value / allowed)
    if not ratios:
        residual = _diagnostic_float(diagnostics, "target_error_constraint_residual_norm_T")
        allowed = _diagnostic_float(diagnostics, "target_error_allowed_residual_norm_T")
        if np.isfinite(residual) and np.isfinite(allowed) and allowed > 0.0:
            ratios.append(residual / allowed)
    if not ratios:
        return 0.0 if _target_error_solution_meets_constraint(solution) else float("inf")
    return float(max(ratios))


def _target_error_retry_solution_is_better(
    candidate: StreamFunctionSolution,
    candidate_violation: float,
    incumbent: StreamFunctionSolution,
    incumbent_violation: float,
) -> bool:
    candidate_finite = bool(np.isfinite(candidate_violation))
    incumbent_finite = bool(np.isfinite(incumbent_violation))
    if candidate_finite != incumbent_finite:
        return candidate_finite
    if not candidate_finite:
        return False

    tolerance = 1.0e-10 * max(abs(float(candidate_violation)), abs(float(incumbent_violation)), 1.0)
    if float(candidate_violation) < float(incumbent_violation) - tolerance:
        return True
    if float(candidate_violation) > float(incumbent_violation) + tolerance:
        return False

    candidate_met = _target_error_solution_meets_constraint(candidate)
    incumbent_met = _target_error_solution_meets_constraint(incumbent)
    if candidate_met != incumbent_met:
        return candidate_met

    candidate_lambda = _diagnostic_float(candidate.diagnostics, "selected_lambda")
    incumbent_lambda = _diagnostic_float(incumbent.diagnostics, "selected_lambda")
    if np.isfinite(candidate_lambda) and np.isfinite(incumbent_lambda):
        lambda_tolerance = 1.0e-12 * max(abs(candidate_lambda), abs(incumbent_lambda), 1.0)
        if candidate_lambda > incumbent_lambda + lambda_tolerance:
            return True
        if candidate_lambda < incumbent_lambda - lambda_tolerance:
            return False

    candidate_resistance = _diagnostic_float(candidate.diagnostics, "target_error_constraint_resistance_norm")
    incumbent_resistance = _diagnostic_float(incumbent.diagnostics, "target_error_constraint_resistance_norm")
    if np.isfinite(candidate_resistance) and np.isfinite(incumbent_resistance):
        resistance_tolerance = 1.0e-12 * max(abs(candidate_resistance), abs(incumbent_resistance), 1.0)
        if candidate_resistance < incumbent_resistance - resistance_tolerance:
            return True
        if candidate_resistance > incumbent_resistance + resistance_tolerance:
            return False

    return True


def _target_error_auto_retry_attempt_summary(
    solution: StreamFunctionSolution,
    config: CoilGenConfig,
    *,
    rerun: int,
) -> dict[str, Any]:
    diagnostics = solution.diagnostics
    return {
        "rerun": int(rerun),
        "met": bool(diagnostics.get("target_error_constraint_met", False)),
        "violation_ratio": _target_error_solution_violation_ratio(solution),
        "selected_lambda": _diagnostic_float(diagnostics, "selected_lambda"),
        "selected_factor": _diagnostic_float(diagnostics, "selected_factor"),
        "field_metric_T": _diagnostic_float(diagnostics, "target_error_constraint_field_metric_T"),
        "allowed_field_metric_T": _diagnostic_float(diagnostics, "target_error_allowed_field_metric_T"),
        "relative_residual_percent": _diagnostic_float(
            diagnostics,
            "target_error_constraint_relative_residual_percent",
        ),
        "regularization_auto_range_min_factor": _finite_config_attr(
            config,
            "regularization_auto_range_min_factor",
            0.0,
        ),
        "target_error_constraint_factor_min_multiplier": _finite_config_attr(
            config,
            "target_error_constraint_factor_min_multiplier",
            1.0e-10,
        ),
        "target_error_constraint_factor_count": int(getattr(config, "target_error_constraint_factor_count", 61)),
        "target_error_infeasible_fallback": str(getattr(config, "target_error_infeasible_fallback", "residual_floor")),
        "fallback": bool(diagnostics.get("fallback", False)),
        "fallback_reason": str(diagnostics.get("fallback_reason", "")),
        "selection_reason": str(diagnostics.get("target_error_constraint_selection_reason", "")),
    }


def _target_error_auto_retry_error_summary(
    config: CoilGenConfig,
    rerun: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "rerun": int(rerun),
        "met": False,
        "violation_ratio": float("inf"),
        "error": str(exc),
        "regularization_auto_range_min_factor": _finite_config_attr(
            config,
            "regularization_auto_range_min_factor",
            0.0,
        ),
        "target_error_constraint_factor_min_multiplier": _finite_config_attr(
            config,
            "target_error_constraint_factor_min_multiplier",
            1.0e-10,
        ),
        "target_error_constraint_factor_count": int(getattr(config, "target_error_constraint_factor_count", 61)),
        "target_error_infeasible_fallback": str(
            getattr(config, "target_error_infeasible_fallback", "minimum_residual")
        ),
    }


def _target_error_auto_retry_diagnostics(
    config: CoilGenConfig,
    attempts: list[dict[str, Any]],
    *,
    selected_index: int,
    retry_enabled: bool,
    max_reruns: int,
) -> dict[str, Any]:
    selected_index = int(np.clip(selected_index, 0, max(len(attempts) - 1, 0)))
    selected_attempt = attempts[selected_index] if attempts else {}
    rerun_count = sum(1 for attempt in attempts if int(attempt.get("rerun", 0)) > 0)
    selected_rerun = int(selected_attempt.get("rerun", 0))
    selected_met = bool(selected_attempt.get("met", False))
    initial_met = bool(attempts[0].get("met", False)) if attempts else False
    if initial_met:
        reason = "initial attempt met the requested target-error specification"
    elif not retry_enabled:
        reason = "automatic target-error retry disabled"
    elif max_reruns <= 0:
        reason = "automatic target-error retry has zero allowed reruns"
    elif selected_met:
        reason = f"target-error specification met after {selected_rerun} automatic rerun(s)"
    else:
        reason = (
            f"target-error specification still unmet after {rerun_count} automatic rerun(s); "
            "returned the attempt with the lowest constraint-violation ratio"
        )
    return {
        "target_error_auto_retry_enabled": bool(retry_enabled),
        "target_error_auto_retry_max_reruns": int(max_reruns),
        "target_error_auto_retry_factor_shrink": _target_error_auto_retry_factor_shrink(config),
        "target_error_auto_retry_candidate_growth": _target_error_auto_retry_candidate_growth(config),
        "target_error_auto_retry_max_candidate_count": _target_error_auto_retry_max_candidate_count(config),
        "target_error_auto_retry_count": int(rerun_count),
        "target_error_auto_retry_initial_met": bool(initial_met),
        "target_error_auto_retry_selected_rerun": int(selected_rerun),
        "target_error_auto_retry_selected_violation_ratio": float(
            selected_attempt.get("violation_ratio", float("nan"))
        ),
        "target_error_auto_retry_reason": reason,
        "target_error_auto_retry_attempts": attempts,
    }


def _suppression_target_diagnostics(
    problem: StreamFunctionOptimizationProblem,
    reduced_stream_function: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, Any]:
    """Report the actual zero-field suppression-target residual for any solver."""

    suppressed_mask = _suppression_point_mask(problem)
    if suppressed_mask is None:
        return _empty_suppression_target_diagnostics(config)
    mask = _active_point_mask(problem) & suppressed_mask
    if not np.any(mask):
        return _empty_suppression_target_diagnostics(config)

    matrix, target, component, weighting, denominator = _suppression_constraint_system(problem, mask, config)
    unweighted_matrix, unweighted_target, _, _, _ = _suppression_constraint_system(
        problem,
        mask,
        config,
        force_uniform=True,
    )
    residual = np.asarray(matrix @ reduced_stream_function - target, dtype=float).reshape(-1)
    unweighted_residual = np.asarray(
        unweighted_matrix @ reduced_stream_function - unweighted_target, dtype=float
    ).reshape(-1)
    finite = np.isfinite(residual)
    if not np.any(finite):
        rms = peak = residual_norm = weighted_norm = weighted_rms = float("nan")
    else:
        selected = residual[finite]
        rms = float(np.linalg.norm(selected) / max(float(denominator), 1.0e-300))
        finite_unweighted = unweighted_residual[np.isfinite(unweighted_residual)]
        peak = float(np.max(np.abs(finite_unweighted))) if finite_unweighted.size else float("nan")
        residual_norm = float(np.linalg.norm(finite_unweighted))
        weighted_norm = float(np.linalg.norm(selected))
        weighted_rms = rms
    selected_weights_all = _suppression_point_weights(
        problem,
        np.flatnonzero(mask),
        config,
        force_uniform=False,
    )
    diagnostics = {
        "suppression_target_enabled": True,
        "suppression_target_component": component,
        "suppression_surface_weighting": weighting,
        "suppression_target_sample_count": int(np.count_nonzero(mask)),
        "suppression_target_weight_mean": float(np.mean(selected_weights_all)),
        "suppression_target_weight_max": float(np.max(selected_weights_all)),
        "suppression_target_rms_residual_T": rms,
        "suppression_target_peak_abs_residual_T": peak,
        "suppression_target_residual_norm_T": residual_norm,
        "suppression_target_weighted_residual_norm_T": weighted_norm,
        "suppression_target_weighted_rms_residual_T": weighted_rms,
    }
    diagnostics["suppression_field_rms_T"] = rms
    return diagnostics


def _empty_suppression_target_diagnostics(config: CoilGenConfig) -> dict[str, Any]:
    return {
        "suppression_target_enabled": False,
        "suppression_target_component": _suppression_component(config),
        "suppression_surface_weighting": _normalise_suppression_surface_weighting(config),
        "suppression_target_sample_count": 0,
        "suppression_target_weight_mean": float("nan"),
        "suppression_target_weight_max": float("nan"),
        "suppression_target_rms_residual_T": float("nan"),
        "suppression_target_peak_abs_residual_T": float("nan"),
        "suppression_target_residual_norm_T": float("nan"),
        "suppression_target_weighted_residual_norm_T": float("nan"),
        "suppression_target_weighted_rms_residual_T": float("nan"),
    }


def _print_constraint_status(label: str, diagnostics: dict[str, Any]) -> None:
    rows = diagnostics.get("target_error_constraint_block_diagnostics", ())
    if not isinstance(rows, list) or not rows:
        return
    print(f"{label}:")
    for row in rows:
        status = "PASS" if bool(row.get("met", False)) else "FAIL"
        print(
            "  "
            f"{row.get('name', 'unknown')} [{row.get('metric', '')}, {row.get('component', '')}, {row.get('weighting', '')}] {status}: "
            f"value={_format_diagnostic_number(row.get('field_metric_T'))} T, "
            f"allowed={_format_diagnostic_number(row.get('allowed_field_metric_T'))} T, "
            f"ratio={_format_diagnostic_number(row.get('violation_ratio'))}"
        )


def _format_diagnostic_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.6g}"


def _constraint_blocks_met(
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    reduced_stream_function: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
) -> bool:
    for block in constraint_blocks:
        residual = _constraint_residual_measure(block, reduced_stream_function)
        if not np.isfinite(residual):
            return False
        if residual > float(block.allowed_residual) * (1.0 + tolerance):
            return False
    return True


def _constraint_violation_ratio(
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    reduced_stream_function: np.ndarray,
) -> float:
    ratios = []
    for block in constraint_blocks:
        residual = _constraint_residual_measure(block, reduced_stream_function)
        allowed = max(float(block.allowed_residual), 1.0e-300)
        ratios.append(residual / allowed if np.isfinite(residual) else float("inf"))
    return float(max(ratios)) if ratios else float("inf")


def _minimum_residual_slack(
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    reduced_stream_function: np.ndarray,
) -> float:
    if not constraint_blocks:
        return float("nan")
    slacks = [
        float(block.allowed_residual) - _constraint_residual_measure(block, reduced_stream_function)
        for block in constraint_blocks
    ]
    return float(np.min(slacks))


def _constrained_quadratic_resistance_weight(config: CoilGenConfig, objective_mode: str) -> float:
    if objective_mode == "minimax_current_density":
        return 0.0
    if objective_mode != "smoothness_priority":
        return 1.0
    try:
        value = float(getattr(config, "constrained_quadratic_resistance_tiebreak_factor", 1.0e-3))
    except (TypeError, ValueError):
        value = 1.0e-3
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "constrained_quadratic_resistance_tiebreak_factor must be finite and positive in smoothness_priority mode."
        )
    return float(value)


def _quadratic_objective_matrix(
    rtr: np.ndarray,
    fixed_quadratic: np.ndarray | None,
) -> np.ndarray:
    objective = _symmetric_quadratic_matrix(rtr)
    if fixed_quadratic is not None:
        objective = objective + np.asarray(fixed_quadratic, dtype=float)
    objective = 0.5 * (objective + objective.T)
    if not np.all(np.isfinite(objective)):
        raise ValueError("Constrained quadratic objective contains non-finite values.")
    return objective


def _regularization_operator_from_quadratic(matrix: np.ndarray) -> np.ndarray | None:
    """Return L with L.T @ L equal to the quadratic metric, when invertible."""

    quadratic = _symmetric_quadratic_matrix(matrix)
    try:
        return np.linalg.cholesky(quadratic).T
    except np.linalg.LinAlgError:
        pass

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        return None
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) <= 1.0e-12 * scale:
        return None
    return np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def _regularization_whitened_sensitivity(
    problem: StreamFunctionOptimizationProblem,
) -> RegularizationWhitenedSystem | None:
    """Return A R^{-1/2} on the positive eigenspace of semidefinite R."""

    quadratic = _symmetric_quadratic_matrix(problem.reduced_resistance)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        return None

    spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    if not np.isfinite(spectral_radius) or spectral_radius <= 0.0:
        return None
    eigenvalue_floor = max(
        spectral_radius * 1.0e-12,
        np.finfo(float).eps * max(int(quadratic.shape[0]), 1) * spectral_radius,
    )
    if np.min(eigenvalues) < -eigenvalue_floor:
        return None

    positive = eigenvalues > eigenvalue_floor
    if not np.any(positive):
        return None

    positive_vectors = eigenvectors[:, positive]
    inverse_sqrt = 1.0 / np.sqrt(eigenvalues[positive])
    transformed = np.asarray(problem.reduced_sensitivity, dtype=float) @ (positive_vectors * inverse_sqrt[None, :])
    if not np.all(np.isfinite(transformed)):
        return None

    nullspace_relative_field_norm = 0.0
    nullity = int(np.count_nonzero(~positive))
    if nullity:
        null_vectors = eigenvectors[:, ~positive]
        null_coupling = np.asarray(problem.reduced_sensitivity, dtype=float) @ null_vectors
        denominator = max(float(np.linalg.norm(problem.reduced_sensitivity)), 1.0e-300)
        nullspace_relative_field_norm = float(np.linalg.norm(null_coupling) / denominator)

    return RegularizationWhitenedSystem(
        transformed_sensitivity=transformed,
        regularization_rank=int(np.count_nonzero(positive)),
        regularization_nullity=nullity,
        eigenvalue_floor=float(eigenvalue_floor),
        nullspace_relative_field_norm=float(nullspace_relative_field_norm),
    )


def _fixed_regularization_choice(
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    scale: float,
    config: CoilGenConfig,
    *,
    selection_method: str,
    fallback_reason: str = "",
    fixed_quadratic: np.ndarray | None = None,
) -> StreamFunctionSolution:
    factor = max(float(getattr(config, "tikonov_reg_factor", 1.0)), 0.0)
    lambda_value = factor * scale
    reduced_sf = _solve_for_regularization_lambda(ata, rtr, rhs, lambda_value, config, fixed_quadratic)
    diagnostics = {
        "selection_method": selection_method,
        "candidate_solver": "fixed",
        "fallback": bool(fallback_reason),
        "fallback_reason": fallback_reason,
        "factor_scale": scale,
        "selected_index": 0,
        "selected_factor": factor,
        "selected_lambda": lambda_value,
        "candidate_factors": [factor],
        "candidate_lambdas": [lambda_value],
        "residual_norms": [],
        "regularization_norms": [],
        "curvature": [],
    }
    return StreamFunctionSolution(reduced_sf, diagnostics)


def _l_curve_regularization_choice(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    scale: float,
    config: CoilGenConfig,
    *,
    fixed_quadratic: np.ndarray | None = None,
) -> StreamFunctionSolution:
    scan_method = _normalise_l_curve_scan_method(getattr(config, "l_curve_scan_method", "normal_equations"))
    factors, candidate_notes = _l_curve_candidate_factors(
        config,
        problem=problem,
        scale=scale,
        scan_method=scan_method,
        fixed_quadratic=fixed_quadratic,
    )
    lambdas = factors * scale
    scan = None
    with progress_scope(0, 80, "Scanning L-curve regularization candidates", label="L-curve"):
        if fixed_quadratic is not None and scan_method in {"auto", "transformed_svd"}:
            candidate_notes.append(
                "transformed_svd scan unavailable with fixed turn-spacing penalty; used normal_equations scan"
            )
        elif scan_method in {"auto", "transformed_svd"}:
            scan = _scan_l_curve_by_transformed_svd(problem, lambdas, config)
            if scan is None and scan_method == "transformed_svd":
                candidate_notes.append("transformed_svd scan failed; used normal_equations scan")
        if scan is None:
            scan = _scan_l_curve_by_normal_equations(problem, ata, rtr, rhs, lambdas, config, fixed_quadratic)
    residuals, regularization_norms, candidate_solver = scan
    smoothing_window = _l_curve_smoothing_window(config)
    selected_index, curvature, fallback_reason = _select_l_curve_knee(
        lambdas,
        residuals,
        regularization_norms,
        smoothing_window,
    )
    if selected_index >= 0:
        selected_index, guard_notes = _apply_auto_range_l_curve_selection_guards(
            problem,
            ata,
            rtr,
            rhs,
            factors,
            lambdas,
            selected_index,
            config,
            fixed_quadratic,
        )
        candidate_notes.extend(guard_notes)
    if selected_index < 0:
        selected_factor = max(float(getattr(config, "tikonov_reg_factor", 1.0)), 0.0)
        selected_lambda = selected_factor * scale
        reduced_sf = _solve_for_regularization_lambda(ata, rtr, rhs, selected_lambda, config, fixed_quadratic)
        estimated_contour_step = _estimated_contour_step_current_A(problem, reduced_sf, config)
        nearest_index = int(np.argmin(np.abs(factors - selected_factor))) if len(factors) else -1
        diagnostics = {
            "selection_method": "l_curve",
            "candidate_solver": candidate_solver,
            "l_curve_scan_method": scan_method,
            "l_curve_curvature_smoothing_window": smoothing_window,
            "regularization_auto_range": _regularization_auto_range_enabled(config),
            "fallback": True,
            "fallback_reason": fallback_reason or "fewer than three valid L-curve candidates",
            "candidate_notes": candidate_notes,
            "estimated_selected_contour_step_A": estimated_contour_step,
            "regularization_auto_range_max_contour_step_A": _nonnegative_config_value(
                config,
                "regularization_auto_range_max_contour_step_A",
                0.0,
            ),
            "factor_scale": scale,
            "selected_index": nearest_index,
            "selected_factor": selected_factor,
            "selected_lambda": selected_lambda,
            "candidate_factors": _float_list(factors),
            "candidate_lambdas": _float_list(lambdas),
            "residual_norms": _float_list(residuals),
            "regularization_norms": _float_list(regularization_norms),
            "curvature": _float_list(curvature),
        }
        return StreamFunctionSolution(reduced_sf, diagnostics)

    selected_lambda = float(lambdas[selected_index])
    selected_factor = float(factors[selected_index])
    reduced_sf = _solve_for_regularization_lambda(ata, rtr, rhs, selected_lambda, config, fixed_quadratic)
    estimated_contour_step = _estimated_contour_step_current_A(problem, reduced_sf, config)
    diagnostics = {
        "selection_method": "l_curve",
        "candidate_solver": candidate_solver,
        "l_curve_scan_method": scan_method,
        "l_curve_curvature_smoothing_window": smoothing_window,
        "regularization_auto_range": _regularization_auto_range_enabled(config),
        "fallback": False,
        "fallback_reason": "",
        "candidate_notes": candidate_notes,
        "estimated_selected_contour_step_A": estimated_contour_step,
        "regularization_auto_range_max_contour_step_A": _nonnegative_config_value(
            config,
            "regularization_auto_range_max_contour_step_A",
            0.0,
        ),
        "factor_scale": scale,
        "selected_index": int(selected_index),
        "selected_factor": selected_factor,
        "selected_lambda": selected_lambda,
        "candidate_factors": _float_list(factors),
        "candidate_lambdas": _float_list(lambdas),
        "residual_norms": _float_list(residuals),
        "regularization_norms": _float_list(regularization_norms),
        "curvature": _float_list(curvature),
    }
    return StreamFunctionSolution(reduced_sf, diagnostics)


def _apply_auto_range_l_curve_selection_guards(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    factors: np.ndarray,
    lambdas: np.ndarray,
    selected_index: int,
    config: CoilGenConfig,
    fixed_quadratic: np.ndarray | None,
) -> tuple[int, list[str]]:
    if not _regularization_auto_range_enabled(config):
        return selected_index, []
    max_contour_step = _nonnegative_config_value(config, "regularization_auto_range_max_contour_step_A", 0.0)
    if max_contour_step <= 0.0:
        return selected_index, []

    def candidate_step(index: int) -> float:
        try:
            candidate = _solve_for_regularization_lambda(
                ata,
                rtr,
                rhs,
                float(lambdas[index]),
                config,
                fixed_quadratic,
            )
        except np.linalg.LinAlgError:
            return float("nan")
        return _estimated_contour_step_current_A(problem, candidate, config)

    selected_step = candidate_step(selected_index)
    if np.isfinite(selected_step) and selected_step <= max_contour_step:
        return selected_index, []

    replacement_index = -1
    replacement_step = float("nan")
    last_finite_index = -1
    last_finite_step = float("nan")
    for index in range(selected_index + 1, len(lambdas)):
        step = candidate_step(index)
        if np.isfinite(step):
            last_finite_index = index
            last_finite_step = step
        if np.isfinite(step) and step <= max_contour_step:
            replacement_index = index
            replacement_step = step
            break
    if replacement_index >= 0:
        return replacement_index, [
            "auto-range contour-step guard raised selected factor "
            f"from {float(factors[selected_index]):.4g} to {float(factors[replacement_index]):.4g} "
            f"because estimated contour step {selected_step:.4g} A exceeded {max_contour_step:.4g} A "
            f"(guarded estimate {replacement_step:.4g} A)"
        ]
    if last_finite_index >= 0:
        return last_finite_index, [
            "auto-range contour-step guard selected the strongest tested regularization "
            f"factor {float(factors[last_finite_index]):.4g}; all tested estimates exceeded "
            f"{max_contour_step:.4g} A (best tested {last_finite_step:.4g} A)"
        ]
    return selected_index, [
        "auto-range contour-step guard could not evaluate candidate stream-function ranges; kept L-curve selection"
    ]


def _estimated_contour_step_current_A(
    problem: StreamFunctionOptimizationProblem,
    reduced_stream_function: np.ndarray,
    config: CoilGenConfig,
) -> float:
    try:
        levels = max(int(getattr(config, "levels", 1)), 1)
    except (TypeError, ValueError):
        levels = 1
    return _estimated_contour_step_current_for_level_count_A(problem, reduced_stream_function, config, levels)


def _estimated_contour_step_current_for_level_count_A(
    problem: StreamFunctionOptimizationProblem,
    reduced_stream_function: np.ndarray,
    config: CoilGenConfig,
    level_count: int,
) -> float:
    full_stream_function = problem.expand(reduced_stream_function)
    values = np.asarray(full_stream_function, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    percentile = _contour_step_range_percentile(config)
    if percentile >= 100.0:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    else:
        tail = 0.5 * (100.0 - percentile)
        lower, upper = (float(value) for value in np.percentile(finite, [tail, 100.0 - tail]))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower = float(np.min(finite))
            upper = float(np.max(finite))
    stream_range = max(float(upper - lower), 0.0)
    denominator = _contour_step_denominator(int(level_count), config)
    return float(stream_range / denominator)


def _contour_step_range_percentile(config: CoilGenConfig) -> float:
    try:
        value = float(getattr(config, "contour_level_range_percentile", 100.0))
    except (TypeError, ValueError):
        return 100.0
    if not np.isfinite(value):
        return 100.0
    return min(max(value, np.finfo(float).eps), 100.0)


def _normalise_l_curve_scan_method(method: str) -> str:
    normalised = str(method).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "normal": "normal_equations",
        "normal_equation": "normal_equations",
        "normal_equations": "normal_equations",
        "direct": "normal_equations",
        "svd": "transformed_svd",
        "transformed_svd": "transformed_svd",
    }
    if normalised not in aliases:
        raise ValueError("l_curve_scan_method must be 'normal_equations', 'transformed_svd', or 'auto'.")
    return aliases[normalised]


def _regularization_auto_range_enabled(config: CoilGenConfig) -> bool:
    value = getattr(config, "regularization_auto_range", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "auto"}
    return bool(value)


def _target_error_constraint_candidate_factors(
    config: CoilGenConfig,
    *,
    problem: StreamFunctionOptimizationProblem,
    scale: float,
    fixed_quadratic: np.ndarray | None = None,
) -> np.ndarray:
    count = max(int(getattr(config, "target_error_constraint_factor_count", 61)), 3)
    center = float(getattr(config, "tikonov_reg_factor", 1.0))
    if not np.isfinite(center) or center <= 0.0:
        center = 1.0
    min_multiplier = float(getattr(config, "target_error_constraint_factor_min_multiplier", 1e-10))
    max_multiplier = float(getattr(config, "target_error_constraint_factor_max_multiplier", 1e10))
    if not np.isfinite(min_multiplier) or min_multiplier <= 0.0:
        min_multiplier = 1e-10
    if not np.isfinite(max_multiplier) or max_multiplier <= min_multiplier:
        max_multiplier = 1e10
    lower_factor = center * min_multiplier
    upper_factor = center * max_multiplier
    configured = np.logspace(np.log10(lower_factor), np.log10(upper_factor), count)

    adaptive_range = None
    if fixed_quadratic is None:
        adaptive_range = _adaptive_l_curve_candidate_factors_from_svd(problem, scale, count, config)
    adaptive = adaptive_range.factors if adaptive_range is not None else None
    if _regularization_auto_range_enabled(config) and adaptive is not None and len(adaptive) >= 3:
        return _positive_finite_unique(adaptive)
    combined = configured
    if adaptive is not None and len(adaptive) >= 3:
        combined = np.r_[configured, adaptive]
    combined = np.asarray(combined, dtype=float)
    if _regularization_auto_range_enabled(config):
        bounded = _auto_range_bounded_candidate_factors(combined, config, count)
        if len(bounded) >= 3:
            return bounded
    valid = (
        np.isfinite(combined)
        & (combined > 0.0)
        & (combined >= lower_factor * (1.0 - 1.0e-12))
        & (combined <= upper_factor * (1.0 + 1.0e-12))
    )
    return np.unique(combined[valid])


def _positive_finite_unique(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array) & (array > 0.0)]
    return np.unique(valid)


def _auto_range_factor_bounds(config: CoilGenConfig) -> tuple[float, float]:
    if not _regularization_auto_range_enabled(config):
        return 0.0, 0.0
    return (
        _nonnegative_config_value(config, "regularization_auto_range_min_factor", 0.0),
        _nonnegative_config_value(config, "regularization_auto_range_max_factor", 0.0),
    )


def _regularization_auto_min_lambda(config: CoilGenConfig, scale: float) -> float:
    min_factor, _max_factor = _auto_range_factor_bounds(config)
    if min_factor <= 0.0 or not np.isfinite(scale) or scale <= 0.0:
        return 0.0
    return float(min_factor * scale)


def _auto_range_bounded_candidate_factors(
    values: np.ndarray | Sequence[float],
    config: CoilGenConfig,
    count: int,
) -> np.ndarray:
    factors = _positive_finite_unique(values)
    min_factor, max_factor = _auto_range_factor_bounds(config)
    if min_factor > 0.0:
        factors = factors[factors >= min_factor * (1.0 - 1.0e-12)]
    if max_factor > 0.0:
        factors = factors[factors <= max_factor * (1.0 + 1.0e-12)]
    if len(factors) >= 3:
        return factors

    all_values = _positive_finite_unique(values)
    lower = min_factor if min_factor > 0.0 else (float(all_values[0]) if len(all_values) else 1.0e-6)
    upper = max_factor if max_factor > 0.0 else (float(all_values[-1]) if len(all_values) else lower * 1.0e8)
    if not np.isfinite(lower) or lower <= 0.0:
        lower = 1.0e-6
    if not np.isfinite(upper) or upper <= lower:
        upper = lower * 1.0e8
    return np.logspace(np.log10(lower), np.log10(upper), max(int(count), 3))


def _apply_auto_range_contour_step_lambda_guard(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    lambdas: np.ndarray,
    selected_lambda: float,
    config: CoilGenConfig,
    fixed_quadratic: np.ndarray | None,
) -> tuple[float, list[str]]:
    if not _regularization_auto_range_enabled(config):
        return float(selected_lambda), []
    max_contour_step = _nonnegative_config_value(config, "regularization_auto_range_max_contour_step_A", 0.0)
    if max_contour_step <= 0.0:
        return float(selected_lambda), []

    selected_lambda = max(float(selected_lambda), 1.0e-300)

    def candidate_step(lambda_value: float) -> float:
        try:
            candidate = _solve_for_regularization_lambda(
                ata,
                rtr,
                rhs,
                float(lambda_value),
                config,
                fixed_quadratic,
            )
        except np.linalg.LinAlgError:
            return float("nan")
        return _estimated_contour_step_current_A(problem, candidate, config)

    selected_step = candidate_step(selected_lambda)
    if np.isfinite(selected_step) and selected_step <= max_contour_step:
        return selected_lambda, []

    tested = _positive_finite_unique(lambdas)
    larger = tested[tested > selected_lambda * (1.0 + 1.0e-12)]
    last_finite_lambda = selected_lambda
    last_finite_step = selected_step
    for lambda_value in larger:
        step = candidate_step(float(lambda_value))
        if np.isfinite(step):
            last_finite_lambda = float(lambda_value)
            last_finite_step = float(step)
        if np.isfinite(step) and step <= max_contour_step:
            return float(lambda_value), [
                "auto-range contour-step guard raised selected lambda "
                f"from {selected_lambda:.4g} to {float(lambda_value):.4g} "
                f"because estimated contour step {selected_step:.4g} A exceeded {max_contour_step:.4g} A "
                f"(guarded estimate {float(step):.4g} A)"
            ]

    lambda_value = max(float(tested[-1]) if len(tested) else selected_lambda, selected_lambda)
    for _ in range(24):
        lambda_value *= 10.0
        step = candidate_step(lambda_value)
        if np.isfinite(step):
            last_finite_lambda = float(lambda_value)
            last_finite_step = float(step)
        if np.isfinite(step) and step <= max_contour_step:
            return float(lambda_value), [
                "auto-range contour-step guard expanded above scan maximum "
                f"to lambda {float(lambda_value):.4g} because estimated contour step "
                f"{selected_step:.4g} A exceeded {max_contour_step:.4g} A "
                f"(guarded estimate {float(step):.4g} A)"
            ]

    if np.isfinite(last_finite_step) and last_finite_lambda > selected_lambda:
        return float(last_finite_lambda), [
            "auto-range contour-step guard selected the strongest evaluated lambda "
            f"{last_finite_lambda:.4g}; all estimates exceeded {max_contour_step:.4g} A "
            f"(best evaluated {last_finite_step:.4g} A)"
        ]
    return selected_lambda, [
        "auto-range contour-step guard could not evaluate candidate stream-function ranges; kept target-error selection"
    ]


def _augment_regularization_scan_with_selected_point(
    factors: np.ndarray,
    lambdas: np.ndarray,
    residuals: np.ndarray,
    regularization_norms: np.ndarray,
    selected_factor: float,
    selected_lambda: float,
    selected_residual: float,
    selected_regularization_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    factors_out = np.r_[np.asarray(factors, dtype=float), float(selected_factor)]
    lambdas_out = np.r_[np.asarray(lambdas, dtype=float), float(selected_lambda)]
    residuals_out = np.r_[np.asarray(residuals, dtype=float), float(selected_residual)]
    regularization_out = np.r_[np.asarray(regularization_norms, dtype=float), float(selected_regularization_norm)]
    order = np.argsort(lambdas_out)
    factors_out = factors_out[order]
    lambdas_out = lambdas_out[order]
    residuals_out = residuals_out[order]
    regularization_out = regularization_out[order]
    selected_candidates = np.flatnonzero(order == len(order) - 1)
    selected_index = (
        int(selected_candidates[0])
        if len(selected_candidates)
        else int(np.argmin(np.abs(lambdas_out - selected_lambda)))
    )
    return factors_out, lambdas_out, residuals_out, regularization_out, selected_index


def _select_error_constrained_lambda(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    lambdas: np.ndarray,
    residuals: np.ndarray,
    regularization_norms: np.ndarray,
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    config: CoilGenConfig,
    fixed_quadratic: np.ndarray | None,
    *,
    minimum_allowed_lambda: float = 0.0,
) -> tuple[int, float, str, bool]:
    finite = np.isfinite(lambdas) & np.isfinite(residuals) & (lambdas > 0.0)
    if not np.any(finite):
        return -1, 0.0, "no finite lambda candidates", False
    candidate_indices = np.flatnonzero(finite)
    sorted_order = candidate_indices[np.argsort(lambdas[candidate_indices])]
    sorted_lambdas = np.asarray(lambdas[sorted_order], dtype=float)
    tolerance = 0.0
    met_values: list[bool] = []
    violation_ratios: list[float] = []
    for lambda_value in sorted_lambdas:
        try:
            candidate = _solve_for_regularization_lambda(
                ata,
                rtr,
                rhs,
                float(lambda_value),
                config,
                fixed_quadratic,
            )
        except np.linalg.LinAlgError:
            met_values.append(False)
            violation_ratios.append(float("inf"))
            continue
        met_values.append(_constraint_blocks_met(constraint_blocks, candidate, tolerance=tolerance))
        violation_ratios.append(_constraint_violation_ratio(constraint_blocks, candidate))
    met = np.asarray(met_values, dtype=bool)
    violation_array = np.asarray(violation_ratios, dtype=float)
    if not np.any(met):
        min_lambda = float(minimum_allowed_lambda) if np.isfinite(minimum_allowed_lambda) else 0.0
        expanded_low = None
        expansion_reason = (
            "expanded below configured scan minimum and selected largest multiplier within target error budget"
        )
        if min_lambda > 0.0:
            if min_lambda < float(sorted_lambdas[0]) * (1.0 - 1.0e-12):
                try:
                    min_candidate = _solve_for_regularization_lambda(
                        ata,
                        rtr,
                        rhs,
                        min_lambda,
                        config,
                        fixed_quadratic,
                    )
                except np.linalg.LinAlgError:
                    min_candidate = None
                if min_candidate is not None and _constraint_blocks_met(
                    constraint_blocks,
                    min_candidate,
                    tolerance=0.0,
                ):
                    expanded_low = min_lambda
                    expansion_reason = (
                        "expanded down to regularization_auto_range_min_factor and selected largest "
                        "multiplier within target error budget"
                    )
        else:
            expanded_low = _shrink_lambda_until_constraint_met(
                problem,
                ata,
                rtr,
                rhs,
                float(sorted_lambdas[0]),
                constraint_blocks,
                config,
                fixed_quadratic,
            )
        if expanded_low is not None:
            refined_lambda = _refine_error_constrained_lambda(
                problem,
                ata,
                rtr,
                rhs,
                float(expanded_low),
                float(sorted_lambdas[0]),
                constraint_blocks,
                config,
                fixed_quadratic,
            )
            return (
                int(sorted_order[0]),
                refined_lambda,
                expansion_reason,
                True,
            )
        fallback_local, fallback_reason = _select_infeasible_target_error_fallback(
            sorted_lambdas,
            np.asarray(residuals[sorted_order], dtype=float),
            np.asarray(regularization_norms[sorted_order], dtype=float),
            violation_array,
            config,
        )
        if min_lambda > 0.0:
            fallback_reason = (
                f"{fallback_reason}; regularization_auto_range_min_factor prevented expansion below "
                f"lambda {min_lambda:.4g}"
            )
        return int(sorted_order[fallback_local]), float(sorted_lambdas[fallback_local]), fallback_reason, False

    last_met_local = int(np.flatnonzero(met)[-1])
    if last_met_local == len(sorted_lambdas) - 1:
        lambda_value, reason, crossed = _grow_lambda_until_constraint_crosses(
            problem,
            ata,
            rtr,
            rhs,
            float(sorted_lambdas[last_met_local]),
            constraint_blocks,
            config,
            fixed_quadratic,
        )
        return int(sorted_order[last_met_local]), lambda_value, reason, bool(crossed)

    low_lambda = float(sorted_lambdas[last_met_local])
    high_lambda = float(sorted_lambdas[last_met_local + 1])
    refined_lambda = _refine_error_constrained_lambda(
        problem,
        ata,
        rtr,
        rhs,
        low_lambda,
        high_lambda,
        constraint_blocks,
        config,
        fixed_quadratic,
    )
    return (
        int(sorted_order[last_met_local]),
        refined_lambda,
        "selected largest multiplier within target error budget",
        True,
    )


def _select_infeasible_target_error_fallback(
    sorted_lambdas: np.ndarray,
    sorted_residuals: np.ndarray,
    sorted_regularization_norms: np.ndarray,
    violation_ratios: np.ndarray,
    config: CoilGenConfig,
) -> tuple[int, str]:
    strategy = _normalise_target_error_infeasible_fallback(config)
    finite_violation = np.isfinite(violation_ratios)
    if not np.any(finite_violation):
        return 0, "requested target error is infeasible and no finite fallback candidate was available"

    best_local = int(np.nanargmin(np.where(finite_violation, violation_ratios, np.inf)))
    if strategy in {"minimum_residual", "best_residual"}:
        return best_local, "requested target error is infeasible; selected minimum-residual fallback candidate"

    valid = (
        finite_violation
        & np.isfinite(sorted_lambdas)
        & np.isfinite(sorted_residuals)
        & np.isfinite(sorted_regularization_norms)
        & (sorted_lambdas > 0.0)
        & (sorted_residuals > 0.0)
        & (sorted_regularization_norms > 0.0)
    )
    if not np.any(valid):
        return best_local, "requested target error is infeasible; selected minimum-residual fallback candidate"

    if strategy == "l_curve":
        local_index, _curvature, reason = _select_l_curve_knee(
            sorted_lambdas,
            sorted_residuals,
            sorted_regularization_norms,
            smoothing_window=_l_curve_smoothing_window(config),
        )
        if local_index >= 0:
            return int(local_index), "requested target error is infeasible; selected L-curve fallback candidate"
        strategy = "residual_floor"

    margin = _target_error_infeasible_residual_margin(config)
    best_violation = float(violation_ratios[best_local])
    threshold = best_violation * (1.0 + margin / 100.0)
    floor_candidates = np.flatnonzero(valid & (violation_ratios <= threshold))
    if len(floor_candidates):
        # Among candidates close to the best achievable residual, choose the
        # largest regularization multiplier. This suppresses high-current,
        # high-spatial-frequency stream functions when the requested error
        # budget is physically infeasible.
        selected = int(floor_candidates[np.argmax(sorted_lambdas[floor_candidates])])
        return (
            selected,
            "requested target error is infeasible; selected largest regularization "
            f"within {margin:.3g}% of the best residual fallback candidate",
        )

    return best_local, "requested target error is infeasible; selected minimum-residual fallback candidate"


def _shrink_lambda_until_constraint_met(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    start_lambda: float,
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    config: CoilGenConfig,
    fixed_quadratic: np.ndarray | None,
) -> float | None:
    lambda_value = max(float(start_lambda), 1.0e-300)
    for _ in range(80):
        lambda_value = max(lambda_value * 0.1, 1.0e-300)
        try:
            candidate = _solve_for_regularization_lambda(ata, rtr, rhs, lambda_value, config, fixed_quadratic)
        except np.linalg.LinAlgError:
            if lambda_value <= 1.0e-300:
                return None
            continue
        if _constraint_blocks_met(constraint_blocks, candidate, tolerance=0.0):
            return float(lambda_value)
        if lambda_value <= 1.0e-300:
            return None
    return None


def _normalise_target_error_infeasible_fallback(config: CoilGenConfig) -> str:
    raw = str(getattr(config, "target_error_infeasible_fallback", "residual_floor")).strip().lower().replace("-", "_")
    aliases = {
        "minimum_residual": "minimum_residual",
        "min_residual": "minimum_residual",
        "best_residual": "best_residual",
        "residual_floor": "residual_floor",
        "near_minimum_residual": "residual_floor",
        "near_minimum": "residual_floor",
        "l_curve": "l_curve",
        "lcurve": "l_curve",
    }
    if raw not in aliases:
        raise ValueError("target_error_infeasible_fallback must be 'minimum_residual', 'residual_floor', or 'l_curve'.")
    return aliases[raw]


def _target_error_infeasible_residual_margin(config: CoilGenConfig) -> float:
    return max(float(getattr(config, "target_error_infeasible_residual_margin_percent", 100.0)), 0.0)


def _grow_lambda_until_constraint_crosses(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    start_lambda: float,
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    config: CoilGenConfig,
    fixed_quadratic: np.ndarray | None,
) -> tuple[float, str, bool]:
    low_lambda = max(float(start_lambda), 1.0e-300)
    high_lambda = low_lambda
    tolerance = 0.0
    for _ in range(24):
        high_lambda *= 10.0
        try:
            candidate = _solve_for_regularization_lambda(ata, rtr, rhs, high_lambda, config, fixed_quadratic)
        except np.linalg.LinAlgError:
            break
        constraints_met = _constraint_blocks_met(constraint_blocks, candidate, tolerance=tolerance)
        violation_ratio = _constraint_violation_ratio(constraint_blocks, candidate)
        if not constraints_met and np.isfinite(violation_ratio):
            refined = _refine_error_constrained_lambda(
                problem,
                ata,
                rtr,
                rhs,
                low_lambda,
                high_lambda,
                constraint_blocks,
                config,
                fixed_quadratic,
            )
            return refined, "selected largest multiplier within target error budget", True
        if np.isfinite(violation_ratio):
            low_lambda = high_lambda
    return low_lambda, "upper scan bound still met the target error budget; selected largest tested multiplier", True


def _refine_error_constrained_lambda(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    low_lambda: float,
    high_lambda: float,
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    config: CoilGenConfig,
    fixed_quadratic: np.ndarray | None,
) -> float:
    low = max(float(low_lambda), 1.0e-300)
    high = max(float(high_lambda), low * (1.0 + 1.0e-12))
    tolerance = 0.0
    for _ in range(48):
        middle = float(np.exp(0.5 * (np.log(low) + np.log(high))))
        try:
            candidate = _solve_for_regularization_lambda(ata, rtr, rhs, middle, config, fixed_quadratic)
        except np.linalg.LinAlgError:
            high = middle
            continue
        if _constraint_blocks_met(constraint_blocks, candidate, tolerance=tolerance):
            low = middle
        else:
            high = middle
    return low


def _l_curve_candidate_factors(
    config: CoilGenConfig,
    *,
    problem: StreamFunctionOptimizationProblem | None = None,
    scale: float = 1.0,
    scan_method: str = "normal_equations",
    fixed_quadratic: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    explicit = np.asarray(tuple(getattr(config, "l_curve_regularization_factors", ())), dtype=float)
    notes: list[str] = []
    auto_range = _regularization_auto_range_enabled(config)
    if explicit.size:
        factors = np.unique(explicit[np.isfinite(explicit) & (explicit > 0.0)])
        if factors.size:
            if auto_range:
                notes.append("explicit l_curve_regularization_factors override regularization_auto_range")
            return np.sort(factors), notes
        notes.append("explicit candidate factors contained no positive finite values")

    center = float(getattr(config, "tikonov_reg_factor", 1.0))
    if not np.isfinite(center) or center <= 0.0:
        notes.append("non-positive tikonov_reg_factor replaced by scan center 1.0")
        center = 1.0
    count = max(int(getattr(config, "l_curve_factor_count", 21)), 3)
    min_multiplier = float(getattr(config, "l_curve_factor_min_multiplier", 1e-4))
    max_multiplier = float(getattr(config, "l_curve_factor_max_multiplier", 1e4))
    if not np.isfinite(min_multiplier) or min_multiplier <= 0.0:
        notes.append("invalid l_curve_factor_min_multiplier replaced by 1e-4")
        min_multiplier = 1e-4
    if not np.isfinite(max_multiplier) or max_multiplier <= min_multiplier:
        notes.append("invalid l_curve_factor_max_multiplier replaced by 1e4")
        max_multiplier = 1e4
    centered = np.unique(center * np.logspace(np.log10(min_multiplier), np.log10(max_multiplier), count))
    use_adaptive_range = auto_range or scan_method in {"auto", "transformed_svd"}
    if problem is None or not use_adaptive_range:
        return centered, notes
    if fixed_quadratic is not None:
        notes.append(
            "generalized singular-value range unavailable with fixed quadratic penalty; used configured factor range"
        )
        return centered, notes

    adaptive_range = _adaptive_l_curve_candidate_factors_from_svd(problem, scale, count, config)
    if adaptive_range is None:
        notes.append("generalized singular-value regularization range unavailable; used configured factor range")
        if auto_range:
            bounded = _auto_range_bounded_candidate_factors(centered, config, count)
            if len(bounded) >= 3:
                notes.append(
                    "regularization_auto_range_min_factor/max_factor bounded the configured fallback factor range"
                )
                return bounded, notes
        return centered, notes
    adaptive = adaptive_range.factors
    rank_note = ""
    if adaptive_range.regularization_rank > 0:
        rank_note = (
            f"; R rank {adaptive_range.regularization_rank}/"
            f"{adaptive_range.regularization_rank + adaptive_range.regularization_nullity}"
        )
        if adaptive_range.regularization_nullity:
            rank_note += f", nullspace field coupling {adaptive_range.regularization_nullspace_relative_field_norm:.3g}"
    notes.append(
        "generalized singular-value regularization range used "
        f"(factor range {float(adaptive[0]):.4g} to {float(adaptive[-1]):.4g}; "
        f"retained {adaptive_range.retained_singular_value_count}/{adaptive_range.singular_value_count} modes; "
        f"singular floor {adaptive_range.singular_value_floor:.4g}{rank_note})"
    )
    return np.unique(adaptive), notes


def _adaptive_l_curve_candidate_factors_from_svd(
    problem: StreamFunctionOptimizationProblem,
    scale: float,
    count: int,
    config: CoilGenConfig,
) -> AdaptiveRegularizationRange | None:
    """Build a problem-scaled interval from generalized singular values of (A, R).

    The resistance/current quadratic R is often positive semidefinite because a
    constant stream-function gauge produces no current. The adaptive range must
    therefore use the positive eigenspace of R instead of requiring a Cholesky
    factor of a strictly positive definite matrix.
    """

    if not np.isfinite(scale) or scale <= 0.0:
        return None
    progress = ProgressReporter.from_config(config, label="L-curve")
    progress.report(None, "factoring resistance to estimate the regularization candidate range")
    whitened = _regularization_whitened_sensitivity(problem)
    if whitened is None:
        return None
    try:
        progress.report(None, "computing the candidate-range SVD; factorization completion is not available")
        u, singular_values, _vt = np.linalg.svd(whitened.transformed_sensitivity, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    singular_values = np.asarray(singular_values, dtype=float)
    valid = np.isfinite(singular_values) & (singular_values > 0.0)
    if np.count_nonzero(valid) < 1:
        return None
    valid_indices = np.flatnonzero(valid)
    singular_values = singular_values[valid]
    projected = np.ones_like(singular_values)
    if valid_indices.size and u.shape[1] > int(valid_indices[-1]):
        projected = np.abs(u[:, valid_indices].T @ np.asarray(problem.target_z, dtype=float))
    spectral_max_all = float(np.max(singular_values))
    singular_floor = spectral_max_all * _positive_config_value(
        config,
        "regularization_auto_range_min_singular_relative",
        1.0e-4,
    )
    projection_floor = (
        float(np.max(projected))
        * _positive_config_value(config, "regularization_auto_range_target_projection_relative", 1.0e-4)
        if projected.size and np.any(projected > 0.0)
        else 0.0
    )
    trustworthy = singular_values >= singular_floor
    target_relevant = projected >= projection_floor if projection_floor > 0.0 else np.ones_like(trustworthy, dtype=bool)
    retained_mask = trustworthy & target_relevant
    if not np.any(retained_mask):
        retained_mask = trustworthy
    if not np.any(retained_mask):
        retained_mask = np.zeros_like(singular_values, dtype=bool)
        retained_mask[0] = True
    retained = singular_values[retained_mask]
    spectral_max = float(np.max(retained))
    spectral_min = float(np.min(retained))
    if not np.isfinite(spectral_max) or spectral_max <= 0.0:
        return None
    spectral_min = max(spectral_min, spectral_max * 1.0e-12)
    lambda_min = spectral_min * _positive_config_value(config, "regularization_auto_range_lower_padding", 5.0e-2)
    lambda_max = spectral_max * _positive_config_value(config, "regularization_auto_range_upper_padding", 1.0e2)
    if not np.isfinite(lambda_min) or not np.isfinite(lambda_max) or lambda_min <= 0.0 or lambda_max <= lambda_min:
        return None
    factor_min = float(lambda_min / scale)
    factor_max = float(lambda_max / scale)
    configured_min_factor = _nonnegative_config_value(config, "regularization_auto_range_min_factor", 0.0)
    configured_max_factor = _nonnegative_config_value(config, "regularization_auto_range_max_factor", 0.0)
    if configured_min_factor > 0.0:
        factor_min = max(factor_min, configured_min_factor)
    if configured_max_factor > 0.0:
        factor_max = min(factor_max, configured_max_factor)
    if not np.isfinite(factor_min) or not np.isfinite(factor_max) or factor_min <= 0.0 or factor_max <= factor_min:
        return None
    factors = np.logspace(np.log10(factor_min), np.log10(factor_max), max(int(count), 3))
    return AdaptiveRegularizationRange(
        factors=factors,
        singular_value_count=int(len(singular_values)),
        retained_singular_value_count=int(np.count_nonzero(retained_mask)),
        singular_value_max=spectral_max_all,
        singular_value_min_retained=spectral_min,
        singular_value_floor=float(singular_floor),
        target_projection_floor=float(projection_floor),
        factor_min=float(factor_min),
        factor_max=float(factor_max),
        regularization_rank=int(whitened.regularization_rank),
        regularization_nullity=int(whitened.regularization_nullity),
        regularization_eigenvalue_floor=float(whitened.eigenvalue_floor),
        regularization_nullspace_relative_field_norm=float(whitened.nullspace_relative_field_norm),
    )


def _positive_config_value(config: CoilGenConfig, name: str, default: float) -> float:
    try:
        value = float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value) or value <= 0.0:
        return float(default)
    return value


def _l_curve_smoothing_window(config: CoilGenConfig) -> int:
    try:
        window = int(getattr(config, "l_curve_curvature_smoothing_window", 5))
    except (TypeError, ValueError):
        return 5
    return max(window, 1)


def _scan_l_curve_by_transformed_svd(
    problem: StreamFunctionOptimizationProblem,
    lambdas: np.ndarray,
    config: CoilGenConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    reporter = ProgressReporter.from_config(config, label="L-curve")
    reporter.report(None, "factoring the resistance operator for the transformed SVD")
    regularization_operator = _regularization_operator_from_quadratic(problem.reduced_resistance)
    if regularization_operator is None:
        return None
    try:
        reporter.report(None, "transforming the sensitivity matrix")
        transformed = np.linalg.solve(regularization_operator.T, problem.reduced_sensitivity.T).T
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(transformed)):
        return None
    try:
        reporter.report(None, "computing the sensitivity SVD; factorization completion is not available")
        u, singular_values, _vt = np.linalg.svd(transformed, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(singular_values)):
        return None

    projected_target = u.T @ problem.target_z
    residuals = np.full(len(lambdas), np.nan, dtype=float)
    regularization_norms = np.full(len(lambdas), np.nan, dtype=float)
    task = reporter.task("regularization candidates", len(lambdas))
    for ind, lambda_value in enumerate(lambdas):
        denom = singular_values * singular_values + lambda_value * lambda_value
        y_coefficients = (singular_values / np.maximum(denom, 1e-300)) * projected_target
        fitted_target = u @ (singular_values * y_coefficients)
        residuals[ind] = float(np.linalg.norm(fitted_target - problem.target_z))
        regularization_norms[ind] = float(np.linalg.norm(y_coefficients))
        task.advance()
    task.done()
    if np.count_nonzero(_valid_l_curve_points(residuals, regularization_norms)) < 3:
        return None
    return residuals, regularization_norms, "transformed_svd"


def _scan_l_curve_by_normal_equations(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    lambdas: np.ndarray,
    config: CoilGenConfig,
    fixed_quadratic: np.ndarray | None = None,
    *,
    progress_label: str = "L-curve",
) -> tuple[np.ndarray, np.ndarray, str]:
    residuals = np.full(len(lambdas), np.nan, dtype=float)
    regularization_norms = np.full(len(lambdas), np.nan, dtype=float)
    task = ProgressReporter.from_config(config, label=progress_label).task("regularization candidates", len(lambdas))
    for ind, lambda_value in enumerate(lambdas):
        try:
            reduced_sf = _solve_for_regularization_lambda(
                ata,
                rtr,
                rhs,
                float(lambda_value),
                config,
                fixed_quadratic,
            )
        except np.linalg.LinAlgError:
            task.advance(1)
            continue
        if np.all(np.isfinite(reduced_sf)):
            residuals[ind] = problem.residual_norm(reduced_sf)
            regularization_norms[ind] = problem.resistance_norm(reduced_sf)
        task.advance(1)
    task.done()
    return residuals, regularization_norms, "normal_equations"


def _select_l_curve_knee(
    lambdas: np.ndarray,
    residuals: np.ndarray,
    regularization_norms: np.ndarray,
    smoothing_window: int = 5,
) -> tuple[int, np.ndarray, str]:
    valid = _valid_l_curve_points(residuals, regularization_norms)
    curvature = np.full(len(residuals), np.nan, dtype=float)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 3:
        return -1, curvature, "fewer than three valid positive L-curve points"

    log_lambda = np.log(np.asarray(lambdas, dtype=float)[valid_indices])
    log_residual = _smooth_l_curve_trace(np.log(residuals[valid_indices]), smoothing_window)
    log_regularization = _smooth_l_curve_trace(np.log(regularization_norms[valid_indices]), smoothing_window)
    if not np.all(np.isfinite(log_lambda)) or len(np.unique(log_lambda)) < 3:
        return _select_l_curve_ideal_corner(residuals, regularization_norms, valid), curvature, "invalid lambda spacing"

    dx = np.gradient(log_residual, log_lambda)
    dy = np.gradient(log_regularization, log_lambda)
    ddx = np.gradient(dx, log_lambda)
    ddy = np.gradient(dy, log_lambda)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    valid_curvature = np.abs(dx * ddy - dy * ddx) / np.maximum(denominator, 1e-300)
    valid_curvature[[0, -1]] = np.nan
    curvature[valid_indices] = valid_curvature

    span_reason = _l_curve_insufficient_span_reason(log_residual, log_regularization)
    if span_reason:
        return -1, curvature, span_reason

    selection_score = _l_curve_knee_selection_score(
        valid_curvature,
        log_residual,
        log_regularization,
        dx,
        dy,
    )
    first_strong_knee_index = _select_first_strong_l_curve_knee_index(selection_score)
    if first_strong_knee_index is not None:
        return int(valid_indices[first_strong_knee_index]), curvature, ""

    geometric_corner_index = _select_l_curve_geometric_corner_index(
        log_residual,
        log_regularization,
        dx,
        dy,
    )
    if geometric_corner_index is not None:
        return int(valid_indices[geometric_corner_index]), curvature, ""

    if np.any(np.isfinite(selection_score)):
        selected_valid_index = int(np.nanargmax(selection_score))
        return int(valid_indices[selected_valid_index]), curvature, ""
    return _select_l_curve_ideal_corner(residuals, regularization_norms, valid), curvature, "zero discrete curvature"


def _select_first_strong_l_curve_knee_index(selection_score: np.ndarray) -> int | None:
    """Return the first robust L-curve knee instead of the late collapse knee.

    Active-shielded inverse problems can contain two bends: first, the useful
    onset where regularization starts reducing current norm while residual is
    still close to its minimum; second, a late overregularized collapse where
    current norm vanishes and residual grows. The first bend is the usable
    Tikhonov knee for contour generation. Numerical plateau wiggles have already
    been masked by `_l_curve_knee_selection_score`, so here we choose the first
    local maximum whose score is comparable to the best candidate.
    """

    score = np.asarray(selection_score, dtype=float)
    finite = np.isfinite(score) & (score > 0.0)
    if not np.any(finite):
        return None

    max_score = float(np.nanmax(score[finite]))
    if not np.isfinite(max_score) or max_score <= 0.0:
        return None

    strong = finite & (score >= 0.75 * max_score)
    if not np.any(strong):
        return None

    previous_score = np.r_[-np.inf, score[:-1]]
    next_score = np.r_[score[1:], -np.inf]
    previous_score[~np.isfinite(previous_score)] = -np.inf
    next_score[~np.isfinite(next_score)] = -np.inf
    local_peak = strong & (score >= previous_score) & (score >= next_score)
    if np.any(local_peak):
        return int(np.flatnonzero(local_peak)[0])
    return int(np.flatnonzero(strong)[0])


def _select_l_curve_geometric_corner_index(
    log_residual: np.ndarray,
    log_regularization: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> int | None:
    """Select the geometric L-curve corner by distance from the endpoint chord.

    The raw curvature maximum is useful for plotting, but it is fragile for
    active-shielded designs where a long flat residual plateau is followed by a
    broad current-norm collapse. The endpoint-chord distance is the stable
    triangle criterion used for noisy L-curves: it selects the point where the
    log-residual/log-norm trace bows furthest away from the line connecting the
    underregularized and overregularized endpoints.
    """

    residual_position = _normalised_l_curve_coordinate(log_residual)
    regularization_position = _normalised_l_curve_coordinate(log_regularization)
    if residual_position is None or regularization_position is None:
        return None
    points = np.column_stack((residual_position, regularization_position))
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if not np.isfinite(chord_length) or chord_length <= 1.0e-12:
        return None

    signed_distance = (points[:, 0] - points[0, 0]) * chord[1] - (points[:, 1] - points[0, 1]) * chord[0]
    signed_distance = -signed_distance / chord_length
    if np.nanmax(signed_distance) < abs(float(np.nanmin(signed_distance))):
        signed_distance = -signed_distance

    speed = np.sqrt(np.asarray(dx, dtype=float) ** 2 + np.asarray(dy, dtype=float) ** 2)
    mask = np.isfinite(signed_distance)
    edge_guard = max(1, int(np.ceil(0.03 * len(signed_distance))))
    if len(signed_distance) > 2 * edge_guard:
        mask[:edge_guard] = False
        mask[-edge_guard:] = False
    if np.any(np.isfinite(speed)):
        mask &= speed > max(float(np.nanmax(speed)) * 1.0e-3, 1.0e-12)
    margin = 0.02
    mask &= residual_position > margin
    mask &= residual_position < 1.0 - margin
    mask &= regularization_position > margin
    mask &= regularization_position < 1.0 - margin
    mask &= signed_distance > 0.0
    if not np.any(mask):
        return None

    score = np.where(mask, signed_distance, np.nan)
    selected = int(np.nanargmax(score))
    if not np.isfinite(score[selected]) or float(score[selected]) < 0.03:
        return None
    return selected


def _l_curve_knee_selection_score(
    curvature: np.ndarray,
    log_residual: np.ndarray,
    log_regularization: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    """Score reliable L-curve knee candidates.

    Raw curvature is unstable on flat L-curve plateaus because the denominator
    contains the parametric speed cubed. Scenario A active shielding exposed this
    failure: tiny numerical wiggles at the low-regularization end were selected
    as the "knee". The score keeps the curvature diagnostic unchanged, but masks
    boundary/plateau/low-speed candidates before selecting the maximum.
    """

    curvature = np.asarray(curvature, dtype=float)
    score = np.full(curvature.shape, np.nan, dtype=float)
    finite = np.isfinite(curvature)
    if np.count_nonzero(finite) == 0:
        return score

    speed = np.sqrt(np.asarray(dx, dtype=float) ** 2 + np.asarray(dy, dtype=float) ** 2)
    residual_position = _normalised_l_curve_coordinate(log_residual)
    regularization_position = _normalised_l_curve_coordinate(log_regularization)
    if residual_position is None or regularization_position is None:
        return score

    interior = finite.copy()
    edge_guard = max(1, int(np.ceil(0.03 * len(curvature))))
    if len(curvature) > 2 * edge_guard:
        interior[:edge_guard] = False
        interior[-edge_guard:] = False
    if np.any(np.isfinite(speed)):
        speed_threshold = max(float(np.nanmax(speed)) * 1.0e-3, 1.0e-12)
        interior &= speed > speed_threshold

    # Exclude the underregularized and overregularized plateaus. A real knee is
    # where the field residual and resistance seminorm both participate in the
    # trade-off, not where one axis is numerically flat.
    tradeoff_margin = 0.02
    interior &= residual_position > tradeoff_margin
    interior &= residual_position < 1.0 - tradeoff_margin
    interior &= regularization_position > tradeoff_margin
    interior &= regularization_position < 1.0 - tradeoff_margin

    if not np.any(interior):
        interior = finite.copy()
        if len(curvature) > 2 * edge_guard:
            interior[:edge_guard] = False
            interior[-edge_guard:] = False

    balance = np.sqrt(
        np.maximum(
            residual_position * (1.0 - residual_position) * regularization_position * (1.0 - regularization_position),
            0.0,
        )
    )
    score[interior] = curvature[interior] * balance[interior]
    return score


def _normalised_l_curve_coordinate(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    value_range = float(np.ptp(values))
    if value_range <= 1.0e-30:
        return None
    return (values - float(np.min(values))) / value_range


def _l_curve_insufficient_span_reason(log_residual: np.ndarray, log_regularization: np.ndarray) -> str:
    residual_span = float(np.ptp(np.asarray(log_residual, dtype=float)))
    regularization_span = float(np.ptp(np.asarray(log_regularization, dtype=float)))
    min_log_span = 1.0e-3
    if residual_span < min_log_span or regularization_span < min_log_span:
        return (
            "L-curve scan has insufficient dynamic range "
            f"(log residual span {residual_span:.3g}, log regularization span {regularization_span:.3g}); "
            "using fixed regularization factor"
        )
    return ""


def _smooth_l_curve_trace(values: np.ndarray, smoothing_window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = int(max(smoothing_window, 1))
    if window < 3 or len(values) < 5:
        return values.copy()
    if window % 2 == 0:
        window += 1
    if window > len(values):
        window = len(values) if len(values) % 2 == 1 else len(values) - 1
    if window < 3:
        return values.copy()
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _select_l_curve_ideal_corner(residuals: np.ndarray, regularization_norms: np.ndarray, valid: np.ndarray) -> int:
    valid_indices = np.flatnonzero(valid)
    log_residual = np.log(residuals[valid_indices])
    log_regularization = np.log(regularization_norms[valid_indices])
    residual_range = max(float(np.ptp(log_residual)), 1e-30)
    regularization_range = max(float(np.ptp(log_regularization)), 1e-30)
    residual_score = (log_residual - float(np.min(log_residual))) / residual_range
    regularization_score = (log_regularization - float(np.min(log_regularization))) / regularization_range
    return int(
        valid_indices[int(np.argmin(residual_score * residual_score + regularization_score * regularization_score))]
    )


def _valid_l_curve_points(residuals: np.ndarray, regularization_norms: np.ndarray) -> np.ndarray:
    return np.isfinite(residuals) & np.isfinite(regularization_norms) & (residuals > 0.0) & (regularization_norms > 0.0)


def _solve_for_regularization_lambda(
    ata: np.ndarray,
    rtr: np.ndarray,
    rhs: np.ndarray,
    lambda_value: float,
    config: CoilGenConfig,
    fixed_quadratic: np.ndarray | None = None,
) -> np.ndarray:
    lhs = ata + float(lambda_value) * float(lambda_value) * rtr
    if fixed_quadratic is not None:
        lhs = lhs + fixed_quadratic
    return _solve_stream_function_system(lhs, rhs, config)


def _stream_function_smoothness_context(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> dict[str, Any]:
    factor = _safe_positive_float(getattr(config, "stream_function_smoothness_factor", 0.0))
    metric = problem.reduced_smoothness_metric
    enabled = factor > 0.0 and metric is not None and metric.size > 0
    if not enabled:
        return {
            "enabled": False,
            "factor": factor,
            "quadratic": None,
            "matrix_scale": 0.0,
            "normalization": "resistance",
        }
    resistance_scale = _matrix_mean_abs_diag(problem.reduced_resistance)
    metric_scale = _matrix_mean_abs_diag(metric)
    matrix_scale = resistance_scale / metric_scale if resistance_scale > 0.0 and metric_scale > 0.0 else 1.0
    return {
        "enabled": True,
        "factor": factor,
        "quadratic": factor * matrix_scale * metric,
        "matrix_scale": float(matrix_scale),
        "normalization": "resistance",
    }


def _stream_function_smoothness_diagnostics(smoothness: dict[str, Any]) -> dict[str, Any]:
    return {
        "stream_function_smoothness_enabled": bool(smoothness["enabled"]),
        "stream_function_smoothness_factor": float(smoothness["factor"]),
        "stream_function_smoothness_matrix_scale": float(smoothness["matrix_scale"]),
        "stream_function_smoothness_normalization": str(smoothness["normalization"]),
        "stream_function_smoothness_operator": "cotangent_fem_biharmonic",
    }


def _turn_spacing_penalty_context(
    problem: StreamFunctionOptimizationProblem,
    ata: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, Any]:
    threshold = _turn_spacing_threshold(config)
    factor = _turn_spacing_regularization_factor(config)
    growth = _safe_positive_float(getattr(config, "turn_spacing_penalty_growth", 10.0))
    if growth <= 1.0:
        growth = 10.0
    max_attempts = max(int(getattr(config, "turn_spacing_max_iterations", 4)), 1)
    metric = problem.reduced_turn_spacing_metric
    reference_length = float(problem.turn_spacing_reference_length)
    enabled = _turn_spacing_penalty_requested(config) and metric is not None and metric.size > 0

    if not enabled:
        return {
            "enabled": False,
            "threshold": threshold,
            "factor": factor,
            "growth": growth,
            "max_attempts": 1,
            "initial_weight": 0.0,
            "scaled_metric": None,
            "reference_length": reference_length,
            "matrix_scale": 0.0,
        }

    ata_scale = _matrix_mean_abs_diag(ata)
    metric_scale = _matrix_mean_abs_diag(metric)
    matrix_scale = ata_scale / metric_scale if ata_scale > 0.0 and metric_scale > 0.0 else 1.0
    reference = max(reference_length, 1e-12)
    initial_weight = factor * (threshold / reference) ** 2
    if not np.isfinite(initial_weight) or initial_weight <= 0.0:
        initial_weight = factor
    return {
        "enabled": True,
        "threshold": threshold,
        "factor": factor,
        "growth": growth,
        "max_attempts": max_attempts,
        "initial_weight": float(initial_weight),
        "scaled_metric": matrix_scale * metric,
        "reference_length": reference_length,
        "matrix_scale": float(matrix_scale),
    }


def _turn_spacing_diagnostics(
    spacing: dict[str, Any],
    attempts: list[dict[str, float | bool]],
) -> dict[str, Any]:
    estimated = [float(attempt["estimated_min_turn_distance"]) for attempt in attempts]
    weights = [float(attempt["penalty_weight"]) for attempt in attempts]
    threshold_met = bool(attempts[-1]["threshold_met"]) if attempts else False
    last_estimated = estimated[-1] if estimated else 0.0
    return {
        "turn_spacing_penalty_enabled": bool(spacing["enabled"]),
        "min_turn_distance": float(spacing["threshold"]),
        "turn_spacing_reference_length": float(spacing["reference_length"]),
        "turn_spacing_regularization_factor": float(spacing["factor"]),
        "turn_spacing_attempts": len(attempts),
        "turn_spacing_threshold_met": threshold_met,
        "estimated_min_turn_distance": last_estimated,
        "turn_spacing_penalty_weights": weights,
        "estimated_min_turn_distances": estimated,
        "turn_spacing_penalty_matrix_scale": float(spacing["matrix_scale"]),
    }


def _estimate_min_turn_distance(
    problem: StreamFunctionOptimizationProblem,
    reduced_stream_function: np.ndarray,
    config: CoilGenConfig,
) -> float:
    full_stream_function = problem.expand(reduced_stream_function)
    offset = 0
    estimates: list[float] = []
    for vertex_count, current_density_mat, faces, face_basis_currents in zip(
        problem.part_vertex_counts,
        problem.part_current_density_mats,
        problem.part_faces,
        problem.part_face_basis_currents,
    ):
        part_stream_function = full_stream_function[offset : offset + vertex_count]
        offset += vertex_count
        contour_step = _estimated_contour_step(part_stream_function, config)
        if contour_step <= 0.0:
            continue
        if current_density_mat is not None:
            current_density = np.tensordot(part_stream_function, current_density_mat, axes=(0, 0))
        else:
            current_density = np.einsum(
                "fv,fva->fa",
                part_stream_function[np.asarray(faces, dtype=np.int64)],
                np.asarray(face_basis_currents, dtype=float),
                optimize=True,
            )
        density_norm = np.linalg.norm(current_density, axis=1)
        valid = np.isfinite(density_norm) & (density_norm > 1e-15)
        if np.any(valid):
            estimates.append(float(np.min(contour_step / density_norm[valid])))
    return min(estimates) if estimates else float("inf")


def _estimated_contour_step(stream_function: np.ndarray, config: CoilGenConfig) -> float:
    if engineering_contouring_enabled(config):
        engineering_step = engineering_max_contour_step_A(config)
        if engineering_step > 0.0:
            return float(engineering_step)
    sf_range = float(np.ptp(stream_function))
    levels = max(int(getattr(config, "levels", 1)), 1)
    denominator = _contour_step_denominator(levels, config)
    if not np.isfinite(sf_range) or sf_range <= 0.0:
        return 0.0
    return sf_range / denominator


def _matrix_mean_abs_diag(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    diag = np.abs(np.diag(matrix))
    finite = diag[np.isfinite(diag) & (diag > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float)]


def _normalise_stream_function_cost_function(cost_function: str) -> str:
    normalised = str(cost_function).strip().lower().replace("-", "_")
    aliases = {
        "standard": "field_error_plus_resistance",
        "field_error_plus_resistance": "field_error_plus_resistance",
        "least_squares_plus_resistance": "field_error_plus_resistance",
    }
    if normalised in aliases:
        return aliases[normalised]
    available = ", ".join(sorted(STREAM_FUNCTION_COST_FORMULAS))
    raise NotImplementedError(
        f"Stream-function cost function '{cost_function}' is not implemented. "
        f"Available cost functions: {available}. "
        "Add a new solver or branch in solve_tikhonov_stream_function()."
    )


def _solve_stream_function_system(lhs: np.ndarray, rhs: np.ndarray, config: CoilGenConfig) -> np.ndarray:
    solver = str(getattr(config, "optimization_solver", "auto")).lower()
    if solver not in {"auto", "solve", "pinv"}:
        raise ValueError("optimization_solver must be 'auto', 'solve', or 'pinv'.")
    progress = ProgressReporter.from_config(config, label="Linear solve")
    if solver == "pinv":
        progress.report(None, "computing a pseudoinverse; factorization completion is not available")
        return np.linalg.pinv(lhs) @ rhs

    tolerance = max(float(getattr(config, "optimization_solver_residual_tolerance", 1e-8)), 0.0)
    try:
        progress.report(None, "solving the stream-function system; factorization completion is not available")
        candidate = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        if solver == "solve":
            raise
        progress.report(None, "direct solve failed; computing a pseudoinverse")
        return np.linalg.pinv(lhs) @ rhs

    progress.report(None, "checking the linear-system residual")
    residual = np.linalg.norm(lhs @ candidate - rhs) / max(np.linalg.norm(rhs), 1e-30)
    if np.all(np.isfinite(candidate)) and float(residual) <= tolerance:
        return candidate
    if solver == "solve":
        raise np.linalg.LinAlgError(f"Direct solve residual {residual:.3e} exceeds tolerance {tolerance:.3e}.")
    progress.report(None, "direct solve residual failed; computing a pseudoinverse")
    return np.linalg.pinv(lhs) @ rhs
