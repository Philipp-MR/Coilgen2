"""CVXPY conic execution over assembled stream-function problems and physical constraints."""

from __future__ import annotations
from .optimization_constraints import (
    ResidualConstraintBlock,
    _constrained_contour_step_limit,
    _constraint_acceptance_relative_tolerance,
    _contract_surface_current_density,
    _shield_gradient_opposition_constraint_data,
    _shield_gradient_opposition_constraint_diagnostics,
)
from .optimization_problem import StreamFunctionOptimizationProblem, _normalise_constrained_quadratic_objective_mode
import os
import time
from typing import Any
import numpy as np
from .config import CoilGenConfig
from .engineering_limits import engineering_limit_diagnostics
from .optimization_solvers import conic_solver_order
from .progress import ProgressReporter


def _solve_constrained_quadratic_cvxpy(
    problem: StreamFunctionOptimizationProblem,
    objective_quadratic: np.ndarray,
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    config: CoilGenConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CVXPY is not installed. Install cvxpy to use the direct constrained quadratic solver."
        ) from exc

    if not constraint_blocks:
        raise ValueError("Constrained quadratic solver requires at least one residual constraint block.")
    n_unknowns = int(objective_quadratic.shape[0])
    residual_row_count = int(sum(int(block.matrix.shape[0]) for block in constraint_blocks))
    for block in constraint_blocks:
        if block.matrix.ndim != 2 or block.target.ndim != 1 or block.matrix.shape[0] != block.target.shape[0]:
            raise ValueError(f"Constraint block {block.name!r} has inconsistent A and b dimensions.")
        if block.matrix.shape[1] != n_unknowns:
            raise ValueError(f"Constraint block {block.name!r} has the wrong number of unknowns.")

    progress = ProgressReporter.from_config(config, label="Constrained quadratic")
    progress.report(
        0.0,
        (
            f"building CVXPY problem ({n_unknowns} unknowns, "
            f"{len(constraint_blocks)} residual block(s), {residual_row_count} residual rows)"
        ),
        force=True,
    )
    variable = cp.Variable(n_unknowns)
    minimax = _normalise_constrained_quadratic_objective_mode(config) == "minimax_current_density"
    peak_variable = None
    density_operators: tuple[np.ndarray, ...] = ()
    if minimax:
        expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
        peak, _face_count, reason = _contract_surface_current_density(problem, np.zeros(expansion.shape[0]))
        if not np.isfinite(peak):
            raise ValueError("Minimax requires complete finite face-current operators. " + reason)
        density_operators = _reduced_surface_current_density_operators(problem)
        # Normalize each basis column by its physical current gain. Then choose
        # an A/m scale from the field drive. Both survive a rescaling of basis DOFs.
        gains = np.maximum.reduce([np.max(np.linalg.norm(operator, axis=1), axis=0) for operator in density_operators])
        if not np.all(np.isfinite(gains)):
            raise ValueError("Minimax current-density basis gains overflowed; rescale the input basis.")
        coefficient_scales = np.ones(n_unknowns)
        np.divide(1.0, gains, out=coefficient_scales, where=gains > 0.0)
        drive_scales = []
        for block in constraint_blocks:
            field_gain = float(np.max(np.abs(block.matrix * coefficient_scales)))
            target_magnitude = float(np.max(np.abs(block.target)))
            if field_gain > 0.0 and target_magnitude > 0.0:
                drive_scales.append(target_magnitude / field_gain)
        objective_scale = _cvxpy_positive_scale(max(drive_scales, default=1.0), "minimax current-density scale")
        variable = cp.multiply(coefficient_scales * objective_scale, variable)
        peak_variable = cp.Variable(nonneg=True)
        objective = cp.Minimize(peak_variable)
    else:
        objective_scale = _cvxpy_matrix_scale(objective_quadratic)
        objective_matrix = np.asarray(objective_quadratic, dtype=float) / objective_scale
        eigenvalues = np.linalg.eigvalsh(objective_matrix)
        if eigenvalues.size and eigenvalues[0] < -1.0e-10 * max(float(np.max(np.abs(eigenvalues))), 1.0):
            raise ValueError("The conic quadratic objective must be positive semidefinite.")
        objective = cp.Minimize(cp.quad_form(variable, cp.psd_wrap(objective_matrix)))
    residual_scales = tuple(
        _cvxpy_positive_scale(block.allowed_residual, f"{block.name} allowed residual") for block in constraint_blocks
    )
    constraints = [
        _cvxpy_residual_constraint(cp, block, variable, scale=scale)
        for block, scale in zip(constraint_blocks, residual_scales)
    ]
    if minimax:
        for operator in density_operators:
            components = cp.vstack([(operator[:, axis, :] / objective_scale) @ variable for axis in range(3)]).T
            constraints.append(cp.norm(components, 2, axis=1) <= peak_variable)
    contour_step_limit = _constrained_contour_step_limit(config)
    contour_step_normalization = float("nan")
    if bool(contour_step_limit["contour_step_constraint_enabled"]):
        expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
        if expansion.ndim != 2 or expansion.shape[1] != n_unknowns:
            raise ValueError("Contour-step constraint expansion has inconsistent dimensions.")
        contour_step_normalization = _cvxpy_positive_scale(
            contour_step_limit["contour_step_constraint_max_range_A"],
            "contour-step maximum range",
        )
        full_stream_function = (expansion / contour_step_normalization) @ variable
        stream_min = cp.Variable()
        stream_max = cp.Variable()
        constraints.extend(
            [
                full_stream_function <= stream_max,
                full_stream_function >= stream_min,
                stream_max - stream_min <= 1.0,
            ]
        )
    engineering_diagnostics = engineering_limit_diagnostics(config)
    current_density_limit = float(engineering_diagnostics["engineering_surface_current_density_limit_A_per_m"])
    current_density_limit_source = str(engineering_diagnostics["engineering_surface_current_density_limit_source"])
    current_density_constraint_count = 0
    if current_density_limit > 0.0 and minimax:
        constraints.append(peak_variable <= current_density_limit / objective_scale)
        current_density_constraint_count = sum(len(operator) for operator in density_operators)
    elif current_density_limit > 0.0:
        density_constraints, current_density_constraint_count = _cvxpy_surface_current_density_constraints(
            cp,
            problem,
            variable,
            float(current_density_limit),
        )
        constraints.extend(density_constraints)
    shield_gradient_constraint = _shield_gradient_opposition_constraint_data(problem, config)
    shield_gradient_constraint_count = 0
    if bool(shield_gradient_constraint["enabled"]):
        scale = _cvxpy_positive_scale(
            shield_gradient_constraint["scale_T_per_m"],
            "shield-gradient opposition scale",
        )
        constraints.append(
            (np.asarray(shield_gradient_constraint["matrix"], dtype=float).reshape(1, -1) / scale) @ variable
            <= float(shield_gradient_constraint["limit_T_per_m"]) / scale
        )
        shield_gradient_constraint_count = 1
    cvx_problem = cp.Problem(objective, constraints)
    solver_order = _cvxpy_conic_solver_order(cp, config)
    solver_verbose = bool(getattr(config, "constrained_optimization_verbose", getattr(config, "progress_flag", False)))
    requested_thread_count = _cvxpy_requested_thread_count(config)
    thread_text = (
        "solver threads=default"
        if requested_thread_count <= 0
        else f"requested solver threads={requested_thread_count}"
    )
    progress.report(
        10.0,
        (
            f"CVXPY problem ready ({len(constraints)} constraint expression(s), "
            f"solvers: {', '.join(str(name) for name in solver_order) or 'none'}, "
            f"verbose={solver_verbose}, {thread_text})"
        ),
        force=True,
    )

    solver_errors: list[str] = []
    task = progress.task("CVXPY solver attempts", len(solver_order), indeterminate=True)
    for solver_index, solver_name in enumerate(solver_order):
        options = _cvxpy_solver_options(solver_name, config)
        thread_option, solver_thread_count = _cvxpy_solver_thread_option(solver_name, config)
        thread_suffix = f", {thread_option}={solver_thread_count}" if solver_thread_count > 0 else ""
        task.update(solver_index, message=f"starting {solver_name}{thread_suffix}", force=True)
        solve_start = time.perf_counter()
        try:
            cvx_problem.solve(solver=solver_name, verbose=solver_verbose, **options)
        except Exception as exc:
            elapsed = time.perf_counter() - solve_start
            solver_errors.append(f"{solver_name}: {exc}")
            task.advance(1, message=f"{solver_name} failed after {elapsed:.2f} s: {exc}", force=True)
            continue
        elapsed = time.perf_counter() - solve_start
        if cvx_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            solver_errors.append(f"{solver_name}: status {cvx_problem.status}")
            task.advance(
                1,
                message=f"{solver_name} returned status {cvx_problem.status} after {elapsed:.2f} s",
                force=True,
            )
            continue
        if variable.value is None:
            solver_errors.append(f"{solver_name}: returned no solution vector")
            task.advance(1, message=f"{solver_name} returned no solution vector after {elapsed:.2f} s", force=True)
            continue
        candidate = np.asarray(variable.value, dtype=float).reshape(-1)
        if candidate.shape[0] != n_unknowns or not np.all(np.isfinite(candidate)):
            solver_errors.append(f"{solver_name}: returned non-finite solution")
            task.advance(1, message=f"{solver_name} returned a non-finite solution after {elapsed:.2f} s", force=True)
            continue
        peak_current = float("nan")
        epigraph_current = float("nan")
        if minimax:
            peak_current = max(
                float(np.max(np.linalg.norm(operator @ candidate, axis=1))) for operator in density_operators
            )
            epigraph_current = (
                float(peak_variable.value) * objective_scale if peak_variable.value is not None else float("nan")
            )
            acceptance = _constraint_acceptance_relative_tolerance(config) * max(
                peak_current, abs(epigraph_current), objective_scale
            )
            if (
                not np.isfinite(peak_current)
                or not np.isfinite(epigraph_current)
                or peak_current > epigraph_current + acceptance
            ):
                solver_errors.append(f"{solver_name}: minimax epigraph failed independent face-current check")
                task.advance(1, message=f"{solver_name} failed minimax face-current check", force=True)
                continue
        task.advance(1, message=f"{solver_name} solved with status {cvx_problem.status} in {elapsed:.2f} s", force=True)
        diagnostics = {
            "solver": str(solver_name),
            "status": str(cvx_problem.status),
            "objective_value": (
                peak_current if minimax else float(candidate @ np.asarray(objective_quadratic, dtype=float) @ candidate)
            ),
            "objective_unit": "A/m" if minimax else "weighted_quadratic",
            "minimax_current_density_enabled": minimax,
            "minimax_current_density_peak_A_per_m": peak_current,
            "minimax_current_density_epigraph_A_per_m": epigraph_current,
            "minimax_current_density_face_count": sum(len(operator) for operator in density_operators),
            "solver_scaled_objective_value": (
                float(cvx_problem.value) if cvx_problem.value is not None else float("nan")
            ),
            "objective_normalization_scale": float(objective_scale),
            "problem_unknown_count": int(n_unknowns),
            "residual_constraint_block_count": int(len(constraint_blocks)),
            "residual_constraint_row_count": int(residual_row_count),
            "cvxpy_constraint_expression_count": int(len(constraints)),
            "residual_constraint_normalization_enabled": True,
            "residual_constraint_normalization_min": float(min(residual_scales)),
            "residual_constraint_normalization_max": float(max(residual_scales)),
            "contour_step_constraint_normalization_A": float(contour_step_normalization),
            "solver_verbose": bool(solver_verbose),
            "solver_thread_count_requested": int(requested_thread_count),
            "solver_thread_count_applied": int(solver_thread_count),
            "solver_thread_option": str(thread_option),
            "solver_elapsed_seconds": float(elapsed),
            "solver_error_count": int(len(solver_errors)),
            "surface_current_density_constraint_enabled": bool(current_density_limit > 0.0),
            "surface_current_density_constraint_limit_A_per_m": float(current_density_limit),
            "surface_current_density_constraint_explicit_limit_A_per_m": float(
                engineering_diagnostics["surface_current_density_limit_A_per_m"]
            ),
            "surface_current_density_constraint_derived_limit_A_per_m": float(
                engineering_diagnostics["derived_surface_current_density_limit_A_per_m"]
            ),
            "surface_current_density_constraint_effective_limit_A_per_m": float(current_density_limit),
            "surface_current_density_constraint_source": str(
                current_density_limit_source if current_density_limit > 0.0 else "disabled"
            ),
            "surface_current_density_constraint_count": int(current_density_constraint_count),
            "shield_gradient_opposition_constraint_count": int(shield_gradient_constraint_count),
            **_shield_gradient_opposition_constraint_diagnostics(shield_gradient_constraint, candidate),
            **contour_step_limit,
        }
        progress.report(100.0, f"accepted {solver_name} solution after {task.completed} solver attempt(s)")
        return candidate, diagnostics

    task.update(task.completed, message="all available solver attempts failed", force=True)
    details = "; ".join(solver_errors) if solver_errors else "no installed conic solver found"
    raise RuntimeError(f"CVXPY constrained quadratic solve failed ({details}).")


def _cvxpy_residual_constraint(cp: Any, block: ResidualConstraintBlock, variable: Any, *, scale: float = 1.0) -> Any:
    scale = _cvxpy_positive_scale(scale, f"{block.name} residual scale")
    residual = (np.asarray(block.matrix, dtype=float) / scale) @ variable - (
        np.asarray(block.target, dtype=float) / scale
    )
    allowed = float(block.allowed_residual) / scale
    if str(block.metric).lower() == "linf":
        group_size = max(int(getattr(block, "group_size", 1) or 1), 1)
        if group_size > 1:
            row_count = int(block.matrix.shape[0])
            if row_count % group_size != 0:
                raise ValueError(
                    f"Constraint block {block.name!r} has {row_count} residual rows, "
                    f"which is not divisible by group_size={group_size}."
                )
            grouped = cp.reshape(residual, (row_count // group_size, group_size), order="C")
            return cp.norm(grouped, 2, axis=1) <= allowed
        return cp.norm(residual, "inf") <= allowed
    return cp.norm(residual, 2) <= allowed


def _cvxpy_matrix_scale(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    scale = float(np.max(finite))
    return scale if scale > 0.0 and np.isfinite(scale) else 1.0


def _cvxpy_positive_scale(value: Any, label: str) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive number.") from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} must be a finite positive number.")
    return max(scale, 1.0e-300)


def _cvxpy_surface_current_density_constraints(
    cp: Any,
    problem: StreamFunctionOptimizationProblem,
    variable: Any,
    current_density_limit: float,
) -> tuple[list[Any], int]:
    scale = _cvxpy_positive_scale(current_density_limit, "surface-current-density limit")
    constraints: list[Any] = []
    constraint_count = 0
    for operator in _reduced_surface_current_density_operators(problem):
        components = cp.vstack([(operator[:, axis, :] / scale) @ variable for axis in range(3)]).T
        constraints.append(cp.norm(components, 2, axis=1) <= 1.0)
        constraint_count += len(operator)
    return constraints, constraint_count


def _reduced_surface_current_density_operators(problem: StreamFunctionOptimizationProblem) -> tuple[np.ndarray, ...]:
    """Return each part's face-by-Cartesian-component-by-reduced-DOF operator."""

    expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
    result: list[np.ndarray] = []
    offset = 0
    for vertex_count, current_density_mat, faces, face_basis_currents in zip(
        problem.part_vertex_counts,
        problem.part_current_density_mats,
        problem.part_faces,
        problem.part_face_basis_currents,
    ):
        part_expansion = expansion[offset : offset + vertex_count]
        offset += vertex_count
        if current_density_mat is not None:
            current_density_mat = np.asarray(current_density_mat, dtype=float)
            if current_density_mat.ndim != 3 or current_density_mat.shape[0] != vertex_count:
                raise ValueError("Surface-current-density matrix has inconsistent dimensions.")
            operators = [
                np.asarray(current_density_mat[:, :, component], dtype=float).T @ part_expansion
                for component in range(3)
            ]
            reduced_operators = np.stack(operators, axis=1)
        else:
            faces = np.asarray(faces, dtype=np.int64)
            face_basis_currents = np.asarray(face_basis_currents, dtype=float)
            if face_basis_currents.shape != (len(faces), 3, 3):
                raise ValueError("Local surface-current-density operators have inconsistent dimensions.")
            reduced_operators = np.einsum(
                "fva,fvr->far",
                face_basis_currents,
                part_expansion[faces],
                optimize=True,
            )
        if not np.all(np.isfinite(reduced_operators)):
            raise ValueError("Surface-current-density operators contain nonfinite values.")
        result.append(reduced_operators)
    return tuple(result)


def _cvxpy_conic_solver_order(cp: Any, config: CoilGenConfig) -> list[str]:
    return list(
        conic_solver_order(
            str(getattr(config, "constrained_optimization_solver", "auto")),
            installed_solvers=cp.installed_solvers(),
        )
    )


def _cvxpy_solver_options(solver_name: str, config: CoilGenConfig) -> dict[str, Any]:
    name = str(solver_name).upper()
    tolerance = max(float(getattr(config, "constrained_optimization_tolerance", 1e-7)), 1e-12)
    max_iterations = max(int(getattr(config, "constrained_optimization_max_iterations", 20000)), 1)
    if name == "CLARABEL":
        options = {
            "max_iter": max_iterations,
            "tol_gap_abs": tolerance,
            "tol_gap_rel": tolerance,
            "tol_feas": tolerance,
        }
        thread_option, solver_thread_count = _cvxpy_solver_thread_option(solver_name, config)
        if solver_thread_count > 0 and thread_option:
            options[thread_option] = solver_thread_count
        return options
    if name == "SCS":
        return {
            "eps": tolerance,
            "max_iters": max_iterations,
        }
    if name == "ECOS":
        return {
            "abstol": tolerance,
            "reltol": tolerance,
            "feastol": tolerance,
            "max_iters": max_iterations,
        }
    return {}


def _cvxpy_requested_thread_count(config: CoilGenConfig) -> int:
    raw = getattr(config, "constrained_optimization_threads", 0)
    if raw is None or raw == "":
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("constrained_optimization_threads must be an integer >= 0.") from exc
    if value < 0:
        raise ValueError("constrained_optimization_threads must be an integer >= 0.")
    return value


def _cvxpy_solver_thread_option(solver_name: str, config: CoilGenConfig) -> tuple[str, int]:
    requested = _cvxpy_requested_thread_count(config)
    if requested <= 0:
        return "", 0
    name = str(solver_name).upper()
    if name != "CLARABEL":
        return "", 0
    logical_cores = os.cpu_count()
    if logical_cores is not None and logical_cores > 0:
        return "max_threads", min(requested, int(logical_cores))
    return "max_threads", requested
