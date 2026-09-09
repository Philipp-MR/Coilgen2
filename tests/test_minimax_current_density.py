"""Analytic convex optima with independent face-vector magnitudes."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.models import Mesh
from coilgen.optimization import (
    BoundaryReducer,
    ResidualConstraintBlock,
    StreamFunctionOptimizationProblem,
    _residual_constraint_diagnostics,
    _solve_constrained_quadratic_cvxpy,
    solve_stream_function_problem,
)


def _problem(*, dense=False, rotation=None):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    expansion = np.zeros((6, 2))
    expansion[0, 0] = expansion[3, 1] = 1.0
    local = np.zeros((2, 3, 3))
    local[0, 0] = [1, 0, 0]
    local[1, 0] = [0, 2, 0]
    local[:, 1] = -local[:, 0]
    if rotation is not None:
        local = local @ rotation.T
    dense_operator = np.zeros((6, 2, 3))
    for face_index, face in enumerate(faces):
        dense_operator[face, face_index, :] = local[face_index]
    reducer = BoundaryReducer(6, [], np.arange(6), expansion)
    return StreamFunctionOptimizationProblem(
        sensitivity_matrix=np.zeros((3, 1, 6)),
        resistance_matrix=np.eye(6),
        combined_mesh=Mesh(vertices=vertices, faces=faces),
        reducer=reducer,
        target_z=np.array([1.0e-6]),
        reduced_sensitivity=np.array([[1.0e-6, 1.0e-6]]),
        reduced_resistance=np.diag([1.0, 4.0]),
        part_vertex_counts=(6,),
        part_current_density_mats=(dense_operator if dense else None,),
        part_faces=(faces,),
        part_face_basis_currents=(local,),
    )


def _config(solver="CLARABEL", **kwargs):
    return CoilGenConfig(
        sf_opt_method="constrained_quadratic",
        constrained_quadratic_objective_mode="minimax_current_density",
        constrained_quadratic_fallback_policy="strict",
        constrained_optimization_solver=solver,
        constrained_optimization_tolerance=1.0e-9,
        target_error_tolerance_percent=1.0,
        target_error_constraint_acceptance_absolute_T=0.0,
        # SCS stops on a normalized system, so its 1e-9 solver tolerance is
        # not a 1e-9 relative tolerance on this 1%-of-target residual budget.
        # Request an explicit 1e-14 T numerical allowance for this analytic
        # optimum comparison; the production acceptance rules stay strict.
        target_error_constraint_acceptance_tolerance=1.0e-6,
        progress_flag=False,
        **kwargs,
    )


@pytest.mark.parametrize("solver", ["CLARABEL", "SCS"])
@pytest.mark.parametrize("dense", [False, True])
def test_minimax_and_power_have_different_analytic_optima(solver, dense):
    cp = pytest.importorskip("cvxpy")
    if solver not in cp.installed_solvers():
        pytest.skip(f"{solver} is not installed")
    problem = _problem(dense=dense)
    minimax = solve_stream_function_problem(problem, _config(solver))
    # x1+x2 >= .99, minimize max(|x1|, 2|x2|): x1=2x2.
    np.testing.assert_allclose(minimax.reduced_stream_function, [0.66, 0.33], atol=3e-7)
    assert minimax.diagnostics["constrained_solver_contract_met"]
    assert minimax.diagnostics["constrained_solver_objective_unit"] == "A/m"
    assert minimax.diagnostics["minimax_current_density_face_count"] == 2
    assert minimax.diagnostics["constrained_solver_objective_value"] == pytest.approx(0.66, abs=3e-7)
    assert minimax.diagnostics["minimax_current_density_epigraph_A_per_m"] == pytest.approx(0.66, abs=3e-7)
    power_config = replace(_config(solver), constrained_quadratic_objective_mode="balanced")
    power = solve_stream_function_problem(problem, power_config)
    # Same field budget, minimize x1^2 + 4*x2^2: x1=4*x2.
    np.testing.assert_allclose(power.reduced_stream_function, [0.792, 0.198], atol=3e-7)


def test_minimax_is_rotation_invariant_and_honors_hard_cap():
    pytest.importorskip("cvxpy")
    angle = 0.41
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    solution = solve_stream_function_problem(
        _problem(rotation=rotation), _config(surface_current_density_limit_A_per_m=0.7)
    )
    np.testing.assert_allclose(solution.reduced_stream_function, [0.66, 0.33], atol=3e-7)
    assert solution.diagnostics["surface_current_density_constraint_met"]


def test_minimax_rejects_infeasible_peak_cap():
    pytest.importorskip("cvxpy")
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_stream_function_problem(_problem(), _config(surface_current_density_limit_A_per_m=0.6))


@pytest.mark.parametrize("basis_scales", [[1e8, 1e8], [1e-8, 1e8]])
def test_minimax_physical_solution_survives_basis_rescaling(basis_scales):
    pytest.importorskip("cvxpy")
    problem = _problem()
    scales = np.asarray(basis_scales)
    problem.reducer.expansion_matrix *= scales
    problem.reduced_sensitivity *= scales
    problem.reduced_resistance *= scales[:, None] * scales[None, :]
    solution = solve_stream_function_problem(problem, _config())
    np.testing.assert_allclose(solution.reduced_stream_function * scales, [0.66, 0.33], atol=3e-7)
    assert solution.diagnostics["minimax_current_density_epigraph_A_per_m"] == pytest.approx(0.66, abs=3e-7)


def test_minimax_retains_engineering_bounds_without_implicit_spacing_penalty():
    pytest.importorskip("cvxpy")
    problem = _problem()
    problem.reduced_turn_spacing_metric = np.eye(2)
    problem.turn_spacing_reference_length = 0.01
    solution = solve_stream_function_problem(
        problem,
        _config(
            contour_discretization_method="engineering",
            min_wire_distance_m=0.01,
            max_wire_current_A=1.0,
        ),
    )
    np.testing.assert_allclose(solution.reduced_stream_function, [0.66, 0.33], atol=3e-7)
    assert solution.diagnostics["constrained_solver_contract_met"]


def test_minimax_requires_operators_for_every_part():
    pytest.importorskip("cvxpy")
    problem = _problem()
    problem.part_faces = ()
    with pytest.raises(RuntimeError, match="complete finite face-current operators"):
        solve_stream_function_problem(problem, _config())


def test_minimax_does_not_silently_add_explicit_quadratic_penalties():
    with pytest.raises(RuntimeError, match="zero smoothness and turn-spacing penalty factors"):
        solve_stream_function_problem(_problem(), _config(stream_function_smoothness_factor=1.0))


def test_conic_solver_rejects_indefinite_quadratic_objective():
    pytest.importorskip("cvxpy")
    problem = _problem()
    config = replace(_config(), constrained_quadratic_objective_mode="balanced")
    block = ResidualConstraintBlock("dsv", problem.reduced_sensitivity, problem.target_z, 1e-8, 1e-6, 1.0, 0.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _solve_constrained_quadratic_cvxpy(problem, np.diag([1.0, -1.0]), (block,), config)


@pytest.mark.parametrize("name", ["gradient_strength", "cross_gradient"])
def test_gradient_acceptance_does_not_add_a_tesla_allowance_to_tesla_per_metre(name):
    problem = _problem()
    block = ResidualConstraintBlock(name, np.array([[1.0, 0.0]]), np.zeros(1), 1e-9, 1e-6, 0.1, 0.0)
    config = replace(_config(), target_error_constraint_acceptance_absolute_T=1e-7)
    diagnostics = _residual_constraint_diagnostics(problem, np.array([5e-8, 0.0]), (block,), config)
    assert not diagnostics[f"{name}_constraint_met"]
    assert diagnostics[f"{name}_constraint_residual_unit"] == "T/m"
