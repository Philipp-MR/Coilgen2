from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import coilgen.optimization as optimization
from coilgen.config import CoilGenConfig
from coilgen.models import CoilPart, Mesh, TargetField
from coilgen.optimization import (
    BoundaryReducer,
    ResidualConstraintBlock,
    StreamFunctionOptimizationProblem,
    StreamFunctionSolution,
)


def _problem(*, local_density: bool = False) -> StreamFunctionOptimizationProblem:
    mesh = Mesh(vertices=np.eye(3), faces=np.array([[0, 1, 2]]))
    reducer = BoundaryReducer(3, [], np.arange(3), np.eye(3))
    matrix = np.eye(3) * 1.0e-6
    sensitivity = np.zeros((3, 3, 3))
    sensitivity[2] = matrix
    currents = np.array([[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    target = np.array([1.0, -1.0, 0.0]) * 1.0e-6
    return StreamFunctionOptimizationProblem(
        sensitivity_matrix=sensitivity,
        resistance_matrix=np.eye(3),
        combined_mesh=mesh,
        reducer=reducer,
        target_z=target,
        reduced_sensitivity=matrix,
        reduced_resistance=np.eye(3),
        unweighted_target_z=target,
        unweighted_reduced_sensitivity=matrix,
        target_coords=np.eye(3),
        target_weights=np.ones(3),
        target_suppressed=np.zeros(3, dtype=bool),
        part_vertex_counts=(3,),
        part_current_density_mats=(None if local_density else currents,),
        part_faces=(mesh.faces,),
        part_face_basis_currents=(currents.transpose(1, 0, 2),),
    )


def _config(**overrides) -> CoilGenConfig:
    values = {
        "sf_opt_method": "constrained_quadratic",
        "target_error_constraint_metric": "rms",
        "target_error_rms_limit_T": 1.0e-7,
        "target_error_constraint_acceptance_absolute_T": 0.0,
        "constrained_quadratic_fallback_policy": "diagnostic",
        "progress_flag": False,
    }
    values.update(overrides)
    return CoilGenConfig(**values)


def _fake_optimal(monkeypatch, values: np.ndarray) -> None:
    def selected(_problem, _config):
        return StreamFunctionSolution(
            values.copy(),
            {
                "constrained_solver_status": "optimal",
                "target_error_constraint_met": True,
                "surface_current_density_constraint_met": True,
                "contour_step_constraint_met": True,
                "shield_gradient_opposition_constraint_met": True,
                "constrained_solver_contract_met": True,
            },
        )

    monkeypatch.setattr(optimization, "_select_constrained_quadratic", selected)


@pytest.mark.parametrize("local_density", [False, True])
@pytest.mark.parametrize("policy", ["diagnostic", "strict"])
def test_false_optimal_density_violation_is_recomputed(monkeypatch, local_density, policy) -> None:
    _fake_optimal(monkeypatch, np.array([1.0, -1.0, 0.0]))
    config = _config(surface_current_density_limit_A_per_m=1.0, constrained_quadratic_fallback_policy=policy)
    if policy == "strict":
        with pytest.raises(RuntimeError, match="surface_current_density"):
            optimization.solve_stream_function_problem(_problem(local_density=local_density), config)
        return
    result = optimization.solve_stream_function_problem(_problem(local_density=local_density), config)
    assert result.diagnostics["target_error_constraint_met"]
    assert not result.diagnostics["constrained_solver_contract_met"]
    assert not result.diagnostics["surface_current_density_constraint_met"]
    assert result.diagnostics["surface_current_density_constraint_max_A_per_m"] == pytest.approx(2.0)


def test_false_optimal_range_violation_uses_full_stream_function(monkeypatch) -> None:
    _fake_optimal(monkeypatch, np.array([1.0, -1.0, 0.0]))
    result = optimization.solve_stream_function_problem(
        _problem(), _config(regularization_auto_range_max_contour_step_A=0.5, levels=2, pot_offset_factor=0.0)
    )
    assert result.diagnostics["target_error_constraint_met"]
    assert not result.diagnostics["constrained_solver_contract_met"]
    assert result.diagnostics["contour_step_constraint_range_A"] == pytest.approx(2.0)
    assert result.diagnostics["contour_step_constraint_max_range_A"] == pytest.approx(0.5)
    assert result.diagnostics["constrained_solver_failed_constraints"] == ["contour_step"]


def test_false_optimal_suppression_violation_is_recomputed(monkeypatch) -> None:
    problem = _problem()
    problem.target_suppressed[-1] = True
    _fake_optimal(monkeypatch, np.array([1.0, -1.0, 1.0]))
    result = optimization.solve_stream_function_problem(problem, _config(suppression_field_rms_limit_T=1.0e-7))
    assert result.diagnostics["dsv_constraint_met"]
    assert not result.diagnostics["suppression_constraint_met"]
    assert not result.diagnostics["constrained_solver_contract_met"]


def _shield_problem() -> StreamFunctionOptimizationProblem:
    coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * 0.01
    target = coords[:, 0] * 1.0e-6
    matrix = np.tile(target[:, None], (1, 4))
    sensitivity = np.zeros((3, 5, 4))
    sensitivity[2] = matrix
    return replace(
        _problem(),
        sensitivity_matrix=sensitivity,
        resistance_matrix=np.eye(4),
        reducer=BoundaryReducer(4, [], np.arange(4), np.eye(4)),
        target_z=target,
        reduced_sensitivity=matrix,
        reduced_resistance=np.eye(4),
        unweighted_target_z=target,
        unweighted_reduced_sensitivity=matrix,
        target_coords=coords,
        target_weights=np.ones(5),
        target_suppressed=np.zeros(5, dtype=bool),
        part_vertex_counts=(1, 1, 1, 1),
        part_outward_scores=(1.0, 1.0, 2.0, 2.0),
    )


def test_false_optimal_shield_sign_violation_is_recomputed(monkeypatch) -> None:
    _fake_optimal(monkeypatch, np.array([0.0, 0.0, 1.0, 0.0]))
    result = optimization.solve_stream_function_problem(
        _shield_problem(), _config(shield_gradient_opposition_enabled=True, target_gradient_strength=1.0e-6)
    )
    assert result.diagnostics["target_error_constraint_met"]
    assert result.diagnostics["shield_gradient_opposition_enabled"]
    assert result.diagnostics["shield_gradient_opposition_value_T_per_m"] == pytest.approx(1.0e-6)
    assert not result.diagnostics["constrained_solver_contract_met"]


@pytest.mark.parametrize("missing", ["density", "shield"])
def test_unavailable_requested_constraint_cannot_pass(monkeypatch, missing) -> None:
    problem = _problem()
    config = _config()
    if missing == "density":
        problem.part_face_basis_currents = ()
        config.surface_current_density_limit_A_per_m = 10.0
    else:
        config.shield_gradient_opposition_enabled = True
    _fake_optimal(monkeypatch, np.array([1.0, -1.0, 0.0]))
    result = optimization.solve_stream_function_problem(problem, config)
    assert not result.diagnostics["constrained_solver_constraint_data_available"]
    assert not result.diagnostics["constrained_solver_hard_constraints_enforced"]
    assert not result.diagnostics["constrained_solver_contract_met"]


@pytest.mark.parametrize(
    "parameter, missing_block",
    [("linearity_rms_tolerance_percent", "linearity_rms"), ("suppression_field_rms_limit_T", "suppression")],
)
def test_unavailable_requested_residual_block_cannot_pass(monkeypatch, parameter, missing_block) -> None:
    _fake_optimal(monkeypatch, np.array([1.0, -1.0, 0.0]))
    result = optimization.solve_stream_function_problem(_problem(), _config(**{parameter: 1.0}))
    assert not result.diagnostics["constrained_solver_contract_met"]
    assert not result.diagnostics["constrained_solver_constraint_data_available"]
    assert missing_block in result.diagnostics["constrained_solver_failed_constraints"]


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_peak_residual_is_never_zero(value) -> None:
    block = ResidualConstraintBlock("dsv", np.eye(1), np.zeros(1), 0.1, 1.0, 10.0, 0.0, metric="linf")
    assert not optimization._constraint_blocks_met((block,), np.array([value]))
    assert optimization._constraint_residual_measure(block, np.array([value])) == np.inf


@pytest.mark.parametrize("hardware", ["surface_current_density", "contour_step", "shield_gradient_opposition"])
def test_default_fallback_refuses_unsupported_hard_constraints(monkeypatch, hardware) -> None:
    def unavailable(*_args):
        raise RuntimeError("No conic solver available")

    monkeypatch.setattr(optimization, "_select_constrained_quadratic", unavailable)
    config = _config(constrained_quadratic_fallback_policy="target_error_scan")
    if hardware == "surface_current_density":
        config.surface_current_density_limit_A_per_m = 1.0
    elif hardware == "contour_step":
        config.regularization_auto_range_max_contour_step_A = 1.0
    else:
        config.shield_gradient_opposition_enabled = True
    with pytest.raises(RuntimeError, match=f"cannot enforce.*{hardware}"):
        optimization.solve_stream_function_problem(_problem(), config)


def test_minimax_objective_cannot_fall_back_to_resistance_scan(monkeypatch) -> None:
    def unavailable(*_args):
        raise RuntimeError("No conic solver available")

    monkeypatch.setattr(optimization, "_select_constrained_quadratic", unavailable)
    with pytest.raises(RuntimeError, match="minimax_current_density_objective"):
        optimization.solve_stream_function_problem(
            _problem(),
            _config(
                constrained_quadratic_objective_mode="minimax_current_density",
                constrained_quadratic_fallback_policy="target_error_scan",
            ),
        )


def test_diagnostic_fallback_reports_unenforced_constraints(monkeypatch) -> None:
    def unavailable(*_args):
        raise RuntimeError("No conic solver available")

    monkeypatch.setattr(optimization, "_select_constrained_quadratic", unavailable)
    monkeypatch.setattr(
        optimization,
        "_select_target_error_constrained_efficiency",
        lambda *_args: StreamFunctionSolution(np.array([1.0, -1.0, 0.0]), {"target_error_constraint_met": True}),
    )
    result = optimization.solve_stream_function_problem(_problem(), _config(surface_current_density_limit_A_per_m=1.0))
    assert not result.diagnostics["constrained_solver_contract_met"]
    assert not result.diagnostics["constrained_solver_physical_constraints_met"]
    assert result.diagnostics["constrained_solver_fallback_unsupported_constraints"] == ["surface_current_density"]


def test_residual_only_scan_fallback_does_not_claim_conic_optimality(monkeypatch) -> None:
    def unavailable(*_args):
        raise RuntimeError("No conic solver available")

    monkeypatch.setattr(optimization, "_select_constrained_quadratic", unavailable)
    result = optimization.solve_stream_function_problem(
        _problem(), _config(constrained_quadratic_fallback_policy="target_error_scan")
    )
    assert result.diagnostics["constrained_solver_physical_constraints_met"]
    assert not result.diagnostics["constrained_solver_contract_met"]
    assert not result.diagnostics["constrained_solver_hard_constraints_enforced"]


@pytest.mark.parametrize("local_density", [False, True])
def test_real_conic_solution_passes_every_enabled_physical_constraint(local_density) -> None:
    pytest.importorskip("cvxpy")
    result = optimization.solve_stream_function_problem(
        _problem(local_density=local_density),
        _config(
            constrained_quadratic_fallback_policy="strict",
            target_error_rms_limit_T=2.0e-7,
            surface_current_density_limit_A_per_m=1.8,
            regularization_auto_range_max_contour_step_A=1.9,
            levels=2,
            pot_offset_factor=0.0,
        ),
    )
    assert result.diagnostics["constrained_solver_contract_met"]
    assert result.diagnostics["surface_current_density_constraint_max_A_per_m"] <= 1.8 * (1.0 + 1.0e-6)
    assert result.diagnostics["contour_step_constraint_range_A"] <= 1.9 * (1.0 + 1.0e-6)
    assert not result.diagnostics["constrained_solver_failed_constraints"]


def test_real_conic_solution_obeys_shield_gradient_opposition() -> None:
    pytest.importorskip("cvxpy")
    result = optimization.solve_stream_function_problem(
        _shield_problem(),
        _config(
            shield_gradient_opposition_enabled=True,
            target_gradient_strength=1.0e-6,
            target_error_rms_limit_T=1.0e-10,
            constrained_quadratic_fallback_policy="strict",
        ),
    )
    assert result.diagnostics["constrained_solver_contract_met"]
    assert result.diagnostics["shield_gradient_opposition_value_T_per_m"] <= 1.0e-12


@pytest.mark.parametrize("minimax", [False, True])
def test_post_solve_density_scaling_refreshes_residual_contract(monkeypatch, minimax) -> None:
    problem = _problem()
    part = CoilPart(mesh=problem.combined_mesh)
    part.current_density_mat = problem.part_current_density_mats[0]
    target = TargetField(
        coords=problem.target_coords,
        b=np.column_stack((np.zeros((3, 2)), problem.target_z)),
        weights=np.ones(3),
        suppressed=np.zeros(3, dtype=bool),
        target_field_group_inds=np.ones(3, dtype=int),
    )
    monkeypatch.setattr(StreamFunctionOptimizationProblem, "from_parts", lambda *_args: problem)
    monkeypatch.setattr(
        optimization,
        "solve_stream_function_problem",
        lambda *_args: StreamFunctionSolution(
            np.array([1.0, -1.0, 0.0]),
            {
                "stream_function_scheme": "constrained_quadratic",
                "constrained_solver_contract_met": True,
                "constrained_solver_hard_constraints_enforced": True,
                "minimax_current_density_enabled": minimax,
                "minimax_current_density_peak_A_per_m": 2.0,
                "minimax_current_density_epigraph_A_per_m": 2.0,
            },
        ),
    )
    _, _, _, diagnostics = optimization.stream_function_optimization(
        [part], target, _config(surface_current_density_limit_A_per_m=1.999999, target_error_rms_limit_T=1.0e-13)
    )
    assert diagnostics["surface_current_density_scaling_applied"]
    assert not diagnostics["constrained_solver_contract_met"]
    assert not diagnostics["dsv_constraint_met"]
    if minimax:
        assert diagnostics["minimax_current_density_peak_A_per_m"] == pytest.approx(1.999999)
        assert diagnostics["constrained_solver_objective_value"] == pytest.approx(1.999999)
        assert diagnostics["minimax_current_density_pre_scale_peak_A_per_m"] == 2.0
        assert diagnostics["minimax_current_density_epigraph_describes_pre_scale_solution"]
