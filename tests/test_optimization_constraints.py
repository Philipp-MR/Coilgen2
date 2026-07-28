from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.models import Mesh
from coilgen.optimization import (
    BoundaryReducer,
    ResidualConstraintBlock,
    StreamFunctionOptimizationProblem,
    StreamFunctionSolution,
    _constrained_contour_step_limit,
    _normalise_constrained_quadratic_fallback_policy,
    _solve_constrained_quadratic_cvxpy,
    _target_error_retry_solution_is_better,
    solve_stream_function_problem,
)


class _FakeExpression:
    __array_priority__ = 1000

    @property
    def T(self):
        return self

    def __rmatmul__(self, _other):
        return _FakeExpression()

    def __sub__(self, _other):
        return _FakeExpression()

    def __rsub__(self, _other):
        return _FakeExpression()

    def __le__(self, _other):
        return ("le", self, _other)

    def __ge__(self, _other):
        return ("ge", self, _other)


class _FakeVariable(_FakeExpression):
    _created: list["_FakeVariable"] = []

    def __init__(self, size=None):
        self.size = size
        self.value = None
        self._created.append(self)


class _FakeProblem:
    last_solve_kwargs: dict | None = None

    def __init__(self, _objective, _constraints):
        self.status = None
        self.value = None

    def solve(self, **kwargs):
        self.__class__.last_solve_kwargs = dict(kwargs)
        self.status = _FakeCvxpy.OPTIMAL
        self.value = 0.0
        for variable in _FakeVariable._created:
            if variable.size is None:
                variable.value = 0.0
            else:
                variable.value = np.zeros(int(variable.size), dtype=float)


class _FakeCvxpy:
    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    Variable = _FakeVariable
    Problem = _FakeProblem

    @staticmethod
    def installed_solvers():
        return ["CLARABEL"]

    @staticmethod
    def psd_wrap(matrix):
        return matrix

    @staticmethod
    def quad_form(_variable, _matrix):
        return _FakeExpression()

    @staticmethod
    def Minimize(_expression):
        return _FakeExpression()

    @staticmethod
    def norm(_expression, *_args, **_kwargs):
        return _FakeExpression()

    @staticmethod
    def vstack(_values):
        return _FakeExpression()


class OptimizationConstraintTest(unittest.TestCase):
    def test_contour_step_constraint_uses_actual_levels_not_repair_ceiling(self) -> None:
        config = SimpleNamespace(
            regularization_auto_range_max_contour_step_A=0.25,
            max_wire_current_A=0.0,
            max_current_A=0.0,
            min_wire_distance_m=0.0,
            contour_discretization_method="level_count",
            levels=22,
            contourability_validation="auto",
            contourability_repair_enabled=True,
            contourability_repair_max_levels=80,
            pot_offset_factor=0.5,
        )

        diagnostics = _constrained_contour_step_limit(config)

        self.assertEqual(diagnostics["contour_step_constraint_level_count"], 22)
        self.assertAlmostEqual(diagnostics["contour_step_constraint_max_range_A"], 5.5)

    def test_tikhonov_reports_weighted_suppression_target_residual(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((1, 3), dtype=float),
            faces=np.array([[0, 0, 0]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=1,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(1),
            expansion_matrix=np.eye(1),
        )
        sensitivity = np.zeros((3, 2, 1), dtype=float)
        sensitivity[2, :, 0] = 1.0
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=sensitivity,
            resistance_matrix=np.eye(1),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0, 0.0], dtype=float),
            reduced_sensitivity=np.array([[1.0], [10.0]], dtype=float),
            reduced_resistance=np.eye(1),
            unweighted_target_z=np.array([1.0, 0.0], dtype=float),
            unweighted_reduced_sensitivity=np.array([[1.0], [1.0]], dtype=float),
            target_coords=np.zeros((2, 3), dtype=float),
            target_weights=np.array([1.0, 100.0], dtype=float),
            target_suppressed=np.array([False, True], dtype=bool),
            target_suppression_area_weights=np.array([1.0, 100.0], dtype=float),
        )
        config = CoilGenConfig(
            sf_opt_method="tikhonov",
            regularization_selection_method="fixed",
            tikonov_reg_factor=0.0,
            suppression_surface_weighting="area",
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        expected_suppression_residual = 1.0 / 101.0
        self.assertTrue(diagnostics["suppression_target_enabled"])
        self.assertFalse(diagnostics.get("suppression_constraint_enabled", False))
        self.assertEqual(diagnostics["suppression_target_sample_count"], 1)
        self.assertAlmostEqual(diagnostics["suppression_target_weight_mean"], 100.0)
        self.assertAlmostEqual(diagnostics["suppression_target_rms_residual_T"], expected_suppression_residual)
        self.assertAlmostEqual(diagnostics["suppression_field_rms_T"], expected_suppression_residual)
        self.assertAlmostEqual(diagnostics["suppression_target_residual_norm_T"], expected_suppression_residual)
        self.assertAlmostEqual(
            diagnostics["suppression_target_weighted_residual_norm_T"],
            10.0 * expected_suppression_residual,
        )
        self.assertAlmostEqual(
            diagnostics["suppression_target_weighted_rms_residual_T"],
            expected_suppression_residual,
        )

    def test_constrained_quadratic_fallback_policy_aliases_are_normalised(self) -> None:
        self.assertEqual(
            _normalise_constrained_quadratic_fallback_policy(
                SimpleNamespace(constrained_quadratic_fallback_policy="target-error-scan")
            ),
            "target_error_scan",
        )
        self.assertEqual(
            _normalise_constrained_quadratic_fallback_policy(
                SimpleNamespace(constrained_quadratic_fallback_policy="tikhonov_scan")
            ),
            "target_error_scan",
        )
        self.assertEqual(
            _normalise_constrained_quadratic_fallback_policy(
                SimpleNamespace(constrained_quadratic_fallback_policy="diagnostic")
            ),
            "diagnostic",
        )
        self.assertEqual(
            _normalise_constrained_quadratic_fallback_policy(
                SimpleNamespace(constrained_quadratic_fallback_policy="fail")
            ),
            "strict",
        )

    def test_cvxpy_contour_step_constraint_uses_diagnostic_keys(self) -> None:
        previous_cvxpy = sys.modules.get("cvxpy")
        sys.modules["cvxpy"] = _FakeCvxpy
        try:
            _FakeVariable._created.clear()
            _FakeProblem.last_solve_kwargs = None
            problem = SimpleNamespace(
                reducer=SimpleNamespace(expansion_matrix=np.eye(1)),
                part_vertex_counts=(),
                part_current_density_mats=(),
            )
            config = SimpleNamespace(
                regularization_auto_range_max_contour_step_A=2.0,
                levels=3,
                contourability_validation="off",
                contourability_repair_enabled=False,
                pot_offset_factor=0.25,
                surface_current_density_limit_A_per_m=0.0,
                max_wire_current_A=0.0,
                max_current_A=0.0,
                min_wire_distance_m=0.0,
                constrained_optimization_solver="auto",
                constrained_optimization_tolerance=1.0e-7,
                constrained_optimization_max_iterations=100,
                constrained_optimization_threads=1,
                progress_flag=True,
            )
            block = ResidualConstraintBlock(
                name="dsv",
                matrix=np.eye(1),
                target=np.zeros(1, dtype=float),
                allowed_residual=1.0,
                reference_norm=1.0,
                relative_tolerance_percent=1.0,
                absolute_tolerance=0.0,
                metric="norm",
            )

            solution, diagnostics = _solve_constrained_quadratic_cvxpy(problem, np.eye(1), (block,), config)
        finally:
            if previous_cvxpy is None:
                sys.modules.pop("cvxpy", None)
            else:
                sys.modules["cvxpy"] = previous_cvxpy

        self.assertEqual(solution.tolist(), [0.0])
        self.assertIsNotNone(_FakeProblem.last_solve_kwargs)
        self.assertTrue(_FakeProblem.last_solve_kwargs["verbose"])
        self.assertEqual(_FakeProblem.last_solve_kwargs["max_threads"], 1)
        self.assertEqual(diagnostics["problem_unknown_count"], 1)
        self.assertEqual(diagnostics["residual_constraint_block_count"], 1)
        self.assertEqual(diagnostics["residual_constraint_row_count"], 1)
        self.assertTrue(diagnostics["residual_constraint_normalization_enabled"])
        self.assertAlmostEqual(diagnostics["residual_constraint_normalization_min"], 1.0)
        self.assertAlmostEqual(diagnostics["residual_constraint_normalization_max"], 1.0)
        self.assertAlmostEqual(diagnostics["objective_normalization_scale"], 1.0)
        self.assertAlmostEqual(diagnostics["contour_step_constraint_normalization_A"], 5.0)
        self.assertTrue(diagnostics["solver_verbose"])
        self.assertEqual(diagnostics["solver_thread_count_requested"], 1)
        self.assertEqual(diagnostics["solver_thread_count_applied"], 1)
        self.assertEqual(diagnostics["solver_thread_option"], "max_threads")
        self.assertIs(diagnostics["contour_step_constraint_enabled"], True)
        self.assertGreater(diagnostics["contour_step_constraint_max_range_A"], 0.0)

    def test_cvxpy_uses_max_wire_current_and_surface_current_density_constraints(self) -> None:
        previous_cvxpy = sys.modules.get("cvxpy")
        sys.modules["cvxpy"] = _FakeCvxpy
        try:
            _FakeVariable._created.clear()
            problem = SimpleNamespace(
                reducer=SimpleNamespace(expansion_matrix=np.eye(1)),
                part_vertex_counts=(1,),
                part_current_density_mats=(np.ones((1, 2, 3), dtype=float),),
                part_faces=(np.empty((0, 3), dtype=np.int64),),
                part_face_basis_currents=(np.empty((0, 3, 3), dtype=float),),
            )
            config = SimpleNamespace(
                regularization_auto_range_max_contour_step_A=0.0,
                levels=3,
                contourability_validation="off",
                contourability_repair_enabled=False,
                pot_offset_factor=0.25,
                contour_discretization_method="engineering",
                engineering_contour_max_levels=12,
                engineering_contour_step_safety_factor=0.5,
                surface_current_density_limit_A_per_m=1500.0,
                max_wire_current_A=8.0,
                max_current_A=0.0,
                min_wire_distance_m=0.002,
                constrained_optimization_solver="auto",
                constrained_optimization_tolerance=1.0e-7,
                constrained_optimization_max_iterations=100,
            )
            block = ResidualConstraintBlock(
                name="dsv",
                matrix=np.eye(1),
                target=np.zeros(1, dtype=float),
                allowed_residual=1.0,
                reference_norm=1.0,
                relative_tolerance_percent=1.0,
                absolute_tolerance=0.0,
                metric="norm",
            )

            _solution, diagnostics = _solve_constrained_quadratic_cvxpy(problem, np.eye(1), (block,), config)
        finally:
            if previous_cvxpy is None:
                sys.modules.pop("cvxpy", None)
            else:
                sys.modules["cvxpy"] = previous_cvxpy

        self.assertIs(diagnostics["contour_step_constraint_enabled"], True)
        self.assertEqual(diagnostics["contour_step_constraint_source"], "max_wire_current")
        self.assertAlmostEqual(diagnostics["contour_step_constraint_max_step_A"], 4.0)
        self.assertEqual(diagnostics["contour_step_constraint_level_count"], 12)
        self.assertIs(diagnostics["surface_current_density_constraint_enabled"], True)
        self.assertAlmostEqual(diagnostics["surface_current_density_constraint_limit_A_per_m"], 1500.0)
        self.assertAlmostEqual(diagnostics["surface_current_density_constraint_derived_limit_A_per_m"], 2000.0)
        self.assertAlmostEqual(diagnostics["surface_current_density_constraint_effective_limit_A_per_m"], 1500.0)
        self.assertEqual(diagnostics["surface_current_density_constraint_source"], "explicit")
        self.assertEqual(diagnostics["surface_current_density_constraint_count"], 2)

    def test_cvxpy_expands_derived_spacing_limit_into_density_soc_constraints(self) -> None:
        previous_cvxpy = sys.modules.get("cvxpy")
        sys.modules["cvxpy"] = _FakeCvxpy
        try:
            _FakeVariable._created.clear()
            problem = SimpleNamespace(
                reducer=SimpleNamespace(expansion_matrix=np.eye(1)),
                part_vertex_counts=(1,),
                part_current_density_mats=(np.ones((1, 2, 3), dtype=float),),
                part_faces=(np.empty((0, 3), dtype=np.int64),),
                part_face_basis_currents=(np.empty((0, 3, 3), dtype=float),),
            )
            config = SimpleNamespace(
                regularization_auto_range_max_contour_step_A=0.0,
                levels=3,
                contourability_validation="off",
                contourability_repair_enabled=False,
                pot_offset_factor=0.25,
                contour_discretization_method="engineering",
                engineering_contour_max_levels=12,
                engineering_contour_step_safety_factor=0.5,
                surface_current_density_limit_A_per_m=0.0,
                max_wire_current_A=8.0,
                max_current_A=0.0,
                min_wire_distance_m=0.002,
                constrained_optimization_solver="auto",
                constrained_optimization_tolerance=1.0e-7,
                constrained_optimization_max_iterations=100,
            )
            block = ResidualConstraintBlock(
                name="dsv",
                matrix=np.eye(1),
                target=np.zeros(1, dtype=float),
                allowed_residual=1.0,
                reference_norm=1.0,
                relative_tolerance_percent=1.0,
                absolute_tolerance=0.0,
                metric="norm",
            )

            _solution, diagnostics = _solve_constrained_quadratic_cvxpy(problem, np.eye(1), (block,), config)
        finally:
            if previous_cvxpy is None:
                sys.modules.pop("cvxpy", None)
            else:
                sys.modules["cvxpy"] = previous_cvxpy

        self.assertIs(diagnostics["surface_current_density_constraint_enabled"], True)
        self.assertAlmostEqual(diagnostics["surface_current_density_constraint_limit_A_per_m"], 2000.0)
        self.assertAlmostEqual(diagnostics["surface_current_density_constraint_derived_limit_A_per_m"], 2000.0)
        self.assertAlmostEqual(diagnostics["surface_current_density_constraint_effective_limit_A_per_m"], 2000.0)
        self.assertEqual(diagnostics["surface_current_density_constraint_source"], "wire_current_and_spacing")
        self.assertEqual(diagnostics["surface_current_density_constraint_count"], 2)

    def test_target_error_retry_tie_break_ignores_roundoff_scale_violation_delta(self) -> None:
        incumbent = StreamFunctionSolution(
            np.zeros(1),
            {
                "target_error_constraint_met": False,
                "selected_lambda": 1.0e-14,
                "target_error_constraint_resistance_norm": 3.0,
            },
        )
        candidate = StreamFunctionSolution(
            np.zeros(1),
            {
                "target_error_constraint_met": False,
                "selected_lambda": 9.0e-13,
                "target_error_constraint_resistance_norm": 2.0,
            },
        )

        self.assertTrue(
            _target_error_retry_solution_is_better(
                candidate,
                0.5618690606367523,
                incumbent,
                0.5618690606367589,
            )
        )
        self.assertFalse(
            _target_error_retry_solution_is_better(
                incumbent,
                0.57,
                candidate,
                0.5618690606367523,
            )
        )


if __name__ == "__main__":
    unittest.main()
