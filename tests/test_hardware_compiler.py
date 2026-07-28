from __future__ import annotations

import unittest

import numpy as np

from coilgen import (
    CoilDesignProblem,
    CoilPart,
    CoilSupportSpec,
    ConductorSpec,
    DesignMetadata,
    HardwareCompilerOptions,
    HardwareConstrainedCoilDesignSolver,
    HardwareSpec,
    LayoutSpec,
    Mesh,
    OptimizationSpec,
    SupportMesh,
    TargetField,
    TargetFieldSpec,
    analyze_stream_function_problem,
    compile_problem_to_hardware_config,
    compile_stream_function_problem,
)
from coilgen.optimization import StreamFunctionOptimizationProblem


class HardwareCompilerTest(unittest.TestCase):
    def test_problem_analysis_reports_rank_limited_target_content(self) -> None:
        problem = _toy_stream_problem(target_bz=np.array([1.0, -1.0]), rank_limited=True)

        analysis = analyze_stream_function_problem(problem)

        self.assertEqual(analysis.target_sample_count, 2)
        # One constant stream-function gauge mode is physically unobservable
        # and is removed before conditioning analysis.
        self.assertEqual(analysis.unknown_count, 2)
        self.assertEqual(analysis.numerical_rank, 1)
        self.assertGreater(analysis.rank_limited_residual_percent, 99.0)

    def test_compiler_prefers_hardware_feasible_candidate_over_unconstrained_fit(self) -> None:
        problem = _toy_stream_problem(target_bz=np.array([10.0, 10.0]), rank_limited=False)
        hardware = HardwareSpec(
            wire_current_A=1.0,
            max_power_W=50.0,
            min_wire_spacing_m=0.1,
            wire_width_m=0.001,
            wire_height_m=0.001,
        )

        report = compile_stream_function_problem(problem, hardware, HardwareCompilerOptions(effort="quick"))

        self.assertEqual(report.status, "best_effort")
        self.assertTrue(report.best_candidate.metrics.satisfies_hard_constraints)
        self.assertLessEqual(report.best_candidate.metrics.power_proxy_W, 50.0 * (1.0 + 1e-7))
        self.assertGreaterEqual(report.best_candidate.metrics.predicted_min_wire_spacing_m, 0.1)
        self.assertIn(
            "power",
            next(
                candidate for candidate in report.candidates if candidate.name == "minimum_error"
            ).metrics.hard_constraint_violations,
        )
        self.assertLess(report.best_candidate.metrics.target_amplitude_fraction, 1.0)
        self.assertIn("achievable_target_amplitude_fraction", report.suggested_hardware_changes)

    def test_hardware_config_bridge_maps_physical_constraints_to_pipeline_settings(self) -> None:
        problem = _high_level_problem()
        hardware = HardwareSpec.from_problem(
            problem,
            wire_current_A=0.75,
            max_power_W=12.0,
            max_current_A=1.0,
            min_wire_spacing_m=0.003,
            min_bend_radius_m=0.02,
        )
        options = HardwareCompilerOptions(field_error_goal_percent=4.0)

        config = compile_problem_to_hardware_config(problem, hardware, options)
        solver_result = HardwareConstrainedCoilDesignSolver(hardware, options, dry_run=True).solve(problem)

        self.assertEqual(config.sf_opt_method, "constrained_quadratic")
        self.assertAlmostEqual(config.min_turn_distance, 0.003)
        self.assertAlmostEqual(config.regularization_auto_range_max_contour_step_A, 0.75)
        self.assertEqual(config.field_evaluation_current_source, "stream_function")
        self.assertAlmostEqual(config.conductor_cross_section_width, 0.002)
        self.assertAlmostEqual(config.conductor_cross_section_height, 0.001)
        self.assertAlmostEqual(config.sweep_min_bend_radius, 0.02)
        self.assertEqual(solver_result.status, "ok")
        self.assertEqual(solver_result.solution.sf_opt_method, "constrained_quadratic")
        self.assertEqual(solver_result.diagnostics["hardware"]["max_power_W"], 12.0)


def _toy_stream_problem(*, target_bz: np.ndarray, rank_limited: bool) -> StreamFunctionOptimizationProblem:
    mesh = Mesh(
        vertices=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    part = CoilPart(mesh=mesh)
    sensitivity = np.zeros((3, 2, 3), dtype=float)
    if rank_limited:
        sensitivity[2] = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    else:
        sensitivity[2] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    part.sensitivity_matrix = sensitivity
    part.resistance_matrix = np.eye(3, dtype=float)
    current_density_mat = np.zeros((3, 1, 3), dtype=float)
    current_density_mat[0, 0, 0] = 1.0
    current_density_mat[1, 0, 1] = 1.0
    part.current_density_mat = current_density_mat

    target = TargetField(
        coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]], dtype=float),
        b=np.column_stack((np.zeros(2), np.zeros(2), np.asarray(target_bz, dtype=float))),
        weights=np.ones(2, dtype=float),
        suppressed=np.zeros(2, dtype=bool),
        target_field_group_inds=np.ones(2, dtype=np.int64),
    )
    return StreamFunctionOptimizationProblem.from_parts([part], target)


def _high_level_problem() -> CoilDesignProblem:
    mesh = Mesh(
        vertices=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    target = TargetFieldSpec.from_arrays(
        np.array([[0.0, 0.0, 0.02]], dtype=float),
        np.array([1.0e-5], dtype=float),
    )
    return CoilDesignProblem(
        target_field=target,
        support=CoilSupportSpec("single_mesh", (SupportMesh(mesh, role="primary"),)),
        conductor=ConductorSpec(width_m=0.002, height_m=0.001, min_turn_spacing_m=0.002),
        optimization=OptimizationSpec(min_turn_spacing_m=0.001),
        layout=LayoutSpec(contour_count=12, min_bend_radius_m=0.01),
        metadata=DesignMetadata(name="hardware_bridge"),
    )


if __name__ == "__main__":
    unittest.main()
