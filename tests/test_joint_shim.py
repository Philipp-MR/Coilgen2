from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.joint_shim import (
    JointShimOptimizationProblem,
    JointShimSolveOptions,
    LinearShimActuator,
    save_joint_shim_solution_npz,
    solve_joint_shim_problem,
    _project_stream_function_to_single_mesh_symmetry,
)
from coilgen.models import CoilPart, Mesh, TargetField
from coilgen.optimization import BoundaryReducer, StreamFunctionOptimizationProblem
from coilgen.preoptimization import load_preoptimized_stream_function_values


CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None


class JointShimTest(unittest.TestCase):
    def test_postsolve_projection_enforces_exact_single_mesh_mirror_symmetry(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        problem = SimpleNamespace(
            surface_problem=SimpleNamespace(
                reducer=SimpleNamespace(expansion_matrix=np.eye(3)),
                combined_mesh=mesh,
            )
        )
        config = CoilGenConfig(
            symmetry_reduction_source="single_mesh",
            symmetry_reduction_mirror_plane_point=(0.0, 0.0, 0.0),
            symmetry_reduction_mirror_plane_normal=(1.0, 0.0, 0.0),
            contour_symmetry_geometry_tolerance=1.0e-8,
            contour_symmetry_stream_function_tolerance=1.0e-8,
        )

        projected, diagnostics = _project_stream_function_to_single_mesh_symmetry(
            problem,
            np.array([1.0, 1.2, 0.4]),
            config,
        )

        self.assertAlmostEqual(projected[0], projected[1], places=12)
        self.assertAlmostEqual(projected[2], 0.4, places=12)
        self.assertTrue(diagnostics["joint_shim_stream_symmetry_projection_applied"])
        self.assertLess(
            diagnostics["joint_shim_stream_symmetry_after_normalized_rms"],
            1.0e-12,
        )

    def test_linear_actuator_validates_coordinates_bounds_and_channels(self) -> None:
        actuator = LinearShimActuator.from_scalar_bz(
            "existing_coil",
            np.ones((3, 2), dtype=float),
            channel_labels=("coil_a", "coil_b"),
            coefficient_unit="A",
            lower_bounds=np.array([-2.0, -3.0]),
            upper_bounds=np.array([2.0, 3.0]),
            sample_coords_m=np.zeros((3, 3)),
        )

        self.assertEqual(actuator.channel_count, 2)
        self.assertEqual(actuator.sample_count, 3)
        self.assertEqual(actuator.coefficient_unit, "A")
        self.assertEqual(actuator.channel_labels, ("coil_a", "coil_b"))
        np.testing.assert_allclose(actuator.reference_values, [2.0, 3.0])
        np.testing.assert_allclose(actuator.evaluate_field([1.0, -1.0])[:, 2], 0.0)

        with self.assertRaisesRegex(ValueError, "lower bound"):
            LinearShimActuator.from_scalar_bz(
                "invalid",
                np.ones((3, 1)),
                lower_bounds=np.array([2.0]),
                upper_bounds=np.array([1.0]),
            )

    def test_problem_rejects_misaligned_calibrated_field_rows(self) -> None:
        surface, target, config = _synthetic_surface_problem()
        actuator = LinearShimActuator.from_scalar_bz(
            "misaligned",
            np.array([[0.0], [1.0], [-1.0]]),
            sample_coords_m=np.asarray(target.coords) + np.array([1.0e-3, 0.0, 0.0]),
        )

        with self.assertRaisesRegex(ValueError, "Interpolate"):
            JointShimOptimizationProblem.from_stream_function_problem(
                surface,
                target,
                config,
                (actuator,),
            )

    @unittest.skipUnless(CVXPY_AVAILABLE, "CVXPY is required for the coupled solve")
    def test_joint_surface_and_existing_coil_span_target_together(self) -> None:
        surface, target, config = _synthetic_surface_problem()
        actuator = LinearShimActuator.from_scalar_bz(
            "existing_coil",
            np.array([[0.0], [1.0], [-1.0]]),
            channel_labels=("channel",),
            coefficient_unit="A",
            lower_bounds=np.array([-2.0]),
            upper_bounds=np.array([2.0]),
            reference_values=np.array([1.0]),
            sample_coords_m=target.coords,
        )
        problem = JointShimOptimizationProblem.from_stream_function_problem(
            surface,
            target,
            config,
            (actuator,),
        )

        solution = solve_joint_shim_problem(
            problem,
            config,
            JointShimSolveOptions(lexicographic_secondary=False),
        )

        self.assertEqual(solution.diagnostics["joint_shim_primary_metric"], "std")
        self.assertLess(solution.diagnostics["joint_shim_selected_field_metric_T"], 1.0e-8)
        self.assertAlmostEqual(
            solution.diagnostics["joint_shim_selected_field_metric_T"],
            solution.diagnostics["joint_shim_primary_floor_field_metric_T"],
            places=10,
        )
        self.assertTrue(solution.diagnostics["joint_shim_target_metric_was_primary_objective"])
        self.assertTrue(solution.diagnostics["joint_shim_selected_retains_primary_metric_minimum"])
        self.assertFalse(solution.diagnostics["joint_shim_lexicographic_secondary_used"])
        self.assertAlmostEqual(solution.reduced_stream_function[0], 1.0, places=6)
        self.assertAlmostEqual(solution.actuator_coefficients["existing_coil"][0], 1.0, places=6)
        np.testing.assert_allclose(solution.combined_field_T[:, 2], target.b[:, 2], atol=1.0e-7)
        with tempfile.TemporaryDirectory() as directory:
            path = save_joint_shim_solution_npz(
                Path(directory) / "joint_solution.npz",
                problem,
                solution,
            )
            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_allclose(loaded["stream_function"], solution.full_stream_function)
                np.testing.assert_allclose(
                    loaded["actuator_0_coefficients"],
                    solution.actuator_coefficients["existing_coil"],
                )
                self.assertEqual(str(loaded["actuator_0_name"]), "existing_coil")
            imported = load_preoptimized_stream_function_values(
                path,
                [CoilPart(mesh=surface.combined_mesh)],
            )
            np.testing.assert_allclose(imported[0], solution.full_stream_function)

    @unittest.skipUnless(CVXPY_AVAILABLE, "CVXPY is required for the coupled solve")
    def test_zero_auxiliary_bound_reduces_to_surface_only_solution(self) -> None:
        surface, target, config = _synthetic_surface_problem()
        actuator = LinearShimActuator.from_scalar_bz(
            "disabled_existing_coil",
            np.array([[0.0], [1.0], [-1.0]]),
            lower_bounds=np.array([0.0]),
            upper_bounds=np.array([0.0]),
            reference_values=np.array([1.0]),
            sample_coords_m=target.coords,
        )
        problem = JointShimOptimizationProblem.from_stream_function_problem(
            surface,
            target,
            config,
            (actuator,),
        )

        solution = solve_joint_shim_problem(
            problem,
            config,
            JointShimSolveOptions(lexicographic_secondary=False),
        )

        self.assertAlmostEqual(solution.actuator_coefficients["disabled_existing_coil"][0], 0.0, places=8)
        self.assertGreater(solution.diagnostics["joint_shim_selected_field_metric_T"], 0.1)

    @unittest.skipUnless(CVXPY_AVAILABLE, "CVXPY is required for the coupled solve")
    def test_optional_joint_surface_smoothing_uses_biharmonic_secondary_objective(self) -> None:
        surface, target, config = _synthetic_redundant_surface_problem()
        actuator = LinearShimActuator.from_scalar_bz(
            "disabled_auxiliary",
            np.zeros((len(target.coords), 1), dtype=float),
            lower_bounds=np.array([0.0]),
            upper_bounds=np.array([0.0]),
            reference_values=np.array([1.0]),
            sample_coords_m=target.coords,
        )
        problem = JointShimOptimizationProblem.from_stream_function_problem(
            surface,
            target,
            config,
            (actuator,),
        )

        solution = solve_joint_shim_problem(
            problem,
            config,
            JointShimSolveOptions(
                lexicographic_secondary=True,
                residual_relative_slack=0.0,
                residual_absolute_slack_field_T=1.0e-8,
                surface_cost_weight=0.0,
                surface_smoothness_cost_weight=1.0,
                default_actuator_cost_weight=0.0,
            ),
        )

        diagnostics = solution.diagnostics
        self.assertTrue(diagnostics["joint_shim_lexicographic_secondary_used"])
        self.assertEqual(diagnostics["joint_shim_surface_smoothness_cost_weight"], 1.0)
        self.assertLess(
            diagnostics["joint_shim_selected_surface_biharmonic_energy_A2_per_m2"],
            0.1 * diagnostics["joint_shim_primary_surface_biharmonic_energy_A2_per_m2"],
        )
        self.assertLessEqual(
            diagnostics["joint_shim_selected_field_metric_T"],
            diagnostics["joint_shim_primary_floor_field_metric_T"] + 1.1e-8,
        )


def _synthetic_surface_problem() -> tuple[StreamFunctionOptimizationProblem, TargetField, CoilGenConfig]:
    coords = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    surface_bz = np.array([1.0, -1.0, 0.0], dtype=float)
    target_bz = np.array([1.0, 0.0, -1.0], dtype=float)
    target_b = np.zeros((3, 3), dtype=float)
    target_b[:, 2] = target_bz
    target = TargetField(
        coords=coords,
        b=target_b,
        weights=np.ones(3, dtype=float),
        suppressed=np.zeros(3, dtype=bool),
        target_field_group_inds=np.ones(3, dtype=np.int64),
    )
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
    sensitivity = np.zeros((3, 3, 1), dtype=float)
    sensitivity[2, :, 0] = surface_bz
    surface = StreamFunctionOptimizationProblem(
        sensitivity_matrix=sensitivity,
        resistance_matrix=np.eye(1),
        combined_mesh=mesh,
        reducer=reducer,
        target_z=target_bz.copy(),
        reduced_sensitivity=surface_bz[:, None],
        reduced_resistance=np.eye(1),
        unweighted_target_z=target_bz.copy(),
        unweighted_reduced_sensitivity=surface_bz[:, None],
        target_coords=coords,
        target_weights=np.ones(3, dtype=float),
        target_suppressed=np.zeros(3, dtype=bool),
        target_field_group_inds=np.ones(3, dtype=np.int64),
        target_b=target_b,
    )
    config = CoilGenConfig(
        target_field_component="Bz",
        target_error_constraint_mode="separate",
        target_error_constraint_metric="std",
        target_error_std_limit_T=1.0,
        target_error_tolerance_percent=0.0,
        regularization_auto_range_max_contour_step_A=0.0,
        max_wire_current_A=0.0,
        max_current_A=0.0,
        min_wire_distance_m=0.0,
        surface_current_density_limit_A_per_m=0.0,
        constrained_optimization_solver="CLARABEL",
        constrained_optimization_tolerance=1.0e-8,
        constrained_optimization_max_iterations=1000,
        progress_flag=False,
    )
    return surface, target, config


def _synthetic_redundant_surface_problem() -> tuple[
    StreamFunctionOptimizationProblem,
    TargetField,
    CoilGenConfig,
]:
    surface, target, config = _synthetic_surface_problem()
    surface_bz = np.asarray(surface.unweighted_reduced_sensitivity[:, 0], dtype=float)
    sensitivity = np.zeros((3, len(surface_bz), 2), dtype=float)
    sensitivity[2, :, 0] = surface_bz
    sensitivity[2, :, 1] = surface_bz
    mesh = Mesh(
        vertices=np.zeros((2, 3), dtype=float),
        faces=np.array([[0, 1, 1]], dtype=np.int64),
    )
    reducer = BoundaryReducer(
        full_node_count=2,
        boundary_nodes=[],
        non_boundary_nodes=np.arange(2),
        expansion_matrix=np.eye(2),
    )
    redundant_surface = StreamFunctionOptimizationProblem(
        sensitivity_matrix=sensitivity,
        resistance_matrix=np.eye(2),
        combined_mesh=mesh,
        reducer=reducer,
        target_z=np.asarray(surface.target_z, dtype=float),
        reduced_sensitivity=np.column_stack((surface_bz, surface_bz)),
        reduced_resistance=np.eye(2),
        unweighted_target_z=np.asarray(surface.unweighted_target_z, dtype=float),
        unweighted_reduced_sensitivity=np.column_stack((surface_bz, surface_bz)),
        target_coords=np.asarray(surface.target_coords, dtype=float),
        target_weights=np.asarray(surface.target_weights, dtype=float),
        target_suppressed=np.asarray(surface.target_suppressed, dtype=bool),
        target_field_group_inds=np.asarray(surface.target_field_group_inds, dtype=np.int64),
        target_b=np.asarray(surface.target_b, dtype=float),
        reduced_smoothness_metric=np.diag([100.0, 1.0]),
    )
    return redundant_surface, target, config


if __name__ == "__main__":
    unittest.main()
