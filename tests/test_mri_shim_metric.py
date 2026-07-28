from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.field_units import PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
from coilgen.joint_shim import (
    JointShimOptimizationProblem,
    JointShimSolveOptions,
    LinearShimActuator,
    solve_joint_shim_problem,
)
from coilgen.models import Mesh, TargetField
from coilgen.mri_shim_metric import MRIShimMetricModel, MRIShimMetricOptions
from coilgen.optimization import BoundaryReducer, StreamFunctionOptimizationProblem


CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None


class MRIShimMetricTest(unittest.TestCase):
    def test_metric_removes_frequency_offset_and_reproduces_linear_gradient(self) -> None:
        coords = _cube_grid()
        options = MRIShimMetricOptions(
            mode="report",
            cvar_alpha=0.95,
            residual_rms_weight=1.0,
            residual_cvar_weight=0.25,
            spatial_gradient_weight=0.10,
            derivative_neighbor_count=18,
        )
        model = MRIShimMetricModel.from_samples(coords, np.ones(len(coords)), options)
        frequency_hz = 37.0 + 100.0 * coords[:, 0] - 40.0 * coords[:, 2]
        residual_T = frequency_hz / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
        metrics = model.evaluate(residual_T)
        shifted = model.evaluate(residual_T + 800.0 / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T)

        self.assertAlmostEqual(metrics["mri_shim_frequency_adjustment_Hz"], 37.0, places=9)
        self.assertAlmostEqual(
            metrics["mri_shim_gradient_rms_Hz_per_m"],
            np.hypot(100.0, 40.0),
            places=7,
        )
        self.assertAlmostEqual(
            metrics["mri_shim_composite_score_Hz"],
            shifted["mri_shim_composite_score_Hz"],
            places=8,
        )

    def test_isotropic_gradient_term_is_rotation_invariant(self) -> None:
        coords = _cube_grid()
        angle = np.deg2rad(31.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        options = MRIShimMetricOptions(mode="report", derivative_neighbor_count=18)
        original = MRIShimMetricModel.from_samples(coords, None, options)
        rotated = MRIShimMetricModel.from_samples(coords @ rotation.T, None, options)
        frequency_hz = 20.0 * coords[:, 0] - 13.0 * coords[:, 1] + 8.0 * coords[:, 2]
        residual_T = frequency_hz / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T

        original_metrics = original.evaluate(residual_T)
        rotated_metrics = rotated.evaluate(residual_T)
        self.assertAlmostEqual(
            original_metrics["mri_shim_scaled_gradient_rms_Hz"],
            rotated_metrics["mri_shim_scaled_gradient_rms_Hz"],
            places=8,
        )

    @unittest.skipUnless(CVXPY_AVAILABLE, "CVXPY is required for the active metric solve")
    def test_joint_solver_uses_active_mri_metric(self) -> None:
        surface, target, config = _joint_problem()
        actuator = LinearShimActuator.from_scalar_bz(
            "existing_shim",
            target.coords[:, 1:2] / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T,
            channel_labels=("y",),
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

        self.assertEqual(
            solution.diagnostics["joint_shim_primary_metric"],
            "mri_acquisition_agnostic_risk",
        )
        self.assertLess(solution.diagnostics["joint_shim_selected_mri_score_Hz"], 1.0e-5)
        self.assertAlmostEqual(solution.reduced_stream_function[0], 1.0, places=5)
        self.assertAlmostEqual(solution.actuator_coefficients["existing_shim"][0], 1.0, places=5)


def _cube_grid() -> np.ndarray:
    values = np.linspace(-0.01, 0.01, 3)
    return np.asarray(
        [[x, y, z] for x in values for y in values for z in values],
        dtype=float,
    )


def _joint_problem() -> tuple[StreamFunctionOptimizationProblem, TargetField, CoilGenConfig]:
    coords = _cube_grid()
    surface_bz = coords[:, 0] / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
    actuator_bz = coords[:, 1] / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
    target_bz = surface_bz + actuator_bz
    target_b = np.zeros((len(coords), 3), dtype=float)
    target_b[:, 2] = target_bz
    target = TargetField(
        coords=coords,
        b=target_b,
        weights=np.ones(len(coords), dtype=float),
        suppressed=np.zeros(len(coords), dtype=bool),
        target_field_group_inds=np.ones(len(coords), dtype=np.int64),
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
    sensitivity = np.zeros((3, len(coords), 1), dtype=float)
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
        target_weights=np.ones(len(coords), dtype=float),
        target_suppressed=np.zeros(len(coords), dtype=bool),
        target_field_group_inds=np.ones(len(coords), dtype=np.int64),
        target_b=target_b,
    )
    config = CoilGenConfig(
        target_field_component="Bz",
        target_error_constraint_mode="separate",
        target_error_constraint_metric="std",
        target_error_std_limit_T=1.0,
        target_error_tolerance_percent=0.0,
        mri_shim_metric_mode="optimize",
        mri_shim_cvar_alpha=0.95,
        mri_shim_residual_rms_weight=1.0,
        mri_shim_residual_cvar_weight=0.25,
        mri_shim_spatial_gradient_weight=0.10,
        mri_shim_derivative_neighbor_count=18,
        regularization_auto_range_max_contour_step_A=0.0,
        max_wire_current_A=0.0,
        max_current_A=0.0,
        min_wire_distance_m=0.0,
        surface_current_density_limit_A_per_m=0.0,
        constrained_optimization_solver="CLARABEL",
        constrained_optimization_tolerance=1.0e-9,
        constrained_optimization_max_iterations=2000,
        progress_flag=False,
    )
    return surface, target, config


if __name__ == "__main__":
    unittest.main()
