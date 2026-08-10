from __future__ import annotations

import numpy as np
import pytest

from coilgen.cohort_shim import (
    ChannelCurrentLimits,
    CohortChannelDesignOptions,
    CohortFitOptions,
    PopulationShimDataset,
    RegisteredSurfaceProblem,
    RigidTransform,
    ShimScenario,
    SurfaceShimScenario,
    calibrate_pod_modes,
    design_cohort_channels,
    fit_cohort_currents,
    fit_scenario_currents,
    resistance_weighted_pod,
    save_cohort_channel_stream_functions,
    solve_reference_stream_functions,
)
from coilgen.config import CoilGenConfig
from coilgen.optimization import StreamFunctionOptimizationProblem


class _IdentityReducer:
    def __init__(self, size: int) -> None:
        self.expansion_matrix = np.eye(size)

    def expand(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)


def _identity_transform() -> RigidTransform:
    return RigidTransform("mask_cad", "scanner", np.eye(4))


def _scenario(disturbance: np.ndarray, scenario_id: str = "s0") -> ShimScenario:
    coords = np.column_stack((np.arange(len(disturbance), dtype=float), np.zeros((len(disturbance), 2))))
    return ShimScenario(
        scenario_id=scenario_id,
        subject_id="subject",
        coords_scanner_m=coords,
        disturbance_T=np.asarray(disturbance, dtype=float),
        scanner_from_mask=_identity_transform(),
    )


def _surface_problem(coords: np.ndarray, matrix: np.ndarray) -> StreamFunctionOptimizationProblem:
    dof_count = matrix.shape[1]
    return StreamFunctionOptimizationProblem(
        sensitivity_matrix=np.zeros((3, len(coords), dof_count)),
        resistance_matrix=np.eye(dof_count),
        combined_mesh=None,
        reducer=_IdentityReducer(dof_count),
        target_z=np.zeros(len(coords)),
        reduced_sensitivity=np.asarray(matrix, dtype=float),
        reduced_resistance=np.eye(dof_count),
        unweighted_target_z=np.zeros(len(coords)),
        unweighted_reduced_sensitivity=np.asarray(matrix, dtype=float),
        target_coords=np.asarray(coords, dtype=float),
        target_weights=np.ones(len(coords)),
        target_suppressed=np.zeros(len(coords), dtype=bool),
        target_field_group_inds=np.ones(len(coords), dtype=np.int64),
        part_vertex_counts=(),
        part_current_density_mats=(),
        part_faces=(),
        part_face_basis_currents=(),
    )


def test_rigid_transform_uses_explicit_source_to_target_convention() -> None:
    matrix = np.eye(4)
    matrix[:3, 3] = (0.1, -0.2, 0.3)
    transform = RigidTransform("mask_cad", "scanner", matrix)
    points = np.asarray([[1.0, 2.0, 3.0]])

    scanner = transform.apply_points(points)

    np.testing.assert_allclose(scanner, [[1.1, 1.8, 3.3]])
    np.testing.assert_allclose(transform.inverse().apply_points(scanner), points)


def test_scenario_current_fit_enforces_two_ampere_bounds() -> None:
    scenario = _scenario(np.asarray([3.0, 0.0]))
    fields = np.asarray([[1.0], [0.0]])
    limits = ChannelCurrentLimits.symmetric(1, 2.0)

    result = fit_scenario_currents(
        scenario,
        fields,
        limits,
        options=CohortFitOptions(center_frequency=False),
    )

    np.testing.assert_allclose(result.channel_currents_A, [-2.0], atol=1.0e-8)
    np.testing.assert_allclose(result.residual_T, [1.0, 0.0], atol=1.0e-8)
    assert result.saturated_channels.tolist() == [True]


def test_centered_fit_reports_the_global_frequency_offset() -> None:
    scenario = _scenario(np.asarray([3.0, 1.0]))

    result = fit_scenario_currents(
        scenario,
        np.asarray([[1.0], [0.0]]),
        ChannelCurrentLimits.symmetric(1, 2.0),
    )

    np.testing.assert_allclose(result.channel_currents_A, [-2.0], atol=1.0e-8)
    assert result.frequency_offset_T == pytest.approx(-1.0)
    np.testing.assert_allclose(result.residual_T, [0.0, 0.0], atol=1.0e-8)


def test_cohort_fit_keeps_scenario_rows_and_reports_saturation() -> None:
    scenarios = (_scenario([1.0, 0.0], "s0"), _scenario([3.0, 0.0], "s1"))
    dataset = PopulationShimDataset(scenarios)
    fields = {scenario.scenario_id: np.asarray([[1.0], [0.0]]) for scenario in scenarios}

    result = fit_cohort_currents(
        dataset,
        fields,
        ChannelCurrentLimits.symmetric(1, 2.0),
        options=CohortFitOptions(center_frequency=False),
    )

    assert result.current_matrix_A.shape == (2, 1)
    assert result.saturation_fraction == pytest.approx(0.5)
    assert result.summary()["scenario_count"] == 2


def test_power_constraint_is_applied_to_simultaneous_channel_currents() -> None:
    pytest.importorskip("cvxpy")
    scenario = _scenario([2.0, 2.0])
    fields = np.eye(2)
    limits = ChannelCurrentLimits.symmetric(
        2,
        2.0,
        resistance_ohm=(1.0, 1.0),
        total_power_limit_W=1.0,
    )

    result = fit_scenario_currents(
        scenario,
        fields,
        limits,
        options=CohortFitOptions(center_frequency=False),
    )

    assert result.power_W is not None
    assert result.power_W <= 1.0 + 1.0e-6
    np.testing.assert_allclose(result.channel_currents_A, [-np.sqrt(0.5), -np.sqrt(0.5)], atol=2.0e-4)


def test_resistance_weighted_pod_is_orthonormal_in_physical_metric() -> None:
    resistance = np.diag([1.0, 4.0, 9.0])
    references = np.asarray(
        [
            [3.0, -3.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.1, -0.1],
        ]
    )

    pod = resistance_weighted_pod(references, resistance, 2)

    np.testing.assert_allclose(pod.modes.T @ resistance @ pod.modes, np.eye(2), atol=1.0e-10)
    assert pod.resistance_orthonormality_error < 1.0e-10
    assert pod.cumulative_energy_fraction[-1] > 0.99


def test_resistance_weighted_pod_rejects_more_channels_than_cohort_rank() -> None:
    references = np.asarray([[1.0, -1.0, 2.0], [2.0, -2.0, 4.0]])

    with pytest.raises(ValueError, match="numerical rank 1"):
        resistance_weighted_pod(references, np.eye(2), 2)


def test_pod_calibration_uses_cohort_quantile_and_two_ampere_limit() -> None:
    resistance = np.eye(2)
    references = np.asarray([[1.0, -1.0, 2.0, -2.0], [1.0, 1.0, -1.0, -1.0]])
    pod = resistance_weighted_pod(references, resistance, 2)
    limits = ChannelCurrentLimits.symmetric(2, 2.0)

    calibrated = calibrate_pod_modes(
        pod,
        references,
        resistance,
        limits,
        coefficient_quantile=1.0,
        preferred_current_fraction=1.0,
    )

    np.testing.assert_allclose(np.max(np.abs(calibrated.calibrated_currents_A), axis=1), 2.0, atol=1.0e-10)
    assert calibrated.quantile_within_current_limit.tolist() == [True, True]


def test_registered_surface_problem_rejects_coordinate_or_pose_mismatch() -> None:
    scenario = _scenario([1.0, 2.0])
    problem = _surface_problem(scenario.coords_scanner_m, np.eye(2))
    registered = RegisteredSurfaceProblem(problem, _identity_transform())
    assert SurfaceShimScenario(scenario, registered).problem is problem

    shifted_matrix = np.eye(4)
    shifted_matrix[0, 3] = 0.01
    shifted_registration = RigidTransform("mask_cad", "scanner", shifted_matrix)
    with pytest.raises(ValueError, match="different mask-to-scanner registrations"):
        SurfaceShimScenario(scenario, RegisteredSurfaceProblem(problem, shifted_registration))

    shifted_problem = _surface_problem(scenario.coords_scanner_m + 0.001, np.eye(2))
    with pytest.raises(ValueError, match="coordinates do not match"):
        SurfaceShimScenario(scenario, RegisteredSurfaceProblem(shifted_problem, _identity_transform()))


def test_reference_solve_rejects_b0_component_mismatch() -> None:
    scenario = _scenario([1.0, -1.0])
    problem = _surface_problem(scenario.coords_scanner_m, np.eye(2))
    surface_scenario = SurfaceShimScenario(
        scenario,
        RegisteredSurfaceProblem(problem, _identity_transform()),
    )

    with pytest.raises(ValueError, match="B0 direction does not match"):
        solve_reference_stream_functions(
            (surface_scenario,),
            CoilGenConfig(target_field_component="Bx"),
        )


def test_end_to_end_four_channel_design_and_contour_export(tmp_path) -> None:
    sensitivity = np.vstack((np.eye(4), -np.ones((1, 4))))
    coords = np.column_stack((np.arange(5, dtype=float), np.zeros((5, 2))))
    problem = _surface_problem(coords, sensitivity)
    registered = RegisteredSurfaceProblem(problem, _identity_transform())
    reference_commands = np.eye(4)
    surface_scenarios = tuple(
        SurfaceShimScenario(
            ShimScenario(
                scenario_id=f"s{index}",
                subject_id="subject_a" if index < 3 else "subject_b",
                coords_scanner_m=coords,
                disturbance_T=-(sensitivity @ reference_commands[:, index]),
                scanner_from_mask=_identity_transform(),
            ),
            registered,
        )
        for index in range(4)
    )
    config = CoilGenConfig(
        sf_opt_method="tikhonov",
        regularization_selection_method="fixed",
        tikonov_reg_factor=1.0e-8,
        target_field_component="Bz",
    )

    progress_messages: list[str] = []
    result = design_cohort_channels(
        surface_scenarios,
        config,
        CohortChannelDesignOptions(
            channel_count=4,
            current_limit_A=2.0,
            coefficient_quantile=1.0,
            preferred_current_fraction=0.99,
        ),
        progress_callback=progress_messages.append,
    )

    assert result.channel_basis_per_A.shape == (4, 4)
    assert result.full_channel_basis_per_A.shape == (4, 4)
    assert all(item["cohort_shared_factorization"] for item in result.reference.diagnostics)
    assert all(item["cohort_shared_rhs_count"] == 4 for item in result.reference.diagnostics)
    target_matrix = np.column_stack([item.scenario.cancellation_target_T for item in surface_scenarios])
    regularization_lambda = config.tikonov_reg_factor * problem.regularization_scale
    expected_reference = np.linalg.solve(
        sensitivity.T @ sensitivity + regularization_lambda**2 * problem.reduced_resistance,
        sensitivity.T @ target_matrix,
    )
    np.testing.assert_allclose(result.reference.reduced_stream_functions, expected_reference)
    np.testing.assert_allclose(result.scenario_weights, [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5])
    assert np.max(np.abs(result.fit.current_matrix_A)) <= 2.0 + 1.0e-8
    assert np.max(result.fit.rms_after_T) < 1.0e-8
    assert any("one factorization with 4 right-hand sides" in message for message in progress_messages)
    assert any("Reference assembly complete" in message for message in progress_messages)
    assert any("Resistance eigensystem complete" in message for message in progress_messages)
    assert any("Cohort channel design stage complete" in message for message in progress_messages)

    paths = save_cohort_channel_stream_functions(tmp_path, result)
    assert len(paths) == 4
    with np.load(paths[0]) as channel:
        np.testing.assert_allclose(
            channel["stream_function"],
            2.0 * result.full_channel_basis_per_A[:, 0],
        )
        assert float(channel["contour_current_A"]) == pytest.approx(2.0)
