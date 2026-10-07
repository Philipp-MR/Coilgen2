from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coilgen.cohort_sectorization import (
    AdaptiveSectorizationOptions,
    _grow_connected_sectors_to_area,
    _project_full_basis_to_reduced,
    _retain_connected_sector_cores,
    _sector_local_expansion,
    adaptively_sectorize_cohort_channels,
)
from coilgen.cohort_shim import (
    ChannelCurrentLimits,
    CohortFitOptions,
    RegisteredSurfaceProblem,
    RigidTransform,
    ShimScenario,
    SurfaceShimScenario,
)
from coilgen.config import CoilGenConfig
from coilgen.models import Mesh
from coilgen.optimization import StreamFunctionOptimizationProblem


class _OuterVertexReducer:
    def __init__(self, face_count: int) -> None:
        self.expansion_matrix = np.vstack((np.eye(face_count), np.zeros((1, face_count))))

    def expand(self, values: np.ndarray) -> np.ndarray:
        return self.expansion_matrix @ np.asarray(values, dtype=float)


def _four_sector_surface_problem() -> tuple[StreamFunctionOptimizationProblem, np.ndarray]:
    face_count = 12
    angle = 2.0 * np.pi * np.arange(face_count) / face_count
    vertices = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(face_count)))
    vertices = np.vstack((vertices, np.zeros((1, 3))))
    faces = np.asarray(
        [[face_count, index, (index + 1) % face_count] for index in range(face_count)],
        dtype=np.int64,
    )
    mesh = Mesh(vertices, faces)
    reducer = _OuterVertexReducer(face_count)

    # This deliberately simple local operator makes face i depend on reduced
    # stream-function dof i. It isolates support hardening from FEM details.
    current_density = np.zeros((mesh.n_vertices, mesh.n_faces, 3), dtype=float)
    current_density[np.arange(face_count), np.arange(face_count), 0] = 1.0
    sensitivity = np.eye(face_count)
    full_sensitivity = np.zeros((3, face_count, mesh.n_vertices), dtype=float)
    full_sensitivity[2, :, :face_count] = sensitivity
    coords = np.column_stack((np.arange(face_count), np.zeros((face_count, 2))))
    problem = StreamFunctionOptimizationProblem(
        sensitivity_matrix=full_sensitivity,
        resistance_matrix=np.eye(mesh.n_vertices),
        combined_mesh=mesh,
        reducer=reducer,
        target_z=np.zeros(face_count),
        reduced_sensitivity=sensitivity,
        reduced_resistance=np.eye(face_count),
        unweighted_target_z=np.zeros(face_count),
        unweighted_reduced_sensitivity=sensitivity,
        target_coords=coords,
        target_weights=np.ones(face_count),
        target_suppressed=np.zeros(face_count, dtype=bool),
        target_field_group_inds=np.ones(face_count, dtype=np.int64),
        part_vertex_counts=(mesh.n_vertices,),
        part_current_density_mats=(current_density,),
        part_faces=(faces,),
        part_face_basis_currents=(np.zeros((face_count, 3, 3)),),
    )

    # Each channel has one dominant three-face lobe plus deliberately shared
    # low-level activity over the rest of the support.
    basis = np.full((face_count, 4), 0.20, dtype=float)
    for channel in range(4):
        basis[3 * channel : 3 * (channel + 1), channel] = 1.0
    return problem, basis


def _surface_scenarios(
    problem: StreamFunctionOptimizationProblem,
    basis: np.ndarray,
) -> tuple[SurfaceShimScenario, ...]:
    transform = RigidTransform("mask_cad", "scanner", np.eye(4))
    registered = RegisteredSurfaceProblem(problem, transform)
    return tuple(
        SurfaceShimScenario(
            ShimScenario(
                scenario_id=f"s{channel}",
                subject_id=f"subject_{channel}",
                coords_scanner_m=problem.target_coords,
                disturbance_T=-(problem.unweighted_reduced_sensitivity @ basis[:, channel]),
                scanner_from_mask=transform,
            ),
            registered,
        )
        for channel in range(4)
    )


def test_sectorization_options_reject_invalid_partition_controls() -> None:
    with pytest.raises(ValueError, match="activity_damping"):
        AdaptiveSectorizationOptions(activity_damping=1.0)
    with pytest.raises(ValueError, match="minimum_sector_area_fraction"):
        AdaptiveSectorizationOptions(minimum_sector_area_fraction=1.0)
    with pytest.raises(ValueError, match="maximum_localized_overlap_fraction"):
        AdaptiveSectorizationOptions(maximum_localized_overlap_fraction=0.0)
    with pytest.raises(ValueError, match="maximum_hardening_rms_ratio"):
        AdaptiveSectorizationOptions(maximum_hardening_rms_ratio=0.99)
    with pytest.raises(ValueError, match="maximum_normalized_field_condition_number"):
        AdaptiveSectorizationOptions(maximum_normalized_field_condition_number=0.99)
    with pytest.raises(ValueError, match="minimum_discovery_field_projection_ratio"):
        AdaptiveSectorizationOptions(minimum_discovery_field_projection_ratio=0.0)


def test_sector_local_parameterization_makes_every_inactive_face_constant() -> None:
    problem, _basis = _four_sector_surface_problem()
    active = np.zeros(problem.combined_mesh.n_faces, dtype=bool)
    active[:3] = True

    expansion = _sector_local_expansion(problem.combined_mesh, active)

    assert expansion.shape == (problem.combined_mesh.n_vertices, 2)
    for face in problem.combined_mesh.faces[~active]:
        face_rows = expansion[np.asarray(face, dtype=np.int64)].toarray()
        np.testing.assert_allclose(face_rows, np.repeat(face_rows[:1], 3, axis=0))


def test_clearance_islands_are_moved_to_guard_while_terminal_core_is_retained() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    seeds = np.asarray([0, 7], dtype=np.int64)
    guard = np.asarray([False, True, False, False, False, False, False, False])
    adjacency: list[list[tuple[int, float, float]]] = [[] for _ in range(len(labels))]
    for face in range(len(labels) - 1):
        adjacency[face].append((face + 1, 1.0, 1.0))
        adjacency[face + 1].append((face, 1.0, 1.0))

    connected_guard = _retain_connected_sector_cores(
        labels,
        seeds,
        guard,
        np.ones(len(labels)),
        adjacency,
    )

    np.testing.assert_array_equal(
        connected_guard,
        np.asarray([False, True, True, True, False, False, False, False]),
    )


def test_area_balancing_grows_region_without_disconnecting_donor() -> None:
    labels = np.asarray([0, 1, 1, 1, 1, 1], dtype=np.int64)
    seeds = np.asarray([0, 5], dtype=np.int64)
    adjacency: list[list[tuple[int, float, float]]] = [[] for _ in range(len(labels))]
    for face in range(len(labels) - 1):
        adjacency[face].append((face + 1, 1.0, 1.0))
        adjacency[face + 1].append((face, 1.0, 1.0))

    balanced = _grow_connected_sectors_to_area(
        labels,
        seeds,
        np.zeros((len(labels), 2)),
        np.ones(len(labels)),
        adjacency,
        np.asarray([2.0, 2.0]),
        np.asarray([2.0, 2.0]),
        partition_smoothness=0.0,
    )

    np.testing.assert_array_equal(balanced, np.asarray([0, 0, 1, 1, 1, 1]))


def test_full_basis_projection_removes_component_gauge_without_changing_current() -> None:
    problem, _basis = _four_sector_surface_problem()
    density = np.zeros((problem.combined_mesh.n_vertices, problem.combined_mesh.n_faces, 3))
    face_indices = np.arange(problem.combined_mesh.n_faces)
    density[face_indices, face_indices, 0] = 1.0
    density[-1, face_indices, 0] = -1.0
    problem = replace(problem, part_current_density_mats=(density,))
    reduced = np.column_stack((np.linspace(-1.0, 1.0, problem.reduced_sensitivity.shape[1]), np.ones(12)))
    full_with_gauge = problem.reducer.expansion_matrix @ reduced + np.asarray([3.0, -2.0])

    projected = _project_full_basis_to_reduced(problem, full_with_gauge)

    np.testing.assert_allclose(projected, reduced, atol=1.0e-12)


def test_adaptive_sectorization_enforces_disjoint_hard_current_supports() -> None:
    pytest.importorskip("cvxpy")
    problem, initial_basis = _four_sector_surface_problem()
    progress_messages: list[str] = []

    hardened, fit, audit = adaptively_sectorize_cohort_channels(
        _surface_scenarios(problem, initial_basis),
        initial_basis,
        ChannelCurrentLimits.symmetric(4, 2.0),
        CoilGenConfig(target_field_component="Bz"),
        AdaptiveSectorizationOptions(
            enabled=True,
            localization_iterations=1,
            overlap_penalty_initial=0.5,
            resistance_weight=1.0e-5,
            basis_anchor_weight=1.0e-3,
            partition_smoothness=0.05,
            partition_compactness=0.20,
            minimum_sector_area_fraction=0.10,
            hard_support_relative_tolerance=1.0e-7,
            maximum_localized_overlap_fraction=1.0,
            maximum_hardening_rms_ratio=1.0e30,
            minimum_channel_field_norm_ratio=1.0e-6,
            maximum_normalized_field_condition_number=1.0e6,
        ),
        fit_options=CohortFitOptions(center_frequency=False),
        progress_callback=progress_messages.append,
    )

    assert hardened.shape == initial_basis.shape
    assert fit.current_matrix_A.shape == (4, 4)
    assert audit.single_layer_support_enforced
    assert audit.initial_overlap_fraction > 0.0
    assert np.sum(audit.local_dof_counts) < initial_basis.size
    np.testing.assert_array_less(np.sum(audit.active_face_masks, axis=1), 2)
    assert np.all(np.count_nonzero(audit.active_face_masks, axis=0) >= 3)
    assert np.max(audit.max_inactive_density_relative) <= 1.0e-7
    assert audit.normalized_field_condition_number >= 1.0
    assert np.all(audit.channel_field_rms_T_per_A > 0.0)
    assert np.all(audit.active_face_masks[audit.terminal_face_indices, np.arange(4)])
    for channel in range(4):
        np.testing.assert_allclose(
            audit.final_current_density_A_per_m_per_A[~audit.active_face_masks[:, channel], channel],
            0.0,
            atol=1.0e-9,
        )
    assert [entry["stage"] for entry in audit.history] == ["initial", "localization", "hardening"]
    accepted_updates = [entry for entry in audit.history if entry["stage"] == "localization" and entry["accepted"]]
    for entry in accepted_updates:
        assert min(entry["field_norm_ratio"]) >= 0.75 - 1.0e-5
    assert any("Adaptive localization 1/1 complete" in message for message in progress_messages)
    assert any("discovery DOFs per channel" in message for message in progress_messages)
    assert any("local parameterization" in message for message in progress_messages)
    assert any("canonicalization complete" in message for message in progress_messages)
    assert any("Adaptive partition complete" in message for message in progress_messages)
    assert any("Adaptive hardening complete" in message for message in progress_messages)
