from __future__ import annotations

import copy
from dataclasses import dataclass
import os
import time
from typing import Any, Sequence

import numpy as np

from .basis import current_density_from_basis
from .config import CoilGenConfig
from .engineering_limits import (
    engineering_contour_max_levels,
    engineering_contouring_enabled,
    engineering_limit_diagnostics,
    engineering_max_contour_step_A,
    engineering_surface_current_density_limit_A_per_m,
    min_wire_distance_m,
)
from .field_components import component_sample_values, field_component_values, sensitivity_component_matrix
from .geometry import combine_meshes
from .gradient_linearity import (
    gradient_linearity_enabled,
    gradient_linearity_reference_scale_T,
    gradient_linearity_region_mask,
)
from .models import CoilPart, Mesh, TargetField
from .optimization_solvers import conic_solver_order
from .progress import ProgressReporter
from .spectral_stream_function import build_spectral_stream_function_basis, cotangent_stiffness_and_lumped_mass
from .symmetry import (
    absolute_tolerance as _symmetry_absolute_tolerance,
    generated_candidate_symmetric_part_pairs as _symmetry_candidate_part_pairs,
    geometry_scale as _symmetry_geometry_scale,
    nearest_indices as _symmetry_nearest_indices,
    normalise_symmetry_mode,
    parity_residual as _symmetry_parity_residual,
    reflect_points as _symmetry_reflect_points,
)


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
class BoundaryReducer:
    full_node_count: int
    boundary_nodes: list[np.ndarray]
    non_boundary_nodes: np.ndarray
    expansion_matrix: np.ndarray

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> "BoundaryReducer":
        boundary_nodes = [np.unique(boundary).astype(np.int64) for boundary in mesh.boundaries]
        boundary_flat = np.concatenate(boundary_nodes) if boundary_nodes else np.empty(0, dtype=np.int64)
        non_boundary_nodes = np.setdiff1d(np.arange(mesh.n_vertices), boundary_flat, assume_unique=False)
        reduced_count = len(boundary_nodes) + len(non_boundary_nodes)
        expansion = np.zeros((mesh.n_vertices, reduced_count), dtype=float)
        for boundary_ind, nodes in enumerate(boundary_nodes):
            expansion[nodes, boundary_ind] = 1.0
        for reduced_ind, node in enumerate(non_boundary_nodes, start=len(boundary_nodes)):
            expansion[node, reduced_ind] = 1.0
        return cls(mesh.n_vertices, boundary_nodes, non_boundary_nodes, expansion)

    def reduce_columns(self, matrix: np.ndarray) -> np.ndarray:
        return matrix @ self.expansion_matrix

    def reduce_square(self, matrix: np.ndarray) -> np.ndarray:
        return self.expansion_matrix.T @ matrix @ self.expansion_matrix

    def expand(self, reduced_vector: np.ndarray) -> np.ndarray:
        return self.expansion_matrix @ reduced_vector


@dataclass
class SymmetryReducer:
    full_node_count: int
    boundary_reducer: BoundaryReducer
    symmetry_expansion_matrix: np.ndarray
    expansion_matrix: np.ndarray
    diagnostics: dict[str, Any]

    @classmethod
    def from_boundary_reducer(
        cls,
        boundary_reducer: BoundaryReducer,
        symmetry_expansion_matrix: np.ndarray,
        diagnostics: dict[str, Any],
    ) -> "SymmetryReducer":
        expansion = boundary_reducer.expansion_matrix @ symmetry_expansion_matrix
        return cls(
            full_node_count=boundary_reducer.full_node_count,
            boundary_reducer=boundary_reducer,
            symmetry_expansion_matrix=symmetry_expansion_matrix,
            expansion_matrix=expansion,
            diagnostics=diagnostics,
        )

    def reduce_columns(self, matrix: np.ndarray) -> np.ndarray:
        return matrix @ self.expansion_matrix

    def reduce_square(self, matrix: np.ndarray) -> np.ndarray:
        return self.expansion_matrix.T @ matrix @ self.expansion_matrix

    def expand(self, reduced_vector: np.ndarray) -> np.ndarray:
        return self.expansion_matrix @ reduced_vector


@dataclass
class GaugeReducer:
    """Remove stream-function directions that are constant on mesh components."""

    full_node_count: int
    base_reducer: BoundaryReducer | SymmetryReducer
    expansion_matrix: np.ndarray
    diagnostics: dict[str, Any]

    @classmethod
    def from_reducer(
        cls,
        mesh: Mesh,
        base_reducer: BoundaryReducer | SymmetryReducer,
    ) -> "GaugeReducer":
        expansion = np.asarray(base_reducer.expansion_matrix, dtype=float)
        _stiffness, mass = cotangent_stiffness_and_lumped_mass(mesh)
        constraints, component_count = _component_mean_gauge_constraints(mesh, mass, expansion)
        free, pivots, pivot_coefficients, rank_tolerance, singular_values = _constraint_nullspace_parameterization(
            constraints
        )
        gauge_count = int(len(pivots))
        if len(free) == 0:
            raise ValueError("Stream-function reduction contains only constant gauge modes.")
        if gauge_count:
            gauge_expansion = expansion[:, free] + expansion[:, pivots] @ pivot_coefficients
            constraint_residual = constraints[:, free] + constraints[:, pivots] @ pivot_coefficients
            max_constraint_residual = float(np.max(np.abs(constraint_residual)))
        else:
            gauge_expansion = expansion.copy()
            max_constraint_residual = 0.0
        return cls(
            full_node_count=base_reducer.full_node_count,
            base_reducer=base_reducer,
            expansion_matrix=gauge_expansion,
            diagnostics={
                "stream_function_gauge_removed": bool(gauge_count > 0),
                "stream_function_gauge_mode_count": gauge_count,
                "stream_function_gauge_unknown_count_before": int(expansion.shape[1]),
                "stream_function_gauge_unknown_count_after": int(gauge_expansion.shape[1]),
                "stream_function_gauge_component_count": int(component_count),
                "stream_function_gauge_constraint_tolerance": float(rank_tolerance),
                "stream_function_gauge_max_constraint_residual": max_constraint_residual,
                "stream_function_gauge_smallest_retained_singular_value": (
                    float(singular_values[gauge_count - 1]) if gauge_count else float("nan")
                ),
            },
        )

    def reduce_columns(self, matrix: np.ndarray) -> np.ndarray:
        return matrix @ self.expansion_matrix

    def reduce_square(self, matrix: np.ndarray) -> np.ndarray:
        return self.expansion_matrix.T @ matrix @ self.expansion_matrix

    def expand(self, reduced_vector: np.ndarray) -> np.ndarray:
        return self.expansion_matrix @ reduced_vector


def _component_mean_gauge_constraints(
    mesh: Mesh,
    mass: np.ndarray,
    expansion: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return area-weighted component-mean constraints in reduced coordinates."""

    labels = _mesh_vertex_component_labels(mesh)
    component_count = int(np.max(labels) + 1) if labels.size else 0
    constraints = np.zeros((component_count, expansion.shape[1]), dtype=float)
    mass_values = np.asarray(mass, dtype=float).reshape(-1)
    for component in range(component_count):
        indices = np.flatnonzero(labels == component)
        weights = mass_values[indices]
        valid = np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            weights = np.ones(len(indices), dtype=float)
        else:
            fallback = float(np.median(weights[valid]))
            weights = np.where(valid, weights, fallback)
        weights = weights / float(np.sum(weights))
        constraints[component] = weights @ expansion[indices]
    return constraints, component_count


def _mesh_vertex_component_labels(mesh: Mesh) -> np.ndarray:
    vertex_count = int(mesh.n_vertices)
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(node: int) -> int:
        root = int(node)
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node]) != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for face in np.asarray(mesh.faces, dtype=np.int64):
        valid = [int(node) for node in face if 0 <= int(node) < vertex_count]
        for node in valid[1:]:
            union(valid[0], node)
    roots = np.asarray([find(node) for node in range(vertex_count)], dtype=np.int64)
    _unique, labels = np.unique(roots, return_inverse=True)
    return np.asarray(labels, dtype=np.int64)


def _constraint_nullspace_parameterization(
    constraints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Parameterize Hx=0 by eliminating a well-conditioned set of pivot columns."""

    matrix = np.asarray(constraints, dtype=float)
    unknown_count = int(matrix.shape[1])
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = np.finfo(float).eps * max(matrix.shape, default=1) * max(largest, 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        return (
            np.arange(unknown_count, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.zeros((0, unknown_count), dtype=float),
            tolerance,
            singular_values,
        )

    column_order = np.argsort(-np.linalg.norm(matrix, axis=0))
    orthonormal_columns: list[np.ndarray] = []
    pivot_list: list[int] = []
    for column_index in column_order:
        residual = matrix[:, int(column_index)].copy()
        for basis_column in orthonormal_columns:
            residual -= float(np.dot(basis_column, residual)) * basis_column
        norm = float(np.linalg.norm(residual))
        if norm <= tolerance:
            continue
        orthonormal_columns.append(residual / norm)
        pivot_list.append(int(column_index))
        if len(pivot_list) == rank:
            break
    if len(pivot_list) != rank:
        raise ValueError("Could not construct an independent stream-function gauge constraint basis.")

    pivots = np.asarray(sorted(pivot_list), dtype=np.int64)
    free = np.setdiff1d(np.arange(unknown_count, dtype=np.int64), pivots, assume_unique=True)
    pivot_matrix = matrix[:, pivots]
    pivot_coefficients = np.linalg.lstsq(pivot_matrix, -matrix[:, free], rcond=None)[0]
    residual = matrix[:, free] + pivot_matrix @ pivot_coefficients
    residual_scale = max(float(np.max(np.abs(matrix))), 1.0)
    if residual.size and float(np.max(np.abs(residual))) > 100.0 * tolerance * residual_scale:
        raise ValueError("Stream-function component-mean gauge elimination is numerically inconsistent.")
    return free, pivots, pivot_coefficients, tolerance, singular_values


@dataclass
class SpectralReducer:
    full_node_count: int
    base_reducer: BoundaryReducer | SymmetryReducer | GaugeReducer
    spectral_basis: np.ndarray
    expansion_matrix: np.ndarray
    diagnostics: dict[str, Any]

    @classmethod
    def from_reducer(
        cls,
        mesh: Mesh,
        base_reducer: BoundaryReducer | SymmetryReducer | GaugeReducer,
        config: CoilGenConfig,
    ) -> "SpectralReducer":
        basis = build_spectral_stream_function_basis(mesh, base_reducer.expansion_matrix, config)
        expansion = base_reducer.expansion_matrix @ basis.basis
        return cls(
            full_node_count=base_reducer.full_node_count,
            base_reducer=base_reducer,
            spectral_basis=basis.basis,
            expansion_matrix=expansion,
            diagnostics=basis.diagnostics,
        )

    def reduce_columns(self, matrix: np.ndarray) -> np.ndarray:
        return matrix @ self.expansion_matrix

    def reduce_square(self, matrix: np.ndarray) -> np.ndarray:
        return self.expansion_matrix.T @ matrix @ self.expansion_matrix

    def expand(self, reduced_vector: np.ndarray) -> np.ndarray:
        return self.expansion_matrix @ reduced_vector


@dataclass(frozen=True)
class ResidualConstraintBlock:
    """One physically meaningful residual constraint in the stream-function solve."""

    name: str
    matrix: np.ndarray
    target: np.ndarray
    allowed_residual: float
    reference_norm: float
    relative_tolerance_percent: float
    absolute_tolerance: float
    rms_field_limit_T: float | None = None
    metric: str = "norm"
    field_limit_T: float | None = None
    field_metric_denominator: float | None = None
    component: str = ""
    weighting: str = "uniform"
    group_size: int = 1


@dataclass
class StreamFunctionOptimizationProblem:
    """Reduced stream-function inverse problem.

    The unknown x is the reduced stream-function vector. Boundary nodes belonging
    to the same mesh boundary share one degree of freedom; the reducer expands x
    back to one stream-function value per mesh vertex.

    A maps x to the selected target-field component, b is the requested target field,
    and R is the reduced resistance/current-energy metric used by the default Tikhonov
    cost function:

        min_x ||A x - b||_2^2 + lambda^2 x^T R x

    New optimization schemes should accept this object and return a
    StreamFunctionSolution.
    """

    sensitivity_matrix: np.ndarray
    resistance_matrix: np.ndarray
    combined_mesh: Mesh
    reducer: BoundaryReducer | SymmetryReducer | GaugeReducer | SpectralReducer
    target_z: np.ndarray
    reduced_sensitivity: np.ndarray
    reduced_resistance: np.ndarray
    unweighted_target_z: np.ndarray | None = None
    unweighted_reduced_sensitivity: np.ndarray | None = None
    target_coords: np.ndarray | None = None
    target_weights: np.ndarray | None = None
    target_suppressed: np.ndarray | None = None
    target_field_group_inds: np.ndarray | None = None
    target_b: np.ndarray | None = None
    target_suppression_normals: np.ndarray | None = None
    target_suppression_area_weights: np.ndarray | None = None
    symmetry_diagnostics: dict[str, Any] | None = None
    reduced_turn_spacing_metric: np.ndarray | None = None
    reduced_smoothness_metric: np.ndarray | None = None
    part_vertex_counts: tuple[int, ...] = ()
    part_current_density_mats: tuple[np.ndarray | None, ...] = ()
    part_faces: tuple[np.ndarray, ...] = ()
    part_face_basis_currents: tuple[np.ndarray, ...] = ()
    part_outward_scores: tuple[float, ...] = ()
    turn_spacing_reference_length: float = 0.0

    @classmethod
    def from_parts(
        cls,
        parts: list[CoilPart],
        target_field: TargetField,
        config: CoilGenConfig | None = None,
    ) -> "StreamFunctionOptimizationProblem":
        sensitivity_matrices = [part.sensitivity_matrix for part in parts if part.sensitivity_matrix is not None]
        resistance_matrices = [part.resistance_matrix for part in parts if part.resistance_matrix is not None]
        if len(sensitivity_matrices) != len(parts):
            raise ValueError("Every coil part must have a sensitivity_matrix before stream-function optimization.")
        if len(resistance_matrices) != len(parts):
            raise ValueError("Every coil part must have a resistance_matrix before stream-function optimization.")

        sensitivity_matrix = np.concatenate(sensitivity_matrices, axis=2)
        resistance_matrix = _block_diag(resistance_matrices)
        combined_mesh = combine_meshes([part.mesh for part in parts])
        boundary_reducer = BoundaryReducer.from_mesh(combined_mesh)
        component = getattr(config, "target_field_component", "Bz") if config is not None else "Bz"
        b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0)) if config is not None else (0.0, 0.0, 1.0)
        component_sensitivity = sensitivity_component_matrix(
            sensitivity_matrix,
            component,
            b0_direction=b0_direction,
        )
        raw_target_z = field_component_values(target_field.b, component, b0_direction=b0_direction).reshape(-1)
        weights = np.asarray(target_field.weights, dtype=float).reshape(-1)
        weights = component_sample_values(weights, component)
        if len(weights) != len(raw_target_z):
            raise ValueError("Target-field weights must match the number of target points.")
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        sqrt_weights = np.sqrt(weights)
        reducer, symmetry_diagnostics = _select_stream_function_reducer(
            parts,
            target_field,
            boundary_reducer,
            component_sensitivity * sqrt_weights[:, None],
            raw_target_z * sqrt_weights,
            config,
        )
        basis_diagnostics = _stream_function_basis_diagnostics(config)
        gauge_reducer = GaugeReducer.from_reducer(combined_mesh, reducer)
        reducer = gauge_reducer
        basis_diagnostics.update(gauge_reducer.diagnostics)
        if config is not None and _spectral_stream_function_basis_requested(config):
            spectral_reducer = SpectralReducer.from_reducer(combined_mesh, reducer, config)
            reducer = spectral_reducer
            basis_diagnostics.update(spectral_reducer.diagnostics)
        if symmetry_diagnostics is not None:
            symmetry_diagnostics.update(basis_diagnostics)
        else:
            symmetry_diagnostics = basis_diagnostics
        unweighted_reduced_sensitivity = reducer.reduce_columns(component_sensitivity)
        # Weighted least squares: min ||sqrt(W) (A x - b)||_2^2 + regularization.
        # The unweighted blocks are kept as well so fixed-error schemes can
        # constrain DSV accuracy and active-shield suppression separately.
        reduced_sensitivity = unweighted_reduced_sensitivity * sqrt_weights[:, None]
        target_z = raw_target_z * sqrt_weights
        reduced_resistance = reducer.reduce_square(resistance_matrix)
        turn_spacing_metric = None
        smoothness_metric = None
        turn_spacing_reference_length = 0.0
        if config is not None and _turn_spacing_metric_requested(config):
            turn_spacing_metric = _reduced_surface_current_density_metric(parts, reducer)
            turn_spacing_reference_length = _median_mesh_edge_length([part.mesh for part in parts])
        if config is not None and _stream_function_smoothness_requested(config):
            smoothness_metric = reducer.reduce_square(_surface_biharmonic_metric(combined_mesh))
        return cls(
            sensitivity_matrix=sensitivity_matrix,
            resistance_matrix=resistance_matrix,
            combined_mesh=combined_mesh,
            reducer=reducer,
            target_z=target_z,
            reduced_sensitivity=reduced_sensitivity,
            reduced_resistance=reduced_resistance,
            unweighted_target_z=raw_target_z,
            unweighted_reduced_sensitivity=unweighted_reduced_sensitivity,
            target_coords=np.asarray(target_field.coords, dtype=float),
            target_weights=weights,
            target_suppressed=component_sample_values(
                np.asarray(target_field.suppressed, dtype=bool), component
            ).astype(bool),
            target_field_group_inds=component_sample_values(
                np.asarray(target_field.target_field_group_inds, dtype=np.int64),
                component,
            ).astype(np.int64),
            target_b=np.asarray(target_field.b, dtype=float),
            target_suppression_normals=None
            if target_field.suppression_normals is None
            else np.asarray(target_field.suppression_normals, dtype=float),
            target_suppression_area_weights=None
            if target_field.suppression_area_weights is None
            else np.asarray(target_field.suppression_area_weights, dtype=float),
            symmetry_diagnostics=symmetry_diagnostics,
            reduced_turn_spacing_metric=turn_spacing_metric,
            reduced_smoothness_metric=smoothness_metric,
            part_vertex_counts=tuple(part.mesh.n_vertices for part in parts),
            part_current_density_mats=tuple(part.current_density_mat for part in parts),
            part_faces=tuple(np.asarray(part.mesh.faces, dtype=np.int64) for part in parts),
            part_face_basis_currents=tuple(_face_local_basis_currents(part) for part in parts),
            part_outward_scores=tuple(_part_outward_score(part) for part in parts),
            turn_spacing_reference_length=turn_spacing_reference_length,
        )

    @property
    def regularization_scale(self) -> float:
        return float(self.reduced_sensitivity.shape[0] / max(self.reduced_sensitivity.shape[1], 1))

    def normal_equation_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.reduced_sensitivity.T @ self.reduced_sensitivity,
            _symmetric_quadratic_matrix(self.reduced_resistance),
            self.reduced_sensitivity.T @ self.target_z,
        )

    def expand(self, reduced_stream_function: np.ndarray) -> np.ndarray:
        return self.reducer.expand(reduced_stream_function)

    def evaluate_full_b_field(self, full_stream_function: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                self.sensitivity_matrix[0] @ full_stream_function,
                self.sensitivity_matrix[1] @ full_stream_function,
                self.sensitivity_matrix[2] @ full_stream_function,
            ]
        )

    def residual_norm(self, reduced_stream_function: np.ndarray) -> float:
        return float(np.linalg.norm(self.reduced_sensitivity @ reduced_stream_function - self.target_z))

    def resistance_norm(self, reduced_stream_function: np.ndarray) -> float:
        return _quadratic_norm(self.reduced_resistance, reduced_stream_function)

    def max_surface_current_density(self, reduced_stream_function: np.ndarray) -> float:
        """Return peak ``|K|`` in A/m for one reduced stream function.

        Population channel design uses this to normalize a POD mode as a
        physical stream function per ampere while respecting the maximum drive
        current.  Keeping the evaluation on the problem guarantees that it uses
        the same reducer and per-part FEM operators as the original solve.
        """

        return _solution_max_surface_current_density(self, self.expand(reduced_stream_function))


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


def _part_outward_score(part: CoilPart) -> float:
    vertices = getattr(getattr(part, "mesh", None), "vertices", None)
    try:
        array = np.asarray(vertices, dtype=float)
    except (TypeError, ValueError):
        return float("nan")
    if array.ndim != 2 or array.shape[1] < 2 or len(array) == 0:
        return float("nan")
    radii = np.linalg.norm(array[:, :2], axis=1)
    finite = radii[np.isfinite(radii)]
    return float(np.mean(finite)) if len(finite) else float("nan")


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


def _constrained_solution_contract_diagnostics(
    problem: StreamFunctionOptimizationProblem,
    reduced_stream_function: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, Any]:
    """Recompute every constrained-solver bound from the returned physical solution."""

    reduced = np.asarray(reduced_stream_function, dtype=float).reshape(-1)
    full = np.asarray(problem.expand(reduced), dtype=float).reshape(-1)
    finite_solution = bool(np.all(np.isfinite(reduced)) and np.all(np.isfinite(full)))
    blocks = _target_error_constraint_blocks(problem, config)
    diagnostics = _residual_constraint_diagnostics(problem, reduced, blocks, config)
    rows: list[dict[str, Any]] = []
    for block in blocks:
        denominator = _constraint_field_metric_denominator(block) if block.metric in {"std", "rms"} else 1.0
        rows.append(
            {
                "name": block.name,
                "value": float(diagnostics[f"{block.name}_constraint_residual_norm_T"]) / denominator,
                "limit": float(block.allowed_residual) / denominator,
                "effective_limit": float(
                    diagnostics[f"{block.name}_constraint_allowed_with_acceptance_residual_norm_T"]
                )
                / denominator,
                "unit": "T/m" if block.name in {"gradient_strength", "cross_gradient"} else "T",
                "met": bool(diagnostics[f"{block.name}_constraint_met"]),
                "available": bool(
                    np.isfinite(diagnostics[f"{block.name}_constraint_residual_norm_T"])
                    and np.isfinite(block.allowed_residual)
                ),
            }
        )
    requested_residual_blocks = {
        "linearity_rms": "linearity_rms_tolerance_percent",
        "linearity_peak": "linearity_peak_tolerance_percent",
        "gradient_strength": "gradient_strength_tolerance_percent",
        "cross_gradient": "linearity_cross_gradient_tolerance_percent",
        "suppression": "suppression_field_rms_limit_T",
        "suppression_peak": "suppression_field_peak_limit_T",
    }
    built_blocks = {block.name for block in blocks}
    for name, parameter in requested_residual_blocks.items():
        if _positive_config_float(config, parameter) <= 0.0 or name in built_blocks:
            continue
        diagnostics[f"{name}_constraint_met"] = False
        rows.append(
            {
                "name": name,
                "value": float("nan"),
                "limit": float("nan"),
                "effective_limit": float("nan"),
                "unit": "T/m" if name in {"gradient_strength", "cross_gradient"} else "T",
                "met": False,
                "available": False,
                "reason": f"Requested {parameter}, but no corresponding residual constraint could be constructed.",
            }
        )

    relative_tolerance = _constraint_acceptance_relative_tolerance(config)
    density_limit = engineering_surface_current_density_limit_A_per_m(config)
    if density_limit > 0.0:
        peak, face_count, reason = _contract_surface_current_density(problem, full)
        available = bool(np.isfinite(peak) and face_count > 0)
        effective_limit = density_limit * (1.0 + relative_tolerance)
        met = bool(available and peak <= effective_limit)
        diagnostics.update(
            {
                "surface_current_density_constraint_enabled": True,
                "surface_current_density_constraint_limit_A_per_m": float(density_limit),
                "surface_current_density_constraint_max_A_per_m": float(peak),
                "surface_current_density_constraint_effective_limit_A_per_m": float(effective_limit),
                "surface_current_density_constraint_count": face_count,
                "surface_current_density_constraint_met": met,
                "surface_current_density_constraint_available": available,
                "surface_current_density_constraint_reason": reason,
            }
        )
        rows.append(
            {
                "name": "surface_current_density",
                "value": peak,
                "limit": density_limit,
                "effective_limit": effective_limit,
                "unit": "A/m",
                "met": met,
                "available": available,
                "reason": reason,
            }
        )

    contour = _constrained_contour_step_limit(config)
    diagnostics.update(contour)
    if contour["contour_step_constraint_enabled"]:
        stream_range = float(np.ptp(full)) if full.size and finite_solution else float("nan")
        limit = float(contour["contour_step_constraint_max_range_A"])
        effective_limit = limit * (1.0 + relative_tolerance)
        available = bool(np.isfinite(stream_range))
        met = bool(available and stream_range <= effective_limit)
        diagnostics.update(
            {
                "contour_step_constraint_range_A": stream_range,
                "contour_step_constraint_effective_max_range_A": effective_limit,
                "contour_step_constraint_met": met,
                "contour_step_constraint_actual_step_A": stream_range
                / float(contour["contour_step_constraint_denominator"]),
            }
        )
        rows.append(
            {
                "name": "contour_step",
                "value": stream_range,
                "limit": limit,
                "effective_limit": effective_limit,
                "unit": "A",
                "met": met,
                "available": available,
            }
        )

    shield = _shield_gradient_opposition_constraint_data(problem, config)
    shield_diagnostics = _shield_gradient_opposition_constraint_diagnostics(shield, reduced)
    diagnostics.update(shield_diagnostics)
    if shield["requested"]:
        value = float(shield_diagnostics["shield_gradient_opposition_value_T_per_m"])
        limit = float(shield["limit_T_per_m"])
        available = bool(shield["enabled"] and np.isfinite(value) and np.isfinite(limit))
        tolerance = relative_tolerance * abs(float(shield["scale_T_per_m"]))
        effective_limit = limit + tolerance
        met = bool(available and value <= effective_limit)
        diagnostics["shield_gradient_opposition_constraint_met"] = met
        diagnostics["shield_gradient_opposition_effective_limit_T_per_m"] = effective_limit
        rows.append(
            {
                "name": "shield_gradient_opposition",
                "value": value,
                "limit": limit,
                "effective_limit": effective_limit,
                "unit": "T/m",
                "met": met,
                "available": available,
                "reason": shield["reason"],
            }
        )

    failed = [str(row["name"]) for row in rows if not row["met"]]
    if not finite_solution:
        failed.append("finite_stream_function")
    diagnostics.update(
        {
            "constrained_solver_physical_constraints_met": not failed,
            "constrained_solver_constraint_data_available": all(row["available"] for row in rows),
            "constrained_solver_solution_finite": finite_solution,
            "constrained_solver_failed_constraints": failed,
            "constrained_solver_contract_constraint_diagnostics": rows,
            "constrained_solver_hardware_acceptance_relative_tolerance": relative_tolerance,
        }
    )
    return diagnostics


def _contract_surface_current_density(
    problem: StreamFunctionOptimizationProblem,
    full_stream_function: np.ndarray,
) -> tuple[float, int, str]:
    counts = problem.part_vertex_counts
    records = (problem.part_current_density_mats, problem.part_faces, problem.part_face_basis_currents)
    if not counts or any(len(record) != len(counts) for record in records) or sum(counts) != len(full_stream_function):
        return float("nan"), 0, "Current-density operators do not cover every stream-function vertex."
    peak = 0.0
    face_count = 0
    offset = 0
    for count, dense, faces, local in zip(counts, *records):
        values = full_stream_function[offset : offset + count]
        offset += count
        faces = np.asarray(faces, dtype=np.int64)
        if dense is not None:
            dense = np.asarray(dense, dtype=float)
            if dense.shape != (count, len(faces), 3) or not np.all(np.isfinite(dense)):
                return float("nan"), face_count, "A dense current-density operator is incomplete or nonfinite."
            components = np.einsum("vfa,v->fa", dense, values, optimize=True)
        else:
            local = np.asarray(local, dtype=float)
            if faces.shape != (len(faces), 3) or local.shape != (len(faces), 3, 3):
                return float("nan"), face_count, "A local current-density operator has inconsistent dimensions."
            if np.any(faces < 0) or np.any(faces >= count) or not np.all(np.isfinite(local)):
                return float("nan"), face_count, "A local current-density operator is invalid or nonfinite."
            components = np.einsum("fva,fv->fa", local, values[faces], optimize=True)
        if not len(faces) or not np.all(np.isfinite(components)):
            return float("nan"), face_count, "A coil part has no finite face-current-density values."
        magnitudes = np.linalg.norm(components, axis=1)
        if not np.all(np.isfinite(magnitudes)):
            return float("nan"), face_count, "Surface-current-density evaluation overflowed."
        peak = max(peak, float(np.max(magnitudes)))
        face_count += len(faces)
    return peak, face_count, "Every coil face was checked."


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


def _solution_max_surface_current_density(
    problem: StreamFunctionOptimizationProblem,
    full_stream_function: np.ndarray,
) -> float:
    """Evaluate the peak face-current density for a full vertex stream function."""

    stream_function = np.asarray(full_stream_function, dtype=float).reshape(-1)
    max_density = 0.0
    offset = 0
    for vertex_count, current_density_mat, faces, face_basis_currents in zip(
        problem.part_vertex_counts,
        problem.part_current_density_mats,
        problem.part_faces,
        problem.part_face_basis_currents,
    ):
        part_stream_function = stream_function[offset : offset + vertex_count]
        offset += vertex_count
        if current_density_mat is not None:
            operator = np.asarray(current_density_mat, dtype=float)
            components = np.column_stack([operator[:, :, component].T @ part_stream_function for component in range(3)])
        else:
            part_faces = np.asarray(faces, dtype=np.int64)
            local_currents = np.asarray(face_basis_currents, dtype=float)
            components = np.einsum(
                "fva,fv->fa",
                local_currents,
                part_stream_function[part_faces],
                optimize=True,
            )
        if components.size:
            max_density = max(max_density, float(np.max(np.linalg.norm(components, axis=1))))
    return max_density


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

    for _ in range(max_attempts):
        fixed_quadratic = smoothness["quadratic"]
        if spacing["enabled"]:
            spacing_quadratic = float(penalty_weight) * spacing["scaled_metric"]
            fixed_quadratic = spacing_quadratic if fixed_quadratic is None else fixed_quadratic + spacing_quadratic
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

    for _ in range(max_attempts):
        fixed_quadratic = smoothness["quadratic"]
        if spacing["enabled"]:
            spacing_quadratic = float(penalty_weight) * spacing["scaled_metric"]
            fixed_quadratic = spacing_quadratic if fixed_quadratic is None else fixed_quadratic + spacing_quadratic

        solution = _target_error_constrained_efficiency_choice_with_retries(
            problem,
            ata,
            rtr,
            rhs,
            scale,
            config,
            fixed_quadratic=fixed_quadratic,
        )

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

    for _ in range(max_attempts):
        fixed_quadratic = smoothness["quadratic"]
        if spacing["enabled"]:
            spacing_quadratic = float(penalty_weight) * spacing["scaled_metric"]
            fixed_quadratic = spacing_quadratic if fixed_quadratic is None else fixed_quadratic + spacing_quadratic

        solution = _constrained_quadratic_choice(problem, rtr, config, fixed_quadratic=fixed_quadratic)

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
    retry_enabled = _target_error_auto_retry_enabled(config)
    max_reruns = _target_error_auto_retry_max_reruns(config)

    if retry_enabled and max_reruns > 0 and not _target_error_solution_meets_constraint(initial_solution):
        for rerun in range(1, max_reruns + 1):
            retry_config = _target_error_retry_config(config, rerun)
            try:
                retry_solution = _target_error_constrained_efficiency_choice(
                    problem,
                    ata,
                    rtr,
                    rhs,
                    scale,
                    retry_config,
                    fixed_quadratic=fixed_quadratic,
                )
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


def _diagnostic_float(diagnostics: dict[str, Any], field_name: str, default: float = float("nan")) -> float:
    try:
        value = float(diagnostics.get(field_name, default))
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


def _target_error_residual_budget(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> tuple[float, float, float, float]:
    target_norm = float(np.linalg.norm(problem.target_z))
    relative_tolerance_percent = float(getattr(config, "target_error_tolerance_percent", 5.0))
    absolute_tolerance = max(float(getattr(config, "target_error_tolerance_absolute_T", 0.0)), 0.0)
    allowed_residual = target_norm * relative_tolerance_percent / 100.0 + absolute_tolerance
    if not np.isfinite(allowed_residual) or allowed_residual <= 0.0:
        raise ValueError(
            "target_error_tolerance_percent and target_error_tolerance_absolute_T define a zero error budget. "
            "Use a positive tolerance for fixed-error stream-function optimization."
        )
    return target_norm, relative_tolerance_percent, absolute_tolerance, float(allowed_residual)


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


def _active_weights_for_points(problem: StreamFunctionOptimizationProblem, point_mask: np.ndarray) -> np.ndarray:
    point_count = len(point_mask)
    weights = problem.target_weights
    if weights is None:
        return np.ones(int(np.count_nonzero(point_mask)), dtype=float)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) == point_count:
        return values[point_mask]
    if len(values) == 3 * point_count:
        return np.mean(values.reshape(point_count, 3)[point_mask], axis=1)
    return np.ones(int(np.count_nonzero(point_mask)), dtype=float)


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


def _normalise_target_error_constraint_mode(config: CoilGenConfig) -> str:
    raw = str(getattr(config, "target_error_constraint_mode", "separate")).strip().lower().replace("-", "_")
    aliases = {
        "combined": "combined",
        "weighted": "combined",
        "weighted_combined": "combined",
        "legacy": "combined",
        "separate": "separate",
        "split": "separate",
        "dsv_and_suppression": "separate",
        "separate_dsv_suppression": "separate",
    }
    if raw not in aliases:
        raise ValueError(
            "target_error_constraint_mode must be 'separate' or 'combined'. "
            "'separate' constrains DSV accuracy and suppression-field residual independently."
        )
    return aliases[raw]


def _normalise_target_error_constraint_metric(config: CoilGenConfig) -> str:
    raw = str(getattr(config, "target_error_constraint_metric", "norm")).strip().lower().replace("-", "_")
    aliases = {
        "norm": "norm",
        "l2": "norm",
        "l2_norm": "norm",
        "residual_norm": "norm",
        "rms": "rms",
        "root_mean_square": "rms",
        "std": "std",
        "standard_deviation": "std",
        "homogeneity": "std",
    }
    if raw not in aliases:
        raise ValueError("target_error_constraint_metric must be 'norm', 'rms', or 'std'.")
    return aliases[raw]


def _unweighted_constraint_system(problem: StreamFunctionOptimizationProblem) -> tuple[np.ndarray, np.ndarray]:
    a_matrix = problem.unweighted_reduced_sensitivity
    target = problem.unweighted_target_z
    if a_matrix is None or target is None:
        return np.asarray(problem.reduced_sensitivity, dtype=float), np.asarray(problem.target_z, dtype=float)
    return np.asarray(a_matrix, dtype=float), np.asarray(target, dtype=float)


def _centered_constraint_system(a_matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(a_matrix, dtype=float) - np.mean(np.asarray(a_matrix, dtype=float), axis=0, keepdims=True),
        np.asarray(target, dtype=float) - float(np.mean(np.asarray(target, dtype=float))),
    )


def _weighted_constraint_system(
    a_matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    center: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(a_matrix, dtype=float)
    values = np.asarray(target, dtype=float)
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    if len(weight_values) != len(values):
        raise ValueError("DSV constraint weights must match the target rows.")
    valid = np.isfinite(weight_values) & (weight_values > 0.0)
    if not np.all(valid):
        raise ValueError("Active DSV constraint weights must be finite and positive.")
    weight_sum = float(np.sum(weight_values))
    if center:
        matrix = matrix - (weight_values @ matrix / weight_sum)[None, :]
        values = values - float(np.dot(weight_values, values) / weight_sum)
    sqrt_weights = np.sqrt(weight_values)
    return matrix * sqrt_weights[:, None], values * sqrt_weights, weight_sum


def _dsv_constraint_block(
    a_matrix: np.ndarray,
    target: np.ndarray,
    config: CoilGenConfig,
    *,
    weights: np.ndarray | None = None,
) -> ResidualConstraintBlock:
    metric = _normalise_target_error_constraint_metric(config)
    row_count = int(target.shape[0])
    weight_values = np.ones(row_count, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    relative_tolerance_percent = float(getattr(config, "target_error_tolerance_percent", 5.0))
    absolute_tolerance = max(float(getattr(config, "target_error_tolerance_absolute_T", 0.0)), 0.0)

    if metric == "std":
        constraint_matrix, constraint_target, weight_sum = _weighted_constraint_system(
            a_matrix,
            target,
            weight_values,
            center=True,
        )
        root_count = np.sqrt(max(weight_sum, 1.0e-30))
        reference_norm = float(np.linalg.norm(constraint_target))
        explicit_limit = max(float(getattr(config, "target_error_std_limit_T", 0.0)), 0.0)
        if explicit_limit > 0.0:
            allowed = root_count * explicit_limit
            field_limit = explicit_limit
        else:
            allowed = reference_norm * relative_tolerance_percent / 100.0 + root_count * absolute_tolerance
            field_limit = allowed / root_count
        zero_budget_message = (
            "target_error_constraint_metric='std' requires a positive target_error_std_limit_T, "
            "target_error_tolerance_percent, or target_error_tolerance_absolute_T."
        )
    elif metric == "rms":
        constraint_matrix, constraint_target, weight_sum = _weighted_constraint_system(
            a_matrix,
            target,
            weight_values,
            center=False,
        )
        root_count = np.sqrt(max(weight_sum, 1.0e-30))
        reference_norm = float(np.linalg.norm(constraint_target))
        explicit_limit = max(float(getattr(config, "target_error_rms_limit_T", 0.0)), 0.0)
        if explicit_limit > 0.0:
            allowed = root_count * explicit_limit
            field_limit = explicit_limit
        else:
            allowed = reference_norm * relative_tolerance_percent / 100.0 + root_count * absolute_tolerance
            field_limit = allowed / root_count
        zero_budget_message = (
            "target_error_constraint_metric='rms' requires a positive target_error_rms_limit_T, "
            "target_error_tolerance_percent, or target_error_tolerance_absolute_T."
        )
    else:
        constraint_matrix, constraint_target, _weight_sum = _weighted_constraint_system(
            a_matrix,
            target,
            weight_values,
            center=False,
        )
        reference_norm = float(np.linalg.norm(constraint_target))
        allowed = reference_norm * relative_tolerance_percent / 100.0 + absolute_tolerance
        field_limit = None
        zero_budget_message = (
            "target_error_tolerance_percent and target_error_tolerance_absolute_T define a zero DSV error budget. "
            "Use a positive fixed-error tolerance."
        )

    if not np.isfinite(allowed) or allowed <= 0.0:
        raise ValueError(zero_budget_message)

    return ResidualConstraintBlock(
        name="dsv",
        matrix=constraint_matrix,
        target=constraint_target,
        allowed_residual=float(allowed),
        reference_norm=reference_norm,
        relative_tolerance_percent=relative_tolerance_percent,
        absolute_tolerance=absolute_tolerance,
        metric=metric,
        field_limit_T=float(field_limit) if field_limit is not None else None,
        field_metric_denominator=float(root_count) if metric in {"std", "rms"} else 1.0,
    )


def _target_error_constraint_blocks(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> tuple[ResidualConstraintBlock, ...]:
    mode = _normalise_target_error_constraint_mode(config)
    metric = _normalise_target_error_constraint_metric(config)
    if mode == "combined":
        if metric != "norm":
            raise ValueError(
                "target_error_constraint_metric='std' or 'rms' requires "
                "target_error_constraint_mode='separate' so the DSV metric is not mixed with suppression rows."
            )
        target_norm, relative_tolerance_percent, absolute_tolerance, allowed_residual = _target_error_residual_budget(
            problem,
            config,
        )
        return (
            ResidualConstraintBlock(
                name="combined_weighted",
                matrix=np.asarray(problem.reduced_sensitivity, dtype=float),
                target=np.asarray(problem.target_z, dtype=float),
                allowed_residual=allowed_residual,
                reference_norm=target_norm,
                relative_tolerance_percent=relative_tolerance_percent,
                absolute_tolerance=absolute_tolerance,
                metric="norm",
            ),
        )

    a_matrix, target = _unweighted_constraint_system(problem)
    row_count = target.shape[0]
    weights = problem.target_weights
    if weights is None:
        active = np.ones(row_count, dtype=bool)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != row_count:
            raise ValueError("Target weights do not match the target vector length.")
        active = np.isfinite(weights) & (weights > 0.0)
    suppressed = problem.target_suppressed
    if suppressed is None:
        suppressed_mask = np.zeros(row_count, dtype=bool)
    else:
        suppressed_mask = np.asarray(suppressed, dtype=bool).reshape(-1)
        if len(suppressed_mask) != row_count:
            raise ValueError("Suppression mask does not match the target vector length.")

    dsv_mask = active & ~suppressed_mask
    if not np.any(dsv_mask):
        dsv_mask = active
    if not np.any(dsv_mask):
        raise ValueError("No active DSV target rows are available for fixed-error optimization.")

    dsv_target = target[dsv_mask]
    dsv_reference_norm = float(np.linalg.norm(dsv_target))
    dsv_weights = np.ones(len(dsv_target), dtype=float) if weights is None else weights[dsv_mask]
    blocks: list[ResidualConstraintBlock] = [
        _dsv_constraint_block(a_matrix[dsv_mask], dsv_target, config, weights=dsv_weights)
    ]
    blocks.extend(_gradient_linearity_constraint_blocks(problem, a_matrix, target, dsv_mask, config))

    suppression_point_mask = _suppression_point_mask(problem)
    if suppression_point_mask is not None:
        active_point_mask = _active_point_mask(problem)
        suppression_point_mask = suppression_point_mask & active_point_mask
        if np.any(suppression_point_mask):
            blocks.extend(
                _suppression_constraint_blocks(
                    problem,
                    suppression_point_mask,
                    config,
                    dsv_reference_norm=dsv_reference_norm,
                    default_relative_tolerance_percent=blocks[0].relative_tolerance_percent,
                )
            )
    return tuple(blocks)


def _suppression_constraint_blocks(
    problem: StreamFunctionOptimizationProblem,
    point_mask: np.ndarray,
    config: CoilGenConfig,
    *,
    dsv_reference_norm: float,
    default_relative_tolerance_percent: float,
) -> list[ResidualConstraintBlock]:
    matrix, target, component, base_weighting, denominator = _suppression_constraint_system(problem, point_mask, config)
    row_count = int(matrix.shape[0])
    if row_count == 0:
        return []
    field_rms_limit = max(float(getattr(config, "suppression_field_rms_limit_T", 0.0)), 0.0)
    peak_limit = max(float(getattr(config, "suppression_field_peak_limit_T", 0.0)), 0.0)
    relative_tolerance_percent = float(
        getattr(config, "suppression_error_tolerance_percent", default_relative_tolerance_percent)
    )
    absolute_tolerance = max(float(getattr(config, "suppression_error_tolerance_absolute_T", 0.0)), 0.0)
    blocks: list[ResidualConstraintBlock] = []
    if field_rms_limit > 0.0:
        allowed = float(denominator) * field_rms_limit
        blocks.append(
            ResidualConstraintBlock(
                name="suppression",
                matrix=matrix,
                target=target,
                allowed_residual=allowed,
                reference_norm=dsv_reference_norm,
                relative_tolerance_percent=100.0 * allowed / max(dsv_reference_norm, 1.0e-30),
                absolute_tolerance=allowed,
                rms_field_limit_T=field_rms_limit,
                metric="rms",
                field_limit_T=field_rms_limit,
                field_metric_denominator=float(denominator),
                component=component,
                weighting=base_weighting,
            )
        )
    elif peak_limit <= 0.0:
        allowed = dsv_reference_norm * relative_tolerance_percent / 100.0 + absolute_tolerance
        if not np.isfinite(allowed) or allowed <= 0.0:
            raise ValueError(
                "suppression_error_tolerance_percent and suppression_error_tolerance_absolute_T define a zero "
                "suppression error budget. Set a positive suppression_field_rms_limit_T for a physical field limit "
                "or use a positive active-shield suppression tolerance."
            )
        unweighted_matrix, unweighted_target, _, _, unweighted_denominator = _suppression_constraint_system(
            problem,
            point_mask,
            config,
            force_uniform=True,
        )
        blocks.append(
            ResidualConstraintBlock(
                name="suppression",
                matrix=unweighted_matrix,
                target=unweighted_target,
                allowed_residual=float(allowed),
                reference_norm=dsv_reference_norm,
                relative_tolerance_percent=relative_tolerance_percent,
                absolute_tolerance=absolute_tolerance,
                metric="norm",
                field_metric_denominator=float(unweighted_denominator),
                component=component,
                weighting="uniform",
            )
        )
    if peak_limit > 0.0:
        peak_matrix, peak_target, _, _, _ = _suppression_constraint_system(
            problem, point_mask, config, force_uniform=True
        )
        peak_group_size = 3 if component in {"vector", "Bmag"} else 1
        blocks.append(
            ResidualConstraintBlock(
                name="suppression_peak",
                matrix=peak_matrix,
                target=peak_target,
                allowed_residual=peak_limit,
                reference_norm=dsv_reference_norm,
                relative_tolerance_percent=100.0 * peak_limit / max(dsv_reference_norm, 1.0e-30),
                absolute_tolerance=peak_limit,
                metric="linf",
                field_limit_T=peak_limit,
                component=component,
                weighting="uniform",
                group_size=peak_group_size,
            )
        )
    return blocks


def _suppression_constraint_system(
    problem: StreamFunctionOptimizationProblem,
    point_mask: np.ndarray,
    config: CoilGenConfig,
    *,
    force_uniform: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, str, float]:
    if problem.target_coords is None:
        raise ValueError("Suppression constraints require target coordinates.")
    point_mask = np.asarray(point_mask, dtype=bool).reshape(-1)
    if len(point_mask) != len(problem.target_coords):
        raise ValueError("Suppression point mask does not match target coordinates.")
    point_indices = np.flatnonzero(point_mask)
    component = _suppression_component(config)
    full_matrix, full_target = _suppression_component_system(problem, point_indices, component, config)
    if full_matrix.shape[0] == 0:
        return full_matrix, full_target, component, "uniform", 1.0
    point_weights = _suppression_point_weights(problem, point_indices, config, force_uniform=force_uniform)
    if component in {"vector", "Bmag", "components"}:
        row_weights = np.repeat(point_weights, 3)
    else:
        row_weights = point_weights
    if len(row_weights) != full_matrix.shape[0]:
        raise ValueError("Suppression row weights do not match suppression constraint rows.")
    sqrt_weights = np.sqrt(row_weights)
    matrix = full_matrix * sqrt_weights[:, None]
    target = full_target * sqrt_weights
    denominator = float(np.sqrt(np.sum(point_weights)))
    weighting = "uniform" if force_uniform else _normalise_suppression_surface_weighting(config)
    return matrix, target, component, weighting, max(denominator, 1.0e-300)


def _suppression_component_system(
    problem: StreamFunctionOptimizationProblem,
    point_indices: np.ndarray,
    component: str,
    config: CoilGenConfig,
) -> tuple[np.ndarray, np.ndarray]:
    sensitivity = np.asarray(problem.sensitivity_matrix, dtype=float)[:, point_indices, :]
    target_b = _suppression_target_b(problem, point_indices)
    if component == "Bnormal":
        normals = _suppression_normals(problem, point_indices, config)
        full_matrix = (
            sensitivity[0] * normals[:, 0, None]
            + sensitivity[1] * normals[:, 1, None]
            + sensitivity[2] * normals[:, 2, None]
        )
        full_target = np.einsum("ij,ij->i", target_b, normals)
    elif component in {"vector", "Bmag", "components"}:
        full_matrix = np.moveaxis(sensitivity, 0, 1).reshape(3 * len(point_indices), sensitivity.shape[2])
        full_target = target_b.reshape(-1)
    else:
        b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
        full_matrix = sensitivity_component_matrix(sensitivity, component, b0_direction=b0_direction)
        full_target = field_component_values(target_b, component, b0_direction=b0_direction)
    return problem.reducer.reduce_columns(full_matrix), np.asarray(full_target, dtype=float).reshape(-1)


def _suppression_target_b(problem: StreamFunctionOptimizationProblem, point_indices: np.ndarray) -> np.ndarray:
    if problem.target_b is None:
        return np.zeros((len(point_indices), 3), dtype=float)
    target_b = np.asarray(problem.target_b, dtype=float)
    if target_b.ndim != 2 or target_b.shape[1] != 3:
        return np.zeros((len(point_indices), 3), dtype=float)
    return target_b[point_indices]


def _suppression_component(config: CoilGenConfig) -> str:
    raw = (
        str(getattr(config, "suppression_field_component", "target"))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if raw in {"", "target", "same", "useful", "target_component"}:
        raw = str(getattr(config, "target_field_component", "Bz")).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "x": "Bx",
        "bx": "Bx",
        "b_x": "Bx",
        "y": "By",
        "by": "By",
        "b_y": "By",
        "z": "Bz",
        "bz": "Bz",
        "b_z": "Bz",
        "b0": "BdotB0",
        "bdotb0": "BdotB0",
        "b_dot_b0": "BdotB0",
        "b_parallel": "BdotB0",
        "parallel_b0": "BdotB0",
        "normal": "Bnormal",
        "bnormal": "Bnormal",
        "b_normal": "Bnormal",
        "normal_field": "Bnormal",
        "vector": "vector",
        "full_vector": "vector",
        "bvector": "vector",
        "components": "components",
        "component": "components",
        "componentwise": "components",
        "component_wise": "components",
        "cartesian_components": "components",
        "xyz": "components",
        "bx_by_bz": "components",
        "bmag": "Bmag",
        "b_mag": "Bmag",
        "magnitude": "Bmag",
    }
    if raw not in aliases:
        raise ValueError(
            "suppression_field_component must be target, Bx, By, Bz, BdotB0, Bnormal, vector, components, or Bmag."
        )
    return aliases[raw]


def _suppression_point_mask(problem: StreamFunctionOptimizationProblem) -> np.ndarray | None:
    if problem.target_coords is None or problem.target_suppressed is None:
        return None
    point_count = len(problem.target_coords)
    suppressed = np.asarray(problem.target_suppressed, dtype=bool).reshape(-1)
    if len(suppressed) == point_count:
        return suppressed
    if len(suppressed) == 3 * point_count:
        return np.any(suppressed.reshape(point_count, 3), axis=1)
    return None


def _active_point_mask(problem: StreamFunctionOptimizationProblem) -> np.ndarray:
    if problem.target_coords is None:
        return np.zeros(0, dtype=bool)
    point_count = len(problem.target_coords)
    weights = problem.target_weights
    if weights is None:
        return np.ones(point_count, dtype=bool)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) == point_count:
        return np.isfinite(values) & (values > 0.0)
    if len(values) == 3 * point_count:
        reshaped = values.reshape(point_count, 3)
        return np.any(np.isfinite(reshaped) & (reshaped > 0.0), axis=1)
    return np.ones(point_count, dtype=bool)


def _suppression_point_weights(
    problem: StreamFunctionOptimizationProblem,
    point_indices: np.ndarray,
    config: CoilGenConfig,
    *,
    force_uniform: bool,
) -> np.ndarray:
    if force_uniform or _normalise_suppression_surface_weighting(config) == "uniform":
        return np.ones(len(point_indices), dtype=float)
    area_weights = problem.target_suppression_area_weights
    if area_weights is None:
        return np.ones(len(point_indices), dtype=float)
    values = np.asarray(area_weights, dtype=float).reshape(-1)
    if problem.target_coords is None or len(values) != len(problem.target_coords):
        return np.ones(len(point_indices), dtype=float)
    selected = values[point_indices]
    finite_positive = selected[np.isfinite(selected) & (selected > 0.0)]
    if finite_positive.size == 0:
        return np.ones(len(point_indices), dtype=float)
    fallback = float(np.median(finite_positive))
    selected = np.where(np.isfinite(selected) & (selected > 0.0), selected, fallback)
    return selected


def _normalise_suppression_surface_weighting(config: CoilGenConfig) -> str:
    raw = (
        str(getattr(config, "suppression_surface_weighting", "uniform"))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if raw in {"", "none", "uniform", "point", "points"}:
        return "uniform"
    if raw in {"area", "mesh_area", "surface_area", "vertex_area"}:
        return "area"
    raise ValueError("suppression_surface_weighting must be 'uniform' or 'area'.")


def _suppression_normals(
    problem: StreamFunctionOptimizationProblem,
    point_indices: np.ndarray,
    config: CoilGenConfig,
) -> np.ndarray:
    source = (
        str(getattr(config, "suppression_normal_source", "mesh")).strip().lower().replace("-", "_").replace(" ", "_")
    )
    coords = np.asarray(problem.target_coords, dtype=float)[point_indices]
    normals = None
    if source in {"mesh", "target", "metadata"} and problem.target_suppression_normals is not None:
        candidate = np.asarray(problem.target_suppression_normals, dtype=float)
        if candidate.shape == np.asarray(problem.target_coords).shape:
            normals = candidate[point_indices]
    if normals is None:
        model = "radial" if source in {"radial", "cylindrical"} else "box"
        normals = _estimate_suppression_normals(coords, model=model)
    return _normalise_normal_rows(normals, coords)


def _estimate_suppression_normals(coords: np.ndarray, *, model: str) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=float)
    center = np.nanmean(coords, axis=0)
    centered = coords - center[None, :]
    if model in {"radial", "cylindrical"}:
        normals = centered
    else:
        half_span = np.nanmax(np.abs(centered), axis=0)
        half_span[~np.isfinite(half_span) | (half_span <= 0.0)] = 1.0
        axis = np.nanargmax(np.abs(centered) / half_span[None, :], axis=1)
        normals = np.zeros_like(coords)
        signs = np.sign(centered[np.arange(coords.shape[0]), axis])
        signs[signs == 0.0] = 1.0
        normals[np.arange(coords.shape[0]), axis] = signs
    return normals


def _normalise_normal_rows(normals: np.ndarray, coords: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=float)
    lengths = np.linalg.norm(values, axis=1)
    safe = np.isfinite(lengths) & (lengths > 1.0e-15)
    out = np.zeros_like(values)
    out[safe] = values[safe] / lengths[safe, None]
    if np.any(~safe):
        fallback = _estimate_suppression_normals(coords[~safe], model="box")
        fallback_lengths = np.linalg.norm(fallback, axis=1)
        fallback_safe = np.isfinite(fallback_lengths) & (fallback_lengths > 1.0e-15)
        out[np.flatnonzero(~safe)[fallback_safe]] = fallback[fallback_safe] / fallback_lengths[fallback_safe, None]
    return out


def _shield_gradient_opposition_constraint_data(
    problem: StreamFunctionOptimizationProblem,
    config: CoilGenConfig,
) -> dict[str, Any]:
    requested = bool(getattr(config, "shield_gradient_opposition_enabled", False))
    disabled = {
        "requested": requested,
        "enabled": False,
        "reason": "disabled" if not requested else "unavailable",
        "matrix": None,
        "limit_T_per_m": float("nan"),
        "scale_T_per_m": float("nan"),
        "shield_part_indices": (),
        "desired_gradient_T_per_m": float("nan"),
        "max_fraction_of_target": float(getattr(config, "shield_gradient_max_fraction_of_target", 0.0)),
    }
    if not requested:
        return disabled
    if problem.target_coords is None or problem.unweighted_target_z is None:
        return {**disabled, "reason": "target data unavailable"}
    target = np.asarray(problem.unweighted_target_z, dtype=float).reshape(-1)
    coords = np.asarray(problem.target_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] != target.shape[0]:
        return {**disabled, "reason": "target coordinate rows do not match scalar target rows"}

    dsv_mask = _active_dsv_point_mask(problem, target.shape[0])
    sample_mask = _linearity_sample_mask(problem, target, dsv_mask, config)
    if np.count_nonzero(sample_mask) < 4:
        return {**disabled, "reason": "not enough DSV points for shield-gradient fit"}

    shield_part_indices = _outer_half_part_indices(problem)
    if not shield_part_indices:
        return {**disabled, "reason": "no distinct outer shield parts identified"}
    shield_columns = _part_column_indices(problem.part_vertex_counts, shield_part_indices)
    if shield_columns.size == 0:
        return {**disabled, "reason": "shield parts have no stream-function vertices"}

    fit_offset = bool(getattr(config, "linearity_fit_offset", True))
    basis = _linearity_affine_basis(coords[sample_mask], fit_offset=fit_offset)
    if basis.shape[0] <= basis.shape[1]:
        return {**disabled, "reason": "not enough DSV points for affine gradient fit"}
    pseudo_inverse = np.linalg.pinv(basis)
    target_coeff = _linearity_coefficients_from_pinv(pseudo_inverse, target[sample_mask], fit_offset=fit_offset)
    desired_gradient = np.asarray(target_coeff[1:4], dtype=float)
    desired_norm = float(np.linalg.norm(desired_gradient))
    requested_gradient = float(getattr(config, "target_gradient_strength", desired_norm))
    reference_gradient = (
        abs(requested_gradient) if np.isfinite(requested_gradient) and abs(requested_gradient) > 0.0 else desired_norm
    )
    if not np.isfinite(desired_norm) or desired_norm <= 1.0e-30 or reference_gradient <= 0.0:
        return {**disabled, "reason": "desired target gradient is zero or unavailable"}

    b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
    component = getattr(config, "target_field_component", "Bz")
    full_component = sensitivity_component_matrix(
        np.asarray(problem.sensitivity_matrix, dtype=float),
        component,
        b0_direction=b0_direction,
    )
    expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
    shield_operator = full_component[:, shield_columns] @ expansion[shield_columns, :]
    shield_selected = shield_operator[sample_mask]
    shield_coeff = _linearity_coefficients_from_pinv(pseudo_inverse, shield_selected, fit_offset=fit_offset)
    shield_gradient_matrix = np.asarray(shield_coeff[1:4], dtype=float)
    desired_unit = desired_gradient / desired_norm
    row = np.asarray(desired_unit.reshape(1, 3) @ shield_gradient_matrix, dtype=float).reshape(-1)
    if row.shape[0] != expansion.shape[1] or not np.all(np.isfinite(row)):
        return {**disabled, "reason": "shield-gradient constraint row is invalid"}

    max_fraction = float(getattr(config, "shield_gradient_max_fraction_of_target", 0.0))
    if not np.isfinite(max_fraction):
        return {**disabled, "reason": "shield-gradient max fraction is not finite"}
    return {
        "requested": requested,
        "enabled": True,
        "reason": "outer shield contribution constrained along desired gradient axis",
        "matrix": row,
        "limit_T_per_m": float(max_fraction * reference_gradient),
        "scale_T_per_m": float(reference_gradient),
        "shield_part_indices": tuple(int(index) for index in shield_part_indices),
        "desired_gradient_T_per_m": float(reference_gradient),
        "max_fraction_of_target": float(max_fraction),
    }


def _active_dsv_point_mask(problem: StreamFunctionOptimizationProblem, row_count: int) -> np.ndarray:
    if problem.target_weights is None:
        active = np.ones(row_count, dtype=bool)
    else:
        weights = np.asarray(problem.target_weights, dtype=float).reshape(-1)
        active = np.isfinite(weights) & (weights > 0.0) if len(weights) == row_count else np.ones(row_count, dtype=bool)
    if problem.target_suppressed is None:
        suppressed = np.zeros(row_count, dtype=bool)
    else:
        values = np.asarray(problem.target_suppressed, dtype=bool).reshape(-1)
        suppressed = values if len(values) == row_count else np.zeros(row_count, dtype=bool)
    dsv_mask = active & ~suppressed
    return dsv_mask if np.any(dsv_mask) else active


def _linearity_sample_mask(
    problem: StreamFunctionOptimizationProblem,
    target: np.ndarray,
    dsv_mask: np.ndarray,
    config: CoilGenConfig,
) -> np.ndarray:
    target_field = _linearity_target_field_view(problem, target)
    try:
        sample_mask = np.asarray(gradient_linearity_region_mask(target_field, config), dtype=bool).reshape(-1)
    except ValueError:
        sample_mask = np.asarray(dsv_mask, dtype=bool).reshape(-1)
    region_key = str(getattr(config, "linearity_region", "dsv")).strip().lower().replace("-", "_")
    if region_key in {"", "dsv", "fov", "roi", "target", "primary", "internal"}:
        sample_mask &= np.asarray(dsv_mask, dtype=bool).reshape(-1)
    sample_mask &= np.all(np.isfinite(np.asarray(problem.target_coords, dtype=float)), axis=1)
    return sample_mask


def _outer_half_part_indices(problem: StreamFunctionOptimizationProblem) -> tuple[int, ...]:
    counts = tuple(int(count) for count in problem.part_vertex_counts)
    if len(counts) < 4 or len(counts) % 2:
        return ()
    scores = tuple(float(value) for value in getattr(problem, "part_outward_scores", ()) or ())
    if len(scores) != len(counts) or not all(np.isfinite(score) for score in scores):
        return ()
    ranked = sorted(range(len(scores)), key=lambda index: scores[index])
    midpoint = len(ranked) // 2
    inner_max = max(scores[index] for index in ranked[:midpoint])
    outer_min = min(scores[index] for index in ranked[midpoint:])
    scale = max(max(abs(score) for score in scores), 1.0)
    if outer_min <= inner_max + 1.0e-6 * scale:
        return ()
    return tuple(sorted(ranked[midpoint:]))


def _part_column_indices(part_vertex_counts: tuple[int, ...], part_indices: tuple[int, ...]) -> np.ndarray:
    offsets = np.concatenate(([0], np.cumsum(np.asarray(part_vertex_counts, dtype=np.int64))))
    columns: list[np.ndarray] = []
    for index in part_indices:
        if 0 <= int(index) < len(part_vertex_counts):
            columns.append(np.arange(offsets[index], offsets[index + 1], dtype=np.int64))
    return np.concatenate(columns) if columns else np.empty(0, dtype=np.int64)


def _shield_gradient_opposition_constraint_diagnostics(
    constraint: dict[str, Any],
    reduced_stream_function: np.ndarray,
) -> dict[str, Any]:
    enabled = bool(constraint.get("enabled", False))
    diagnostics: dict[str, Any] = {
        "shield_gradient_opposition_requested": bool(constraint.get("requested", False)),
        "shield_gradient_opposition_enabled": enabled,
        "shield_gradient_opposition_reason": str(constraint.get("reason", "")),
        "shield_gradient_opposition_limit_T_per_m": float(constraint.get("limit_T_per_m", np.nan)),
        "shield_gradient_opposition_target_T_per_m": float(constraint.get("desired_gradient_T_per_m", np.nan)),
        "shield_gradient_opposition_max_fraction_of_target": float(constraint.get("max_fraction_of_target", np.nan)),
        "shield_gradient_opposition_part_indices": ",".join(
            str(int(index) + 1) for index in constraint.get("shield_part_indices", ())
        ),
        "shield_gradient_opposition_value_T_per_m": float("nan"),
        "shield_gradient_opposition_margin_T_per_m": float("nan"),
        "shield_gradient_opposition_constraint_met": not bool(constraint.get("requested", False)),
    }
    if not enabled:
        return diagnostics
    row = np.asarray(constraint.get("matrix"), dtype=float).reshape(-1)
    value = float(row @ np.asarray(reduced_stream_function, dtype=float).reshape(-1))
    limit = float(constraint.get("limit_T_per_m", np.nan))
    scale = max(abs(float(constraint.get("scale_T_per_m", 0.0))), 1.0e-30)
    tolerance = max(1.0e-12, 1.0e-9 * scale)
    diagnostics.update(
        {
            "shield_gradient_opposition_value_T_per_m": value,
            "shield_gradient_opposition_margin_T_per_m": float(limit - value),
            "shield_gradient_opposition_constraint_met": bool(value <= limit + tolerance),
        }
    )
    return diagnostics


def _gradient_linearity_constraint_blocks(
    problem: StreamFunctionOptimizationProblem,
    a_matrix: np.ndarray,
    target: np.ndarray,
    dsv_mask: np.ndarray,
    config: CoilGenConfig,
) -> list[ResidualConstraintBlock]:
    if not gradient_linearity_enabled(config):
        return []
    if problem.target_coords is None:
        return []
    target_field = _linearity_target_field_view(problem, target)
    try:
        region_mask = gradient_linearity_region_mask(target_field, config)
        sample_mask = np.asarray(region_mask, dtype=bool).reshape(-1)
    except ValueError:
        sample_mask = np.asarray(dsv_mask, dtype=bool).reshape(-1)
    region_key = str(getattr(config, "linearity_region", "dsv")).strip().lower().replace("-", "_")
    if region_key in {"", "dsv", "fov", "roi", "target", "primary", "internal"}:
        sample_mask &= np.asarray(dsv_mask, dtype=bool).reshape(-1)
    sample_mask &= np.all(np.isfinite(np.asarray(problem.target_coords, dtype=float)), axis=1)
    if np.count_nonzero(sample_mask) < 4:
        return []

    coords = np.asarray(problem.target_coords, dtype=float)[sample_mask]
    a_selected = np.asarray(a_matrix, dtype=float)[sample_mask]
    target_selected = np.asarray(target, dtype=float)[sample_mask]
    fit_offset = bool(getattr(config, "linearity_fit_offset", True))
    basis = _linearity_affine_basis(coords, fit_offset=fit_offset)
    if basis.shape[0] <= basis.shape[1]:
        return []
    pseudo_inverse = np.linalg.pinv(basis)
    residual_projection = np.eye(basis.shape[0]) - basis @ pseudo_inverse
    reference_scale = gradient_linearity_reference_scale_T(target_field, config, sample_mask)
    if not np.isfinite(reference_scale) or reference_scale <= 0.0:
        return []

    target_coeff = _linearity_coefficients_from_pinv(pseudo_inverse, target_selected, fit_offset=fit_offset)
    desired_gradient = np.asarray(target_coeff[1:4], dtype=float)
    desired_norm = float(np.linalg.norm(desired_gradient))
    requested_gradient = float(getattr(config, "target_gradient_strength", desired_norm))
    reference_gradient = (
        abs(requested_gradient) if np.isfinite(requested_gradient) and abs(requested_gradient) > 0.0 else desired_norm
    )
    blocks: list[ResidualConstraintBlock] = []

    rms_tolerance = _positive_config_float(config, "linearity_rms_tolerance_percent")
    peak_tolerance = _positive_config_float(config, "linearity_peak_tolerance_percent")
    cross_tolerance = _positive_config_float(config, "linearity_cross_gradient_tolerance_percent")
    gradient_tolerance = _positive_config_float(config, "gradient_strength_tolerance_percent")
    row_count = int(np.count_nonzero(sample_mask))
    if rms_tolerance > 0.0:
        field_limit = reference_scale * rms_tolerance / 100.0
        blocks.append(
            ResidualConstraintBlock(
                name="linearity_rms",
                matrix=residual_projection @ a_selected,
                target=np.zeros(row_count, dtype=float),
                allowed_residual=float(np.sqrt(float(row_count)) * field_limit),
                reference_norm=float(np.sqrt(float(row_count)) * reference_scale),
                relative_tolerance_percent=float(rms_tolerance),
                absolute_tolerance=0.0,
                metric="rms",
                field_limit_T=float(field_limit),
            )
        )
    if peak_tolerance > 0.0:
        field_limit = reference_scale * peak_tolerance / 100.0
        blocks.append(
            ResidualConstraintBlock(
                name="linearity_peak",
                matrix=residual_projection @ a_selected,
                target=np.zeros(row_count, dtype=float),
                allowed_residual=float(field_limit),
                reference_norm=float(reference_scale),
                relative_tolerance_percent=float(peak_tolerance),
                absolute_tolerance=0.0,
                metric="linf",
                field_limit_T=float(field_limit),
            )
        )
    if desired_norm > 0.0 and reference_gradient > 0.0:
        coeff_matrix = _linearity_coefficients_from_pinv(pseudo_inverse, a_selected, fit_offset=fit_offset)
        gradient_matrix = coeff_matrix[1:4]
        desired_unit = desired_gradient / desired_norm
        if gradient_tolerance > 0.0:
            field_limit = reference_gradient * gradient_tolerance / 100.0
            blocks.append(
                ResidualConstraintBlock(
                    name="gradient_strength",
                    matrix=(desired_unit.reshape(1, 3) @ gradient_matrix),
                    target=np.asarray([reference_gradient], dtype=float),
                    allowed_residual=float(field_limit),
                    reference_norm=float(reference_gradient),
                    relative_tolerance_percent=float(gradient_tolerance),
                    absolute_tolerance=0.0,
                    metric="linf",
                    field_limit_T=float(field_limit),
                )
            )
        if cross_tolerance > 0.0:
            field_limit = reference_gradient * cross_tolerance / 100.0
            orthogonal = np.eye(3) - np.outer(desired_unit, desired_unit)
            blocks.append(
                ResidualConstraintBlock(
                    name="cross_gradient",
                    matrix=orthogonal @ gradient_matrix,
                    target=np.zeros(3, dtype=float),
                    allowed_residual=float(field_limit),
                    reference_norm=float(reference_gradient),
                    relative_tolerance_percent=float(cross_tolerance),
                    absolute_tolerance=0.0,
                    metric="norm",
                    field_limit_T=float(field_limit),
                )
            )
    return blocks


def _linearity_target_field_view(problem: StreamFunctionOptimizationProblem, target: np.ndarray) -> Any:
    class _TargetView:
        pass

    view = _TargetView()
    view.coords = np.asarray(problem.target_coords, dtype=float)
    values = np.asarray(target, dtype=float).reshape(-1)
    b = np.zeros((len(values), 3), dtype=float)
    b[:, 2] = values
    view.b = b
    if problem.target_suppressed is None:
        view.suppressed = np.zeros(len(values), dtype=bool)
    else:
        view.suppressed = np.asarray(problem.target_suppressed, dtype=bool).reshape(-1)
    return view


def _linearity_affine_basis(coords: np.ndarray, *, fit_offset: bool) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if fit_offset:
        return np.column_stack([np.ones(coords.shape[0]), coords])
    return coords


def _linearity_coefficients_from_pinv(
    pseudo_inverse: np.ndarray, values: np.ndarray, *, fit_offset: bool
) -> np.ndarray:
    coeff = np.asarray(pseudo_inverse @ np.asarray(values, dtype=float), dtype=float)
    if fit_offset:
        return coeff.reshape(4, -1) if coeff.ndim == 2 else coeff.reshape(4)
    if coeff.ndim == 2:
        return np.vstack([np.zeros((1, coeff.shape[1]), dtype=float), coeff])
    return np.concatenate(([0.0], coeff.reshape(3)))


def _residual_constraint_diagnostics(
    problem: StreamFunctionOptimizationProblem,
    reduced_stream_function: np.ndarray,
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    config: CoilGenConfig,
) -> dict[str, Any]:
    tolerance = _constraint_acceptance_relative_tolerance(config)
    absolute_acceptance = _constraint_acceptance_absolute_tolerance(config)
    diagnostics: dict[str, Any] = {
        "target_error_constraint_mode": _normalise_target_error_constraint_mode(config),
        "target_error_constraint_metric": _normalise_target_error_constraint_metric(config),
        "target_error_constraint_block_count": int(len(constraint_blocks)),
        "target_error_constraint_acceptance_relative_tolerance": float(tolerance),
        "target_error_constraint_acceptance_absolute_T": float(absolute_acceptance),
        "suppression_constraint_enabled": any(str(block.name).startswith("suppression") for block in constraint_blocks),
    }
    all_met = True
    primary_block = constraint_blocks[0]
    primary_residual = float("nan")
    primary_relative_percent = float("nan")
    primary_metric_value = float("nan")
    primary_allowed_metric_value = float("nan")
    for block in constraint_blocks:
        residual_values = np.asarray(block.matrix @ reduced_stream_function - block.target, dtype=float).reshape(-1)
        norm_residual = float(np.linalg.norm(residual_values))
        residual = _constraint_residual_measure(block, reduced_stream_function)
        field_denominator = _constraint_field_metric_denominator(block)
        rms_residual = norm_residual / field_denominator
        linf_residual = _constraint_linf_residual(block, residual_values)
        std_residual = float(np.std(residual_values)) if len(residual_values) else float("nan")
        mean_residual = float(np.mean(residual_values)) if len(residual_values) else float("nan")
        reference = max(float(block.reference_norm), 1.0e-30)
        relative_percent = 100.0 * residual / reference
        # Gradient residuals are T/m; a field-valued T allowance cannot be added
        # to them. Keep relative numerical acceptance for these fitted gradients.
        residual_unit = "T/m" if block.name in {"gradient_strength", "cross_gradient"} else "T"
        block_absolute_acceptance = absolute_acceptance if residual_unit == "T" else 0.0
        allowed_with_acceptance = float(block.allowed_residual) * (1.0 + tolerance) + block_absolute_acceptance
        met = residual <= allowed_with_acceptance
        all_met = all_met and bool(met)
        prefix = block.name
        metric = str(block.metric)
        field_metric = (
            std_residual
            if metric == "std"
            else rms_residual
            if metric == "rms"
            else linf_residual
            if metric == "linf"
            else residual
        )
        allowed_field_metric = (
            float(block.field_limit_T)
            if block.field_limit_T is not None
            else float(block.allowed_residual) / field_denominator
            if metric in {"std", "rms"}
            else float(block.allowed_residual)
        )
        diagnostics.update(
            {
                f"{prefix}_constraint_met": bool(met),
                f"{prefix}_constraint_residual_unit": residual_unit,
                f"{prefix}_constraint_metric": metric,
                f"{prefix}_constraint_component": str(block.component),
                f"{prefix}_constraint_weighting": str(block.weighting),
                f"{prefix}_constraint_residual_norm_T": residual,
                f"{prefix}_constraint_l2_residual_norm_T": norm_residual,
                f"{prefix}_constraint_allowed_residual_norm_T": float(block.allowed_residual),
                f"{prefix}_constraint_allowed_with_acceptance_residual_norm_T": float(allowed_with_acceptance),
                f"{prefix}_constraint_reference_norm_T": float(block.reference_norm),
                f"{prefix}_constraint_relative_residual_percent": float(relative_percent),
                f"{prefix}_constraint_rms_residual_T": float(rms_residual),
                f"{prefix}_constraint_linf_residual_T": float(linf_residual),
                f"{prefix}_constraint_std_T": float(std_residual),
                f"{prefix}_constraint_mean_T": float(mean_residual),
                f"{prefix}_constraint_field_metric_T": float(field_metric),
                f"{prefix}_constraint_allowed_field_metric_T": float(allowed_field_metric),
                f"{prefix}_constraint_tolerance_percent": float(block.relative_tolerance_percent),
                f"{prefix}_constraint_absolute_tolerance_T": float(block.absolute_tolerance),
            }
        )
        if metric == "std":
            diagnostics[f"{prefix}_constraint_allowed_std_T"] = float(allowed_field_metric)
        if metric == "rms":
            diagnostics[f"{prefix}_constraint_allowed_rms_T"] = float(allowed_field_metric)
        if metric == "linf":
            diagnostics[f"{prefix}_constraint_allowed_linf_T"] = float(allowed_field_metric)
        if block.rms_field_limit_T is not None:
            diagnostics[f"{prefix}_constraint_rms_field_limit_T"] = float(block.rms_field_limit_T)
        if block.name == "suppression":
            diagnostics["suppression_field_rms_T"] = float(rms_residual)
            diagnostics["suppression_field_rms_limit_T"] = (
                float(block.rms_field_limit_T) if block.rms_field_limit_T is not None else float("nan")
            )
            diagnostics["suppression_field_component"] = str(block.component)
            diagnostics["suppression_surface_weighting"] = str(block.weighting)
        if block.name == "suppression_peak":
            diagnostics["suppression_field_peak_T"] = float(field_metric)
            diagnostics["suppression_field_peak_limit_T"] = float(allowed_field_metric)
            diagnostics.setdefault("suppression_field_component", str(block.component))
            diagnostics.setdefault("suppression_surface_weighting", str(block.weighting))
        if block.name == "dsv":
            primary_block = block
            primary_residual = residual
            primary_relative_percent = relative_percent
            primary_metric_value = float(field_metric)
            primary_allowed_metric_value = float(allowed_field_metric)
    if not np.isfinite(primary_residual):
        primary_values = np.asarray(primary_block.matrix @ reduced_stream_function - primary_block.target, dtype=float)
        primary_residual = _constraint_residual_measure(primary_block, reduced_stream_function)
        primary_relative_percent = 100.0 * primary_residual / max(float(primary_block.reference_norm), 1.0e-30)
        primary_metric = str(primary_block.metric)
        if primary_metric == "std":
            primary_metric_value = float(np.std(primary_values))
        elif primary_metric == "rms":
            primary_metric_value = float(np.linalg.norm(primary_values)) / _constraint_field_metric_denominator(
                primary_block
            )
        elif primary_metric == "linf":
            primary_metric_value = _constraint_linf_residual(primary_block, primary_values)
        else:
            primary_metric_value = primary_residual
        primary_allowed_metric_value = (
            float(primary_block.field_limit_T)
            if primary_block.field_limit_T is not None
            else float(primary_block.allowed_residual) / _constraint_field_metric_denominator(primary_block)
            if primary_metric in {"std", "rms"}
            else float(primary_block.allowed_residual)
        )

    diagnostics.update(
        {
            "target_error_constraint_met": bool(all_met),
            "target_error_constraint_metric": str(primary_block.metric),
            "target_error_tolerance_percent": float(primary_block.relative_tolerance_percent),
            "target_error_tolerance_absolute_T": float(primary_block.absolute_tolerance),
            "target_error_allowed_residual_norm_T": float(primary_block.allowed_residual),
            "target_error_constraint_field_metric_T": float(primary_metric_value),
            "target_error_allowed_field_metric_T": float(primary_allowed_metric_value),
            "target_error_target_norm_T": float(primary_block.reference_norm),
            "target_error_constraint_relative_residual_percent": float(primary_relative_percent),
            "target_error_constraint_residual_norm_T": float(primary_residual),
            "combined_weighted_residual_norm_T": problem.residual_norm(reduced_stream_function),
        }
    )
    block_diagnostics = _constraint_block_diagnostic_rows(constraint_blocks, diagnostics)
    suppression_rows = [row for row in block_diagnostics if str(row["name"]).startswith("suppression")]
    failed_blocks = [row["name"] for row in block_diagnostics if not bool(row["met"])]
    finite_ratios = [
        float(row["violation_ratio"]) for row in block_diagnostics if np.isfinite(float(row["violation_ratio"]))
    ]
    diagnostics.update(
        {
            "target_error_constraint_block_diagnostics": block_diagnostics,
            "target_error_failed_constraint_blocks": failed_blocks,
            "suppression_constraint_enabled": bool(suppression_rows),
            "suppression_constraint_met": bool(suppression_rows) and all(bool(row["met"]) for row in suppression_rows),
            "target_error_constraint_max_violation_ratio": (
                float(max(finite_ratios)) if finite_ratios else float("nan")
            ),
        }
    )
    return diagnostics


def _constraint_acceptance_relative_tolerance(config: CoilGenConfig) -> float:
    explicit = getattr(config, "target_error_constraint_acceptance_tolerance", 0.0)
    try:
        explicit_value = float(explicit)
    except (TypeError, ValueError):
        explicit_value = 0.0
    if np.isfinite(explicit_value) and explicit_value > 0.0:
        return float(explicit_value)

    try:
        solver_tolerance = float(getattr(config, "constrained_optimization_tolerance", 1.0e-7))
    except (TypeError, ValueError):
        solver_tolerance = 1.0e-7
    if not np.isfinite(solver_tolerance) or solver_tolerance <= 0.0:
        solver_tolerance = 1.0e-7
    return float(max(1.0e-8, 10.0 * solver_tolerance))


def _constraint_acceptance_absolute_tolerance(config: CoilGenConfig) -> float:
    explicit = getattr(config, "target_error_constraint_acceptance_absolute_T", 1.0e-7)
    try:
        explicit_value = float(explicit)
    except (TypeError, ValueError):
        explicit_value = 1.0e-7
    if np.isfinite(explicit_value) and explicit_value >= 0.0:
        return float(explicit_value)
    return 1.0e-7


def _constraint_block_diagnostic_rows(
    constraint_blocks: tuple[ResidualConstraintBlock, ...],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in constraint_blocks:
        prefix = block.name
        residual = _diagnostic_float(diagnostics, f"{prefix}_constraint_residual_norm_T")
        allowed = _diagnostic_float(diagnostics, f"{prefix}_constraint_allowed_residual_norm_T")
        field_metric = _diagnostic_float(diagnostics, f"{prefix}_constraint_field_metric_T")
        allowed_field_metric = _diagnostic_float(diagnostics, f"{prefix}_constraint_allowed_field_metric_T")
        ratio = residual / allowed if np.isfinite(residual) and np.isfinite(allowed) and allowed > 0.0 else float("nan")
        rows.append(
            {
                "name": prefix,
                "metric": str(diagnostics.get(f"{prefix}_constraint_metric", block.metric)),
                "component": str(diagnostics.get(f"{prefix}_constraint_component", block.component)),
                "weighting": str(diagnostics.get(f"{prefix}_constraint_weighting", block.weighting)),
                "met": bool(diagnostics.get(f"{prefix}_constraint_met", False)),
                "residual_norm_T": float(residual),
                "allowed_residual_norm_T": float(allowed),
                "field_metric_T": float(field_metric),
                "allowed_field_metric_T": float(allowed_field_metric),
                "violation_ratio": float(ratio),
            }
        )
    return rows


def _constraint_field_metric_denominator(block: ResidualConstraintBlock) -> float:
    if block.field_metric_denominator is not None:
        try:
            value = float(block.field_metric_denominator)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0.0:
            return value
    return float(np.sqrt(max(int(block.matrix.shape[0]), 1)))


def _constraint_failure_summary(diagnostics: dict[str, Any]) -> str:
    rows = diagnostics.get("target_error_constraint_block_diagnostics", ())
    failed = [row for row in rows if not bool(row.get("met", False))] if isinstance(rows, list) else []
    if not failed:
        return "No block-level residual diagnostics were available."
    pieces = []
    for row in failed:
        pieces.append(
            (
                f"{row.get('name', 'unknown')}: "
                f"value={_format_diagnostic_number(row.get('field_metric_T'))} T, "
                f"allowed={_format_diagnostic_number(row.get('allowed_field_metric_T'))} T, "
                f"ratio={_format_diagnostic_number(row.get('violation_ratio'))}"
            )
        )
    return "Failed residual block(s): " + "; ".join(pieces) + "."


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


def _constraint_residual_measure(block: ResidualConstraintBlock, reduced_stream_function: np.ndarray) -> float:
    residual_values = np.asarray(block.matrix @ reduced_stream_function - block.target, dtype=float).reshape(-1)
    if str(block.metric).lower() == "linf":
        value = _constraint_linf_residual(block, residual_values)
        return float(value) if np.isfinite(value) else float("inf")
    return float(np.linalg.norm(residual_values))


def _constraint_linf_residual(block: ResidualConstraintBlock, residual_values: np.ndarray) -> float:
    values = np.asarray(residual_values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    group_size = max(int(getattr(block, "group_size", 1) or 1), 1)
    if group_size <= 1:
        return float(np.max(np.abs(values)))
    if values.size % group_size != 0:
        raise ValueError(
            f"Constraint block {block.name!r} has {values.size} residual rows, "
            f"which is not divisible by group_size={group_size}."
        )
    grouped = values.reshape((-1, group_size))
    return float(np.max(np.linalg.norm(grouped, axis=1)))


def _positive_config_float(config: CoilGenConfig, name: str) -> float:
    try:
        value = float(getattr(config, name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return float(value) if np.isfinite(value) and value > 0.0 else 0.0


def _normalise_constrained_quadratic_objective_mode(config: CoilGenConfig) -> str:
    raw = str(getattr(config, "constrained_quadratic_objective_mode", "balanced"))
    key = raw.strip().lower().replace("-", "_")
    aliases = {
        "balanced": "balanced",
        "resistance": "balanced",
        "resistance_plus_penalties": "balanced",
        "smoothness_priority": "smoothness_priority",
        "smoothest_feasible": "smoothness_priority",
        "contourability_priority": "smoothness_priority",
        "minimax_current_density": "minimax_current_density",
    }
    if key not in aliases:
        raise ValueError(
            "constrained_quadratic_objective_mode must be 'balanced', 'smoothness_priority', "
            "or 'minimax_current_density'."
        )
    return aliases[key]


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


def _symmetric_quadratic_matrix(matrix: np.ndarray) -> np.ndarray:
    quadratic = np.asarray(matrix, dtype=float)
    if quadratic.ndim != 2 or quadratic.shape[0] != quadratic.shape[1]:
        raise ValueError("Regularization matrix must be square.")
    quadratic = 0.5 * (quadratic + quadratic.T)
    if not np.all(np.isfinite(quadratic)):
        raise ValueError("Regularization matrix contains non-finite values.")
    return quadratic


def _quadratic_norm(matrix: np.ndarray, vector: np.ndarray) -> float:
    quadratic = _symmetric_quadratic_matrix(matrix)
    x = np.asarray(vector, dtype=float).reshape(-1)
    if quadratic.shape[0] != len(x):
        raise ValueError("Regularization matrix and vector dimensions do not match.")
    if len(x) == 0:
        return 0.0
    value = float(x @ quadratic @ x)
    scale = max(float(np.linalg.norm(quadratic, ord=np.inf)) * float(np.dot(x, x)), 1.0)
    if value < 0.0 and abs(value) <= 1.0e-10 * scale:
        value = 0.0
    return float(np.sqrt(max(value, 0.0)))


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
    task = progress.task("CVXPY solver attempts", len(solver_order))
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
        task.done(message="done")
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
        return candidate, diagnostics

    task.done(message="failed")
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


def _constrained_contour_step_limit(config: CoilGenConfig) -> dict[str, float | int | bool | str]:
    configured_step = _nonnegative_config_value(config, "regularization_auto_range_max_contour_step_A", 0.0)
    engineering_step = engineering_max_contour_step_A(config)
    step_candidates: list[tuple[str, float]] = []
    if configured_step > 0.0 and np.isfinite(configured_step):
        step_candidates.append(("regularization_auto_range", float(configured_step)))
    if engineering_step > 0.0 and np.isfinite(engineering_step):
        step_candidates.append(("max_wire_current", float(engineering_step)))

    if step_candidates:
        min_step = min(step for _source, step in step_candidates)
        tied_sources = [
            source for source, step in step_candidates if abs(step - min_step) <= 1.0e-12 * max(abs(min_step), 1.0)
        ]
        source = "_and_".join(tied_sources)
        max_step = float(min_step)
    else:
        source = "disabled"
        max_step = 0.0

    level_count = _contour_step_constraint_level_count(
        config,
        prefer_engineering_max_levels="max_wire_current" in source,
    )
    denominator = _contour_step_denominator(level_count, config)
    enabled = bool(max_step > 0.0 and np.isfinite(max_step) and denominator > 0.0)
    max_range = float(max_step * denominator) if enabled else float("nan")
    return {
        "contour_step_constraint_enabled": enabled,
        "contour_step_constraint_max_step_A": float(max_step),
        "contour_step_constraint_regularization_max_step_A": float(configured_step),
        "contour_step_constraint_engineering_max_step_A": float(engineering_step),
        "contour_step_constraint_source": source,
        "contour_step_constraint_level_count": int(level_count),
        "contour_step_constraint_denominator": float(denominator),
        "contour_step_constraint_max_range_A": max_range,
    }


def _contour_step_constraint_level_count(
    config: CoilGenConfig,
    *,
    prefer_engineering_max_levels: bool = False,
) -> int:
    if prefer_engineering_max_levels and engineering_contouring_enabled(config):
        max_levels = engineering_contour_max_levels(config)
        if max_levels > 0:
            return int(max_levels)
    try:
        level_count = max(int(getattr(config, "levels", 1)), 1)
    except (TypeError, ValueError):
        level_count = 1
    return int(level_count)


def _contour_step_denominator(level_count: int, config: CoilGenConfig) -> float:
    grid_mode = str(getattr(config, "contour_level_grid_mode", "centered_margin")).strip().lower().replace("-", "_")
    if grid_mode in {"fixed_step_phase", "fixed_phase", "quantizer_phase"}:
        return max(float(max(int(level_count), 1)), np.finfo(float).eps)
    try:
        level_offset = float(getattr(config, "pot_offset_factor", 0.5))
    except (TypeError, ValueError):
        level_offset = 0.5
    return max(float(max(int(level_count), 1) - 1) + 2.0 * level_offset, np.finfo(float).eps)


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
    if fixed_quadratic is not None and scan_method in {"auto", "transformed_svd"}:
        candidate_notes.append(
            "transformed_svd scan unavailable with fixed turn-spacing penalty; used normal_equations scan"
        )
    elif scan_method in {"auto", "transformed_svd"}:
        scan = _scan_l_curve_by_transformed_svd(problem, lambdas)
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
    whitened = _regularization_whitened_sensitivity(problem)
    if whitened is None:
        return None
    try:
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


def _nonnegative_config_value(config: CoilGenConfig, name: str, default: float) -> float:
    try:
        value = float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value) or value < 0.0:
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
) -> tuple[np.ndarray, np.ndarray, str] | None:
    regularization_operator = _regularization_operator_from_quadratic(problem.reduced_resistance)
    if regularization_operator is None:
        return None
    try:
        transformed = np.linalg.solve(regularization_operator.T, problem.reduced_sensitivity.T).T
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(transformed)):
        return None
    try:
        u, singular_values, _vt = np.linalg.svd(transformed, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(singular_values)):
        return None

    projected_target = u.T @ problem.target_z
    residuals = np.full(len(lambdas), np.nan, dtype=float)
    regularization_norms = np.full(len(lambdas), np.nan, dtype=float)
    for ind, lambda_value in enumerate(lambdas):
        denom = singular_values * singular_values + lambda_value * lambda_value
        y_coefficients = (singular_values / np.maximum(denom, 1e-300)) * projected_target
        fitted_target = u @ (singular_values * y_coefficients)
        residuals[ind] = float(np.linalg.norm(fitted_target - problem.target_z))
        regularization_norms[ind] = float(np.linalg.norm(y_coefficients))
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


def _select_stream_function_reducer(
    parts: list[CoilPart],
    target_field: TargetField,
    boundary_reducer: BoundaryReducer,
    weighted_full_sensitivity_z: np.ndarray,
    weighted_target_z: np.ndarray,
    config: CoilGenConfig | None,
) -> tuple[BoundaryReducer | SymmetryReducer, dict[str, Any]]:
    mode = _normalise_symmetry_reduction_mode(config)
    base_unknown_count = int(boundary_reducer.expansion_matrix.shape[1])
    diagnostics: dict[str, Any] = {
        "symmetry_reduction_mode": mode,
        "symmetry_reduction_active": False,
        "symmetry_reduction_reason": "",
        "symmetry_reduction_unknown_count_before": base_unknown_count,
        "symmetry_reduction_unknown_count_after": base_unknown_count,
        "symmetry_reduction_pairs": [],
    }
    if mode == "off":
        diagnostics["symmetry_reduction_reason"] = "disabled"
        return boundary_reducer, diagnostics
    if config is None:
        diagnostics["symmetry_reduction_reason"] = "no config supplied"
        return boundary_reducer, diagnostics

    part_offsets = _part_vertex_offsets(parts)
    full_to_boundary = _full_to_boundary_reduced_column(boundary_reducer)
    constraints: list[tuple[int, int, float]] = []
    zero_columns: list[int] = []
    pair_reports: list[dict[str, Any]] = []
    failures: list[str] = []
    source = _normalise_symmetry_reduction_source(config)

    if source in {"generated_parts", "auto"}:
        candidate_pairs = _generated_candidate_symmetric_part_pairs(parts, config)
        for pair in candidate_pairs:
            pair_report: dict[str, Any] = {
                "pair": [int(part_ind + 1) for part_ind in pair],
                "symmetry_reduction_source": "generated_parts",
            }
            geometry = _mirror_vertex_mapping_for_pair(parts, pair, config)
            pair_report.update(geometry["diagnostics"])
            if not geometry["ok"]:
                failures.append(f"pair {pair_report['pair']}: {geometry['reason']}")
                pair_reports.append(pair_report)
                continue

            target = _target_symmetry_mapping(target_field, geometry["plane_point"], geometry["plane_normal"], config)
            pair_report.update(target["diagnostics"])
            if not target["ok"]:
                failures.append(f"pair {pair_report['pair']}: {target['reason']}")
                pair_reports.append(pair_report)
                continue

            sign = _select_symmetry_pair_stream_sign(
                weighted_full_sensitivity_z,
                weighted_target_z,
                target["match_indices"],
                float(target["target_parity"]),
                pair,
                geometry["vertex_match_indices"],
                part_offsets,
                config,
            )
            pair_report.update(sign["diagnostics"])
            if not sign["ok"]:
                failures.append(f"pair {pair_report['pair']}: {sign['reason']}")
                pair_reports.append(pair_report)
                continue

            stream_sign = float(sign["stream_function_sign"])
            part_a, part_b = pair
            offset_a = part_offsets[part_a]
            offset_b = part_offsets[part_b]
            vertex_match = np.asarray(geometry["vertex_match_indices"], dtype=np.int64)
            for local_a, local_b in enumerate(vertex_match):
                full_a = offset_a + int(local_a)
                full_b = offset_b + int(local_b)
                col_a = int(full_to_boundary[full_a])
                col_b = int(full_to_boundary[full_b])
                constraints.append((col_a, col_b, stream_sign))
            pair_report["symmetry_reduction_pair_accepted"] = True
            pair_reports.append(pair_report)

    if source in {"single_mesh", "auto"} and len(parts) == 1:
        single_report: dict[str, Any] = {"pair": [1], "symmetry_reduction_source": "single_mesh"}
        geometry = _mirror_vertex_mapping_for_single_mesh(parts[0], config)
        single_report.update(geometry["diagnostics"])
        if not geometry["ok"]:
            failures.append(f"single mesh: {geometry['reason']}")
            pair_reports.append(single_report)
        else:
            target = _target_symmetry_mapping(target_field, geometry["plane_point"], geometry["plane_normal"], config)
            single_report.update(target["diagnostics"])
            if not target["ok"]:
                failures.append(f"single mesh: {target['reason']}")
                pair_reports.append(single_report)
            else:
                sign = _select_single_mesh_stream_sign(
                    weighted_full_sensitivity_z,
                    target["match_indices"],
                    float(target["target_parity"]),
                    geometry["vertex_match_indices"],
                    config,
                )
                single_report.update(sign["diagnostics"])
                if not sign["ok"]:
                    failures.append(f"single mesh: {sign['reason']}")
                    pair_reports.append(single_report)
                else:
                    stream_sign = float(sign["stream_function_sign"])
                    vertex_match = np.asarray(geometry["vertex_match_indices"], dtype=np.int64)
                    seen: set[tuple[int, int]] = set()
                    for local_a, local_b in enumerate(vertex_match):
                        col_a = int(full_to_boundary[int(local_a)])
                        col_b = int(full_to_boundary[int(local_b)])
                        if col_a == col_b:
                            if stream_sign < 0.0:
                                zero_columns.append(col_a)
                            continue
                        key = tuple(sorted((col_a, col_b)))
                        if key in seen:
                            continue
                        seen.add(key)
                        constraints.append((col_a, col_b, stream_sign))
                    single_report["symmetry_reduction_pair_accepted"] = True
                    pair_reports.append(single_report)

    diagnostics["symmetry_reduction_pairs"] = pair_reports
    if failures:
        diagnostics["symmetry_reduction_reason"] = "; ".join(failures)
        if mode == "strict":
            raise ValueError("Symmetry reduction validation failed: " + diagnostics["symmetry_reduction_reason"])
        return boundary_reducer, diagnostics
    if not constraints and not zero_columns:
        diagnostics["symmetry_reduction_reason"] = "no valid symmetry constraints"
        if source == "generated_parts" and not pair_reports:
            diagnostics["symmetry_reduction_reason"] = "no generated symmetric part pairs"
        elif source == "single_mesh" and len(parts) != 1:
            diagnostics["symmetry_reduction_reason"] = "single-mesh symmetry reduction requires exactly one coil part"
        if mode == "strict":
            raise ValueError("Symmetry reduction validation failed: no valid symmetry constraints")
        return boundary_reducer, diagnostics

    expansion, reduction_report = _signed_constraint_expansion(
        base_unknown_count, constraints, zero_columns=zero_columns
    )
    diagnostics.update(reduction_report)
    if not reduction_report["symmetry_reduction_constraints_ok"]:
        diagnostics["symmetry_reduction_reason"] = str(reduction_report["symmetry_reduction_constraint_failure"])
        if mode == "strict":
            raise ValueError("Symmetry reduction validation failed: " + diagnostics["symmetry_reduction_reason"])
        return boundary_reducer, diagnostics
    reduced_unknown_count = int(expansion.shape[1])
    if reduced_unknown_count >= base_unknown_count:
        diagnostics["symmetry_reduction_reason"] = "constraints did not reduce the unknown count"
        return boundary_reducer, diagnostics

    diagnostics["symmetry_reduction_active"] = True
    diagnostics["symmetry_reduction_reason"] = f"validated {source} mirror symmetry"
    diagnostics["symmetry_reduction_unknown_count_after"] = reduced_unknown_count
    diagnostics["symmetry_reduction_factor"] = float(base_unknown_count / max(reduced_unknown_count, 1))
    reducer = SymmetryReducer.from_boundary_reducer(boundary_reducer, expansion, diagnostics)
    return reducer, diagnostics


def _normalise_symmetry_reduction_mode(config: CoilGenConfig | None) -> str:
    if config is None:
        return "off"
    return normalise_symmetry_mode(getattr(config, "symmetry_reduction", "auto"))


def _normalise_symmetry_reduction_source(config: CoilGenConfig | None) -> str:
    if config is None:
        return "generated_parts"
    source = str(getattr(config, "symmetry_reduction_source", "generated_parts")).strip().lower().replace("-", "_")
    if source in {"generated", "generated_part", "generated_parts"}:
        return "generated_parts"
    if source in {"single", "single_mesh", "mesh"}:
        return "single_mesh"
    if source in {"half", "half_mesh"}:
        return "single_mesh"
    if source == "auto":
        return "auto"
    return "generated_parts"


def _generated_candidate_symmetric_part_pairs(parts: list[CoilPart], config: CoilGenConfig) -> list[tuple[int, ...]]:
    return _symmetry_candidate_part_pairs(str(getattr(config, "coil_mesh_file", "")), len(parts))


def _part_vertex_offsets(parts: list[CoilPart]) -> np.ndarray:
    counts = [part.mesh.n_vertices for part in parts]
    return np.asarray([0, *np.cumsum(counts[:-1])], dtype=np.int64)


def _full_to_boundary_reduced_column(boundary_reducer: BoundaryReducer) -> np.ndarray:
    expansion = np.asarray(boundary_reducer.expansion_matrix, dtype=float)
    if expansion.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.argmax(np.abs(expansion), axis=1).astype(np.int64)


def _mirror_vertex_mapping_for_pair(
    parts: list[CoilPart],
    pair: tuple[int, ...],
    config: CoilGenConfig,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "symmetry_reduction_geometry_ok": False,
        "symmetry_reduction_geometry_reason": "",
    }
    if len(pair) != 2:
        return {"ok": False, "reason": "only two-part mirror pairs are supported", "diagnostics": diagnostics}
    part_a, part_b = parts[pair[0]], parts[pair[1]]
    if part_a.mesh.n_vertices != part_b.mesh.n_vertices or part_a.mesh.n_faces != part_b.mesh.n_faces:
        diagnostics.update(
            {
                "symmetry_reduction_geometry_vertex_count_a": int(part_a.mesh.n_vertices),
                "symmetry_reduction_geometry_vertex_count_b": int(part_b.mesh.n_vertices),
            }
        )
        return {"ok": False, "reason": "pair mesh sizes differ", "diagnostics": diagnostics}

    vertices_a = np.asarray(part_a.mesh.vertices, dtype=float)
    vertices_b = np.asarray(part_b.mesh.vertices, dtype=float)
    center_a = np.mean(vertices_a, axis=0)
    center_b = np.mean(vertices_b, axis=0)
    normal = center_b - center_a
    centroid_distance = float(np.linalg.norm(normal))
    scale = _geometry_scale(np.vstack((vertices_a, vertices_b)))
    if centroid_distance <= 1e-14 * max(scale, 1.0):
        return {"ok": False, "reason": "pair centroids are coincident", "diagnostics": diagnostics}
    normal = normal / centroid_distance
    plane_point = 0.5 * (center_a + center_b)
    reflected_a = _reflect_points(vertices_a, plane_point, normal)
    match_indices = _nearest_indices(reflected_a, vertices_b)
    if match_indices is None:
        return {
            "ok": False,
            "reason": "vertex matching would be too expensive without scipy",
            "diagnostics": diagnostics,
        }
    distances = np.linalg.norm(reflected_a - vertices_b[match_indices], axis=1)
    tolerance = _geometry_tolerance(config, scale)
    max_distance = float(np.max(distances)) if distances.size else float("inf")
    rms_distance = float(np.sqrt(np.mean(distances * distances))) if distances.size else float("inf")
    unique_match = len(np.unique(match_indices)) == len(match_indices)
    ok = bool(unique_match and np.isfinite(max_distance) and max_distance <= tolerance)
    diagnostics.update(
        {
            "symmetry_reduction_geometry_ok": ok,
            "symmetry_reduction_geometry_reason": "" if ok else "reflected vertex cloud mismatch",
            "symmetry_reduction_geometry_tolerance_m": float(tolerance),
            "symmetry_reduction_geometry_max_distance_m": max_distance,
            "symmetry_reduction_geometry_rms_distance_m": rms_distance,
            "symmetry_reduction_geometry_unique_mapping": bool(unique_match),
            "symmetry_reduction_mirror_plane_point_m": [float(value) for value in plane_point],
            "symmetry_reduction_mirror_plane_normal": [float(value) for value in normal],
        }
    )
    return {
        "ok": ok,
        "reason": "" if ok else str(diagnostics["symmetry_reduction_geometry_reason"]),
        "vertex_match_indices": match_indices,
        "plane_point": plane_point,
        "plane_normal": normal,
        "diagnostics": diagnostics,
    }


def _mirror_vertex_mapping_for_single_mesh(part: CoilPart, config: CoilGenConfig) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "symmetry_reduction_geometry_ok": False,
        "symmetry_reduction_geometry_reason": "",
    }
    vertices = np.asarray(part.mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return {"ok": False, "reason": "single mesh has no vertices", "diagnostics": diagnostics}
    plane_point, normal = _configured_mirror_plane(config)
    reflected = _reflect_points(vertices, plane_point, normal)
    match_indices = _nearest_indices(reflected, vertices)
    if match_indices is None:
        return {
            "ok": False,
            "reason": "vertex matching would be too expensive without scipy",
            "diagnostics": diagnostics,
        }
    match_indices = np.asarray(match_indices, dtype=np.int64)
    distances = np.linalg.norm(reflected - vertices[match_indices], axis=1)
    scale = _geometry_scale(vertices)
    tolerance = _geometry_tolerance(config, scale)
    max_distance = float(np.max(distances)) if distances.size else float("inf")
    rms_distance = float(np.sqrt(np.mean(distances * distances))) if distances.size else float("inf")
    unique_match = len(np.unique(match_indices)) == len(match_indices)
    involution_ok = bool(np.all(match_indices[match_indices] == np.arange(len(match_indices), dtype=np.int64)))
    ok = bool(unique_match and involution_ok and np.isfinite(max_distance) and max_distance <= tolerance)
    diagnostics.update(
        {
            "symmetry_reduction_geometry_ok": ok,
            "symmetry_reduction_geometry_reason": "" if ok else "reflected single-mesh vertex cloud mismatch",
            "symmetry_reduction_geometry_vertex_count": int(part.mesh.n_vertices),
            "symmetry_reduction_geometry_tolerance_m": float(tolerance),
            "symmetry_reduction_geometry_max_distance_m": max_distance,
            "symmetry_reduction_geometry_rms_distance_m": rms_distance,
            "symmetry_reduction_geometry_unique_mapping": bool(unique_match),
            "symmetry_reduction_geometry_involution_mapping": bool(involution_ok),
            "symmetry_reduction_mirror_plane_point_m": [float(value) for value in plane_point],
            "symmetry_reduction_mirror_plane_normal": [float(value) for value in normal],
        }
    )
    return {
        "ok": ok,
        "reason": "" if ok else str(diagnostics["symmetry_reduction_geometry_reason"]),
        "vertex_match_indices": match_indices,
        "plane_point": plane_point,
        "plane_normal": normal,
        "diagnostics": diagnostics,
    }


def _target_symmetry_mapping(
    target_field: TargetField,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "symmetry_reduction_target_ok": False,
        "symmetry_reduction_target_reason": "",
    }
    coords = np.asarray(target_field.coords, dtype=float)
    if len(coords) == 0:
        return {"ok": False, "reason": "target field has no samples", "diagnostics": diagnostics}
    reflected = _reflect_points(coords, plane_point, plane_normal)
    match_indices = _nearest_indices(reflected, coords)
    if match_indices is None:
        return {
            "ok": False,
            "reason": "target matching would be too expensive without scipy",
            "diagnostics": diagnostics,
        }
    distances = np.linalg.norm(reflected - coords[match_indices], axis=1)
    scale = _geometry_scale(coords)
    tolerance = _target_tolerance(config, scale)
    max_distance = float(np.max(distances)) if distances.size else float("inf")
    unique_match = len(np.unique(match_indices)) == len(match_indices)
    weights = np.asarray(target_field.weights, dtype=float)
    suppressed = np.asarray(target_field.suppressed, dtype=bool)
    groups = np.asarray(target_field.target_field_group_inds, dtype=np.int64)
    weight_tolerance = float(getattr(config, "symmetry_reduction_target_weight_tolerance", 1e-8))
    weights_ok = bool(np.allclose(weights[match_indices], weights, rtol=weight_tolerance, atol=weight_tolerance))
    suppressed_ok = bool(np.array_equal(suppressed[match_indices], suppressed))
    groups_ok = bool(np.array_equal(groups[match_indices], groups))

    component = getattr(config, "target_field_component", "Bz")
    target_z = field_component_values(
        target_field.b,
        component,
        b0_direction=getattr(config, "b_0_direction", (0.0, 0.0, 1.0)),
    )
    if len(target_z) != len(match_indices):
        component = "Bz"
        target_z = np.asarray(target_field.b[:, 2], dtype=float)
    plus = _target_parity_residual(target_z, match_indices, sign=1.0)
    minus = _target_parity_residual(target_z, match_indices, sign=-1.0)
    selected = _select_configured_target_parity(plus, minus, config)
    parity_tolerance = _target_parity_tolerance(config)
    parity_ok = bool(selected["normalized_rms"] <= parity_tolerance)
    parity_policy = str(getattr(config, "symmetry_reduction_target_parity_policy", "require")).strip().lower()
    parity_required = parity_policy != "diagnostic"
    ok = bool(
        unique_match
        and np.isfinite(max_distance)
        and max_distance <= tolerance
        and weights_ok
        and suppressed_ok
        and groups_ok
        and (parity_ok or not parity_required)
    )
    reasons = []
    if not unique_match or not np.isfinite(max_distance) or max_distance > tolerance:
        reasons.append("reflected target samples do not match")
    if not weights_ok:
        reasons.append("target weights are not symmetric")
    if not suppressed_ok:
        reasons.append("suppression flags are not symmetric")
    if not groups_ok:
        reasons.append("target groups are not symmetric")
    if not parity_ok and parity_required:
        reasons.append(f"target {component} is neither mirror-even nor mirror-odd")
    diagnostics.update(
        {
            "symmetry_reduction_target_ok": ok,
            "symmetry_reduction_target_reason": "; ".join(reasons),
            "symmetry_reduction_target_tolerance_m": float(tolerance),
            "symmetry_reduction_target_max_distance_m": max_distance,
            "symmetry_reduction_target_unique_mapping": bool(unique_match),
            "symmetry_reduction_target_weights_ok": weights_ok,
            "symmetry_reduction_target_suppressed_ok": suppressed_ok,
            "symmetry_reduction_target_groups_ok": groups_ok,
            "symmetry_reduction_target_parity": int(selected["sign"]),
            "symmetry_reduction_target_parity_mode": str(getattr(config, "symmetry_reduction_target_parity", "auto")),
            "symmetry_reduction_target_parity_normalized_rms": float(selected["normalized_rms"]),
            "symmetry_reduction_target_parity_normalized_max": float(selected["normalized_max"]),
            "symmetry_reduction_target_parity_tolerance": float(parity_tolerance),
            "symmetry_reduction_target_parity_policy": parity_policy,
            "symmetry_reduction_target_parity_required": parity_required,
            "symmetry_reduction_target_weight_tolerance": float(weight_tolerance),
        }
    )
    return {
        "ok": ok,
        "reason": str(diagnostics["symmetry_reduction_target_reason"]),
        "match_indices": match_indices,
        "target_parity": float(selected["sign"]),
        "diagnostics": diagnostics,
    }


def _target_parity_residual(target_z: np.ndarray, match_indices: np.ndarray, *, sign: float) -> dict[str, float]:
    return _symmetry_parity_residual(target_z, match_indices, sign=sign)


def _select_configured_target_parity(
    plus: dict[str, float], minus: dict[str, float], config: CoilGenConfig
) -> dict[str, float]:
    mode = str(getattr(config, "symmetry_reduction_target_parity", "auto")).strip().lower()
    if mode == "even":
        return plus
    if mode == "odd":
        return minus
    return plus if plus["normalized_rms"] <= minus["normalized_rms"] else minus


def _select_symmetry_pair_stream_sign(
    weighted_full_sensitivity_z: np.ndarray,
    weighted_target_z: np.ndarray,
    target_match_indices: np.ndarray,
    target_parity: float,
    pair: tuple[int, ...],
    vertex_match_indices: np.ndarray,
    part_offsets: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "symmetry_reduction_sign_ok": False,
        "symmetry_reduction_stream_function_sign": 0,
    }
    part_a, part_b = pair
    offset_a = int(part_offsets[part_a])
    offset_b = int(part_offsets[part_b])
    local_a = np.arange(len(vertex_match_indices), dtype=np.int64)
    full_a = offset_a + local_a
    full_b = offset_b + np.asarray(vertex_match_indices, dtype=np.int64)
    plus = _sensitivity_parity_residual(
        weighted_full_sensitivity_z,
        target_match_indices,
        target_parity,
        full_a,
        full_b,
        stream_sign=1.0,
    )
    minus = _sensitivity_parity_residual(
        weighted_full_sensitivity_z,
        target_match_indices,
        target_parity,
        full_a,
        full_b,
        stream_sign=-1.0,
    )
    selected = plus if plus["normalized_rms"] <= minus["normalized_rms"] else minus
    tolerance = max(
        _target_parity_tolerance(config),
        10.0 * float(getattr(config, "contour_symmetry_geometry_tolerance", 1e-6)),
    )
    target_norm = float(np.linalg.norm(weighted_target_z))
    # If the target is exactly zero on a suppression-only set, the sensitivity
    # parity still decides; the threshold stays relative to the column scale.
    ok = bool(
        selected["normalized_rms"] <= tolerance
        or (target_norm <= 1e-30 and selected["normalized_rms"] <= 10.0 * tolerance)
    )
    diagnostics.update(
        {
            "symmetry_reduction_sign_ok": ok,
            "symmetry_reduction_stream_function_sign": int(selected["stream_sign"]),
            "symmetry_reduction_sign_normalized_rms": float(selected["normalized_rms"]),
            "symmetry_reduction_sign_normalized_max": float(selected["normalized_max"]),
            "symmetry_reduction_sign_tolerance": float(tolerance),
            "symmetry_reduction_sign_plus_normalized_rms": float(plus["normalized_rms"]),
            "symmetry_reduction_sign_minus_normalized_rms": float(minus["normalized_rms"]),
        }
    )
    return {
        "ok": ok,
        "reason": "" if ok else "sensitivity columns do not support a consistent mirror sign",
        "stream_function_sign": float(selected["stream_sign"]),
        "diagnostics": diagnostics,
    }


def _select_single_mesh_stream_sign(
    weighted_full_sensitivity_z: np.ndarray,
    target_match_indices: np.ndarray,
    target_parity: float,
    vertex_match_indices: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "symmetry_reduction_sign_ok": False,
        "symmetry_reduction_stream_function_sign": 0,
    }
    full_a = np.arange(len(vertex_match_indices), dtype=np.int64)
    full_b = np.asarray(vertex_match_indices, dtype=np.int64)
    plus = _sensitivity_parity_residual(
        weighted_full_sensitivity_z,
        target_match_indices,
        target_parity,
        full_a,
        full_b,
        stream_sign=1.0,
    )
    minus = _sensitivity_parity_residual(
        weighted_full_sensitivity_z,
        target_match_indices,
        target_parity,
        full_a,
        full_b,
        stream_sign=-1.0,
    )
    selected = plus if plus["normalized_rms"] <= minus["normalized_rms"] else minus
    tolerance = max(
        _target_parity_tolerance(config),
        10.0 * float(getattr(config, "contour_symmetry_geometry_tolerance", 1e-6)),
    )
    ok = bool(selected["normalized_rms"] <= tolerance)
    diagnostics.update(
        {
            "symmetry_reduction_sign_ok": ok,
            "symmetry_reduction_stream_function_sign": int(selected["stream_sign"]),
            "symmetry_reduction_sign_normalized_rms": float(selected["normalized_rms"]),
            "symmetry_reduction_sign_normalized_max": float(selected["normalized_max"]),
            "symmetry_reduction_sign_tolerance": float(tolerance),
            "symmetry_reduction_sign_plus_normalized_rms": float(plus["normalized_rms"]),
            "symmetry_reduction_sign_minus_normalized_rms": float(minus["normalized_rms"]),
        }
    )
    return {
        "ok": ok,
        "reason": "" if ok else "single-mesh sensitivity columns do not support a consistent mirror sign",
        "stream_function_sign": float(selected["stream_sign"]),
        "diagnostics": diagnostics,
    }


def _sensitivity_parity_residual(
    sensitivity_z: np.ndarray,
    target_match_indices: np.ndarray,
    target_parity: float,
    full_a: np.ndarray,
    full_b: np.ndarray,
    *,
    stream_sign: float,
) -> dict[str, float]:
    chunk_size = 256
    sum_sq = 0.0
    sum_max = 0.0
    norm_sq = 0.0
    count = 0
    for start in range(0, len(full_a), chunk_size):
        stop = min(start + chunk_size, len(full_a))
        columns = sensitivity_z[:, full_a[start:stop]] + float(stream_sign) * sensitivity_z[:, full_b[start:stop]]
        residual = columns[target_match_indices, :] - float(target_parity) * columns
        sum_sq += float(np.sum(residual * residual))
        sum_max = max(sum_max, float(np.max(np.abs(residual))) if residual.size else 0.0)
        norm_sq += float(np.sum(columns * columns))
        count += int(residual.size)
    rms = np.sqrt(sum_sq / max(count, 1))
    scale = np.sqrt(norm_sq / max(count, 1))
    scale = max(float(scale), 1e-30)
    return {
        "stream_sign": float(stream_sign),
        "normalized_rms": float(rms / scale),
        "normalized_max": float(sum_max / scale),
    }


def _signed_constraint_expansion(
    unknown_count: int,
    constraints: list[tuple[int, int, float]],
    *,
    zero_columns: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    parent = np.arange(unknown_count, dtype=np.int64)
    sign_to_parent = np.ones(unknown_count, dtype=float)
    failures: list[str] = []
    zero_set = {int(column) for column in (zero_columns or []) if 0 <= int(column) < int(unknown_count)}

    def find(node: int) -> tuple[int, float]:
        root = int(node)
        sign = 1.0
        while parent[root] != root:
            sign *= sign_to_parent[root]
            root = int(parent[root])
        current = int(node)
        path_sign = 1.0
        while parent[current] != current:
            next_node = int(parent[current])
            next_sign = sign_to_parent[current]
            parent[current] = root
            sign_to_parent[current] = sign / path_sign
            path_sign *= next_sign
            current = next_node
        return root, sign

    def union(col_a: int, col_b: int, relation_sign: float) -> None:
        if not (0 <= col_a < unknown_count and 0 <= col_b < unknown_count):
            failures.append("constraint column index out of range")
            return
        sign = 1.0 if relation_sign >= 0.0 else -1.0
        root_a, sign_a = find(col_a)
        root_b, sign_b = find(col_b)
        if root_a == root_b:
            if not np.isclose(sign_b, sign * sign_a, rtol=1e-10, atol=1e-12):
                failures.append(f"inconsistent signed symmetry constraint for column {col_a} and {col_b}")
            return
        parent[root_b] = root_a
        sign_to_parent[root_b] = sign * sign_a / sign_b

    for col_a, col_b, sign in constraints:
        union(int(col_a), int(col_b), float(sign))

    roots = []
    signs = []
    for col in range(unknown_count):
        root, sign = find(col)
        roots.append(root)
        signs.append(sign)
    zero_roots = {find(column)[0] for column in zero_set}
    root_to_reduced: dict[int, int] = {}
    for root in roots:
        if int(root) in zero_roots:
            continue
        root_to_reduced.setdefault(int(root), len(root_to_reduced))
    expansion = np.zeros((unknown_count, len(root_to_reduced)), dtype=float)
    for col, (root, sign) in enumerate(zip(roots, signs)):
        if int(root) in zero_roots:
            continue
        expansion[col, root_to_reduced[int(root)]] = float(sign)
    report = {
        "symmetry_reduction_constraints_ok": len(failures) == 0,
        "symmetry_reduction_constraint_failure": "; ".join(failures),
        "symmetry_reduction_constraint_count": int(len(constraints)),
        "symmetry_reduction_zero_constraint_count": int(len(zero_set)),
    }
    return expansion, report


def _geometry_scale(points: np.ndarray) -> float:
    return _symmetry_geometry_scale(points)


def _geometry_tolerance(config: CoilGenConfig, scale: float) -> float:
    return _symmetry_absolute_tolerance(float(getattr(config, "contour_symmetry_geometry_tolerance", 1e-6)), scale)


def _target_tolerance(config: CoilGenConfig, scale: float) -> float:
    absolute = float(getattr(config, "symmetry_reduction_target_geometry_tolerance_m", 0.0))
    if np.isfinite(absolute) and absolute > 0.0:
        return absolute
    return _symmetry_absolute_tolerance(float(getattr(config, "symmetry_reduction_target_tolerance", 1e-6)), scale)


def _target_parity_tolerance(config: CoilGenConfig) -> float:
    tolerance = float(getattr(config, "symmetry_reduction_target_parity_tolerance", 0.0))
    if np.isfinite(tolerance) and tolerance > 0.0:
        return tolerance
    return float(getattr(config, "symmetry_reduction_target_tolerance", 1e-6))


def _configured_mirror_plane(config: CoilGenConfig) -> tuple[np.ndarray, np.ndarray]:
    plane_point = np.asarray(
        getattr(config, "symmetry_reduction_mirror_plane_point", (0.0, 0.0, 0.0)), dtype=float
    ).reshape(3)
    normal = np.asarray(
        getattr(config, "symmetry_reduction_mirror_plane_normal", (1.0, 0.0, 0.0)), dtype=float
    ).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("symmetry_reduction_mirror_plane_normal must be finite and non-zero.")
    return plane_point, normal / norm


def _reflect_points(points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return _symmetry_reflect_points(points, plane_point, normal)


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    return _symmetry_nearest_indices(source, target)


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


def _turn_spacing_metric_requested(config: CoilGenConfig) -> bool:
    return _turn_spacing_penalty_requested(config) or engineering_surface_current_density_limit_A_per_m(config) > 0.0


def _turn_spacing_penalty_requested(config: CoilGenConfig) -> bool:
    # Minimax directly spreads peak current; retain geometric spacing as a bound,
    # without silently changing the objective through the engineering-mode heuristic.
    if (
        _normalise_stream_function_method(getattr(config, "sf_opt_method", "tikhonov")) == "constrained_quadratic"
        and _normalise_constrained_quadratic_objective_mode(config) == "minimax_current_density"
    ):
        return False
    threshold = _turn_spacing_threshold(config)
    factor = _turn_spacing_regularization_factor(config)
    return threshold > 0.0 and factor > 0.0


def _turn_spacing_threshold(config: CoilGenConfig) -> float:
    if engineering_contouring_enabled(config):
        return min_wire_distance_m(config)
    return _safe_positive_float(getattr(config, "min_turn_distance", 0.0))


def _turn_spacing_regularization_factor(config: CoilGenConfig) -> float:
    factor = _safe_positive_float(getattr(config, "turn_spacing_regularization_factor", 0.0))
    if factor <= 0.0 and engineering_contouring_enabled(config) and _turn_spacing_threshold(config) > 0.0:
        return 1.0
    return factor


def _stream_function_smoothness_requested(config: CoilGenConfig) -> bool:
    return _safe_positive_float(getattr(config, "stream_function_smoothness_factor", 0.0)) > 0.0


def _spectral_stream_function_basis_requested(config: CoilGenConfig) -> bool:
    return (
        _normalise_stream_function_basis(getattr(config, "stream_function_basis", "vertex"))
        == "cotangent_spectral_lowpass"
    )


def _normalise_stream_function_basis(basis: str) -> str:
    normalised = str(basis).strip().lower().replace("-", "_")
    if normalised in {"vertex", "vertices", "full", ""}:
        return "vertex"
    if normalised in {"cotangent_spectral_lowpass", "spectral_lowpass", "laplace_beltrami_lowpass"}:
        return "cotangent_spectral_lowpass"
    raise ValueError(f"stream_function_basis must be 'vertex' or 'cotangent_spectral_lowpass' (got {basis!r}).")


def _stream_function_basis_diagnostics(config: CoilGenConfig | None) -> dict[str, Any]:
    basis = (
        "vertex"
        if config is None
        else _normalise_stream_function_basis(getattr(config, "stream_function_basis", "vertex"))
    )
    return {
        "stream_function_basis": basis,
        "stream_function_spectral_active": basis == "cotangent_spectral_lowpass",
    }


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


def _combined_surface_current_density_metric(parts: list[CoilPart]) -> np.ndarray:
    metrics: list[np.ndarray] = []
    for part in parts:
        if part.current_density_mat is None:
            raise ValueError("Every coil part must have current_density_mat before turn-spacing regularization.")
        metrics.append(_surface_current_density_metric(part))
    return _block_diag(metrics)


def _reduced_surface_current_density_metric(
    parts: list[CoilPart],
    reducer: BoundaryReducer | SymmetryReducer | GaugeReducer | SpectralReducer,
) -> np.ndarray:
    """Assemble integral |K|^2 dA directly in the active reduced basis.

    This is algebraically E.T Q E, where E expands reduced stream-function
    coefficients and Q is the full nodal current-density Gram matrix. Direct
    element assembly avoids the O(vertices * faces) dense current tensor.
    """

    expansion = np.asarray(reducer.expansion_matrix, dtype=float)
    reduced_count = int(expansion.shape[1])
    metric = np.zeros((reduced_count, reduced_count), dtype=float)
    vertex_offset = 0
    chunk_size = 256
    for part in parts:
        faces = np.asarray(part.mesh.faces, dtype=np.int64)
        face_currents = _face_local_basis_currents(part)
        face_areas = _mesh_face_areas(part.mesh)
        for start in range(0, len(faces), chunk_size):
            stop = min(start + chunk_size, len(faces))
            face_expansion = expansion[vertex_offset + faces[start:stop]]
            reduced_density = np.einsum(
                "fvr,fva->fra",
                face_expansion,
                face_currents[start:stop],
                optimize=True,
            )
            metric += np.einsum(
                "fra,fsa,f->rs",
                reduced_density,
                reduced_density,
                face_areas[start:stop],
                optimize=True,
            )
        vertex_offset += int(part.mesh.n_vertices)
    return 0.5 * (metric + metric.T)


def _face_local_basis_currents(part: CoilPart) -> np.ndarray:
    """Return each triangle's three local nodal surface-current vectors."""

    faces = np.asarray(part.mesh.faces, dtype=np.int64)
    currents = np.zeros((len(faces), 3, 3), dtype=float)
    for node_ind, basis in enumerate(part.basis_elements):
        triangle_indices = np.asarray(basis.triangles, dtype=np.int64)
        if len(triangle_indices) == 0:
            continue
        local_slots = np.argmax(faces[triangle_indices] == int(node_ind), axis=1)
        currents[triangle_indices, local_slots] = np.asarray(basis.current, dtype=float)
    return currents


def _surface_current_density_metric(part: CoilPart) -> np.ndarray:
    current_density_mat = np.asarray(part.current_density_mat, dtype=float)
    face_areas = _mesh_face_areas(part.mesh)
    return np.einsum("ifa,jfa,f->ij", current_density_mat, current_density_mat, face_areas, optimize=True)


def _surface_biharmonic_metric(mesh: Mesh) -> np.ndarray:
    """Return the intrinsic FEM biharmonic metric C.T M^-1 C.

    C is the cotangent stiffness matrix and M is the lumped surface mass.
    Unlike an unweighted graph Laplacian, this discretizes
    integral_S (Delta_S psi)^2 dA and remains physically meaningful under
    nonuniform but shape-regular remeshing.
    """

    vertex_count = int(mesh.n_vertices)
    if vertex_count == 0 or len(mesh.faces) == 0:
        return np.zeros((vertex_count, vertex_count), dtype=float)
    stiffness, mass = cotangent_stiffness_and_lumped_mass(mesh)
    valid_mass = np.isfinite(mass) & (mass > 0.0)
    if not np.all(valid_mass):
        raise ValueError(
            "Cannot assemble stream-function FEM smoothness: "
            f"{int(np.count_nonzero(~valid_mass))} vertices have zero surface mass."
        )
    inverse_mass = 1.0 / mass
    try:
        from scipy import sparse  # type: ignore

        stiffness_sparse = sparse.csr_matrix(stiffness)
        metric = (stiffness_sparse.T @ sparse.diags(inverse_mass, offsets=0, format="csr") @ stiffness_sparse).toarray()
    except Exception:
        metric = stiffness.T @ (inverse_mass[:, None] * stiffness)
    return 0.5 * (np.asarray(metric, dtype=float) + np.asarray(metric, dtype=float).T)


def _mesh_face_areas(mesh: Mesh) -> np.ndarray:
    vertices = mesh.vertices
    faces = mesh.faces
    point_a = vertices[faces[:, 0]]
    point_b = vertices[faces[:, 1]]
    point_c = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(point_b - point_a, point_c - point_a), axis=1)


def _median_mesh_edge_length(meshes: list[Mesh]) -> float:
    lengths: list[np.ndarray] = []
    for mesh in meshes:
        vertices = mesh.vertices
        faces = mesh.faces
        edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        finite = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
        if finite.size:
            lengths.append(finite)
    if not lengths:
        return 0.0
    return float(np.median(np.concatenate(lengths)))


def _matrix_mean_abs_diag(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    diag = np.abs(np.diag(matrix))
    finite = diag[np.isfinite(diag) & (diag > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _safe_positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number) or number <= 0.0:
        return 0.0
    return number


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float)]


def _normalise_stream_function_method(method: str) -> str:
    normalised = str(method).strip().lower().replace("-", "_")
    if normalised in {"tikkonov", "tikhonov", "tikhonov_regularized_least_squares"}:
        return "tikhonov"
    if normalised in {
        "target_error_constrained_efficiency",
        "efficiency_with_target_error_constraint",
        "fixed_error_efficiency",
        "minimum_resistance_subject_to_field_error",
    }:
        return "target_error_constrained_efficiency"
    if normalised in {
        "constrained_quadratic",
        "cvxpy_constrained",
        "minimum_resistance_constrained",
        "minimum_resistance_subject_to_field_error_socp",
        "direct_fixed_error",
        "socp",
        "second_order_cone",
    }:
        return "constrained_quadratic"
    return normalised


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
    if solver == "pinv":
        return np.linalg.pinv(lhs) @ rhs

    tolerance = max(float(getattr(config, "optimization_solver_residual_tolerance", 1e-8)), 0.0)
    try:
        candidate = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        if solver == "solve":
            raise
        return np.linalg.pinv(lhs) @ rhs

    residual = np.linalg.norm(lhs @ candidate - rhs) / max(np.linalg.norm(rhs), 1e-30)
    if np.all(np.isfinite(candidate)) and float(residual) <= tolerance:
        return candidate
    if solver == "solve":
        raise np.linalg.LinAlgError(f"Direct solve residual {residual:.3e} exceeds tolerance {tolerance:.3e}.")
    return np.linalg.pinv(lhs) @ rhs


def _block_diag(matrices: list[np.ndarray]) -> np.ndarray:
    total_rows = sum(matrix.shape[0] for matrix in matrices)
    total_cols = sum(matrix.shape[1] for matrix in matrices)
    out = np.zeros((total_rows, total_cols), dtype=float)
    row = 0
    col = 0
    for matrix in matrices:
        rows, cols = matrix.shape
        out[row : row + rows, col : col + cols] = matrix
        row += rows
        col += cols
    return out
