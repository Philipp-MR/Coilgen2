"""Stream-function problem assembly, gauges, basis reduction and physical quadratic metrics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .config import CoilGenConfig
from .config_semantics import normalize_optimization_method, normalize_objective_mode
from .engineering_limits import (
    engineering_contouring_enabled,
    engineering_surface_current_density_limit_A_per_m,
    min_wire_distance_m,
)
from .field_components import component_sample_values, field_component_values, sensitivity_component_matrix
from .geometry import combine_meshes
from .models import CoilPart, Mesh, TargetField
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


def _normalise_constrained_quadratic_objective_mode(config: CoilGenConfig) -> str:
    key = normalize_objective_mode(getattr(config, "constrained_quadratic_objective_mode", "balanced"))
    if key not in {"balanced", "smoothness_priority", "minimax_current_density"}:
        raise ValueError(
            "constrained_quadratic_objective_mode must be 'balanced', 'smoothness_priority', "
            "or 'minimax_current_density'."
        )
    return key


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


def _safe_positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number) or number <= 0.0:
        return 0.0
    return number


def _normalise_stream_function_method(method: str) -> str:
    return normalize_optimization_method(method)


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
