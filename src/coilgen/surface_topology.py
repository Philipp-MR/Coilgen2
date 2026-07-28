from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .models import CoilPart, ContourLine, Mesh
from .progress import ProgressReporter


@dataclass(frozen=True)
class FaceAdjacency:
    neighbors: list[list[int]]
    boundary_faces: np.ndarray
    non_manifold_edges: list[tuple[int, int]]
    edge_to_faces: dict[tuple[int, int], list[int]]


@dataclass(frozen=True)
class FaceCut:
    face_index: int
    bary_a: np.ndarray
    bary_b: np.ndarray
    cell_negative: int
    cell_positive: int


@dataclass(frozen=True)
class VirtualCutGraph:
    neighbors: list[list[int]]
    cell_face: np.ndarray
    cell_side: np.ndarray
    cell_areas: np.ndarray
    cell_centroids: np.ndarray
    boundary_cells: np.ndarray
    face_cells: list[tuple[int, ...]]
    cuts: dict[int, FaceCut]


@dataclass(frozen=True)
class BoundaryComponent:
    """One connected boundary component of an open surface mesh."""

    index: int
    vertices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    centroid: np.ndarray


@dataclass(frozen=True)
class LoopTopologySignature:
    """Topology classification for one contour loop on a surface."""

    loop_index: int
    component_boundary_sets: tuple[tuple[int, ...], ...]
    component_areas: tuple[float, ...]
    component_centroids: tuple[tuple[float, float, float], ...]
    boundary_signature: tuple[tuple[int, ...], ...]
    separating: bool
    has_boundary_free_side: bool
    boundary_associated: bool
    contractible_candidate: bool
    ambiguous: bool

    @property
    def kind(self) -> str:
        if self.contractible_candidate:
            return "contractible"
        if self.boundary_associated:
            return "boundary_associated"
        if self.separating:
            return "essential_separating"
        return "nonseparating"


@dataclass(frozen=True)
class TopologyValidationReport:
    errors: list[str]
    warnings: list[str]
    non_separating_loops: list[int]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "non_separating_loops": self.non_separating_loops.copy(),
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"Intrinsic topology preflight failed:\n{joined}")


@dataclass(frozen=True)
class MeshQualityReport:
    """User-facing mesh quality summary for stream-function support surfaces."""

    diagnostics: dict[str, object]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def as_dict(self) -> dict[str, object]:
        return {
            **self.diagnostics,
            "mesh_quality_ok": self.ok,
            "mesh_quality_error_count": len(self.errors),
            "mesh_quality_warning_count": len(self.warnings),
            "mesh_quality_errors": self.errors.copy(),
            "mesh_quality_warnings": self.warnings.copy(),
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"Mesh quality preflight failed:\n{joined}")


def mesh_surface_diagnostics(mesh: Mesh, *, check_self_intersections: bool = False) -> dict[str, object]:
    """Return compact diagnostics for mesh assumptions used by CoilGen."""

    diagnostics: dict[str, object] = {
        "vertex_count": mesh.n_vertices,
        "face_count": mesh.n_faces,
        "nonfinite_vertex_count": 0,
        "invalid_face_index_count": 0,
        "degenerate_face_vertex_count": 0,
        "zero_area_face_count": 0,
        "duplicate_face_count": 0,
        "boundary_edge_count": 0,
        "boundary_loop_count": 0,
        "boundary_loop_lengths": [],
        "connected_face_component_count": 0,
        "non_manifold_edge_count": 0,
        "orientable": False,
        "inconsistent_face_normal_count": 0,
        "self_intersection_count": 0,
        "edge_length_min_m": 0.0,
        "edge_length_p05_m": 0.0,
        "edge_length_mean_m": 0.0,
        "edge_length_p95_m": 0.0,
        "edge_length_max_m": 0.0,
        "face_area_min_m2": 0.0,
        "face_area_mean_m2": 0.0,
        "face_area_max_m2": 0.0,
        "triangle_edge_ratio_mean": 0.0,
        "triangle_edge_ratio_p95": 0.0,
        "triangle_edge_ratio_max": 0.0,
        "triangle_altitude_aspect_ratio_mean": 0.0,
        "triangle_altitude_aspect_ratio_p95": 0.0,
        "triangle_altitude_aspect_ratio_max": 0.0,
    }
    if mesh.n_faces == 0 or mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        return diagnostics
    diagnostics["nonfinite_vertex_count"] = int(np.count_nonzero(~np.all(np.isfinite(mesh.vertices), axis=1)))
    if mesh.n_vertices == 0:
        return diagnostics
    invalid_faces = np.any((mesh.faces < 0) | (mesh.faces >= mesh.n_vertices), axis=1)
    diagnostics["invalid_face_index_count"] = int(np.count_nonzero(invalid_faces))
    if bool(np.any(invalid_faces)):
        return diagnostics
    diagnostics["degenerate_face_vertex_count"] = int(
        np.count_nonzero([len(set(map(int, face))) < 3 for face in mesh.faces])
    )

    areas = face_areas(mesh)
    diagnostics["zero_area_face_count"] = int(np.count_nonzero(areas <= 1e-18))
    if len(areas):
        diagnostics["face_area_min_m2"] = float(np.min(areas))
        diagnostics["face_area_mean_m2"] = float(np.mean(areas))
        diagnostics["face_area_max_m2"] = float(np.max(areas))
    sorted_faces = np.sort(mesh.faces, axis=1)
    diagnostics["duplicate_face_count"] = int(len(sorted_faces) - len({tuple(map(int, face)) for face in sorted_faces}))
    edge_lengths, edge_ratios, altitude_ratios = _mesh_shape_quality_arrays(mesh, areas)
    if len(edge_lengths):
        diagnostics["edge_length_min_m"] = float(np.min(edge_lengths))
        diagnostics["edge_length_p05_m"] = float(np.percentile(edge_lengths, 5.0))
        diagnostics["edge_length_mean_m"] = float(np.mean(edge_lengths))
        diagnostics["edge_length_p95_m"] = float(np.percentile(edge_lengths, 95.0))
        diagnostics["edge_length_max_m"] = float(np.max(edge_lengths))
    if len(edge_ratios):
        diagnostics["triangle_edge_ratio_mean"] = float(np.mean(edge_ratios))
        diagnostics["triangle_edge_ratio_p95"] = float(np.percentile(edge_ratios, 95.0))
        diagnostics["triangle_edge_ratio_max"] = float(np.max(edge_ratios))
    if len(altitude_ratios):
        diagnostics["triangle_altitude_aspect_ratio_mean"] = float(np.mean(altitude_ratios))
        diagnostics["triangle_altitude_aspect_ratio_p95"] = float(np.percentile(altitude_ratios, 95.0))
        diagnostics["triangle_altitude_aspect_ratio_max"] = float(np.max(altitude_ratios))

    adjacency = build_face_adjacency(mesh)
    boundary_edges = [edge for edge, faces in adjacency.edge_to_faces.items() if len(faces) == 1]
    diagnostics["boundary_edge_count"] = int(len(boundary_edges))
    boundary_loop_lengths = _boundary_loop_lengths(boundary_edges)
    diagnostics["boundary_loop_count"] = int(len(boundary_loop_lengths))
    diagnostics["boundary_loop_lengths"] = [int(length) for length in boundary_loop_lengths]
    diagnostics["connected_face_component_count"] = int(_connected_face_component_count(adjacency))
    diagnostics["non_manifold_edge_count"] = int(len(adjacency.non_manifold_edges))
    diagnostics["orientable"] = bool(not adjacency.non_manifold_edges and _mesh_is_orientable(mesh, adjacency))

    if mesh.face_normals is not None and len(mesh.face_normals) == mesh.n_faces:
        computed = _computed_face_normals(mesh)
        provided = np.asarray(mesh.face_normals, dtype=float)
        norms = np.linalg.norm(provided, axis=1)
        valid = norms > 1e-15
        dots = np.einsum("ij,ij->i", computed[valid], provided[valid] / norms[valid, None])
        diagnostics["inconsistent_face_normal_count"] = int(np.count_nonzero(dots < -0.5))

    if check_self_intersections:
        diagnostics["self_intersection_count"] = int(_obvious_self_intersection_count(mesh))
    return diagnostics


def mesh_quality_report(
    mesh: Mesh,
    *,
    check_self_intersections: bool = False,
    max_triangle_edge_ratio: float = 25.0,
    max_altitude_aspect_ratio: float = 50.0,
    max_edge_length_ratio: float = 100.0,
) -> MeshQualityReport:
    """Return errors and warnings for a mesh before stream-function design.

    Errors mark conditions that invalidate topology or current orientation.
    Warnings mark poor numerical quality that can still run but may produce
    unstable contours or poorly conditioned optimization matrices.
    """

    diagnostics = mesh_surface_diagnostics(mesh, check_self_intersections=check_self_intersections)
    errors: list[str] = []
    warnings: list[str] = []

    if int(diagnostics["vertex_count"]) == 0:
        errors.append("Mesh has no vertices.")
    if int(diagnostics["face_count"]) == 0:
        errors.append("Mesh has no triangular faces.")
    if int(diagnostics["nonfinite_vertex_count"]):
        errors.append(f"Mesh has {diagnostics['nonfinite_vertex_count']} non-finite vertices.")
    if int(diagnostics["invalid_face_index_count"]):
        errors.append(f"Mesh has {diagnostics['invalid_face_index_count']} faces with invalid vertex indices.")
    if int(diagnostics["degenerate_face_vertex_count"]):
        errors.append(f"Mesh has {diagnostics['degenerate_face_vertex_count']} faces with repeated vertices.")
    if int(diagnostics["zero_area_face_count"]):
        errors.append(f"Mesh has {diagnostics['zero_area_face_count']} zero-area faces.")
    if int(diagnostics["non_manifold_edge_count"]):
        errors.append(f"Mesh has {diagnostics['non_manifold_edge_count']} non-manifold edges.")
    if int(diagnostics["face_count"]) and not bool(diagnostics["orientable"]):
        errors.append("Mesh is not orientable.")
    if int(diagnostics["self_intersection_count"]):
        errors.append(f"Mesh has {diagnostics['self_intersection_count']} obvious self-intersecting triangle pairs.")

    if int(diagnostics["duplicate_face_count"]):
        warnings.append(f"Mesh has {diagnostics['duplicate_face_count']} duplicate faces.")
    if int(diagnostics["connected_face_component_count"]) > 1:
        warnings.append(f"Mesh has {diagnostics['connected_face_component_count']} disconnected face components.")
    if int(diagnostics["inconsistent_face_normal_count"]):
        warnings.append(
            f"Mesh has {diagnostics['inconsistent_face_normal_count']} supplied normals opposed to face winding."
        )
    if float(diagnostics["triangle_edge_ratio_max"]) > float(max_triangle_edge_ratio):
        warnings.append(
            f"Mesh has elongated triangles: max edge-length ratio {diagnostics['triangle_edge_ratio_max']:.3g}."
        )
    if float(diagnostics["triangle_altitude_aspect_ratio_max"]) > float(max_altitude_aspect_ratio):
        warnings.append(
            "Mesh has very small triangle altitudes: "
            f"max altitude aspect ratio {diagnostics['triangle_altitude_aspect_ratio_max']:.3g}."
        )
    min_edge = float(diagnostics["edge_length_min_m"])
    max_edge = float(diagnostics["edge_length_max_m"])
    if min_edge > 0.0 and max_edge / min_edge > float(max_edge_length_ratio):
        warnings.append(f"Mesh edge lengths vary strongly: max/min edge ratio {max_edge / min_edge:.3g}.")

    return MeshQualityReport(diagnostics=diagnostics, errors=errors, warnings=warnings)


def validate_intrinsic_topology_inputs(
    part: CoilPart,
    *,
    check_separating: bool = False,
    check_self_intersections: bool = False,
    progress: ProgressReporter | None = None,
) -> TopologyValidationReport:
    """Validate mesh and contour assumptions required by intrinsic topology."""

    errors: list[str] = []
    warnings: list[str] = []
    non_separating_loops: list[int] = []

    mesh = part.mesh
    if mesh.n_vertices == 0:
        errors.append("Mesh has no vertices.")
    if mesh.n_faces == 0:
        errors.append("Mesh has no triangular faces.")
    if not np.all(np.isfinite(mesh.vertices)):
        errors.append("Mesh vertices contain non-finite coordinates.")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        errors.append("Mesh faces must be triangular with shape (n_faces, 3).")

    faces_in_range = bool(
        mesh.faces.ndim == 2
        and mesh.faces.shape[1] == 3
        and (mesh.n_faces == 0 or (np.min(mesh.faces) >= 0 and np.max(mesh.faces) < mesh.n_vertices))
    )
    if not faces_in_range and mesh.n_faces:
        errors.append("Mesh faces reference vertex indices outside the mesh.")

    duplicate_faces = int(np.count_nonzero([len(set(map(int, face))) < 3 for face in mesh.faces]))
    if duplicate_faces:
        errors.append(f"Mesh contains {duplicate_faces} degenerate faces with repeated vertices.")

    adjacency: FaceAdjacency | None = None
    if faces_in_range and mesh.n_faces:
        mesh_diagnostics = mesh_surface_diagnostics(mesh, check_self_intersections=check_self_intersections)
        areas = face_areas(mesh)
        zero_area_faces = int(np.count_nonzero(areas <= 1e-18))
        if zero_area_faces:
            errors.append(f"Mesh contains {zero_area_faces} zero-area faces.")
        adjacency = build_face_adjacency(mesh)
        if adjacency.non_manifold_edges:
            errors.append(f"Mesh contains {len(adjacency.non_manifold_edges)} non-manifold edges.")
        elif not _mesh_is_orientable(mesh, adjacency):
            errors.append("Mesh is not orientable; intrinsic loop containment requires an orientable surface.")
        if int(mesh_diagnostics["connected_face_component_count"]) > 1:
            warnings.append(
                f"Mesh contains {mesh_diagnostics['connected_face_component_count']} disconnected face components; "
                "split disconnected coil parts before layout interconnection."
            )
        if int(mesh_diagnostics["inconsistent_face_normal_count"]):
            warnings.append(
                f"Mesh has {mesh_diagnostics['inconsistent_face_normal_count']} provided face normals opposed to computed orientation."
            )
        if int(mesh_diagnostics["self_intersection_count"]):
            errors.append(
                f"Mesh contains {mesh_diagnostics['self_intersection_count']} obvious self-intersecting triangle pairs."
            )

    boundary_edges = set()
    if adjacency is not None:
        boundary_edges = {edge for edge, faces in adjacency.edge_to_faces.items() if len(faces) == 1}

    for loop_ind, contour in enumerate(part.contour_lines):
        _validate_contour_provenance(contour, loop_ind, mesh.n_faces, boundary_edges, errors)

    if check_separating and adjacency is not None and not errors:
        task = progress.task("separating-loop validation", len(part.contour_lines)) if progress else None
        for loop_ind, contour in enumerate(part.contour_lines):
            cut_graph = build_virtual_cut_graph(mesh, contour, adjacency)
            labels = label_cut_components(cut_graph)
            components = np.unique(labels[labels >= 0])
            interior_component = choose_interior_component(labels, cut_graph.cell_areas, cut_graph.boundary_cells)
            if len(components) < 2 or interior_component is None:
                non_separating_loops.append(loop_ind)
                warnings.append(
                    f"Contour {loop_ind} is non-separating, leaking through the mesh, or does not bound an interior surface component."
                )
            if task is not None:
                task.advance(
                    1,
                    message=f"loop {loop_ind + 1}/{len(part.contour_lines)}; non-separating warnings={len(non_separating_loops)}",
                )
        if task is not None:
            task.done()

    return TopologyValidationReport(errors=errors, warnings=warnings, non_separating_loops=non_separating_loops)


def build_intrinsic_loop_inclusion_matrix(
    part: CoilPart,
    membership_threshold: float = 0.9,
    *,
    validate: bool = True,
    progress: ProgressReporter | None = None,
) -> np.ndarray:
    """Build contour containment from intrinsic surface reachability.

    Each tested contour is converted into a virtual cut graph. Crossed mesh
    faces are split into two side-cells, and adjacency across original mesh
    edges is connected only where the same side of the contour touches the edge.
    This makes the contour an impenetrable wall without deleting a band of
    crossed faces.
    """

    if validate:
        validate_intrinsic_topology_inputs(part, progress=progress).raise_for_errors()

    loops = part.contour_lines
    contains = np.zeros((len(loops), len(loops)), dtype=bool)
    if len(loops) < 2:
        return contains

    adjacency = build_face_adjacency(part.mesh)
    task = progress.task("intrinsic loop inclusion", len(loops)) if progress else None
    for outer_ind, outer in enumerate(loops):
        if task is not None:
            task.update(
                outer_ind,
                message=f"outer loop {outer_ind + 1}/{len(loops)}; voting against {max(len(loops) - 1, 0)} loops",
                force=outer_ind == 0,
            )
        cut_graph = build_virtual_cut_graph(part.mesh, outer, adjacency)
        if not cut_graph.cuts:
            if task is not None:
                task.advance(1, message=f"outer loop {outer_ind + 1}/{len(loops)} has no surface cuts")
            continue
        labels = label_cut_components(cut_graph)
        interior_component = choose_interior_component(labels, cut_graph.cell_areas, cut_graph.boundary_cells)
        if interior_component is None:
            if task is not None:
                task.advance(1, message=f"outer loop {outer_ind + 1}/{len(loops)} has no interior component")
            continue

        for inner_ind, inner in enumerate(loops):
            if inner_ind == outer_ind:
                continue
            component, confidence = contour_component_vote(inner, cut_graph, labels)
            contains[outer_ind, inner_ind] = component == interior_component and confidence >= membership_threshold
        if task is not None:
            contained_count = int(np.count_nonzero(contains[outer_ind]))
            task.advance(1, message=f"outer loop {outer_ind + 1}/{len(loops)} contains {contained_count} loop(s)")
    if task is not None:
        task.done()
    return contains


def boundary_components(mesh: Mesh, adjacency: FaceAdjacency | None = None) -> list[BoundaryComponent]:
    """Return connected components of the mesh boundary graph."""

    adjacency = build_face_adjacency(mesh) if adjacency is None else adjacency
    boundary_edges = [edge for edge, faces in adjacency.edge_to_faces.items() if len(faces) == 1]
    if not boundary_edges:
        return []

    neighbors: dict[int, set[int]] = defaultdict(set)
    for node_a, node_b in boundary_edges:
        neighbors[int(node_a)].add(int(node_b))
        neighbors[int(node_b)].add(int(node_a))

    edge_set = {tuple(sorted(edge)) for edge in boundary_edges}
    seen_vertices: set[int] = set()
    components: list[BoundaryComponent] = []
    for start in sorted(neighbors):
        if start in seen_vertices:
            continue
        queue: deque[int] = deque([start])
        seen_vertices.add(start)
        vertices: set[int] = set()
        edges: set[tuple[int, int]] = set()
        while queue:
            node = queue.popleft()
            vertices.add(int(node))
            for neighbor in neighbors[node]:
                edge = tuple(sorted((int(node), int(neighbor))))
                if edge in edge_set:
                    edges.add(edge)
                if neighbor not in seen_vertices:
                    seen_vertices.add(neighbor)
                    queue.append(neighbor)
        ordered_vertices = tuple(sorted(vertices))
        centroid = (
            np.mean(mesh.vertices[np.asarray(ordered_vertices, dtype=np.int64)], axis=0)
            if ordered_vertices
            else np.zeros(3, dtype=float)
        )
        components.append(
            BoundaryComponent(
                index=len(components),
                vertices=ordered_vertices,
                edges=tuple(sorted(edges)),
                centroid=np.asarray(centroid, dtype=float),
            )
        )
    return components


def classify_loop_topology_signatures(
    part: CoilPart,
    *,
    adjacency: FaceAdjacency | None = None,
) -> list[LoopTopologySignature]:
    """Classify contour loops by the boundary components of their cut regions."""

    adjacency = build_face_adjacency(part.mesh) if adjacency is None else adjacency
    components = boundary_components(part.mesh, adjacency)
    edge_to_boundary = {edge: component.index for component in components for edge in component.edges}
    signatures: list[LoopTopologySignature] = []
    for loop_ind, contour in enumerate(part.contour_lines):
        signatures.append(_classify_loop_topology_signature(part.mesh, contour, loop_ind, adjacency, edge_to_boundary))
    return signatures


def _classify_loop_topology_signature(
    mesh: Mesh,
    contour: ContourLine,
    loop_ind: int,
    adjacency: FaceAdjacency,
    edge_to_boundary: dict[tuple[int, int], int],
) -> LoopTopologySignature:
    cut_graph = build_virtual_cut_graph(mesh, contour, adjacency)
    labels = label_cut_components(cut_graph)
    components = tuple(int(component) for component in np.unique(labels[labels >= 0]))
    if not components:
        return LoopTopologySignature(
            loop_index=loop_ind,
            component_boundary_sets=(),
            component_areas=(),
            component_centroids=(),
            boundary_signature=(),
            separating=False,
            has_boundary_free_side=False,
            boundary_associated=False,
            contractible_candidate=False,
            ambiguous=True,
        )

    boundary_sets = _cut_component_boundary_sets(mesh, adjacency, cut_graph, labels, edge_to_boundary)
    component_boundary_sets: list[tuple[int, ...]] = []
    component_areas: list[float] = []
    component_centroids: list[tuple[float, float, float]] = []
    for component in components:
        mask = labels == component
        boundary_set = tuple(sorted(boundary_sets.get(component, ())))
        component_boundary_sets.append(boundary_set)
        area = float(np.sum(cut_graph.cell_areas[mask]))
        component_areas.append(area)
        component_centroids.append(tuple(float(value) for value in _weighted_component_centroid(cut_graph, mask)))

    boundary_signature = tuple(sorted(set(component_boundary_sets)))
    separating = len(components) >= 2
    has_mesh_boundaries = bool(edge_to_boundary)
    has_boundary_free_side = bool(
        has_mesh_boundaries and any(len(boundary_set) == 0 for boundary_set in component_boundary_sets)
    )
    all_sides_touch_boundary = all(len(boundary_set) > 0 for boundary_set in component_boundary_sets)
    boundary_associated = bool(separating and all_sides_touch_boundary and len(boundary_signature) >= 2)
    contractible_candidate = bool(separating and has_boundary_free_side)
    ambiguous = bool(len(components) > 2)
    return LoopTopologySignature(
        loop_index=loop_ind,
        component_boundary_sets=tuple(component_boundary_sets),
        component_areas=tuple(component_areas),
        component_centroids=tuple(component_centroids),
        boundary_signature=boundary_signature,
        separating=separating,
        has_boundary_free_side=has_boundary_free_side,
        boundary_associated=boundary_associated,
        contractible_candidate=contractible_candidate,
        ambiguous=ambiguous,
    )


def _cut_component_boundary_sets(
    mesh: Mesh,
    adjacency: FaceAdjacency,
    cut_graph: VirtualCutGraph,
    labels: np.ndarray,
    edge_to_boundary: dict[tuple[int, int], int],
) -> dict[int, set[int]]:
    boundary_sets: dict[int, set[int]] = defaultdict(set)
    for edge, faces in adjacency.edge_to_faces.items():
        if len(faces) != 1:
            continue
        boundary_index = edge_to_boundary.get(tuple(sorted(edge)))
        if boundary_index is None:
            continue
        face = faces[0]
        for cell, _start, _end in _cells_touching_edge(mesh, face, edge, cut_graph.face_cells, cut_graph.cuts):
            component = int(labels[cell])
            if component >= 0:
                boundary_sets[component].add(int(boundary_index))
    return boundary_sets


def _weighted_component_centroid(cut_graph: VirtualCutGraph, mask: np.ndarray) -> np.ndarray:
    centroids = np.asarray(cut_graph.cell_centroids[mask], dtype=float)
    areas = np.asarray(cut_graph.cell_areas[mask], dtype=float)
    if len(centroids) == 0:
        return np.zeros(3, dtype=float)
    total_area = float(np.sum(areas))
    if total_area <= 1e-30 or not np.isfinite(total_area):
        return np.mean(centroids, axis=0)
    return np.sum(centroids * areas[:, None], axis=0) / total_area


def build_face_adjacency(mesh: Mesh) -> FaceAdjacency:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_ind, face in enumerate(mesh.faces):
        for node_a, node_b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(node_a, node_b)), int(max(node_a, node_b)))
            edge_to_faces[edge].append(face_ind)

    neighbors = [set() for _ in range(mesh.n_faces)]
    boundary_faces = np.zeros(mesh.n_faces, dtype=bool)
    non_manifold_edges: list[tuple[int, int]] = []
    for edge, faces in edge_to_faces.items():
        if len(faces) == 1:
            boundary_faces[faces[0]] = True
        elif len(faces) == 2:
            face_a, face_b = faces
            neighbors[face_a].add(face_b)
            neighbors[face_b].add(face_a)
        else:
            non_manifold_edges.append(edge)
            for face_a in faces:
                for face_b in faces:
                    if face_a != face_b:
                        neighbors[face_a].add(face_b)
    return FaceAdjacency(
        neighbors=[sorted(face_neighbors) for face_neighbors in neighbors],
        boundary_faces=boundary_faces,
        non_manifold_edges=non_manifold_edges,
        edge_to_faces={edge: faces.copy() for edge, faces in edge_to_faces.items()},
    )


def _mesh_is_orientable(mesh: Mesh, adjacency: FaceAdjacency) -> bool:
    face_edge_signs: list[dict[tuple[int, int], int]] = []
    for face in mesh.faces:
        signs: dict[tuple[int, int], int] = {}
        for node_a, node_b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            directed = (int(node_a), int(node_b))
            edge = (min(directed), max(directed))
            signs[edge] = 1 if directed == edge else -1
        face_edge_signs.append(signs)

    constraints: list[list[tuple[int, int]]] = [[] for _ in range(mesh.n_faces)]
    for edge, faces in adjacency.edge_to_faces.items():
        if len(faces) != 2:
            continue
        face_a, face_b = faces
        same_direction = face_edge_signs[face_a][edge] == face_edge_signs[face_b][edge]
        required_flip = 1 if same_direction else 0
        constraints[face_a].append((face_b, required_flip))
        constraints[face_b].append((face_a, required_flip))

    orientation = np.full(mesh.n_faces, -1, dtype=np.int8)
    for start_face in range(mesh.n_faces):
        if orientation[start_face] >= 0:
            continue
        orientation[start_face] = 0
        queue: deque[int] = deque([start_face])
        while queue:
            face = queue.popleft()
            for neighbor, required_flip in constraints[face]:
                expected = orientation[face] ^ required_flip
                if orientation[neighbor] < 0:
                    orientation[neighbor] = expected
                    queue.append(neighbor)
                elif orientation[neighbor] != expected:
                    return False
    return True


def _boundary_loop_count(boundary_edges: list[tuple[int, int]]) -> int:
    return len(_boundary_loop_lengths(boundary_edges))


def _boundary_loop_lengths(boundary_edges: list[tuple[int, int]]) -> list[int]:
    if not boundary_edges:
        return []
    neighbors: dict[int, set[int]] = defaultdict(set)
    for node_a, node_b in boundary_edges:
        neighbors[int(node_a)].add(int(node_b))
        neighbors[int(node_b)].add(int(node_a))
    seen: set[int] = set()
    lengths: list[int] = []
    for node in neighbors:
        if node in seen:
            continue
        length = 0
        queue: deque[int] = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            length += 1
            for neighbor in neighbors[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        lengths.append(length)
    return lengths


def _mesh_shape_quality_arrays(mesh: Mesh, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = mesh.vertices[mesh.faces]
    edge_a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    edge_b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    edge_c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    face_edges = np.column_stack((edge_a, edge_b, edge_c))
    finite = np.all(np.isfinite(face_edges), axis=1)
    positive_edges = np.all(face_edges > 1e-15, axis=1)
    positive_area = np.asarray(areas, dtype=float) > 1e-18
    valid = finite & positive_edges & positive_area
    if not np.any(valid):
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    valid_edges = face_edges[valid]
    valid_areas = areas[valid]
    edge_lengths = np.unique(np.round(valid_edges.reshape(-1), 15))
    shortest = np.min(valid_edges, axis=1)
    longest = np.max(valid_edges, axis=1)
    edge_ratios = longest / shortest
    min_altitude = 2.0 * valid_areas / longest
    altitude_ratios = longest / min_altitude
    return edge_lengths, edge_ratios, altitude_ratios


def _connected_face_component_count(adjacency: FaceAdjacency) -> int:
    seen = np.zeros(len(adjacency.neighbors), dtype=bool)
    count = 0
    for face in range(len(adjacency.neighbors)):
        if seen[face]:
            continue
        count += 1
        queue: deque[int] = deque([face])
        seen[face] = True
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.neighbors[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
    return count


def _computed_face_normals(mesh: Mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-15
    out = np.zeros_like(normals)
    out[valid] = normals[valid] / norms[valid, None]
    return out


def _obvious_self_intersection_count(mesh: Mesh) -> int:
    tri = mesh.vertices[mesh.faces]
    mins = np.min(tri, axis=1)
    maxs = np.max(tri, axis=1)
    count = 0
    for ind_a in range(mesh.n_faces):
        nodes_a = set(map(int, mesh.faces[ind_a]))
        for ind_b in range(ind_a + 1, mesh.n_faces):
            if nodes_a.intersection(map(int, mesh.faces[ind_b])):
                continue
            if np.any(maxs[ind_a] < mins[ind_b]) or np.any(maxs[ind_b] < mins[ind_a]):
                continue
            if _triangle_pair_intersects(tri[ind_a], tri[ind_b]):
                count += 1
    return count


def _triangle_pair_intersects(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    for point_a, point_b in ((tri_a[0], tri_a[1]), (tri_a[1], tri_a[2]), (tri_a[2], tri_a[0])):
        if _segment_triangle_intersection(point_a, point_b, tri_b):
            return True
    for point_a, point_b in ((tri_b[0], tri_b[1]), (tri_b[1], tri_b[2]), (tri_b[2], tri_b[0])):
        if _segment_triangle_intersection(point_a, point_b, tri_a):
            return True
    return False


def _segment_triangle_intersection(start: np.ndarray, end: np.ndarray, triangle: np.ndarray) -> bool:
    direction = end - start
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    h_vec = np.cross(direction, edge_2)
    det = float(np.dot(edge_1, h_vec))
    if abs(det) < 1e-12:
        return False
    inv_det = 1.0 / det
    s_vec = start - triangle[0]
    u_val = inv_det * float(np.dot(s_vec, h_vec))
    if u_val < -1e-10 or u_val > 1.0 + 1e-10:
        return False
    q_vec = np.cross(s_vec, edge_1)
    v_val = inv_det * float(np.dot(direction, q_vec))
    if v_val < -1e-10 or u_val + v_val > 1.0 + 1e-10:
        return False
    t_val = inv_det * float(np.dot(edge_2, q_vec))
    return bool(-1e-10 <= t_val <= 1.0 + 1e-10)


def _validate_contour_provenance(
    contour: ContourLine,
    loop_ind: int,
    n_faces: int,
    boundary_edges: set[tuple[int, int]],
    errors: list[str],
) -> None:
    prefix = f"Contour {loop_ind}"
    if len(contour.v) < 4:
        errors.append(f"{prefix} has fewer than four points.")
    if len(contour.v) and not np.all(np.isfinite(contour.v)):
        errors.append(f"{prefix} contains non-finite 3D points.")
    if len(contour.v) >= 2 and np.linalg.norm(contour.v[0] - contour.v[-1]) > 1e-8:
        errors.append(f"{prefix} is not closed.")

    segment_count = max(len(contour.v) - 1, 0)
    if contour.segment_faces is None or contour.segment_barycentric is None:
        errors.append(f"{prefix} has no surface segment provenance.")
        return
    segment_faces = np.asarray(contour.segment_faces, dtype=np.int64)
    segment_barycentric = np.asarray(contour.segment_barycentric, dtype=float)
    if len(segment_faces) != segment_count:
        errors.append(f"{prefix} segment face count does not match contour segment count.")
    if segment_barycentric.shape != (segment_count, 2, 3):
        errors.append(f"{prefix} segment barycentric coordinates must have shape ({segment_count}, 2, 3).")
        return
    if len(segment_faces) and (np.min(segment_faces) < 0 or np.max(segment_faces) >= n_faces):
        errors.append(f"{prefix} references faces outside the mesh.")
    face_runs = _contour_face_runs(segment_faces, segment_barycentric)
    repeated_passage_count = len(face_runs) - len({face for face, _, _ in face_runs})
    if repeated_passage_count:
        errors.append(
            f"{prefix} passes through at least one face in disconnected runs; "
            "the current virtual cut graph supports one passage per face."
        )
    if not np.all(np.isfinite(segment_barycentric)):
        errors.append(f"{prefix} segment barycentric coordinates contain non-finite values.")
    bary_sums = np.sum(segment_barycentric, axis=2)
    if not np.all(np.isclose(bary_sums, 1.0, atol=1e-7)):
        errors.append(f"{prefix} segment barycentric coordinates do not sum to one.")
    if np.any(segment_barycentric < -1e-8) or np.any(segment_barycentric > 1.0 + 1e-8):
        errors.append(f"{prefix} segment barycentric coordinates fall outside triangle bounds.")
    if _touches_triangle_vertex(segment_barycentric):
        errors.append(f"{prefix} touches a mesh vertex; the virtual cut would be ambiguous.")
    if _has_degenerate_segments(contour.v):
        errors.append(f"{prefix} contains degenerate 3D segments.")

    if contour.surface_edge_vertices is not None and boundary_edges:
        edge_vertices = np.asarray(contour.surface_edge_vertices, dtype=np.int64)
        touched_boundary = any(
            (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))) in boundary_edges
            for edge in edge_vertices
        )
        if touched_boundary:
            errors.append(f"{prefix} touches the mesh boundary; containment is undefined for this loop.")


def _touches_triangle_vertex(segment_barycentric: np.ndarray) -> bool:
    flat = segment_barycentric.reshape(-1, 3)
    return bool(np.any(np.max(flat, axis=1) >= 1.0 - 1e-8))


def _has_degenerate_segments(points: np.ndarray) -> bool:
    if len(points) < 2:
        return False
    return bool(np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-12))


def face_areas(mesh: Mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def face_centroids(mesh: Mesh) -> np.ndarray:
    return np.mean(mesh.vertices[mesh.faces], axis=1)


def build_virtual_cut_graph(
    mesh: Mesh,
    contour: ContourLine,
    adjacency: FaceAdjacency | None = None,
) -> VirtualCutGraph:
    if contour.segment_faces is None or contour.segment_barycentric is None:
        raise ValueError("Virtual cut topology requires contour segment face and barycentric provenance.")
    adjacency = build_face_adjacency(mesh) if adjacency is None else adjacency
    cuts_without_cells = _contour_face_cuts(contour, mesh.n_faces)

    cell_face: list[int] = []
    cell_side: list[int] = []
    cell_areas: list[float] = []
    cell_centroids: list[np.ndarray] = []
    face_cells: list[tuple[int, ...]] = []
    cuts: dict[int, FaceCut] = {}
    original_areas = face_areas(mesh)
    original_centroids = face_centroids(mesh)

    for face_ind in range(mesh.n_faces):
        raw_cut = cuts_without_cells.get(face_ind)
        if raw_cut is None:
            cell_id = len(cell_face)
            cell_face.append(face_ind)
            cell_side.append(0)
            cell_areas.append(float(original_areas[face_ind]))
            cell_centroids.append(original_centroids[face_ind])
            face_cells.append((cell_id,))
            continue

        neg_cell = len(cell_face)
        pos_cell = neg_cell + 1
        neg_area, neg_centroid = _cut_side_geometry(mesh, face_ind, raw_cut[0], raw_cut[1], side=-1)
        pos_area, pos_centroid = _cut_side_geometry(mesh, face_ind, raw_cut[0], raw_cut[1], side=1)
        cell_face.extend([face_ind, face_ind])
        cell_side.extend([-1, 1])
        cell_areas.extend([neg_area, pos_area])
        cell_centroids.extend([neg_centroid, pos_centroid])
        face_cells.append((neg_cell, pos_cell))
        cuts[face_ind] = FaceCut(
            face_index=face_ind,
            bary_a=raw_cut[0],
            bary_b=raw_cut[1],
            cell_negative=neg_cell,
            cell_positive=pos_cell,
        )

    neighbors = [set() for _ in range(len(cell_face))]
    boundary_cells = np.zeros(len(cell_face), dtype=bool)
    for edge, edge_faces in adjacency.edge_to_faces.items():
        if len(edge_faces) == 1:
            face = edge_faces[0]
            for cell, _, _ in _cells_touching_edge(mesh, face, edge, face_cells, cuts):
                boundary_cells[cell] = True
            continue
        if len(edge_faces) != 2:
            continue
        face_a, face_b = edge_faces
        intervals_a = _cells_touching_edge(mesh, face_a, edge, face_cells, cuts)
        intervals_b = _cells_touching_edge(mesh, face_b, edge, face_cells, cuts)
        for cell_a, start_a, end_a in intervals_a:
            for cell_b, start_b, end_b in intervals_b:
                if _interval_overlap(start_a, end_a, start_b, end_b) > 1e-12:
                    neighbors[cell_a].add(cell_b)
                    neighbors[cell_b].add(cell_a)

    return VirtualCutGraph(
        neighbors=[sorted(cell_neighbors) for cell_neighbors in neighbors],
        cell_face=np.asarray(cell_face, dtype=np.int64),
        cell_side=np.asarray(cell_side, dtype=np.int64),
        cell_areas=np.asarray(cell_areas, dtype=float),
        cell_centroids=np.asarray(cell_centroids, dtype=float),
        boundary_cells=boundary_cells,
        face_cells=face_cells,
        cuts=cuts,
    )


def _contour_face_cuts(contour: ContourLine, n_faces: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if contour.segment_faces is None:
        raise ValueError("Intrinsic topology requires contour segment face provenance.")
    if contour.segment_barycentric is None:
        raise ValueError("Intrinsic topology requires contour segment barycentric provenance.")
    segment_faces = np.asarray(contour.segment_faces, dtype=np.int64)
    segment_barycentric = np.asarray(contour.segment_barycentric, dtype=float)
    cuts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for face, start_barycentric, end_barycentric in _contour_face_runs(
        segment_faces,
        segment_barycentric,
    ):
        if not (0 <= face < n_faces):
            continue
        if face not in cuts:
            cuts[face] = (start_barycentric.copy(), end_barycentric.copy())
    return cuts


def _contour_face_runs(
    segment_faces: np.ndarray,
    segment_barycentric: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Collapse consecutive samples in one triangle into one topological cut.

    A parameterized B-spline needs several samples within a coarse triangle to
    retain its shape.  Those consecutive segments are one connected passage
    and have the same topology as the chord joining the first and last sample.
    Re-entering a face later in the loop remains invalid and is reported by the
    provenance validator.
    """

    faces = np.asarray(segment_faces, dtype=np.int64)
    barycentric = np.asarray(segment_barycentric, dtype=float)
    if len(faces) == 0:
        return []
    runs: list[tuple[int, np.ndarray, np.ndarray]] = []
    run_face = int(faces[0])
    run_start = barycentric[0, 0].copy()
    run_end = barycentric[0, 1].copy()
    for segment_index in range(1, len(faces)):
        face = int(faces[segment_index])
        if face == run_face:
            run_end = barycentric[segment_index, 1].copy()
            continue
        runs.append((run_face, run_start, run_end))
        run_face = face
        run_start = barycentric[segment_index, 0].copy()
        run_end = barycentric[segment_index, 1].copy()
    runs.append((run_face, run_start, run_end))
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        merged = (runs[0][0], runs[-1][1].copy(), runs[0][2].copy())
        runs = [merged, *runs[1:-1]]
    return runs


def label_cut_components(cut_graph: VirtualCutGraph) -> np.ndarray:
    labels = np.full(len(cut_graph.neighbors), -1, dtype=np.int64)
    component = 0
    for start_cell in range(len(cut_graph.neighbors)):
        if labels[start_cell] >= 0:
            continue
        queue: deque[int] = deque([start_cell])
        labels[start_cell] = component
        while queue:
            cell = queue.popleft()
            for neighbor in cut_graph.neighbors[cell]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    queue.append(neighbor)
        component += 1
    return labels


def choose_interior_component(
    labels: np.ndarray,
    areas: np.ndarray,
    boundary_faces: np.ndarray,
) -> int | None:
    components = np.unique(labels[labels >= 0])
    if len(components) < 2:
        return None

    component_areas = np.asarray([np.sum(areas[labels == component]) for component in components], dtype=float)
    component_touches_boundary = np.asarray(
        [np.any(boundary_faces[labels == component]) for component in components],
        dtype=bool,
    )
    candidates = components[~component_touches_boundary]
    candidate_areas = component_areas[~component_touches_boundary]
    if len(candidates) == 0:
        if np.any(boundary_faces):
            return None
        candidates = components
        candidate_areas = component_areas
    return int(candidates[int(np.argmin(candidate_areas))])


def contour_component_vote(
    contour: ContourLine,
    cut_graph: VirtualCutGraph,
    labels: np.ndarray,
) -> tuple[int | None, float]:
    if contour.segment_faces is None:
        raise ValueError("Intrinsic topology requires contour face provenance.")
    if contour.segment_barycentric is None:
        raise ValueError("Intrinsic topology requires contour segment barycentric provenance.")

    valid_labels = []
    segment_faces = np.asarray(contour.segment_faces, dtype=np.int64)
    segment_barycentric = np.asarray(contour.segment_barycentric, dtype=float)
    for segment_ind, face in enumerate(segment_faces):
        sample_bary = 0.5 * (segment_barycentric[segment_ind, 0] + segment_barycentric[segment_ind, 1])
        cell = locate_cut_cell(cut_graph, int(face), sample_bary)
        if cell is None:
            continue
        label = int(labels[cell])
        if label >= 0:
            valid_labels.append(label)
    if not valid_labels:
        return None, 0.0

    values, counts = np.unique(np.asarray(valid_labels, dtype=np.int64), return_counts=True)
    winner_ind = int(np.argmax(counts))
    confidence = float(counts[winner_ind] / len(valid_labels))
    return int(values[winner_ind]), confidence


def locate_cut_cell(cut_graph: VirtualCutGraph, face: int, barycentric: np.ndarray) -> int | None:
    if not (0 <= face < len(cut_graph.face_cells)):
        return None
    cells = cut_graph.face_cells[face]
    if len(cells) == 1:
        return int(cells[0])
    cut = cut_graph.cuts.get(face)
    if cut is None:
        return int(cells[0])
    sign = _cut_side_value(cut.bary_a, cut.bary_b, np.asarray(barycentric, dtype=float))
    return cut.cell_positive if sign >= 0 else cut.cell_negative


def _cut_side_geometry(
    mesh: Mesh,
    face_ind: int,
    bary_a: np.ndarray,
    bary_b: np.ndarray,
    side: int,
) -> tuple[float, np.ndarray]:
    local_polygon = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    clipped = _clip_local_triangle_by_cut(local_polygon, bary_a, bary_b, side)
    face_area = face_areas(mesh)[face_ind]
    local_area = abs(_polygon_area_2d(clipped))
    area = float(face_area * local_area / 0.5) if len(clipped) >= 3 else 0.0
    centroid_local = _polygon_centroid_2d(clipped) if len(clipped) >= 3 else np.array([1.0 / 3.0, 1.0 / 3.0])
    bary = np.array([1.0 - centroid_local[0] - centroid_local[1], centroid_local[0], centroid_local[1]], dtype=float)
    centroid = bary @ mesh.vertices[mesh.faces[face_ind]]
    return area, centroid


def _clip_local_triangle_by_cut(
    polygon: np.ndarray,
    bary_a: np.ndarray,
    bary_b: np.ndarray,
    side: int,
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output: list[np.ndarray] = []

    def value(point: np.ndarray) -> float:
        bary = np.array([1.0 - point[0] - point[1], point[0], point[1]], dtype=float)
        return side * _cut_side_value(bary_a, bary_b, bary)

    prev = polygon[-1]
    prev_value = value(prev)
    prev_inside = prev_value >= -1e-12
    for current in polygon:
        current_value = value(current)
        current_inside = current_value >= -1e-12
        if current_inside != prev_inside:
            denom = prev_value - current_value
            t = 0.0 if abs(denom) < 1e-15 else prev_value / denom
            output.append(prev + np.clip(t, 0.0, 1.0) * (current - prev))
        if current_inside:
            output.append(current)
        prev = current
        prev_value = current_value
        prev_inside = current_inside
    return np.asarray(output, dtype=float)


def _polygon_area_2d(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _polygon_centroid_2d(points: np.ndarray) -> np.ndarray:
    area = _polygon_area_2d(points)
    if abs(area) < 1e-15:
        return np.mean(points, axis=0)
    x = points[:, 0]
    y = points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
    return np.array([cx, cy], dtype=float)


def _cells_touching_edge(
    mesh: Mesh,
    face: int,
    edge: tuple[int, int],
    face_cells: list[tuple[int, ...]],
    cuts: dict[int, FaceCut],
) -> list[tuple[int, float, float]]:
    cells = face_cells[face]
    cut = cuts.get(face)
    if cut is None:
        return [(int(cells[0]), 0.0, 1.0)]

    intervals = _edge_intervals_for_cut(mesh.faces[face], edge, cut)
    touching: list[tuple[int, float, float]] = []
    for start, end in intervals:
        if end - start <= 1e-12:
            continue
        midpoint = 0.5 * (start + end)
        bary = _edge_barycentric_at(mesh.faces[face], edge, midpoint)
        sign = _cut_side_value(cut.bary_a, cut.bary_b, bary)
        cell = cut.cell_positive if sign >= 0 else cut.cell_negative
        touching.append((cell, start, end))
    return touching


def _edge_intervals_for_cut(
    face_vertices: np.ndarray, edge: tuple[int, int], cut: FaceCut
) -> list[tuple[float, float]]:
    split_values = []
    for bary in (cut.bary_a, cut.bary_b):
        t = _edge_t_for_bary(face_vertices, edge, bary)
        if t is not None:
            split_values.append(t)
    if not split_values:
        return [(0.0, 1.0)]
    t = float(np.clip(split_values[0], 0.0, 1.0))
    return [(0.0, t), (t, 1.0)]


def _edge_t_for_bary(face_vertices: np.ndarray, edge: tuple[int, int], bary: np.ndarray) -> float | None:
    low, high = edge
    local_low = _local_vertex_index(face_vertices, low)
    local_high = _local_vertex_index(face_vertices, high)
    if local_low is None or local_high is None:
        return None
    local_other = next(ind for ind in range(3) if ind not in (local_low, local_high))
    if abs(float(bary[local_other])) > 1e-8:
        return None
    return float(bary[local_high])


def _edge_barycentric_at(face_vertices: np.ndarray, edge: tuple[int, int], t: float) -> np.ndarray:
    low, high = edge
    local_low = _local_vertex_index(face_vertices, low)
    local_high = _local_vertex_index(face_vertices, high)
    if local_low is None or local_high is None:
        raise ValueError("Edge is not part of the requested face.")
    bary = np.zeros(3, dtype=float)
    bary[local_low] = 1.0 - t
    bary[local_high] = t
    return bary


def _local_vertex_index(face_vertices: np.ndarray, vertex: int) -> int | None:
    matches = np.where(np.asarray(face_vertices, dtype=np.int64) == int(vertex))[0]
    return int(matches[0]) if len(matches) else None


def _interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _cut_side_value(bary_a: np.ndarray, bary_b: np.ndarray, bary: np.ndarray) -> float:
    a = _bary_to_local_2d(bary_a)
    b = _bary_to_local_2d(bary_b)
    p = _bary_to_local_2d(bary)
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


def _bary_to_local_2d(bary: np.ndarray) -> np.ndarray:
    return np.array([float(bary[1]), float(bary[2])], dtype=float)


def geodesic_distance_field_with_barrier(
    mesh: Mesh,
    contour: ContourLine,
    seed_faces: np.ndarray,
    adjacency: FaceAdjacency | None = None,
) -> np.ndarray:
    """Approximate intrinsic distance over the virtual cut-face graph."""

    adjacency = build_face_adjacency(mesh) if adjacency is None else adjacency
    cut_graph = build_virtual_cut_graph(mesh, contour, adjacency)
    distances = np.full(len(cut_graph.neighbors), np.inf, dtype=float)
    heap: list[tuple[float, int]] = []

    for seed in np.asarray(seed_faces, dtype=np.int64):
        if 0 <= seed < len(cut_graph.face_cells):
            for cell in cut_graph.face_cells[int(seed)]:
                distances[cell] = 0.0
                heapq.heappush(heap, (0.0, int(cell)))

    while heap:
        dist, cell = heapq.heappop(heap)
        if dist > distances[cell]:
            continue
        for neighbor in cut_graph.neighbors[cell]:
            step = float(np.linalg.norm(cut_graph.cell_centroids[neighbor] - cut_graph.cell_centroids[cell]))
            next_dist = dist + step
            if next_dist < distances[neighbor]:
                distances[neighbor] = next_dist
                heapq.heappush(heap, (next_dist, neighbor))
    return distances
