from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ContourLine, Mesh


@dataclass(frozen=True)
class ParameterizedContourSmoothingResult:
    """Result and compact diagnostics for one parameterized contour fit."""

    contour: ContourLine
    applied: bool
    reason: str
    parameterization: str = "harmonic_local_patch"
    patch_face_count: int = 0
    output_segment_count: int = 0
    max_displacement_m: float = 0.0
    segment_length_cv_before: float = 0.0
    segment_length_cv_after: float = 0.0
    scale_mode: str = "legacy"
    mesh_scale_m: float = 0.0
    mesh_gain_requested: float = 1.0
    mesh_gain_applied: float = 1.0
    work_sample_count: int = 0
    output_target_segment_length_m: float = 0.0


@dataclass(frozen=True)
class _SurfaceChart:
    mesh: Mesh
    global_faces: np.ndarray
    global_vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    triangles_uv: np.ndarray
    neighbors_across_edge: tuple[dict[tuple[int, int], int], ...]
    parameterization: str


class MeshContourBSplineSmoother:
    """Smooth closed contours through a developed or harmonic mesh chart.

    Circular cylinders use an exact developed chart. Other surfaces use a
    harmonic chart built from the contour-crossed faces and neighboring face
    rings. Fitting and arc-length regularization happen in two dimensions. The
    fitted polyline is then traced through the chart triangles, which gives
    exact face and barycentric provenance for every returned 3D segment.
    """

    def __init__(self, mesh: Mesh, *, patch_rings: int = 4) -> None:
        self.mesh = mesh
        self.patch_rings = max(int(patch_rings), 1)
        self._face_neighbors = _face_neighbors(mesh.faces)

    def smooth(
        self,
        contour: ContourLine,
        *,
        smoothing_factor: float,
        target_segment_length_m: float = 0.0,
        max_displacement_m: float = 0.0,
        scale_mode: str = "legacy",
        mesh_gain: float = 0.1,
        work_samples_per_mesh_length: int = 6,
        output_samples_per_mesh_length: int = 8,
    ) -> ParameterizedContourSmoothingResult:
        """Fit one periodic cubic B-spline and map it back to the mesh."""

        points = np.asarray(contour.v, dtype=float)
        if not _has_surface_provenance(contour):
            return _unchanged(contour, "missing_surface_provenance")
        if len(points) < 4:
            return _unchanged(contour, "insufficient_contour_points")
        if not _is_closed(points):
            return _unchanged(contour, "contour_not_closed")
        factor = float(smoothing_factor)
        scale_mode = _normalise_scale_mode(scale_mode)
        requested_mesh_gain = float(mesh_gain)
        if scale_mode == "legacy" and (not np.isfinite(factor) or factor <= 0.0):
            return _unchanged(contour, "smoothing_disabled")
        if scale_mode == "mesh_coupled" and (
            not np.isfinite(requested_mesh_gain) or requested_mesh_gain <= 0.0 or requested_mesh_gain >= 1.0
        ):
            return _unchanged(contour, "mesh_coupled_smoothing_disabled")

        try:
            chart = self._build_chart(contour)
            contour_uv = _contour_uv(chart, contour)
            parameters = _closed_arclength_parameter(points)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return _unchanged(contour, f"parameterization_failed:{type(exc).__name__}")

        original_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        original_length = float(np.sum(original_lengths))
        unique_count = len(points) - 1
        mesh_scale = _contour_mesh_scale(self.mesh, contour, original_lengths) if scale_mode == "mesh_coupled" else 0.0
        if scale_mode == "mesh_coupled" and mesh_scale <= 0.0:
            return _unchanged(contour, "mesh_scale_unavailable")
        target_length = float(target_segment_length_m)
        if not np.isfinite(target_length) or target_length <= 0.0:
            target_length = original_length / max(4 * unique_count, 1)
        if scale_mode == "mesh_coupled":
            mesh_target_length = mesh_scale / max(int(output_samples_per_mesh_length), 2)
            target_length = min(target_length, mesh_target_length)
        sample_count = int(np.clip(np.ceil(original_length / max(target_length, 1.0e-14)), unique_count, 8192))

        result = self._fit_with_backtracking(
            contour,
            chart,
            contour_uv,
            parameters,
            factor,
            sample_count,
            max(float(max_displacement_m), 0.0),
            scale_mode=scale_mode,
            mesh_scale_m=mesh_scale,
            mesh_gain=requested_mesh_gain,
            work_samples_per_mesh_length=max(int(work_samples_per_mesh_length), 4),
        )
        if result is None:
            return _unchanged(contour, "fitted_curve_left_parameterized_patch")

        smoothed, displacement, applied_mesh_gain, work_sample_count = result
        return ParameterizedContourSmoothingResult(
            contour=smoothed,
            applied=True,
            reason="applied",
            parameterization=chart.parameterization,
            patch_face_count=int(len(np.unique(chart.global_faces))),
            output_segment_count=int(len(smoothed.v) - 1),
            max_displacement_m=float(displacement),
            segment_length_cv_before=_coefficient_of_variation(original_lengths),
            segment_length_cv_after=_coefficient_of_variation(np.linalg.norm(np.diff(smoothed.v, axis=0), axis=1)),
            scale_mode=scale_mode,
            mesh_scale_m=float(mesh_scale),
            mesh_gain_requested=float(requested_mesh_gain if scale_mode == "mesh_coupled" else 1.0),
            mesh_gain_applied=float(applied_mesh_gain),
            work_sample_count=int(work_sample_count),
            output_target_segment_length_m=float(target_length),
        )

    def _build_chart(self, contour: ContourLine) -> _SurfaceChart:
        seed_faces = np.unique(np.asarray(contour.segment_faces, dtype=np.int64))
        if len(seed_faces) == 0:
            raise ValueError("Contour has no crossed faces.")
        for ring_count in range(self.patch_rings, 0, -1):
            patch_faces = _expanded_face_patch(seed_faces, self._face_neighbors, ring_count)
            cylindrical_chart = _cylindrical_surface_chart(self.mesh, patch_faces)
            if cylindrical_chart is not None:
                return cylindrical_chart
            try:
                return _harmonic_surface_chart(self.mesh, patch_faces)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
        raise ValueError("No valid local harmonic chart could be constructed.")

    def _fit_with_backtracking(
        self,
        contour: ContourLine,
        chart: _SurfaceChart,
        contour_uv: np.ndarray,
        parameters: np.ndarray,
        smoothing_factor: float,
        sample_count: int,
        max_displacement_m: float,
        *,
        scale_mode: str,
        mesh_scale_m: float,
        mesh_gain: float,
        work_samples_per_mesh_length: int,
    ) -> tuple[ContourLine, float, float, int] | None:
        dense_count = int(np.clip(max(16 * sample_count, 512), 512, 32768))
        dense_u = np.linspace(0.0, 1.0, dense_count + 1)
        original_points = np.asarray(contour.v, dtype=float)
        original_length = float(np.sum(np.linalg.norm(np.diff(original_points, axis=0), axis=1)))

        if scale_mode == "mesh_coupled":
            fit_values = [1.0 - (1.0 - mesh_gain) * np.power(0.5, index) for index in range(8)]
        else:
            fit_values = [smoothing_factor * np.power(0.5, index) for index in range(8)]
            fit_values.append(0.0)
        for fit_value in fit_values:
            if scale_mode == "mesh_coupled":
                applied_mesh_gain = float(fit_value)
                fit_data = _mesh_coupled_periodic_bspline(
                    contour_uv,
                    parameters,
                    original_length_m=original_length,
                    mesh_scale_m=mesh_scale_m,
                    mesh_gain=applied_mesh_gain,
                    work_samples_per_mesh_length=work_samples_per_mesh_length,
                )
            else:
                applied_mesh_gain = 1.0
                fit_data = (*_periodic_bspline(contour_uv, parameters, float(fit_value)), 0)
            fit, _fitted_parameters, translation, work_sample_count = fit_data
            fit = (fit, _fitted_parameters, translation)
            candidate_dense = _evaluate_bspline(fit, dense_u)
            if _polyline_self_intersects(candidate_dense):
                continue
            mapped_dense = _map_uv_points(chart, candidate_dense)
            if mapped_dense is None:
                continue
            dense_lengths = np.linalg.norm(np.diff(mapped_dense, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(dense_lengths)))
            total_length = float(cumulative[-1])
            if total_length <= 1.0e-14:
                return None
            sample_arclength = np.linspace(0.0, total_length, sample_count + 1)
            sample_u = np.interp(sample_arclength, cumulative, dense_u)
            sample_uv = _evaluate_bspline(fit, sample_u)
            candidate = _trace_uv_polyline(
                chart,
                sample_uv,
                contour,
                target_segment_length_m=total_length / max(sample_count, 1),
            )
            if candidate is None:
                continue
            if _has_disconnected_face_passages(candidate.segment_faces):
                continue
            displacement = _max_point_to_polyline_distance(candidate.v[:-1], original_points)
            if max_displacement_m > 0.0 and displacement > max_displacement_m * (1.0 + 1.0e-6):
                continue
            return candidate, displacement, float(applied_mesh_gain), int(work_sample_count)
        return None


def smooth_contour_on_parameterized_mesh(
    mesh: Mesh,
    contour: ContourLine,
    *,
    smoothing_factor: float,
    target_segment_length_m: float = 0.0,
    max_displacement_m: float = 0.0,
    patch_rings: int = 4,
    scale_mode: str = "legacy",
    mesh_gain: float = 0.1,
    work_samples_per_mesh_length: int = 6,
    output_samples_per_mesh_length: int = 8,
) -> ParameterizedContourSmoothingResult:
    """Smooth one closed contour through a local 2D mesh parameterization."""

    return MeshContourBSplineSmoother(mesh, patch_rings=patch_rings).smooth(
        contour,
        smoothing_factor=smoothing_factor,
        target_segment_length_m=target_segment_length_m,
        max_displacement_m=max_displacement_m,
        scale_mode=scale_mode,
        mesh_gain=mesh_gain,
        work_samples_per_mesh_length=work_samples_per_mesh_length,
        output_samples_per_mesh_length=output_samples_per_mesh_length,
    )


def _unchanged(contour: ContourLine, reason: str) -> ParameterizedContourSmoothingResult:
    return ParameterizedContourSmoothingResult(contour=contour, applied=False, reason=reason)


def _has_surface_provenance(contour: ContourLine) -> bool:
    return (
        contour.segment_faces is not None
        and contour.segment_barycentric is not None
        and len(contour.segment_faces) == len(contour.v) - 1
        and np.asarray(contour.segment_barycentric).shape == (len(contour.v) - 1, 2, 3)
    )


def _is_closed(points: np.ndarray) -> bool:
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    return bool(np.linalg.norm(points[0] - points[-1]) <= 1.0e-10 * scale)


def _face_neighbors(faces: np.ndarray) -> tuple[tuple[int, ...], ...]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(int(start), int(end)), max(int(start), int(end)))
            edge_faces.setdefault(edge, []).append(int(face_index))
    neighbors = [set() for _ in range(len(faces))]
    for incident in edge_faces.values():
        if len(incident) == 2:
            first, second = incident
            neighbors[first].add(second)
            neighbors[second].add(first)
    return tuple(tuple(sorted(items)) for items in neighbors)


def _expanded_face_patch(
    seed_faces: np.ndarray,
    neighbors: tuple[tuple[int, ...], ...],
    ring_count: int,
) -> np.ndarray:
    selected = {int(face) for face in seed_faces if 0 <= int(face) < len(neighbors)}
    frontier = set(selected)
    for _ in range(max(int(ring_count), 0)):
        next_frontier = {neighbor for face in frontier for neighbor in neighbors[face]} - selected
        if not next_frontier:
            break
        selected.update(next_frontier)
        frontier = next_frontier
    return np.asarray(sorted(selected), dtype=np.int64)


def _harmonic_surface_chart(mesh: Mesh, global_faces: np.ndarray) -> _SurfaceChart:
    patch_global_faces = mesh.faces[np.asarray(global_faces, dtype=np.int64)]
    global_vertices = np.unique(patch_global_faces.reshape(-1))
    global_to_local = {int(vertex): index for index, vertex in enumerate(global_vertices)}
    faces = np.asarray(
        [[global_to_local[int(vertex)] for vertex in face] for face in patch_global_faces],
        dtype=np.int64,
    )
    boundary_loops = _oriented_boundary_loops(faces)
    if len(boundary_loops) not in {1, 2}:
        raise ValueError("Local patch must have one or two boundary components.")

    vertices = mesh.vertices[global_vertices]
    loop_perimeters = np.asarray([_loop_perimeter(vertices, loop) for loop in boundary_loops])
    if np.any(loop_perimeters <= 1.0e-14):
        raise ValueError("Degenerate parameterization boundary.")
    loop_order = np.argsort(loop_perimeters)[::-1]
    loops = [boundary_loops[int(index)] for index in loop_order]
    perimeters = loop_perimeters[loop_order]

    candidates: list[np.ndarray] = []
    for outer_direction in (1.0, -1.0):
        direction_sets = (
            [(outer_direction,)]
            if len(loops) == 1
            else [
                (outer_direction, outer_direction),
                (outer_direction, -outer_direction),
            ]
        )
        for directions in direction_sets:
            phase_candidates: tuple[float | None, ...] = (None,)
            if len(loops) == 2:
                phase_candidates = tuple(
                    _inner_boundary_phase_candidates(
                        vertices[loops[0]],
                        vertices[loops[1]],
                        directions[0],
                        directions[1],
                    )
                )
            for inner_phase in phase_candidates:
                boundary_uv = _boundary_circle_coordinates(
                    vertices,
                    loops,
                    perimeters,
                    directions,
                    inner_phase=inner_phase,
                )
                uv = _solve_uniform_harmonic_map(faces, boundary_uv, len(vertices))
                if _chart_has_consistent_faces(uv, faces):
                    candidates.append(uv)
                    break
            if candidates:
                break
        if candidates:
            break
    if not candidates:
        raise ValueError("Harmonic chart contains folded triangles.")
    uv = max(candidates, key=lambda candidate: _minimum_scaled_triangle_area(candidate, faces))
    return _SurfaceChart(
        mesh=mesh,
        global_faces=np.asarray(global_faces, dtype=np.int64),
        global_vertices=global_vertices,
        faces=faces,
        uv=uv,
        triangles_uv=uv[faces],
        neighbors_across_edge=_chart_face_neighbors(faces),
        parameterization="harmonic_local_patch",
    )


def _cylindrical_surface_chart(mesh: Mesh, global_faces: np.ndarray) -> _SurfaceChart | None:
    """Return an exact unwrapped chart for a triangulated circular cylinder.

    The chart uses axial distance and radius-scaled azimuth, so a smooth curve
    in 2D remains intrinsically smooth after mapping back.  Five periodic tiles
    allow both contractible gradient loops and contours that wind once around
    the cylinder to cross the angular seam without a discontinuity.
    """

    frame = _circular_cylinder_frame(mesh)
    if frame is None:
        return None
    center, axis, first_radial, second_radial, radius = frame
    relative = np.asarray(mesh.vertices, dtype=float) - center
    axial = relative @ axis
    radial_first = relative @ first_radial
    radial_second = relative @ second_radial
    azimuth = np.arctan2(radial_second, radial_first)

    local_vertex_lookup: dict[tuple[int, int], int] = {}
    local_global_vertices: list[int] = []
    local_uv: list[np.ndarray] = []
    local_faces: list[list[int]] = []
    tiled_global_faces: list[int] = []
    for tile in range(-1, 2):
        for global_face_raw in np.asarray(global_faces, dtype=np.int64):
            global_face = int(global_face_raw)
            face_vertices = np.asarray(mesh.faces[global_face], dtype=np.int64)
            face_azimuth = azimuth[face_vertices]
            reference = float(face_azimuth[0])
            face_shifts = np.rint((reference - face_azimuth) / (2.0 * np.pi)).astype(np.int64)
            chart_face: list[int] = []
            for global_vertex_raw, base_shift_raw in zip(face_vertices, face_shifts):
                global_vertex = int(global_vertex_raw)
                shift = int(base_shift_raw) + tile
                key = (global_vertex, shift)
                local_vertex = local_vertex_lookup.get(key)
                if local_vertex is None:
                    local_vertex = len(local_global_vertices)
                    local_vertex_lookup[key] = local_vertex
                    local_global_vertices.append(global_vertex)
                    local_uv.append(
                        np.array(
                            [
                                radius * (float(azimuth[global_vertex]) + 2.0 * np.pi * shift),
                                float(axial[global_vertex]),
                            ],
                            dtype=float,
                        )
                    )
                chart_face.append(local_vertex)
            local_faces.append(chart_face)
            tiled_global_faces.append(global_face)
    faces = np.asarray(local_faces, dtype=np.int64)
    uv = np.asarray(local_uv, dtype=float)
    if len(faces) == 0 or not np.all(np.isfinite(uv)):
        return None
    return _SurfaceChart(
        mesh=mesh,
        global_faces=np.asarray(tiled_global_faces, dtype=np.int64),
        global_vertices=np.asarray(local_global_vertices, dtype=np.int64),
        faces=faces,
        uv=uv,
        triangles_uv=uv[faces],
        neighbors_across_edge=_chart_face_neighbors(faces),
        parameterization="developed_circular_cylinder",
    )


def _circular_cylinder_frame(
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Detect an open circular cylinder and return its intrinsic frame."""

    try:
        boundaries = _oriented_boundary_loops(np.asarray(mesh.faces, dtype=np.int64))
    except ValueError:
        return None
    if len(boundaries) != 2:
        return None
    vertices = np.asarray(mesh.vertices, dtype=float)
    boundary_centers = [np.mean(vertices[boundary], axis=0) for boundary in boundaries]
    axis_vector = boundary_centers[1] - boundary_centers[0]
    axis_length = float(np.linalg.norm(axis_vector))
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0e-12)
    if axis_length <= 1.0e-8 * scale:
        return None
    axis = axis_vector / axis_length
    center = 0.5 * (boundary_centers[0] + boundary_centers[1])
    relative = vertices - center
    radial = relative - (relative @ axis)[:, None] * axis
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.mean(radii))
    if radius <= 1.0e-10 * scale:
        return None
    # This exact chart is deliberately restricted to circular cylinders.  A
    # loose fit here would silently impose the wrong metric on arbitrary meshes.
    if float(np.max(np.abs(radii - radius))) > max(1.0e-7 * radius, 1.0e-12):
        return None
    boundary_axial_spread = max(float(np.ptp((vertices[boundary] - center) @ axis)) for boundary in boundaries)
    if boundary_axial_spread > max(1.0e-7 * axis_length, 1.0e-12):
        return None
    radial_norms = np.linalg.norm(radial, axis=1)
    reference_index = int(np.argmax(radial_norms))
    first_radial = radial[reference_index] / radial_norms[reference_index]
    second_radial = np.cross(axis, first_radial)
    second_norm = float(np.linalg.norm(second_radial))
    if second_norm <= 1.0e-12:
        return None
    second_radial /= second_norm
    return center, axis, first_radial, second_radial, radius


def _oriented_boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(int(start), int(end)), max(int(start), int(end)))
            edge_occurrences.setdefault(key, []).append((int(start), int(end)))
    directed = [items[0] for items in edge_occurrences.values() if len(items) == 1]
    if not directed:
        return []
    outgoing: dict[int, int] = {}
    incoming_count: dict[int, int] = {}
    for start, end in directed:
        if start in outgoing and outgoing[start] != end:
            raise ValueError("Non-manifold patch boundary.")
        outgoing[start] = end
        incoming_count[end] = incoming_count.get(end, 0) + 1
    if any(count != 1 for count in incoming_count.values()) or set(outgoing) != set(incoming_count):
        raise ValueError("Patch boundary is not a collection of closed loops.")

    loops: list[np.ndarray] = []
    unused = set(outgoing)
    while unused:
        start = min(unused)
        loop = [start]
        current = start
        for _ in range(len(outgoing) + 1):
            current = outgoing[current]
            if current == start:
                break
            if current not in unused:
                raise ValueError("Patch boundary traversal repeated a vertex.")
            loop.append(current)
        else:
            raise ValueError("Patch boundary did not close.")
        unused.difference_update(loop)
        if len(loop) < 3:
            raise ValueError("Patch boundary has fewer than three vertices.")
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _loop_perimeter(vertices: np.ndarray, loop: np.ndarray) -> float:
    points = vertices[loop]
    return float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))


def _boundary_circle_coordinates(
    vertices: np.ndarray,
    loops: list[np.ndarray],
    perimeters: np.ndarray,
    directions: tuple[float, ...],
    *,
    inner_phase: float | None = None,
) -> dict[int, np.ndarray]:
    coordinates: dict[int, np.ndarray] = {}
    radii = [1.0]
    if len(loops) == 2:
        radii.append(float(np.clip(perimeters[1] / perimeters[0], 0.2, 0.82)))
    phases = [0.0] * len(loops)
    if len(loops) == 2:
        outer_points = vertices[loops[0]]
        inner_points = vertices[loops[1]]
        phases[1] = (
            _closest_inner_boundary_phase(outer_points, inner_points, directions[0], directions[1])
            if inner_phase is None
            else float(inner_phase)
        )
    for loop_index, (loop, radius, direction, phase) in enumerate(zip(loops, radii, directions, phases)):
        del loop_index
        fractions = _loop_fractions(vertices[loop])
        angles = phase + direction * 2.0 * np.pi * fractions
        for vertex, angle in zip(loop, angles):
            coordinates[int(vertex)] = radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
    return coordinates


def _closest_inner_boundary_phase(
    outer_points: np.ndarray,
    inner_points: np.ndarray,
    outer_direction: float,
    inner_direction: float,
) -> float:
    outer_fractions = _loop_fractions(outer_points)
    inner_fractions = _loop_fractions(inner_points)
    distances = np.linalg.norm(outer_points[:, None, :] - inner_points[None, :, :], axis=2)
    outer_index, inner_index = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return float(
        2.0 * np.pi * (outer_direction * outer_fractions[outer_index] - inner_direction * inner_fractions[inner_index])
    )


def _inner_boundary_phase_candidates(
    outer_points: np.ndarray,
    inner_points: np.ndarray,
    outer_direction: float,
    inner_direction: float,
) -> list[float]:
    base = _closest_inner_boundary_phase(
        outer_points,
        inner_points,
        outer_direction,
        inner_direction,
    )
    angular_step = 2.0 * np.pi / max(len(outer_points), len(inner_points), 1)
    offsets = (0, 1, -1, 2, -2, 3, -3, 4, -4)
    return [float(base + offset * angular_step) for offset in offsets]


def _loop_fractions(points: np.ndarray) -> np.ndarray:
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    total = float(np.sum(edge_lengths))
    return np.concatenate(([0.0], np.cumsum(edge_lengths[:-1]))) / max(total, 1.0e-30)


def _solve_uniform_harmonic_map(
    faces: np.ndarray,
    boundary_uv: dict[int, np.ndarray],
    vertex_count: int,
) -> np.ndarray:
    neighbors = [set() for _ in range(vertex_count)]
    for face in faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            neighbors[int(first)].add(int(second))
            neighbors[int(second)].add(int(first))
    boundary = np.asarray(sorted(boundary_uv), dtype=np.int64)
    interior = np.asarray([index for index in range(vertex_count) if index not in boundary_uv], dtype=np.int64)
    uv = np.zeros((vertex_count, 2), dtype=float)
    uv[boundary] = np.asarray([boundary_uv[int(index)] for index in boundary])
    if len(interior) == 0:
        return uv

    interior_lookup = {int(vertex): row for row, vertex in enumerate(interior)}
    matrix = np.zeros((len(interior), len(interior)), dtype=float)
    right_hand_side = np.zeros((len(interior), 2), dtype=float)
    for row, vertex in enumerate(interior):
        degree = len(neighbors[int(vertex)])
        if degree == 0:
            raise ValueError("Isolated parameterization vertex.")
        matrix[row, row] = float(degree)
        for neighbor in neighbors[int(vertex)]:
            neighbor_row = interior_lookup.get(int(neighbor))
            if neighbor_row is None:
                right_hand_side[row] += uv[int(neighbor)]
            else:
                matrix[row, neighbor_row] -= 1.0
    uv[interior] = np.linalg.solve(matrix, right_hand_side)
    return uv


def _chart_has_consistent_faces(uv: np.ndarray, faces: np.ndarray) -> bool:
    areas = _signed_triangle_areas(uv[faces])
    tolerance = 1.0e-12 * max(float(np.max(np.abs(areas))), 1.0)
    valid = np.abs(areas) > tolerance
    if not np.all(valid):
        return False
    return bool(np.all(areas > 0.0) or np.all(areas < 0.0))


def _minimum_scaled_triangle_area(uv: np.ndarray, faces: np.ndarray) -> float:
    areas = np.abs(_signed_triangle_areas(uv[faces]))
    return float(np.min(areas) / max(np.mean(areas), 1.0e-30))


def _signed_triangle_areas(triangles: np.ndarray) -> np.ndarray:
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    return 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def _chart_face_neighbors(faces: np.ndarray) -> tuple[dict[tuple[int, int], int], ...]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(int(first), int(second)), max(int(first), int(second)))
            edge_faces.setdefault(edge, []).append(int(face_index))
    neighbors: list[dict[tuple[int, int], int]] = [dict() for _ in range(len(faces))]
    for edge, incident in edge_faces.items():
        if len(incident) == 2:
            first, second = incident
            neighbors[first][edge] = second
            neighbors[second][edge] = first
    return tuple(neighbors)


def _contour_uv(chart: _SurfaceChart, contour: ContourLine) -> np.ndarray:
    face_lookup: dict[int, list[int]] = {}
    for local_face, global_face in enumerate(chart.global_faces):
        face_lookup.setdefault(int(global_face), []).append(int(local_face))
    segment_faces = np.asarray(contour.segment_faces, dtype=np.int64)
    barycentric = np.asarray(contour.segment_barycentric, dtype=float)
    points: list[np.ndarray] = []
    for segment_index, global_face in enumerate(segment_faces):
        candidate_faces = face_lookup.get(int(global_face), [])
        if not candidate_faces:
            raise ValueError("Contour face is outside the chart.")
        candidates = [
            (
                barycentric[segment_index, 0] @ chart.uv[chart.faces[local_face]],
                barycentric[segment_index, 1] @ chart.uv[chart.faces[local_face]],
            )
            for local_face in candidate_faces
        ]
        if not points:
            start_uv, end_uv = min(candidates, key=lambda candidate: abs(float(candidate[0][0])))
            points.append(start_uv)
        else:
            start_uv, end_uv = min(
                candidates,
                key=lambda candidate: float(np.linalg.norm(candidate[0] - points[-1])),
            )
            tolerance = 1.0e-8 * max(float(np.linalg.norm(np.ptp(chart.uv, axis=0))), 1.0)
            if float(np.linalg.norm(start_uv - points[-1])) > tolerance:
                raise ValueError("Contour is discontinuous in the selected surface chart.")
        points.append(end_uv)
    result = np.asarray(points, dtype=float)
    if chart.parameterization == "harmonic_local_patch":
        result[-1] = result[0]
    return result


def _closed_arclength_parameter(points: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 1.0e-14):
        raise ValueError("Contour contains zero-length segments.")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1.0e-14:
        raise ValueError("Contour has zero length.")
    return cumulative / cumulative[-1]


def _normalise_scale_mode(value: object) -> str:
    key = str(value).strip().lower().replace("-", "_")
    return "mesh_coupled" if key in {"mesh", "mesh_coupled", "mesh_scaled"} else "legacy"


def _contour_mesh_scale(mesh: Mesh, contour: ContourLine, segment_lengths: np.ndarray) -> float:
    """Return a robust intrinsic element length along a contour.

    ``sqrt(2 A)`` is the geometric-mean leg length for a right triangle.  It
    captures both directions of an anisotropic structured mesh without being
    dominated by its diagonal.  Segment-length weighting prevents grazing
    intersections and vertex-adjacent slivers from setting the smoothing
    wavelength.
    """

    if contour.segment_faces is None:
        return 0.0
    face_indices = np.asarray(contour.segment_faces, dtype=np.int64).reshape(-1)
    lengths = np.asarray(segment_lengths, dtype=float).reshape(-1)
    count = min(len(face_indices), len(lengths))
    if count == 0:
        return 0.0
    face_indices = face_indices[:count]
    lengths = lengths[:count]
    valid_faces = (face_indices >= 0) & (face_indices < len(mesh.faces))
    if not np.any(valid_faces):
        return 0.0
    triangles = np.asarray(mesh.vertices, dtype=float)[
        np.asarray(mesh.faces, dtype=np.int64)[face_indices[valid_faces]]
    ]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    scales = np.sqrt(np.maximum(double_areas, 0.0))
    weights = lengths[valid_faces]
    valid = np.isfinite(scales) & (scales > 1.0e-14) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return 0.0
    return _weighted_quantile(scales[valid], weights[valid], 0.75)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    total = float(cumulative[-1])
    if total <= 0.0:
        return 0.0
    index = int(np.searchsorted(cumulative, float(np.clip(quantile, 0.0, 1.0)) * total, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _mesh_coupled_periodic_bspline(
    points_uv: np.ndarray,
    parameters: np.ndarray,
    *,
    original_length_m: float,
    mesh_scale_m: float,
    mesh_gain: float,
    work_samples_per_mesh_length: int,
) -> tuple[object, np.ndarray, np.ndarray, int]:
    """Fit a periodic spline after calibrated mesh-wavelength attenuation."""

    try:
        from scipy.interpolate import splprep  # type: ignore
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("Parameterized contour smoothing requires scipy.") from exc

    source_unique_count = len(points_uv) - 1
    degree = min(3, source_unique_count - 1)
    if degree < 2:
        raise ValueError("A periodic B-spline requires at least three unique points.")
    translation = np.asarray(points_uv[-1] - points_uv[0], dtype=float)
    periodic_points = np.asarray(points_uv, dtype=float) - parameters[:, None] * translation
    periodic_points[-1] = periodic_points[0]

    work_step = mesh_scale_m / max(int(work_samples_per_mesh_length), 4)
    work_count = int(np.clip(np.ceil(original_length_m / max(work_step, 1.0e-14)), 32, 8192))
    work_parameters = np.linspace(0.0, 1.0, work_count, endpoint=False)
    work_points = _periodic_linear_interpolation(parameters, periodic_points, work_parameters)

    actual_step = original_length_m / max(work_count, 1)
    mesh_frequency_eigenvalue = 16.0 * np.sin(np.pi * actual_step / mesh_scale_m) ** 4
    gain = float(np.clip(mesh_gain, 1.0e-6, 1.0))
    penalty = (1.0 / gain - 1.0) / max(float(mesh_frequency_eigenvalue), 1.0e-14)
    frequencies = np.arange(work_count, dtype=float)
    cyclic_eigenvalues = 16.0 * np.sin(np.pi * frequencies / work_count) ** 4
    transfer = 1.0 / (1.0 + penalty * cyclic_eigenvalues)
    filtered = np.real(np.fft.ifft(np.fft.fft(work_points, axis=0) * transfer[:, None], axis=0))

    fitted_parameters = np.linspace(0.0, 1.0, work_count + 1)
    fitted_points = np.vstack((filtered, filtered[:1]))
    fit, spline_parameters = splprep(
        [fitted_points[:, 0], fitted_points[:, 1]],
        u=fitted_parameters,
        k=degree,
        s=0.0,
        per=True,
    )
    return fit, np.asarray(spline_parameters, dtype=float), translation, work_count


def _periodic_bspline(
    points_uv: np.ndarray,
    parameters: np.ndarray,
    factor: float,
) -> tuple[object, np.ndarray, np.ndarray]:
    try:
        from scipy.interpolate import splprep  # type: ignore
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("Parameterized contour smoothing requires scipy.") from exc
    source_unique_count = len(points_uv) - 1
    degree = min(3, source_unique_count - 1)
    if degree < 2:
        raise ValueError("A periodic B-spline requires at least three unique points.")
    translation = np.asarray(points_uv[-1] - points_uv[0], dtype=float)
    periodic_points = np.asarray(points_uv, dtype=float) - parameters[:, None] * translation
    periodic_points[-1] = periodic_points[0]
    segment_lengths = np.linalg.norm(np.diff(points_uv, axis=0), axis=1)
    median_segment = float(np.median(segment_lengths[segment_lengths > 1.0e-14]))
    smoothing_bound = source_unique_count * np.square(max(float(factor), 0.0) * median_segment)
    regularized_count = max(4 * source_unique_count, 32)
    regularized_parameters = np.linspace(0.0, 1.0, regularized_count + 1)
    regularized_points = _periodic_linear_interpolation(parameters, periodic_points, regularized_parameters)
    regularized_points[-1] = regularized_points[0]
    fit, fitted_parameters = splprep(
        [regularized_points[:, 0], regularized_points[:, 1]],
        u=regularized_parameters,
        k=degree,
        s=float(smoothing_bound),
        per=True,
    )
    return fit, np.asarray(fitted_parameters, dtype=float), translation


def _evaluate_bspline(fit: tuple[object, np.ndarray, np.ndarray], parameters: np.ndarray) -> np.ndarray:
    try:
        from scipy.interpolate import splev  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Parameterized contour smoothing requires scipy.") from exc
    parameters = np.asarray(parameters, dtype=float)
    values = np.column_stack(splev(parameters, fit[0]))
    return values + parameters[:, None] * fit[2]


def _periodic_linear_interpolation(
    parameters: np.ndarray,
    points: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    return np.column_stack(
        (
            np.interp(query, parameters, points[:, 0]),
            np.interp(query, parameters, points[:, 1]),
        )
    )


def _map_uv_points(chart: _SurfaceChart, points_uv: np.ndarray) -> np.ndarray | None:
    mapped = []
    hint: int | None = None
    for point in points_uv:
        located = _locate_uv_point(chart, point, hint)
        if located is None:
            return None
        hint, barycentric = located
        global_face = int(chart.global_faces[hint])
        mapped.append(barycentric @ chart.mesh.vertices[chart.mesh.faces[global_face]])
    result = np.asarray(mapped, dtype=float)
    if len(result):
        result[-1] = result[0]
    return result


def _locate_uv_point(
    chart: _SurfaceChart,
    point: np.ndarray,
    hint: int | None = None,
) -> tuple[int, np.ndarray] | None:
    if hint is not None:
        barycentric = _barycentric_2d(chart.triangles_uv[int(hint)], point)
        if np.min(barycentric) >= -1.0e-9:
            return int(hint), _normalise_barycentric(barycentric)
        for neighbor in set(chart.neighbors_across_edge[int(hint)].values()):
            barycentric = _barycentric_2d(chart.triangles_uv[int(neighbor)], point)
            if np.min(barycentric) >= -1.0e-9:
                return int(neighbor), _normalise_barycentric(barycentric)
    best_face = -1
    best_barycentric: np.ndarray | None = None
    best_violation = float("inf")
    for face_index, triangle in enumerate(chart.triangles_uv):
        barycentric = _barycentric_2d(triangle, point)
        violation = max(0.0, -float(np.min(barycentric)))
        if violation < best_violation:
            best_face = int(face_index)
            best_barycentric = barycentric
            best_violation = violation
            if violation <= 1.0e-10:
                break
    if best_barycentric is None or best_violation > 1.0e-7:
        return None
    return best_face, _normalise_barycentric(best_barycentric)


def _trace_uv_polyline(
    chart: _SurfaceChart,
    points_uv: np.ndarray,
    source: ContourLine,
    *,
    target_segment_length_m: float,
) -> ContourLine | None:
    if len(points_uv) < 4:
        return None
    located = _locate_uv_point(chart, points_uv[0])
    if located is None:
        return None
    face_hint, _ = located
    traced: list[tuple[int, np.ndarray, np.ndarray]] = []
    for start, end in zip(points_uv[:-1], points_uv[1:]):
        pieces = _trace_uv_segment(chart, start, end, face_hint)
        if not pieces:
            return None
        traced.extend(pieces)
        face_hint = pieces[-1][0]
    if not traced:
        return None
    traced = _regularize_traced_face_runs(chart, traced, target_segment_length_m)

    vertices: list[np.ndarray] = []
    segment_faces: list[int] = []
    segment_barycentric: list[np.ndarray] = []
    surface_faces: list[int] = []
    surface_barycentric: list[np.ndarray] = []
    for piece_index, (local_face, start_bary, end_bary) in enumerate(traced):
        global_face = int(chart.global_faces[local_face])
        triangle = chart.mesh.vertices[chart.mesh.faces[global_face]]
        start_point = start_bary @ triangle
        end_point = end_bary @ triangle
        if float(np.linalg.norm(end_point - start_point)) <= 1.0e-13:
            continue
        if not vertices:
            vertices.append(start_point)
            surface_faces.append(global_face)
            surface_barycentric.append(start_bary)
        elif float(np.linalg.norm(vertices[-1] - start_point)) > 1.0e-9:
            return None
        vertices.append(end_point)
        surface_faces.append(global_face)
        surface_barycentric.append(end_bary)
        segment_faces.append(global_face)
        segment_barycentric.append(np.vstack((start_bary, end_bary)))
        del piece_index
    if len(vertices) < 4:
        return None
    vertices[-1] = vertices[0].copy()
    surface_faces[-1] = surface_faces[0]
    surface_barycentric[-1] = surface_barycentric[0].copy()
    return ContourLine(
        v=np.asarray(vertices, dtype=float),
        potential=source.potential,
        current_orientation=source.current_orientation,
        surface_faces=np.asarray(surface_faces, dtype=np.int64),
        surface_barycentric=np.asarray(surface_barycentric, dtype=float),
        surface_edge_vertices=None,
        surface_edge_t=None,
        segment_faces=np.asarray(segment_faces, dtype=np.int64),
        segment_barycentric=np.asarray(segment_barycentric, dtype=float),
        turn_multiplicity=source.turn_multiplicity,
    )


def _regularize_traced_face_runs(
    chart: _SurfaceChart,
    pieces: list[tuple[int, np.ndarray, np.ndarray]],
    target_segment_length_m: float,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Redistribute samples within each face passage at near-uniform spacing."""

    if not pieces or target_segment_length_m <= 0.0:
        return pieces
    if len(pieces) > 1 and pieces[0][0] == pieces[-1][0]:
        transition = next(
            (index for index in range(1, len(pieces)) if pieces[index][0] != pieces[index - 1][0]),
            None,
        )
        if transition is not None:
            pieces = [*pieces[transition:], *pieces[:transition]]

    regularized: list[tuple[int, np.ndarray, np.ndarray]] = []
    start = 0
    while start < len(pieces):
        face = pieces[start][0]
        end = start + 1
        while end < len(pieces) and pieces[end][0] == face:
            end += 1
        run = pieces[start:end]
        global_face = int(chart.global_faces[face])
        triangle = chart.mesh.vertices[chart.mesh.faces[global_face]]
        bary_points = np.asarray([run[0][1], *[piece[2] for piece in run]], dtype=float)
        points = bary_points @ triangle
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        run_length = float(cumulative[-1])
        if run_length <= 1.0e-13:
            start = end
            continue
        split_count = max(1, int(round(run_length / target_segment_length_m)))
        sample_distances = np.linspace(0.0, run_length, split_count + 1)
        sample_barycentric = np.column_stack(
            [np.interp(sample_distances, cumulative, bary_points[:, component]) for component in range(3)]
        )
        sample_barycentric = np.asarray(
            [_normalise_barycentric(barycentric) for barycentric in sample_barycentric],
            dtype=float,
        )
        regularized.extend(
            (face, sample_barycentric[index], sample_barycentric[index + 1]) for index in range(split_count)
        )
        start = end
    return regularized


def _trace_uv_segment(
    chart: _SurfaceChart,
    start: np.ndarray,
    end: np.ndarray,
    start_face: int,
) -> list[tuple[int, np.ndarray, np.ndarray]] | None:
    if float(np.linalg.norm(end - start)) <= 1.0e-14:
        return []
    current_point = np.asarray(start, dtype=float)
    current_face = int(start_face)
    if np.min(_barycentric_2d(chart.triangles_uv[current_face], current_point)) < -1.0e-8:
        located = _locate_uv_point(chart, current_point)
        if located is None:
            return None
        current_face = located[0]
    pieces: list[tuple[int, np.ndarray, np.ndarray]] = []

    for _ in range(len(chart.faces) + 8):
        triangle = chart.triangles_uv[current_face]
        start_bary = _normalise_barycentric(_barycentric_2d(triangle, current_point))
        end_bary_unclipped = _barycentric_2d(triangle, end)
        if np.min(end_bary_unclipped) >= -1.0e-9:
            pieces.append((current_face, start_bary, _normalise_barycentric(end_bary_unclipped)))
            return pieces

        departing_components = [
            component
            for component in range(3)
            if start_bary[component] <= 1.0e-9 and end_bary_unclipped[component] < -1.0e-10
        ]
        if departing_components:
            probe = current_point + 1.0e-8 * (end - current_point)
            next_face = None
            for component in departing_components:
                face = chart.faces[current_face]
                edge_vertices = [int(face[index]) for index in range(3) if index != component]
                edge = (min(edge_vertices), max(edge_vertices))
                candidate = chart.neighbors_across_edge[current_face].get(edge)
                if candidate is None:
                    continue
                if np.min(_barycentric_2d(chart.triangles_uv[candidate], probe)) >= -1.0e-8:
                    next_face = candidate
                    break
            if next_face is None:
                located = _locate_uv_point(chart, probe)
                if located is None or located[0] == current_face:
                    return None
                next_face = located[0]
            current_face = int(next_face)
            continue

        crossing_parameters = []
        for component in range(3):
            denominator = start_bary[component] - end_bary_unclipped[component]
            if end_bary_unclipped[component] < -1.0e-10 and denominator > 1.0e-14:
                crossing = start_bary[component] / denominator
                if crossing > 1.0e-10:
                    crossing_parameters.append((float(crossing), component))
        if not crossing_parameters:
            return None
        crossing, crossed_component = min(crossing_parameters)
        crossing = float(np.clip(crossing, 0.0, 1.0))
        crossing_point = current_point + crossing * (end - current_point)
        crossing_bary = _normalise_barycentric(_barycentric_2d(triangle, crossing_point))
        pieces.append((current_face, start_bary, crossing_bary))

        face = chart.faces[current_face]
        edge_vertices = [int(face[index]) for index in range(3) if index != crossed_component]
        edge = (min(edge_vertices), max(edge_vertices))
        next_face = chart.neighbors_across_edge[current_face].get(edge)
        if next_face is None:
            probe = crossing_point + 1.0e-8 * (end - crossing_point)
            located = _locate_uv_point(chart, probe)
            if located is None or located[0] == current_face:
                return None
            next_face = located[0]
        current_point = crossing_point
        current_face = int(next_face)
    return None


def _barycentric_2d(triangle: np.ndarray, point: np.ndarray) -> np.ndarray:
    matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1.0e-15:
        return np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    beta, gamma = np.linalg.solve(matrix, np.asarray(point, dtype=float) - triangle[0])
    return np.array([1.0 - beta - gamma, beta, gamma], dtype=float)


def _normalise_barycentric(barycentric: np.ndarray) -> np.ndarray:
    barycentric = np.clip(np.asarray(barycentric, dtype=float), 0.0, 1.0)
    total = float(np.sum(barycentric))
    return barycentric / total if total > 0.0 else np.array([1.0, 0.0, 0.0], dtype=float)


def _max_point_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline[:-1]
    directions = np.diff(polyline, axis=0)
    lengths_squared = np.einsum("ij,ij->i", directions, directions)
    maximum = 0.0
    for point in np.asarray(points, dtype=float):
        relative = point - starts
        parameters = np.divide(
            np.einsum("ij,ij->i", relative, directions),
            lengths_squared,
            out=np.zeros_like(lengths_squared),
            where=lengths_squared > 1.0e-30,
        )
        parameters = np.clip(parameters, 0.0, 1.0)
        closest = starts + parameters[:, None] * directions
        maximum = max(maximum, float(np.min(np.linalg.norm(closest - point, axis=1))))
    return maximum


def _polyline_self_intersects(points: np.ndarray) -> bool:
    points = np.asarray(points, dtype=float)
    if len(points) > 513:
        indices = np.linspace(0, len(points) - 1, 513, dtype=np.int64)
        points = points[indices]
    segment_count = len(points) - 1
    closed = bool(
        np.linalg.norm(points[-1] - points[0]) <= 1.0e-10 * max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    )
    for first in range(segment_count):
        for second in range(first + 2, segment_count):
            if closed and first == 0 and second == segment_count - 1:
                continue
            if _segments_intersect_2d(points[first], points[first + 1], points[second], points[second + 1]):
                return True
    return False


def _segments_intersect_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def cross(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    direction_ab = b - a
    direction_cd = d - c
    denominator = cross(direction_ab, direction_cd)
    if abs(denominator) <= 1.0e-14:
        return False
    relative = c - a
    first_parameter = cross(relative, direction_cd) / denominator
    second_parameter = cross(relative, direction_ab) / denominator
    tolerance = 1.0e-10
    return bool(tolerance < first_parameter < 1.0 - tolerance and tolerance < second_parameter < 1.0 - tolerance)


def _coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values)) if len(values) else 0.0
    return 0.0 if mean <= 1.0e-30 else float(np.std(values) / mean)


def _has_disconnected_face_passages(segment_faces: np.ndarray | None) -> bool:
    if segment_faces is None:
        return True
    faces = [int(face) for face in np.asarray(segment_faces, dtype=np.int64)]
    if not faces:
        return True
    runs = [faces[0]]
    runs.extend(face for previous, face in zip(faces[:-1], faces[1:]) if face != previous)
    if len(runs) > 1 and runs[0] == runs[-1]:
        runs.pop()
    return len(runs) != len(set(runs))
