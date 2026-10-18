from __future__ import annotations

import numpy as np

from .config import CoilGenConfig
from .models import BasisElement, CoilPart, TargetField
from .parallel import (
    balanced_ranges,
    chunk_ranges,
    memory_limited_chunk_size,
    parallel_map,
    resolve_worker_count,
    should_parallelize,
)
from .progress import ProgressReporter
from .quadrature import gauss_legendre_integration_points_triangle

BIOT_SAVART_COEFF = 1e-7


def calculate_sensitivity_matrix(
    parts: list[CoilPart], target_field: TargetField, config: CoilGenConfig
) -> list[CoilPart]:
    u_coord, v_coord, gauss_weight = gauss_legendre_integration_points_triangle(config.gauss_order)
    target_points = target_field.coords
    worker_count = resolve_worker_count(config)
    reporter = ProgressReporter.from_config(config, label="Sensitivity")
    method = str(getattr(config, "sensitivity_calculation_method", "vectorized")).lower()
    if method == "auto":
        method = "vectorized"
    if method not in {"vectorized", "scalar"}:
        raise ValueError("sensitivity_calculation_method must be 'vectorized', 'scalar', or 'auto'.")
    # Plan the existing partitioning before reporting, so several supports share
    # one completion count rather than each restarting at zero percent.
    plans = []
    for part in parts:
        node_count = len(part.basis_elements)
        work_size = node_count * len(target_points) * max(len(gauss_weight), 1)
        ranges = (
            balanced_ranges(
                node_count,
                worker_count,
                chunks_per_worker=int(config.sensitivity_chunks_per_worker),
            )
            if should_parallelize(work_size, config, worker_count=worker_count)
            else None
        )
        plans.append(ranges)
    task = reporter.task(
        "support work chunks",
        sum(len(ranges) if ranges is not None else len(part.basis_elements) for part, ranges in zip(parts, plans)),
    )
    for part_ind, part in enumerate(parts, start=1):
        node_count = len(part.basis_elements)
        sensitivity = np.zeros((3, len(target_points), node_count), dtype=float)
        ranges = plans[part_ind - 1]
        if ranges is not None:
            results = parallel_map(
                lambda node_range: _sensitivity_node_chunk(
                    part.basis_elements,
                    target_points,
                    u_coord,
                    v_coord,
                    gauss_weight,
                    node_range,
                    method,
                ),
                ranges,
                worker_count=worker_count,
                progress_callback=lambda done, total: task.advance(),
            )
            for start, chunk in results:
                sensitivity[:, :, start : start + chunk.shape[2]] = chunk
        else:
            for node_ind, basis in enumerate(part.basis_elements):
                sensitivity[:, :, node_ind] = _sensitivity_for_basis(
                    basis,
                    target_points,
                    u_coord,
                    v_coord,
                    gauss_weight,
                    method,
                )
                task.advance()
        part.sensitivity_matrix = sensitivity
    task.done()
    return parts


def _sensitivity_node_chunk(
    basis_elements: list[BasisElement],
    target_points: np.ndarray,
    u_coord: np.ndarray,
    v_coord: np.ndarray,
    gauss_weight: np.ndarray,
    node_range: tuple[int, int],
    method: str,
) -> tuple[int, np.ndarray]:
    start, stop = node_range
    chunk = np.zeros((3, len(target_points), stop - start), dtype=float)
    for out_ind, basis in enumerate(basis_elements[start:stop]):
        chunk[:, :, out_ind] = _sensitivity_for_basis(basis, target_points, u_coord, v_coord, gauss_weight, method)
    return start, chunk


def _sensitivity_for_basis(
    basis: BasisElement,
    target_points: np.ndarray,
    u_coord: np.ndarray,
    v_coord: np.ndarray,
    gauss_weight: np.ndarray,
    method: str = "vectorized",
) -> np.ndarray:
    if method == "scalar":
        return _sensitivity_for_basis_scalar(basis, target_points, u_coord, v_coord, gauss_weight)
    try:
        return _sensitivity_for_basis_vectorized(basis, target_points, u_coord, v_coord, gauss_weight)
    except MemoryError:
        return _sensitivity_for_basis_scalar(basis, target_points, u_coord, v_coord, gauss_weight)


def _sensitivity_for_basis_scalar(
    basis: BasisElement,
    target_points: np.ndarray,
    u_coord: np.ndarray,
    v_coord: np.ndarray,
    gauss_weight: np.ndarray,
) -> np.ndarray:
    d_b = np.zeros((len(target_points), 3), dtype=float)
    for tri_ind, area in enumerate(basis.area):
        if area <= 0:
            continue
        point_a, point_b, point_c = basis.triangle_points_abc[tri_ind]
        current = basis.current[tri_ind]
        for u, v, weight in zip(u_coord, v_coord, gauss_weight):
            gauss_point = point_a * u + point_b * v + point_c * (1.0 - u - v)
            r_vec = target_points - gauss_point
            distance = np.linalg.norm(r_vec, axis=1)
            safe = distance > 1e-15
            contribution = np.zeros_like(r_vec)
            contribution[safe] = (
                np.cross(np.broadcast_to(current, r_vec[safe].shape), r_vec[safe])
                * (distance[safe] ** -3)[:, None]
                * 2.0
                * area
                * weight
            )
            d_b += contribution
    return (BIOT_SAVART_COEFF * d_b).T


def _sensitivity_for_basis_vectorized(
    basis: BasisElement,
    target_points: np.ndarray,
    u_coord: np.ndarray,
    v_coord: np.ndarray,
    gauss_weight: np.ndarray,
) -> np.ndarray:
    valid = np.asarray(basis.area, dtype=float) > 0.0
    if not np.any(valid):
        return np.zeros((3, len(target_points)), dtype=float)

    areas = np.asarray(basis.area[valid], dtype=float)
    triangles = np.asarray(basis.triangle_points_abc[valid], dtype=float)
    currents = np.asarray(basis.current[valid], dtype=float)
    point_a = triangles[:, 0, :]
    point_b = triangles[:, 1, :]
    point_c = triangles[:, 2, :]
    u = np.asarray(u_coord, dtype=float)
    v = np.asarray(v_coord, dtype=float)
    weights = np.asarray(gauss_weight, dtype=float)

    gauss_points = (
        point_a[:, None, :] * u[None, :, None]
        + point_b[:, None, :] * v[None, :, None]
        + point_c[:, None, :] * (1.0 - u - v)[None, :, None]
    )
    r_vec = target_points[:, None, None, :] - gauss_points[None, :, :, :]
    distance = np.linalg.norm(r_vec, axis=3)
    distance_factor = np.zeros_like(distance)
    safe = distance > 1e-15
    distance_factor[safe] = distance[safe] ** -3
    current_cross_r = np.cross(currents[None, :, None, :], r_vec, axis=3)
    integration_weight = 2.0 * areas[:, None] * weights[None, :]
    d_b = np.sum(current_cross_r * distance_factor[:, :, :, None] * integration_weight[None, :, :, None], axis=(1, 2))
    return (BIOT_SAVART_COEFF * d_b).T


def calculate_resistance_matrix(parts: list[CoilPart], config: CoilGenConfig) -> list[CoilPart]:
    material_factor = config.specific_conductivity_conductor / config.conductor_thickness
    reporter = ProgressReporter.from_config(config, label="Resistance")
    entry_plans = []
    for part in parts:
        mesh = part.mesh
        adjacency = np.zeros((mesh.n_vertices, mesh.n_vertices), dtype=bool)
        for face in mesh.faces:
            a, b, c = [int(x) for x in face]
            adjacency[a, b] = True
            adjacency[b, c] = True
            adjacency[c, a] = True
        adjacency = adjacency | adjacency.T | np.eye(mesh.n_vertices, dtype=bool)
        entry_plans.append(np.where(adjacency))
    task = reporter.task("support matrix entries", sum(len(rows) for rows, _cols in entry_plans))
    for part, (rows, cols) in zip(parts, entry_plans):
        mesh = part.mesh
        resistance = np.zeros((mesh.n_vertices, mesh.n_vertices), dtype=float)
        basis_by_triangle = _basis_triangle_lookup(part)
        for node_a, node_b in zip(rows, cols):
            shared = set(part.basis_elements[node_a].triangles).intersection(part.basis_elements[node_b].triangles)
            total = 0.0
            for face_ind in shared:
                local_a = basis_by_triangle[(node_a, int(face_ind))]
                local_b = basis_by_triangle[(node_b, int(face_ind))]
                area = part.basis_elements[node_a].area[local_a]
                current_a = part.basis_elements[node_a].current[local_a]
                current_b = part.basis_elements[node_b].current[local_b]
                total += float(np.dot(current_a, current_b) * area)
            resistance[node_a, node_b] = total
            task.advance()
        part.resistance_matrix = (resistance + resistance.T) * material_factor / 2.0
    task.done()
    return parts


def _basis_triangle_lookup(part: CoilPart) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for node_ind, basis in enumerate(part.basis_elements):
        for local_ind, face_ind in enumerate(basis.triangles):
            lookup[(node_ind, int(face_ind))] = local_ind
    return lookup


def biot_savart_wire_path(
    points: np.ndarray,
    target_coords: np.ndarray,
    chunk_size: int = 512,
    *,
    worker_count: int = 1,
    min_work_size: int = 50000,
    max_chunk_memory_mb: float = 64.0,
    progress: ProgressReporter | None = None,
    progress_label: str = "Biot-Savart chunks",
) -> np.ndarray:
    """Evaluate the magnetic field of a thin polyline wire at target points.

    Each polyline edge is treated as a finite straight current segment and
    integrated analytically. The returned field is per ampere.
    """

    points = np.asarray(points, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)
    if len(points) < 2:
        return np.zeros((len(target_coords), 3), dtype=float)
    segment_starts = points[:-1]
    segment_vectors = points[1:] - points[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid_segments = segment_lengths > 1e-15
    if not np.any(valid_segments):
        return np.zeros((len(target_coords), 3), dtype=float)
    segment_starts = segment_starts[valid_segments]
    segment_vectors = segment_vectors[valid_segments]
    segment_lengths = segment_lengths[valid_segments]
    segment_directions = segment_vectors / segment_lengths[:, None]
    effective_chunk_size = memory_limited_chunk_size(
        chunk_size,
        len(target_coords),
        max_chunk_memory_mb=max_chunk_memory_mb,
        # Include the rationalized exterior-endpoint intermediates.
        bytes_per_pair=192,
    )
    ranges = chunk_ranges(len(segment_starts), effective_chunk_size)
    work_size = len(segment_starts) * len(target_coords)
    task = progress.task(progress_label, len(ranges)) if progress is not None else None
    if worker_count > 1 and len(ranges) > 1 and work_size >= int(min_work_size):
        chunks = parallel_map(
            lambda segment_range: _biot_savart_segment_chunk(
                segment_starts,
                segment_directions,
                segment_lengths,
                target_coords,
                segment_range,
            ),
            ranges,
            worker_count=worker_count,
            progress_callback=None if task is None else lambda done, total: task.update(done),
        )
        if task is not None:
            task.done()
        return np.sum(chunks, axis=0)
    field = np.zeros((len(target_coords), 3), dtype=float)
    for chunk_ind, segment_range in enumerate(ranges, start=1):
        field += _biot_savart_segment_chunk(
            segment_starts,
            segment_directions,
            segment_lengths,
            target_coords,
            segment_range,
        )
        if task is not None:
            task.update(chunk_ind)
    if task is not None:
        task.done()
    return field


def _biot_savart_segment_chunk(
    segment_starts: np.ndarray,
    segment_directions: np.ndarray,
    segment_lengths: np.ndarray,
    target_coords: np.ndarray,
    segment_range: tuple[int, int],
) -> np.ndarray:
    start, stop = segment_range
    seg_start = segment_starts[start:stop]
    direction = segment_directions[start:stop]
    length = segment_lengths[start:stop]

    r_start = target_coords[:, None, :] - seg_start[None, :, :]
    axial_start = np.einsum("ijk,jk->ij", r_start, direction)
    r_perp = r_start - axial_start[:, :, None] * direction[None, :, :]
    rho2 = np.einsum("ijk,ijk->ij", r_perp, r_perp)
    axial_end = length[None, :] - axial_start
    distance_start = np.sqrt(rho2 + axial_start * axial_start)
    distance_end = np.sqrt(rho2 + axial_end * axial_end)

    safe = (rho2 > 1e-30) & (distance_start > 1e-15) & (distance_end > 1e-15)
    finite_segment_factor = np.zeros_like(rho2)
    finite_segment_factor[safe] = (
        axial_start[safe] / distance_start[safe] + axial_end[safe] / distance_end[safe]
    ) / rho2[safe]
    # Beyond either endpoint, the signed endpoint terms nearly cancel for
    # samples close to the supporting line. Rationalizing their difference
    # retains the small physical field without dividing a rounded zero by rho².
    exterior = safe & ((axial_start < 0.0) | (axial_end < 0.0))
    a, b = np.abs(axial_start[exterior]), np.abs(axial_end[exterior])
    ra, rb = distance_start[exterior], distance_end[exterior]
    finite_segment_factor[exterior] = (
        np.broadcast_to(length, rho2.shape)[exterior] * (a + b) / (ra * rb * (a * rb + b * ra))
    )
    d_b = np.cross(direction[None, :, :], r_perp, axis=2) * finite_segment_factor[:, :, None]
    return BIOT_SAVART_COEFF * d_b.sum(axis=1)
