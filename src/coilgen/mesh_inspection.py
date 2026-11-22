"""Swept-mesh quality, topology and intersection inspection."""

from __future__ import annotations
import numpy as np
from .config import CoilGenConfig
from .geometry_collision import _SegmentBvhNode, accelerated_mesh_collision_diagnostics
from .geometry_kernels import active_geometry_backend
from .models import Mesh
from .progress import ProgressReporter


_DEFAULT_MESH_AREA_TOLERANCE_M2 = 1e-12


def mesh_manufacturing_diagnostics(
    mesh: Mesh, tolerance: float = _DEFAULT_MESH_AREA_TOLERANCE_M2
) -> dict[str, float | int | bool]:
    """Check whether a triangular shell is a consistently oriented solid boundary.

    ``tolerance`` is an area tolerance in square metres. Its square root is
    used as the corresponding edge-length tolerance. This preserves the legacy
    face-area interpretation while making short-edge rejection dimensionally
    consistent.
    """

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    nonfinite_vertices = int(np.count_nonzero(~np.all(np.isfinite(vertices), axis=1))) if len(vertices) else 0
    invalid_face_indices = int(np.count_nonzero((faces < 0) | (faces >= len(vertices)))) if len(faces) else 0
    if invalid_face_indices == 0:
        valid_faces = faces
    else:
        valid_faces = faces[np.all((faces >= 0) & (faces < len(vertices)), axis=1)]
    areas = _face_areas(vertices, valid_faces) if len(valid_faces) else np.zeros(0, dtype=float)
    finite_areas = areas[np.isfinite(areas)]
    area_tolerance = max(float(tolerance), 0.0)
    edge_tolerance = float(np.sqrt(area_tolerance))
    volume_tolerance = edge_tolerance**3
    degenerate_faces = int(np.count_nonzero(~np.isfinite(areas) | (areas <= area_tolerance)))
    edge_lengths = _mesh_edge_lengths(vertices, valid_faces) if len(valid_faces) else np.zeros(0, dtype=float)
    finite_edge_lengths = edge_lengths[np.isfinite(edge_lengths)]
    short_edges = int(np.count_nonzero(~np.isfinite(edge_lengths) | (edge_lengths <= edge_tolerance)))
    open_edges, nonmanifold_edges, misoriented_edges = _edge_topology_counts(valid_faces)
    signed_volume = _signed_mesh_volume(vertices, valid_faces) if len(valid_faces) else 0.0
    positive_volume = bool(np.isfinite(signed_volume) and signed_volume > volume_tolerance)
    manufacturing_ok = (
        nonfinite_vertices == 0
        and invalid_face_indices == 0
        and degenerate_faces == 0
        and short_edges == 0
        and open_edges == 0
        and nonmanifold_edges == 0
        and misoriented_edges == 0
        and positive_volume
    )
    return {
        "swept_body_created": bool(len(vertices) > 0 and len(faces) > 0),
        "manufacturing_ok": bool(manufacturing_ok),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "nonfinite_vertex_count": int(nonfinite_vertices),
        "invalid_face_index_count": int(invalid_face_indices),
        "degenerate_face_count": int(degenerate_faces),
        "short_edge_count": int(short_edges),
        "open_edge_count": int(open_edges),
        "nonmanifold_edge_count": int(nonmanifold_edges),
        "misoriented_edge_count": int(misoriented_edges),
        "watertight": bool(open_edges == 0 and nonmanifold_edges == 0 and misoriented_edges == 0 and len(faces) > 0),
        "signed_volume_m3": float(signed_volume),
        "absolute_volume_m3": float(abs(signed_volume)) if np.isfinite(signed_volume) else 0.0,
        "positive_volume": bool(positive_volume),
        "edge_length_tolerance_m": float(edge_tolerance),
        "min_face_area": float(np.min(finite_areas)) if len(finite_areas) else 0.0,
        "max_face_area": float(np.max(finite_areas)) if len(finite_areas) else 0.0,
        "min_edge_length_m": float(np.min(finite_edge_lengths)) if len(finite_edge_lengths) else 0.0,
        "max_edge_length_m": float(np.max(finite_edge_lengths)) if len(finite_edge_lengths) else 0.0,
    }


def _swept_mesh_quality_diagnostics(
    mesh: Mesh,
    config: CoilGenConfig,
) -> dict[str, float | int | bool]:
    triangles = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    minimum_edges = np.min(edge_lengths, axis=1)
    maximum_edges = np.max(edge_lengths, axis=1)
    edge_ratios = np.divide(
        maximum_edges,
        minimum_edges,
        out=np.full(len(triangles), np.inf),
        where=minimum_edges > 1e-30,
    )
    area_twice = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    altitude_ratios = np.divide(
        maximum_edges**2,
        area_twice,
        out=np.full(len(triangles), np.inf),
        where=area_twice > 1e-30,
    )
    edge_ratio = float(np.max(edge_ratios)) if len(edge_ratios) else 0.0
    altitude_ratio = float(np.max(altitude_ratios)) if len(altitude_ratios) else 0.0
    edge_limit = float(config.mesh_quality_max_triangle_edge_ratio)
    altitude_limit = float(config.mesh_quality_max_altitude_aspect_ratio)
    return {
        "sweep_triangle_edge_ratio_max": edge_ratio,
        "sweep_triangle_altitude_aspect_ratio_max": altitude_ratio,
        "sweep_triangle_edge_ratio_limit": edge_limit,
        "sweep_triangle_altitude_aspect_ratio_limit": altitude_limit,
        "sweep_triangle_quality_violation": bool(edge_ratio > edge_limit or altitude_ratio > altitude_limit),
    }


def _swept_mesh_self_intersection_diagnostics(
    mesh: Mesh,
    *,
    maximum_reported_intersections: int = 100,
) -> dict[str, float | int | bool | str]:
    """Check non-adjacent swept triangles using a BVH broad phase."""

    reporter = ProgressReporter(label="Swept collision")
    reporter.report(None, "Check non-adjacent swept triangles", force=True)
    backend = active_geometry_backend()
    if backend is not None:
        return accelerated_mesh_collision_diagnostics(
            mesh, backend=backend, maximum_reported_intersections=maximum_reported_intersections
        )

    triangles = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
    if len(triangles) < 2:
        return {
            "sweep_mesh_collision_broad_phase": "triangle_bvh",
            "sweep_mesh_collision_candidate_pair_count": 0,
            "sweep_mesh_self_intersection_count": 0,
            "sweep_mesh_self_intersection_truncated": False,
            "sweep_mesh_self_intersection": False,
        }
    minima = np.min(triangles, axis=1)
    maxima = np.max(triangles, axis=1)
    centroids = 0.5 * (minima + maxima)
    scale = max(float(np.max(np.max(maxima, axis=0) - np.min(minima, axis=0))), 1.0)
    tolerance = 1e-12 * scale

    def build(indices: np.ndarray) -> _SegmentBvhNode:
        minimum = np.min(minima[indices], axis=0)
        maximum = np.max(maxima[indices], axis=0)
        if len(indices) <= 12:
            return _SegmentBvhNode(indices, minimum, maximum)
        spread = np.ptp(centroids[indices], axis=0)
        axis = int(np.argmax(spread))
        order = indices[np.argsort(centroids[indices, axis], kind="stable")]
        middle = len(order) // 2
        return _SegmentBvhNode(
            np.zeros(0, dtype=np.int64),
            minimum,
            maximum,
            build(order[:middle]),
            build(order[middle:]),
        )

    def boxes_overlap(first: _SegmentBvhNode, second: _SegmentBvhNode) -> bool:
        return bool(
            np.all(first.maximum + tolerance >= second.minimum) and np.all(second.maximum + tolerance >= first.minimum)
        )

    candidate_count = 0
    intersection_count = 0
    truncated = False
    visited = 0
    faces = np.asarray(mesh.faces, dtype=np.int64)

    def test_pair(first_index: int, second_index: int) -> None:
        nonlocal candidate_count, intersection_count, truncated
        if truncated or np.intersect1d(faces[first_index], faces[second_index], assume_unique=True).size:
            return
        if np.any(maxima[first_index] + tolerance < minima[second_index]) or np.any(
            maxima[second_index] + tolerance < minima[first_index]
        ):
            return
        candidate_count += 1
        if _triangle_pair_intersects_robust(
            triangles[first_index],
            triangles[second_index],
            tolerance,
        ):
            intersection_count += 1
            truncated = intersection_count >= maximum_reported_intersections

    def collect(first: _SegmentBvhNode, second: _SegmentBvhNode) -> None:
        nonlocal visited
        visited += 1
        if visited % 256 == 0:
            reporter.report(None, f"Sweep collision · {visited:,} BVH visits · {candidate_count:,} candidate pairs")
        if truncated or not boxes_overlap(first, second):
            return
        if first is second:
            if first.leaf:
                for local_first in range(len(first.indices)):
                    for local_second in range(local_first + 1, len(first.indices)):
                        test_pair(int(first.indices[local_first]), int(first.indices[local_second]))
                return
            collect(first.left, first.left)
            collect(first.left, first.right)
            collect(first.right, first.right)
            return
        if first.leaf and second.leaf:
            for first_index in first.indices:
                for second_index in second.indices:
                    test_pair(int(first_index), int(second_index))
            return
        first_extent = float(np.dot(first.maximum - first.minimum, first.maximum - first.minimum))
        second_extent = float(np.dot(second.maximum - second.minimum, second.maximum - second.minimum))
        if not first.leaf and (second.leaf or first_extent >= second_extent):
            collect(first.left, second)
            collect(first.right, second)
        else:
            collect(first, second.left)
            collect(first, second.right)

    root = build(np.arange(len(triangles), dtype=np.int64))
    collect(root, root)
    return {
        "sweep_mesh_collision_broad_phase": "triangle_bvh",
        "sweep_mesh_collision_candidate_pair_count": int(candidate_count),
        "sweep_mesh_self_intersection_count": int(intersection_count),
        "sweep_mesh_self_intersection_truncated": bool(truncated),
        "sweep_mesh_self_intersection": bool(intersection_count > 0),
    }


def _triangle_pair_intersects_robust(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> bool:
    first_normal = np.cross(first[1] - first[0], first[2] - first[0])
    second_normal = np.cross(second[1] - second[0], second[2] - second[0])
    first_length = float(np.linalg.norm(first_normal))
    second_length = float(np.linalg.norm(second_normal))
    first_edge_length = float(np.max(np.linalg.norm(np.roll(first, -1, axis=0) - first, axis=1)))
    second_edge_length = float(np.max(np.linalg.norm(np.roll(second, -1, axis=0) - second, axis=1)))
    if first_length <= tolerance * first_edge_length or second_length <= tolerance * second_edge_length:
        return False
    unit_first = first_normal / first_length
    unit_second = second_normal / second_length
    parallel = float(np.linalg.norm(np.cross(unit_first, unit_second))) <= 1e-10
    coplanar = parallel and abs(float(np.dot(unit_first, second[0] - first[0]))) <= tolerance
    if coplanar:
        drop_axis = int(np.argmax(np.abs(unit_first)))
        return _coplanar_triangles_intersect(
            np.delete(first, drop_axis, axis=1),
            np.delete(second, drop_axis, axis=1),
            tolerance,
        )
    for triangle_a, triangle_b in ((first, second), (second, first)):
        for edge_index in range(3):
            if _segment_triangle_intersects(
                triangle_a[edge_index],
                triangle_a[(edge_index + 1) % 3],
                triangle_b,
                tolerance,
            ):
                return True
    return False


def _segment_triangle_intersects(
    start: np.ndarray,
    end: np.ndarray,
    triangle: np.ndarray,
    tolerance: float,
) -> bool:
    direction = end - start
    edge_first = triangle[1] - triangle[0]
    edge_second = triangle[2] - triangle[0]
    cross_direction = np.cross(direction, edge_second)
    determinant = float(np.dot(edge_first, cross_direction))
    direction_length = float(np.linalg.norm(direction))
    edge_first_length = float(np.linalg.norm(edge_first))
    edge_second_length = float(np.linalg.norm(edge_second))
    # The determinant has units m^3, whereas the geometric tolerance is in m.
    # An absolute determinant cutoff misses crossings in fine conductor meshes.
    determinant_tolerance = 32.0 * np.finfo(float).eps * direction_length * edge_first_length * edge_second_length
    if abs(determinant) <= determinant_tolerance or direction_length <= tolerance:
        return False
    area_twice = float(np.linalg.norm(np.cross(edge_first, edge_second)))
    maximum_edge_length = max(edge_first_length, edge_second_length, float(np.linalg.norm(edge_second - edge_first)))
    if area_twice <= tolerance * maximum_edge_length:
        return False
    barycentric_tolerance = tolerance * maximum_edge_length / area_twice
    segment_tolerance = tolerance / direction_length
    inverse = 1.0 / determinant
    offset = start - triangle[0]
    first_coordinate = inverse * float(np.dot(offset, cross_direction))
    if first_coordinate < -barycentric_tolerance or first_coordinate > 1.0 + barycentric_tolerance:
        return False
    cross_offset = np.cross(offset, edge_first)
    second_coordinate = inverse * float(np.dot(direction, cross_offset))
    if second_coordinate < -barycentric_tolerance or first_coordinate + second_coordinate > 1.0 + barycentric_tolerance:
        return False
    segment_coordinate = inverse * float(np.dot(edge_second, cross_offset))
    return bool(-segment_tolerance <= segment_coordinate <= 1.0 + segment_tolerance)


def _coplanar_triangles_intersect(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> bool:
    for triangle_a, triangle_b in ((first, second), (second, first)):
        for edge_a in range(3):
            for edge_b in range(3):
                if _segments_intersect_or_touch_2d(
                    triangle_a[edge_a],
                    triangle_a[(edge_a + 1) % 3],
                    triangle_b[edge_b],
                    triangle_b[(edge_b + 1) % 3],
                    tolerance,
                ):
                    return True
        if _point_in_triangle_2d(triangle_a[0], triangle_b, tolerance):
            return True
    return False


def _segments_intersect_or_touch_2d(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> bool:
    def orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        edge = end - start
        offset = point - start
        return float(edge[0] * offset[1] - edge[1] * offset[0])

    values = (
        orientation(first_start, first_end, second_start),
        orientation(first_start, first_end, second_end),
        orientation(second_start, second_end, first_start),
        orientation(second_start, second_end, first_end),
    )
    first_tolerance = tolerance * float(np.linalg.norm(first_end - first_start))
    second_tolerance = tolerance * float(np.linalg.norm(second_end - second_start))
    if values[0] * values[1] < -(first_tolerance**2) and values[2] * values[3] < -(second_tolerance**2):
        return True
    for value, point, start, end, area_tolerance in (
        (values[0], second_start, first_start, first_end, first_tolerance),
        (values[1], second_end, first_start, first_end, first_tolerance),
        (values[2], first_start, second_start, second_end, second_tolerance),
        (values[3], first_end, second_start, second_end, second_tolerance),
    ):
        if (
            abs(value) <= area_tolerance
            and np.all(point >= np.minimum(start, end) - tolerance)
            and np.all(point <= np.maximum(start, end) + tolerance)
        ):
            return True
    return False


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, tolerance: float) -> bool:
    signs = []
    tolerances = []
    for edge_index in range(3):
        start = triangle[edge_index]
        end = triangle[(edge_index + 1) % 3]
        edge = end - start
        offset = point - start
        signs.append(float(edge[0] * offset[1] - edge[1] * offset[0]))
        tolerances.append(tolerance * float(np.linalg.norm(edge)))
    return bool(
        all(value >= -limit for value, limit in zip(signs, tolerances))
        or all(value <= limit for value, limit in zip(signs, tolerances))
    )


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=float)
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)


def _mesh_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=float)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)


def _signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    triangles = vertices[faces]
    return float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def _edge_topology_counts(faces: np.ndarray) -> tuple[int, int, int]:
    edge_counts: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            start = int(a)
            end = int(b)
            edge = (min(start, end), max(start, end))
            state = edge_counts.setdefault(edge, [0, 0])
            state[0] += 1
            state[1] += 1 if start < end else -1
    open_edges = sum(1 for count, _orientation in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count, _orientation in edge_counts.values() if count > 2)
    misoriented_edges = sum(1 for count, orientation in edge_counts.values() if count == 2 and orientation != 0)
    return int(open_edges), int(nonmanifold_edges), int(misoriented_edges)
