from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .field_components import field_component_values as select_field_component_values
from .field_components import normalize_field_component
from .models import Mesh


def normalise_symmetry_mode(value: Any, *, default: str = "auto") -> str:
    """Return a supported symmetry mode: ``auto``, ``strict``, or ``off``."""

    mode = str(value).strip().lower()
    if mode in {"auto", "strict", "off"}:
        return mode
    return default


def generated_candidate_symmetric_part_pairs(coil_mesh_file: str, part_count: int) -> list[tuple[int, ...]]:
    """Known generated geometries where CoilGen can infer mirror-paired parts.

    The returned indices are zero-based. These are only geometry candidates; the
    caller must still validate vertex matching, target-field parity, and current
    direction before relying on symmetry.
    """

    mesh_name = str(coil_mesh_file).strip().lower()
    count = int(part_count)
    if count < 2:
        return []
    if mesh_name in {
        "create bi-planary mesh",
        "create circular bi-planary mesh",
        "create flanged split bore mesh",
    }:
        return [(0, 1)]
    if mesh_name in {
        "create shielded bi-planary mesh",
        "create shielded flanged split bore mesh",
    }:
        return [pair for pair in ((0, 1), (2, 3)) if max(pair) < count]
    return []


def geometry_scale(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return 1.0
    return max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0e-12)


def absolute_tolerance(relative_tolerance: float, scale: float) -> float:
    return max(float(relative_tolerance) * max(float(scale), 1.0e-12), 1e-12)


def reflect_points(points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Mirror-plane normal must be a finite non-zero vector.")
    normal = normal / norm
    signed_distance = (points - plane_point) @ normal
    return points - 2.0 * signed_distance[:, None] * normal[None, :]


def mirror_symmetric_mesh_from_half(
    mesh: Mesh,
    plane_point: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    plane_normal: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
    *,
    half_space: str = "positive",
    clip_tolerance: float = 1e-9,
    merge_tolerance: float | None = None,
    plane_snap_tolerance: float | None = None,
) -> tuple[Mesh, dict[str, Any]]:
    """Clip a support mesh to one mirror half and mirror it back to a full mesh.

    This is a geometry-preparation step for strict symmetry solves on imported
    meshes that are physically intended to be symmetric but cannot pass exact
    vertex-pair validation because of scan/STL noise.  The returned mesh is a
    full support surface, so contour extraction and interconnection still run on
    the ordinary full geometry.
    """

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh must contain (n, 3) vertices and triangular (m, 3) faces.")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Cannot build a symmetric mesh from an empty mesh.")

    point, normal = _normalised_plane(plane_point, plane_normal)
    signed_distances = (vertices - point) @ normal
    side = _half_space_sign(half_space, signed_distances, faces)
    scale = geometry_scale(vertices)
    clip_tol = max(float(clip_tolerance), 1e-12 * max(scale, 1.0))
    snap_tol = float(plane_snap_tolerance) if plane_snap_tolerance is not None else clip_tol
    if not np.isfinite(snap_tol) or snap_tol < clip_tol:
        snap_tol = clip_tol
    merge_tol = float(merge_tolerance) if merge_tolerance is not None else max(clip_tol * 0.01, 1e-12 * max(scale, 1.0))
    if merge_tol <= 0.0 or not np.isfinite(merge_tol):
        raise ValueError("merge_tolerance must be positive.")

    half_vertices, half_faces, clip_summary = _clip_mesh_to_halfspace(
        vertices,
        faces,
        point,
        normal,
        side=side,
        clip_tolerance=clip_tol,
        merge_tolerance=merge_tol,
    )
    seam_collapse_summary = {
        "plane_snapped_vertex_count": 0,
        "plane_short_edge_collapse_count": 0,
    }
    if snap_tol > clip_tol:
        half_vertices, half_faces, seam_collapse_summary = _collapse_short_edges_near_plane(
            half_vertices,
            half_faces,
            point,
            normal,
            threshold=snap_tol,
            plane_tolerance=max(clip_tol, 4.0 * merge_tol),
        )
    if len(half_vertices) == 0 or len(half_faces) == 0:
        raise ValueError("The selected symmetry half-space contains no mesh faces.")

    full_mesh, mirror_summary = _mirror_half_mesh(
        half_vertices,
        half_faces,
        point,
        normal,
        plane_tolerance=max(clip_tol, 4.0 * merge_tol),
    )
    topology_summary = _mesh_topology_diagnostics(
        full_mesh.vertices,
        full_mesh.faces,
        plane_point=point,
        plane_normal=normal,
        plane_tolerance=max(clip_tol, 4.0 * merge_tol),
    )
    _raise_for_invalid_half_mesh_topology(topology_summary)
    full_mesh.metadata.update(dict(mesh.metadata))
    summary: dict[str, Any] = {
        "enabled": True,
        "method": "half_mesh_clip_and_mirror",
        "half_space": "positive" if side > 0 else "negative",
        "plane_point_m": [float(value) for value in point],
        "plane_normal": [float(value) for value in normal],
        "clip_tolerance_m": float(clip_tol),
        "plane_snap_tolerance_m": float(snap_tol),
        **seam_collapse_summary,
        "merge_tolerance_m": float(merge_tol),
        "original_vertex_count": int(len(vertices)),
        "original_face_count": int(len(faces)),
        **clip_summary,
        **mirror_summary,
        **topology_summary,
        "result_vertex_count": int(full_mesh.n_vertices),
        "result_face_count": int(full_mesh.n_faces),
    }
    full_mesh.metadata["symmetry_preparation"] = dict(summary)
    return full_mesh, summary


def _normalised_plane(
    plane_point: tuple[float, float, float] | np.ndarray,
    plane_normal: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(plane_point, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Mirror-plane normal must be a finite non-zero vector.")
    return point, normal / norm


def _half_space_sign(half_space: str, signed_distances: np.ndarray, faces: np.ndarray) -> int:
    mode = str(half_space).strip().lower().replace("-", "_")
    if mode in {"positive", "plus", "+", "right"}:
        return 1
    if mode in {"negative", "minus", "-", "left"}:
        return -1
    if mode != "auto":
        raise ValueError("half_space must be 'positive', 'negative', or 'auto'.")
    face_center_distances = np.mean(signed_distances[np.asarray(faces, dtype=np.int64)], axis=1)
    positive_count = int(np.count_nonzero(face_center_distances >= 0.0))
    negative_count = int(np.count_nonzero(face_center_distances <= 0.0))
    return 1 if positive_count >= negative_count else -1


def _clip_mesh_to_halfspace(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    *,
    side: int,
    clip_tolerance: float,
    merge_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    out_vertices: list[np.ndarray] = []
    out_faces: list[tuple[int, int, int]] = []
    coordinate_to_index: dict[tuple[int, int, int], int] = {}
    clipped_face_count = 0
    dropped_face_count = 0
    kept_face_count = 0

    def add_vertex(point: np.ndarray) -> int:
        point = np.asarray(point, dtype=float).reshape(3)
        distance = float((point - plane_point) @ plane_normal)
        if abs(distance) <= clip_tolerance:
            point = point - distance * plane_normal
        key = tuple(np.round(point / merge_tolerance).astype(np.int64).tolist())
        existing = coordinate_to_index.get(key)
        if existing is not None:
            return int(existing)
        index = len(out_vertices)
        coordinate_to_index[key] = index
        out_vertices.append(point)
        return index

    for face in faces:
        points = vertices[np.asarray(face, dtype=np.int64)]
        signed = side * ((points - plane_point) @ plane_normal)
        inside = signed >= -clip_tolerance
        if not np.any(inside):
            dropped_face_count += 1
            continue
        if np.all(inside):
            polygon = [point.copy() for point in points]
            kept_face_count += 1
        else:
            polygon = _clip_triangle_polygon(points, signed, clip_tolerance)
            clipped_face_count += 1
        if len(polygon) < 3:
            dropped_face_count += 1
            continue
        polygon_indices = [add_vertex(point) for point in polygon]
        for tri in _triangulate_polygon_indices(polygon_indices):
            if len(set(tri)) < 3:
                continue
            tri_points = np.asarray([out_vertices[ind] for ind in tri], dtype=float)
            area2 = float(np.linalg.norm(np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])))
            if area2 <= merge_tolerance * merge_tolerance:
                continue
            out_faces.append(tuple(int(ind) for ind in tri))

    return (
        np.asarray(out_vertices, dtype=float),
        np.asarray(out_faces, dtype=np.int64),
        {
            "half_vertex_count": int(len(out_vertices)),
            "half_face_count": int(len(out_faces)),
            "kept_original_face_count": int(kept_face_count),
            "clipped_original_face_count": int(clipped_face_count),
            "dropped_original_face_count": int(dropped_face_count),
        },
    )


def _clip_triangle_polygon(points: np.ndarray, signed_distances: np.ndarray, tolerance: float) -> list[np.ndarray]:
    polygon = [(points[ind].copy(), float(signed_distances[ind])) for ind in range(3)]
    clipped: list[tuple[np.ndarray, float]] = []
    for current_ind, current in enumerate(polygon):
        previous = polygon[current_ind - 1]
        previous_point, previous_distance = previous
        current_point, current_distance = current
        previous_inside = previous_distance >= -tolerance
        current_inside = current_distance >= -tolerance
        if previous_inside != current_inside:
            denom = previous_distance - current_distance
            if abs(denom) > 1e-30:
                fraction = previous_distance / denom
                fraction = float(np.clip(fraction, 0.0, 1.0))
                clipped.append((previous_point + fraction * (current_point - previous_point), 0.0))
        if current_inside:
            clipped.append((current_point, current_distance))
    return [point for point, _distance in clipped]


def _collapse_short_edges_near_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    *,
    threshold: float,
    plane_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collapse seam-adjacent sliver edges without changing half-space selection."""

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0 or threshold <= 0.0:
        return vertices, faces, {"plane_snapped_vertex_count": 0, "plane_short_edge_collapse_count": 0}
    signed = (vertices - plane_point) @ plane_normal
    seam = np.abs(signed) <= float(plane_tolerance)
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(node: int) -> int:
        root = int(node)
        while parent[root] != root:
            root = int(parent[root])
        while parent[int(node)] != int(node):
            next_node = int(parent[int(node)])
            parent[int(node)] = root
            node = next_node
        return root

    def union(node_a: int, node_b: int) -> None:
        root_a = find(node_a)
        root_b = find(node_b)
        if root_a == root_b:
            return
        if seam[root_a] and not seam[root_b]:
            parent[root_b] = root_a
        elif seam[root_b] and not seam[root_a]:
            parent[root_a] = root_b
        elif root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    collapse_count = 0
    for node_a_raw, node_b_raw in edges:
        node_a = int(node_a_raw)
        node_b = int(node_b_raw)
        if not (seam[node_a] or seam[node_b]):
            continue
        length = float(np.linalg.norm(vertices[node_a] - vertices[node_b]))
        if np.isfinite(length) and length < threshold:
            union(node_a, node_b)
            collapse_count += 1

    roots = np.asarray([find(ind) for ind in range(len(vertices))], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    if len(unique_roots) == len(vertices):
        return vertices, faces, {"plane_snapped_vertex_count": 0, "plane_short_edge_collapse_count": 0}

    out_vertices = np.zeros((len(unique_roots), 3), dtype=float)
    snapped_count = 0
    for out_ind, root in enumerate(unique_roots):
        members = np.flatnonzero(roots == int(root))
        point = np.mean(vertices[members], axis=0)
        if np.any(seam[members]):
            distance = float((point - plane_point) @ plane_normal)
            point = point - distance * plane_normal
            snapped_count += int(len(members) - 1)
        out_vertices[out_ind] = point
    out_faces = inverse[faces]
    valid = np.asarray([len(set(int(value) for value in face)) == 3 for face in out_faces], dtype=bool)
    out_faces = out_faces[valid]
    if len(out_faces):
        sorted_faces = np.sort(out_faces, axis=1)
        _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
        out_faces = out_faces[np.sort(unique_indices)]
    return (
        out_vertices,
        out_faces,
        {
            "plane_snapped_vertex_count": int(snapped_count),
            "plane_short_edge_collapse_count": int(collapse_count),
        },
    )


def _triangulate_polygon_indices(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) == 3:
        return [(indices[0], indices[1], indices[2])]
    return [(indices[0], indices[ind], indices[ind + 1]) for ind in range(1, len(indices) - 1)]


def _mesh_topology_diagnostics(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    plane_tolerance: float,
) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    diagnostics: dict[str, Any] = {
        "result_invalid_face_count": 0,
        "result_duplicate_index_face_count": 0,
        "result_degenerate_face_count": 0,
        "result_boundary_edge_count": 0,
        "result_boundary_loop_count": 0,
        "result_nonmanifold_edge_count": 0,
        "result_mirror_seam_edge_count": 0,
        "result_mirror_seam_boundary_edge_count": 0,
        "result_min_face_area_m2": float("nan"),
        "result_p01_face_area_m2": float("nan"),
        "result_median_face_area_m2": float("nan"),
        "result_min_triangle_quality": float("nan"),
        "result_p01_triangle_quality": float("nan"),
        "result_median_triangle_quality": float("nan"),
        "result_min_edge_length_m": float("nan"),
        "result_p01_edge_length_m": float("nan"),
        "result_median_edge_length_m": float("nan"),
    }
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        diagnostics["result_invalid_face_count"] = int(len(faces)) if faces.ndim else 0
        return diagnostics
    if len(vertices) == 0 or len(faces) == 0:
        return diagnostics

    invalid = np.any((faces < 0) | (faces >= len(vertices)), axis=1)
    duplicate = np.asarray([len(set(int(value) for value in face)) < 3 for face in faces], dtype=bool)
    valid_faces = faces[~invalid & ~duplicate]
    diagnostics["result_invalid_face_count"] = int(np.count_nonzero(invalid))
    diagnostics["result_duplicate_index_face_count"] = int(np.count_nonzero(duplicate))
    if len(valid_faces) == 0:
        return diagnostics

    tri = vertices[valid_faces]
    edge01 = tri[:, 1] - tri[:, 0]
    edge12 = tri[:, 2] - tri[:, 1]
    edge20 = tri[:, 0] - tri[:, 2]
    edge_lengths = np.concatenate(
        (
            np.linalg.norm(edge01, axis=1),
            np.linalg.norm(edge12, axis=1),
            np.linalg.norm(edge20, axis=1),
        )
    )
    areas = 0.5 * np.linalg.norm(np.cross(edge01, tri[:, 2] - tri[:, 0]), axis=1)
    scale = geometry_scale(vertices)
    area_tolerance = max((1e-12 * max(scale, 1.0)) ** 2, 1e-30)
    diagnostics["result_degenerate_face_count"] = int(np.count_nonzero(~np.isfinite(areas) | (areas <= area_tolerance)))

    squared_edge_sum = (
        np.sum(edge01 * edge01, axis=1) + np.sum(edge12 * edge12, axis=1) + np.sum(edge20 * edge20, axis=1)
    )
    quality = np.divide(
        4.0 * np.sqrt(3.0) * areas,
        squared_edge_sum,
        out=np.zeros_like(areas),
        where=squared_edge_sum > 0.0,
    )
    _update_distribution_diagnostics(diagnostics, "face_area_m2", areas)
    _update_distribution_diagnostics(diagnostics, "triangle_quality", quality)
    _update_distribution_diagnostics(diagnostics, "edge_length_m", edge_lengths)

    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in valid_faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[(int(min(u, v)), int(max(u, v)))] += 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    nonmanifold_edges = [edge for edge, count in edge_count.items() if count > 2]
    diagnostics["result_boundary_edge_count"] = int(len(boundary_edges))
    diagnostics["result_boundary_loop_count"] = int(_boundary_loop_count(boundary_edges))
    diagnostics["result_nonmanifold_edge_count"] = int(len(nonmanifold_edges))

    point = np.asarray(plane_point, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm > 0.0 and np.isfinite(norm):
        normal = normal / norm
        plane_vertices = np.abs((vertices - point[None, :]) @ normal) <= float(plane_tolerance)
        seam_edges = [
            edge for edge in edge_count if bool(plane_vertices[int(edge[0])]) and bool(plane_vertices[int(edge[1])])
        ]
        seam_boundary_edges = [edge for edge in seam_edges if edge_count[edge] == 1]
        diagnostics["result_mirror_plane_vertex_count_checked"] = int(np.count_nonzero(plane_vertices))
        diagnostics["result_mirror_seam_edge_count"] = int(len(seam_edges))
        diagnostics["result_mirror_seam_boundary_edge_count"] = int(len(seam_boundary_edges))
    return diagnostics


def _update_distribution_diagnostics(diagnostics: dict[str, Any], key: str, values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    diagnostics[f"result_min_{key}"] = float(np.min(finite))
    diagnostics[f"result_p01_{key}"] = float(np.percentile(finite, 1.0))
    diagnostics[f"result_median_{key}"] = float(np.median(finite))


def _boundary_loop_count(boundary_edges: list[tuple[int, int]]) -> int:
    if not boundary_edges:
        return 0
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    visited: set[tuple[int, int]] = set()
    loop_count = 0
    for start_a, start_b in boundary_edges:
        edge = (min(int(start_a), int(start_b)), max(int(start_a), int(start_b)))
        if edge in visited:
            continue
        loop_count += 1
        prev = -1
        current = int(start_a)
        next_node = int(start_b)
        guard = 0
        while True:
            visited.add((min(current, next_node), max(current, next_node)))
            prev, current = current, next_node
            if current == int(start_a):
                break
            candidates = [node for node in adjacency[current] if node != prev]
            if not candidates:
                break
            unused = [node for node in candidates if (min(current, node), max(current, node)) not in visited]
            next_node = int(unused[0] if unused else candidates[0])
            guard += 1
            if guard > len(boundary_edges) + 5:
                break
    return loop_count


def _raise_for_invalid_half_mesh_topology(diagnostics: dict[str, Any]) -> None:
    fatal = {
        "invalid faces": int(diagnostics.get("result_invalid_face_count", 0)),
        "duplicate-index faces": int(diagnostics.get("result_duplicate_index_face_count", 0)),
        "degenerate faces": int(diagnostics.get("result_degenerate_face_count", 0)),
        "nonmanifold edges": int(diagnostics.get("result_nonmanifold_edge_count", 0)),
        "open mirror-seam edges": int(diagnostics.get("result_mirror_seam_boundary_edge_count", 0)),
    }
    failures = [f"{label}={count}" for label, count in fatal.items() if count > 0]
    if failures:
        raise ValueError("Half-mesh symmetry reconstruction produced invalid topology: " + ", ".join(failures))


def _mirror_half_mesh(
    half_vertices: np.ndarray,
    half_faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    *,
    plane_tolerance: float,
) -> tuple[Mesh, dict[str, Any]]:
    signed = (half_vertices - plane_point) @ plane_normal
    plane_mask = np.abs(signed) <= float(plane_tolerance)
    mirror_index = np.empty(len(half_vertices), dtype=np.int64)
    full_vertices = [vertex.copy() for vertex in half_vertices]
    for ind, vertex in enumerate(half_vertices):
        if bool(plane_mask[ind]):
            mirror_index[ind] = ind
        else:
            mirror_index[ind] = len(full_vertices)
            full_vertices.append(reflect_points(vertex[None, :], plane_point, plane_normal)[0])

    full_faces: list[tuple[int, int, int]] = [tuple(int(value) for value in face) for face in half_faces]
    mirrored_face_count = 0
    for face in half_faces:
        mirrored = (int(mirror_index[face[0]]), int(mirror_index[face[2]]), int(mirror_index[face[1]]))
        if len(set(mirrored)) < 3:
            continue
        if all(int(mirror_index[int(vertex)]) == int(vertex) for vertex in face):
            continue
        full_faces.append(mirrored)
        mirrored_face_count += 1

    return (
        Mesh(vertices=np.asarray(full_vertices, dtype=float), faces=np.asarray(full_faces, dtype=np.int64)),
        {
            "mirror_plane_vertex_count": int(np.count_nonzero(plane_mask)),
            "mirrored_vertex_count": int(len(full_vertices) - len(half_vertices)),
            "mirrored_face_count": int(mirrored_face_count),
            "mirror_pair_count": int(len(half_vertices) - np.count_nonzero(plane_mask)),
        },
    )


def nearest_indices(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_bruteforce_work: int = 1_000_000,
    max_total_work: int = 100_000_000,
) -> np.ndarray | None:
    """Nearest target index for every source point.

    SciPy is used when available. Without SciPy, a quantized dictionary lookup
    handles exact or near-exact paired point sets such as mirrored STL meshes,
    and a chunked brute-force fallback handles modest imported meshes without
    allocating a full all-pairs distance matrix.
    """

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        _distances, indices = cKDTree(target).query(source)
        return np.asarray(indices, dtype=np.int64)
    except Exception:
        pass
    quantized = _quantized_nearest_indices(source, target)
    if quantized is not None:
        return quantized
    work_size = int(len(source)) * int(len(target))
    if work_size > int(max_total_work):
        return None
    if work_size <= int(max_bruteforce_work):
        diff = source[:, None, :] - target[None, :, :]
        return np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int64)
    return _chunked_nearest_indices(source, target, max_chunk_work=int(max_bruteforce_work))


def _chunked_nearest_indices(source: np.ndarray, target: np.ndarray, *, max_chunk_work: int) -> np.ndarray | None:
    if len(source) == 0:
        return np.zeros(0, dtype=np.int64)
    if len(target) == 0:
        return None
    chunk_size = max(int(max_chunk_work) // max(int(len(target)), 1), 1)
    matches = np.empty(len(source), dtype=np.int64)
    target_points = np.asarray(target, dtype=float)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        diff = source[start:stop, None, :] - target_points[None, :, :]
        matches[start:stop] = np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int64)
    return matches


def _quantized_nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    if len(source) == 0:
        return np.zeros(0, dtype=np.int64)
    if len(target) == 0:
        return None
    scale = geometry_scale(np.vstack((source, target)))
    step = max(1.0e-8 * max(scale, 1.0), 1.0e-12)
    target_keys = np.rint(target / step).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for ind, key in enumerate(target_keys):
        buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(int(ind))
    source_keys = np.rint(source / step).astype(np.int64)
    offsets = np.asarray([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], dtype=np.int64)
    matches = np.empty(len(source), dtype=np.int64)
    for source_ind, key in enumerate(source_keys):
        best_index = -1
        best_distance_sq = float("inf")
        for offset in offsets:
            bucket = buckets.get((int(key[0] + offset[0]), int(key[1] + offset[1]), int(key[2] + offset[2])))
            if not bucket:
                continue
            candidates = np.asarray(bucket, dtype=np.int64)
            diff = target[candidates] - source[source_ind]
            distances_sq = np.sum(diff * diff, axis=1)
            local = int(np.argmin(distances_sq))
            distance_sq = float(distances_sq[local])
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_index = int(candidates[local])
        if best_index < 0:
            return None
        matches[source_ind] = best_index
    return matches


def parity_residual(values: np.ndarray, match_indices: np.ndarray, *, sign: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    matched = values[np.asarray(match_indices, dtype=np.int64)]
    residual = matched - float(sign) * values
    scale = max(float(np.ptp(values)), float(np.max(np.abs(values))) if values.size else 0.0, 1e-30)
    rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {
        "sign": float(sign),
        "normalized_rms": rms / scale,
        "normalized_max": max_abs / scale,
    }


def target_field_symmetry_audit(
    target_field: Any,
    plane_point: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    plane_normal: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
    *,
    coordinate_tolerance_m: float | None = None,
    field_relative_tolerance: float = 1e-6,
    component: str = "Bz",
    b0_direction: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Audit mirror symmetry of an imported or generated target field.

    The check verifies that reflected sample coordinates exist, that weights and
    suppression/group metadata are paired, and that the selected field component
    is either mirror-even or mirror-odd.
    """

    coords = np.asarray(target_field.coords, dtype=float)
    report: dict[str, Any] = {
        "target_symmetry_checked": True,
        "target_symmetry_ok": False,
        "target_symmetry_reason": "",
        "target_symmetry_sample_count": int(len(coords)),
        "target_symmetry_component": component,
    }
    if len(coords) == 0:
        report["target_symmetry_reason"] = "target field has no samples"
        return report

    reflected = reflect_points(coords, plane_point, plane_normal)
    match_indices = nearest_indices(reflected, coords)
    if match_indices is None:
        report["target_symmetry_reason"] = "target matching would be too expensive without scipy"
        return report

    distances = np.linalg.norm(reflected - coords[match_indices], axis=1)
    scale = geometry_scale(coords)
    tolerance = absolute_tolerance(1e-6, scale) if coordinate_tolerance_m is None else float(coordinate_tolerance_m)
    max_distance = float(np.max(distances)) if distances.size else float("inf")
    rms_distance = float(np.sqrt(np.mean(distances * distances))) if distances.size else float("inf")
    unique_match = len(np.unique(match_indices)) == len(match_indices)
    coordinate_ok = bool(unique_match and np.isfinite(max_distance) and max_distance <= tolerance)

    weights = np.asarray(getattr(target_field, "weights", np.ones(len(coords))), dtype=float)
    suppressed = np.asarray(getattr(target_field, "suppressed", np.zeros(len(coords), dtype=bool)), dtype=bool)
    groups = np.asarray(
        getattr(target_field, "target_field_group_inds", np.ones(len(coords), dtype=np.int64)), dtype=np.int64
    )
    weights_ok = bool(
        len(weights) == len(coords) and np.allclose(weights[match_indices], weights, rtol=1e-8, atol=tolerance)
    )
    suppressed_ok = bool(len(suppressed) == len(coords) and np.array_equal(suppressed[match_indices], suppressed))
    groups_ok = bool(len(groups) == len(coords) and np.array_equal(groups[match_indices], groups))

    resolved_component = normalize_field_component(component, allow_bmag=True)
    values = select_field_component_values(
        np.asarray(target_field.b, dtype=float),
        resolved_component,
        b0_direction=b0_direction,
    )
    if len(values) != len(coords):
        values = select_field_component_values(np.asarray(target_field.b, dtype=float), "Bz")
        resolved_component = "Bz"
    plus = parity_residual(values, match_indices, sign=1.0)
    minus = parity_residual(values, match_indices, sign=-1.0)
    selected = plus if plus["normalized_rms"] <= minus["normalized_rms"] else minus
    parity_ok = bool(selected["normalized_rms"] <= float(field_relative_tolerance))

    reasons: list[str] = []
    if not coordinate_ok:
        reasons.append("reflected target samples do not match")
    if not weights_ok:
        reasons.append("target weights are not symmetric")
    if not suppressed_ok:
        reasons.append("suppression flags are not symmetric")
    if not groups_ok:
        reasons.append("target groups are not symmetric")
    if not parity_ok:
        reasons.append(f"{resolved_component} is neither mirror-even nor mirror-odd within tolerance")

    report.update(
        {
            "target_symmetry_ok": bool(coordinate_ok and weights_ok and suppressed_ok and groups_ok and parity_ok),
            "target_symmetry_reason": "; ".join(reasons),
            "target_symmetry_component": resolved_component,
            "target_symmetry_coordinate_tolerance_m": float(tolerance),
            "target_symmetry_max_coordinate_error_m": max_distance,
            "target_symmetry_rms_coordinate_error_m": rms_distance,
            "target_symmetry_unique_mapping": bool(unique_match),
            "target_symmetry_weights_ok": weights_ok,
            "target_symmetry_suppressed_ok": suppressed_ok,
            "target_symmetry_groups_ok": groups_ok,
            "target_symmetry_parity": "even" if selected["sign"] > 0.0 else "odd",
            "target_symmetry_parity_sign": int(1 if selected["sign"] > 0.0 else -1),
            "target_symmetry_field_relative_tolerance": float(field_relative_tolerance),
            "target_symmetry_even_normalized_rms": float(plus["normalized_rms"]),
            "target_symmetry_even_normalized_max": float(plus["normalized_max"]),
            "target_symmetry_odd_normalized_rms": float(minus["normalized_rms"]),
            "target_symmetry_odd_normalized_max": float(minus["normalized_max"]),
            "target_symmetry_selected_normalized_rms": float(selected["normalized_rms"]),
            "target_symmetry_selected_normalized_max": float(selected["normalized_max"]),
            "target_symmetry_mirror_plane_point_m": [float(value) for value in np.asarray(plane_point, dtype=float)],
            "target_symmetry_mirror_plane_normal": [
                float(value) for value in np.asarray(plane_normal, dtype=float) / np.linalg.norm(plane_normal)
            ],
        }
    )
    return report
