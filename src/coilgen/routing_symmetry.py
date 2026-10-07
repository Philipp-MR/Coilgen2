"""Conservative symmetry audit for prioritising joint-routing proposals.

These checks describe the actual discretised support and directed source loops.
They do not certify symmetry of cuts, obstacles, mounted profiles or the field
evaluation domain, and therefore never permit skipping physical validation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .models import CoilPart, ContourLine, Mesh
from .symmetry import geometry_scale, nearest_indices


def audit_routing_symmetry(
    part: CoilPart,
    *,
    geometry_rtol: float = 1e-8,
    geometry_atol_m: float = 1e-10,
    sf_rtol: float = 1e-6,
    sf_atol_A: float = 1e-10,
) -> dict[str, Any]:
    """Check world-axis mirrors and half turns about the mesh centroid.

    SF parity is fitted as ``psi(T x) = sign * psi(x) + offset`` using both
    signs, with maximum residual and a gauge-independent range scale. Surface
    current transforms as ``J(T x) = sign * det(T) * T J(x)`` when the oriented
    support normals transform as ``n(T x) = T n(x)``. The loop check uses this
    direction, not just an unordered point-cloud match.

    A verified transform is a proposal-priority hint. Even exact source
    symmetry does not prove that routing cuts or fabrication constraints share
    it. All ordinary acceptance checks remain necessary.
    """
    tolerances = (geometry_rtol, geometry_atol_m, sf_rtol, sf_atol_A)
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("Routing symmetry tolerances must be finite and non-negative.")
    vertices = np.asarray(part.mesh.vertices, dtype=float)
    report: dict[str, Any] = {
        "checked": False,
        "verified_transform_count": 0,
        "use": "topology_priority_only",
        "routing_reuse_allowed": False,
        "transforms": [],
    }
    if not len(vertices) or not np.all(np.isfinite(vertices)):
        report["reason"] = "Support vertices are empty or non-finite."
        return report
    center = np.mean(vertices, axis=0)
    geometry_tol = max(float(geometry_atol_m), float(geometry_rtol) * geometry_scale(vertices))
    sf = part.stream_function if part.stream_function is not None else part.mesh.stream_function
    # Routing represents real directed series currents. A complex RF mode
    # needs its own phase convention; dropping its imaginary SF cannot certify it.
    sf_values = None if sf is None or np.iscomplexobj(sf) else np.asarray(sf, dtype=float).reshape(-1)
    valid_sf = sf_values is not None and len(sf_values) == len(vertices) and np.all(np.isfinite(sf_values))
    sf_tol = max(float(sf_atol_A), float(sf_rtol) * float(np.ptp(sf_values))) if valid_sf else float(sf_atol_A)
    current_check = _source_current_check(part, sf_values if valid_sf else None)
    names = ("mirror_x", "mirror_y", "mirror_z", "half_turn_x", "half_turn_y", "half_turn_z")
    diagonals = ((-1, 1, 1), (1, -1, 1), (1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    for name, diagonal in zip(names, diagonals):
        matrix = np.diag(diagonal).astype(float)
        check, vertex_map, face_map = _mesh_check(part.mesh, matrix, center, geometry_tol)
        check.update({"name": name, "matrix": matrix.tolist(), "center_m": center.tolist()})
        sf_check = _sf_check(sf_values if valid_sf else None, vertex_map, sf_tol)
        check["stream_function"] = sf_check
        contour_check = _contour_check(part, matrix, center, geometry_tol, sf_tol, sf_check, vertex_map, face_map)
        contour_check["sf_current_direction"] = current_check
        if current_check["status"] == "failed":
            contour_check.update(
                status="failed", reason="Source contour direction disagrees with the nodal SF current."
            )
        check["contours"] = contour_check
        check["verified"] = (
            all(item["status"] == "verified" for item in (check["mesh"], sf_check, contour_check))
            and current_check["status"] == "verified"
        )
        check["group_permutation"] = (
            _group_permutation(part, contour_check["permutation"]) if check["verified"] else None
        )
        report["transforms"].append(check)
    report.update(
        {
            "checked": True,
            "geometry_tolerance_m": geometry_tol,
            "stream_function_tolerance_A": sf_tol,
            "verified_transform_count": sum(check["verified"] for check in report["transforms"]),
        }
    )
    return report


def _residual_status(error: float, tolerance: float) -> str:
    if error <= tolerance:
        return "verified"
    return "approximate" if error <= 100.0 * tolerance else "failed"


def _mesh_check(
    mesh: Mesh, matrix: np.ndarray, center: np.ndarray, tolerance: float
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    vertices, faces = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)
    result: dict[str, Any] = {
        "status": "failed",
        "vertex_bijection": False,
        "faces_ok": False,
        "boundaries_ok": False,
        "oriented_normals_ok": False,
    }
    check = {"mesh": result}
    transformed = (vertices - center) @ matrix.T + center
    mapping = nearest_indices(transformed, vertices, max_total_work=4_000_000)
    if mapping is None:
        result.update(status="not_checked", reason="Vertex matching exceeds the available spatial-search budget.")
        return check, None, None
    error = float(np.max(np.linalg.norm(transformed - vertices[mapping], axis=1), initial=0.0))
    unique = len(np.unique(mapping)) == len(vertices)
    involution = unique and bool(np.array_equal(mapping[mapping], np.arange(len(vertices))))
    result.update(max_vertex_error_m=error, vertex_bijection=bool(involution))
    if not involution or error > tolerance:
        result["status"] = _residual_status(error, tolerance) if involution else "failed"
        result["reason"] = "Transformed vertices do not form a bijective symmetry within tolerance."
        return check, mapping if involution else None, None
    if not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        result["reason"] = "Triangular support faces are empty or invalid."
        return check, mapping, None
    keys = [tuple(sorted(face)) for face in faces]
    lookup = {key: index for index, key in enumerate(keys)}
    mapped_keys = [tuple(sorted(face)) for face in mapping[faces]]
    if len(lookup) != len(faces) or set(mapped_keys) != set(keys):
        result["reason"] = "Vertex symmetry does not preserve the actual triangle connectivity."
        return check, mapping, None
    face_map = np.asarray([lookup[key] for key in mapped_keys], dtype=np.int64)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= np.finfo(float).tiny):
        result["reason"] = "Support contains a degenerate triangle."
        return check, mapping, face_map
    normals /= lengths[:, None]
    normal_error = float(np.max(np.linalg.norm(normals[face_map] - normals @ matrix.T, axis=1)))
    for stored, permutation in ((mesh.face_normals, face_map), (mesh.vertex_normals, mapping)):
        if stored is None:
            continue
        stored = np.asarray(stored, dtype=float)
        if len(stored) != len(permutation) or not np.all(np.isfinite(stored)):
            normal_error = float("inf")
            break
        normal_error = max(normal_error, float(np.max(np.linalg.norm(stored[permutation] - stored @ matrix.T, axis=1))))
    boundaries = _boundary_edges(faces)
    mapped_boundaries = Counter(tuple(sorted((int(mapping[a]), int(mapping[b])))) for a, b in boundaries)
    valid_boundaries = all(
        np.all((np.asarray(boundary) >= 0) & (np.asarray(boundary) < len(vertices))) for boundary in mesh.boundaries
    )
    explicit = Counter(_boundary_key(boundary) for boundary in mesh.boundaries)
    mapped_explicit = (
        Counter(_boundary_key(mapping[np.asarray(boundary, dtype=np.int64)]) for boundary in mesh.boundaries)
        if valid_boundaries
        else None
    )
    boundary_ok = valid_boundaries and mapped_boundaries == Counter(boundaries) and explicit == mapped_explicit
    result.update(
        faces_ok=True,
        boundaries_ok=boundary_ok,
        oriented_normals_ok=bool(normal_error <= 1e-6),
        max_normal_error=normal_error,
    )
    if boundary_ok and normal_error <= 1e-6:
        result["status"] = "verified"
    else:
        result["reason"] = "Boundary loops or oriented support normals break this symmetry."
    return check, mapping, face_map


def _boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges = Counter(tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))) for face in faces for i in range(3))
    return [edge for edge, count in edges.items() if count == 1]


def _boundary_key(boundary: np.ndarray) -> tuple[tuple[int, int], ...]:
    indices = np.asarray(boundary, dtype=np.int64).reshape(-1)
    if len(indices) > 1 and indices[-1] == indices[0]:
        indices = indices[:-1]
    return tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in zip(indices, np.roll(indices, -1))))


def _sf_check(values: np.ndarray | None, mapping: np.ndarray | None, tolerance: float) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "not_checked"}
    if values is None or mapping is None:
        result["reason"] = "A finite nodal SF and a vertex bijection are required."
        return result
    residuals = []
    for sign in (1, -1):
        offset = float(np.mean(values[mapping] - sign * values))
        residual = values[mapping] - sign * values - offset
        residuals.append((float(np.max(np.abs(residual))), float(np.sqrt(np.mean(residual * residual))), sign, offset))
    error, rms, sign, offset = min(residuals)
    result.update(
        status=_residual_status(error, tolerance),
        parity=sign,
        offset_A=offset,
        max_residual_A=error,
        rms_residual_A=rms,
    )
    if float(np.ptp(values)) <= tolerance:
        result.update(status="not_checked", reason="Constant SF has no resolved current-direction parity.")
    return result


def _contour_check(
    part: CoilPart,
    matrix: np.ndarray,
    center: np.ndarray,
    geometry_tol: float,
    sf_tol: float,
    sf_check: dict[str, Any],
    vertex_map: np.ndarray | None,
    face_map: np.ndarray | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_checked",
        "permutation": None,
        "current_direction_sign": None,
        "provenance_status": "not_available",
    }
    if "parity" not in sf_check or not part.contour_lines:
        result["reason"] = "Resolved SF parity and nonempty source contours are required."
        return result
    loops = part.contour_lines
    if any(
        len(loop.v) < 4
        or not np.all(np.isfinite(loop.v))
        or np.linalg.norm(loop.v[-1] - loop.v[0]) > geometry_tol
        or not np.isfinite(loop.potential)
        for loop in loops
    ):
        result.update(status="failed", reason="Source contours must be finite closed loops with finite levels.")
        return result
    if any(
        not np.isfinite(loop.current_orientation) or abs(float(loop.current_orientation) - 1) > 1e-12 for loop in loops
    ):
        result.update(
            status="failed", reason="Routing requires canonical directed contours with current_orientation=1."
        )
        return result
    points = [_loop_points(loop.v, geometry_tol) for loop in loops]
    parity, offset = int(sf_check["parity"]), float(sf_check["offset_A"])
    direction = parity * int(round(np.linalg.det(matrix)))
    permutation, directions = [], []
    worst_error = 0.0
    provenance_seen = False
    for index, loop in enumerate(loops):
        expected = (points[index] - center) @ matrix.T + center
        eligible = [
            j
            for j, other in enumerate(loops)
            if other.turn_multiplicity == loop.turn_multiplicity
            and abs(float(other.potential) - (parity * float(loop.potential) + offset)) <= sf_tol
        ]
        matches = []
        for j in eligible:
            if len(points[j]) != len(expected) or len(expected) < 3:
                continue
            point_map = nearest_indices(expected, points[j], max_total_work=1_000_000)
            if point_map is None or len(np.unique(point_map)) != len(expected):
                continue
            error = float(np.max(np.linalg.norm(expected - points[j][point_map], axis=1)))
            if error > geometry_tol:
                continue
            expected_direction = direction
            if not np.all((np.roll(point_map, -1) - point_map) % len(point_map) == expected_direction % len(point_map)):
                continue
            provenance, seen = _provenance_match(
                part.mesh, loop, loops[j], point_map, vertex_map, face_map, geometry_tol
            )
            provenance_seen |= seen
            if provenance:
                matches.append((j, error, expected_direction))
        if len(matches) != 1:
            result.update(
                status="failed",
                reason=f"Contour {index} has no unique directed, level and provenance match.",
                matched_count=len(permutation),
            )
            return result
        match, error, current_sign = matches[0]
        permutation.append(match)
        directions.append(current_sign)
        worst_error = max(worst_error, error)
    if len(set(permutation)) != len(loops) or any(permutation[permutation[i]] != i for i in range(len(loops))):
        result.update(status="failed", reason="Contour matching is not an involutive bijection.")
        return result
    result.update(
        status="verified",
        permutation=permutation,
        current_direction_sign=directions,
        max_point_error_m=worst_error,
        provenance_status="verified" if provenance_seen else "not_available",
    )
    return result


def _loop_points(points: np.ndarray, tolerance: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points[:-1] if len(points) > 1 and np.linalg.norm(points[-1] - points[0]) <= tolerance else points


def _provenance_match(
    mesh: Mesh,
    source: ContourLine,
    target: ContourLine,
    point_map: np.ndarray,
    vertex_map: np.ndarray | None,
    face_map: np.ndarray | None,
    tolerance: float,
) -> tuple[bool, bool]:
    attributes = (
        "surface_faces",
        "surface_barycentric",
        "surface_edge_vertices",
        "surface_edge_t",
        "segment_faces",
        "segment_barycentric",
    )
    seen = any(getattr(loop, name) is not None for loop in (source, target) for name in attributes)
    if not seen:
        return True, False
    if vertex_map is None or face_map is None:
        return False, True
    count = len(point_map)
    for name in attributes:
        if (getattr(source, name) is None) != (getattr(target, name) is None):
            return False, True
    for loop in (source, target):
        if not _valid_attachment(mesh, loop, tolerance):
            return False, True
    if source.surface_edge_vertices is not None:
        mapped_edges = np.sort(vertex_map[np.asarray(source.surface_edge_vertices[:count], dtype=np.int64)], axis=1)
        target_edges = np.sort(np.asarray(target.surface_edge_vertices, dtype=np.int64)[point_map], axis=1)
        if not np.array_equal(mapped_edges, target_edges):
            return False, True
    if source.segment_faces is not None:
        reverse = (int(point_map[1]) - int(point_map[0])) % count != 1
        segment_map = (point_map - int(reverse)) % count
        if len(source.segment_faces) > count or np.any(
            segment_map[: len(source.segment_faces)] >= len(target.segment_faces)
        ):
            return False, True
        if not np.array_equal(
            face_map[source.segment_faces], target.segment_faces[segment_map[: len(source.segment_faces)]]
        ):
            return False, True
    return True, True


def _valid_attachment(mesh: Mesh, loop: ContourLine, tolerance: float) -> bool:
    if not _attachment_shapes_valid(loop):
        return False
    if loop.surface_edge_vertices is not None:
        indices = np.asarray(loop.surface_edge_vertices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(mesh.vertices)):
            return False
        edges = mesh.vertices[indices]
        vectors = edges[:, 1] - edges[:, 0]
        lengths2 = np.sum(vectors * vectors, axis=1)
        if np.any(lengths2 <= np.finfo(float).tiny):
            return False
        fractions = (
            np.sum((loop.v - edges[:, 0]) * vectors, axis=1) / lengths2
            if loop.surface_edge_t is None
            else np.asarray(loop.surface_edge_t, dtype=float)
        )
        reconstructed = edges[:, 0] + fractions[:, None] * vectors
        if (
            not np.all(np.isfinite(fractions))
            or np.any(fractions < -1e-8)
            or np.any(fractions > 1 + 1e-8)
            or np.max(np.linalg.norm(reconstructed - loop.v, axis=1), initial=0.0) > tolerance
        ):
            return False
    elif loop.surface_edge_t is not None:
        return False
    # Point faces can legitimately differ at shared triangle edges. Check
    # their actual attachment instead of requiring an arbitrary point face ID.
    for faces, barycentric, points in (
        (loop.surface_faces, loop.surface_barycentric, loop.v),
        (loop.segment_faces, loop.segment_barycentric, np.stack((loop.v[:-1], loop.v[1:]), axis=1)),
    ):
        if faces is None:
            if barycentric is not None:
                return False
            continue
        if np.any(faces < 0) or np.any(faces >= len(mesh.faces)):
            return False
        triangles = mesh.vertices[mesh.faces[faces]]
        if points.ndim == 3:
            triangles = triangles[:, None]
        if barycentric is None:
            a, b = triangles[..., 1, :] - triangles[..., 0, :], triangles[..., 2, :] - triangles[..., 0, :]
            delta = points - triangles[..., 0, :]
            aa, ab, bb = np.sum(a * a, axis=-1), np.sum(a * b, axis=-1), np.sum(b * b, axis=-1)
            determinant = aa * bb - ab * ab
            if np.any(determinant <= np.finfo(float).tiny):
                return False
            da, db = np.sum(delta * a, axis=-1), np.sum(delta * b, axis=-1)
            v, w = (bb * da - ab * db) / determinant, (aa * db - ab * da) / determinant
            barycentric = np.stack((1 - v - w, v, w), axis=-1)
        reconstructed = np.einsum("...j,...jk->...k", barycentric, triangles)
        if (
            not np.all(np.isfinite(barycentric))
            or np.any(barycentric < -1e-8)
            or np.any(barycentric > 1 + 1e-8)
            or np.max(np.abs(np.sum(barycentric, axis=-1) - 1), initial=0.0) > 1e-8
            or np.max(np.linalg.norm(reconstructed - points, axis=-1), initial=0.0) > tolerance
        ):
            return False
    return True


def _attachment_shapes_valid(loop: ContourLine) -> bool:
    n, segments = len(loop.v), max(len(loop.v) - 1, 0)
    expected = {
        "surface_faces": (n,),
        "surface_barycentric": (n, 3),
        "surface_edge_vertices": (n, 2),
        "surface_edge_t": (n,),
        "segment_faces": (segments,),
        "segment_barycentric": (segments, 2, 3),
    }
    for name, shape in expected.items():
        value = getattr(loop, name)
        if value is None:
            continue
        value = np.asarray(value)
        if value.shape != shape or value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
            return False
        if name in {"surface_faces", "surface_edge_vertices", "segment_faces"} and value.dtype.kind not in "iu":
            return False
    return True


def _source_current_check(part: CoilPart, sf: np.ndarray | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_checked",
        "checked_segment_count": 0,
        "checked_contour_count": 0,
        "unresolved_segment_count": 0,
    }
    loops = [loop for loop in part.contour_lines if loop.segment_faces is not None]
    if sf is None or not loops:
        return result
    faces = np.asarray(part.mesh.faces, dtype=np.int64)
    if not len(faces) or np.any(faces < 0) or np.any(faces >= len(sf)):
        return result
    triangles = part.mesh.vertices[faces]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    normal = np.cross(a, b)
    aa, ab, bb = np.sum(a * a, axis=1), np.sum(a * b, axis=1), np.sum(b * b, axis=1)
    determinant = aa * bb - ab * ab
    if np.any(determinant <= np.finfo(float).tiny):
        return result
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    values = sf[faces]
    gradient = ((values[:, 1] - values[:, 0]) * bb - (values[:, 2] - values[:, 0]) * ab)[:, None] * a + (
        (values[:, 2] - values[:, 0]) * aa - (values[:, 1] - values[:, 0]) * ab
    )[:, None] * b
    # Match calculate_basis_functions: CoilGen uses J = grad(psi) x n.
    current = np.cross(gradient / determinant[:, None], normal)
    worst_alignment = 1.0
    for loop in loops:
        if not _attachment_shapes_valid(loop):
            result.update(status="failed", reason="Contour attachment shapes or values are invalid.")
            return result
        if not np.isfinite(loop.current_orientation) or abs(float(loop.current_orientation) - 1) > 1e-12:
            result.update(status="failed", reason="Source contour current_orientation is not canonical.")
            return result
        indices = np.asarray(loop.segment_faces, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(current)):
            result.update(status="failed", reason="Contour segment-face provenance is invalid.")
            return result
        tangents = np.diff(loop.v, axis=0)
        local_current = current[indices]
        scale = np.linalg.norm(tangents, axis=1) * np.linalg.norm(local_current, axis=1)
        resolved = scale > np.finfo(float).tiny
        result["unresolved_segment_count"] += int(
            np.count_nonzero((np.linalg.norm(tangents, axis=1) > np.finfo(float).tiny) & ~resolved)
        )
        alignment = np.sum(tangents[resolved] * local_current[resolved], axis=1) / scale[resolved]
        worst_alignment = min(worst_alignment, float(np.min(alignment, initial=1.0)))
        result["checked_segment_count"] += int(np.count_nonzero(resolved))
        result["checked_contour_count"] += 1
    result.update(status="failed" if worst_alignment < -1e-8 else "verified", min_cosine_alignment=worst_alignment)
    if result["status"] != "failed" and (
        worst_alignment <= 1e-8
        or not result["checked_segment_count"]
        or result["unresolved_segment_count"]
        or len(loops) != len(part.contour_lines)
    ):
        result["status"] = "not_checked"
    return result


def _group_permutation(part: CoilPart, contour_map: list[int]) -> list[int] | None:
    memberships = [tuple(sorted(group.loop_indices)) for group in part.groups]
    if not memberships or any(not indices for indices in memberships):
        return None
    lookup = {indices: index for index, indices in enumerate(memberships)}
    if len(lookup) != len(memberships):
        return None
    mapped = []
    for indices in memberships:
        if any(index < 0 or index >= len(contour_map) for index in indices):
            return None
        counterpart = lookup.get(tuple(sorted(contour_map[index] for index in indices)))
        if counterpart is None:
            return None
        mapped.append(counterpart)
    return mapped if len(set(mapped)) == len(mapped) else None
