from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .models import Mesh


@dataclass(frozen=True, slots=True)
class SuppressionClearanceReport:
    """Distance report for a generated suppression target surface.

    The clearance check is purely geometric. It verifies that generated
    zero-field target vertices are not closer than the configured clearance to
    the source conductor/support mesh used for the check.
    """

    status: str
    generator: str
    clearance_action: str
    clearance_method: str
    requested_clearance_m: float
    min_allowed_clearance_m: float
    min_clearance_m: float
    p01_clearance_m: float
    p05_clearance_m: float
    median_clearance_m: float
    max_clearance_m: float
    input_vertex_count: int
    input_face_count: int
    output_vertex_count: int
    output_face_count: int
    dropped_face_count: int
    violating_input_vertex_count: int
    violating_output_vertex_count: int
    source_vertex_count: int
    source_face_count: int
    warning: str = ""

    @classmethod
    def not_checked(cls) -> "SuppressionClearanceReport":
        nan = float("nan")
        return cls(
            status="not_checked",
            generator="none",
            clearance_action="none",
            clearance_method="none",
            requested_clearance_m=nan,
            min_allowed_clearance_m=nan,
            min_clearance_m=nan,
            p01_clearance_m=nan,
            p05_clearance_m=nan,
            median_clearance_m=nan,
            max_clearance_m=nan,
            input_vertex_count=0,
            input_face_count=0,
            output_vertex_count=0,
            output_face_count=0,
            dropped_face_count=0,
            violating_input_vertex_count=0,
            violating_output_vertex_count=0,
            source_vertex_count=0,
            source_face_count=0,
        )

    def as_dict(self, *, prefix: str = "suppression_clearance_") -> dict[str, Any]:
        return {
            f"{prefix}status": self.status,
            f"{prefix}generator": self.generator,
            f"{prefix}action": self.clearance_action,
            f"{prefix}method": self.clearance_method,
            f"{prefix}requested_m": self.requested_clearance_m,
            f"{prefix}min_allowed_m": self.min_allowed_clearance_m,
            f"{prefix}min_m": self.min_clearance_m,
            f"{prefix}p01_m": self.p01_clearance_m,
            f"{prefix}p05_m": self.p05_clearance_m,
            f"{prefix}median_m": self.median_clearance_m,
            f"{prefix}max_m": self.max_clearance_m,
            f"{prefix}input_vertex_count": self.input_vertex_count,
            f"{prefix}input_face_count": self.input_face_count,
            f"{prefix}output_vertex_count": self.output_vertex_count,
            f"{prefix}output_face_count": self.output_face_count,
            f"{prefix}dropped_face_count": self.dropped_face_count,
            f"{prefix}violating_input_vertex_count": self.violating_input_vertex_count,
            f"{prefix}violating_output_vertex_count": self.violating_output_vertex_count,
            f"{prefix}source_vertex_count": self.source_vertex_count,
            f"{prefix}source_face_count": self.source_face_count,
            f"{prefix}warning": self.warning,
        }


def apply_suppression_clearance(
    candidate_mesh: Mesh,
    source_meshes: Iterable[Mesh],
    *,
    requested_clearance_m: float,
    min_clearance_m: float,
    action: str = "none",
    generator: str = "mesh_topology",
    tolerance_m: float = 1.0e-9,
) -> Mesh:
    """Attach a clearance report and optionally drop violating target faces.

    ``action`` can be:

    - ``"none"``: return the candidate unchanged and mark clearance unchecked.
    - ``"report_only"``: measure distances but keep all candidate faces.
    - ``"filter_faces"``: remove every candidate face with at least one vertex
      closer than ``min_clearance_m`` to the source mesh.
    """

    mode = str(action).strip().lower()
    if mode in {"", "false"}:
        mode = "none"
    if mode not in {"none", "report_only", "filter_faces"}:
        raise ValueError("suppression clearance action must be 'none', 'report_only', or 'filter_faces'.")

    candidate = _compact_mesh(candidate_mesh)
    if mode == "none" or float(min_clearance_m) <= 0.0:
        report = SuppressionClearanceReport.not_checked()
        candidate.metadata["suppression_clearance_report"] = report.as_dict()
        return candidate

    source = _merge_meshes(list(source_meshes))
    input_distances = point_to_mesh_distances(candidate.vertices, source)
    min_allowed = float(min_clearance_m)
    allowed = input_distances + float(tolerance_m) >= min_allowed

    output = candidate
    dropped_face_count = 0
    warning = ""
    if mode == "filter_faces":
        keep_faces = np.all(allowed[candidate.faces], axis=1)
        dropped_face_count = int(np.count_nonzero(~keep_faces))
        if not np.any(keep_faces):
            raise ValueError(
                "Suppression clearance filtering removed every face. "
                "Reduce mesh_topology_suppression_min_clearance, increase the suppression separation, "
                "or use a less aggressive suppression source."
            )
        output = _mesh_from_selected_faces(candidate.vertices, candidate.faces[keep_faces], metadata=candidate.metadata)
        if dropped_face_count:
            warning = f"Filtered {dropped_face_count} suppression face(s) below the requested clearance."

    output_distances = point_to_mesh_distances(output.vertices, source)
    violating_output = int(np.count_nonzero(output_distances + float(tolerance_m) < min_allowed))
    report = _clearance_report_from_distances(
        input_distances=input_distances,
        output_distances=output_distances,
        candidate=candidate,
        output=output,
        source=source,
        requested_clearance_m=float(requested_clearance_m),
        min_allowed_clearance_m=min_allowed,
        action=mode,
        generator=generator,
        dropped_face_count=dropped_face_count,
        violating_input_vertex_count=int(np.count_nonzero(~allowed)),
        violating_output_vertex_count=violating_output,
        warning=warning,
    )
    output.metadata["suppression_clearance_report"] = report.as_dict()
    return output


def point_to_mesh_distances(points: np.ndarray, mesh: Mesh) -> np.ndarray:
    """Return nearest point-to-triangle distances for each point."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if mesh.n_faces == 0 or mesh.n_vertices == 0:
        return np.full(len(points), np.nan, dtype=float)
    triangles = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
    distances = np.empty(len(points), dtype=float)
    for index, point in enumerate(points):
        distances[index] = float(np.sqrt(np.min(_point_triangle_distance_squared(point, triangles))))
    return distances


def _clearance_report_from_distances(
    *,
    input_distances: np.ndarray,
    output_distances: np.ndarray,
    candidate: Mesh,
    output: Mesh,
    source: Mesh,
    requested_clearance_m: float,
    min_allowed_clearance_m: float,
    action: str,
    generator: str,
    dropped_face_count: int,
    violating_input_vertex_count: int,
    violating_output_vertex_count: int,
    warning: str,
) -> SuppressionClearanceReport:
    distances = np.asarray(output_distances, dtype=float)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        values = (float("nan"),) * 5
    else:
        values = tuple(float(v) for v in np.percentile(finite, [0, 1, 5, 50, 100]))
    if violating_output_vertex_count:
        status = "WARNING" if action == "report_only" else "FAIL"
    else:
        status = "PASS"
    return SuppressionClearanceReport(
        status=status,
        generator=generator,
        clearance_action=action,
        clearance_method="point_to_triangle_surface",
        requested_clearance_m=float(requested_clearance_m),
        min_allowed_clearance_m=float(min_allowed_clearance_m),
        min_clearance_m=values[0],
        p01_clearance_m=values[1],
        p05_clearance_m=values[2],
        median_clearance_m=values[3],
        max_clearance_m=values[4],
        input_vertex_count=int(candidate.n_vertices),
        input_face_count=int(candidate.n_faces),
        output_vertex_count=int(output.n_vertices),
        output_face_count=int(output.n_faces),
        dropped_face_count=int(dropped_face_count),
        violating_input_vertex_count=int(violating_input_vertex_count),
        violating_output_vertex_count=int(violating_output_vertex_count),
        source_vertex_count=int(source.n_vertices),
        source_face_count=int(source.n_faces),
        warning=str(warning),
    )


def _point_triangle_distance_squared(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Vectorized squared distance from one point to many triangles."""

    p = np.asarray(point, dtype=float)
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]
    ab = b - a
    ac = c - a

    dist2 = np.full(len(triangles), np.inf, dtype=float)
    remaining = np.ones(len(triangles), dtype=bool)

    ap = p[None, :] - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    _assign_region_distance(dist2, remaining, (d1 <= 0.0) & (d2 <= 0.0), ap)

    bp = p[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    _assign_region_distance(dist2, remaining, (d3 >= 0.0) & (d4 <= d3), bp)

    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denom = d1 - d3
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(np.abs(denom) > 0.0, d1 / denom, 0.0)
    projection_delta = ap - v[:, None] * ab
    _assign_region_distance(dist2, remaining, mask, projection_delta)

    cp = p[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    _assign_region_distance(dist2, remaining, (d6 >= 0.0) & (d5 <= d6), cp)

    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denom = d2 - d6
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(denom) > 0.0, d2 / denom, 0.0)
    projection_delta = ap - w[:, None] * ac
    _assign_region_distance(dist2, remaining, mask, projection_delta)

    va = d3 * d6 - d5 * d4
    bc = c - b
    mask = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    denom = (d4 - d3) + (d5 - d6)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(denom) > 0.0, (d4 - d3) / denom, 0.0)
    projection_delta = bp - w[:, None] * bc
    _assign_region_distance(dist2, remaining, mask, projection_delta)

    inside = remaining
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(np.abs(denom) > 0.0, vb / denom, 0.0)
        w = np.where(np.abs(denom) > 0.0, vc / denom, 0.0)
    projection = a + v[:, None] * ab + w[:, None] * ac
    delta = p[None, :] - projection
    _assign_region_distance(dist2, remaining, inside, delta)

    invalid = ~np.isfinite(dist2)
    if np.any(invalid):
        vertex_distances = np.minimum(
            np.einsum("ij,ij->i", ap, ap),
            np.minimum(np.einsum("ij,ij->i", bp, bp), np.einsum("ij,ij->i", cp, cp)),
        )
        dist2[invalid] = vertex_distances[invalid]
    return dist2


def _assign_region_distance(dist2: np.ndarray, remaining: np.ndarray, mask: np.ndarray, delta: np.ndarray) -> None:
    active = np.asarray(mask, dtype=bool) & remaining
    if not np.any(active):
        return
    dist2[active] = np.einsum("ij,ij->i", delta[active], delta[active])
    remaining[active] = False


def _compact_mesh(mesh: Mesh) -> Mesh:
    if mesh.n_faces == 0:
        return Mesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), metadata=dict(mesh.metadata))
    return _mesh_from_selected_faces(mesh.vertices, mesh.faces, metadata=mesh.metadata)


def _mesh_from_selected_faces(
    vertices: np.ndarray, faces: np.ndarray, *, metadata: dict[str, Any] | None = None
) -> Mesh:
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("Cannot create a mesh from zero selected faces.")
    used_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return Mesh(
        vertices=np.asarray(vertices, dtype=float)[used_vertices],
        faces=inverse.reshape(faces.shape),
        metadata={} if metadata is None else dict(metadata),
    )


def _merge_meshes(meshes: list[Mesh]) -> Mesh:
    if not meshes:
        raise ValueError("At least one source mesh is required for suppression clearance checking.")
    vertices = []
    faces = []
    offset = 0
    for mesh in meshes:
        vertices.append(np.asarray(mesh.vertices, dtype=float))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        offset += mesh.n_vertices
    return Mesh(vertices=np.vstack(vertices), faces=np.vstack(faces))
