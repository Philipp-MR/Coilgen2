"""Whole-body certificates for actual convex mesh caps with one planar rim."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .group_route_convex_shell import _boundary_cycles, certify_convex_body


@dataclass(frozen=True)
class ConvexCapSupport:
    plane_points: np.ndarray
    plane_normals: np.ndarray
    rim_origin: np.ndarray
    rim_normal: np.ndarray  # points out of the closed solid, through its missing base


def convex_mesh_cap(mesh, tolerance, maximum_work=50_000_000):
    """Verify a complete outward convex disk from its triangles, never metadata.

    The caller checks oriented edge-manifoldness. Supporting planes, a single
    convex planar rim, disk topology and closed convex-hull area independently
    exclude concavity, missing patches, inward faces and duplicate coverage.
    """
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    cycles = _boundary_cycles(faces)
    if cycles is None or len(cycles) != 1:
        return None, "unsupported_convex_cap_boundary"
    rim = vertices[cycles[0]]
    if len(rim) < 3 or len(faces) * len(vertices) > maximum_work:
        return None, "support_face_edge_work_budget_exceeded"
    origin = rim.mean(axis=0)
    area_vector = np.cross(rim - origin, np.roll(rim, -1, axis=0) - origin).sum(axis=0) / 2
    area = float(np.linalg.norm(area_vector))
    if area <= tolerance**2:
        return None, "degenerate_convex_cap_rim"
    normal = area_vector / area
    if (vertices.mean(axis=0) - origin) @ normal > 0:
        normal = -normal
    signed = (vertices - origin) @ normal
    if np.max(np.abs((rim - origin) @ normal)) > tolerance or signed.max() > tolerance:
        return None, "nonplanar_or_inward_convex_cap_rim"
    if signed.min() >= -tolerance:
        return None, "degenerate_convex_cap"
    # Every rim edge must bound the same convex polygon.
    edges = np.roll(rim, -1, axis=0) - rim
    inward = np.cross(area_vector / area, edges)
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    if np.any(np.einsum("pij,ij->pi", rim[:, None] - rim, inward) < -tolerance):
        return None, "nonconvex_cap_rim"
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / twice_area[:, None]
    for start in range(0, len(vertices), 256):
        heights = np.einsum("pij,ij->pi", vertices[start : start + 256, None] - triangles[:, 0], normals)
        if np.any(heights > tolerance):
            return None, "unsupported_nonconvex_support"
    mesh_edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique_edges = np.unique(mesh_edges, axis=0)
    if len(np.unique(faces)) != len(vertices) or len(vertices) - len(unique_edges) + len(faces) != 1:
        return None, "uncertified_convex_cap_topology"
    # Check connectedness explicitly; Euler characteristic alone is insufficient.
    connected = {0}
    adjacency = [[] for _ in vertices]
    for a, b in unique_edges:
        adjacency[a].append(int(b))
        adjacency[b].append(int(a))
    pending = [0]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in connected:
                connected.add(neighbor)
                pending.append(neighbor)
    if len(connected) != len(vertices):
        return None, "uncertified_convex_cap_topology"
    try:
        hull = ConvexHull(vertices)
    except QhullError:
        return None, "uncertified_convex_cap_hull"
    perimeter = np.linalg.norm(edges, axis=1).sum()
    if abs(hull.area - (twice_area.sum() / 2 + area)) > max(tolerance * perimeter, hull.area * 1e-10):
        return None, "uncertified_convex_cap_coverage"
    return ConvexCapSupport(np.vstack((triangles[:, 0], origin)), np.vstack((normals, normal)), origin, normal), ""


def certify_cap_body(cap, points, faces, mode, tolerance, maximum_work, diagnostics):
    margin = -(points - cap.rim_origin) @ cap.rim_normal
    diagnostics["body_cap_rim_margin_m"] = float(margin.min())
    if margin.min() < -tolerance:
        return "body_support_domain_violation"
    return certify_convex_body(
        cap.plane_points, cap.plane_normals, points, faces, mode, tolerance, maximum_work, diagnostics
    )
