"""Whole-triangle certificates for verified circular polygonal support shells.

The physical support is the supplied triangulation, not an ideal radius inferred
from metadata. Only complete convex cylindrical/frustum shells with two rims
are recognized. Exterior certification clips complete body triangles against
the shell's solid halfspaces; vertices alone cannot prove exterior clearance.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConvexSupportShell:
    axis: np.ndarray
    origin: np.ndarray
    axial_length: float
    plane_points: np.ndarray
    plane_normals: np.ndarray
    bottom_radius: float
    top_radius: float


def _boundary_cycles(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse] == 1]
    following = dict(map(tuple, boundary))
    remaining, cycles = set(following), []
    while remaining:
        initial = min(remaining)
        current, cycle = initial, []
        while current in remaining:
            remaining.remove(current)
            cycle.append(current)
            current = int(following[current])
        if current != initial:
            return []
        cycles.append(np.asarray(cycle, dtype=int))
    return cycles


def _corners(polygon, tolerance):
    before = polygon - np.roll(polygon, 1, axis=0)
    after = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    if np.any(lengths == 0):
        return None
    # Exact subdivision samples along a rim do not change its polygon.
    distance = np.linalg.norm(np.cross(before, after), axis=1) / np.linalg.norm(before, axis=1)
    keep = (distance > tolerance) | (np.einsum("ij,ij->i", before, after) <= 0)
    return polygon[keep]


def convex_circular_shell(mesh, tolerance, *, maximum_work=50_000_000):
    """Prove actual rim geometry, side-plane coverage and outward orientation.

    The caller first verifies finite, coherently oriented edge-manifold faces.
    Face-order, vertex-order and collinear mesh subdivision are immaterial.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    cycles = _boundary_cycles(faces)
    if len(cycles) != 2:
        return None, "unsupported_curved_support_rims"
    bottom, top = (_corners(vertices[cycle], tolerance) for cycle in cycles)
    if bottom is None or top is None or len(bottom) < 3 or len(bottom) != len(top) or len(bottom) > 2048:
        return None, "unsupported_curved_support_rims"
    count = len(bottom)
    work = count * (len(vertices) + 4 * len(faces))
    if work > maximum_work:
        return None, "support_face_edge_work_budget_exceeded"
    origin, top_center = bottom.mean(axis=0), top.mean(axis=0)
    axis = top_center - origin
    length = float(np.linalg.norm(axis))
    if length <= tolerance:
        return None, "unsupported_curved_support_rims"
    axis /= length
    radial_bottom, radial_top = bottom - origin, top - top_center
    if max(np.max(np.abs(radial_bottom @ axis)), np.max(np.abs(radial_top @ axis))) > tolerance:
        return None, "unsupported_curved_support_rims"
    radii_bottom, radii_top = np.linalg.norm(radial_bottom, axis=1), np.linalg.norm(radial_top, axis=1)
    radius_bottom, radius_top = float(radii_bottom.mean()), float(radii_top.mean())
    if (
        min(radius_bottom, radius_top) <= tolerance
        or max(np.max(np.abs(radii_bottom - radius_bottom)), np.max(np.abs(radii_top - radius_top))) > tolerance
    ):
        return None, "unsupported_non_circular_support_rims"
    unit_bottom, unit_top = radial_bottom / radii_bottom[:, None], radial_top / radii_top[:, None]
    # Establish correspondence through actual radial directions, not metadata.
    match = np.argmax(unit_bottom @ unit_top.T, axis=1)
    if (
        len(np.unique(match)) != count
        or np.max(np.linalg.norm(unit_bottom - unit_top[match], axis=1)) * max(radius_bottom, radius_top) > tolerance
    ):
        return None, "unsupported_nonhomothetic_support_rims"
    top = top[match]
    edge = np.roll(bottom, -1, axis=0) - bottom
    normal = np.cross(edge, top - bottom)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    orientation = np.einsum("ij,ij->i", normal, radial_bottom)
    if np.any(np.abs(orientation) <= tolerance):
        return None, "unsupported_curved_support_rims"
    normal *= np.where(orientation > 0, 1, -1)[:, None]
    # Every rim must be convex; every vertex must lie in the finite solid.
    axial = (vertices - origin) @ axis
    if axial.min() < -tolerance or axial.max() > length + tolerance:
        return None, "uncertified_curved_support_coverage"
    for offset in range(0, len(vertices), 512):
        signed = np.einsum("pij,ij->pi", vertices[offset : offset + 512, None] - bottom, normal)
        if np.any(signed > tolerance):
            return None, "unsupported_nonconvex_support"
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas2 = np.linalg.norm(cross, axis=1)
    face_normals = cross / areas2[:, None]
    accumulated = np.zeros(count)
    for offset in range(0, len(faces), 512):
        local = triangles[offset : offset + 512]
        signed = np.einsum("pfij,ij->pfi", local[:, :, None] - bottom, normal)
        on_plane = np.max(np.abs(signed), axis=1) <= tolerance
        oriented = face_normals[offset : offset + 512] @ normal.T >= 1 - 1e-10
        eligible = on_plane & oriented
        if np.any(eligible.sum(axis=1) != 1):
            return None, "uncertified_curved_support_face_planes"
        selected = np.argmax(eligible, axis=1)
        np.add.at(accumulated, selected, areas2[offset : offset + 512] / 2)
    quadrilateral_area = 0.5 * (
        np.linalg.norm(np.cross(edge, top - bottom), axis=1)
        + np.linalg.norm(np.cross(np.roll(top, -1, axis=0) - top, bottom - top), axis=1)
    )
    if np.any(
        np.abs(accumulated - quadrilateral_area)
        > np.maximum(
            tolerance * (np.linalg.norm(edge, axis=1) + np.linalg.norm(top - bottom, axis=1)),
            quadrilateral_area * 1e-10,
        )
    ):
        return None, "uncertified_curved_support_coverage"
    # Connected annulus Euler characteristic rejects unused/disconnected input.
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    if len(np.unique(faces)) != len(vertices) or len(vertices) - len(np.unique(edges, axis=0)) + len(faces) != 0:
        return None, "uncertified_curved_support_topology"
    return ConvexSupportShell(axis, origin, length, bottom, normal, radius_bottom, radius_top), ""


def _clip_polygon(polygon, point, normal, threshold):
    signed = (polygon - point) @ normal - threshold
    inside = signed <= 0
    if np.all(inside):
        return polygon
    if not np.any(inside):
        return np.empty((0, 3))
    clipped = []
    for index in range(len(polygon)):
        previous = index - 1
        if inside[previous] != inside[index]:
            fraction = signed[previous] / (signed[previous] - signed[index])
            clipped.append(polygon[previous] + fraction * (polygon[index] - polygon[previous]))
        if inside[index]:
            clipped.append(polygon[index])
    return np.asarray(clipped)


def certify_shell_body(shell, body_points, faces, mode, tolerance, maximum_work, diagnostics):
    """Certify footprint and full triangular faces against the actual solid.

    Inside is a convex intersection of halfspaces. Outside is nonconvex: a
    triangle whose vertices are outside can cross the solid, so undecided faces
    are clipped against every side halfspace. Budget exhaustion rejects.
    """
    axial = (body_points - shell.origin) @ shell.axis
    diagnostics.update(
        body_axial_domain_minimum_m=float(axial.min()),
        body_axial_domain_maximum_m=float(axial.max()),
        physical_support_axial_length_m=shell.axial_length,
        whole_body_exterior_to_support=False,
    )
    if axial.min() < -tolerance or axial.max() > shell.axial_length + tolerance:
        return "body_support_domain_violation"
    side_count = len(shell.plane_points)
    work = len(body_points) * side_count
    if work > maximum_work:
        return "body_support_exterior_budget_exceeded"
    values = np.empty((len(body_points), side_count))
    for offset in range(0, len(body_points), 512):
        values[offset : offset + 512] = np.einsum(
            "pij,ij->pi", body_points[offset : offset + 512, None] - shell.plane_points, shell.plane_normals
        )
    diagnostics["body_support_halfspace_work"] = work
    if mode == "surface_inner":
        diagnostics["maximum_body_outward_halfspace_height_m"] = float(values.max())
        if np.any(values > tolerance):
            return "body_support_penetration"
        diagnostics["whole_body_mounting_certificate"] = "whole_triangles_inside_convex_support_halfspaces"
        return ""
    if np.any(values.max(axis=1) < -tolerance):
        return "body_support_penetration"
    clipped_count = 0
    for face in faces:
        work += 3 * side_count
        if work > maximum_work:
            diagnostics["body_support_halfspace_work"] = work
            return "body_support_exterior_budget_exceeded"
        if np.any(np.min(values[face], axis=0) >= -tolerance):
            continue
        polygon = body_points[face]
        clipped_count += 1
        for point, normal in zip(shell.plane_points, shell.plane_normals, strict=True):
            work += len(polygon)
            if work > maximum_work:
                diagnostics["body_support_halfspace_work"] = work
                return "body_support_exterior_budget_exceeded"
            polygon = _clip_polygon(polygon, point, normal, -tolerance)
            if not len(polygon):
                break
        if len(polygon):
            diagnostics["penetrating_body_face_indices"] = face.tolist()
            diagnostics["body_support_penetration_witness_m"] = polygon.mean(axis=0).tolist()
            return "body_support_penetration"
    diagnostics.update(
        body_support_halfspace_work=work,
        body_triangles_requiring_solid_clipping=clipped_count,
        whole_body_exterior_to_support=True,
        whole_body_mounting_certificate="whole_triangles_excluded_from_verified_convex_mesh_support_solid",
    )
    return ""
