"""Isometric charts of actual, complete homothetic polygonal frustum meshes.

Every side is unfolded as its physical plane.  This is a nominal surface
representation, never a finite-conductor, curvature or clearance certificate.
The chart does not substitute an analytic cone for the supplied triangulation.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .group_route_convex_shell import convex_circular_shell


def _cross(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


@dataclass(frozen=True)
class FrustumChart:
    apex: np.ndarray
    axis: np.ndarray
    origin: np.ndarray
    height: float
    corners: np.ndarray
    starts: np.ndarray
    angles: np.ndarray
    jacobians: np.ndarray
    normals: np.ndarray
    tolerance: float

    @property
    def period(self):
        return float(self.starts[-1] + self.angles[-1])

    def _develop(self, points, side):
        return (np.asarray(points) - self.apex) @ self.jacobians[side]

    def port(self, point, tangent, *, start):
        """Choose an actual incident facet with the directed attachment tangent."""
        point, tangent = np.asarray(point), np.asarray(tangent)
        if point.shape != (3,) or tangent.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("invalid_frustum_port")
        norm = float(np.linalg.norm(tangent))
        if norm <= 0 or not np.all(np.isfinite(tangent)):
            raise ValueError("frustum_port_tangent_not_supported")
        height = float((point - self.origin) @ self.axis)
        if height < -self.tolerance or height > self.height + self.tolerance:
            raise ValueError("frustum_ports_off_nominal_support")
        choices = []
        for side, jacobian in enumerate(self.jacobians):
            planar = self._develop(point, side)
            recovered = self.apex + jacobian @ planar
            if np.linalg.norm(recovered - point) > self.tolerance:
                continue
            left = np.array([np.cos(self.starts[side]), np.sin(self.starts[side])])
            right = np.array(
                [np.cos(self.starts[side] + self.angles[side]), np.sin(self.starts[side] + self.angles[side])]
            )
            if _cross(left, planar) < -self.tolerance or _cross(planar, right) < -self.tolerance:
                continue
            developed = jacobian.T @ tangent
            error = float(np.linalg.norm(jacobian @ developed - tangent)) / norm
            entering = (1 if start else -1) * developed
            if (abs(_cross(left, planar)) <= self.tolerance and _cross(left, entering) < -norm * 1e-10) or (
                abs(_cross(planar, right)) <= self.tolerance and _cross(entering, right) < -norm * 1e-10
            ):
                continue
            choices.append((error, side, planar, developed))
        if not choices or min(value[0] for value in choices) > 1e-8:
            raise ValueError("frustum_port_tangent_not_supported")
        _error, side, planar, developed = min(choices, key=lambda value: (round(value[0], 12), value[1]))
        return side, planar, developed

    def locate(self, points):
        """Locate supported ports without moving them; seams have two images."""
        values, sides = [], []
        for point in np.asarray(points):
            found = None
            for side, jacobian in enumerate(self.jacobians):
                planar = self._develop(point, side)
                recovered = self.apex + jacobian @ planar
                left = np.array([np.cos(self.starts[side]), np.sin(self.starts[side])])
                right = np.array(
                    [np.cos(self.starts[side] + self.angles[side]), np.sin(self.starts[side] + self.angles[side])]
                )
                if (
                    np.linalg.norm(recovered - point) <= self.tolerance
                    and _cross(left, planar) >= -self.tolerance
                    and _cross(planar, right) >= -self.tolerance
                ):
                    found = side, planar
                    break
            height = float((point - self.origin) @ self.axis)
            if found is None or not -self.tolerance <= height <= self.height + self.tolerance:
                raise ValueError("frustum_ports_off_nominal_support")
            sides.append(found[0])
            values.append(np.r_[found[1], 0.0])
        return np.asarray(values), np.asarray(sides)

    def map(self, points, reference_angle):
        """Map a bounded chart image back to its exact physical side planes."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] not in (2, 3) or not np.all(np.isfinite(points)):
            raise ValueError("invalid_frustum_chart_curve")
        if not np.isfinite(reference_angle):
            raise ValueError("invalid_frustum_chart_reference")
        if points.shape[1] == 3 and np.any(np.abs(points[:, 2]) > self.tolerance):
            raise ValueError("frustum_chart_height_requires_spatial_curve")
        radial = np.linalg.norm(points[:, :2], axis=1)
        if np.any(radial <= self.tolerance):
            raise ValueError("frustum_curve_near_apex")
        angles = np.arctan2(points[:, 1], points[:, 0])
        angles += 2 * np.pi * np.rint((reference_angle - angles) / (2 * np.pi))
        if np.ptp(angles) >= np.pi - 1e-8:
            raise ValueError("frustum_curve_angular_domain_exceeded")
        turns = np.floor(angles / self.period).astype(int)
        local = angles - turns * self.period
        sides = np.clip(np.searchsorted(self.starts, local, side="right") - 1, 0, len(self.starts) - 1)
        rotations = -turns * self.period
        cosine, sine = np.cos(rotations), np.sin(rotations)
        planar = np.column_stack(
            (cosine * points[:, 0] - sine * points[:, 1], sine * points[:, 0] + cosine * points[:, 1])
        )
        mapped = self.apex + np.einsum("nij,nj->ni", self.jacobians[sides], planar)
        axial = (mapped - self.origin) @ self.axis
        if np.any(axial < -self.tolerance) or np.any(axial > self.height + self.tolerance):
            raise ValueError("frustum_curve_outside_axial_domain")
        return mapped, sides

    def boundary_rays(self, low, high, maximum_work):
        first, last = int(np.floor(low / self.period)), int(np.floor(high / self.period))
        if (last - first + 1) * len(self.starts) > maximum_work:
            raise ValueError("frustum_curve_sample_budget_exceeded")
        angles = np.concatenate([self.starts + turn * self.period for turn in range(first, last + 1)])
        angles = angles[(angles > low - 1e-10) & (angles < high + 1e-10)]
        return np.column_stack((np.cos(angles), np.sin(angles)))


def frustum_chart(projector, *, maximum_vertices=100000, maximum_sides=2048, maximum_work=50_000_000):
    """Verify the finite annulus, homothetic rims, all actual facets and coverage."""
    vertices, raw_faces = np.asarray(projector.vertices, dtype=float), np.asarray(projector.faces)
    faces = np.asarray(raw_faces, dtype=int)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not 6 <= len(vertices) <= maximum_vertices
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or len(faces) > 2 * maximum_vertices
        or not np.array_equal(raw_faces, faces)
        or not np.all(np.isfinite(vertices))
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("unsupported_frustum_mesh_or_budget")
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered = np.sort(edges, axis=1)
    _unique, inverse, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    balance = np.bincount(inverse, weights=np.where(edges[:, 0] == ordered[:, 0], 1, -1))
    if np.any(counts > 2) or np.any((counts == 2) & (balance != 0)):
        raise ValueError("unsupported_frustum_mesh_topology")
    boundary = edges[counts[inverse] == 1]
    beginning, beginning_counts = np.unique(boundary[:, 0], return_counts=True)
    ending, ending_counts = np.unique(boundary[:, 1], return_counts=True)
    if (
        not len(boundary)
        or not np.array_equal(beginning, ending)
        or np.any(beginning_counts != 1)
        or np.any(ending_counts != 1)
    ):
        raise ValueError("unsupported_frustum_boundaries")
    scale = float(np.max(np.ptp(vertices, axis=0)))
    tolerance = max(1e-10 * scale, 1e-12)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) <= tolerance**2):
        raise ValueError("unsupported_frustum_degenerate_faces")
    # Recognition is geometric, and may read either coherent winding. The
    # physical support/body checker still owns declared mounting orientation.
    oriented = faces
    sign = float(np.einsum("ij,ij->", cross, triangles.mean(axis=1) - vertices.mean(axis=0)))
    if sign < 0:
        oriented = faces[:, [0, 2, 1]]
    shell, reason = convex_circular_shell(
        SimpleNamespace(vertices=vertices, faces=oriented), tolerance, maximum_work=maximum_work
    )
    if shell is None:
        raise ValueError(reason)
    if len(shell.plane_points) > maximum_sides:
        raise ValueError("unsupported_frustum_mesh_or_budget")
    difference = shell.top_radius - shell.bottom_radius
    if abs(difference) <= tolerance:
        raise ValueError("uniform_cylinder_requires_cylinder_chart")
    # Near-cylinder developments have remote apex coordinates and ill-conditioned
    # roots; reject instead of inventing a flat or ideal-cylinder approximation.
    if abs(difference) / shell.axial_length < 1e-6:
        raise ValueError("frustum_development_ill_conditioned")
    apex = shell.origin - shell.axis * (shell.axial_length * shell.bottom_radius / difference)
    corners = shell.plane_points.copy()
    away = np.sign(difference) * shell.axis
    if float(np.sum(np.cross(corners - apex, np.roll(corners - apex, -1, axis=0)), axis=0) @ away) < 0:
        corners = corners[np.r_[0, np.arange(len(corners) - 1, 0, -1)]]
    rays = corners - apex
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    cosines = np.clip(np.einsum("ij,ij->i", rays, np.roll(rays, -1, axis=0)), -1.0, 1.0)
    angles = np.arctan2(np.linalg.norm(np.cross(rays, np.roll(rays, -1, axis=0)), axis=1), cosines)
    if np.any(angles < 1e-10) or not 0 < float(angles.sum()) < 2 * np.pi - 1e-10:
        raise ValueError("frustum_development_ill_conditioned")
    starts = np.r_[0.0, np.cumsum(angles[:-1])]
    jacobians, normals = [], []
    for side, first in enumerate(rays):
        second = (rays[(side + 1) % len(rays)] - cosines[side] * first) / np.sin(angles[side])
        rotation = np.array(
            [[np.cos(starts[side]), np.sin(starts[side])], [-np.sin(starts[side]), np.cos(starts[side])]]
        )
        jacobians.append(np.column_stack((first, second)) @ rotation)
        normal = np.cross(first, second)
        radial = corners[side] - shell.origin
        radial -= float(radial @ shell.axis) * shell.axis
        normals.append(normal * (1 if float(normal @ radial) > 0 else -1))
    return FrustumChart(
        apex,
        shell.axis.copy(),
        shell.origin.copy(),
        shell.axial_length,
        corners,
        starts,
        angles,
        np.asarray(jacobians),
        np.asarray(normals),
        tolerance,
    )
