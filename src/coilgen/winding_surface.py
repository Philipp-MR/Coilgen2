"""Piecewise isometric drawing charts; the physical mesh remains authoritative."""

import numpy as np


class WindingSurfaceChart:
    def __init__(self, mesh):
        from .manufacturing import _SurfaceProjector
        from .group_route_convex_shell import convex_circular_shell
        from .developed_frustum import frustum_chart

        self.mesh = mesh
        self.projector = _SurfaceProjector(mesh)
        triangles = mesh.vertices[mesh.faces]
        tolerance = max(float(np.ptp(mesh.vertices, axis=0).max()) * 1e-8, 1e-10)
        shell, _reason = convex_circular_shell(mesh, tolerance)
        if shell is not None and abs(shell.top_radius - shell.bottom_radius) > tolerance:
            chart = frustum_chart(self.projector)
            _, sides = chart.locate(triangles.mean(axis=1))
            self.face_uv = np.einsum("fvi,fij->fvj", triangles - chart.apex, chart.jacobians[sides])
            self.label = "Developed conical facets · metres"
        elif shell is not None:
            origin, axis = shell.origin, shell.axis
            corners = shell.plane_points
            first = (corners[0] - origin) / shell.bottom_radius
            second = np.cross(axis, first)
            angles = np.mod(np.arctan2((corners - origin) @ second, (corners - origin) @ first), 2 * np.pi)
            corners = corners[np.argsort(angles)]
            edges = np.roll(corners, -1, axis=0) - corners
            lengths = np.linalg.norm(edges, axis=1)
            tangents = edges / lengths[:, None]
            normals = np.cross(tangents, axis)
            distances = np.abs(np.einsum("fki,ki->fk", triangles.mean(axis=1)[:, None] - corners, normals))
            sides = distances.argmin(axis=1)
            offsets = triangles - corners[sides, None]
            starts = np.r_[0.0, np.cumsum(lengths[:-1])]
            self.face_uv = np.stack(
                (starts[sides, None] + np.einsum("fvi,fi->fv", offsets, tangents[sides]), (triangles - origin) @ axis),
                axis=2,
            )
            self.label = "Unfolded cylindrical facets · metres"
        else:
            origin = mesh.vertices.mean(axis=0)
            _, _, axes = np.linalg.svd(mesh.vertices - origin, full_matrices=False)
            if np.max(np.abs((mesh.vertices - origin) @ axes[2])) > tolerance:
                raise ValueError("A flat drawing chart requires a planar, cylindrical or conical support.")
            self.face_uv = (triangles - origin) @ axes[:2].T
            self.label = "Planar support · metres"
        self._cache = {}

    def project(self, points):
        points = np.asarray(points, dtype=float)
        token = points.tobytes()
        if token not in self._cache:
            _, _, _, faces, bary = self.projector.project_with_details(points)
            values = np.einsum("fi,fij->fj", bary, self.face_uv[faces])
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[token] = values
        return self._cache[token].copy()

    def polyline(self, points):
        points = np.asarray(points)
        values = self.project(points)
        result = values.tolist()
        jumps = (
            np.linalg.norm(np.diff(values, axis=0), axis=1) > 3 * np.linalg.norm(np.diff(points, axis=0), axis=1) + 1e-6
        )
        for index in np.flatnonzero(jumps) + 1:
            result[index] = None
        return result

    def pick(self, point):
        """Exact barycentric mapping from a chart triangle to its physical facet."""
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("A drawing point needs two finite surface coordinates.")
        a = self.face_uv[:, 0]
        matrix = np.stack((self.face_uv[:, 1] - a, self.face_uv[:, 2] - a), axis=2)
        uv = np.linalg.solve(matrix, (point - a)[..., None])[..., 0]
        valid = np.flatnonzero((uv.min(axis=1) >= -1e-9) & (uv.sum(axis=1) <= 1 + 1e-9))
        if not len(valid):
            raise ValueError("Keep the stroke inside the developed support.")
        face = int(valid[0])
        weights = np.r_[1 - uv[face].sum(), uv[face]]
        return weights @ self.mesh.vertices[self.mesh.faces[face]]


def seam_intersections(chart, contour, stroke):
    """Intersect a stroke with the exact piecewise-linear directed contour."""
    from .routing_ports import ContourArc

    arc = ContourArc(contour)
    curve = chart.project(arc.points)
    stroke = np.asarray(stroke, float)
    hits = []

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    delta = np.diff(curve, axis=0)
    supported = np.linalg.norm(delta, axis=1) <= 3 * arc.lengths + 1e-6
    for start, end in zip(stroke[:-1], stroke[1:]):
        direction = end - start
        denominator = cross(delta, direction)
        valid = supported & (np.abs(denominator) > 1e-16)
        t = np.divide(cross(start - curve[:-1], direction), denominator, out=np.zeros(len(delta)), where=valid)
        u = np.divide(cross(start - curve[:-1], delta), denominator, out=np.zeros(len(delta)), where=valid)
        for index in np.flatnonzero(valid & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
            position = (arc.positions[index] + t[index] * arc.lengths[index]) % arc.perimeter
            if not any(min(abs(position - p), arc.perimeter - abs(position - p)) < 1e-8 for p in hits):
                hits.append(float(position))
    return hits
