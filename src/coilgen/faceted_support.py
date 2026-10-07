"""Distance queries and whole-chord bounds on a verified convex extrusion.

The support is the actual triangle-covered lateral polygon, never its nominal
circular cylinder. This facade supplies geometric facet normals, so it is for
distance screens, not interpolated mounting or swept profile frames.
"""

from __future__ import annotations

import hashlib

import numpy as np


class _FacetedCylinderSupport:
    def __init__(self, projector):
        from .joint_cylinder_curves import _cylinder_chart

        # The indexed projector keeps an explicit triangle snapshot. Its
        # geometry must agree with the vertices/faces used by chart coverage.
        # A content hash invalidates a cache but cannot resolve inconsistent
        # independently edited geometry arrays.
        if not np.array_equal(projector.triangles, np.asarray(projector.vertices)[projector.faces]):
            raise ValueError("inconsistent_extrusion_triangle_snapshot")
        chart = _cylinder_chart(projector)
        self.origin, self.axis = chart.origin, chart.axis
        self.polygon, self.directions = chart.polygon, chart.directions
        self.lengths, self.lower, self.upper = chart.lengths, chart.lower, chart.upper
        self.normals = np.cross(self.directions, self.axis)
        scale = max(float(np.max(np.ptp(projector.vertices, axis=0))), 1e-12)
        self.roundoff = 256 * np.finfo(np.float64).eps * scale
        triangles = np.asarray(projector.triangles, dtype=float)
        center = triangles.mean(axis=1) - self.origin
        signed = np.einsum("fij,ij->fi", center[:, None] - self.polygon, self.normals)
        sides = np.argmin(np.abs(signed), axis=1)
        expected = self.normals[sides]
        actual = np.asarray(projector.face_normals, dtype=float)
        agreement = np.einsum("ij,ij->i", actual, expected)
        self.orientation = 1.0 if np.all(agreement > 1 - 1e-12) else -1.0
        if not np.all(np.linalg.norm(actual - self.orientation * expected, axis=1) <= 1e-12):
            raise ValueError("inconsistent_extrusion_normals")
        # Chart recognition has a looser modelling tolerance. The distance
        # facade requires coplanar actual facets to floating-point roundoff.
        residual = np.einsum("fij,fj->fi", triangles - self.origin - self.polygon[sides, None], expected)
        if np.max(np.abs(residual)) > self.roundoff:
            raise ValueError("nonplanar_extrusion_facets")
        vertices = np.asarray(projector.vertices, dtype=float) - self.origin
        axial = vertices @ self.axis
        if np.any(axial < self.lower - self.roundoff) or np.any(axial > self.upper + self.roundoff):
            raise ValueError("nonuniform_extrusion_ends")
        self.triangles, self.face_normals = triangles.copy(), actual.copy()
        self.low, self.high = self.triangles.min(axis=1), self.triangles.max(axis=1)
        self.side_faces = tuple(np.flatnonzero(sides == side) for side in range(len(self.polygon)))

    def _query(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
            raise ValueError("Extrusion distances require finite (n, 3) points.")
        projected, normals = np.empty_like(points), np.empty_like(points)
        # Bound temporary memory independently of curve sampling density.
        for offset in range(0, len(points), 1024):
            batch = points[offset : offset + 1024] - self.origin
            axial = batch @ self.axis
            radial = batch - axial[:, None] * self.axis
            delta = radial[:, None] - self.polygon
            along = np.clip(np.einsum("pij,ij->pi", delta, self.directions), 0, self.lengths)
            candidates = self.polygon + along[:, :, None] * self.directions
            separation = radial[:, None] - candidates
            squared = np.einsum("pij,pij->pi", separation, separation)
            side = np.argmin(squared, axis=1)
            rows = np.arange(len(batch))
            projected[offset : offset + len(batch)] = (
                self.origin + candidates[rows, side] + np.clip(axial, self.lower, self.upper)[:, None] * self.axis
            )
            normals[offset : offset + len(batch)] = self.orientation * self.normals[side]
        return projected, normals

    def project(self, points):
        """Closest points on the verified lateral rectangles and facet normals."""
        return self._query(points)

    def crosses_surface(self, segments, sign, tolerance):
        """Original triangle-hit predicates after a conservative facet screen.

        A hit on any actual lateral triangle crosses its facet's plane. Plane
        crossings therefore prune entire facet groups before the original
        exact AABB, signed-height, endpoint trimming and triangle-hit tests.
        """
        from .geometry_kernels import _segment_hits_triangle

        segments = np.asarray(segments, dtype=float)
        for offset in range(0, len(segments), 1024):
            block = segments[offset : offset + 1024]
            length = np.linalg.norm(block[:, 1] - block[:, 0], axis=1)
            eligible = np.flatnonzero(length > 2 * tolerance)
            if not len(eligible):
                continue
            chosen = block[eligible]
            direction = (chosen[:, 1] - chosen[:, 0]) / length[eligible, None]
            first, last = chosen[:, 0] + tolerance * direction, chosen[:, 1] - tolerance * direction
            signed_first = np.einsum("sij,ij->si", first[:, None] - self.origin - self.polygon, self.normals)
            signed_last = np.einsum("sij,ij->si", last[:, None] - self.origin - self.polygon, self.normals)
            # Include actual facet-plane roundoff and query coordinate scale.
            # This is broad-phase padding only; final predicates are unchanged.
            scale = np.maximum(np.max(np.abs(first), axis=1), np.max(np.abs(last), axis=1))
            scale = np.maximum(scale, np.max(np.abs(self.origin)) + np.max(np.abs(self.polygon)))
            padding = self.roundoff + 64 * np.finfo(np.float64).eps * scale
            possible = (np.minimum(signed_first, signed_last) <= padding[:, None]) & (
                np.maximum(signed_first, signed_last) >= -padding[:, None]
            )
            for row in np.flatnonzero(np.any(possible, axis=1)):
                start, end = chosen[row]
                # Recompute trimming exactly as the scalar reference does.
                segment_length = float(np.linalg.norm(end - start))
                if segment_length <= 2 * tolerance:
                    continue
                segment_direction = (end - start) / segment_length
                first_point, last_point = start + tolerance * segment_direction, end - tolerance * segment_direction
                indices = np.concatenate([self.side_faces[side] for side in np.flatnonzero(possible[row])])
                mask = np.all(self.low[indices] <= np.maximum(first_point, last_point), axis=1) & np.all(
                    self.high[indices] >= np.minimum(first_point, last_point), axis=1
                )
                for index in indices[mask]:
                    triangle = self.triangles[index]
                    signed = (np.vstack((start, end)) - triangle[0]) @ self.face_normals[index] * sign
                    if float(signed.min()) >= -tolerance:
                        continue
                    if _segment_hits_triangle(first_point, last_point, triangle):
                        return True
        return False

    def lift_violation(self, segments, height, sign, tolerance):
        """Prove infeasibility at original chord vertices or midpoints only.

        This is a necessary-condition rejection, not a sample-only acceptance.
        Distance to one supplied chord point already exceeds the allowed lift
        from the actual finite triangle support. General whole-chord support
        certificates remain required when this test finds no violation.
        """
        segments = np.asarray(segments, dtype=float)
        if not len(segments):
            return None
        points = np.vstack((segments[:, 0], segments[-1, 1], np.mean(segments, axis=1)))
        projected, normals = self._query(points)
        distances = np.linalg.norm(points - projected, axis=1)
        scale = np.maximum(np.max(np.abs(points), axis=1), np.max(np.abs(self.origin)) + np.max(np.abs(self.polygon)))
        margin = self.roundoff + 256 * np.finfo(np.float64).eps * scale
        if not np.any(distances > height + tolerance + margin):
            return None
        signed_height = np.einsum("ij,ij->i", points - projected, normals) * sign
        return {
            "scope": "candidate_polyline_exact_faceted_lift_violation",
            "support_certificate": "exact_faceted_distance_violation",
            "support_evaluation_count": len(points),
            "minimum_signed_height_m": float(np.min(signed_height)),
            "maximum_signed_height_m": float(np.max(signed_height)),
            "maximum_support_distance_m": float(np.max(distances)),
        }

    def certificate(self, segments, max_lift, tolerance):
        """A whole-chord certificate, or None when adaptive checks are needed.

        Each chord must lie in one exterior facet halfspace. Distance to the
        convex solid is convex there, so endpoint distances bound its complete
        lateral-support distance. Axial interiors exclude both physical rims;
        full oriented triangle coverage was checked when constructing the chart.
        """
        if self.orientation * (1.0 if max_lift >= 0 else -1.0) < 0:
            return None  # Inward lifts require the general signed surface test.
        segments = np.asarray(segments, dtype=float)
        if not len(segments) or not np.array_equal(segments[:-1, 1], segments[1:, 0]):
            return None
        points = np.vstack((segments[:, 0], segments[-1, 1]))
        axial = (points - self.origin) @ self.axis
        if np.any(axial <= self.lower + tolerance) or np.any(axial >= self.upper - tolerance):
            return None
        # A facet separates each whole segment from the convex interior. Keep
        # only arithmetic roundoff, well below the caller's physical tolerance.
        slack = min(self.roundoff, tolerance / 16)
        signed = np.einsum("pij,ij->pi", points[:, None] - self.origin - self.polygon, self.normals)
        separation = np.max(np.minimum(signed[:-1], signed[1:]), axis=1)
        if np.any(separation < -slack):
            return None
        projected, normals = self._query(points)
        distances = np.linalg.norm(points - projected, axis=1)
        height = np.einsum("ij,ij->i", points - projected, normals) * (1.0 if max_lift >= 0 else -1.0)
        if np.max(distances) + self.roundoff > abs(max_lift) + tolerance or np.min(height) < -slack:
            return None
        # Match the general support-domain contract, including its tolerance:
        # merely having an axial footprint inside a rim is insufficient when
        # the closest rim and lateral wall are indistinguishable at that scale.
        # D bounds whole-chord wall distance and M bounds axial rim separation.
        # Rationalization avoids subtracting nearly equal distances near a rim.
        distance_bound = float(np.max(distances)) + self.roundoff
        rim_margin = float(np.min(np.minimum(axial - self.lower, self.upper - axial))) - self.roundoff
        boundary_gap = rim_margin**2 / (np.hypot(distance_bound, rim_margin) + distance_bound)
        if boundary_gap <= tolerance + 4 * self.roundoff:
            return None
        return {
            "scope": "candidate_polyline_convex_faceted_support_and_outside_gap_nonregression",
            "support_certificate": "exact_convex_faceted_extrusion",
            "coplanar_support_triangle_count": 0,
            "unsupported_chord_count": 0,
            "support_evaluation_count": len(points),
            "minimum_signed_height_m": float(np.min(height)),
            "maximum_signed_height_m": float(np.max(height)),
            "maximum_support_distance_m": float(np.max(distances)),
        }


def faceted_cylinder_support(projector):
    """Return a content-checked distance facade, or None for other supports."""
    names = ("vertices", "faces", "triangles", "face_normals")
    if not all(hasattr(projector, name) for name in names):
        return None
    signature = hashlib.blake2b(digest_size=16)
    signature.update(b"convex-faceted-support-v3")
    for name in names:
        value = np.asarray(getattr(projector, name))
        signature.update(str((value.shape, value.dtype.str)).encode())
        signature.update(np.ascontiguousarray(value).tobytes())
    key = signature.digest()
    cached = getattr(projector, "_faceted_support_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        support = _FacetedCylinderSupport(projector)
    except (ValueError, AttributeError, IndexError, TypeError):
        support = None
    projector._faceted_support_cache = key, support
    return support
