"""Bounded checks for replacing one owned connector over a triangle support.

The certificate concerns the supplied piecewise-linear centerline. It does not
certify an unsampled analytic curve, conductor-profile clearance, or the rest of
an already invalid route; those remain the final sweep's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry_kernels import _point_triangle_sq, _segment_hits_triangle, segment_distances
from .faceted_support import faceted_cylinder_support


@dataclass(frozen=True, slots=True)
class BridgeValidation:
    accepted: bool
    reasons: tuple[str, ...]
    diagnostics: dict[str, float | int | str | bool]


def _segments(points: np.ndarray) -> np.ndarray:
    return np.stack((points[:-1], points[1:]), axis=1)


def _near_pairs(first: np.ndarray, second: np.ndarray, distance: float):
    """Exact distances after a bounded, conservative AABB broad phase."""
    second_low, second_high = second.min(axis=1), second.max(axis=1)
    for a in range(0, len(first), 32):
        block = first[a : a + 32]
        low, high = block.min(axis=1), block.max(axis=1)
        for b in range(0, len(second), 1024):
            gap = np.maximum(
                np.maximum(
                    low[:, None] - second_high[None, b : b + 1024], second_low[None, b : b + 1024] - high[:, None]
                ),
                0.0,
            )
            i, j = np.nonzero(np.einsum("ijk,ijk->ij", gap, gap) <= distance**2)
            if not len(i):
                continue
            j += b
            values = segment_distances(
                block[i, 0], block[i, 1] - block[i, 0], second[j, 0], second[j, 1] - second[j, 0]
            )
            keep = values <= distance
            yield i[keep] + a, j[keep], values[keep]


def _capped_minima(first: np.ndarray, second: np.ndarray, cap: float) -> np.ndarray:
    minima = np.full(len(second), cap)
    for _i, j, values in _near_pairs(first, second, cap):
        np.minimum.at(minima, j, values)
    return minima


def _contact_point(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """A closest point on the first segment, including parallel contacts."""
    p, q = first
    a, b = second
    u, v, r = q - p, b - a, p - a
    uu, vv, uv, ur, vr = u @ u, v @ v, u @ v, u @ r, v @ r
    cross = np.cross(u, v)
    denominator = cross @ cross
    s = np.clip(np.cross(-r, v) @ cross / denominator, 0.0, 1.0) if denominator > 0 else 0.0
    t = (uv * s + vr) / vv if vv > 0 else 0.0
    if t < 0 or vv == 0:
        s = np.clip(-ur / uu, 0.0, 1.0) if uu > 0 else 0.0
    elif t > 1:
        s = np.clip((uv - ur) / uu, 0.0, 1.0) if uu > 0 else 0.0
    return p + s * u


def _adjacent_overlap(first: np.ndarray, second: np.ndarray, shared: np.ndarray, tolerance: float) -> bool:
    a = first[np.argmax(np.linalg.norm(first - shared, axis=1))] - shared
    b = second[np.argmax(np.linalg.norm(second - shared, axis=1))] - shared
    lengths = np.linalg.norm(a), np.linalg.norm(b)
    if min(lengths) <= tolerance or a @ b <= 0:
        return False
    return bool(np.linalg.norm(np.cross(a, b)) / max(lengths) <= tolerance)


def _contact_is_inherited(first: np.ndarray, second: np.ndarray, owned: np.ndarray, tolerance: float) -> bool:
    direction = first[1] - first[0]
    length = float(np.linalg.norm(direction))
    axis = direction / length
    other_direction = second[1] - second[0]
    parallel = np.linalg.norm(np.cross(axis, other_direction)) <= 1e-12 * np.linalg.norm(other_direction)
    parameters = (second - first[0]) @ axis
    low, high = max(0.0, float(parameters.min())), min(length, float(parameters.max()))
    if parallel and high - low > tolerance:
        # One old point contact cannot excuse a newly overlapping interval.
        # Certify the whole overlap by the old collinear segment union.
        projections = (owned - first[0]) @ axis
        residual = owned - first[0] - projections[:, :, None] * axis
        on_line = np.max(np.linalg.norm(residual, axis=2), axis=1) <= tolerance
        intervals = sorted(zip(projections[on_line].min(axis=1), projections[on_line].max(axis=1)))
        covered = low
        for begin, end in intervals:
            if end < covered - tolerance:
                continue
            if begin > covered + tolerance:
                return False
            covered = max(covered, float(end))
            if covered >= high - tolerance:
                return True
        return False
    point = _contact_point(first, second)
    distance = segment_distances(
        owned[:, 0], owned[:, 1] - owned[:, 0], np.broadcast_to(point, owned[:, 0].shape), np.zeros_like(owned[:, 0])
    )
    return bool(float(np.min(distance, initial=np.inf)) <= tolerance)


class _PlanarSupport:
    """Exact chord-footprint coverage by a union of coplanar mesh triangles."""

    def __init__(self, support, origin: np.ndarray, normal: np.ndarray, tolerance: float):
        self.origin, self.normal, self.tolerance = origin, normal, tolerance
        if hasattr(support, "triangles"):
            triangles = np.asarray(support.triangles, dtype=float)
        else:
            triangles = np.asarray(support.vertices, dtype=float)[np.asarray(support.faces, dtype=int)]
        self.triangles = triangles
        heights = (triangles - origin) @ normal
        self.planar = np.max(np.abs(heights), axis=1) <= tolerance if len(triangles) else np.empty(0, dtype=bool)
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        first = np.cross(normal, axis)
        first /= np.linalg.norm(first)
        self.basis = np.column_stack((first, np.cross(normal, first)))
        flat = (triangles[self.planar] - origin) @ self.basis
        e1, e2 = flat[:, 1] - flat[:, 0], flat[:, 2] - flat[:, 0]
        determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        valid = np.abs(determinant) > 1e-30
        self.flat, e1, e2, determinant = flat[valid], e1[valid], e2[valid], determinant[valid]
        self.grad_u = np.column_stack((e2[:, 1], -e2[:, 0])) / determinant[:, None]
        self.grad_v = np.column_stack((-e1[:, 1], e1[:, 0])) / determinant[:, None]
        self.bary_tolerance = tolerance * np.linalg.norm(
            np.stack((-self.grad_u - self.grad_v, self.grad_u, self.grad_v), axis=1), axis=-1
        )
        self.low, self.high = self.flat.min(axis=1), self.flat.max(axis=1)

    def covers(self, start: np.ndarray, end: np.ndarray) -> bool:
        p, q = (np.stack((start, end)) - self.origin) @ self.basis
        mask = np.all(self.low <= np.maximum(p, q) + self.tolerance, axis=1) & np.all(
            self.high >= np.minimum(p, q) - self.tolerance, axis=1
        )
        indices = np.flatnonzero(mask)
        if not len(indices):
            return False
        offset = p - self.flat[indices, 0]
        u = np.einsum("ij,ij->i", self.grad_u[indices], offset)
        v = np.einsum("ij,ij->i", self.grad_v[indices], offset)
        initial = np.column_stack((1 - u - v, u, v))
        du, dv = self.grad_u[indices] @ (q - p), self.grad_v[indices] @ (q - p)
        change = np.column_stack((-du - dv, du, dv))
        bounds = -self.bary_tolerance[indices]
        lower, upper = np.zeros(len(indices)), np.ones(len(indices))
        for coordinate in range(3):
            delta = change[:, coordinate]
            crossing = np.divide(
                bounds[:, coordinate] - initial[:, coordinate], delta, out=np.zeros(len(indices)), where=delta != 0
            )
            lower = np.where(delta > 0, np.maximum(lower, crossing), lower)
            upper = np.where(delta < 0, np.minimum(upper, crossing), upper)
            upper[(delta == 0) & (initial[:, coordinate] < bounds[:, coordinate])] = -1
        intervals = sorted(zip(lower[lower <= upper], upper[lower <= upper]))
        covered = 0.0
        parameter_tolerance = self.tolerance / max(float(np.linalg.norm(q - p)), self.tolerance)
        for first, last in intervals:
            if first > covered + parameter_tolerance:
                return False
            covered = max(covered, float(last))
            if covered >= 1.0 - parameter_tolerance:
                return True
        return False

    def crosses_other_faces(self, segments: np.ndarray) -> bool:
        triangles = self.triangles[~self.planar]
        low, high = triangles.min(axis=1), triangles.max(axis=1)
        for start, end in segments:
            mask = np.all(low <= np.maximum(start, end) + self.tolerance, axis=1) & np.all(
                high >= np.minimum(start, end) - self.tolerance, axis=1
            )
            for triangle in triangles[mask]:
                if _segment_hits_triangle(start, end, triangle):
                    return True
                if min(_point_triangle_sq(start, triangle), _point_triangle_sq(end, triangle)) <= self.tolerance**2:
                    return True
                distances = segment_distances(
                    np.broadcast_to(start, (3, 3)),
                    np.broadcast_to(end - start, (3, 3)),
                    triangle,
                    np.roll(triangle, -1, axis=0) - triangle,
                )
                if np.min(distances) <= self.tolerance:
                    return True
        return False


class _SpatialSupport:
    """Adaptive certificates for polyline height and support-boundary avoidance.

    Distance to a closed set is 1-Lipschitz. The distance to one triangle is
    also convex, giving a tighter whole-chord upper bound near the lift limit.
    The difference between distance to boundary edges and distance to the
    support is 2-Lipschitz; a positive whole-chord margin excludes projection
    onto a physical boundary or hole. Uncertified chords are subdivided, never
    accepted merely because a fixed set of samples happened to pass.
    """

    def __init__(self, projector, height: float, sign: float, tolerance: float):
        self.projector, self.height, self.sign, self.tolerance = projector, height, sign, tolerance
        self.extrusion = faceted_cylinder_support(projector)
        self.triangles = np.asarray(projector.triangles)
        self.low, self.high = self.triangles.min(axis=1), self.triangles.max(axis=1)
        edges = np.asarray(sorted(projector.boundary_edges), dtype=int).reshape(-1, 2)
        self.boundaries = projector.vertices[edges]
        self.evaluations = 0
        self.minimum_height, self.maximum_height = np.inf, -np.inf
        self.maximum_distance = 0.0
        self.negative_certificate = None

    def sample(self, point):
        # The projector's tree lookup is a useful initial upper bound, but its
        # finite nearest-vertex/centroid candidate set is not globally exact.
        # Verify every triangle whose AABB can improve that bound.
        from .manufacturing import _closest_point_and_barycentric

        projected, _normal, _face_normal, indices, _bary = self.projector.project_with_details(point[None, :])
        best_point, face = projected[0], int(indices[0])
        best_squared = float(np.sum((point - best_point) ** 2))
        gap = np.maximum(np.maximum(self.low - point, point - self.high), 0.0)
        possible = np.flatnonzero(np.einsum("ij,ij->i", gap, gap) <= best_squared)
        for index in possible:
            closest, _weights = _closest_point_and_barycentric(point, self.triangles[index])
            squared = float(np.sum((point - closest) ** 2))
            if squared < best_squared:
                best_squared, best_point, face = squared, closest, int(index)
        distance = float(np.sqrt(best_squared))
        normal = self.projector.face_normals[face]
        signed_height = float((point - best_point) @ normal) * self.sign
        boundary_distance = np.inf
        if len(self.boundaries):
            boundary_distance = float(
                np.min(
                    segment_distances(
                        self.boundaries[:, 0],
                        self.boundaries[:, 1] - self.boundaries[:, 0],
                        np.broadcast_to(point, (len(self.boundaries), 3)),
                        np.zeros((len(self.boundaries), 3)),
                    )
                )
            )
        self.evaluations += 1
        self.minimum_height = min(self.minimum_height, signed_height)
        self.maximum_height = max(self.maximum_height, signed_height)
        self.maximum_distance = max(self.maximum_distance, distance)
        return distance, boundary_distance, signed_height, face

    def crosses_surface(self, segments):
        if self.extrusion is not None:
            return self.extrusion.crosses_surface(segments, self.sign, self.tolerance)
        for start, end in segments:
            length = float(np.linalg.norm(end - start))
            if length <= 2 * self.tolerance:
                continue
            # Attachments may touch their support. Remove only a numerical
            # endpoint neighborhood; all remaining triangle hits are rejected.
            direction = (end - start) / length
            first, last = start + self.tolerance * direction, end - self.tolerance * direction
            mask = np.all(self.low <= np.maximum(first, last), axis=1) & np.all(
                self.high >= np.minimum(first, last), axis=1
            )
            for index in np.flatnonzero(mask):
                triangle = self.triangles[index]
                signed = (np.vstack((start, end)) - triangle[0]) @ self.projector.face_normals[index] * self.sign
                if float(signed.min()) >= -self.tolerance:
                    continue  # Contact within the existing support tolerance is permitted.
                if _segment_hits_triangle(first, last, triangle):
                    return True
        return False

    def validate(self, segments, *, maximum_evaluations=8192):
        if self.crosses_surface(segments):
            return ["support_surface_intersection"]
        if self.extrusion is not None:
            self.negative_certificate = self.extrusion.lift_violation(segments, self.height, self.sign, self.tolerance)
            if self.negative_certificate is not None:
                self.evaluations = self.negative_certificate["support_evaluation_count"]
                self.minimum_height = self.negative_certificate["minimum_signed_height_m"]
                self.maximum_height = self.negative_certificate["maximum_signed_height_m"]
                self.maximum_distance = self.negative_certificate["maximum_support_distance_m"]
                return ["lift_limit_violation"]
        cache = {}

        def sample(point):
            key = tuple(point)
            if key not in cache:
                cache[key] = self.sample(point)
            return cache[key]

        pending = [(start, end) for start, end in segments]
        while pending:
            if self.evaluations + 3 > maximum_evaluations:
                return ["support_certificate_budget_exceeded"]
            start, end = pending.pop()
            midpoint, length = (start + end) / 2, float(np.linalg.norm(end - start))
            values = [sample(point) for point in (start, midpoint, end)]
            if any(value[0] > self.height + self.tolerance or value[2] < -self.tolerance for value in values):
                return ["lift_limit_violation"]
            # A closest physical boundary at a nonzero offset gives no support
            # footprint. Internal shared triangle edges are deliberately absent
            # from this set, so crossing a real mesh crease remains supported.
            if any(value[0] > self.tolerance and value[1] <= value[0] + self.tolerance for value in values):
                return ["support_domain_violation"]
            middle = values[1]
            height_upper = middle[0] + length / 2
            for face in {value[3] for value in values}:
                triangle = self.triangles[face]
                triangle_upper = np.sqrt(max(_point_triangle_sq(start, triangle), _point_triangle_sq(end, triangle)))
                height_upper = min(height_upper, triangle_upper)
            boundary_safe = middle[1] - middle[0] > length + self.tolerance
            if height_upper <= self.height + self.tolerance and boundary_safe:
                continue
            if length <= self.tolerance:
                return ["support_domain_uncertified"]
            pending.extend(((start, midpoint), (midpoint, end)))
        return []


def _validate_support(projector, original, proposed, normal, max_lift, tolerance, diagnostics):
    extrusion = faceted_cylinder_support(projector)
    if extrusion is not None:
        certificate = extrusion.certificate(proposed, max_lift, tolerance)
        if certificate is not None:
            diagnostics.update(certificate)
            return []
    support = _PlanarSupport(projector, original, normal, tolerance)
    diagnostics["coplanar_support_triangle_count"] = len(support.flat)
    missing = sum(not support.covers(start, end) for start, end in proposed) if len(support.flat) else len(proposed)
    diagnostics["unsupported_chord_count"] = missing
    points = np.vstack((proposed[:, 0], proposed[-1, 1]))
    heights = (points - original) @ normal * (1.0 if max_lift >= 0 else -1.0)
    diagnostics["minimum_signed_height_m"], diagnostics["maximum_signed_height_m"] = (
        float(heights.min()),
        float(heights.max()),
    )
    within_height = heights.min() >= -tolerance and heights.max() <= abs(max_lift) + tolerance
    if len(support.flat) and not missing and within_height:
        diagnostics["support_certificate"] = "exact_planar_triangle_union"
        if not support.crosses_other_faces(proposed):
            return []
        if not hasattr(projector, "project_with_details"):
            return ["support_surface_intersection"]
        # A legal attachment can touch the neighboring facet at a crease.
        # Certify its signed distance instead of rejecting every such contact.
    if not hasattr(projector, "project_with_details") or not len(support.triangles):
        reasons = ["support_domain_violation"] if len(support.flat) and missing else []
        if not len(support.flat):
            reasons.append("unsupported_planar_patch")
        if not within_height:
            reasons.append("lift_limit_violation")
        return reasons
    spatial = _SpatialSupport(projector, abs(max_lift), 1.0 if max_lift >= 0 else -1.0, tolerance)
    reasons = spatial.validate(proposed)
    diagnostics["planar_uncovered_chord_count"] = diagnostics.pop("unsupported_chord_count")
    if not reasons:
        diagnostics["unsupported_chord_count"] = 0
    diagnostics.update(
        scope="candidate_polyline_spatial_support_and_outside_gap_nonregression",
        support_certificate="adaptive_distance_and_boundary_bounds_with_triangle_intersections",
        support_evaluation_count=spatial.evaluations,
    )
    if spatial.evaluations:
        diagnostics.update(
            minimum_signed_height_m=spatial.minimum_height,
            maximum_signed_height_m=spatial.maximum_height,
            maximum_support_distance_m=spatial.maximum_distance,
        )
    else:
        diagnostics.pop("minimum_signed_height_m", None)
        diagnostics.pop("maximum_signed_height_m", None)
    if spatial.negative_certificate is not None:
        diagnostics.update(spatial.negative_certificate)
    return reasons


def validate_connector_bridge(
    original_points: np.ndarray,
    start_index: int,
    end_index: int,
    candidate_points: np.ndarray,
    *,
    support_projector,
    max_lift_m: float,
    support_normal: np.ndarray,
    physical_clearance_m: float,
    target_clearance_m: float,
    source_fractions: np.ndarray | None = None,
    closed: bool = False,
    tolerance_m: float = 1e-8,
) -> BridgeValidation:
    """Check one inclusive, non-wrapping connector replacement without mutation.

    Each outside segment retains at least its old minimum gap to the owned
    connector, capped at max(physical clearance, design target). Existing
    deficits are reported rather than forcing the entire input route to pass.
    Optional source fractions document monotone provenance; the comparison is
    over the entire owned span and therefore independent of its sampling.

    A coplanar triangle-union certificate is used when available. Otherwise an
    indexed projector permits bounded adaptive certification against the actual
    support, including creases: whole-chord support distance, signed orientation,
    physical boundary avoidance, and exact segment/triangle intersection checks.
    """
    original, candidate = np.asarray(original_points, dtype=float), np.asarray(candidate_points, dtype=float)
    tolerance = float(tolerance_m)
    if original.ndim != 2 or original.shape[1:] != (3,) or candidate.ndim != 2 or candidate.shape[1:] != (3,):
        raise ValueError("Original and candidate paths must have shape (N, 3).")
    if not 0 <= start_index < end_index < len(original) or len(candidate) < 2:
        raise ValueError("The inclusive connector range must contain at least two vertices.")
    scalars = [max_lift_m, physical_clearance_m, target_clearance_m, tolerance]
    normal = np.asarray(support_normal, dtype=float)
    if not np.all(np.isfinite(scalars)) or tolerance <= 0 or min(physical_clearance_m, target_clearance_m) < 0:
        raise ValueError("Clearances must be finite and nonnegative, and tolerance must be positive.")
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) <= 1e-15:
        raise ValueError("The support normal must be a finite nonzero vector.")
    normal = normal / np.linalg.norm(normal)
    if source_fractions is not None:
        fractions = np.asarray(source_fractions, dtype=float)
        if (
            fractions.shape != (len(candidate),)
            or not np.all(np.isfinite(fractions))
            or np.any(np.diff(fractions) < 0)
            or abs(fractions[0]) > 1e-12
            or abs(fractions[-1] - 1) > 1e-12
        ):
            raise ValueError("Source fractions must increase monotonically from zero to one, one per candidate vertex.")
    diagnostics: dict[str, float | int | str | bool] = {
        "scope": "candidate_polyline_planar_support_and_outside_gap_nonregression",
        "candidate_segment_count": len(candidate) - 1,
        "source_correspondence_supplied": source_fractions is not None,
        "clearance_comparison_m": max(float(physical_clearance_m), float(target_clearance_m)),
    }
    reasons: list[str] = []
    if (
        not np.all(np.isfinite(original))
        or not np.all(np.isfinite(candidate))
        or np.any(np.linalg.norm(np.diff(candidate, axis=0), axis=1) == 0)
    ):
        return BridgeValidation(False, ("invalid_centerline",), diagnostics)
    if (
        max(np.linalg.norm(candidate[0] - original[start_index]), np.linalg.norm(candidate[-1] - original[end_index]))
        > tolerance
    ):
        return BridgeValidation(False, ("endpoint_mismatch",), diagnostics)

    owned, proposed = _segments(original[start_index : end_index + 1]), _segments(candidate)
    reasons.extend(
        _validate_support(
            support_projector, original[start_index], proposed, normal, max_lift_m, tolerance, diagnostics
        )
    )

    full = original
    if closed and np.linalg.norm(full[-1] - full[0]) > tolerance:
        full = np.vstack((full, full[0]))
    route = _segments(full)
    outside_indices = np.flatnonzero((np.arange(len(route)) < start_index) | (np.arange(len(route)) >= end_index))
    outside = route[outside_indices]
    before = start_index - 1 if start_index > 0 else len(route) - 1 if closed else -1
    after = end_index if end_index < len(route) else 0 if closed else -1
    neighboring = (outside_indices == before) | (outside_indices == after)
    cap = max(float(physical_clearance_m), float(target_clearance_m), tolerance)
    old_minima, new_minima = _capped_minima(owned, outside, cap), _capped_minima(proposed, outside, cap)
    regressions = (~neighboring) & (new_minima + tolerance < old_minima)
    physical_regressions = (~neighboring) & (
        np.minimum(new_minima, physical_clearance_m) + tolerance < np.minimum(old_minima, physical_clearance_m)
    )
    diagnostics.update(
        outside_segment_count=len(outside),
        baseline_below_clearance_count=int(np.count_nonzero((~neighboring) & (old_minima < cap - tolerance))),
        candidate_below_clearance_count=int(np.count_nonzero((~neighboring) & (new_minima < cap - tolerance))),
        clearance_regression_count=int(np.count_nonzero(regressions)),
        physical_clearance_regression_count=int(np.count_nonzero(physical_regressions)),
        maximum_gap_regression_m=float(np.max(np.where(~neighboring, old_minima - new_minima, 0.0), initial=0.0)),
    )
    if np.any(regressions):
        reasons.append("clearance_regression")
        diagnostics["first_regressed_route_segment"] = int(outside_indices[np.flatnonzero(regressions)[0]])

    introduced, inherited = 0, 0
    for ii, jj, _distances in _near_pairs(proposed, outside, tolerance):
        for i, j in zip(ii, jj):
            adjacent_start = i == 0 and outside_indices[j] == before
            adjacent_end = i == len(proposed) - 1 and outside_indices[j] == after
            if adjacent_start or adjacent_end:
                shared = candidate[0] if adjacent_start else candidate[-1]
                if not _adjacent_overlap(proposed[i], outside[j], shared, tolerance):
                    continue
                introduced += 1
                continue
            if _contact_is_inherited(proposed[i], outside[j], owned, tolerance):
                inherited += 1
            else:
                introduced += 1
    diagnostics["new_route_intersection_count"], diagnostics["inherited_route_contact_count"] = introduced, inherited
    if introduced:
        reasons.append("new_route_intersection")
    self_contacts = 0
    for ii, jj, _distances in _near_pairs(proposed, proposed, tolerance):
        for i, j in zip(ii, jj):
            if j <= i:
                continue
            if j == i + 1 and not _adjacent_overlap(proposed[i], proposed[j], candidate[j], tolerance):
                continue
            self_contacts += 1
    diagnostics["candidate_self_intersection_count"] = self_contacts
    if self_contacts:
        reasons.append("candidate_self_intersection")
    return BridgeValidation(not reasons, tuple(reasons), diagnostics)
