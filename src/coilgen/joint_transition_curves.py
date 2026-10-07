"""Bounded smooth proposals for a whole group-return/next-group transition.

Unlike the local crossing bridge generator, these curves may initially point
away from their endpoint chord. Reentrant tangents are not themselves a cusp.
These are proposals only: support distance/domain, other conductors and the
complete finite swept body remain the caller's acceptance checks.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial

from .connector_bridges import (
    BridgeCandidate,
    BridgeGeneration,
    _adaptive_parameters,
    _baseline,
    _evaluate,
    _maximum_distance_to_polyline,
)
from .geometry_collision import _SegmentBvhNode, _segment_bvh_candidate_pairs
from .geometry_kernels import segment_distances
from .joint_curve_search import validate_search_level
from .routing_objective import routing_cost


class _DistanceBudgetExceeded(RuntimeError):
    """No clearance ranking is available within the bounded spatial query."""


class _SegmentDistanceIndex:
    """Reusable conservative midpoint-ball broad phase for segment distances.

    The largest segment half-length expands every query, so even a long
    obstacle whose midpoint is remote cannot be missed. Individual boxes then
    prune the candidates. Raw neighbor counts are checked before list allocation;
    neither distant segments nor obstacle-obstacle pairs are compared.
    """

    def __init__(self, segments):
        self.segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
        self.minimum = self.segments.min(axis=1)
        self.maximum = self.segments.max(axis=1)
        self.centers = self.segments.mean(axis=1)
        self.half_lengths = np.linalg.norm(np.diff(self.segments, axis=1)[:, 0], axis=1) / 2
        self.maximum_half_length = float(np.max(self.half_lengths, initial=0))
        self.tree = self.fallback_tree = None
        if len(self.segments):
            try:
                from scipy.spatial import cKDTree
            except ImportError:
                self.fallback_tree = self._build_bounds(np.arange(len(self.segments)))
            else:
                self.tree = cKDTree(self.centers)

    def _build_bounds(self, indices):
        low, high = self.minimum[indices].min(axis=0), self.maximum[indices].max(axis=0)
        if len(indices) <= 12:
            return _SegmentBvhNode(indices, low, high)
        axis = int(np.argmax(np.ptp(self.centers[indices], axis=0)))
        ordered = indices[np.argsort(self.centers[indices, axis], kind="stable")]
        middle = len(ordered) // 2
        return _SegmentBvhNode(
            np.empty(0, dtype=np.int64),
            low,
            high,
            self._build_bounds(ordered[:middle]),
            self._build_bounds(ordered[middle:]),
        )

    def _numpy_neighbors(self, segments, distance, maximum_pairs):
        """NumPy-only AABB traversal with at most twelve leaf indices per chunk."""
        neighbors, count = [], 0
        for segment in segments:
            low, high = segment.min(axis=0), segment.max(axis=0)
            stack, found = [self.fallback_tree], []
            while stack:
                node = stack.pop()
                delta = np.maximum(0, np.maximum(low - node.maximum, node.minimum - high))
                if float(delta @ delta) > (distance + 1e-12) ** 2:
                    continue
                if node.leaf:
                    count += len(node.indices)
                    if count > maximum_pairs:
                        raise _DistanceBudgetExceeded("curve_distance_pair_budget_exceeded")
                    found.extend(node.indices)
                else:
                    stack.extend((node.right, node.left))
            neighbors.append(sorted(found))
        return neighbors, count

    def gap(self, segments, cap, *, excluded=(), consume=None, maximum_pairs=200000):
        if not len(self.segments) or not len(segments):
            return cap
        first = np.asarray(segments).reshape(-1, 2, 3)
        excluded = np.asarray(tuple(excluded), dtype=np.int64)
        allowed = np.ones(len(self.segments), dtype=bool)
        allowed[excluded[(excluded >= 0) & (excluded < len(allowed))]] = False
        minimum, evaluated = float(cap), 0
        for offset in range(0, len(first), 16):
            batch = first[offset : offset + 16]
            if self.tree is None:
                neighbors, count = self._numpy_neighbors(batch, minimum, maximum_pairs - evaluated)
            else:
                centers = batch.mean(axis=1)
                radii = np.linalg.norm(np.diff(batch, axis=1)[:, 0], axis=1) / 2
                radii += self.maximum_half_length + minimum + 1e-12
                count = int(np.sum(self.tree.query_ball_point(centers, radii, return_length=True)))
            evaluated += count
            if evaluated > maximum_pairs:
                raise _DistanceBudgetExceeded("curve_distance_pair_budget_exceeded")
            if consume is not None:
                consume(count)
            if self.tree is not None:
                neighbors = self.tree.query_ball_point(centers, radii, return_sorted=True)
            sizes = np.fromiter((len(indices) for indices in neighbors), dtype=np.int64, count=len(batch))
            if not np.any(sizes):
                continue
            # The raw-neighbor budget above bounds this allocation. All pairs
            # share the batch's initial cap; a later smaller minimum could only
            # discard pairs that cannot improve the same final minimum.
            rows = np.repeat(np.arange(len(batch)), sizes)
            indices = np.concatenate(neighbors).astype(np.int64, copy=False)
            keep = allowed[indices]
            rows, indices = rows[keep], indices[keep]
            low, high = batch.min(axis=1), batch.max(axis=1)
            gap = np.maximum(0, np.maximum(low[rows] - self.maximum[indices], self.minimum[indices] - high[rows]))
            keep = np.einsum("ij,ij->i", gap, gap) <= (minimum + 1e-12) ** 2
            rows, indices = rows[keep], indices[keep]
            if len(indices):
                other, queried = self.segments[indices], batch[rows]
                distances = segment_distances(
                    queried[:, 0], queried[:, 1] - queried[:, 0], other[:, 0], other[:, 1] - other[:, 0]
                )
                minimum = min(minimum, float(np.min(distances)))
        return minimum


def _minimum_speed(controls, normal, height, peak):
    """Check all polynomial stationary points of speed on both quintic spans."""
    coefficients = np.array(
        [
            controls[0],
            3 * (controls[1] - controls[0]),
            3 * (controls[2] - 2 * controls[1] + controls[0]),
            controls[3] - 3 * controls[2] + 3 * controls[1] - controls[0],
        ]
    )
    lift = height * Polynomial([0, 0, 0, 10, -15, 6])
    minimum = np.inf
    for origin, width in ((0.0, peak), (1.0, -(1.0 - peak))):
        parameter = Polynomial([origin, width])
        derivatives = [
            (Polynomial(coefficients[:, axis])(parameter) + normal[axis] * lift).deriv() / width for axis in range(3)
        ]
        # Normalize coefficients before root finding; physical scaling should
        # not alter the speed critical points or the cusp decision.
        scale = max(float(np.max(np.abs(poly.coef))) for poly in derivatives)
        if scale == 0:
            return 0.0
        squared = sum((poly / scale) ** 2 for poly in derivatives)
        roots = squared.deriv().roots()
        interior = [float(root.real) for root in roots if abs(root.imag) <= 1e-7 and 0 < root.real < 1]
        samples = np.asarray([0.0, 1.0, *interior])
        # Evaluate the derivative components, not their expanded squared
        # polynomial, to avoid cancellation close to a zero-speed point.
        speed = np.hypot.reduce(np.column_stack([poly(samples) for poly in derivatives]), axis=1)
        minimum = min(minimum, float(np.min(speed)))
    return minimum


def _self_contact_reason(points, tolerance):
    starts, ends = points[:-1], points[1:]
    pairs = _segment_bvh_candidate_pairs(starts, ends, tolerance, max_pairs=100000)
    if pairs is None:
        return "self_contact_pair_budget_exceeded"
    if not len(pairs):
        return ""
    minima, maxima = np.minimum(starts, ends), np.maximum(starts, ends)
    first, second = pairs.T
    gap = np.maximum(0, np.maximum(minima[first] - maxima[second], minima[second] - maxima[first]))
    keep = (second > first + 1) & (np.einsum("ij,ij->i", gap, gap) <= tolerance**2)
    first, second = first[keep], second[keep]
    if not len(first):
        return ""
    distances = segment_distances(
        starts[first], ends[first] - starts[first], starts[second], ends[second] - starts[second]
    )
    return "curve_self_intersection" if np.any(distances <= tolerance) else ""


def _rebalance_parameters(controls, normal, height, peak, segment_length, chord_error, angle_deg):
    """Distribute knots by physical length/rotation, then verify each interval.

    Very short accepted leaves can produce poor sweep triangles despite a
    smooth curve. Every proposed interval is checked directly on analytic
    points, tangents and guide frames; the arch peak and endpoints remain exact.
    """
    cosine = np.cos(np.radians(angle_deg))

    def admissible(low, high):
        sampled = np.linspace(low, high, 9)
        points, first, _, _ = _evaluate(controls, normal, height, peak, sampled)
        tangents = first / np.linalg.norm(first, axis=1)[:, None]
        frames = normal - (tangents @ normal)[:, None] * tangents
        lengths = np.linalg.norm(frames, axis=1)
        if np.min(lengths) <= 1e-8:
            return False
        frames /= lengths[:, None]
        linear = points[0] + np.linspace(0, 1, 9)[:, None] * (points[-1] - points[0])
        return (
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum() <= segment_length
            and np.linalg.norm(points - linear, axis=1).max() <= chord_error
            and np.min(tangents @ tangents.T) >= cosine
            and np.min(frames @ frames.T) >= cosine
        )

    knots = [0.0]
    for low, high in ((0.0, peak), (peak, 1.0)):
        parameters = np.linspace(low, high, 2049)
        _, first, second, _ = _evaluate(controls, normal, height, peak, parameters)
        speed = np.linalg.norm(first, axis=1)
        tangents = first / speed[:, None]
        tangent_derivative = (second - tangents * np.einsum("ij,ij->i", tangents, second)[:, None]) / speed[:, None]
        along = tangents @ normal
        frames = normal - along[:, None] * tangents
        frame_lengths = np.linalg.norm(frames, axis=1)
        if np.min(frame_lengths) <= 1e-8:
            return None
        frames /= frame_lengths[:, None]
        derivative = -(tangent_derivative @ normal)[:, None] * tangents - along[:, None] * tangent_derivative
        frame_derivative = (derivative - frames * np.einsum("ij,ij->i", frames, derivative)[:, None]) / frame_lengths[
            :, None
        ]
        angle = np.radians(angle_deg)
        density = np.maximum.reduce(
            (
                speed / segment_length,
                np.linalg.norm(tangent_derivative, axis=1) / angle,
                np.linalg.norm(frame_derivative, axis=1) / angle,
                np.sqrt(np.linalg.norm(second, axis=1) / (8 * chord_error)),
            )
        )
        cumulative = np.r_[0.0, np.cumsum((density[1:] + density[:-1]) * np.diff(parameters) / 2)]
        count = max(1, int(np.ceil(cumulative[-1] / 0.97)))
        if len(knots) + count > 4096:
            return None
        proposed = np.interp(np.linspace(0, cumulative[-1], count + 1), cumulative, parameters)
        intervals = [(a, b, 0) for a, b in zip(proposed[:-1], proposed[1:])][::-1]
        while intervals:
            a, b, depth = intervals.pop()
            if admissible(a, b):
                knots.append(b)
            else:
                if depth >= 20 or len(knots) + len(intervals) >= 4096:
                    return None
                middle = (a + b) / 2
                intervals.extend(((middle, b, depth + 1), (a, middle, depth + 1)))
    return np.asarray(knots)


def _local_clearance(points, controls, normal, height, obstacles, *, index=None, cap=np.inf, consume=None):
    """Exact polyline distance score with a conservative hull broad phase."""
    if not len(obstacles):
        return np.inf
    # Shared attachment faces are continuous conductor, not remote obstacles.
    touching = np.any(np.linalg.norm(obstacles - points[0], axis=2) <= 1e-12, axis=1)
    touching |= np.any(np.linalg.norm(obstacles - points[-1], axis=2) <= 1e-12, axis=1)
    if np.all(touching):
        return np.inf
    if np.isfinite(cap):
        upper = cap
    else:
        remote = obstacles[~touching]
        directions = remote[:, 1] - remote[:, 0]
        squared = np.einsum("ij,ij->i", directions, directions)
        delta = points[len(points) // 2] - remote[:, 0]
        fractions = np.divide(
            np.einsum("ij,ij->i", delta, directions), squared, out=np.zeros(len(remote)), where=squared > 0
        )
        upper = float(np.min(np.linalg.norm(delta - np.clip(fractions, 0, 1)[:, None] * directions, axis=1)))
    index = index if index is not None else _SegmentDistanceIndex(obstacles)
    return index.gap(
        np.stack((points[:-1], points[1:]), axis=1), upper, excluded=np.flatnonzero(touching), consume=consume
    )


def generate_joint_transition_curves(
    source_points,
    tangent_before,
    tangent_after,
    normal,
    obstacle_segments,
    *,
    maximum_height_m: float,
    minimum_radius_m: float,
    maximum_segment_length_m: float,
    chord_tolerance_m: float,
    maximum_tangent_angle_deg: float = 2.0,
    maximum_deviation_m: float | None = None,
    maximum_candidates: int = 8,
    b0_direction=(0.0, 0.0, 1.0),
    b0_weight: float = 0.0,
    clearance_score_cap_m: float = np.inf,
    obstacle_index=None,
    distance_budget=None,
    search_level=None,
) -> BridgeGeneration:
    """Search at most 64 tangent-preserving cubic/quintic transition shapes.

    The physical signed normal sets the nonnegative added arch direction; it
    does not certify height above a curved support. Endpoint derivatives equal
    the specified directed tangents with positive scales. Radius is screened
    independently on 4097 analytic-derivative samples with a 0.5% margin, and
    speed critical points are checked polynomially to reject hidden cusps.
    Sampling uses the existing chord, tangent and guide-frame angle controls.
    Nonincident polyline contacts reject loops; complete swept-body validation
    must still check finite thickness and between-sample geometry.

    After clearance, rank the alternatives by length plus the supplied B0
    transverse-length penalty. The squared tangent projection treats +B0 and
    -B0 equally; zero weight disables that preference. Height diversity is
    retained because this local score cannot predict whole-body interactions.
    """
    validate_search_level(search_level)
    source = np.asarray(source_points, dtype=float)
    obstacles = np.asarray(obstacle_segments, dtype=float)
    vectors = [np.asarray(value, dtype=float) for value in (tangent_before, tangent_after, normal)]
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or len(source) < 2
        or obstacles.ndim != 3
        or obstacles.shape[1:] != (2, 3)
        or any(value.shape != (3,) for value in vectors)
        or not all(np.all(np.isfinite(value)) for value in (source, obstacles, *vectors))
    ):
        return BridgeGeneration(reason="invalid_geometry_or_input_budget")
    if len(source) > 4096 or len(obstacles) > 100000:
        return BridgeGeneration(reason="curve_input_budget_exceeded")
    try:
        routing_cost(source, b0_direction, b0_weight)
    except (TypeError, ValueError):
        return BridgeGeneration(reason="invalid_b0_objective")
    positive = (minimum_radius_m, maximum_segment_length_m, chord_tolerance_m, maximum_tangent_angle_deg)
    if (
        not all(np.isfinite(value) and value > 0 for value in positive)
        or not np.isfinite(maximum_height_m)
        or maximum_height_m < 0
        or not isinstance(maximum_candidates, int)
        or not 1 <= maximum_candidates <= 64
        or np.isnan(clearance_score_cap_m)
        or clearance_score_cap_m <= 0
        or (maximum_deviation_m is not None and (not np.isfinite(maximum_deviation_m) or maximum_deviation_m <= 0))
    ):
        return BridgeGeneration(reason="invalid_limits")
    vector_lengths = np.linalg.norm(vectors, axis=1)
    span = float(np.linalg.norm(source[-1] - source[0]))
    if min(vector_lengths) <= 1e-15 or span <= 1e-12:
        return BridgeGeneration(reason="degenerate_span_or_tangent")
    incoming, outgoing, signed_normal = np.asarray(vectors) / vector_lengths[:, None]
    obstacle_index = obstacle_index if obstacle_index is not None else _SegmentDistanceIndex(obstacles)
    handles = ((0.25, 0.25), (0.4, 0.4), (0.6, 0.6), (0.9, 0.9), (0.25, 0.6), (0.6, 0.25), (0.4, 0.9), (0.9, 0.4))
    profiles = ((1.0, 0.5), (0.75, 0.5), (0.5, 0.5), (0.0, 0.5), (1.0, 0.35), (1.0, 0.65), (0.5, 0.35), (0.5, 0.65))
    if search_level == 0:
        handles = ((0.25, 0.25), (0.6, 0.6))
        profiles = ((0.0, 0.5), (1.0, 0.5))
    elif search_level == 1:
        handles = handles[:6]
        profiles = ((0.0, 0.5), (0.5, 0.5), (1.0, 0.5))
    candidates, rejected, seen_shapes = [], set(), set()
    for before, after in handles:
        controls = np.array(
            [source[0], source[0] + incoming * span * before, source[-1] - outgoing * span * after, source[-1]]
        )
        for fraction, peak in profiles:
            height = float(maximum_height_m) * fraction
            peak = peak if height > 0 else 0.5
            shape = (before, after, height, peak)
            if shape in seen_shapes:
                continue
            seen_shapes.add(shape)
            if _minimum_speed(controls, signed_normal, height, peak) <= span * 1e-8:
                rejected.add("curve_cusp_or_zero_speed")
                continue
            dense = np.unique(np.r_[np.linspace(0, peak, 2049), np.linspace(peak, 1, 2049)])
            dense_points, first, second, _ = _evaluate(controls, signed_normal, height, peak, dense)
            speeds = np.linalg.norm(first, axis=1)
            curvature = np.linalg.norm(np.cross(first, second), axis=1) / speeds**3
            radius = 1 / max(float(np.max(curvature)), np.finfo(float).tiny)
            if radius < minimum_radius_m * 1.005:
                rejected.add("insufficient_bend_radius")
                continue
            parameters = _adaptive_parameters(
                controls,
                signed_normal,
                height,
                peak,
                maximum_segment_length_m,
                chord_tolerance_m,
                min(maximum_tangent_angle_deg, 2.0),
            )
            if parameters is None:
                rejected.add("sampling_budget_or_singular_guide")
                continue
            parameters = _rebalance_parameters(
                controls,
                signed_normal,
                height,
                peak,
                maximum_segment_length_m,
                chord_tolerance_m,
                min(maximum_tangent_angle_deg, 2.0),
            )
            if parameters is None:
                rejected.add("sampling_budget_or_singular_guide")
                continue
            curve, _first, _second, lift = _evaluate(controls, signed_normal, height, peak, parameters)
            curve[[0, -1]] = source[[0, -1]]
            reason = _self_contact_reason(curve, max(span * 1e-10, 1e-12))
            if reason:
                if "budget_exceeded" in reason:
                    return BridgeGeneration(reason=reason, rejected_reasons=tuple(sorted(rejected | {reason})))
                rejected.add(reason)
                continue
            deviation = _maximum_distance_to_polyline(dense_points, source)
            if maximum_deviation_m is not None and deviation > maximum_deviation_m:
                rejected.add("deviation_limit")
                continue
            baseline = _baseline(controls, dense)[0]
            arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(baseline, axis=0), axis=1))]
            fractions = np.interp(parameters, dense, arclength / arclength[-1])
            fractions[[0, -1]] = [0.0, 1.0]
            try:
                clearance = _local_clearance(
                    curve,
                    controls,
                    signed_normal,
                    height,
                    obstacles,
                    index=obstacle_index,
                    cap=clearance_score_cap_m,
                    consume=distance_budget,
                )
            except _DistanceBudgetExceeded as error:
                return BridgeGeneration(reason=str(error), rejected_reasons=tuple(sorted(rejected | {str(error)})))
            candidates.append(
                BridgeCandidate(
                    curve,
                    fractions,
                    lift,
                    parameters,
                    controls,
                    signed_normal.copy(),
                    height,
                    peak,
                    radius,
                    deviation,
                    clearance,
                )
            )
    candidates.sort(
        key=lambda candidate: (
            -round(candidate.local_clearance_m, 12),
            round(routing_cost(candidate.points, b0_direction, b0_weight), 12),
            candidate.maximum_deviation_m,
            -candidate.minimum_radius_m,
        )
    )
    # Preserve a zero-height and lower-height alternative when available.
    # Otherwise shared endpoint-neighbour gaps can make an arbitrary tiny
    # clearance-score tie remove every option with a smaller field change.
    selected = candidates[:1]
    heights = sorted({candidate.height_m for candidate in candidates})
    for height in heights:
        option = next(candidate for candidate in candidates if candidate.height_m == height)
        if all(option is not value for value in selected) and len(selected) < maximum_candidates:
            selected.append(option)
    for candidate in candidates:
        if len(selected) >= maximum_candidates:
            break
        if all(candidate is not value for value in selected):
            selected.append(candidate)
    return BridgeGeneration(
        tuple(selected),
        "candidates_available" if candidates else "no_feasible_joint_transition",
        tuple(sorted(rejected)),
    )


def _corridor_waypoints(source, normal, b0_direction, tolerance):
    """At most three deterministic corridors, with no world-axis preference.

    A greedy polyline approximation retains the largest source bends first.
    The two additional doglegs traverse tangential B0 and its complement in
    opposite orders. They are candidate corridors, not a free-space proof.
    """
    source = np.asarray(source)
    chosen = [0, len(source) - 1]
    while len(chosen) < 10:
        worst, split = tolerance, None
        for first, last in zip(chosen[:-1], chosen[1:]):
            if last <= first + 1:
                continue
            delta = source[last] - source[first]
            squared = float(delta @ delta)
            points = source[first + 1 : last]
            fractions = np.clip((points - source[first]) @ delta / squared, 0, 1) if squared else np.zeros(len(points))
            distances = np.linalg.norm(points - source[first] - fractions[:, None] * delta, axis=1)
            index = int(np.argmax(distances))
            if distances[index] > worst:
                worst, split = float(distances[index]), first + 1 + index
        if split is None:
            break
        chosen.append(split)
        chosen.sort()
    corridors = [("support_waypoints", source[chosen])] if len(chosen) > 2 else []
    b0 = np.array(b0_direction, dtype=float, copy=True)
    b0 /= np.max(np.abs(b0))
    b0 /= np.linalg.norm(b0)
    axial = b0 - (b0 @ normal) * normal
    if np.linalg.norm(axial) > 1e-10:
        axial /= np.linalg.norm(axial)
        delta = source[-1] - source[0]
        along = (delta @ axial) * axial
        across = delta - along
        if min(np.linalg.norm(along), np.linalg.norm(across)) > tolerance:
            for component in (along, across):
                corridors.append(("b0_corridor", np.array([source[0], source[0] + component, source[-1]])))
    return corridors


AXIAL_RETURN_SPAN_TRIALS = 24  # two directions, two heights, two handles, three spans


def generate_axial_return_curves(
    source_points,
    tangent_before,
    tangent_after,
    normal,
    obstacle_segments,
    *,
    maximum_height_m,
    minimum_radius_m,
    maximum_segment_length_m,
    chord_tolerance_m,
    maximum_tangent_angle_deg=2.0,
    maximum_candidates=8,
    b0_direction=(0, 0, 1),
    b0_weight=0.0,
    clearance_score_cap_m=np.inf,
    obstacle_index=None,
    distance_budget=None,
    search_level=None,
):
    """Propose two smooth bends joined by a genuinely straight +/-B0 span.

    The middle cubic is exactly a line in the supplied three-dimensional B0
    direction, not its projection onto the surface. Both signed directions are
    explored. Every attachment and internal join preserves directed tangency
    (G1); no G2 claim is made. The full bends are retained in the resulting
    geometry, objective and external support/finite-body/field checks.

    Eight compound candidates require at most 24 analytic span trials. A
    support-normal B0 may admit no useful span within the permitted height;
    that is reported as a bounded-family failure, never repaired by rotating B0.
    """
    validate_search_level(search_level)
    source, obstacles = np.asarray(source_points, dtype=float), np.asarray(obstacle_segments, dtype=float)
    vectors = [np.asarray(v, dtype=float) for v in (tangent_before, tangent_after, normal, b0_direction)]
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or len(source) < 2
        or obstacles.ndim != 3
        or obstacles.shape[1:] != (2, 3)
        or any(v.shape != (3,) or not np.all(np.isfinite(v)) or not np.any(v) for v in vectors)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(obstacles))
    ):
        return BridgeGeneration(reason="invalid_geometry_or_input_budget")
    if len(source) > 4096 or len(obstacles) > 100000:
        return BridgeGeneration(reason="curve_input_budget_exceeded")
    if (
        not all(
            np.isfinite(v) and v > 0
            for v in (minimum_radius_m, maximum_segment_length_m, chord_tolerance_m, maximum_tangent_angle_deg)
        )
        or not np.isfinite(maximum_height_m)
        or maximum_height_m < 0
        or isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or not 1 <= maximum_candidates <= 64
        or np.isnan(clearance_score_cap_m)
        or clearance_score_cap_m <= 0
    ):
        return BridgeGeneration(reason="invalid_limits")
    try:
        routing_cost(source, b0_direction, b0_weight)
    except (ValueError, TypeError):
        return BridgeGeneration(reason="invalid_b0_objective")
    incoming, outgoing, guide, axis = [v / np.max(np.abs(v)) for v in vectors]
    incoming, outgoing, guide, axis = [v / np.linalg.norm(v) for v in (incoming, outgoing, guide, axis)]
    chord = source[-1] - source[0]
    span = float(np.linalg.norm(chord))
    if span <= 1e-12:
        return BridgeGeneration(reason="degenerate_span_or_tangent")
    # Canonical ordering depends on physical vectors, never a world coordinate.
    for reference in (chord / span, incoming, outgoing, guide):
        projection = float(axis @ reference)
        if abs(projection) > 1e-10:
            axis *= 1 if projection > 0 else -1
            break
    obstacle_index = obstacle_index if obstacle_index is not None else _SegmentDistanceIndex(obstacles)
    candidates, rejected = [], set()
    desired = max(0.6 * abs(float(chord @ axis)), 0.3 * span)
    for height in sorted({0.0, float(maximum_height_m)}):
        normal_component = abs(float(axis @ guide))
        straight_length = desired if normal_component < 1e-10 else min(desired, 0.8 * height / normal_component)
        if straight_length <= max(1e-12, chord_tolerance_m):
            continue
        offset = height if normal_component < 1e-10 else height / 2
        center = (source[0] + source[-1]) / 2 + offset * guide
        for direction in (axis, -axis):
            first, last = center - straight_length / 2 * direction, center + straight_length / 2 * direction
            for ratio in (0.5,) if search_level == 0 else (0.25, 0.5):
                left = ratio * float(np.linalg.norm(first - source[0]))
                right = ratio * float(np.linalg.norm(source[-1] - last))
                controls = np.array(
                    [
                        [source[0], source[0] + left * incoming, first - left * direction, first],
                        [first, first + (last - first) / 3, first + 2 * (last - first) / 3, last],
                        [last, last + right * direction, source[-1] - right * outgoing, source[-1]],
                    ]
                )
                # A vertical straight span must not be rejected merely because
                # the surface-normal guide is parallel to its tangent. Choose
                # a regular constant guide for sampling; support uses `guide`.
                derivatives = np.vstack([_baseline(c, np.linspace(0, 1, 65))[1] for c in controls])
                speed = np.linalg.norm(derivatives, axis=1)
                if np.min(speed) <= span * 1e-8:
                    rejected.add("curve_cusp_or_zero_speed")
                    continue
                tangents = derivatives / speed[:, None]
                guides = [
                    guide,
                    np.cross(direction, chord),
                    np.cross(direction, incoming),
                    np.cross(direction, outgoing),
                ]
                guides = [v / np.linalg.norm(v) for v in guides if np.linalg.norm(v) > 1e-12]
                sample_guide = max(guides, key=lambda v: float(np.min(1 - (tangents @ v) ** 2)))
                pieces, radii, reason = [], [], ""
                for control in controls:
                    if _minimum_speed(control, sample_guide, 0.0, 0.5) <= span * 1e-8:
                        reason = "curve_cusp_or_zero_speed"
                        break
                    _, derivative, second = _baseline(control, np.linspace(0, 1, 4097))
                    curvature = (
                        np.linalg.norm(np.cross(derivative, second), axis=1) / np.linalg.norm(derivative, axis=1) ** 3
                    )
                    radius = 1 / max(float(np.max(curvature)), np.finfo(float).tiny)
                    if radius < minimum_radius_m * 1.005:
                        reason = "insufficient_bend_radius"
                        break
                    knots = _rebalance_parameters(
                        control,
                        sample_guide,
                        0.0,
                        0.5,
                        maximum_segment_length_m,
                        chord_tolerance_m,
                        min(maximum_tangent_angle_deg, 2.0),
                    )
                    if knots is None:
                        reason = "sampling_budget_or_singular_guide"
                        break
                    points = _baseline(control, knots)[0]
                    points[[0, -1]] = control[[0, -1]]
                    pieces.append(points if not pieces else points[1:])
                    radii.append(radius)
                if reason:
                    rejected.add(reason)
                    continue
                points = np.vstack(pieces)
                if len(points) > 4096:
                    return BridgeGeneration(
                        reason="axial_sample_budget_exceeded", rejected_reasons=tuple(sorted(rejected))
                    )
                points[[0, -1]] = source[[0, -1]]
                reason = _self_contact_reason(points, max(span * 1e-10, 1e-12))
                if reason:
                    if "budget_exceeded" in reason:
                        return BridgeGeneration(reason=reason, rejected_reasons=tuple(sorted(rejected)))
                    rejected.add(reason)
                    continue
                try:
                    clearance = _local_clearance(
                        points,
                        controls,
                        guide,
                        height,
                        obstacles,
                        index=obstacle_index,
                        cap=clearance_score_cap_m,
                        consume=distance_budget,
                    )
                except _DistanceBudgetExceeded as error:
                    return BridgeGeneration(reason=str(error), rejected_reasons=tuple(sorted(rejected)))
                arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
                parameters = arclength / arclength[-1]
                candidates.append(
                    BridgeCandidate(
                        points,
                        parameters,
                        np.zeros(len(points)),
                        parameters,
                        controls,
                        guide.copy(),
                        height,
                        0.5,
                        min(radii),
                        _maximum_distance_to_polyline(points, source),
                        clearance,
                        profile_family="b0_straight_return",
                    )
                )
    candidates.sort(
        key=lambda c: (-round(c.local_clearance_m, 12), round(routing_cost(c.points, b0_direction, b0_weight), 12))
    )
    return BridgeGeneration(
        tuple(candidates[:maximum_candidates]),
        "candidates_available" if candidates else "no_feasible_axial_return",
        tuple(sorted(rejected)),
    )


def generate_joint_corridor_curves(
    source_points,
    tangent_before,
    tangent_after,
    normal,
    obstacle_segments,
    *,
    maximum_height_m,
    minimum_radius_m,
    maximum_segment_length_m,
    chord_tolerance_m,
    maximum_tangent_angle_deg=2.0,
    maximum_candidates=8,
    b0_direction=(0, 0, 1),
    b0_weight=0.0,
    clearance_score_cap_m=np.inf,
    obstacle_index=None,
    distance_budget=None,
    search_level=None,
):
    """Bounded multi-span alternatives when direct arches have no corridor.

    Up to three corridors use three handle ratios and zero/full permitted
    lift. There are at most nine cubic spans each (162 analytic span trials).
    Every span uses the direct family's analytic cusp/radius and adaptive
    chord/frame checks. Consecutive derivatives have the same directed tangent
    with positive scales (G1, not a G2 curvature-continuity claim). Support and
    complete conductor checks remain external.

    ``control_points_m`` is an array of cubic controls for these compound
    candidates, identified by ``profile_family``; no single-cubic certificate
    is claimed for them.
    """
    validate_search_level(search_level)
    source = np.asarray(source_points, dtype=float)
    obstacles = np.asarray(obstacle_segments, dtype=float)
    incoming, outgoing, guide = (np.asarray(value, dtype=float) for value in (tangent_before, tangent_after, normal))
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or len(source) < 2
        or obstacles.ndim != 3
        or obstacles.shape[1:] != (2, 3)
        or any(
            value.shape != (3,) or not np.all(np.isfinite(value)) or np.linalg.norm(value) <= 1e-15
            for value in (incoming, outgoing, guide)
        )
        or not np.all(np.isfinite(source))
    ):
        return BridgeGeneration(reason="invalid_geometry_or_input_budget")
    if len(source) > 4096 or len(obstacles) > 100000:
        return BridgeGeneration(reason="curve_input_budget_exceeded")
    if (
        not all(
            np.isfinite(value) and value > 0
            for value in (minimum_radius_m, maximum_segment_length_m, chord_tolerance_m, maximum_tangent_angle_deg)
        )
        or not np.isfinite(maximum_height_m)
        or maximum_height_m < 0
        or not isinstance(maximum_candidates, int)
        or not 1 <= maximum_candidates <= 64
        or not np.all(np.isfinite(obstacles))
        or np.isnan(clearance_score_cap_m)
        or clearance_score_cap_m <= 0
    ):
        return BridgeGeneration(reason="invalid_limits")
    try:
        routing_cost(source, b0_direction, b0_weight)
    except (TypeError, ValueError):
        return BridgeGeneration(reason="invalid_b0_objective")
    incoming, outgoing, guide = [value / np.linalg.norm(value) for value in (incoming, outgoing, guide)]
    corridors = _corridor_waypoints(source, guide, b0_direction, max(minimum_radius_m / 5, chord_tolerance_m))
    obstacle_index = obstacle_index if obstacle_index is not None else _SegmentDistanceIndex(obstacles)
    rejected, candidates = set(), []
    for family, baseline in corridors:
        lengths = np.linalg.norm(np.diff(baseline, axis=0), axis=1)
        if np.any(lengths <= 1e-12):
            rejected.add("degenerate_corridor_span")
            continue
        arclength = np.r_[0.0, np.cumsum(lengths)]
        fractions = arclength / arclength[-1]
        phase = 2 * np.minimum(fractions, 1 - fractions)
        profile = phase**3 * (10 - 15 * phase + 6 * phase**2)
        for height in sorted({0.0, float(maximum_height_m)}):
            waypoints = baseline + height * profile[:, None] * guide
            lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
            if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1e-12):
                rejected.add("degenerate_corridor_span")
                continue
            directions = np.diff(waypoints, axis=0) / lengths[:, None]
            interior = directions[:-1] + directions[1:]
            norms = np.linalg.norm(interior, axis=1)
            if np.any(norms <= 1e-10):
                rejected.add("corridor_reversal")
                continue
            tangents = np.vstack((incoming, interior / norms[:, None], outgoing))
            for ratio in (0.4,) if search_level == 0 else (0.25, 0.6) if search_level == 1 else (0.25, 0.4, 0.6):
                pieces, controls_list, radii, reason = [], [], [], ""
                for index, length in enumerate(lengths):
                    controls = np.array(
                        [
                            waypoints[index],
                            waypoints[index] + ratio * length * tangents[index],
                            waypoints[index + 1] - ratio * length * tangents[index + 1],
                            waypoints[index + 1],
                        ]
                    )
                    if _minimum_speed(controls, guide, 0.0, 0.5) <= length * 1e-8:
                        reason = "curve_cusp_or_zero_speed"
                        break
                    dense = np.linspace(0, 1, 4097)
                    _, first, second, _ = _evaluate(controls, guide, 0.0, 0.5, dense)
                    curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.linalg.norm(first, axis=1) ** 3
                    radius = 1 / max(float(np.max(curvature)), np.finfo(float).tiny)
                    if radius < minimum_radius_m * 1.005:
                        reason = "insufficient_bend_radius"
                        break
                    parameters = _rebalance_parameters(
                        controls,
                        guide,
                        0.0,
                        0.5,
                        maximum_segment_length_m,
                        chord_tolerance_m,
                        min(maximum_tangent_angle_deg, 2.0),
                    )
                    if parameters is None:
                        reason = "sampling_budget_or_singular_guide"
                        break
                    points = _evaluate(controls, guide, 0.0, 0.5, parameters)[0]
                    points[[0, -1]] = waypoints[index : index + 2]
                    pieces.append(points if index == 0 else points[1:])
                    controls_list.append(controls)
                    radii.append(radius)
                if reason:
                    rejected.add(reason)
                    continue
                points = np.vstack(pieces)
                if len(points) > 4096:
                    return BridgeGeneration(
                        reason="corridor_sample_budget_exceeded", rejected_reasons=tuple(sorted(rejected))
                    )
                points[[0, -1]] = source[[0, -1]]
                reason = _self_contact_reason(points, max(arclength[-1] * 1e-10, 1e-12))
                if reason:
                    if "budget_exceeded" in reason:
                        return BridgeGeneration(reason=reason, rejected_reasons=tuple(sorted(rejected)))
                    rejected.add(reason)
                    continue
                try:
                    clearance = _local_clearance(
                        points,
                        controls_list[0],
                        guide,
                        height,
                        obstacles,
                        index=obstacle_index,
                        cap=clearance_score_cap_m,
                        consume=distance_budget,
                    )
                except _DistanceBudgetExceeded as error:
                    return BridgeGeneration(reason=str(error), rejected_reasons=tuple(sorted(rejected | {str(error)})))
                distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
                parameters = distances / distances[-1]
                candidates.append(
                    BridgeCandidate(
                        points,
                        parameters,
                        np.zeros(len(points)),
                        parameters,
                        np.asarray(controls_list),
                        guide.copy(),
                        height,
                        0.5,
                        min(radii),
                        _maximum_distance_to_polyline(points, source),
                        clearance,
                        profile_family=family,
                    )
                )
    candidates.sort(
        key=lambda candidate: (
            -round(candidate.local_clearance_m, 12),
            round(routing_cost(candidate.points, b0_direction, b0_weight), 12),
            candidate.profile_family,
            candidate.height_m,
        )
    )
    return BridgeGeneration(
        tuple(candidates[:maximum_candidates]),
        "candidates_available" if candidates else "no_feasible_corridor_family",
        tuple(sorted(rejected)),
    )
