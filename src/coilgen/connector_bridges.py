"""Bounded connector bridge proposals; acceptance belongs to the routing validator.

Only the owned connector is reconstructed. A candidate is not a clearance or
manufacturing certificate: its entire route and finite conductor still need
checking against the physical support and every other conductor.
"""

from dataclasses import dataclass

import numpy as np

from .geometry_kernels import segment_distances


@dataclass(frozen=True)
class BridgeCandidate:
    points: np.ndarray
    source_fractions: np.ndarray
    lift_distances_m: np.ndarray
    parameters: np.ndarray
    control_points_m: np.ndarray
    normal: np.ndarray
    height_m: float
    peak_parameter: float
    minimum_radius_m: float
    maximum_deviation_m: float
    local_clearance_m: float
    profile_family: str = "quintic"


@dataclass(frozen=True)
class BridgeGeneration:
    candidates: tuple[BridgeCandidate, ...] = ()
    reason: str = ""
    rejected_reasons: tuple[str, ...] = ()


def generate_connector_bridges(
    source_points,
    tangent_before,
    tangent_after,
    normal,
    underpass_segments,
    *,
    maximum_height_m: float,
    minimum_radius_m: float,
    maximum_segment_length_m: float,
    chord_tolerance_m: float,
    maximum_tangent_angle_deg: float = 2.0,
    maximum_deviation_m: float | None = None,
    planarity_tolerance_m: float = 1e-8,
    spatial_mode: bool = False,
) -> BridgeGeneration:
    """Propose a fixed-end connector arch; independently validate every candidate."""
    return _generate_connector_curves(
        source_points,
        tangent_before,
        tangent_after,
        normal,
        underpass_segments,
        maximum_height_m=maximum_height_m,
        minimum_radius_m=minimum_radius_m,
        maximum_segment_length_m=maximum_segment_length_m,
        chord_tolerance_m=chord_tolerance_m,
        maximum_tangent_angle_deg=maximum_tangent_angle_deg,
        maximum_deviation_m=maximum_deviation_m,
        planarity_tolerance_m=planarity_tolerance_m,
        spatial_mode=spatial_mode,
    )


def generate_return_ramps(source_points, tangent_before, tangent_after, normal, underpass_segments, **limits):
    """Propose 64 ramps between the actual unlifted and lifted attachments.

    Endpoints and neighboring 3D tangents remain exact. ``normal`` is the signed
    average physical endpoint face normal. The cubic baseline already includes
    the endpoint lift; ``lift_distances_m`` records only the added hump, not total
    support distance. The caller must validate the actual support, all other
    conductors, and the final finite body. Early/late humps permit clearance of
    a physical support crease without overshooting the lifted attachment.
    """
    return _generate_connector_curves(
        source_points,
        tangent_before,
        tangent_after,
        normal,
        underpass_segments,
        spatial_mode=True,
        return_ramp=True,
        **limits,
    )


def _generate_connector_curves(
    source_points,
    tangent_before,
    tangent_after,
    normal,
    underpass_segments,
    *,
    maximum_height_m: float,
    minimum_radius_m: float,
    maximum_segment_length_m: float,
    chord_tolerance_m: float,
    maximum_tangent_angle_deg: float = 2.0,
    maximum_deviation_m: float | None = None,
    planarity_tolerance_m: float = 1e-8,
    spatial_mode: bool = False,
    return_ramp: bool = False,
) -> BridgeGeneration:
    """Propose fixed-end, tangent-continuous bridges in a bounded shape search.

    ``normal`` must already have the requested physical lift sign. Heights are
    nonnegative and never exceed ``maximum_height_m``. The radius input must
    include the caller's conductor-envelope margin. Planarity of this source
    span does not certify the support domain or its boundary.

    ``source_fractions`` maps normalized *baseline* arclength to normalized
    source arclength, monotonically. It records correspondence, not unchanged
    intermediate source coordinates. Collinear source subdivision has no
    effect on candidate geometry. The analytic curve has a C2 normal profile;
    output sampling bounds chord error and tangent/support-normal frame angles.

    The default searches eight full-height coplanar shapes, then eight lower
    asymmetric proposals only if those fail. Explicit ``spatial_mode``
    searches 64 combinations of independent tangent handles and arch heights,
    keeping the full 3D endpoint tangents. In that mode the supplied normal is
    only the arch direction: arch height is not distance from the actual
    support. The routing validator must certify support distance, sidedness,
    footprint and intersections before accepting any spatial candidate.
    """
    points = np.asarray(source_points, dtype=float)
    underpasses = np.asarray(underpass_segments, dtype=float)
    vectors = [np.asarray(value, dtype=float) for value in (tangent_before, tangent_after, normal)]
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 2
        or underpasses.ndim != 3
        or underpasses.shape[1:] != (2, 3)
        or not len(underpasses)
        or any(value.shape != (3,) for value in vectors)
        or not all(np.all(np.isfinite(value)) for value in [points, underpasses, *vectors])
    ):
        return BridgeGeneration(reason="invalid_geometry")
    limits = [
        maximum_height_m,
        minimum_radius_m,
        maximum_segment_length_m,
        chord_tolerance_m,
        maximum_tangent_angle_deg,
        planarity_tolerance_m,
    ]
    if not all(np.isfinite(value) and value > 0 for value in limits):
        return BridgeGeneration(reason="invalid_limits")
    if maximum_deviation_m is not None and (not np.isfinite(maximum_deviation_m) or maximum_deviation_m <= 0):
        return BridgeGeneration(reason="invalid_limits")
    lengths = [float(np.linalg.norm(value)) for value in vectors]
    chord = points[-1] - points[0]
    span = float(np.linalg.norm(chord))
    if min(lengths) <= 1e-15 or span <= 1e-12:
        return BridgeGeneration(reason="degenerate_span_or_tangent")
    incoming, outgoing, signed_normal = [value / length for value, length in zip(vectors, lengths)]
    if not spatial_mode and np.max(np.abs((points - points[0]) @ signed_normal)) > planarity_tolerance_m:
        return BridgeGeneration(reason="unsupported_nonplanar_span")
    if (
        not spatial_mode
        and max(abs(incoming @ signed_normal), abs(outgoing @ signed_normal)) * span > planarity_tolerance_m
    ):
        return BridgeGeneration(reason="unsupported_nonplanar_tangent")
    if min(incoming @ chord, outgoing @ chord) <= 1e-12 * span:
        return BridgeGeneration(reason="unsupported_reentrant_tangent")

    candidates = []
    rejected = set()
    if return_ramp:
        handle_pairs = (
            (0.25, 0.45),
            (0.45, 0.25),
            (0.35, 0.35),
            (0.35, 0.45),
            (0.45, 0.35),
            (0.45, 0.45),
            (0.45, 0.65),
            (0.65, 0.45),
        )
        profiles = (
            (0, "quartic", None),
            (1 / 3, "quartic", 0.35),
            (1 / 3, "quartic", 0.65),
            (1 / 3, "quintic", None),
            (0.5, "quintic", None),
            (0.8, "quintic", 0.35),
            (0.8, "quintic", 0.65),
            (0.8, "quintic", None),
        )
        shapes = [(a, b, height, family, peak) for a, b in handle_pairs for height, family, peak in profiles]
    elif spatial_mode:
        shapes = [
            (a, b, height, "quartic", None)
            for height in (1.0, 0.95, 0.85, 2.0 / 3.0)
            for a in (0.35, 0.45, 0.55, 0.65)
            for b in (0.35, 0.45, 0.55, 0.65)
        ]
    else:
        shapes = [(value, value, 1.0, "quintic", None) for value in (0.28, 0.34, 0.40, 0.46, 0.52, 0.55, 0.58, 0.64)]
        # A short, asymmetric approach can lack room for the full-height ramp
        # while retaining room for a lower, tangent-continuous crossing. These
        # remain proposals: the routing validator decides whether their gaps
        # are acceptable, and the finite conductor still needs its own check.
        shapes += [
            (first, last, height, "quintic", None)
            for a, b, height in ((0.4, 0.25, 0.7), (0.45, 0.25, 2 / 3), (0.55, 0.25, 2 / 3), (0.8, 0.35, 2 / 3))
            for first, last in ((a, b), (b, a))
        ]
    for shape_index, (first_handle, last_handle, height_fraction, family, fixed_peak) in enumerate(shapes):
        if not spatial_mode and shape_index == 8 and candidates:
            break  # Preserve existing full-height proposals when they suffice.
        height = float(maximum_height_m) * height_fraction
        controls = np.array(
            [
                points[0],
                points[0] + incoming * span * first_handle,
                points[-1] - outgoing * span * last_handle,
                points[-1],
            ]
        )
        peak, reason = _crossing_parameter(controls, signed_normal, underpasses)
        if reason:
            rejected.add(reason)
            continue
        if fixed_peak is not None:
            peak = fixed_peak
        if not 0.03 < peak < 0.97:
            rejected.add("crossing_too_close_to_attachment")
            continue
        # Independent of output sampling. The peak and both one-sided shoulder
        # grids are explicit, so an asymmetric profile cannot hide its short ramp.
        dense = np.unique(np.r_[np.linspace(0, peak, 2049), np.linspace(peak, 1, 2049)])
        dense_points, first, second, _height = _evaluate(controls, signed_normal, height, peak, dense, family)
        speed = np.linalg.norm(first, axis=1)
        if np.min(speed) <= span * 1e-8 or np.min(first @ chord) <= span * span * 1e-8:
            rejected.add("curve_reversal")
            continue
        curvature = np.linalg.norm(np.cross(first, second), axis=1) / speed**3
        radius = 1.0 / max(float(curvature.max()), np.finfo(float).tiny)
        # Reserve 0.5% for finite sampling of analytic curvature extrema.
        if radius < minimum_radius_m * 1.005:
            rejected.add("insufficient_bend_radius")
            continue
        sampled = _adaptive_parameters(
            controls,
            signed_normal,
            height,
            peak,
            maximum_segment_length_m,
            chord_tolerance_m,
            min(maximum_tangent_angle_deg, 2.0),
            family,
        )
        if sampled is None:
            rejected.add("sampling_budget_exceeded")
            continue
        curve, _first, _second, lift = _evaluate(controls, signed_normal, height, peak, sampled, family)
        curve[[0, -1]] = points[[0, -1]]  # Bitwise preserve the shared attachments.
        deviation = _maximum_distance_to_polyline(dense_points, points)
        if maximum_deviation_m is not None and deviation > maximum_deviation_m:
            rejected.add("deviation_limit")
            continue
        baseline = _baseline(controls, dense)[0]
        baseline_arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(baseline, axis=0), axis=1))]
        fractions = np.interp(sampled, dense, baseline_arclength / baseline_arclength[-1])
        fractions[[0, -1]] = [0.0, 1.0]
        candidates.append(
            BridgeCandidate(
                curve,
                fractions,
                lift,
                sampled,
                controls,
                signed_normal.copy(),
                height,
                peak,
                radius,
                deviation,
                _curve_clearance(curve, underpasses),
                family,
            )
        )
    if spatial_mode or (candidates and candidates[0].height_m < maximum_height_m):
        candidates.sort(
            key=lambda candidate: (
                -candidate.local_clearance_m,
                candidate.maximum_deviation_m,
                -candidate.minimum_radius_m,
            )
        )
    else:
        candidates.sort(key=lambda candidate: (candidate.maximum_deviation_m, -candidate.minimum_radius_m))
    return BridgeGeneration(
        tuple(candidates), "candidates_available" if candidates else "no_feasible_bridge", tuple(sorted(rejected))
    )


def _baseline(controls, parameters):
    t = np.asarray(parameters, dtype=float).reshape(-1, 1)
    q = 1 - t
    points = q**3 * controls[0] + 3 * q * q * t * controls[1] + 3 * q * t * t * controls[2] + t**3 * controls[3]
    first = 3 * (
        q * q * (controls[1] - controls[0])
        + 2 * q * t * (controls[2] - controls[1])
        + t * t * (controls[3] - controls[2])
    )
    second = 6 * (q * (controls[2] - 2 * controls[1] + controls[0]) + t * (controls[3] - 2 * controls[2] + controls[1]))
    return points, first, second


def _evaluate(controls, normal, height, peak, parameters, family="quintic"):
    t = np.asarray(parameters, dtype=float).reshape(-1)
    if family == "quartic":
        # Rational time warp places a quartic arch's unique maximum at the
        # continuous crossing. Both endpoint slopes vanish, while finite
        # outward endpoint curvature can keep a 3D baseline above a convex
        # support crease. The arch is C-infinity throughout the closed span.
        denominator = peak + (1 - 2 * peak) * t
        u = (1 - peak) * t / denominator
        first_u = peak * (1 - peak) / denominator**2
        second_u = -2 * peak * (1 - peak) * (1 - 2 * peak) / denominator**3
        lift = 16 * height * u**2 * (1 - u) ** 2
        slope = 32 * height * u * (1 - u) * (1 - 2 * u)
        curvature = 32 * height * (1 - 6 * u + 6 * u**2)
        first_lift = slope * first_u
        second_lift = curvature * first_u**2 + slope * second_u
    else:
        rising = t <= peak
        width = np.where(rising, peak, 1 - peak)
        u = np.where(rising, t, 1 - t) / width
        u = np.clip(u, 0.0, 1.0)
        sign = np.where(rising, 1.0, -1.0)
        lift = height * u**3 * (10 + u * (-15 + 6 * u))
        first_lift = height * 30 * u * u * (1 - u) ** 2 * sign / width
        second_lift = height * 60 * u * (1 - u) * (1 - 2 * u) / width**2
    points, first, second = _baseline(controls, t)
    return (
        points + lift[:, None] * normal,
        first + first_lift[:, None] * normal,
        second + second_lift[:, None] * normal,
        lift,
    )


def _crossing_parameter(controls, normal, segments):
    # Exact roots of the cubic baseline crossing each projected underpass line.
    coefficients = np.array(
        [
            controls[0],
            3 * (controls[1] - controls[0]),
            3 * (controls[2] - 2 * controls[1] + controls[0]),
            controls[3] - 3 * controls[2] + 3 * controls[1] - controls[0],
        ]
    )
    roots = []
    for start, end in segments:
        direction = end - start
        planar_direction = direction - (direction @ normal) * normal
        squared = float(planar_direction @ planar_direction)
        if squared <= 1e-24:
            continue
        offset = coefficients.copy()
        offset[0] -= start
        polynomial = np.cross(offset, planar_direction) @ normal
        scale = np.max(np.abs(polynomial))
        if scale <= 1e-20:
            return 0.0, "unsupported_collinear_underpass"
        polynomial[np.abs(polynomial) < scale * 1e-12] = 0
        for root in np.polynomial.polynomial.polyroots(polynomial):
            if abs(root.imag) > 1e-8 or not 0 <= root.real <= 1:
                continue
            parameter = float(root.real)
            position = _baseline(controls, [parameter])[0][0]
            along = float((position - start) @ planar_direction / squared)
            if -1e-8 <= along <= 1 + 1e-8:
                roots.append(parameter)
    if roots:
        if max(roots) - min(roots) > 0.02:
            return 0.0, "unsupported_multiple_crossings"
        return float(np.mean(roots)), ""
    # Near contacts without a projected crossing: refine the closest baseline
    # location continuously instead of snapping the lift to a source vertex.
    grid = np.linspace(0, 1, 129)
    distances = _point_segment_minimum(_baseline(controls, grid)[0], segments)
    index = int(np.argmin(distances))
    left, right = grid[max(index - 1, 0)], grid[min(index + 1, len(grid) - 1)]
    ratio = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        first, second = right - ratio * (right - left), left + ratio * (right - left)
        values = _point_segment_minimum(_baseline(controls, [first, second])[0], segments)
        if values[0] < values[1]:
            right = second
        else:
            left = first
    return float((left + right) / 2), ""


def _adaptive_parameters(controls, normal, height, peak, segment_length, chord_error, angle_deg, family="quintic"):
    regular = np.linspace(0, 1, 17)
    # Polynomial crossing roots may differ by a few ulps from a regular knot.
    # Keeping both creates an effectively zero-length segment at the apex.
    knots = np.sort(np.r_[regular[np.abs(regular - peak) > 1e-12], peak])
    intervals = [(a, b, 0) for a, b in zip(knots[:-1], knots[1:])]
    result = [0.0]
    cosine = np.cos(np.radians(angle_deg))
    while intervals:
        pending = []
        # Interval predicates are independent. Batch the same five samples and
        # dyadic splits; sorting the terminal endpoints recovers scalar DFS
        # order. The leaf/depth limits remain identical to that traversal.
        leaf_count = len(result) - 1 + len(intervals)
        for offset in range(0, len(intervals), 128):
            batch = np.asarray(intervals[offset : offset + 128])
            parameters = np.linspace(batch[:, 0], batch[:, 1], 5, axis=1)
            points, first, _second, _lift = _evaluate(controls, normal, height, peak, parameters.ravel(), family)
            points, first = points.reshape(-1, 5, 3), first.reshape(-1, 5, 3)
            tangents = first / np.linalg.norm(first, axis=2)[:, :, None]
            frames = normal - (tangents @ normal)[:, :, None] * tangents
            frame_lengths = np.linalg.norm(frames, axis=2)
            if np.min(frame_lengths) <= 1e-8:
                return None
            frames /= frame_lengths[:, :, None]
            linear = points[:, :1] + np.linspace(0, 1, 5)[None, :, None] * (points[:, -1:] - points[:, :1])
            split = (
                (np.linalg.norm(np.diff(points, axis=1), axis=2).sum(axis=1) > segment_length)
                | (np.linalg.norm(points - linear, axis=2).max(axis=1) > chord_error)
                | (np.min(tangents @ tangents.swapaxes(1, 2), axis=(1, 2)) < cosine)
                | (np.min(frames @ frames.swapaxes(1, 2), axis=(1, 2)) < cosine)
            )
            leaf_count += int(np.count_nonzero(split))
            if leaf_count > 4096 or np.any(batch[split, 2] >= 20):
                return None
            result.extend(batch[~split, 1])
            for low, high, depth in batch[split]:
                middle = (low + high) / 2
                pending.extend(((low, middle, depth + 1), (middle, high, depth + 1)))
        intervals = pending
    return np.sort(result)


def _point_segment_minimum(points, segments):
    minimum = np.full(len(points), np.inf)
    # Linear memory even for a heavily subdivided original connector.
    for start, end in segments:
        direction = end - start
        squared = float(direction @ direction)
        fractions = np.zeros(len(points)) if squared == 0 else np.clip((points - start) @ direction / squared, 0, 1)
        minimum = np.minimum(minimum, np.linalg.norm(points - start - fractions[:, None] * direction, axis=1))
    return minimum


def _maximum_distance_to_polyline(points, source):
    return float(np.max(_point_segment_minimum(points, np.stack((source[:-1], source[1:]), axis=1))))


def _curve_clearance(points, segments):
    minimum = np.inf
    directions = np.diff(points, axis=0)
    for start, end in segments:
        distances = segment_distances(
            points[:-1],
            directions,
            np.broadcast_to(start, directions.shape),
            np.broadcast_to(end - start, directions.shape),
        )
        minimum = min(minimum, float(distances.min()))
    return minimum
