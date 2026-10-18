"""Bounded tangent-preserving nominal curves on verified frustum developments.

Each physical facet is unfolded isometrically, in metres. Cubic and two-bend
families are sampled at their exact facet crossings before mapping back. Facet
creases, finite-body clearance, support, mounting, bends and fields remain the
independent production validator's responsibility. No analytic-cone retraction
or discontinuous facet-normal lift is introduced here; spatial bridge families
continue to own lifted curves and their physical height checks.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .connector_bridges import BridgeCandidate, BridgeGeneration, _baseline, _maximum_distance_to_polyline
from .developed_frustum import FrustumChart, _cross
from .joint_curve_search import cylinder_trial_count, validate_search_level
from .joint_surface_curves import PLANAR_RETURN_TRIALS, _turn, generate_planar_return_curves
from .joint_transition_curves import _self_contact_reason, generate_joint_transition_curves
from .models import WirePathGeometry
from .routing_objective import routing_cost


@dataclass(frozen=True)
class FrustumTransitionCandidate(BridgeCandidate):
    segment_geometry: tuple[WirePathGeometry, ...] = ()
    minimum_radius_scope: str = "smooth_metric_development_screen_only"


def _rotate(values, angle):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result = np.array(values, dtype=float, copy=True)
    result[..., :2] = result[..., :2] @ rotation.T
    return result


class _ChartCurve:
    """Evaluate existing cubic/biarc proposals without interpolating their chords."""

    def __init__(self, candidate, before, after):
        self.candidate = candidate
        self.pieces = []
        if candidate.profile_family == "quintic":
            controls = candidate.control_points_m
            self.coefficients = np.array(
                [
                    controls[0],
                    3 * (controls[1] - controls[0]),
                    3 * (controls[2] - 2 * controls[1] + controls[0]),
                    controls[3] - 3 * controls[2] + 3 * controls[1] - controls[0],
                ]
            )
            return
        first, first_end, second_start, last = candidate.control_points_m
        middle = second_start - first_end
        middle /= np.linalg.norm(middle)
        first_index = int(np.argmin(np.linalg.norm(candidate.points - first_end, axis=1)))
        last_index = int(np.argmin(np.linalg.norm(candidate.points - second_start, axis=1)))
        limits = candidate.parameters[[0, first_index, last_index, -1]]
        for low, high, origin, end, incoming, outgoing in (
            (limits[0], limits[1], first, first_end, before, middle),
            (limits[2], limits[3], second_start, last, middle, after),
        ):
            incoming, outgoing = incoming / np.linalg.norm(incoming), outgoing / np.linalg.norm(outgoing)
            turn = _turn(incoming, outgoing)
            if turn is None or turn[0] < 1e-10 or high <= low:
                raise ValueError("frustum_degenerate_biarc")
            radius = float(np.linalg.norm(end - origin) / np.linalg.norm(turn[2]))
            self.pieces.append((low, high, origin, incoming, turn[1], radius, turn[0]))
        self.straight = limits[1], limits[2], first_end, second_start

    def evaluate(self, parameters):
        parameters = np.atleast_1d(parameters)
        if not self.pieces:
            return _baseline(self.candidate.control_points_m, parameters)[0]
        low, high, first, last = self.straight
        result = first + np.clip((parameters - low) / (high - low), 0, 1)[:, None] * (last - first)
        for low, high, origin, direction, across, radius, angle in self.pieces:
            selected = (parameters >= low) & (parameters <= high)
            phi = angle * (parameters[selected] - low) / (high - low)
            result[selected] = origin + radius * (
                np.sin(phi)[:, None] * direction + (1 - np.cos(phi))[:, None] * across
            )
        return result

    def roots(self, ray):
        if not self.pieces:
            polynomial = _cross(ray, self.coefficients[:, :2])
            roots = np.roots(np.trim_zeros(polynomial[::-1], "f"))
            return [float(root.real) for root in roots if abs(root.imag) <= 1e-9 and 0 < root.real < 1]
        roots = []
        low, high, first, last = self.straight
        slope = float(_cross(ray, (last - first)[:2]))
        if abs(slope) > 1e-15:
            fraction = -float(_cross(ray, first[:2])) / slope
            if 0 < fraction < 1:
                roots.append(low + fraction * (high - low))
        for low, high, origin, direction, across, radius, angle in self.pieces:
            sine = radius * float(_cross(ray, direction[:2]))
            cosine = -radius * float(_cross(ray, across[:2]))
            constant = float(_cross(ray, (origin + radius * across)[:2]))
            amplitude = float(np.hypot(sine, cosine))
            if amplitude <= 1e-15 or abs(constant) > amplitude * (1 + 1e-12):
                continue
            phase = float(np.arctan2(cosine, sine))
            value = float(np.arcsin(np.clip(-constant / amplitude, -1, 1)))
            for base in (value - phase, np.pi - value - phase):
                for turn in (-1, 0, 1):
                    phi = base + 2 * np.pi * turn
                    if 0 < phi < angle:
                        roots.append(low + phi / angle * (high - low))
        return roots


def _facet_samples(chart, curve, reference, maximum_samples, spacing, chord_error, tangent_angle):
    candidate = curve.candidate
    dense = np.vstack((curve.evaluate(np.linspace(0, 1, 257)), candidate.control_points_m))
    angles = np.arctan2(dense[:, 1], dense[:, 0])
    angles += 2 * np.pi * np.rint((reference - angles) / (2 * np.pi))
    low, high = float(angles.min()), float(angles.max())
    for _low, _high, origin, _direction, across, radius, _angle in curve.pieces:
        center = origin + radius * across
        distance = np.linalg.norm(center[:2])
        if distance <= radius:
            low, high = reference - np.pi / 2, reference + np.pi / 2
            break
        phi = float(np.arctan2(center[1], center[0]))
        phi += 2 * np.pi * round((reference - phi) / (2 * np.pi))
        extent = float(np.arcsin(radius / distance))
        low, high = min(low, phi - extent), max(high, phi + extent)
    if high - low >= np.pi + 1e-8:
        raise ValueError("frustum_curve_angular_domain_exceeded")
    rays = chart.boundary_rays(low, high, maximum_samples)
    roots = []
    for ray in rays:
        for root in curve.roots(ray):
            point = curve.evaluate([root])[0]
            if (
                float(point[:2] @ ray) > chart.tolerance
                and min(np.linalg.norm(point - candidate.points[0]), np.linalg.norm(point - candidate.points[-1]))
                > chart.tolerance
            ):
                roots.append(root)
    # Evaluate exact curve/facet roots. Coalesce only ordinary adaptive samples
    # close to those roots, leaving protected ports and distinct roots intact.
    root_values = np.asarray(sorted(set(roots)))
    parameters = candidate.parameters.copy()
    retained = np.ones(len(parameters), dtype=bool)
    tolerance = min(chord_error, 0.001 * candidate.minimum_radius_m)
    for root in root_values:
        point = curve.evaluate([root])[0]
        nearest = int(np.searchsorted(parameters, root))
        for index in (nearest - 1, nearest):
            if (
                0 < index < len(parameters) - 1
                and np.linalg.norm(curve.evaluate([parameters[index]])[0] - point) < tolerance
            ):
                retained[index] = False
    knots = np.unique(np.r_[parameters[retained], root_values])
    if len(knots) > maximum_samples:
        raise ValueError("frustum_curve_sample_budget_exceeded")
    intervals = [(low, high, 0) for low, high in zip(knots[:-1], knots[1:])][::-1]
    result = [knots[0]]
    while intervals:
        low, high, depth = intervals.pop()
        probes = np.linspace(low, high, 5)
        points = curve.evaluate(probes)
        linear = points[0] + np.linspace(0, 1, 5)[:, None] * (points[-1] - points[0])
        chords = np.diff(points, axis=0)
        lengths = np.linalg.norm(chords, axis=1)
        directions = chords / np.maximum(lengths[:, None], np.finfo(float).tiny)
        split = (
            lengths.sum() > spacing
            or np.max(np.linalg.norm(points - linear, axis=1)) > chord_error
            or np.min(directions @ directions.T) < np.cos(np.radians(tangent_angle))
        )
        if split:
            if depth >= 20 or len(result) + len(intervals) >= maximum_samples:
                raise ValueError("frustum_curve_sample_budget_exceeded")
            middle = (low + high) / 2
            intervals.extend(((middle, high, depth + 1), (low, middle, depth + 1)))
        else:
            result.append(high)
    parameters = np.asarray(result)
    planar = curve.evaluate(parameters)
    # A curve may bulge beyond a rim between samples. Check interior probes
    # independently; final mapped chords are then contained in convex facets.
    probes = np.unique(np.r_[parameters, ((parameters[:-1] + parameters[1:]) / 2)])
    chart.map(curve.evaluate(probes), reference)
    return planar, parameters, np.interp(parameters, candidate.parameters, candidate.source_fractions)


def generate_frustum_transition_curves(
    source_points,
    tangent_before,
    tangent_after,
    chart: FrustumChart,
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
    maximum_samples=8192,
    lift_sign=1.0,
    shape_trial_budget=None,
    shape_trials_reserved=False,
    search_level=None,
):
    """Propose nominal intrinsic cubic/biarc curves; full validation is external.

    ``minimum_radius_m`` is the smooth metric-development candidate radius.
    It does not certify the mapped 3D facet creases or the finite conductor's
    bends. The complete production sweep and independent body checks remain
    mandatory, including when every nominal chord is on the actual substrate.
    """
    validate_search_level(search_level)
    source = np.asarray(source_points, dtype=float)
    tangents = tuple(np.asarray(value, dtype=float) for value in (tangent_before, tangent_after))
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or not 2 <= len(source) <= 4096
        or not np.all(np.isfinite(source))
        or any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in tangents)
        or isinstance(maximum_samples, bool)
        or not isinstance(maximum_samples, int)
        or not 2 <= maximum_samples <= 65536
    ):
        return BridgeGeneration(reason="invalid_frustum_curve_input_or_budget")
    if (
        not np.isfinite(maximum_height_m)
        or maximum_height_m < 0
        or lift_sign not in (-1, 1)
        or any(
            not np.isfinite(value) or value <= 0
            for value in (minimum_radius_m, maximum_segment_length_m, chord_tolerance_m, maximum_tangent_angle_deg)
        )
        or isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or not 1 <= maximum_candidates <= 64
        or np.isnan(clearance_score_cap_m)
        or clearance_score_cap_m <= 0
    ):
        return BridgeGeneration(reason="invalid_limits")
    try:
        routing_cost(source, b0_direction, b0_weight)
        side, first, before = chart.port(source[0], tangents[0], start=True)
        _side, last, after = chart.port(source[-1], tangents[1], start=False)
        start_angle, end_angle = np.arctan2(first[1], first[0]), np.arctan2(last[1], last[0])
        shift = int(np.rint((start_angle - end_angle) / chart.period))
        windings = [shift]
        displacement = end_angle + shift * chart.period - start_angle
        if abs(displacement) >= 0.4 * chart.period:
            windings.append(shift - int(np.sign(displacement)))
        if shape_trial_budget is not None and not shape_trials_reserved:
            shape_trial_budget(cylinder_trial_count(search_level))
        if shape_trial_budget is not None and len(windings) > 1:
            shape_trial_budget(cylinder_trial_count(search_level))
        candidates, rejected = [], set()
        for winding in windings:
            # The cylinder allocation covers the nominal cubic and one axial
            # biarc grid. Two tilted middle directions each require at most
            # PLANAR_RETURN_TRIALS additional radius/closure trials.
            if shape_trial_budget is not None:
                shape_trial_budget(2 * PLANAR_RETURN_TRIALS)
            developed = np.array([np.r_[first, 0.0], _rotate(np.r_[last, 0.0], winding * chart.period)])
            incoming, outgoing = np.r_[before, 0.0], _rotate(np.r_[after, 0.0], winding * chart.period)
            reference = (start_angle + end_angle + winding * chart.period) / 2
            generation = generate_joint_transition_curves(
                developed,
                incoming,
                outgoing,
                [0, 0, 1],
                np.empty((0, 2, 3)),
                maximum_height_m=0.0,
                minimum_radius_m=minimum_radius_m,
                maximum_segment_length_m=maximum_segment_length_m,
                chord_tolerance_m=chord_tolerance_m,
                maximum_tangent_angle_deg=maximum_tangent_angle_deg,
                maximum_candidates=64,
                b0_direction=(0, 1, 0),
                b0_weight=0.0,
                clearance_score_cap_m=clearance_score_cap_m,
                search_level=search_level,
            )
            rejected.update(generation.rejected_reasons)
            axis = np.r_[chart.jacobians[side].T @ chart.axis, 0.0]
            # The planar helper only sees the genuine metric development; its
            # plane is not a surrogate physical support certificate.
            developed_corners, _ = chart.locate(chart.corners)
            proxy = SimpleNamespace(
                vertices=developed_corners,
                project_with_face_normals=lambda points: (
                    np.asarray(points),
                    None,
                    np.tile([0.0, 0.0, 1.0], (len(points), 1)),
                ),
            )
            planar_options = []
            # A mandatory quarter-turn middle can be wider than the pitch of
            # packed rings. These tangent directions preserve physical units
            # and endpoint tangency while permitting a compact oblique S join.
            for tilt in (0.0, np.pi / 12, -np.pi / 12):
                planar = generate_planar_return_curves(
                    proxy,
                    developed,
                    incoming,
                    outgoing,
                    [0, 0, 1],
                    np.empty((0, 2, 3)),
                    maximum_height_m=0.0,
                    minimum_radius_m=minimum_radius_m,
                    maximum_segment_length_m=maximum_segment_length_m,
                    chord_tolerance_m=chord_tolerance_m,
                    maximum_tangent_angle_deg=maximum_tangent_angle_deg,
                    maximum_candidates=64,
                    b0_direction=_rotate(axis, tilt),
                    b0_weight=0.0,
                    clearance_score_cap_m=clearance_score_cap_m,
                    search_level=search_level,
                )
                rejected.update(planar.rejected_reasons)
                planar_options.extend(planar.candidates)
            for option in (*generation.candidates, *planar_options):
                try:
                    curve = _ChartCurve(option, incoming, outgoing)
                    points, parameters, fractions = _facet_samples(
                        chart,
                        curve,
                        reference,
                        maximum_samples,
                        maximum_segment_length_m,
                        chord_tolerance_m,
                        maximum_tangent_angle_deg,
                    )
                    mapped, _sides = chart.map(points, reference)
                    mapped[[0, -1]] = source[[0, -1]]
                    reason = _self_contact_reason(mapped, max(chart.tolerance, 1e-12))
                    if reason:
                        if "budget_exceeded" in reason:
                            return BridgeGeneration(reason=reason, rejected_reasons=tuple(sorted(rejected)))
                        rejected.add(reason)
                        continue
                    cosine = np.cos(np.radians(maximum_tangent_angle_deg))
                    if any(
                        float(chord @ tangent) < cosine * np.linalg.norm(chord) * np.linalg.norm(tangent)
                        for chord, tangent in (
                            (mapped[1] - mapped[0], tangents[0]),
                            (mapped[-1] - mapped[-2], tangents[1]),
                        )
                    ):
                        rejected.add("frustum_port_tangent_sampling")
                        continue
                    candidates.append(
                        FrustumTransitionCandidate(
                            mapped,
                            fractions,
                            np.zeros(len(mapped)),
                            parameters,
                            option.control_points_m.copy(),
                            lift_sign * chart.normals[side],
                            0.0,
                            0.5,
                            option.minimum_radius_m,
                            _maximum_distance_to_polyline(mapped, source),
                            clearance_score_cap_m,
                            "frustum_surface",
                            (WirePathGeometry.SURFACE,) * (len(mapped) - 1),
                        )
                    )
                except ValueError as error:
                    if "budget_exceeded" in str(error):
                        return BridgeGeneration(reason=str(error), rejected_reasons=tuple(sorted(rejected)))
                    rejected.add(str(error))
        candidates.sort(
            key=lambda option: (
                round(routing_cost(option.points, b0_direction, b0_weight), 12),
                round(option.minimum_radius_m, 12),
                tuple(np.round((option.control_points_m - option.control_points_m[0]).ravel(), 12)),
            )
        )
        return BridgeGeneration(
            tuple(candidates[:maximum_candidates]),
            "candidates_available" if candidates else "no_feasible_frustum_transition",
            tuple(sorted(rejected)),
        )
    except (TypeError, ValueError, AttributeError) as error:
        return BridgeGeneration(reason=str(error))
