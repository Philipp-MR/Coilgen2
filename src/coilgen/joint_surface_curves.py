"""Compact tangent-preserving returns in an existing planar support facet.

Circular terminal bends avoid the chord-scaled cubic handles that can leave a
faceted cylindrical support even when its cut corridor admits an axial return.
These are proposals: the joint evaluator still certifies support, clearance,
the complete finite conductor, and the fixed-current field independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connector_bridges import BridgeCandidate, BridgeGeneration, _maximum_distance_to_polyline
from .joint_curve_search import representative_values, validate_search_level
from .joint_transition_curves import _DistanceBudgetExceeded, _local_clearance, _self_contact_reason
from .models import WirePathGeometry
from .routing_objective import routing_cost


PLANAR_RETURN_TRIALS = 32


@dataclass(frozen=True)
class PlanarReturnCandidate(BridgeCandidate):
    segment_geometry: tuple[WirePathGeometry, ...] = ()


def _plateau_lift(parameters, height, width, length):
    """C2 ramps at the straight-span ends, followed by a constant plateau."""
    values = np.asarray(parameters)
    remaining = length - values
    rising, falling = values < width, remaining < width
    local = np.minimum(values, remaining) / width
    local = np.clip(local, 0, 1)
    lift = height * (10 * local**3 - 15 * local**4 + 6 * local**5)
    first = height / width * (30 * local**2 - 60 * local**3 + 30 * local**4)
    first *= np.where(falling, -1.0, 1.0)
    second = height / width**2 * (60 * local - 180 * local**2 + 120 * local**3)
    first[~(rising | falling)] = 0
    second[~(rising | falling)] = 0
    return lift, first, second


def _lifted_middle(first_end, second_start, guide, height, width, minimum_radius, sampling):
    direction = second_start - first_end
    length = float(np.linalg.norm(direction))
    direction /= length
    width = min(width, length / 2)
    dense = np.unique(np.r_[np.linspace(0, length, 4097), width, length - width])
    _lift, slope, second = _plateau_lift(dense, height, width, length)
    curvature = np.abs(second) / (1 + slope**2) ** 1.5
    radius = 1 / max(float(curvature.max()), np.finfo(float).tiny)
    if radius < minimum_radius * 1.005:
        return None, "insufficient_lifted_bend_radius"
    segment, chord, angle = sampling
    speed = float(np.sqrt(1 + np.max(slope**2)))
    acceleration = float(np.max(np.abs(second)))
    # Global derivative bounds conservatively control every sampled interval.
    step = min(segment / speed, np.sqrt(8 * chord / acceleration), angle / acceleration)
    count = max(1, int(np.ceil(length / step)))
    if count > 4096:
        return None, "planar_return_sample_budget_exceeded"
    positions = np.unique(np.r_[np.linspace(0, length, count + 1), width, length - width])
    lift, _first, _second = _plateau_lift(positions, height, width, length)
    points = first_end + positions[:, None] * direction + lift[:, None] * guide
    points[[0, -1]] = [first_end, second_start]
    return (points, lift, radius), ""


def _distinct_radii(values):
    selected = []
    for value in sorted(values):
        if not selected or value - selected[-1] > max(1e-12, 1e-10 * value):
            selected.append(float(value))
    return selected


def _turn(first, last):
    cosine = float(np.clip(first @ last, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return angle, np.zeros(3), np.zeros(3)
    if np.pi - angle < 1e-6:
        return None
    across = (last - cosine * first) / np.sin(angle)
    displacement = np.sin(angle) * first + (1 - cosine) * across
    return angle, across, displacement


def _arc(origin, tangent, turn, radius, limits):
    angle, across, _displacement = turn
    if angle < 1e-8:
        return origin[None, :]
    length, chord, rotation = limits
    step = min(length / radius, 2 * np.arccos(max(-1.0, 1 - chord / radius)), rotation)
    count = max(1, int(np.ceil(angle / step - 1e-12)))
    if count > 4096:
        return None
    parameters = np.linspace(0, angle, count + 1)
    return origin + radius * (np.sin(parameters)[:, None] * tangent + (1 - np.cos(parameters))[:, None] * across)


def generate_planar_return_curves(
    projector,
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
    shape_trial_budget=None,
    search_level=None,
) -> BridgeGeneration:
    """Propose at most 32 nominal two-bend returns in a common support plane.

    Each terminal bend is a circular arc with its exact directed attachment
    tangent; a straight tangent joins them. Radii are solved from transverse
    endpoint closure, including candidates whose straight span follows a
    physical support edge. The requested radius, sampling controls, and input
    budgets are retained. A facet or edge suggestion is never a support or
    clearance certificate. Six optional raised-middle profiles per admitted
    nominal curve are charged through ``shape_trial_budget`` before sampling.
    Terminal arcs retain SURFACE geometry; only the raised straight middle is
    FREE_3D. Their shared attachments must still pass production realization.
    """
    validate_search_level(search_level)
    source = np.asarray(source_points, dtype=float)
    obstacles = np.asarray(obstacle_segments, dtype=float)
    vectors = [np.asarray(value, dtype=float) for value in (tangent_before, tangent_after, normal, b0_direction)]
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or len(source) < 2
        or obstacles.ndim != 3
        or obstacles.shape[1:] != (2, 3)
        or any(value.shape != (3,) or not np.all(np.isfinite(value)) or not np.any(value) for value in vectors)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(obstacles))
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
        or isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or not 1 <= maximum_candidates <= 64
        or np.isnan(clearance_score_cap_m)
        or clearance_score_cap_m <= 0
    ):
        return BridgeGeneration(reason="invalid_limits")
    try:
        routing_cost(source, b0_direction, b0_weight)
    except (TypeError, ValueError):
        return BridgeGeneration(reason="invalid_b0_objective")
    incoming, outgoing, guide, axis = [value / np.linalg.norm(value) for value in vectors]
    chord = source[-1] - source[0]
    span = float(np.linalg.norm(chord))
    if span <= 1e-12:
        return BridgeGeneration(reason="degenerate_span_or_tangent")
    face_normals = projector.project_with_face_normals(source[[0, -1]])[2]
    plane_normal = face_normals[0] / np.linalg.norm(face_normals[0])
    tolerance = max(1e-9, span * 1e-8)
    if (
        float(face_normals[1] @ plane_normal) < 1 - 1e-10
        or abs(float(guide @ plane_normal)) < 1 - 1e-10
        or abs(float(chord @ plane_normal)) > tolerance
        or max(abs(float(incoming @ plane_normal)), abs(float(outgoing @ plane_normal))) > 1e-7
    ):
        return BridgeGeneration(reason="no_common_planar_attachment_facet")
    axis = axis - float(axis @ plane_normal) * plane_normal
    if np.linalg.norm(axis) < 1e-10:
        return BridgeGeneration(reason="no_tangential_axial_direction")
    axis /= np.linalg.norm(axis)
    if axis @ chord < 0:
        axis *= -1
    samples = (
        float(maximum_segment_length_m),
        float(chord_tolerance_m),
        np.radians(min(float(maximum_tangent_angle_deg), 2.0)),
    )
    vertices = np.asarray(projector.vertices, dtype=float)
    plane_vertices = vertices[np.abs((vertices - source[0]) @ plane_normal) <= tolerance]
    candidates, rejected = [], set()
    for direction in (axis, -axis):
        first_turn, second_turn = _turn(incoming, direction), _turn(direction, outgoing)
        if first_turn is None or second_turn is None:
            rejected.add("antiparallel_terminal_turn")
            continue
        first_delta, second_delta = first_turn[2], second_turn[2]
        first_across = first_delta - float(first_delta @ direction) * direction
        second_across = second_delta - float(second_delta @ direction) * direction
        transverse = chord - float(chord @ direction) * direction
        if np.linalg.norm(first_across) < 1e-10 or np.linalg.norm(second_across) < 1e-10:
            continue  # Existing direct families already cover straight attachments.
        radii = [minimum_radius_m * factor for factor in (1.01, 1.5, 2.0, 3.0, 4.0)]
        if np.isfinite(clearance_score_cap_m):
            radii.extend((clearance_score_cap_m, 1.5 * clearance_score_cap_m))
        # Align the straight span with a real facet edge, rather than assuming
        # the center of a cut corridor is aligned with a world coordinate.
        if len(plane_vertices):
            edge_radii = (plane_vertices - source[0]) @ first_across / float(first_across @ first_across)
            radii.extend(_distinct_radii(edge_radii[edge_radii >= minimum_radius_m * 1.005])[:9])
        radii = _distinct_radii(radius for radius in radii if radius >= minimum_radius_m * 1.005)[:16]
        radii = representative_values(radii, search_level)
        for first_radius in radii:
            remaining = transverse - first_radius * first_across
            second_radius = float(remaining @ second_across) / float(second_across @ second_across)
            if second_radius < minimum_radius_m * 1.005:
                rejected.add("insufficient_bend_radius")
                continue
            if np.linalg.norm(remaining - second_radius * second_across) > tolerance:
                rejected.add("transverse_terminal_closure")
                continue
            first_end = source[0] + first_radius * first_delta
            second_start = source[-1] - second_radius * second_delta
            straight = second_start - first_end
            if float(straight @ direction) <= tolerance:
                rejected.add("nonpositive_straight_return")
                continue
            first_arc = _arc(source[0], incoming, first_turn, first_radius, samples)
            second_arc = _arc(second_start, direction, second_turn, second_radius, samples)
            straight_count = max(1, int(np.ceil(np.linalg.norm(straight) / maximum_segment_length_m - 1e-12)))
            if first_arc is None or second_arc is None or straight_count > 4096:
                rejected.add("planar_return_sample_budget_exceeded")
                continue
            middle = np.linspace(first_end, second_start, straight_count + 1)
            points = np.vstack((first_arc, middle[1:], second_arc[1:]))
            if len(points) > 4096:
                rejected.add("planar_return_sample_budget_exceeded")
                continue
            points[[0, -1]] = source[[0, -1]]
            reason = _self_contact_reason(points, max(span * 1e-10, 1e-12))
            if reason:
                if "budget_exceeded" in reason:
                    return BridgeGeneration(reason=reason, rejected_reasons=tuple(sorted(rejected | {reason})))
                rejected.add(reason)
                continue
            try:
                clearance = _local_clearance(
                    points,
                    np.asarray([source[0], first_end, second_start, source[-1]]),
                    guide,
                    0.0,
                    obstacles,
                    index=obstacle_index,
                    cap=clearance_score_cap_m,
                    consume=distance_budget,
                )
            except _DistanceBudgetExceeded as error:
                return BridgeGeneration(reason=str(error), rejected_reasons=tuple(sorted(rejected | {str(error)})))
            length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            parameters = length / length[-1]
            nominal_geometry = (WirePathGeometry.SURFACE,) * (len(points) - 1)
            candidates.append(
                PlanarReturnCandidate(
                    points,
                    parameters,
                    np.zeros(len(points)),
                    parameters,
                    np.asarray([source[0], first_end, second_start, source[-1]]),
                    guide.copy(),
                    0.0,
                    0.5,
                    min(first_radius, second_radius),
                    _maximum_distance_to_polyline(points, source),
                    clearance,
                    profile_family="planar_biarc_return",
                    segment_geometry=nominal_geometry,
                )
            )
            if maximum_height_m > 0:
                for height in (0.5 * maximum_height_m, maximum_height_m):
                    for ramp_factor in (
                        (4.0,) if search_level == 0 else (2.0, 4.0) if search_level == 1 else (1.0, 2.0, 4.0)
                    ):
                        if shape_trial_budget is not None:
                            shape_trial_budget(1)
                        raised, failure = _lifted_middle(
                            first_end,
                            second_start,
                            guide,
                            height,
                            ramp_factor * (height + minimum_radius_m),
                            minimum_radius_m,
                            samples,
                        )
                        if raised is None:
                            rejected.add(failure)
                            continue
                        raised_middle, middle_lift, middle_radius = raised
                        raised_points = np.vstack((first_arc, raised_middle[1:], second_arc[1:]))
                        if len(raised_points) > 4096:
                            rejected.add("planar_return_sample_budget_exceeded")
                            continue
                        raised_points[[0, -1]] = source[[0, -1]]
                        try:
                            raised_clearance = _local_clearance(
                                raised_points,
                                np.asarray([source[0], first_end, second_start, source[-1]]),
                                guide,
                                height,
                                obstacles,
                                index=obstacle_index,
                                cap=clearance_score_cap_m,
                                consume=distance_budget,
                            )
                        except _DistanceBudgetExceeded as error:
                            return BridgeGeneration(
                                reason=str(error),
                                rejected_reasons=tuple(sorted(rejected | {str(error)})),
                            )
                        raised_length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raised_points, axis=0), axis=1))]
                        raised_parameters = raised_length / raised_length[-1]
                        geometry = (
                            (WirePathGeometry.SURFACE,) * (len(first_arc) - 1)
                            + (WirePathGeometry.FREE_3D,) * (len(raised_middle) - 1)
                            + (WirePathGeometry.SURFACE,) * (len(second_arc) - 1)
                        )
                        lift = np.r_[np.zeros(len(first_arc)), middle_lift[1:], np.zeros(len(second_arc) - 1)]
                        candidates.append(
                            PlanarReturnCandidate(
                                raised_points,
                                raised_parameters,
                                lift,
                                raised_parameters,
                                np.asarray([source[0], first_end, second_start, source[-1]]),
                                guide.copy(),
                                height,
                                0.5,
                                min(first_radius, second_radius, middle_radius),
                                _maximum_distance_to_polyline(raised_points, source),
                                raised_clearance,
                                profile_family="planar_biarc_lifted_return",
                                segment_geometry=geometry,
                            )
                        )
    candidates.sort(
        key=lambda curve: (
            -round(curve.local_clearance_m, 12),
            round(routing_cost(curve.points, b0_direction, b0_weight), 12),
        )
    )
    return BridgeGeneration(
        tuple(candidates[:maximum_candidates]),
        "candidates_available" if candidates else "no_feasible_planar_return",
        tuple(sorted(rejected)),
    )
