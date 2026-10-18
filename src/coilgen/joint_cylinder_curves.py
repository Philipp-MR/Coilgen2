"""Intrinsic transition proposals on a verified faceted cylindrical support.

The lateral polygon is developed isometrically into perimeter/axial coordinates.
Smooth chart curves are returned as nominal surface polylines, with a sample at
every crossed facet edge. Creases belong to the nominal support; the existing
analytic-cylinder manufacturing realization must still validate physical bends,
clearance and fields. Outward lifted returns use a smooth analytic raised middle
and C2 radial corrections on mutable nominal terminal arcs. Their SURFACE tags
describe the intended analytic support realization; raised spans remain FREE_3D.
All radial correction counts toward the supplied nominal-support lift allowance.
"""

from dataclasses import dataclass

import numpy as np

from .connector_bridges import BridgeCandidate, BridgeGeneration, _baseline, _maximum_distance_to_polyline
from .joint_curve_search import cylinder_trial_count, representative_values, validate_search_level
from .joint_surface_curves import _arc, _distinct_radii, _lifted_middle, _plateau_lift, _turn
from .joint_transition_curves import _self_contact_reason, generate_joint_transition_curves
from .models import WirePathGeometry
from .routing_objective import routing_cost


CYLINDER_TRANSITION_TRIALS = 96


@dataclass(frozen=True)
class CylinderTransitionCandidate(BridgeCandidate):
    segment_geometry: tuple[WirePathGeometry, ...] = ()


@dataclass(frozen=True)
class _CylinderChart:
    origin: np.ndarray
    axis: np.ndarray
    polygon: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    directions: np.ndarray
    lower: float
    upper: float
    tolerance: float

    @property
    def perimeter(self):
        return float(self.starts[-1] + self.lengths[-1])

    def locate(self, points):
        """Locate supported points on polygon sides without projecting ports."""
        points = np.asarray(points, dtype=float)
        axial = (points - self.origin) @ self.axis
        radial = points - self.origin - axial[:, None] * self.axis
        delta = radial[:, None] - self.polygon[None]
        distance = np.einsum("pij,ij->pi", delta, self.directions)
        fraction = np.clip(distance / self.lengths, 0.0, 1.0)
        nearest = self.polygon[None] + fraction[:, :, None] * self.lengths[None, :, None] * self.directions[None]
        errors = np.linalg.norm(radial[:, None] - nearest, axis=2)
        sides = np.argmin(errors, axis=1)
        selected = errors[np.arange(len(points)), sides]
        if (
            np.any(selected > self.tolerance)
            or np.any(axial < self.lower - self.tolerance)
            or np.any(axial > self.upper + self.tolerance)
        ):
            raise ValueError("cylinder_ports_off_nominal_support")
        circumferential = self.starts[sides] + fraction[np.arange(len(points)), sides] * self.lengths[sides]
        return np.column_stack((circumferential, axial, np.zeros(len(points)))), sides

    def tangent(self, point, tangent, *, start):
        """Select an incident facet compatible with the directed port tangent."""
        radial = point - self.origin - self.axis * float((point - self.origin) @ self.axis)
        best = None
        norm = float(np.linalg.norm(tangent))
        for side, (corner, direction, length) in enumerate(zip(self.polygon, self.directions, self.lengths)):
            along = float((radial - corner) @ direction)
            closest = corner + np.clip(along, 0.0, length) * direction
            if np.linalg.norm(radial - closest) > self.tolerance:
                continue
            developed = np.array([float(tangent @ direction), float(tangent @ self.axis), 0.0])
            recovered = developed[0] * direction + developed[1] * self.axis
            error = float(np.linalg.norm(tangent - recovered)) / norm
            # The first/last curve interval must remain on the same incident
            # facet as its exact directed tangent, including edge attachments.
            sign = 1 if start else -1
            interior = sign * developed[0]
            if (along <= self.tolerance and interior < -norm * 1e-10) or (
                along >= length - self.tolerance and interior > norm * 1e-10
            ):
                continue
            if best is None or error < best[0]:
                best = error, side, developed
        if best is None or best[0] > 1e-8:
            raise ValueError("cylinder_port_tangent_not_supported")
        return best[1], best[2]

    def map(self, points):
        circumferential = np.mod(points[:, 0], self.perimeter)
        sides = np.minimum(np.searchsorted(self.starts, circumferential, side="right") - 1, len(self.starts) - 1)
        along = circumferential - self.starts[sides]
        return (
            self.origin + self.polygon[sides] + along[:, None] * self.directions[sides] + points[:, 1, None] * self.axis
        )


def _cylinder_chart(projector, *, maximum_vertices=100000, maximum_sides=2048):
    """Verify an open regular polygon extrusion, including lateral coverage."""
    vertices, faces = np.asarray(projector.vertices, dtype=float), np.asarray(projector.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not 6 <= len(vertices) <= maximum_vertices
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or len(faces) > 2 * maximum_vertices
        or not np.all(np.isfinite(vertices))
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("unsupported_cylinder_mesh_or_budget")
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    balance = np.bincount(inverse, weights=np.where(edges[:, 0] == ordered[:, 0], 1, -1), minlength=len(unique))
    if np.any(counts > 2) or np.any((counts == 2) & (balance != 0)):
        raise ValueError("unsupported_cylinder_mesh_topology")
    boundary = edges[counts[inverse] == 1]
    if len(boundary) < 6:
        raise ValueError("unsupported_cylinder_boundaries")
    starts, nstarts = np.unique(boundary[:, 0], return_counts=True)
    ends, nends = np.unique(boundary[:, 1], return_counts=True)
    if not np.array_equal(starts, ends) or np.any(nstarts != 1) or np.any(nends != 1):
        raise ValueError("unsupported_cylinder_boundaries")
    following = dict(map(tuple, boundary))
    remaining, rings = set(following), []
    while remaining:
        first = min(remaining)
        current, ring = first, []
        while current in remaining:
            ring.append(current)
            remaining.remove(current)
            current = int(following[current])
        if current != first:
            raise ValueError("unsupported_cylinder_boundaries")
        rings.append(ring)
    if len(rings) != 2 or not 3 <= len(rings[0]) == len(rings[1]) <= maximum_sides:
        raise ValueError("unsupported_cylinder_boundaries")
    centers = np.array([vertices[ring].mean(axis=0) for ring in rings])
    axis = centers[1] - centers[0]
    height = float(np.linalg.norm(axis))
    scale = max(float(np.max(np.ptp(vertices, axis=0))), height)
    tolerance = max(1e-10 * scale, 1e-12)
    if height <= tolerance:
        raise ValueError("degenerate_cylinder_axis")
    axis /= height
    origin = centers.mean(axis=0)
    axial = (vertices - origin) @ axis
    radial = vertices - origin - axial[:, None] * axis
    polygon = radial[rings[0]].copy()
    if float(np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0) @ axis) < 0:
        polygon = polygon[np.r_[0, np.arange(len(polygon) - 1, 0, -1)]]
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.mean(radii))
    lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
    if (
        radius <= tolerance
        or np.max(np.abs(radii - radius)) > tolerance
        or np.max(np.abs(lengths - lengths.mean())) > tolerance
        or np.max(np.abs(axial[rings[0]] + height / 2)) > tolerance
        or np.max(np.abs(axial[rings[1]] - height / 2)) > tolerance
        or np.any(lengths <= tolerance)
    ):
        raise ValueError("unsupported_nonuniform_cylinder")
    # Every vertex is on an extruded polygon corner, and every triangle is
    # confined to one side. A missing/extra patch fails the total area test.
    corners = np.empty(len(vertices), dtype=np.int64)
    for offset in range(0, len(vertices), 128):
        batch = radial[offset : offset + 128]
        distance = np.linalg.norm(batch[:, None] - polygon[None], axis=2)
        nearest = np.argmin(distance, axis=1)
        if np.max(distance[np.arange(len(batch)), nearest]) > tolerance:
            raise ValueError("unsupported_cylinder_radial_vertices")
        corners[offset : offset + len(batch)] = nearest
    labels = corners[faces]
    side_faces = np.zeros(len(faces), dtype=bool)
    for side in range(len(polygon)):
        side_faces |= np.all((labels == side) | (labels == (side + 1) % len(polygon)), axis=1)
    triangles = vertices[faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(area_vectors, axis=1)
    expected_area = float(lengths.sum()) * height
    if (
        not np.all(side_faces)
        or np.any(areas <= tolerance**2)
        or abs(float(areas.sum()) / 2 - expected_area) > max(tolerance * float(lengths.sum()), expected_area * 1e-9)
    ):
        raise ValueError("uncertified_cylinder_coverage")
    directions = (np.roll(polygon, -1, axis=0) - polygon) / lengths[:, None]
    turns = np.cross(directions, np.roll(directions, -1, axis=0)) @ axis
    cosines = np.einsum("ij,ij->i", directions, np.roll(directions, -1, axis=0))
    if np.any(turns <= 0) or abs(float(np.arctan2(turns, cosines).sum()) - 2 * np.pi) > 1e-8:
        raise ValueError("unsupported_nonconvex_cylinder")
    circumferential = np.r_[0.0, np.cumsum(lengths[:-1])]
    return _CylinderChart(
        origin, axis, polygon, circumferential, lengths, directions, -height / 2, height / 2, tolerance
    )


def _developable_chart(projector, **limits):
    """Retain uniform-cylinder behavior; otherwise verify the actual frustum."""
    try:
        return _cylinder_chart(projector, **limits)
    except ValueError:
        from .developed_frustum import frustum_chart

        return frustum_chart(projector, **limits)


def _split_facets(chart, points, parameters, fractions, maximum_samples, *, surface_only=False):
    """Split chart-polyline intervals at every exact polygon-edge crossing."""
    sections, times, source = [points[0]], [parameters[0]], [fractions[0]]
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        low, high = sorted((float(start[0]), float(end[0])))
        cuts = []
        if high - low > chart.tolerance and not (surface_only and (start[2] > 0 or end[2] > 0)):
            first = int(np.floor(low / chart.perimeter))
            last = int(np.floor(high / chart.perimeter))
            if (last - first + 1) * len(chart.starts) > maximum_samples:
                raise ValueError("cylinder_curve_sample_budget_exceeded")
            for turn in range(first, last + 1):
                boundaries = chart.starts + turn * chart.perimeter
                for boundary in boundaries[
                    (boundaries > low + chart.tolerance) & (boundaries < high - chart.tolerance)
                ]:
                    cuts.append(float((boundary - start[0]) / (end[0] - start[0])))
        for fraction in (*sorted(cuts), 1.0):
            sections.append(start + fraction * (end - start))
            times.append(parameters[index] + fraction * (parameters[index + 1] - parameters[index]))
            source.append(fractions[index] + fraction * (fractions[index + 1] - fractions[index]))
        if len(sections) > maximum_samples:
            raise ValueError("cylinder_curve_sample_budget_exceeded")
    return np.asarray(sections), np.asarray(times), np.asarray(source)


def _split_nominal_cubic_facets(chart, candidate, maximum_samples, sampling):
    """Evaluate nominal cubic facet crossings on the curve, not its chords.

    Ports and every exact facet root are pinned. An adjacent generated interior
    sample may be omitted only within the supplied approximation tolerance and
    one thousandth of the certified curve bend radius. The underlying cubic is
    unchanged. Distinct facet roots and nonadjacent curve portions remain exact;
    final spacing, tangent and manufacturing checks remain independent.
    """
    controls = candidate.control_points_m
    coefficients = np.array(
        [
            controls[0],
            3 * (controls[1] - controls[0]),
            3 * (controls[2] - 2 * controls[1] + controls[0]),
            controls[3] - 3 * controls[2] + 3 * controls[1] - controls[0],
        ]
    )
    low, high = float(controls[:, 0].min()), float(controls[:, 0].max())
    first, last = int(np.floor(low / chart.perimeter)), int(np.floor(high / chart.perimeter))
    if (last - first + 1) * len(chart.starts) > maximum_samples:
        raise ValueError("cylinder_curve_sample_budget_exceeded")
    roots = []
    for turn in range(first, last + 1):
        for boundary in chart.starts + turn * chart.perimeter:
            if not low < boundary < high:
                continue
            polynomial = coefficients[:, 0].copy()
            polynomial[0] -= boundary
            for root in np.roots(np.trim_zeros(polynomial[::-1], "f")):
                if abs(root.imag) <= 64 * np.finfo(float).eps * max(1.0, abs(root.real)) and 0 < root.real < 1:
                    roots.append(float(root.real))
    parameters = candidate.parameters
    retained = np.ones(len(parameters), dtype=bool)
    segment_length, chord_tolerance, rotation = sampling
    coalescing_tolerance = min(chord_tolerance, 0.001 * candidate.minimum_radius_m)
    root_points = _baseline(controls, np.asarray(roots))[0] if roots else np.empty((0, 3))
    for root, point in zip(roots, root_points):
        following = int(np.searchsorted(parameters, root))
        for index in (following - 1, following):
            if (
                0 < index < len(parameters) - 1
                and np.linalg.norm(candidate.points[index] - point) <= coalescing_tolerance
            ):
                retained[index] = False
    knots = np.unique(np.r_[parameters[retained], roots])
    if len(knots) > maximum_samples:
        raise ValueError("cylinder_curve_sample_budget_exceeded")
    # Recheck merged intervals with the original adaptive sampling criteria.
    # Facet roots stay pinned; only intervals exceeding spacing, approximation
    # or tangent limits are bisected, under the same explicit sample budget.
    intervals = [(low, high, 0) for low, high in zip(knots[:-1], knots[1:])][::-1]
    result = [knots[0]]
    cosine = np.cos(rotation)
    while intervals:
        low, high, depth = intervals.pop()
        probes = np.linspace(low, high, 5)
        points, tangents, _second = _baseline(controls, probes)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        linear = points[0] + np.linspace(0, 1, 5)[:, None] * (points[-1] - points[0])
        split = (
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > segment_length
            or np.linalg.norm(points - linear, axis=1).max() > chord_tolerance
            or np.min(tangents @ tangents.T) < cosine
        )
        if split:
            if depth >= 20 or len(result) + len(intervals) >= maximum_samples:
                raise ValueError("cylinder_curve_sample_budget_exceeded")
            middle = (low + high) / 2
            intervals.extend(((middle, high, depth + 1), (low, middle, depth + 1)))
        else:
            result.append(high)
    knots = np.asarray(result)
    return _baseline(controls, knots)[0], knots, np.interp(knots, parameters, candidate.source_fractions)


def _restore_detour_arc_intersections(points, parameters, candidate, before, after):
    """Inserted facet points lie on the circular proposal, not its chord.

    Very short intervals adjacent to an inserted chord point otherwise appear
    to have arbitrarily tight curvature after analytic support realization.
    """
    controls = candidate.control_points_m
    axial = np.array([0.0, np.sign(controls[2, 1] - controls[1, 1]), 0.0])
    if candidate.profile_family.startswith("cylinder_compact"):
        directions = (before / np.linalg.norm(before), axial, after / np.linalg.norm(after))
    else:
        around = np.array([np.sign(controls[-1, 0] - controls[0, 0]), 0.0, 0.0])
        directions = (before / np.linalg.norm(before), axial, around, -axial, after / np.linalg.norm(after))
    for section in range(len(controls) // 2):
        origin, end = controls[2 * section : 2 * section + 2]
        turn = _turn(directions[section], directions[section + 1])
        radius = np.linalg.norm(end - origin) / np.linalg.norm(turn[2])
        first = int(np.argmin(np.linalg.norm(candidate.points[:, :2] - origin[:2], axis=1)))
        last = int(np.argmin(np.linalg.norm(candidate.points[:, :2] - end[:2], axis=1)))
        selected = np.flatnonzero(
            (parameters > candidate.parameters[first]) & (parameters < candidate.parameters[last])
        )
        if not len(selected):
            continue
        low, high = np.zeros(len(selected)), np.full(len(selected), turn[0])
        increasing = end[0] > origin[0]
        for _iteration in range(48):
            angle = (low + high) / 2
            circumferential = origin[0] + radius * (
                np.sin(angle) * directions[section][0] + (1 - np.cos(angle)) * turn[1][0]
            )
            advance = (circumferential < points[selected, 0]) == increasing
            low, high = np.where(advance, angle, low), np.where(advance, high, angle)
        angle = (low + high) / 2
        points[selected, 1] = origin[1] + radius * (
            np.sin(angle) * directions[section][1] + (1 - np.cos(angle)) * turn[1][1]
        )


def _compact_chart_returns(
    chart,
    source,
    before,
    after,
    minimum_radius,
    sampling,
    clearance,
    maximum_height,
    reserve,
    search_level=None,
):
    """Keep terminal bend sizes independent of the distance between groups.

    Two circular bends and an axial straight span can fit a narrow opening
    even when chord-scaled cubic handles leave it. Radii solve transverse
    closure in the developed surface; support and conductor clearance remain
    independent three-dimensional acceptance checks after mapping.
    """
    incoming, outgoing = before / np.linalg.norm(before), after / np.linalg.norm(after)
    chord = source[-1] - source[0]
    tolerance = max(1e-12, np.linalg.norm(chord) * 1e-10)
    candidates = []
    for direction in (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])):
        first_turn, second_turn = _turn(incoming, direction), _turn(direction, outgoing)
        if first_turn is None or second_turn is None:
            continue
        first_delta, second_delta = first_turn[2], second_turn[2]
        first_across = first_delta - float(first_delta @ direction) * direction
        second_across = second_delta - float(second_delta @ direction) * direction
        if min(np.linalg.norm(first_across), np.linalg.norm(second_across)) < 1e-10:
            continue
        transverse = chord - float(chord @ direction) * direction
        radii = [minimum_radius * factor for factor in (1.01, 1.5, 2.0, 3.0, 4.0)]
        if np.isfinite(clearance):
            radii.extend((clearance, 1.5 * clearance))
        # Include real facet-edge corridors, accounting for perimeter wraps.
        boundaries = np.concatenate([chart.starts + turn * chart.perimeter for turn in (-1, 0, 1)])
        edge_radii = (boundaries - source[0, 0]) / first_across[0]
        radii.extend(_distinct_radii(edge_radii[edge_radii >= minimum_radius * 1.005])[:9])
        for first_radius in representative_values(_distinct_radii(radii)[:16], search_level):
            remaining = transverse - first_radius * first_across
            second_radius = float(remaining @ second_across) / float(second_across @ second_across)
            if (
                second_radius < minimum_radius * 1.005
                or np.linalg.norm(remaining - second_radius * second_across) > tolerance
            ):
                continue
            first_end = source[0] + first_radius * first_delta
            second_start = source[-1] - second_radius * second_delta
            span = float((second_start - first_end) @ direction)
            if span <= tolerance:
                continue
            first_arc = _arc(source[0], incoming, first_turn, first_radius, sampling)
            second_arc = _arc(second_start, direction, second_turn, second_radius, sampling)
            count = max(1, int(np.ceil(span / sampling[0])))
            if first_arc is None or second_arc is None or count > 4096:
                continue
            points = np.vstack((first_arc, np.linspace(first_end, second_start, count + 1)[1:], second_arc[1:]))
            if len(points) > 4096:
                continue
            points[[0, -1]] = source[[0, -1]]
            if _self_contact_reason(points, tolerance):
                continue
            distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            parameters = distance / distance[-1]
            candidates.append(
                BridgeCandidate(
                    points,
                    parameters,
                    np.zeros(len(points)),
                    parameters,
                    np.asarray([source[0], first_end, second_start, source[-1]]),
                    np.array([0.0, 0.0, 1.0]),
                    0.0,
                    0.5,
                    min(first_radius, second_radius),
                    _maximum_distance_to_polyline(points, source),
                    clearance,
                    profile_family="cylinder_compact_return",
                )
            )
            if maximum_height > 0:
                for height in (0.5 * maximum_height, maximum_height):
                    for ramp_factor in (
                        (4.0,) if search_level == 0 else (2.0, 4.0) if search_level == 1 else (1.0, 2.0, 4.0)
                    ):
                        if reserve is not None:
                            reserve(1)
                        raised, _reason = _lifted_middle(
                            first_end,
                            second_start,
                            np.array([0.0, 0.0, 1.0]),
                            height,
                            ramp_factor * (height + minimum_radius),
                            minimum_radius,
                            sampling,
                        )
                        if raised is None:
                            continue
                        middle, middle_lift, middle_radius = raised
                        lifted = np.vstack((first_arc, middle[1:], second_arc[1:]))
                        if len(lifted) > 4096:
                            continue
                        lifted[[0, -1]] = source[[0, -1]]
                        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(lifted, axis=0), axis=1))]
                        parameters = distance / distance[-1]
                        lift = np.r_[np.zeros(len(first_arc)), middle_lift[1:], np.zeros(len(second_arc) - 1)]
                        candidates.append(
                            BridgeCandidate(
                                lifted,
                                parameters,
                                lift,
                                parameters,
                                np.asarray([source[0], first_end, second_start, source[-1]]),
                                np.array([0.0, 0.0, 1.0]),
                                height,
                                0.5,
                                min(first_radius, second_radius, middle_radius),
                                _maximum_distance_to_polyline(lifted, source),
                                clearance,
                                profile_family="cylinder_compact_lifted_return",
                            )
                        )
    return candidates


def _detour_chart_returns(
    chart,
    source,
    before,
    after,
    minimum_radius,
    sampling,
    clearance,
    maximum_height,
    reserve,
    search_level=None,
):
    """Four compact bends connect opposing ports through a parallel rail.

    An axial detour lets near-level antipodal ports reverse circumferential
    direction without a cusp. Interior lift has a common constant plateau;
    the first and last circular bends retain their nominal attachments.
    These are bounded proposals, subject to the unchanged physical screens.
    """
    incoming, outgoing = before / np.linalg.norm(before), after / np.linalg.norm(after)
    chord = source[-1] - source[0]
    if abs(chord[0]) < 0.4 * chart.perimeter or abs(chord[1]) > 0.2 * abs(chord[0]) or incoming[0] * outgoing[0] >= 0:
        return []
    circumferential = np.array([np.sign(chord[0]), 0.0, 0.0])
    spacing, chord_error, rotation = sampling
    arc_sampling = spacing, chord_error, rotation / 2
    candidates = []

    def line(first, last, step):
        count = max(1, int(np.ceil(np.linalg.norm(last - first) / step)))
        return np.linspace(first, last, count + 1)

    for sign in (-1.0, 1.0):
        axial = np.array([0.0, sign, 0.0])
        turns = (
            _turn(incoming, axial),
            _turn(axial, circumferential),
            _turn(circumferential, -axial),
            _turn(-axial, outgoing),
        )
        if any(turn is None for turn in turns):
            continue
        metric_factor = 1.0 + maximum_height / float(np.min(np.linalg.norm(chart.polygon, axis=1)))
        terminal_radii = (
            (minimum_radius * 1.03,) if search_level == 0 else (minimum_radius * 1.03, minimum_radius * 2.0)
        )
        for terminal_radius in terminal_radii:
            first_radius = last_radius = terminal_radius
            radius = terminal_radius * metric_factor
            offsets = [
                max(
                    first_radius + radius, last_radius + radius, clearance if np.isfinite(clearance) else minimum_radius
                )
                * factor
                for factor in (1.01, 1.25, 2.0, 4.0, 8.0)
            ]
            offsets = [
                offset
                for offset in offsets
                if chart.lower < (max(source[:, 1]) if sign > 0 else min(source[:, 1])) + sign * offset < chart.upper
            ]
            offsets = representative_values(offsets, search_level)
            for offset in offsets:
                rail = (max(source[:, 1]) if sign > 0 else min(source[:, 1])) + sign * offset
                if not chart.lower < rail < chart.upper:
                    continue
                first_end = source[0] + first_radius * turns[0][2]
                second_start = np.array([first_end[0], rail - radius * turns[1][2][1], 0.0])
                second_end = second_start + radius * turns[1][2]
                fourth_start = source[-1] - last_radius * turns[3][2]
                third_start = np.array([fourth_start[0] - radius * turns[2][2][0], rail, 0.0])
                third_end = third_start + radius * turns[2][2]
                if (
                    min(
                        (second_start - first_end) @ axial,
                        (third_start - second_end) @ circumferential,
                        (fourth_start - third_end) @ -axial,
                    )
                    <= chart.tolerance
                ):
                    continue
                if reserve is not None:
                    reserve(1)
                arcs = [
                    _arc(origin, tangent, turn, bend_radius, arc_sampling)
                    for origin, tangent, turn, bend_radius in zip(
                        (source[0], second_start, third_start, fourth_start),
                        (incoming, axial, circumferential, -axial),
                        turns,
                        (first_radius, radius, radius, last_radius),
                    )
                ]
                if any(arc is None for arc in arcs):
                    continue
                middle = np.vstack(
                    (
                        line(first_end, second_start, spacing),
                        arcs[1][1:],
                        line(second_end, third_start, spacing)[1:],
                        arcs[2][1:],
                        line(third_end, fourth_start, spacing)[1:],
                    )
                )
                distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(middle, axis=0), axis=1))]
                controls = np.array(
                    [source[0], first_end, second_start, second_end, third_start, third_end, fourth_start, source[-1]]
                )
                profiles = [(0.0, 1.0)] + [
                    (height, width)
                    for height in ((maximum_height,) if search_level == 0 else (0.5 * maximum_height, maximum_height))
                    for width in ((4.0,) if search_level == 0 else (2.0, 4.0))
                    if height > 0
                ]
                for height, ramp_factor in profiles:
                    if height > 0 and reserve is not None:
                        reserve(1)
                    if height == 0:
                        interior, lift = middle, np.zeros(len(middle))
                    else:
                        width = min(ramp_factor * (height + minimum_radius), distance[-1] / 2)
                        dense = np.linspace(0, distance[-1], 4097)
                        _values, slopes, seconds = _plateau_lift(dense, height, width, distance[-1])
                        acceleration = max(float(np.max(np.abs(seconds))), np.finfo(float).tiny)
                        step = min(
                            spacing / np.sqrt(1 + float(np.max(slopes**2))),
                            np.sqrt(8 * chord_error / acceleration),
                            rotation / (2 * acceleration),
                        )
                        # Keep every circular sample; subdivide only straight
                        # intervals when the radial ramp needs finer sampling.
                        knots = np.concatenate(
                            [
                                np.linspace(
                                    a,
                                    b,
                                    (
                                        1
                                        if abs(delta[0]) > 1e-15 and abs(delta[1]) > 1e-15
                                        else max(1, int(np.ceil((b - a) / step)))
                                    )
                                    + 1,
                                )[:-1]
                                for a, b, delta in zip(distance[:-1], distance[1:], np.diff(middle, axis=0))
                            ]
                            + [distance[-1:]]
                        )
                        if len(knots) > 8192:
                            continue
                        interior = np.column_stack(
                            [np.interp(knots, distance, middle[:, coordinate]) for coordinate in range(3)]
                        )
                        lift = _plateau_lift(knots, height, width, distance[-1])[0]
                        interior[:, 2] += lift
                    points = np.vstack((arcs[0], interior[1:], arcs[3][1:]))
                    if len(points) > 8192:
                        continue
                    points[[0, -1]] = source[[0, -1]]
                    intervals = np.diff(points, axis=0)
                    a, b = intervals[:-1], intervals[1:]
                    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(a + b, axis=1)
                    if np.any(denominator <= 0):
                        continue
                    curvature = 2 * np.linalg.norm(np.cross(a, b), axis=1) / denominator
                    attained_radius = 1 / max(float(curvature.max()), np.finfo(float).tiny)
                    if attained_radius < minimum_radius * 1.005:
                        continue
                    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(intervals, axis=1))]
                    fractions = lengths / lengths[-1]
                    lifts = np.r_[np.zeros(len(arcs[0])), lift[1:], np.zeros(len(arcs[3]) - 1)]
                    candidates.append(
                        BridgeCandidate(
                            points,
                            fractions,
                            lifts,
                            fractions,
                            controls,
                            np.array([0.0, 0.0, 1.0]),
                            height,
                            0.5,
                            min(first_radius, radius, last_radius, attained_radius),
                            _maximum_distance_to_polyline(points, source),
                            clearance,
                            profile_family="cylinder_detour_return" if height == 0 else "cylinder_detour_lifted_return",
                        )
                    )
    return candidates


def generate_cylindrical_transition_curves(
    source_points,
    tangent_before,
    tangent_after,
    projector,
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
    cylinder_chart=None,
    cylinder_chart_error=None,
    **unused,
):
    """Propose supported chart curves with unchanged directed attachments.

    Chart-height arches follow the support instead of cutting across facets.
    Their signed radial displacement never exceeds the supplied lift allowance.
    ``minimum_radius_m`` screens the smooth developed curve; it is not a
    physical bend certificate after realization. The caller must screen all
    obstacles in 3-D and validate the complete body. The trial callback reserves
    the entire bounded cubic family before any curve generation.
    """
    validate_search_level(search_level)
    source = np.asarray(source_points, dtype=float)
    tangents = [np.asarray(value, dtype=float) for value in (tangent_before, tangent_after)]
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or not 2 <= len(source) <= 4096
        or not np.all(np.isfinite(source))
        or any(
            value.shape != (3,) or not np.all(np.isfinite(value)) or np.linalg.norm(value) <= 1e-15
            for value in tangents
        )
        or isinstance(maximum_samples, bool)
        or not isinstance(maximum_samples, int)
        or maximum_samples < 2
        or not np.isfinite(maximum_height_m)
        or maximum_height_m < 0
        or not np.isfinite(lift_sign)
        or lift_sign not in (-1.0, 1.0)
        or isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or not 1 <= maximum_candidates <= 64
        or not isinstance(shape_trials_reserved, bool)
    ):
        return BridgeGeneration(reason="invalid_cylinder_curve_input_or_budget")
    try:
        if cylinder_chart_error is not None:
            raise ValueError(cylinder_chart_error)
        chart = cylinder_chart if cylinder_chart is not None else _developable_chart(projector)
        from .developed_frustum import FrustumChart

        if isinstance(chart, FrustumChart):
            from .joint_frustum_curves import generate_frustum_transition_curves

            return generate_frustum_transition_curves(
                source,
                *tangents,
                chart,
                maximum_height_m=maximum_height_m,
                minimum_radius_m=minimum_radius_m,
                maximum_segment_length_m=maximum_segment_length_m,
                chord_tolerance_m=chord_tolerance_m,
                maximum_tangent_angle_deg=maximum_tangent_angle_deg,
                maximum_candidates=maximum_candidates,
                b0_direction=b0_direction,
                b0_weight=b0_weight,
                clearance_score_cap_m=clearance_score_cap_m,
                maximum_samples=maximum_samples,
                lift_sign=lift_sign,
                shape_trial_budget=shape_trial_budget,
                shape_trials_reserved=shape_trials_reserved,
                search_level=search_level,
            )
        developed, _sides = chart.locate(source[[0, -1]])
        # Shortest perimeter displacement is the useful compact spiral join.
        developed[-1, 0] = developed[0, 0] + (
            (developed[-1, 0] - developed[0, 0] + chart.perimeter / 2) % chart.perimeter - chart.perimeter / 2
        )
        side_before, before = chart.tangent(source[0], tangents[0], start=True)
        side_after, after = chart.tangent(source[-1], tangents[1], start=False)
        # locate's nearest-side tie must agree with the tangent-selected side
        # at an edge; edge chart coordinates themselves remain continuous.
        for endpoint, side in ((0, side_before), (-1, side_after)):
            radial = source[endpoint] - chart.origin - developed[endpoint, 1] * chart.axis
            value = chart.starts[side] + float((radial - chart.polygon[side]) @ chart.directions[side])
            developed[endpoint, 0] += (
                value - developed[endpoint, 0] + chart.perimeter / 2
            ) % chart.perimeter - chart.perimeter / 2
        windings = [developed]
        displacement = developed[-1, 0] - developed[0, 0]
        if abs(displacement) >= 0.4 * chart.perimeter:
            alternate = developed.copy()
            alternate[-1, 0] -= np.copysign(chart.perimeter, displacement)
            windings.append(alternate)
            # Half-turn ports have equally short windings. Order the directed
            # tangent-aligned one first, without snapping either physical port.
            direction = before[0] / np.linalg.norm(before) + after[0] / np.linalg.norm(after)
            if abs(abs(displacement) - chart.perimeter / 2) <= chart.tolerance and direction * displacement < 0:
                windings.reverse()
        if shape_trial_budget is not None and not shape_trials_reserved:
            shape_trial_budget(cylinder_trial_count(search_level))
        if shape_trial_budget is not None and len(windings) > 1:
            # The original 64 cubic + 32 compact trials were reserved by the
            # caller; the second winding is additional work in either mode.
            shape_trial_budget(cylinder_trial_count(search_level))
        # The radial frame rotates around the polygon; its additional speed is
        # bounded by displacement/radius. Reserve that factor in chart sampling.
        inner_radius = float(np.min(np.linalg.norm((chart.polygon + np.roll(chart.polygon, -1, axis=0)) / 2, axis=1)))
        sampling_factor = 1.0 + maximum_height_m / inner_radius
        proposals, rejected = [], set()
        for winding in windings:
            generation = generate_joint_transition_curves(
                winding,
                before,
                after,
                [0, 0, 1],
                np.empty((0, 2, 3)),
                maximum_height_m=maximum_height_m,
                minimum_radius_m=minimum_radius_m,
                maximum_segment_length_m=maximum_segment_length_m / sampling_factor,
                chord_tolerance_m=chord_tolerance_m / sampling_factor,
                maximum_tangent_angle_deg=maximum_tangent_angle_deg,
                maximum_candidates=64,
                b0_direction=(0, 1, 0),
                b0_weight=0.0,
                search_level=search_level,
            )
            rejected.update(generation.rejected_reasons)
            if "budget_exceeded" in generation.reason:
                return BridgeGeneration(reason=generation.reason, rejected_reasons=tuple(sorted(rejected)))
            compact = _compact_chart_returns(
                chart,
                winding,
                before,
                after,
                minimum_radius_m,
                (
                    maximum_segment_length_m / sampling_factor,
                    chord_tolerance_m / sampling_factor,
                    np.radians(min(maximum_tangent_angle_deg, 2.0)),
                ),
                clearance_score_cap_m,
                maximum_height_m,
                shape_trial_budget,
                search_level,
            )
            proposals.extend((*generation.candidates, *compact))
            proposals.extend(
                _detour_chart_returns(
                    chart,
                    winding,
                    before,
                    after,
                    minimum_radius_m,
                    (
                        maximum_segment_length_m / sampling_factor,
                        chord_tolerance_m / sampling_factor,
                        np.radians(min(maximum_tangent_angle_deg, 2.0)),
                    ),
                    clearance_score_cap_m,
                    maximum_height_m,
                    shape_trial_budget,
                    search_level,
                )
            )
        candidates = []
        for candidate in proposals:
            if candidate.profile_family == "quintic" and candidate.height_m == 0:
                planar, parameters, fractions = _split_nominal_cubic_facets(
                    chart,
                    candidate,
                    maximum_samples,
                    (
                        maximum_segment_length_m / sampling_factor,
                        chord_tolerance_m / sampling_factor,
                        np.radians(min(maximum_tangent_angle_deg, 2.0)),
                    ),
                )
            else:
                planar, parameters, fractions = _split_facets(
                    chart,
                    candidate.points,
                    candidate.parameters,
                    candidate.source_fractions,
                    maximum_samples,
                    surface_only=(
                        candidate.profile_family in ("cylinder_detour_lifted_return", "cylinder_compact_lifted_return")
                        and lift_sign > 0
                    ),
                )
            if candidate.profile_family.startswith("cylinder_detour") or (
                candidate.profile_family == "cylinder_compact_lifted_return" and lift_sign > 0
            ):
                _restore_detour_arc_intersections(planar, parameters, candidate, before, after)
            if (
                np.min(planar[:, 1]) < chart.lower - chart.tolerance
                or np.max(planar[:, 1]) > chart.upper + chart.tolerance
            ):
                rejected.add("cylinder_curve_outside_axial_domain")
                continue
            mapped = chart.map(planar)
            heights = planar[:, 2]
            radial = mapped - chart.origin - planar[:, 1, None] * chart.axis
            radial_lengths = np.linalg.norm(radial, axis=1)
            radial /= radial_lengths[:, None]
            lifts = lift_sign * heights
            if (
                candidate.profile_family in ("cylinder_detour_lifted_return", "cylinder_compact_lifted_return")
                and lift_sign > 0
            ):
                # FREE intervals must be physically smooth around facet edges,
                # just as nominal SURFACE intervals are after manufacturing's
                # analytic retraction. Reserve the maximum polygon sag in the
                # allowed height; never add it to the caller's lift allowance.
                radius = float(np.mean(np.linalg.norm(chart.polygon, axis=1)))
                compact = candidate.profile_family == "cylinder_compact_lifted_return"
                # Axial compact middles have a fixed radial frame. Reserve
                # their actual sag, preserving full lift on a facet-edge rail.
                sag = radius - float(radial_lengths[heights > 0].min()) if compact else radius - inner_radius
                if candidate.height_m <= sag:
                    continue
                effective_height = heights * (1.0 - sag / candidate.height_m)
                angle = 2 * np.pi * planar[:, 0] / chart.perimeter
                first_radial = chart.polygon[0] / np.linalg.norm(chart.polygon[0])
                second_radial = np.cross(chart.axis, first_radial)
                raised = heights > 0
                raised_indices = np.flatnonzero(raised)
                start, end = int(raised_indices[0] - 1), int(raised_indices[-1] + 1)
                # Match the analytic retraction of both shared SURFACE/FREE
                # boundary points. A C2 angular correction bridges their
                # polygon-to-circle chart offsets without introducing creases.
                attachments = np.array([start, end])
                attachment_angles = np.arctan2(radial[attachments] @ second_radial, radial[attachments] @ first_radial)
                offsets = (attachment_angles - angle[attachments] + np.pi) % (2 * np.pi) - np.pi
                if compact:
                    progress = np.clip(
                        (parameters - parameters[start]) / (parameters[end] - parameters[start]), 0.0, 1.0
                    )
                else:
                    progress = np.clip(
                        (planar[:, 0] - planar[start, 0]) / (planar[end, 0] - planar[start, 0]), 0.0, 1.0
                    )
                blend = progress**3 * (10 - 15 * progress + 6 * progress**2)
                angle += offsets[0] + (offsets[1] - offsets[0]) * blend
                analytic_radial = np.cos(angle)[:, None] * first_radial + np.sin(angle)[:, None] * second_radial
                mapped[raised] = (
                    chart.origin
                    + planar[raised, 1, None] * chart.axis
                    + (radius + effective_height[raised])[:, None] * analytic_radial[raised]
                )
                lifts = np.where(raised, radius + effective_height - radial_lengths, 0.0)
                # These mutable terminal arcs represent the nominal analytic
                # support realized by manufacturing's SURFACE retraction.
                # Grow its radial correction C2 from zero at each protected
                # port to full polygon sag at the shared FREE boundary. This
                # keeps raw mixed geometry continuous without moving ports or
                # treating the raised middle as a surface-constrained segment.
                for indices, progress in (
                    (np.arange(start + 1), parameters[: start + 1] / parameters[start]),
                    (
                        np.arange(end, len(planar)),
                        (parameters[-1] - parameters[end:]) / (parameters[-1] - parameters[end]),
                    ),
                ):
                    blend = progress**3 * (10 - 15 * progress + 6 * progress**2)
                    correction = (radius - radial_lengths[indices]) * blend
                    mapped[indices] += correction[:, None] * radial[indices]
                    lifts[indices] = correction
            else:
                mapped += lift_sign * heights[:, None] * radial
            mapped[[0, -1]] = source[[0, -1]]
            contact = _self_contact_reason(mapped, max(chart.tolerance, 1e-12))
            if contact:
                if "budget_exceeded" in contact:
                    return BridgeGeneration(reason=contact, rejected_reasons=tuple(sorted(rejected)))
                rejected.add(contact)
                continue
            first, last = mapped[1] - mapped[0], mapped[-1] - mapped[-2]
            # Nominal polyline endpoint chords approximate analytic tangency;
            # the positive directed tangent is exact in the chart controls.
            cosine = np.cos(np.radians(maximum_tangent_angle_deg))
            if first @ tangents[0] < cosine * np.linalg.norm(first) * np.linalg.norm(tangents[0]) or last @ tangents[
                1
            ] < cosine * np.linalg.norm(last) * np.linalg.norm(tangents[1]):
                rejected.add("cylinder_port_tangent_sampling")
                continue
            geometry = tuple(
                WirePathGeometry.SURFACE if first_height == second_height == 0 else WirePathGeometry.FREE_3D
                for first_height, second_height in zip(heights[:-1], heights[1:])
            )
            values = {
                **vars(candidate),
                "points": mapped,
                "parameters": parameters,
                "source_fractions": fractions,
                "lift_distances_m": lifts,
                "normal": lift_sign * np.cross(chart.directions[side_before], chart.axis),
                "profile_family": (
                    candidate.profile_family
                    if candidate.profile_family.startswith(("cylinder_compact", "cylinder_detour"))
                    else "cylinder_surface"
                    if candidate.height_m == 0
                    else "cylinder_lifted_transition"
                ),
                "local_clearance_m": clearance_score_cap_m,
                "segment_geometry": geometry,
            }
            candidates.append(CylinderTransitionCandidate(**values))
        candidates.sort(
            key=lambda value: (
                round(routing_cost(value.points, b0_direction, b0_weight), 12),
                tuple(np.round((value.control_points_m - developed[0]).ravel(), 12)),
                round(value.height_m, 12),
                round(value.peak_parameter, 12),
            )
        )
        # Preserve lift, family, winding and detour-side diversity. Keep both
        # the shortest and longest rail for each detour signature: complete
        # circuit field cancellation can favor a farther paired rail even when
        # its individual scalar routing cost is higher.
        selected = candidates[:1]
        groups = {}
        for option in candidates:
            controls = option.control_points_m.reshape(-1, 3)
            signature = option.height_m, option.profile_family, np.sign(controls[-1, 0] - controls[0, 0])
            if option.profile_family.startswith("cylinder_detour"):
                signature += (np.sign(controls[2, 1] - controls[1, 1]),)
            groups.setdefault(signature, []).append(option)
        representatives = [options[0] for options in groups.values()]
        for options in groups.values():
            if options[0].profile_family.startswith("cylinder_detour"):

                def extent(option):
                    return abs(option.control_points_m[2, 1] - option.control_points_m[1, 1])

                representatives.extend((min(options, key=extent), max(options, key=extent)))
        for option in representatives:
            if len(selected) >= maximum_candidates:
                break
            if all(option is not prior for prior in selected):
                selected.append(option)
        for option in candidates:
            if len(selected) >= maximum_candidates:
                break
            if all(option is not prior for prior in selected):
                selected.append(option)
        return BridgeGeneration(
            tuple(selected),
            "candidates_available" if candidates else "no_feasible_cylinder_transition",
            tuple(sorted(rejected)),
        )
    except (AttributeError, TypeError, ValueError) as error:
        return BridgeGeneration(reason=str(error))
