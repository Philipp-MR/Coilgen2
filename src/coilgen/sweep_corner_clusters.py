"""Bounded intrinsic two-arc realization of an adjacent corner cluster.

This never edits nominal source samples. The caller must independently certify
that the selected source support interval belongs to one protected contour.
"""

from dataclasses import dataclass

import numpy as np

from .joint_surface_curves import _turn
from .manufacturing import _corner_circumradius, _polyline_circumradii, _sweep_arc_step, _vector_angle_deg
from .models import WirePathGeometry


@dataclass(frozen=True)
class CylinderClusterRealization:
    points: np.ndarray
    source_first: int
    source_last: int
    coarse_deviation_lower_bound_m: float
    minimum_radius_m: float
    chart_radii_m: tuple[float, float]


def _distance(points, line, work):
    work.consume_pairs(len(points) * max(len(line) - 1, 0))
    starts = line[:-1]
    vectors = np.diff(line, axis=0)
    squared = np.einsum("ij,ij->i", vectors, vectors)
    offsets = points[:, None, :] - starts[None, :, :]
    parameter = np.divide(
        np.einsum("nsi,si->ns", offsets, vectors),
        squared,
        out=np.zeros((len(points), len(vectors))),
        where=squared > 1e-30,
    )
    differences = offsets - np.clip(parameter, 0, 1)[:, :, None] * vectors
    return float(np.sqrt(np.einsum("nsi,nsi->ns", differences, differences).min(axis=1)).max())


def _endpoint_sampled_cluster_arc(origin, tangent, turn, radius, limits):
    """Use configured interior bounds with fine intrinsic endpoint tangents."""
    angle, across, _displacement = turn
    if angle < 1e-8:
        return origin[None, :]
    length, chord, rotation = limits
    step = min(length / radius, 2 * np.arccos(max(-1.0, 1 - chord / radius)), rotation)
    endpoint_step = min(step, np.deg2rad(1.0), angle / 2)
    count = max(1, int(np.ceil((angle - 2 * endpoint_step) / step - 1e-12)))
    if count + 3 > 4096:
        return None
    parameters = np.r_[0.0, np.linspace(endpoint_step, angle - endpoint_step, count + 1), angle]
    parameters = parameters[np.r_[True, np.diff(parameters) > 1e-14]]
    return origin + radius * (np.sin(parameters)[:, None] * tangent + (1 - np.cos(parameters))[:, None] * across)


def _propose_cylinder_cluster(
    points, labels, first, last, surface, minimum_radius, maximum_deviation, work, sampling, *, maximum_support_steps=8
):
    """Find two C1 circular arcs within a strict source-distance corridor.

    The two radii are solved from endpoint displacement for each bounded
    middle-tangent direction. No radius/clearance tolerance is reduced.
    Endpoints and their adjacent intrinsic source tangents remain exact.
    """
    points = np.asarray(points, dtype=float)
    if (
        not (0 < first <= last < len(points) - 1)
        or minimum_radius <= 0
        or maximum_deviation <= 0
        or not 1 <= maximum_support_steps <= 8
    ):
        return None
    origin = points[first]
    chart = surface.log_map(origin, points)
    directions = np.diff(chart[first - 1 : last + 2], axis=0)
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= 1e-12):
        return None
    directions /= lengths[:, None]
    if np.any(np.einsum("ij,ij->i", directions[:-1], directions[1:]) < -1 + 1e-8):
        return None
    angles = np.linspace(-np.pi, np.pi, 361)[:-1]
    middles = np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))]
    minimum_chart_radius = minimum_radius * 1.005
    shortlist = []
    for before in range(1, maximum_support_steps + 1):
        start_index = first - before
        if start_index < 1:
            break
        for after in range(1, maximum_support_steps + 1):
            end_index = last + after
            if end_index >= len(points) - 1:
                break
            if any(
                WirePathGeometry(value) != WirePathGeometry.SURFACE for value in labels[start_index - 1 : end_index + 1]
            ):
                continue
            start, end = np.r_[chart[start_index], 0.0], np.r_[chart[end_index], 0.0]
            incoming = start - np.r_[chart[start_index - 1], 0.0]
            outgoing = np.r_[chart[end_index + 1], 0.0] - end
            if min(np.linalg.norm(incoming), np.linalg.norm(outgoing)) < 1e-12:
                continue
            incoming /= np.linalg.norm(incoming)
            outgoing /= np.linalg.norm(outgoing)
            original_chart = chart[start_index : end_index + 1]
            original = points[start_index : end_index + 1]
            bbox_minimum = original.min(axis=0) - maximum_deviation
            bbox_maximum = original.max(axis=0) + maximum_deviation
            chord = end - start
            for middle in middles:
                work.consume_trial()
                turns = _turn(incoming, middle), _turn(middle, outgoing)
                if any(turn is None or turn[0] < 1e-5 for turn in turns):
                    continue
                matrix = np.column_stack((turns[0][2][:2], turns[1][2][:2]))
                determinant = np.linalg.det(matrix)
                if abs(determinant) < 1e-10:
                    continue
                radii = np.linalg.solve(matrix, chord[:2])
                if np.any(radii < minimum_chart_radius):
                    continue
                first_arc = _endpoint_sampled_cluster_arc(start, incoming, turns[0], radii[0], sampling)
                if first_arc is None:
                    continue
                second_arc = _endpoint_sampled_cluster_arc(first_arc[-1], middle, turns[1], radii[1], sampling)
                if second_arc is None or len(first_arc) + len(second_arc) > 4096:
                    continue
                local = np.vstack((first_arc, second_arc[1:]))
                if np.linalg.norm(local[-1] - end) > 1e-10:
                    continue
                proposed = surface.exp_map(origin, local[:, :2])
                proposed[[0, -1]] = points[[start_index, end_index]]
                if not np.all(surface.within_domain(proposed, 1e-9)):
                    continue
                if np.any(proposed.min(axis=0) < bbox_minimum) or np.any(proposed.max(axis=0) > bbox_maximum):
                    continue
                controls = local[np.linspace(0, len(local) - 1, min(len(local), 64), dtype=int), :2]
                deviation = max(
                    _distance(controls, original_chart, work), _distance(original_chart, local[:, :2], work)
                )
                if deviation > maximum_deviation:
                    continue
                work.consume_curvature_points(len(proposed))
                physical_radii = np.array(
                    [_corner_circumradius(*proposed[index - 1 : index + 2]) for index in range(1, len(proposed) - 1)]
                )
                physical_minimum = float(physical_radii.min())
                if physical_minimum < minimum_radius:
                    continue
                value = CylinderClusterRealization(
                    proposed, start_index, end_index, deviation, physical_minimum, (float(radii[0]), float(radii[1]))
                )
                shortlist.append(((deviation, before + after, before), value))
                shortlist.sort(key=lambda item: item[0])
                del shortlist[32:]
    return [item[1] for item in shortlist]


def _subdivide(points, spacing, maximum_points):
    samples = [points[0]]
    maximum_step = 0.0
    for first, second in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(second - first))
        count = max(1, int(np.ceil(length / spacing)))
        if len(samples) + count > maximum_points:
            raise ValueError("cluster_validation_sample_budget")
        maximum_step = max(maximum_step, length / count)
        samples.extend(first + fraction * (second - first) for fraction in np.linspace(0, 1, count + 1)[1:])
    return np.asarray(samples), maximum_step


def _distances_to_path(samples, path, work):
    work.consume_pairs(len(samples) * max(len(path) - 1, 0))
    minimum = np.full(len(samples), np.inf)
    for first, second in zip(path[:-1], path[1:]):
        vector = second - first
        squared = float(vector @ vector)
        if squared <= 1e-30:
            distances = np.linalg.norm(samples - first, axis=1)
        else:
            fraction = np.clip((samples - first) @ vector / squared, 0, 1)
            distances = np.linalg.norm(samples - first - fraction[:, None] * vector, axis=1)
        np.minimum(minimum, distances, out=minimum)
    return minimum


def _certify_cluster(
    source_points,
    realized_points,
    surface,
    *,
    required_radius_m,
    maximum_source_deviation_m,
    maximum_chord_error_m,
    tangent_tolerance_deg,
    source_owner_keys,
    source_geometry,
    work,
    maximum_validation_points=4096,
    incoming_source_point=None,
    outgoing_source_point=None,
):
    """Check a supplied local realization; never alter or approve a whole route.

    Both endpoint positions and directions, the analytic support, exact uniform
    contour ownership, and physical curvature are hard requirements. Dense
    bidirectional intrinsic distances have a Lipschitz sampling margin; an
    explicit source-deviation bound remains separate from support-leg error.
    """
    reasons = []
    metrics = {"scope": "local source-corridor certificate only", "whole_body_certified": False}
    try:
        source = np.asarray(source_points, dtype=float)
        realized = np.asarray(realized_points, dtype=float)
        if any(points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 for points in (source, realized)):
            raise ValueError("invalid_cluster_points")
        if not np.isfinite(source).all() or not np.isfinite(realized).all():
            raise ValueError("nonfinite_cluster_points")
        if surface is None:
            raise ValueError("analytic_cylinder_required")
        if not np.isfinite(required_radius_m) or required_radius_m <= 0:
            raise ValueError("invalid_cluster_radius")
        if not np.isfinite(maximum_source_deviation_m) or maximum_source_deviation_m <= 0:
            raise ValueError("invalid_source_deviation_bound")
        if len(source_owner_keys) != len(source) - 1 or len(set(source_owner_keys)) != 1:
            raise ValueError("cluster_crosses_source_owner_boundary")
        owner = source_owner_keys[0]
        if owner[0] != "contour_loop":
            raise ValueError("cluster_requires_protected_contour_owner")
        if len(source_geometry) != len(source) - 1 or any(
            str(getattr(label, "value", label)).lower() != "surface" for label in source_geometry
        ):
            raise ValueError("cluster_crosses_surface_contract")
        if not np.isfinite(maximum_chord_error_m) or maximum_chord_error_m <= 0:
            raise ValueError("invalid_cluster_chord_error")
        if not np.isfinite(tangent_tolerance_deg) or tangent_tolerance_deg < 0:
            raise ValueError("invalid_cluster_tangent_tolerance")
        tolerance = 64 * np.finfo(float).eps * max(float(np.max(np.abs(source))), surface.radius, 1)
        if not np.all(np.linalg.norm(realized[[0, -1]] - source[[0, -1]], axis=1) <= tolerance):
            reasons.append("cluster_endpoint_positions_changed")
        if not np.all(surface.within_domain(realized, 1e-9)):
            reasons.append("cluster_leaves_support_domain")
        projected = surface.project(realized)[0]
        if float(np.max(np.linalg.norm(projected - realized, axis=1))) > tolerance:
            reasons.append("cluster_leaves_analytic_surface")
        local_source = surface.log_map(source[0], source)
        local_realized = surface.log_map(source[0], realized)
        if max(np.max(np.abs(local_source[:, 0])), np.max(np.abs(local_realized[:, 0]))) >= np.pi * surface.radius:
            raise ValueError("cluster_winding_ambiguous")
        incoming_direction = local_source[1] - local_source[0]
        outgoing_direction = local_source[-1] - local_source[-2]
        if incoming_source_point is not None:
            previous_local = surface.log_map(source[0], np.asarray(incoming_source_point)[None])[0]
            incoming_direction = local_source[0] - previous_local
        if outgoing_source_point is not None:
            following_local = surface.log_map(source[0], np.asarray(outgoing_source_point)[None])[0]
            outgoing_direction = following_local - local_source[-1]
        incoming = _vector_angle_deg(incoming_direction, local_realized[1] - local_realized[0])
        outgoing = _vector_angle_deg(outgoing_direction, local_realized[-1] - local_realized[-2])
        metrics.update(incoming_tangent_error_deg=incoming, outgoing_tangent_error_deg=outgoing)
        if max(incoming, outgoing) > tangent_tolerance_deg:
            reasons.append("cluster_endpoint_tangent_changed")
        radii = _polyline_circumradii(realized, closed=False, minimum_turn_angle_deg=1e-7)
        minimum_radius = float(np.min(radii))
        metrics["minimum_physical_radius_m"] = minimum_radius
        if minimum_radius < 0.999 * required_radius_m:
            reasons.append("cluster_physical_radius_violation")
        spacing = min(required_radius_m / 64, max(maximum_chord_error_m / 4, 1e-9))
        source_samples, source_step = _subdivide(local_source, spacing, maximum_validation_points)
        realized_samples, realized_step = _subdivide(local_realized, spacing, maximum_validation_points)
        source_delta = float(np.max(_distances_to_path(source_samples, local_realized, work)))
        realized_delta = float(np.max(_distances_to_path(realized_samples, local_source, work)))
        # Distance to a fixed path is 1-Lipschitz, so sample midintervals cannot
        # exceed the measured maximum by more than half the longest interval.
        upper_bound = max(source_delta + source_step / 2, realized_delta + realized_step / 2)
        metrics.update(
            source_owner_key=owner,
            source_sample_count=len(source_samples),
            realized_sample_count=len(realized_samples),
            maximum_source_to_realized_distance_m=source_delta,
            maximum_realized_to_source_distance_m=realized_delta,
            maximum_two_way_source_deviation_bound_m=upper_bound,
            allowed_source_deviation_m=maximum_source_deviation_m,
        )
        if upper_bound > maximum_source_deviation_m:
            reasons.append("cluster_source_deviation_bound")
    except (TypeError, ValueError, FloatingPointError) as error:
        reasons.append(str(error))
    return {"accepted": not reasons, "reasons": list(dict.fromkeys(reasons)), "metrics": metrics}


class _ClusterBudgetExceeded(RuntimeError):
    pass


@dataclass
class _ClusterWork:
    maximum_trials: int = 16 * 64 * 360
    maximum_pairs: int = 128_000_000
    maximum_curvature_points: int = 2_000_000
    maximum_clusters: int = 16
    clusters: int = 0
    trials: int = 0
    pairs: int = 0
    curvature_points: int = 0
    attempted_pairs: int = 0

    def consume_cluster(self):
        if self.clusters >= min(16, self.maximum_clusters):
            raise _ClusterBudgetExceeded("corner_cluster_count_budget")
        self.clusters += 1

    def consume_trial(self):
        if self.trials >= self.maximum_trials:
            raise _ClusterBudgetExceeded("corner_cluster_trial_budget")
        self.trials += 1

    def consume_pairs(self, count):
        self.attempted_pairs = self.pairs + count
        if self.attempted_pairs > self.maximum_pairs:
            raise _ClusterBudgetExceeded("corner_cluster_distance_pair_budget")
        self.pairs += count

    def consume_curvature_points(self, count):
        if self.curvature_points + count > self.maximum_curvature_points:
            raise _ClusterBudgetExceeded("corner_cluster_curvature_point_budget")
        self.curvature_points += count


def _source_owner_keys(part, source_points, source_geometry, surface):
    """Accept only a sequence-matching original wire with exact owner records."""
    from dataclasses import asdict
    from .manufacturing import _surface_path_point_mask

    wire = part.wire_path
    if wire is None or not wire.segment_owners:
        return None
    expected = np.asarray(wire.v, dtype=float)
    closed = bool(len(expected) > 1 and np.array_equal(expected[0], expected[-1]))
    if closed:
        expected = expected[:-1]
    if len(expected) != len(source_points) or len(wire.segment_owners) != len(source_geometry):
        return None
    if [WirePathGeometry(value) for value in wire.segment_geometry] != list(source_geometry):
        return None
    expected = expected.copy()
    mask = _surface_path_point_mask(source_geometry, len(expected), closed=closed)
    expected[mask] = surface.project(expected[mask])[0]
    tolerance = 64 * np.finfo(float).eps * max(float(np.max(np.abs(source_points))), 1)
    if not np.all(np.linalg.norm(expected - source_points, axis=1) <= tolerance):
        return None
    return [tuple(asdict(owner).values()) for owner in wire.segment_owners]


def resolve_source_corner_clusters(
    part,
    points,
    labels,
    source_points,
    source_geometry,
    surface,
    candidates,
    conflicts,
    minimum_radius,
    config,
    *,
    work=None,
    maximum_clusters=16,
):
    """Replace only certified private-analysis contour intervals, then replan."""
    from .manufacturing import conductor_cross_section_points, _mounting_centerline_offset

    work = _ClusterWork() if work is None else work
    if isinstance(maximum_clusters, bool) or not isinstance(maximum_clusters, int) or not 0 <= maximum_clusters <= 16:
        raise ValueError("maximum_clusters must be an integer in [0, 16]")
    remaining_clusters = min(16 - work.clusters, maximum_clusters, work.maximum_clusters - work.clusters)
    details = {
        "sweep_corner_cluster_enabled": True,
        "sweep_corner_cluster_max_source_deviation_m": float(config.sweep_corner_cluster_max_source_deviation_m),
        "sweep_corner_cluster_initial_conflict_count": len(conflicts),
        "sweep_corner_cluster_initial_candidate_count": len(candidates),
        "sweep_corner_cluster_resolved_count": 0,
        "sweep_corner_cluster_budget_exhausted": False,
        "sweep_corner_cluster_source_owner_checked": False,
        "sweep_corner_cluster_certificates": [],
        "sweep_corner_cluster_rejected_certificate_reasons": {},
    }
    owner_keys = _source_owner_keys(part, source_points, source_geometry, surface)
    if owner_keys is None:
        details["sweep_corner_cluster_unavailable_reason"] = "original_source_owners_not_certified"
        return points, labels, details
    details["sweep_corner_cluster_source_owner_checked"] = True
    lookup = {}
    for index, point in enumerate(source_points):
        lookup.setdefault(point.tobytes(), []).append(index)
    groups = []
    for index in sorted(conflicts):
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    replacements = []
    section_step = float(config.sweep_max_segment_length)
    profile = conductor_cross_section_points(config)
    envelope = float(np.max(np.linalg.norm(profile, axis=1))) + abs(_mounting_centerline_offset(config, profile))
    # If the centerline sagitta is <= e*Rmin/(Rmin+envelope), the outer
    # profile sagitta is <= e for every proposed radius R >= Rmin.
    outer_profile_chord_allowance = float(config.sweep_max_chord_error) * minimum_radius / (minimum_radius + envelope)
    output_sampling = (
        min(section_step, 0.001) if section_step > 0 else 0.001,
        max(outer_profile_chord_allowance, 1e-9),
        _sweep_arc_step(config, minimum_radius),
    )
    try:
        for group in groups:
            if len(group) != 2:
                continue
            if len(replacements) >= remaining_clusters:
                raise _ClusterBudgetExceeded("corner_cluster_count_budget")
            proposals = _propose_cylinder_cluster(
                points,
                labels,
                group[0],
                group[-1],
                surface,
                minimum_radius,
                float(config.sweep_corner_cluster_max_source_deviation_m),
                work,
                output_sampling,
            )
            for proposal in proposals or ():
                if any(
                    proposal.source_first <= previous.source_last and proposal.source_last >= previous.source_first
                    for previous in replacements
                ):
                    continue
                starts = lookup.get(points[proposal.source_first].tobytes(), [])
                ends = lookup.get(points[proposal.source_last].tobytes(), [])
                if len(starts) != 1 or len(ends) != 1:
                    continue
                begin, end = starts[0], ends[0]
                count = len(source_points)
                indices = list(range(begin, end + 1)) if end >= begin else [*range(begin, count), *range(end + 1)]
                extended_owners = [
                    owner_keys[(begin - 1) % len(owner_keys)],
                    *(owner_keys[index % len(owner_keys)] for index in indices[:-1]),
                    owner_keys[end % len(owner_keys)],
                ]
                if len(set(extended_owners)) != 1 or extended_owners[0][0] != "contour_loop":
                    continue
                certificate = _certify_cluster(
                    source_points[indices],
                    proposal.points,
                    surface,
                    required_radius_m=minimum_radius,
                    maximum_source_deviation_m=float(config.sweep_corner_cluster_max_source_deviation_m),
                    maximum_chord_error_m=float(config.sweep_max_chord_error),
                    tangent_tolerance_deg=max(float(config.sweep_tangent_angle_tolerance_deg), 0.75),
                    source_owner_keys=[owner_keys[index % len(owner_keys)] for index in indices[:-1]],
                    source_geometry=[source_geometry[index % len(source_geometry)] for index in indices[:-1]],
                    incoming_source_point=source_points[(begin - 1) % count],
                    outgoing_source_point=source_points[(end + 1) % count],
                    work=work,
                )
                if not certificate["accepted"]:
                    for reason in certificate["reasons"]:
                        counts = details["sweep_corner_cluster_rejected_certificate_reasons"]
                        counts[reason] = counts.get(reason, 0) + 1
                    continue
                work.consume_cluster()
                replacements.append(proposal)
                details["sweep_corner_cluster_certificates"].append(
                    {
                        "analysis_cluster": group,
                        "analysis_support_interval": [proposal.source_first, proposal.source_last],
                        "original_source_interval": [begin, end],
                        "original_source_owner": extended_owners[0],
                        "chart_radii_m": proposal.chart_radii_m,
                        **certificate,
                    }
                )
                break
    except _ClusterBudgetExceeded as error:
        details["sweep_corner_cluster_budget_exhausted"] = True
        details["sweep_corner_cluster_budget_reason"] = str(error)
        # Exhaustion is an incomplete search, never a geometric certificate.
        # Do not apply a partial set of replacements to the physical path.
        replacements = []
    details.update(
        sweep_corner_cluster_resolved_count=len(replacements),
        sweep_corner_cluster_global_certified_count=work.clusters,
        sweep_corner_cluster_global_count_limit=min(16, work.maximum_clusters),
        sweep_corner_cluster_pass_remaining_limit=remaining_clusters,
        sweep_corner_cluster_trial_count=work.trials,
        sweep_corner_cluster_trial_limit=work.maximum_trials,
        sweep_corner_cluster_distance_pair_count=work.pairs,
        sweep_corner_cluster_attempted_distance_pair_count=work.attempted_pairs,
        sweep_corner_cluster_distance_pair_limit=work.maximum_pairs,
        sweep_corner_cluster_curvature_point_count=work.curvature_points,
        sweep_corner_cluster_curvature_point_limit=work.maximum_curvature_points,
        sweep_corner_cluster_validation_point_limit=4096,
        sweep_corner_cluster_proposal_point_limit=4096,
        sweep_corner_cluster_shortlist_limit=32,
        sweep_corner_cluster_support_step_limit=8,
        sweep_corner_cluster_direction_count=360,
    )
    if not replacements:
        return points, labels, details
    replacements.sort(key=lambda value: value.source_first)
    output = [points[0]]
    geometry = []
    cursor = 0
    for value in replacements:
        while cursor < value.source_first:
            geometry.append(labels[cursor])
            output.append(points[cursor + 1])
            cursor += 1
        output.extend(value.points[1:])
        geometry.extend([WirePathGeometry.SURFACE] * (len(value.points) - 1))
        cursor = value.source_last
    while cursor < len(points) - 1:
        geometry.append(labels[cursor])
        output.append(points[cursor + 1])
        cursor += 1
    return np.asarray(output), geometry, details
