"""Locally condition unevenly sampled cut approaches which fold during mounting."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .models import WirePath, WirePathGeometry, WireSegmentOwner
from .layout import _within_group_cut_width, _inter_group_cut_width, _wire_backtrack_indices


def _taper_join_budget(points, stations, owners, geometry, heights, displacement, cut, turn_budget, radius, inherited):
    """Avoid a new reversal when a planar join meets a contour with a wider lateral allowance."""
    allowed = np.full(len(points), np.inf)
    if turn_budget <= cut:
        return displacement
    for pivot in _wire_backtrack_indices(points):
        if pivot < 1 or pivot >= len(owners) or np.any(abs(inherited - stations[pivot]) <= 1e-12):
            continue
        join = owners[pivot]
        if owners[pivot - 1] != join or join.connector_role != "joint_spiral_join":
            continue
        for sign in (1, -1):
            contact = pivot
            while 0 < contact < len(owners) and owners[contact if sign > 0 else contact - 1] == join:
                contact += sign
            if not 0 < contact < len(owners) or abs(stations[contact] - stations[pivot]) > radius:
                continue
            segment = contact if sign > 0 else contact - 1
            turn = owners[segment]
            if turn.kind != "contour_loop" or turn.group_index != join.group_index:
                continue
            i = contact
            while 0 < i < len(owners) and owners[i if sign > 0 else i - 1] == turn:
                distance = abs(stations[i] - stations[contact]) / radius
                if (
                    distance >= 1
                    or abs(heights[i]) > 1e-10
                    or geometry[i if sign > 0 else i - 1] != WirePathGeometry.SURFACE
                ):
                    break
                # Both owners share this boundary knot. Ramp the contour's allowance
                # continuously from the cut's half width, with zero slope at each end.
                allowed[i] = min(allowed[i], cut + (turn_budget - cut) * distance**2 * (3 - 2 * distance))
                i += sign
    return displacement * np.minimum(1, allowed / np.maximum(np.linalg.norm(displacement, axis=1), 1e-15))[:, None]


def _short_ring_stations(part):
    """Locate longitudinal loft edges rejected by the actual solid check."""
    if not part.manufacturing_diagnostics.get("short_edge_count", 0) or len(part.swept_bodies) != 1:
        return set()
    from .manufacturing import _sweep_lateral_edge_lengths

    body = part.swept_bodies[0]
    path = body.fabrication_path
    n, sides = len(path.points_m), len(path.cross_section_m)
    # An open loft has two extra cap-center vertices after the profile rings.
    lengths = _sweep_lateral_edge_lengths(body.mesh.vertices[: n * sides], n, sides, closed=path.closed)
    tolerance = body.diagnostics["edge_length_tolerance_m"]
    return set(np.flatnonzero(np.any(lengths <= tolerance, axis=1)))


def conditioned_lift(part, config, *, scale=1.0, include_tight_bends=False):
    """An alternative source path; acceptance is always on a new physical sweep."""
    d = part.normal_shift_provenance
    if d.get("method") != "arclength_crossing_layers" or part.fabrication_path is None:
        return None
    witnesses = part.manufacturing_diagnostics.get("copper_clearance_witness_intervals", ())
    local = _short_ring_stations(part)
    if not witnesses and not local:
        return None
    raw = d["points_after_m"]
    o = [WireSegmentOwner(**v) for v in d["output_segment_owners"]]
    geom = list(map(WirePathGeometry, d["output_segment_geometry"]))
    closed = np.linalg.norm(raw[0] - raw[-1]) < 1e-10
    lengths = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    if np.any(lengths <= 1e-12):
        return None
    arc = np.r_[0, np.cumsum(lengths)]
    physical = part.fabrication_path
    p = physical.wire_points_m
    pa = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    profile = physical.cross_section_m
    diameter = np.max(np.linalg.norm(profile[:, None] - profile[None, :], axis=2))
    for a, b, _ in witnesses:
        dist = abs(pa[a] - pa[b])
        if closed:
            dist = min(dist, pa[-1] - dist)
        if dist <= 4 * diameter:
            local.update((a, b))
    if not local:
        return None
    if include_tight_bends:
        # Moving the ramp changes arclength sampling further along the wire.
        # Round other physically near-inverting bends in this same proposal
        # so a new tessellation cannot simply move a fold to a nearby corner.
        v = np.diff(p, axis=0)
        lengths_p = np.linalg.norm(v, axis=1)
        u = v / np.maximum(lengths_p[:, None], 1e-15)
        sine = np.linalg.norm(np.cross(u[:-1], u[1:]), axis=1)
        chord = np.linalg.norm(v[:-1] + v[1:], axis=1)
        acute_radius = chord / np.maximum(2 * sine, 1e-15)
        for k in np.flatnonzero((sine > 0.05) & (acute_radius < 0.8 * diameter)):
            local.update((int(k), int(k + 1)))
    from .manufacturing import _path_reference_point
    from .wire_provenance import ordered_source_stations

    reference = _path_reference_point(profile, config)
    offset = part.manufacturing_diagnostics["sweep_centerline_offset_m"]
    carrier = physical.points_m - (offset + reference[1]) * physical.normals - reference[0] * physical.transverse
    if physical.closed:
        carrier = np.vstack((carrier, carrier[0]))
    ids, fractions = ordered_source_stations(raw, carrier)
    centers = (arc[ids] + fractions * lengths[ids])[list(local)]
    radius = max(config.sweep_min_bend_radius, diameter)
    sigma = scale * radius * 0.8
    span = 3 * sigma
    positions = np.array(sorted(centers))
    if closed:
        positions = np.r_[positions[-1] - arc[-1], positions, positions[0] + arc[-1]]

    def locality(at):
        k = np.searchsorted(positions, at, side="right")
        near = np.minimum(
            abs(at - positions[np.maximum(k - 1, 0)]), abs(at - positions[np.minimum(k, len(positions) - 1)])
        )
        return 0.5 + 0.5 * np.cos(np.pi * np.clip((near - span / 3) / (2 * span / 3), 0, 1))

    # Estimate a smooth path from uniformly spaced samples so that a cluster
    # of nearly coincident ramp vertices has no extra numerical weight.
    u = np.linspace(0, arc[-1], max(8, int(np.ceil(arc[-1] / min(diameter / 4, sigma / 6)))) + 1)

    def sample(at, x, y):
        return np.stack([np.interp(at, x, col) for col in y.T], axis=-1)

    uniform = sample(u, arc, raw)
    if closed:
        g = gaussian_filter1d(uniform[:-1], sigma / (u[1] - u[0]), axis=0, mode="wrap")
        g = np.vstack((g, g[0]))
    else:
        g = gaussian_filter1d(uniform, sigma / (u[1] - u[0]), axis=0, mode="nearest")
    boundaries = np.r_[True, [a != b or x != y for a, b, x, y in zip(o[:-1], o[1:], geom[:-1], geom[1:])], True]
    adjacent_turn = (
        np.r_[False, [item.kind == "contour_loop" for item in o]]
        | np.r_[[item.kind == "contour_loop" for item in o], False]
    )
    fixed = adjacent_turn & (config.sweep_corner_cluster_max_source_deviation_m <= 0)
    anchors = arc[(locality(arc) < 1e-10) | boundaries | fixed]
    new_stations = u[locality(u) > 1e-10]
    k = np.searchsorted(anchors, new_stations, side="right")
    keep = np.minimum(
        abs(new_stations - anchors[np.maximum(k - 1, 0)]), abs(new_stations - anchors[np.minimum(k, len(anchors) - 1)])
    ) > min(0.05 * diameter, 1e-4)
    s = np.unique(np.r_[anchors, new_stations[keep]])
    srcpos = np.interp(s, arc, d["output_source_segment_indices"] + d["output_source_segment_fractions"])
    at = sample(s, arc, raw)
    n = sample(s, arc, d["output_normals"])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-15)[:, None]
    height = np.interp(s, arc, d["lift_distance_m"])
    fp = sample(s, arc, d["output_footprint_m"])
    diff = (sample(s, u, g) - at) * locality(s)[:, None]
    normal = np.einsum("ij,ij->i", diff, n)
    lateral = diff - normal[:, None] * n
    ix = np.minimum(np.searchsorted(arc, s, side="right") - 1, len(o) - 1)
    is_turn = np.array([o[i].kind == "contour_loop" for i in ix])
    cut = 0.5 * min(_within_group_cut_width(config), _inter_group_cut_width(config))
    budget = np.where(is_turn, config.sweep_corner_cluster_max_source_deviation_m, cut)
    previous = sample(s, arc, d["lateral_displacement_m"])
    total = previous + lateral
    total *= np.minimum(1, budget / np.maximum(np.linalg.norm(total, axis=1), 1e-15))[:, None]
    lateral = total - previous
    # On a planar or cylindrical carrier the requested signed layer height
    # remains bounded. Unraised support traces return to the support on sweep.
    uniform_h = np.interp(u, arc, d["lift_distance_m"])
    if closed:
        filtered_h = gaussian_filter1d(uniform_h[:-1], sigma / (u[1] - u[0]), mode="wrap")
        filtered_h = np.r_[filtered_h, filtered_h[0]]
    else:
        filtered_h = gaussian_filter1d(uniform_h, sigma / (u[1] - u[0]), mode="nearest")
    # Extend the shoulder smoothly before a previously pinned zero-height
    # station. Filter signed height rather than the carrier's own curvature.
    h = height + locality(s) * (np.interp(s, u, filtered_h) - height)
    h = np.sign(config.normal_shift_length) * np.clip(
        np.sign(config.normal_shift_length) * h, 0, abs(config.normal_shift_length)
    )
    new = at + lateral + (h - height)[:, None] * n
    if closed:
        new[-1] = new[0]
    else:
        new[[0, -1]] = raw[[0, -1]]
    mids = np.minimum(np.searchsorted(arc, (s[1:] + s[:-1]) / 2, side="right") - 1, len(o) - 1)
    tapered = _taper_join_budget(
        new,
        s,
        [o[i] for i in mids],
        [geom[i] for i in mids],
        h,
        total,
        cut,
        config.sweep_corner_cluster_max_source_deviation_m,
        radius,
        arc[_wire_backtrack_indices(raw)],
    )
    new += tapered - total
    total = tapered
    # Check both polylines, including source vertices removed on resampling.
    # Their difference is affine between the union of their knots, so this
    # bound applies throughout every intervening segment.
    combined = np.unique(np.r_[arc, s])
    baseline = raw - d["lateral_displacement_m"] - d["lift_distance_m"][:, None] * d["output_normals"]
    offset = sample(combined, s, new - h[:, None] * n) - sample(combined, arc, baseline)
    segment = np.minimum(np.searchsorted(arc, combined, side="right") - 1, len(o) - 1)
    allowed = np.where(
        [o[i].kind == "contour_loop" for i in segment], config.sweep_corner_cluster_max_source_deviation_m, cut
    )
    if np.any(np.linalg.norm(offset, axis=1) > allowed + 1e-8):
        return None
    stamps = [
        WirePathGeometry.OFFSET_SURFACE
        if geom[i] != WirePathGeometry.FREE_3D and max(abs(h[k]), abs(h[k + 1])) > 1e-10
        else geom[i]
        for k, i in enumerate(mids)
    ]
    wire = WirePath(new, [o[i] for i in mids], stamps)
    return wire, dict(
        source_positions=srcpos,
        source_points_m=d["points_before_m"],
        routing_footprint_m=d["routing_footprint_m"],
        source_normals=d["normals"],
        output_normals=n,
        output_footprint_m=fp,
        source_owners=[WireSegmentOwner(**v) for v in d["segment_owners"]],
        source_geometry=list(map(WirePathGeometry, d["segment_geometry_before"])),
        lift_distance_m=h,
        lateral_displacement_m=total,
        contacts_before=d["contacts_before"],
        selected_contacts=d["crossing_pairs_over_under_distance_m"],
        return_shortcuts=d["return_shortcuts"],
        shoulder_length_m=d["shoulder_length_m"],
        target_clearance_m=part.normal_shift_diagnostics["normal_shift_target_clearance_m"],
        physical_clearance_m=part.normal_shift_diagnostics["normal_shift_physical_clearance_m"],
    )
