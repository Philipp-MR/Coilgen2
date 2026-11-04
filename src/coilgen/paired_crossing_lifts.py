"""Choose opposite heights for two distinct visits to an unresolved mounted crossing."""

from collections import defaultdict

import numpy as np

from .models import WirePath, WirePathGeometry
from .mounted_crossing_lifts import incumbent_lift_proposal


def _assign_levels(edges, preference, reverse):
    side = {}
    for start in edges:
        if start in side:
            continue
        side[start] = 0
        component = [start]
        for node in component:
            for other in edges[node]:
                if other not in side:
                    side[other] = 1 - side[node]
                    component.append(other)
                elif side[other] == side[node]:
                    return {}
        priority = [sum(preference[k] for k in component if side[k] == level) for level in (0, 1)]
        if bool(np.argmax(priority) == 0) != reverse:
            for node in component:
                side[node] = 1 - side[node]
    return side


def smooth_lift_ceiling(proposal, physical_config):
    """Fit a detached proposal inside the configured signed layer, leaving its footprint fixed."""
    if proposal is None:
        return None
    path, detail = proposal
    cap = abs(physical_config.normal_shift_length)
    sign = np.sign(physical_config.normal_shift_length)
    old = np.asarray(detail["lift_distance_m"])
    if not cap or np.any(sign * old < -1e-12) or not np.isfinite(old).all():
        return None
    bounded = {**detail, "lift_height_ceiling_m": cap}
    if np.max(np.abs(old)) <= cap:
        return path, bounded
    closed = np.linalg.norm(path.v[0] - path.v[-1]) <= 1e-10
    if not closed and np.any(abs(old[[0, -1]]) > cap):
        return None
    # Match the identity with unit slope below the transition. Saturating only
    # the highest layer retains clearance that uniform amplitude scaling loses.
    transition = min(0.1 * physical_config.conductor_cross_section_width, cap / 2)
    threshold = cap - transition
    signed = sign * old
    high = signed > threshold
    signed[high] = threshold - transition * np.expm1(-(signed[high] - threshold) / transition)
    height = sign * signed
    if not closed:
        height[[0, -1]] = old[[0, -1]]
    moved = path.v + (height - old)[:, None] * detail["output_normals"]
    if closed:
        moved[-1] = moved[0]
    return WirePath(moved, list(path.segment_owners), list(path.segment_geometry)), {
        **bounded,
        "lift_distance_m": height,
    }


def paired_witness_lift_proposal(part, config, *, reverse=False):
    """Move both source visits to opposite sides of the prescribed lift envelope."""
    body = part.fabrication_path
    diagnostics = part.manufacturing_diagnostics
    witnesses = diagnostics.get("copper_clearance_witness_intervals", ())
    if body is None or not witnesses:
        return None
    seed = incumbent_lift_proposal(part, config)
    if seed is None:
        return None
    path, detail = seed
    spa = np.r_[0, np.cumsum(np.linalg.norm(np.diff(body.wire_points_m, axis=0), axis=1))]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path.v, axis=0), axis=1))]
    margin = 4 * np.ptp(body.cross_section_m, axis=0).max()
    selected = []
    for i, j, gap in witnesses:
        distance = abs(spa[i] - spa[j])
        if (min(distance, spa[-1] - distance) if body.closed else distance) > margin:
            selected.append((i, j, gap))
    if not selected:
        return None
    from .manufacturing import _path_reference_point
    from .wire_provenance import ordered_source_stations

    reference = _path_reference_point(body.cross_section_m, config)
    carrier = (
        body.points_m
        - (diagnostics["sweep_centerline_offset_m"] + reference[1]) * body.normals
        - reference[0] * body.transverse
    )
    if body.closed:
        carrier = np.vstack((carrier, carrier[:1]))
    source_ids, fractions = ordered_source_stations(path.v, carrier)
    source_arc = arc[source_ids] + fractions * np.diff(arc)[source_ids]
    locations = {k: source_arc[k] for i, j, _ in selected for k in (i, j)}
    ramp = max(config.sweep_min_bend_radius, config.conductor_cross_section_width)
    station = arc
    if body.closed:
        # Group on the periodic arclength axis, cutting in the largest empty sector.
        ordered = np.unique(list(locations.values()))
        cut = ordered[(np.argmax(np.diff(np.r_[ordered, ordered[0] + arc[-1]])) + 1) % len(ordered)]
        station = (arc - cut) % arc[-1]
    else:
        cut = 0.0
    positions = {k: (s - cut) % arc[-1] if body.closed else s for k, s in locations.items()}
    ordered = np.unique(list(positions.values()))
    spans = np.split(ordered, np.flatnonzero(np.diff(ordered) > ramp) + 1)
    group = {s: k for k, span in enumerate(spans) for s in span}
    edges, preference = defaultdict(set), defaultdict(float)
    old = np.asarray(detail["lift_distance_m"])
    cap = abs(config.normal_shift_length)
    sign = np.sign(config.normal_shift_length)
    for i, j, _ in selected:
        a, b = group[positions[i]], group[positions[j]]
        if a == b:
            continue
        edges[a].add(b)
        edges[b].add(a)
        for node, k in ((a, i), (b, j)):
            connector = path.segment_owners[source_ids[k]].kind in {"within_group_return", "inter_group_connector"}
            preference[node] += (connector + 0.1 * sign * np.interp(locations[k], arc, old) / cap) / len(spans[node])
    levels = _assign_levels(edges, preference, reverse)
    if not levels:
        return None
    height = old.copy()
    pad = 0.3 * config.conductor_cross_section_width
    for k, side in levels.items():
        lo, hi = spans[k][0] - pad, spans[k][-1] + pad
        distance = np.maximum(0, np.maximum(lo - station, station - hi))
        if body.closed:
            distance = np.minimum(distance, np.minimum(station + arc[-1] - hi, lo + arc[-1] - station))
        weight = 0.5 + 0.5 * np.cos(np.pi * np.clip(distance / ramp, 0, 1))
        height = (1 - weight) * height + weight * side * config.normal_shift_length
    if not body.closed:
        height[[0, -1]] = old[[0, -1]]
    moved = path.v + (height - old)[:, None] * detail["output_normals"]
    if body.closed:
        moved[-1] = moved[0]
    original = detail["source_geometry"]
    source_positions = detail["source_positions"]
    ids = np.minimum(((source_positions[:-1] + source_positions[1:]) / 2).astype(int), len(original) - 1)
    labels = [
        WirePathGeometry.FREE_3D
        if current == WirePathGeometry.FREE_3D
        else WirePathGeometry.OFFSET_SURFACE
        if max(abs(height[k : k + 2])) > 1e-10
        else original[ids[k]]
        for k, current in enumerate(path.segment_geometry)
    ]
    contacts = {}
    for i, j, gap in selected:
        a, b = group[positions[i]], group[positions[j]]
        if a not in levels or b not in levels or levels[a] == levels[b]:
            continue
        high, low = (i, j) if levels[a] else (j, i)
        pair = tuple(int(np.interp(locations[k], arc, source_positions)) for k in (high, low))
        contacts[pair] = min(contacts.get(pair, np.inf), gap)
    return WirePath(moved, list(path.segment_owners), labels), {
        **detail,
        "lift_distance_m": height,
        "shoulder_length_m": ramp,
        "selected_contacts": [*detail["selected_contacts"], *((a, b, gap) for (a, b), gap in contacts.items())],
    }
