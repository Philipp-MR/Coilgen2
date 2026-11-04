"""Add localized overpasses to an incumbent whose other copper already clears."""

from copy import copy

import numpy as np

from .crossing_lifts import _active_intervals, _cosine_plateaus, _crossing_plateaus
from .layout import _normal_shift_bridge_radius, _normal_shift_physical_clearance, _wire_clearance_failure_threshold
from .models import WirePath, WirePathGeometry, WireSegmentOwner


def incumbent_lift_proposal(part, config):
    """Expose the incumbent's exact pre-cleanup wire and signed layers for local repair."""
    d = part.normal_shift_provenance
    if (
        config.skip_normal_shift
        or not config.normal_shift_length
        or part.wire_path is None
        or not d.get("segment_owners")
    ):
        return None
    original, raw = d["points_before_m"], d["points_after_m"]
    owners = [WireSegmentOwner(**o) for o in d["segment_owners"]]
    geometry = [WirePathGeometry(v) for v in d["segment_geometry_before"]]
    source_pos = np.arange(len(original), dtype=float)
    if "output_source_segment_indices" in d:
        source_pos = d["output_source_segment_indices"] + d["output_source_segment_fractions"]
    heights = d["lift_distance_m"]
    if (
        len(source_pos) != len(raw)
        or len(heights) != len(raw)
        or len(raw) < 2
        or len(owners) != len(original) - 1
        or len(geometry) != len(owners)
    ):
        return None
    signed = np.sign(config.normal_shift_length) * heights
    if np.min(signed) < -1e-10 or np.max(signed) > abs(config.normal_shift_length) + 1e-10:
        return None
    base_index = np.minimum(np.floor((source_pos[:-1] + source_pos[1:]) / 2).astype(int), len(owners) - 1)
    raw_owners = (
        [owners[i] for i in base_index]
        if "output_segment_owners" not in d
        else [WireSegmentOwner(**o) for o in d["output_segment_owners"]]
    )
    if "output_segment_geometry" in d:
        raw_geometry = [WirePathGeometry(v) for v in d["output_segment_geometry"]]
    elif np.array_equal(part.wire_path.v, raw):
        raw_geometry = list(part.wire_path.segment_geometry)
    else:
        raw_geometry = [
            WirePathGeometry.OFFSET_SURFACE
            if geometry[i] != WirePathGeometry.FREE_3D and max(abs(heights[j]), abs(heights[j + 1])) > 1e-12
            else geometry[i]
            for j, i in enumerate(base_index)
        ]
    if len(raw_owners) != len(raw) - 1 or len(raw_geometry) != len(raw) - 1:
        return None
    normals = d.get("output_normals")
    if normals is None:
        normals = np.column_stack([np.interp(source_pos, np.arange(len(original)), col) for col in d["normals"].T])
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms < 1e-10):
        return None
    normals = normals / norms[:, None]
    lateral = d.get("lateral_displacement_m", np.zeros_like(raw))
    footprint = d.get("output_footprint_m", raw - heights[:, None] * normals - lateral)
    return WirePath(raw.copy(), raw_owners, raw_geometry), dict(
        source_positions=source_pos,
        lift_distance_m=heights.copy(),
        physical_clearance_m=_normal_shift_physical_clearance(config),
        target_clearance_m=_wire_clearance_failure_threshold(config),
        source_points_m=original,
        source_normals=d["normals"],
        output_normals=normals,
        source_owners=owners,
        source_geometry=geometry,
        routing_footprint_m=d.get("routing_footprint_m", original),
        output_footprint_m=footprint,
        return_shortcuts=d.get("return_shortcuts", []),
        lateral_displacement_m=lateral,
        contacts_before=d.get("contacts_before", []),
        shoulder_length_m=d.get("shoulder_length_m", 0.0),
        selected_contacts=d.get("crossing_pairs_over_under_distance_m", []),
    )


def mounted_gap_proposal(part, config, *, scale=1.25):
    """Correct positive facet gaps by their measured shortfall along a feasible separating axis."""
    from .crossing_relaxation import _CutMotion

    pairs = part.manufacturing_diagnostics.get("copper_clearance_witness_intervals", ())
    if (
        not np.isfinite(scale)
        or scale <= 0
        or len(part.swept_bodies) != 1
        or not pairs
        or any(g <= 1e-10 for _, _, g in pairs)
    ):
        return None
    seed = incumbent_lift_proposal(part, config)
    if seed is None:
        return None
    path, detail = seed
    # The motion solver works in the same directed source intervals as the
    # incumbent sweep. Its constraints use the actual copper closest points,
    # required insulation/tolerance gap, and a small cushion. A side-on pair
    # can use its footprint's lateral room when the support normal is ineffective.
    adapted = copy(part)
    positions = detail["source_positions"]
    indices = np.minimum(positions.astype(int), len(detail["source_points_m"]) - 2)
    adapted.normal_shift_provenance = {
        **part.normal_shift_provenance,
        "output_source_segment_indices": indices,
        "output_source_segment_fractions": positions - indices,
        "output_normals": detail["output_normals"],
        "output_footprint_m": detail["output_footprint_m"],
        "output_segment_owners": [dict(vars(o)) for o in path.segment_owners],
        "output_segment_geometry": [g.value for g in path.segment_geometry],
        "lateral_displacement_m": detail["lateral_displacement_m"],
    }
    motion = _CutMotion(adapted, config)
    constraints = motion.constraints(path.v, part.swept_bodies[0], pairs)
    if not constraints:
        return None
    moved = motion.step(path.v, constraints, scale)
    if np.max(np.linalg.norm(moved - path.v, axis=1)) < 1e-9:
        return None
    result = motion.mount(moved)
    if result is None:
        return None
    moved, height, lateral, normals = result
    geometry = [
        WirePathGeometry.OFFSET_SURFACE if g != WirePathGeometry.FREE_3D and max(abs(height[i : i + 2])) > 1e-12 else g
        for i, g in enumerate(path.segment_geometry)
    ]
    return WirePath(moved, list(path.segment_owners), geometry), {
        **detail,
        "lift_distance_m": height,
        "lateral_displacement_m": lateral,
        "output_normals": normals,
    }


def mounted_lift_proposal(part, config, *, reverse=False, ramp_scale=1.0):
    witnesses = part.manufacturing_diagnostics.get("copper_clearance_witness_intervals", ())
    if not witnesses or part.fabrication_path is None:
        return None
    seed = incumbent_lift_proposal(part, config)
    if seed is None:
        return None
    route, source = seed
    raw = route.v
    source_pos = source["source_positions"]
    raw_owners, raw_geometry = route.segment_owners, route.segment_geometry
    vectors = np.diff(raw, axis=0)
    length = np.linalg.norm(vectors, axis=1)
    arc = np.r_[0.0, np.cumsum(length)]
    if np.any(length <= 1e-12):
        return None
    closed = np.linalg.norm(raw[0] - raw[-1]) < 1e-10
    f = part.fabrication_path
    ordered = None
    if len(part.swept_bodies) == 1 and bool(f.closed) == closed:
        from .wire_provenance import ordered_source_stations

        body = part.swept_bodies[0]
        carrier = f.points_m - f.normals * body.diagnostics.get("sweep_centerline_offset_m", 0.0)
        if f.closed:
            carrier = np.vstack((carrier, carrier[0]))
        ordered, _ = ordered_source_stations(raw, carrier)
    by_owner = {}
    for i, owner in enumerate(raw_owners):
        by_owner.setdefault(tuple(vars(owner).values()), []).append(i)
    station_id = {}

    def locate(k):
        if ordered is not None:
            return int(ordered[k])
        if k not in station_id:
            ids = np.array(by_owner.get(tuple(vars(f.segment_owners[k]).values()), []), dtype=int)
            if not len(ids):
                return None
            center = (f.wire_points_m[k] + f.wire_points_m[k + 1]) / 2
            v = vectors[ids]
            t = np.clip(np.einsum("ij,ij->i", center - raw[ids], v) / length[ids] ** 2, 0, 1)
            dist = np.linalg.norm(raw[ids] + t[:, None] * v - center, axis=1)
            station_id[k] = int(ids[np.argmin(dist)])
        return station_id[k]

    pairs = {}
    for a, b, gap in witnesses:
        i, j = locate(a), locate(b)
        if i is None or j is None or i == j:
            continue
        # Adjacent-ring defects need a tangent-continuous bend, not a jump.
        if abs(a - b) <= 3 or (closed and len(f.segment_owners) - abs(a - b) <= 3):
            continue
        key = tuple(sorted((i, j)))
        pairs[key] = min(pairs.get(key, np.inf), gap)
    contacts = [(i, j, gap) for (i, j), gap in pairs.items()]
    physical = _normal_shift_physical_clearance(config)
    cap = abs(config.normal_shift_length)
    step = min(cap, physical * 1.25)
    ramp = ramp_scale * max(2 * step, np.pi * np.sqrt(step * _normal_shift_bridge_radius(config) / 2))
    if min(step, ramp) <= 0:
        return None
    high, selected = _crossing_plateaus(raw_owners, contacts, arc, ramp, closed)
    if high is None:
        return None
    if reverse:
        high[:] = False
        selected = [(j, i, gap) for i, j, gap in selected]
        high[[i for i, j, gap in selected]] = True
    begins, ends = _active_intervals(high, arc, ramp, closed)
    extra = []
    for lo, hi in zip(begins - ramp, ends + ramp):
        lo, hi = max(lo, 0), min(hi, arc[-1])
        if hi > lo:
            extra.extend(np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / (ramp / 12))) + 1)))
    s = np.unique(np.r_[arc, extra])
    idx = np.minimum(np.searchsorted(arc, s, side="right") - 1, len(length) - 1)
    frac = (s - arc[idx]) / length[idx]
    kept = np.isin(s, arc) | ((frac * length[idx] > 1e-9) & ((1 - frac) * length[idx] > 1e-9))
    s, idx, frac = s[kept], idx[kept], frac[kept]
    pos = np.interp(s, arc, source_pos)

    def interp_rows(abscissa, at, rows):
        return np.stack([np.interp(at, abscissa, col) for col in rows.T], axis=-1)

    n = interp_rows(arc, s, source["output_normals"])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-15)[:, None]
    old_height = np.interp(s, arc, source["lift_distance_m"])
    old = raw[idx] + frac[:, None] * vectors[idx]
    factor = _cosine_plateaus(s, begins, ends, ramp)
    if not closed:
        factor *= 0.5 - 0.5 * np.cos(np.pi * np.minimum(np.minimum(s, arc[-1] - s) / ramp, 1))
    signed = np.sign(config.normal_shift_length)
    extra_height = signed * np.maximum(0, np.minimum(step * factor, cap - signed * old_height))
    new = old + extra_height[:, None] * n
    if closed:
        new[-1] = new[0]
    if np.max(abs(extra_height)) < 1e-8:
        return None
    mids = np.minimum(np.searchsorted(arc, (s[:-1] + s[1:]) / 2, side="right") - 1, len(length) - 1)
    stamps = [
        WirePathGeometry.OFFSET_SURFACE
        if raw_geometry[k] != WirePathGeometry.FREE_3D
        and max(abs(old_height[q] + extra_height[q]), abs(old_height[q + 1] + extra_height[q + 1])) > 1e-10
        else raw_geometry[k]
        for q, k in enumerate(mids)
    ]
    path = WirePath(new, [raw_owners[k] for k in mids], stamps)
    lateral = interp_rows(arc, s, source["lateral_displacement_m"])
    footprint = old - old_height[:, None] * n - lateral
    return path, {
        **source,
        "source_positions": pos,
        "lift_distance_m": old_height + extra_height,
        "output_normals": n,
        "output_footprint_m": footprint,
        "lateral_displacement_m": lateral,
        "shoulder_length_m": ramp,
        "selected_contacts": [(int(source_pos[i]), int(source_pos[j]), v) for i, j, v in selected],
    }
