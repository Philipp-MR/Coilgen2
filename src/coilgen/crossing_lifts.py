"""Build a bounded alternative overpass on the original ordered winding.

The support footprint and directed turns are retained. An overpass can start on
the approach to a connector when its two cuts are too close to provide the
required ramp length. The caller must validate the resulting mounted conductor
before adopting the alternative.
"""

from __future__ import annotations

import numpy as np

from .models import WirePath, WirePathGeometry, WireSegmentOwner


def _precedence(owner):
    priority = {"within_group_return": 3, "inter_group_connector": 2, "contour_group": 1}.get(owner.kind, 0)
    return priority, owner.group_index, owner.connector_index, owner.connector_role


def _active_intervals(mask, arclength, ramp, closed):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = arclength[np.flatnonzero(edges == 1)], arclength[np.flatnonzero(edges == -1)]
    if not len(starts):
        return starts, ends
    if closed:
        period = arclength[-1]
        starts = np.concatenate((starts - period, starts, starts + period))
        ends = np.concatenate((ends - period, ends, ends + period))
    # Two neighboring ramps cannot make a short down/up excursion while
    # meeting the same bend radius. Keep a single plateau between the events.
    gaps = np.flatnonzero(starts[1:] - ends[:-1] >= 2 * ramp)
    return starts[np.r_[0, gaps + 1]], ends[np.r_[gaps, len(ends) - 1]]


def _cosine_plateaus(s, starts, ends, ramp):
    if not len(starts):
        return np.zeros_like(s)
    index = np.searchsorted(starts, s, side="right")
    previous = np.where(index > 0, ends[np.maximum(index - 1, 0)], -np.inf)
    following = np.where(index < len(starts), starts[np.minimum(index, len(starts) - 1)], np.inf)
    distance = np.clip(np.minimum(s - previous, following - s) / ramp, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * distance))


def _crossing_plateaus(owners, contacts, arc, ramp, closed, protect_contours=False, layer_count=2):
    """Assign high/low consistently across all crossings sharing an approach."""
    pairs = [(a, b, d) for a, b, d in contacts if _precedence(owners[a])[0] or _precedence(owners[b])[0]]
    if not pairs:
        return None, []
    touched = np.zeros(len(owners), dtype=bool)
    for a, b, _d in pairs:
        touched[[a, b]] = True
    indices = np.flatnonzero(touched)
    # Connect neighboring obstacle samples before choosing which branch lifts.
    # An arm cannot be a low underpass and a raised overpass at the same point.
    breaks = np.r_[True, arc[indices[1:]] - arc[indices[:-1] + 1] > 1.8 * ramp]
    groups = np.cumsum(breaks) - 1
    if closed and arc[indices[0]] + arc[-1] - arc[indices[-1] + 1] <= 1.8 * ramp:
        groups[groups == groups[-1]] = 0
    label = np.full(len(owners), -1, dtype=int)
    label[indices] = groups
    n = int(groups.max()) + 1
    edges = {}
    preference = np.zeros(n, dtype=float)
    for a, b, d in pairs:
        g, h = label[a], label[b]
        if g == h:
            continue
        edge = tuple(sorted((g, h)))
        edges[edge] = min(d, edges.get(edge, np.inf))
        score = 1 if (_precedence(owners[a]), a) > (_precedence(owners[b]), b) else -1
        preference[g] += score
        preference[h] -= score
    if protect_contours:
        for group in {label[i] for i in indices if owners[i].kind == "contour_loop"}:
            preference[group] -= 2 * len(pairs) + 1
    if layer_count > 2:
        layer_count = min(layer_count, n)
        neighbors = [{} for _ in range(n)]
        for (g, h), distance in edges.items():
            neighbors[g][h] = neighbors[h][g] = distance
        color = np.full(n, -1, dtype=int)
        ordinary = {label[i] for i in indices if owners[i].kind == "contour_loop"}
        # Assign constrained winding patches first. The most constrained
        # remaining patch then takes the lowest layer that preserves previously
        # assigned separations. A third layer resolves triangles in the
        # crossing graph if the user's allowed height accommodates it.
        pending = set(np.unique(groups))
        while pending:
            node = max(
                pending,
                key=lambda g: (
                    g in ordinary,
                    len({color[h] for h in neighbors[g] if color[h] >= 0}),
                    len(neighbors[g]),
                    -preference[g],
                    -g,
                ),
            )
            penalties = [
                sum(1 / max(d, 1e-9) for h, d in neighbors[node].items() if color[h] == k) for k in range(layer_count)
            ]
            color[node] = 0 if protect_contours and node in ordinary else int(np.argmin(penalties))
            pending.remove(node)
        levels = np.zeros(len(owners), dtype=int)
        levels[indices] = color[groups]
        if protect_contours:
            levels[np.array([o.kind == "contour_loop" for o in owners])] = 0
        selected = [(a, b, d) if levels[a] > levels[b] else (b, a, d) for a, b, d in pairs if levels[a] != levels[b]]
        return (levels if levels.any() else None), selected
    # Prioritize physical crossings over conservative envelope near misses.
    # A triangle of mutually close arms may not admit a two-layer solution.
    # In that case retain the most severe independent constraints and let the
    # downstream mounted-solid check decide whether the alternative is usable.
    parent = list(range(n))
    size = np.ones(n, dtype=int)
    parity = np.ones(n, dtype=int)

    def root(i):
        if parent[i] != i:
            old = parent[i]
            parent[i] = root(old)
            parity[i] *= parity[old]
        return parent[i]

    for (g, h), _dist in sorted(edges.items(), key=lambda item: item[1]):
        u, v = root(g), root(h)
        if u != v:
            if size[u] < size[v]:
                u, v = v, u
            parent[v] = u
            parity[v] = -parity[g] * parity[h]
            size[u] += size[v]
    roots = np.array([root(i) for i in range(n)])
    vote = np.bincount(roots, weights=preference * parity, minlength=n)
    color = parity * np.where(vote[roots] < 0, -1, 1)
    high = np.zeros(len(owners), dtype=bool)
    selected = []
    for a, b, d in pairs:
        g, h = label[a], label[b]
        if color[g] == color[h]:
            continue
        over, under = (a, b) if color[g] > 0 else (b, a)
        if protect_contours and owners[over].kind == "contour_loop":
            over, under = under, over
        high[over] = True
        selected.append((over, under, d))
    return (high if high.any() else None), selected


def _protected_contour_envelope(arc, s, owners, ramp, closed):
    fixed = np.asarray([o.kind == "contour_loop" for o in owners], dtype=bool)
    start, end = _active_intervals(fixed, arc, 0.0, closed)
    if not len(start):
        return np.ones_like(s)
    i = np.searchsorted(start, s, side="right")
    previous = np.where(i > 0, end[np.maximum(i - 1, 0)], -np.inf)
    following = np.where(i < len(start), start[np.minimum(i, len(start) - 1)], np.inf)
    width = np.minimum(ramp, np.maximum(0.0, 0.5 * (following - previous)))
    distance = np.maximum(np.minimum(s - previous, following - s), 0.0)
    ratio = np.clip(np.divide(distance, width, out=np.zeros_like(s), where=width > 0), 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * ratio))


def _separate_close_joins(lifted, normals, arc, owners, clearance, config, ramp, budget, closed):
    """Relieve residual same-layer crowding with bounded tangent-plane motions."""
    from .geometry_kernels import segment_closest_parameters
    from .layout import find_wire_segment_contacts_3d

    original = lifted
    moved = lifted.copy()
    limit = 1.015 * clearance
    contacts = find_wire_segment_contacts_3d(moved, limit, config=config)

    def penalty(pairs):
        return sum((1 - d / limit) ** 2 for _, _, d in pairs)

    for _ in range(5):
        pairs = [
            (a, b, d) for a, b, d in contacts if owners[a].kind != "contour_loop" or owners[b].kind != "contour_loop"
        ]
        if not pairs:
            break
        a, b, d = np.asarray(pairs).T
        a, b = a.astype(int), b.astype(int)
        delta = np.diff(moved, axis=0)
        ta, tb = segment_closest_parameters(moved[a], delta[a], moved[b], delta[b])
        separation = moved[a] + ta[:, None] * delta[a] - moved[b] - tb[:, None] * delta[b]
        push = np.zeros_like(moved)
        weight = np.zeros(len(moved))
        for i, which, fraction, direction in [(a, b, ta, separation), (b, a, tb, -separation)]:
            for k, index in enumerate(i):
                if owners[index].kind == "contour_loop":
                    continue
                normal = normals[index] * (1 - fraction[k]) + normals[index + 1] * fraction[k]
                tangent = direction[k] - normal * (normal @ direction[k]) / max(normal @ normal, 1e-20)
                magnitude = np.linalg.norm(tangent)
                if magnitude < 1e-10:
                    continue
                # For two movable branches each takes half of the correction.
                share = 1 if owners[which[k]].kind == "contour_loop" else 0.5
                displacement = tangent * (limit - d[k]) * share / magnitude
                position = arc[index] + fraction[k] * (arc[index + 1] - arc[index])
                for center in (position - arc[-1], position, position + arc[-1]) if closed else (position,):
                    lo, hi = np.searchsorted(arc, [center - ramp, center + ramp])
                    s = slice(lo, hi)
                    profile = 0.5 * (1 + np.cos(np.pi * (arc[s] - center) / ramp))
                    push[s] += profile[:, None] * displacement
                    weight[s] += profile
        step = push / np.maximum(weight[:, None], 1)
        step -= normals * np.einsum("ij,ij->i", step, normals)[:, None]
        if closed:
            step[-1] = step[0]
        else:
            step[[0, -1]] = 0
        if np.max(np.linalg.norm(step, axis=1)) < 1e-7:
            break
        for scale in (1.0, 0.5, 0.25):
            motion = moved - original + scale * step
            motion *= np.minimum(1, budget / np.maximum(np.linalg.norm(motion, axis=1), 1e-15))[:, None]
            trial = original + motion
            trial_pairs = find_wire_segment_contacts_3d(trial, limit, config=config)
            if penalty(trial_pairs) < penalty(contacts) - 1e-8 and sum(d <= 1e-9 for _, _, d in trial_pairs) <= sum(
                d <= 1e-9 for _, _, d in contacts
            ):
                moved, contacts = trial, trial_pairs
                break
        else:
            break
    return moved


def _straighten_reversed_returns(part, source, normals, owners, config, clearance):
    """Remove narrow, backtracking excursions on otherwise straight return rails."""
    from .layout import _within_group_cut_width
    from .manufacturing import _SurfaceProjector

    points = source.copy()
    normals = normals.copy()
    projector = None
    events = []
    closed = np.linalg.norm(source[0] - source[-1]) <= 1e-10
    boundaries = [
        i
        for i, owner in enumerate(owners)
        if owner.kind == "within_group_return" and (i == 0 or owner != owners[i - 1])
    ]
    if closed and len(boundaries) > 1 and boundaries[0] == 0 and owners[0] == owners[-1]:
        boundaries.pop(0)
    for start in boundaries:
        ids = [start]
        nxt = (start + 1) % len(owners)
        while nxt != start and owners[nxt] == owners[start] and (closed or nxt != 0):
            ids.append(nxt)
            nxt = (nxt + 1) % len(owners)
        ids = np.r_[ids, ids[-1] + 1]
        raw = source[ids]
        chord = raw[-1] - raw[0]
        distance = np.linalg.norm(chord)
        if distance < 2 * clearance or len(raw) < 4:
            continue
        axis = chord / distance
        progress = (raw - raw[0]) @ axis
        lateral = np.linalg.norm(raw - raw[0] - progress[:, None] * axis, axis=1)
        if np.max(lateral) > 1.1 * _within_group_cut_width(config):
            continue
        lag = np.maximum.accumulate(progress) - progress
        if np.max(lag) < max(2 * clearance, 0.1 * distance):
            continue
        tip = int(np.argmax(lag))
        first = int(np.argmax(progress[: tip + 1]))
        resumed = np.flatnonzero(progress[tip + 1 :] >= progress[first] + 2 * clearance)
        last = tip + 1 + int(resumed[0]) if len(resumed) else len(progress) - 1
        span = raw[first : last + 1]
        span_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(span, axis=0), axis=1))]
        if span_arc[-1] <= 0:
            continue
        interior = ids[first + 1 : last]
        proposed = span[0] + (span_arc[1:-1] / span_arc[-1])[:, None] * (span[-1] - span[0])
        if projector is None:
            projector = _SurfaceProjector(part.mesh)
        projected, directors, _fn, faces, _bary = projector.project_with_details(proposed)
        # A straight shortcut must stay inside the same connected local strip;
        # reject a chord that leaves its support or jumps to another surface.
        if np.max(np.linalg.norm(projected - proposed, axis=1)) > clearance or np.any(
            np.abs(directors @ normals[ids[first]]) < 0.8
        ):
            continue
        assembly = part.mesh.metadata.get("routing_assembly", {})
        if assembly:
            endpoint_faces = projector.project_with_details(span[[0, -1]])[3]
            tested_faces = np.r_[faces, endpoint_faces]
            if not any(
                np.all((tested_faces >= s["face_range"][0]) & (tested_faces < s["face_range"][1]))
                for s in assembly["sources"]
            ):
                continue
        directors[np.einsum("ij,ij->i", directors, normals[interior]) < 0] *= -1
        points[interior] = projected
        normals[interior] = directors
        if closed and 0 in interior:
            points[-1] = points[0]
            normals[-1] = normals[0]
        events.append(
            dict(
                group_index=owners[start].group_index,
                source_indices=ids[first : last + 1].tolist(),
                removed_axial_backtrack_m=float(lag[tip]),
            )
        )
    return points, normals, events


def _return_elbows(points, owners, arc, radius):
    """Inscribed circular bends on steep, same-owner return handoffs."""
    lengths = np.diff(arc)
    delta = np.diff(points, axis=0)
    tangent = np.divide(delta, lengths[:, None], out=np.zeros_like(delta), where=lengths[:, None] > 1e-12)
    elbows = []
    for j in range(1, len(owners)):
        if owners[j].kind != "within_group_return" or owners[j - 1] != owners[j]:
            continue
        u, v = tangent[j - 1], tangent[j]
        cosine = u @ v
        if not (-0.95 < cosine < 0.65) or min(lengths[j - 1], lengths[j]) < 1e-12:
            continue
        theta = np.arccos(cosine)
        left = j - 1
        right = j + 1
        while left > 0 and owners[left - 1] == owners[j] and tangent[left - 1] @ u > 0.995:
            left -= 1
        while right < len(owners) and owners[right] == owners[j] and tangent[right] @ v > 0.995:
            right += 1
        trim = min(radius * np.tan(theta / 2), 0.8 * (arc[j] - arc[left]), 0.8 * (arc[right] - arc[j]))
        if trim < 1e-8 or (elbows and arc[j] - trim < elbows[-1][1]):
            continue
        # Actual trims stay within both owned legs. Use their measured ends so
        # facet chords on a curved support attach at the original positions.
        before = np.array([np.interp(arc[j] - trim, arc, col) for col in points.T])
        after = np.array([np.interp(arc[j] + trim, arc, col) for col in points.T])
        elbows.append((arc[j] - trim, arc[j] + trim, before, points[j], after))
    return elbows


def crossing_lift_candidate(
    part,
    config,
    *,
    ramp_scale=1.0,
    clearance_factor=1.12,
    separate_joins=False,
    straighten_returns=False,
    protect_contours=False,
    round_returns=False,
    multiple_layers=False,
):
    """Return an isolated `(WirePath, provenance)` proposal, or ``None``.

    A signed configured lift is a hard upper bound. The reference is the
    original unshifted path stored in ``shift_return_paths`` provenance.
    Contact classification never uses a projected drawing of the coil.
    """
    from .layout import (
        _normal_shift_bridge_radius,
        _normal_shift_physical_clearance,
        _wire_clearance_failure_threshold,
        _wire_path_normal_samples,
        find_wire_segment_contacts_3d,
    )

    provenance = part.normal_shift_provenance
    if config.skip_normal_shift or not config.normal_shift_length or part.wire_path is None:
        return None
    if not np.isfinite([ramp_scale, clearance_factor]).all() or min(ramp_scale, clearance_factor) <= 0:
        raise ValueError("Crossing lift factors must be positive and finite.")
    if provenance.get("segment_owners"):
        points = provenance["points_before_m"]
        normals = provenance["normals"]
        owners = [WireSegmentOwner(**row) for row in provenance["segment_owners"]]
        geometry = [WirePathGeometry(value) for value in provenance["segment_geometry_before"]]
    else:
        points = part.wire_path.v
        owners = part.wire_path.segment_owners
        geometry = part.wire_path.segment_geometry
        if not owners:
            return None
        normals = _wire_path_normal_samples(part)[0]
    if len(points) < 4 or len(normals) != len(points):
        return None
    physical = _normal_shift_physical_clearance(config)
    target = _wire_clearance_failure_threshold(config)
    clearance = max(physical, target)
    height = min(abs(config.normal_shift_length), clearance_factor * clearance)
    if min(clearance, height) <= 0:
        return None
    original_points, original_normals = points, normals
    straightened = []
    if straighten_returns:
        points, normals, straightened = _straighten_reversed_returns(part, points, normals, owners, config, clearance)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(lengths)]
    closed = np.linalg.norm(points[0] - points[-1]) <= 1e-10
    contacts = find_wire_segment_contacts_3d(points, clearance, config=config)
    radius = _normal_shift_bridge_radius(config)
    ramp = max(np.pi * np.sqrt(0.5 * height * radius) * 1.12, 2 * height) * ramp_scale
    elbows = _return_elbows(points, owners, arc, radius) if round_returns else []
    layer_count = 1 + int(abs(config.normal_shift_length) / height + 1e-12) if multiple_layers else 2
    high, selected = _crossing_plateaus(owners, contacts, arc, ramp, closed, protect_contours, layer_count)
    if high is None:
        if not straightened and not elbows:
            return None
        high = np.zeros(len(owners), dtype=bool)
    start, end = _active_intervals(high > 0, arc, ramp, closed)
    # Evaluate plateaus and shoulders continuously in arclength, independent
    # of the number of vertices on a nominally straight connector.
    step = min(ramp / 12, clearance * 0.4)
    sample_start, sample_end = start, end
    if separate_joins:
        all_contacts = high > 0
        for a, b, _d in contacts:
            all_contacts[[a, b]] = True
        sample_start, sample_end = _active_intervals(all_contacts, arc, ramp, closed)
    ranges = np.column_stack((np.maximum(sample_start - ramp, 0), np.minimum(sample_end + ramp, arc[-1])))
    additions = []
    additions.extend(np.linspace(left, right, 25) for left, right, *_ in elbows)
    for left, right in ranges:
        if right > left:
            additions.append(np.linspace(left, right, max(2, int(np.ceil((right - left) / step)) + 1)))
    extra = np.unique(np.concatenate(additions)) if additions else np.empty(0)
    segments = np.clip(np.searchsorted(arc, extra, side="right") - 1, 0, len(lengths) - 1)
    keep = (extra - arc[segments] > 0.15 * step) & (arc[segments + 1] - extra > 0.15 * step)
    if protect_contours:
        keep &= np.array([owners[i].kind != "contour_loop" for i in segments], dtype=bool)
    # Overlapping elbow/plateau grids can add two almost identical stations
    # between the same original vertices. Avoid micrometre-sized loft rings
    # without discarding any supplied contour or attachment sample.
    if elbows:
        previous = -np.inf
        for index in np.flatnonzero(keep):
            if extra[index] - previous < 0.15 * step:
                keep[index] = False
            else:
                previous = extra[index]
    segments = segments[keep]
    extra = extra[keep]
    positions = np.sort(np.r_[np.arange(len(points)), segments + (extra - arc[segments]) / lengths[segments]])
    indices = np.minimum(positions.astype(int), len(lengths) - 1)
    fraction = positions - indices
    s = arc[indices] + fraction * lengths[indices]
    factor = _cosine_plateaus(s, start, end, ramp)
    for layer in range(2, int(high.max()) + 1):
        additional = _active_intervals(high >= layer, arc, ramp, closed)
        factor += _cosine_plateaus(s, *additional, ramp)
    allowance = _protected_contour_envelope(arc, s, owners, ramp, closed) if protect_contours else np.ones_like(s)
    factor *= allowance
    if not closed:
        boundary = np.minimum(s, arc[-1] - s)
        factor *= 0.5 * (1 - np.cos(np.pi * np.clip(boundary / ramp, 0, 1)))
    interpolated = points[indices] + fraction[:, None] * (points[indices + 1] - points[indices])
    normal = normals[indices] + fraction[:, None] * (normals[indices + 1] - normals[indices])
    if elbows:
        from .manufacturing import _SurfaceProjector

        projector = _SurfaceProjector(part.mesh)
        for left, right, p0, corner, p1 in elbows:
            active = np.flatnonzero((s > left) & (s < right))
            if not len(active):
                continue
            t = (s[active] - left) / (right - left)
            # Rational quadratic circular arc for equal-length legs. The
            # tangent-intersection representation also handles slightly
            # unequal mesh chords without moving either attachment.
            u = corner - p0
            v = p1 - corner
            cosine = np.clip((u @ v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-24), -1, 1)
            w = np.sqrt(0.5 * (1 + cosine))
            a = (1 - t) ** 2
            b = 2 * w * t * (1 - t)
            c = t**2
            curve = (a[:, None] * p0 + b[:, None] * corner + c[:, None] * p1) / (a + b + c)[:, None]
            projected, directors = projector.project(curve)
            if np.any(np.linalg.norm(projected - curve, axis=1) > clearance):
                continue
            directors[np.einsum("ij,ij->i", directors, normal[active]) < 0] *= -1
            # The trim attachments can lie inside long cut chords, below the
            # triangulated carrier. Retract the adjoining chord samples all
            # the way back to their actual supported source vertices, so the
            # straight approach and arc share a continuous carrier footprint.
            lo = max(int(np.searchsorted(arc, left, side="right")) - 1, 0)
            hi = min(int(np.searchsorted(arc, right)), len(points) - 1)
            anchors = points[[lo, hi]]
            supported = np.max(np.linalg.norm(projector.project(anchors)[0] - anchors, axis=1)) <= 1e-8
            interpolated[active] = projected if supported else curve
            if supported:
                approach = np.flatnonzero(((s > arc[lo]) & (s <= left)) | ((s >= right) & (s < arc[hi])))
                if len(approach):
                    landing, landing_normals = projector.project(interpolated[approach])
                    landing_normals[np.einsum("ij,ij->i", landing_normals, normal[approach]) < 0] *= -1
                    interpolated[approach] = landing
                    normal[approach] = landing_normals
            normal[active] = directors
    norm = np.linalg.norm(normal, axis=1)
    if np.any((factor > 0) & (norm < 1e-10)):
        return None
    normal /= np.where(norm > 0, norm, 1)[:, None]
    offset = np.sign(config.normal_shift_length) * height * factor
    lifted = interpolated + normal * offset[:, None]
    if closed:
        lifted[-1] = lifted[0]
    mid = (0.5 * (positions[1:] + positions[:-1])).astype(int)
    stamped = [
        WirePathGeometry.FREE_3D
        if geometry[i] == WirePathGeometry.FREE_3D
        else WirePathGeometry.OFFSET_SURFACE
        if max(abs(offset[k]), abs(offset[k + 1])) > 1e-12
        else geometry[i]
        for k, i in enumerate(mid)
    ]
    mapped_owners = [owners[i] for i in mid]
    lateral = np.zeros_like(lifted)
    if separate_joins:
        from .layout import _inter_group_cut_width, _within_group_cut_width

        budget = min(physical, 0.5 * min(_within_group_cut_width(config), _inter_group_cut_width(config)))
        if budget > 0:
            moved = _separate_close_joins(lifted, normal, s, mapped_owners, clearance, config, ramp, budget, closed)
            lateral = (moved - lifted) * allowance[:, None]
            moved = lifted + lateral
            # Transport supported displacements across the actual mesh facets,
            # retaining the original normal height above the moved footprint.
            # Free-space connector spans have no support projection contract.
            from .manufacturing import _SurfaceProjector

            active = np.flatnonzero(
                (np.linalg.norm(lateral, axis=1) > 1e-12)
                & np.array([geometry[i] != WirePathGeometry.FREE_3D for i in indices])
            )
            if len(active):
                projector = _SurfaceProjector(part.mesh)
                original_support, _ = projector.project(interpolated[active])
                moved_support, moved_normals = projector.project(interpolated[active] + lateral[active])
                if np.any(np.linalg.norm(moved_support - original_support, axis=1) > budget + 1e-8):
                    return None
                lateral[active] = moved_support - original_support
                moved_normals[np.einsum("ij,ij->i", moved_normals, normal[active]) < 0] *= -1
                normal[active] = moved_normals
                moved[active] = interpolated[active] + lateral[active] + normal[active] * offset[active, None]
                if closed:
                    moved[-1] = moved[0]
            lifted = moved
    path = WirePath(lifted, segment_owners=mapped_owners, segment_geometry=stamped)
    return path, dict(
        source_positions=positions,
        lift_distance_m=offset,
        selected_contacts=selected,
        contacts_before=contacts,
        physical_clearance_m=physical,
        target_clearance_m=target,
        source_points_m=original_points,
        source_normals=original_normals,
        routing_footprint_m=points,
        return_shortcuts=straightened,
        output_normals=normal,
        output_footprint_m=interpolated,
        protected_contours=protect_contours,
        lateral_displacement_m=lateral,
        source_owners=owners,
        source_geometry=geometry,
        shoulder_length_m=ramp,
        plateau_height_m=height,
    )


def _update_group_paths(part, points, path, positions, owners):
    """Update surviving group intervals while retaining precursor cut geometry.

    Each connected-group outline is recorded before group-to-group sections
    are excised. Map surviving directed source edges by group membership. At
    repeated coordinates this must not borrow the neighboring return branch.
    """
    # A later source-conditioning proposal may replace a cluster of original
    # vertices with fewer tangent-matched samples. Its old stations then lie
    # *inside* output edges. Interpolate those exact positions once so group
    # polylines still begin and end at the correct locations in the new wire.
    stations = np.arange(len(points))
    anchors = np.column_stack([np.interp(stations, positions, v) for v in path.v.T])
    first_inside = np.searchsorted(positions, stations, side="right")
    last_inside = np.searchsorted(positions, stations, side="left")
    indexed = {}
    for i, owner in enumerate(owners):
        if owner.group_index < 0 or owner.kind == "inter_group_connector":
            continue
        first, second = points[i].tobytes(), points[i + 1].tobytes()
        indexed.setdefault((owner.group_index, first, second), []).append(i)
    for group_index, group in enumerate(part.connected_groups):
        for name in ("v", "return_path_v", "spiral_in_v", "spiral_out_v"):
            reference = getattr(group, name)
            if len(reference) < 2:
                continue
            chunks = []
            vertices = reference.copy()
            for j, (a, b) in enumerate(zip(reference[:-1], reference[1:])):
                ka, kb = a.tobytes(), b.tobytes()
                forward = indexed.get((group_index, ka, kb), [])
                reverse = indexed.get((group_index, kb, ka), [])
                if name == "return_path_v":
                    forward = [i for i in forward if owners[i].kind == "within_group_return"]
                    reverse = [i for i in reverse if owners[i].kind == "within_group_return"]
                if len(forward) + len(reverse) == 1:
                    i = forward[0] if forward else reverse[0]
                    interior = path.v[first_inside[i] : last_inside[i + 1]]
                    chunk = np.vstack((anchors[i], interior, anchors[i + 1]))
                    if reverse:
                        chunk = chunk[::-1]
                    vertices[j : j + 2] = chunk[[0, -1]]
                else:
                    chunk = None
                # Retain the surviving endpoint where a discarded cut edge
                # touches the geometry that remains in the final wire.
                chunks.append(chunk)
            if chunks:
                curve = [vertices[i : i + 1] if c is None else c[:-1] for i, c in enumerate(chunks)]
                setattr(group, name, np.concatenate(curve + [vertices[-1:]]))


def _sync_cut_spans(part, path, original):
    """Use exact interval ownership even when the pre-cut and final sampling differ."""
    spans = {}
    for i, owner in enumerate(path.segment_owners):
        if owner.kind == "inter_group_connector":
            spans.setdefault((owner.connector_index, owner.connector_role), []).append(i)
    for index, cut in enumerate(part.opening_cuts_among_groups):
        for role, attr in [("primary", "connection_path_v"), ("closing", "closing_connection_path_v")]:
            indices = np.asarray(spans.get((index, role), []), dtype=int)
            ref = original[index][role]
            if not len(indices) or not len(ref):
                continue
            breaks = np.flatnonzero(np.diff(indices) != 1)
            if len(breaks) == 1 and indices[0] == 0 and indices[-1] == len(path.v) - 2:
                split = breaks[0] + 1
                segment = path.v[np.r_[indices[split:], indices[:split], indices[split - 1] + 1]]
            elif not len(breaks):
                segment = path.v[indices[0] : indices[-1] + 2]
            else:
                # A later cut may excise intervals of an earlier bridge. A
                # discontinuous set must never introduce a fictitious chord.
                continue
            if np.linalg.norm(segment[0] - ref[-1]) + np.linalg.norm(segment[-1] - ref[0]) < np.linalg.norm(
                segment[0] - ref[0]
            ) + np.linalg.norm(segment[-1] - ref[-1]):
                segment = segment[::-1]
            setattr(cut, attr, segment.copy())
            for aux, prior in enumerate(original[index]["auxiliary"]):
                if prior.shape != ref.shape:
                    continue
                if np.allclose(ref, prior, rtol=0, atol=1e-12):
                    cut.normal_shift_paths_v[aux] = segment.copy()
                elif np.allclose(ref[::-1], prior, rtol=0, atol=1e-12):
                    cut.normal_shift_paths_v[aux] = segment[::-1].copy()


def apply_crossing_lifts(
    part,
    config,
    *,
    ramp_scale=1.0,
    clearance_factor=1.12,
    separate_joins=False,
    straighten_returns=False,
    protect_contours=False,
    round_returns=False,
    multiple_layers=False,
):
    """Apply a proposal to an *isolated* routing part, ready for cleanup/sweep.

    Returns False if no eligible crossing exists or a hard centerline spacing
    constraint remains violated. True means a candidate was produced; the
    actual mounted, swept conductor still requires validation before adoption.
    """
    candidate = crossing_lift_candidate(
        part,
        config,
        ramp_scale=ramp_scale,
        clearance_factor=clearance_factor,
        separate_joins=separate_joins,
        straighten_returns=straighten_returns,
        protect_contours=protect_contours,
        round_returns=round_returns,
        multiple_layers=multiple_layers,
    )
    return apply_lift_proposal(part, config, candidate)


def apply_lift_proposal(part, config, candidate):
    """Attach a proposed path with its source-to-output correspondence to a detached part."""
    from .layout import _update_inter_group_paths_after_shift, _wire_backtrack_indices

    if candidate is None:
        return False
    path, detail = candidate
    physical = detail["physical_clearance_m"]
    target = detail["target_clearance_m"]
    final_contacts = _proposal_contacts(path.v, max(physical, target, 1e-9), config)
    if any(d + 1e-9 < target for _a, _b, d in final_contacts):
        return False
    points = detail["source_points_m"]
    owners = detail["source_owners"]
    positions = detail["source_positions"]
    offset = detail["lift_distance_m"]
    factors = offset / float(config.normal_shift_length)
    crossings = sum(d <= 1e-9 for _, _, d in final_contacts)
    physical_pairs = sum(d + 1e-9 < physical for _, _, d in final_contacts)
    before = detail["contacts_before"]
    backtracks = _wire_backtrack_indices(path.v)
    mapped = positions[backtracks]
    inherited = np.isin(np.rint(mapped).astype(int), _wire_backtrack_indices(points)) & (
        np.abs(mapped - np.rint(mapped)) <= 1e-10
    )
    reasons = []
    if crossings:
        reasons.append("centerline_intersection")
    if np.any(~inherited):
        reasons.append("centerline_backtrack")
    # The legacy pass may already have updated cut references. In that case,
    # compose the two exact source-position maps rather than guessing the
    # branch by Cartesian proximity in a wire with repeated crossings.
    previous = part.normal_shift_provenance
    previous_v = previous.get("points_after_m", points)
    if "output_source_segment_indices" in previous:
        old_positions = previous["output_source_segment_indices"] + previous["output_source_segment_fractions"]
        cut_positions = np.interp(positions, old_positions, np.arange(len(old_positions)))
    else:
        cut_positions = positions
    original_cuts = [
        dict(
            primary=cut.connection_path_v,
            closing=cut.closing_connection_path_v,
            auxiliary=list(cut.normal_shift_paths_v),
        )
        for cut in part.opening_cuts_among_groups
    ]
    _update_inter_group_paths_after_shift(part, previous_v, path.v, cut_positions)
    _sync_cut_spans(part, path, original_cuts)
    if previous.get("group_paths_follow_output"):
        previous_owners = [WireSegmentOwner(**row) for row in previous["output_segment_owners"]]
        _update_group_paths(part, previous_v, path, cut_positions, previous_owners)
    else:
        _update_group_paths(part, points, path, positions, owners)
    part.wire_path = path
    part.shift_array = factors
    part.points_to_shift = factors > 0
    part.wire_self_contacts = before
    physical_status = "requires_swept_body_check" if physical_pairs else "reference_path_screen_clear"
    part.normal_shift_diagnostics = {
        "normal_shift_status": "unresolved" if reasons else "complete",
        "normal_shift_failure_reasons": ";".join(reasons),
        "normal_shift_unresolved": bool(reasons),
        "normal_shift_quality_failure": bool(reasons),
        "normal_shift_method": detail.get("method", "arclength_crossing_layers"),
        "normal_shift_contours_protected": bool(detail.get("protected_contours", False)),
        "normal_shift_target_clearance_m": target,
        "normal_shift_physical_clearance_m": physical,
        "normal_shift_contact_count_before": len(before),
        "normal_shift_unresolved_contact_count": 0,
        "normal_shift_centerline_intersection_count": crossings,
        "normal_shift_physical_clearance_candidate_count": physical_pairs,
        "normal_shift_physical_clearance_status": physical_status,
        "normal_shift_maximum_lift_m": abs(config.normal_shift_length),
        "normal_shift_peak_lift_m": float(np.max(abs(offset))),
        "normal_shift_backtrack_count": len(backtracks),
        "normal_shift_introduced_backtrack_count": int(sum(~inherited)),
    }
    seg = np.minimum(positions.astype(int), len(points) - 2)
    part.normal_shift_provenance = {
        "schema_version": 2,
        "index_convention": "zero_based_before_cleanup",
        "method": detail.get("method", "arclength_crossing_layers"),
        "group_paths_follow_output": True,
        "points_before_m": points,
        "routing_footprint_m": detail.get("routing_footprint_m", points),
        "return_shortcuts": detail.get("return_shortcuts", []),
        "contacts_before_reference": "routing_footprint_m",
        "points_after_m": path.v.copy(),
        "normals": detail["source_normals"],
        "segment_owners": [dict(vars(o)) for o in owners],
        "segment_geometry_before": [g.value for g in detail["source_geometry"]],
        "output_source_segment_indices": seg,
        "output_source_segment_fractions": positions - seg,
        "output_segment_owners": [dict(vars(o)) for o in path.segment_owners],
        "output_segment_geometry": [g.value for g in path.segment_geometry],
        "lift_factors": factors.copy(),
        "lift_distance_m": offset,
        "lateral_displacement_m": detail["lateral_displacement_m"],
        "contacts_before": before,
        "contacts_after": final_contacts,
        "crossing_pairs_over_under_distance_m": detail["selected_contacts"],
        "shoulder_length_m": detail["shoulder_length_m"],
        "backtrack_point_indices": backtracks,
    }
    for key in ("output_normals", "output_footprint_m"):
        if key in detail:
            part.normal_shift_provenance[key] = detail[key]
    # A caller can evaluate a previous result's routed source. Invalidate
    # cached physical bodies and verdicts that belong to the previous route.
    part.route_diagnostics = {}
    part.fabrication_path = None
    part.swept_body = None
    part.swept_bodies = []
    return True


def _proposal_contacts(points, clearance, config):
    """Screen the stitched path with cleanup's 0.1-nm consecutive-duplicate tolerance."""
    from .layout import find_wire_segment_contacts_3d

    # A cut computed from each side can contribute two or more numerically
    # coincident stations. Their incoming/outgoing edges meet at the same
    # attachment, just as they do on a once-sampled vertex. Keep original
    # source indices in all reported witnesses and retain true return visits.
    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.all(length > 1e-9):
        return find_wire_segment_contacts_3d(points, clearance, config=config)
    retained = np.flatnonzero(np.r_[True, length > 1e-10])
    trimmed = points if len(retained) == len(points) else points[retained]
    pairs = find_wire_segment_contacts_3d(trimmed, clearance, config=config)
    segments = retained[1:] - 1
    # An attachment can also have a real but sub-nanometre connector edge,
    # above duplicate cleanup's tolerance. If the measured *positive* gap
    # equals its intervening arclength, the closest points are the consecutive
    # endpoints of the same through-running wire. Preserve actual crossings
    # and near-reversals, including ones separated by a similarly short edge.
    delta = np.diff(trimmed, axis=0)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(delta, axis=1))]
    closed = np.linalg.norm(trimmed[0] - trimmed[-1]) <= 1e-10
    result = []
    for a, b, distance in pairs:
        if 0.0 < distance <= 1e-9 and delta[a] @ delta[b] > 0:
            gap = arc[b] - arc[a + 1]
            if closed:
                gap = min(gap, arc[-1] - arc[b + 1] + arc[a])
            if gap <= 1e-9 and abs(gap - distance) <= 1e-13:
                continue
        result.append((int(segments[a]), int(segments[b]), distance))
    return result
