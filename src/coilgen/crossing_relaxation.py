"""Relax crowded connection cuts against their realized copper, preserving the solved turns."""

from __future__ import annotations

from copy import copy, deepcopy

import numpy as np

from .models import WirePath, WirePathGeometry, WireSegmentOwner
from .swept_clearance import required_copper_gap


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-13)


def _opposing_cut_legs(a, b):
    return (
        a.kind == b.kind == "inter_group_connector"
        and a.connector_index == b.connector_index
        and a.connector_role != b.connector_role
    )


def _positive_gap_normal(body, a, b, nsection):
    """Closest features of two already separated loft intervals determine the gap gradient."""
    from .geometry_kernels import segment_closest_parameters, triangle_pair_distances

    x = body.mesh.vertices[body.mesh.faces[2 * nsection * a : 2 * nsection * (a + 1)]]
    y = body.mesh.vertices[body.mesh.faces[2 * nsection * b : 2 * nsection * (b + 1)]]
    ix, iy = np.indices((len(x), len(y))).reshape(2, -1)
    distances = triangle_pair_distances(x[ix], y[iy], backend="numpy").values_m
    k = np.argmin(distances)
    x, y = x[ix[k]], y[iy[k]]
    # For disjoint triangles the minimizer is vertex/face or edge/edge.
    solutions = []
    for first, second, sign in ((x, y, 1), (y, x, -1)):
        e = np.roll(second, -1, axis=0) - second
        n = _unit(np.cross(e[0], e[1]))
        heights = (first - second[0]) @ n
        foot = first - heights[:, None] * n
        inside = np.all(np.cross(e[None, :, :], foot[:, None, :] - second[None, :, :]) @ n >= -1e-18, axis=1)
        solutions.extend(sign * heights[inside, None] * n)
    i, j = np.indices((3, 3)).reshape(2, -1)
    p, q = x[i], y[j]
    u, v = x[(i + 1) % 3] - p, y[(j + 1) % 3] - q
    s, t = segment_closest_parameters(p, u, q, v)
    solutions.extend(p + s[:, None] * u - q - t[:, None] * v)
    solutions = np.asarray(solutions)
    best = np.argmin(np.linalg.norm(solutions, axis=1))
    return _unit(solutions[best])


class _CutMotion:
    def __init__(self, part, config):
        from .layout import _inter_group_cut_width, _within_group_cut_width

        self.config = config
        self.detail = part.normal_shift_provenance
        d = self.detail
        self.points = d["points_after_m"]
        i = d["output_source_segment_indices"]
        f = d["output_source_segment_fractions"][:, None]
        source = d["points_before_m"]
        self.origin = d.get("output_footprint_m", source[i] + f * (source[i + 1] - source[i]))
        self.normals = _unit(d.get("output_normals", d["normals"][i] + f * (d["normals"][i + 1] - d["normals"][i])))
        self.owners = [WireSegmentOwner(**o) for o in d["output_segment_owners"]]
        self.geometry = list(map(WirePathGeometry, d["output_segment_geometry"]))
        before = list(map(WirePathGeometry, d["segment_geometry_before"]))
        self.footprints = np.asarray([before[index] != WirePathGeometry.FREE_3D for index in i])
        self.part = part
        self._mapped_wire = self._mapped_body = None
        self.closed = np.linalg.norm(self.points[-1] - self.points[0]) <= 1e-10
        self.arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        self.limit = 0.5 * min(_inter_group_cut_width(config), _within_group_cut_width(config))
        self.height = abs(config.normal_shift_length)
        self.sign = np.sign(config.normal_shift_length)
        self.ramp = max(2.5 * config.conductor_cross_section_width, 1.3 * self.limit)
        self.fraction = np.array([o.kind in {"inter_group_connector", "within_group_return"} for o in self.owners])
        edges = np.flatnonzero(self.fraction)
        self.authorized = np.zeros(len(self.arc))
        if len(edges):
            endpoints = np.unique(self.arc[np.r_[edges, edges + 1]])
            if self.closed:
                endpoints = np.r_[endpoints[-1] - self.arc[-1], endpoints, endpoints[0] + self.arc[-1]]
            index = np.searchsorted(endpoints, self.arc)
            dist = np.minimum(
                abs(self.arc - endpoints[np.maximum(0, index - 1)]),
                abs(self.arc - endpoints[np.minimum(len(endpoints) - 1, index)]),
            )
            self.authorized = np.clip(2.0 - dist / self.ramp, 0.0, 1.0)
        self.cut_fraction = self.authorized.copy()
        # Source turns can move in-plane only when both a contour conditioning
        # limit and a source-to-manufacture corridor have actually been enabled.
        turn_limit = 0.0
        if str(config.contour_path_regularization_mode).lower() != "off":
            turn_limit = min(
                config.contour_smoothing_max_displacement, config.sweep_corner_cluster_max_source_deviation_m
            )
        self.lateral_limit = np.maximum(
            self.limit * self.cut_fraction, np.full(len(self.arc), min(self.limit, turn_limit))
        )
        self.lateral_limit = np.maximum(self.lateral_limit, np.linalg.norm(d["lateral_displacement_m"], axis=1))
        self.height_limit = np.maximum(self.height * self.cut_fraction, np.abs(d["lift_distance_m"]))
        self.authorized = np.where(self.lateral_limit > 0, 1.0, self.cut_fraction)
        if not self.closed:
            self.authorized[[0, -1]] = 0.0

    def mount(self, coordinates):
        """Return bounded displacements with the footprint on the actual triangulated support."""
        from .manufacturing import _SurfaceProjector

        delta = coordinates - self.origin
        height = self.sign * np.clip(self.sign * np.einsum("ij,ij->i", delta, self.normals), 0.0, self.height)
        lateral = delta - height[:, None] * self.normals
        final_normals = self.normals.copy()
        changed = (np.linalg.norm(coordinates - self.points, axis=1) > 1e-13) & self.footprints
        if np.any(changed):
            project = _SurfaceProjector(self.part.mesh)
            old, _ = project.project(self.origin[changed])
            new, normal = project.project(self.origin[changed] + lateral[changed])
            if np.any(np.linalg.norm(new - old, axis=1) > self.lateral_limit[changed] + 1e-8):
                return None
            normal[np.einsum("ij,ij->i", normal, self.normals[changed]) < 0] *= -1
            final_normals[changed] = normal
            lateral[changed] = new - old
            coordinates[changed] = self.origin[changed] + lateral[changed] + height[changed, None] * normal
        if self.closed:
            coordinates[-1], height[-1], lateral[-1] = coordinates[0], height[0], lateral[0]
        return coordinates, height, lateral, final_normals

    def body_stations(self, wire, body):
        from .wire_provenance import ordered_source_stations
        from .manufacturing import _path_reference_point

        if self._mapped_wire is not wire or self._mapped_body is not body:
            f = body.fabrication_path
            reference = _path_reference_point(f.cross_section_m, self.config)
            carrier = (
                f.points_m
                - f.normals * (body.diagnostics.get("sweep_centerline_offset_m", 0.0) + reference[1])
                - f.transverse * reference[0]
            )
            if f.closed:
                carrier = np.vstack((carrier, carrier[0]))
            self._stations = ordered_source_stations(wire, carrier)
            self._mapped_wire, self._mapped_body = wire, body
        return self._stations

    def radial_limit(self, wire, indices):
        delta = wire - self.origin[indices]
        n = self.normals[indices]
        h = np.einsum("ij,ij->i", delta, n)
        lateral = delta - h[:, None] * n
        lateral *= np.minimum(1.0, self.lateral_limit[indices] / np.maximum(np.linalg.norm(lateral, axis=1), 1e-15))[
            :, None
        ]
        h = self.sign * np.clip(self.sign * h, 0.0, self.height_limit[indices])
        return self.origin[indices] + lateral + h[:, None] * n

    def window(self, index, t, radius):
        center = self.arc[index] + t * (self.arc[index + 1] - self.arc[index])
        distance = abs(self.arc - center)
        if self.closed:
            distance = np.minimum(distance, self.arc[-1] - distance)
        ids = np.flatnonzero((distance < radius) & (self.authorized > 0))
        weight = 0.5 * (1 + np.cos(np.pi * distance[ids] / radius)) * self.authorized[ids]
        return ids, weight

    def nearby_guards(self, rings, intervals, nsection, pairs, gap):
        from .group_route_validation import GroupRouteValidationLimits, _TriangleAuditBudgetExceeded
        from .swept_clearance import loft_pair_batches

        stations = np.asarray([x[:2] for x in pairs], dtype=int).ravel()
        arc = intervals.arc
        center = (arc[:-1] + arc[1:]) / 2
        active = np.zeros(len(center), dtype=bool)
        for station in np.unique(stations):
            dist = abs(center - center[station])
            if intervals.closed:
                dist = np.minimum(dist, arc[-1] - dist)
            active |= dist < 2.7 * self.ramp
        low = rings.min(axis=1)
        high = rings.max(axis=1)
        if intervals.closed:
            low = np.concatenate((low, low[:1]))
            high = np.concatenate((high, high[:1]))
        lower = np.minimum(low[:-1], low[1:])
        upper = np.maximum(high[:-1], high[1:])
        pairs_found = []
        try:
            for batch in loft_pair_batches(lower, upper, gap + self.limit, GroupRouteValidationLimits(), {}):
                i, j = batch.T
                keep = (active[i] | active[j]) & intervals.separate_visits(i * (2 * nsection), j * (2 * nsection))
                pairs_found.extend((int(a), int(b), None) for a, b in batch[keep])
        except _TriangleAuditBudgetExceeded:
            return []
        return pairs_found

    def constraints(self, wire, body, pairs, lateral_lanes=False):
        from .geometry_kernels import segment_closest_parameters
        from .swept_clearance import SweepIntervals

        fp = body.fabrication_path
        physical = fp.wire_points_m
        ring_count = len(fp.points_m)
        nsection = len(fp.cross_section_m)
        rings = body.mesh.vertices[: ring_count * nsection].reshape(ring_count, nsection, 3)
        required = required_copper_gap(self.config)
        intervals = SweepIntervals(fp, body.mesh.faces)
        pair_ids = np.asarray([p[:2] for p in pairs], dtype=int)
        separate = intervals.separate_visits(pair_ids[:, 0] * (2 * nsection), pair_ids[:, 1] * (2 * nsection))
        # Adjacent rings meeting in one tight bend need curb/fillet repair.
        # Forcing a gap across their intentional common section tears the bend.
        pairs = [pair for pair, keep in zip(pairs, separate) if keep]
        if not pairs:
            return []
        buffer = max(
            0.12 * required,
            0.025 * min(self.config.conductor_cross_section_width, self.config.conductor_cross_section_height),
        )
        obstacles = self.nearby_guards(rings, intervals, nsection, pairs, required)
        active_pairs = {(a, b) for a, b, _ in pairs}
        obstacles = [p for p in obstacles if (p[0], p[1]) not in active_pairs]
        cache = {}
        source_indices, source_fractions = self.body_stations(wire, body)
        for station in np.unique(np.array([row[:2] for row in pairs + obstacles], dtype=int)):
            next_station = (station + 1) % ring_count
            n = _unit(fp.normals[station] + fp.normals[next_station])
            center = 0.5 * (physical[station] + physical[station + 1])
            i, t = int(source_indices[station]), float(source_fractions[station])
            where = self.window(i, t, self.ramp)
            cache[station] = (
                i,
                t,
                where,
                np.concatenate((rings[station], rings[next_station])),
                center,
                _unit(physical[station + 1] - physical[station]),
                n,
            )
        constraints = []
        # Choose one consistent high/low ordering for opposing legs of the
        # same physical cut, even where the original lifts meet at mid-height.
        leads = {}
        for a, b, _ in pairs:
            if a not in cache or b not in cache:
                continue
            oa, ob = self.owners[source_indices[a]], self.owners[source_indices[b]]
            if _opposing_cut_legs(oa, ob):
                direction = _unit(cache[a][6] + cache[b][6]) * self.sign
                if lateral_lanes:
                    tangent = cache[a if oa.connector_role == "primary" else b][5]
                    direction = _unit(np.cross(tangent, direction))
                signed = (cache[a][4] - cache[b][4]) @ direction
                leads.setdefault(oa.connector_index, []).append(signed if oa.connector_role == "primary" else -signed)
        for a, b, pair_gap in pairs + obstacles:
            if a not in cache or b not in cache or a == b:
                continue
            i, t, wi, ra, ca, ta, na = cache[a]
            j, u, wj, rb, cb, tb, nb = cache[b]
            if i == j or not (len(wi[0]) or len(wj[0])):
                continue
            # A separating axis can be the support normal, either in-plane
            # transverse, or the shortest connector between the station axes.
            s, v = segment_closest_parameters(
                physical[[a]], np.diff(physical[a : a + 2], axis=0), physical[[b]], np.diff(physical[b : b + 2], axis=0)
            )
            sep = (
                physical[a]
                + s[0] * (physical[a + 1] - physical[a])
                - physical[b]
                - v[0] * (physical[b + 1] - physical[b])
            )
            axis = _unit(np.vstack((sep, na, nb, np.cross(ta, na), np.cross(tb, nb), np.cross(ta, tb), ca - cb)))
            axis = np.vstack((axis, -axis))
            present = (ra @ axis.T).min(axis=0) - (rb @ axis.T).max(axis=0)
            if pair_gap is not None and pair_gap > 1e-10:
                direction = _positive_gap_normal(body, a, b, nsection)
                if np.linalg.norm(direction) > 0.5:
                    axis = np.vstack((axis, direction))
                    present = np.r_[present, pair_gap]
            need = required + buffer - present

            def response(index, fraction, tangent, dirs):
                # Tangential sliding on a straight segment cannot change its
                # separation. Use the transverse component and clip its trial
                # endpoint against the signed lift and lateral motion limits.
                dirs = dirs - np.outer(dirs @ tangent, tangent)
                k = index + int(fraction > 0.5)
                ids = np.full(len(dirs), k)
                trial = self.radial_limit(wire[k] + dirs * (self.limit + self.height), ids) - wire[k]
                return trial * self.authorized[k]

            up = response(i, t, ta, axis)
            down = response(j, u, tb, -axis)
            room = np.einsum("ij,ij->i", up - down, axis)
            viable = (room > 1e-9) & (np.linalg.norm(axis, axis=1) > 0.5)
            if not viable.any():
                continue
            # Minimize demanded fractional travel while avoiding large motion
            # when another axis already almost separates these two solids.
            cost = np.where(viable, need / np.maximum(room, 1e-15) + 0.15 * need / max(self.limit, 1e-12), np.inf)
            oa, ob = self.owners[source_indices[a]], self.owners[source_indices[b]]
            if pair_gap is None:
                # Preserve the existing separating plane of a close, currently
                # clear turn before the repelled branch is allowed to descend.
                k = int(np.argmax(present))
                if present[k] < required or not viable[k]:
                    continue
                constraints.append((min(0.0, need[k]), axis[k], room[k], i, t, up[k], wi, j, u, down[k], wj))
                continue
            if oa.connector_index in leads and _opposing_cut_legs(oa, ob):
                expected = self.sign * (1 if np.median(leads[oa.connector_index]) >= 0 else -1)
                expected *= 1 if oa.connector_role == "primary" else -1
                preferred = _unit(na + nb)
                if lateral_lanes:
                    preferred = _unit(np.cross(ta if oa.connector_role == "primary" else tb, preferred))
                # Use the same lane ordering along both opposing arms, so a
                # correction at one witness need not swap them at the next.
                choice = np.argmax(expected * (axis @ preferred))
                if viable[choice] and need[choice] < room[choice] * 1.1:
                    cost[choice] = 0.0
                    wi = self.window(i, t, 1.7 * self.ramp)
                    wj = self.window(j, u, 1.7 * self.ramp)
            k = int(np.argmin(cost))
            constraints.append((need[k], axis[k], room[k], i, t, up[k], wi, j, u, down[k], wj))
        constraints.sort(key=lambda x: -x[0])
        return constraints

    def step(self, wire, constraints, scale):
        current = wire.copy()
        for _ in range(3):
            for need, axis, room, i, t, up, wi, j, u, down, wj in constraints:
                pi = (1 - t) * (current[i] - wire[i]) + t * (current[i + 1] - wire[i + 1])
                pj = (1 - u) * (current[j] - wire[j]) + u * (current[j + 1] - wire[j + 1])
                target = scale * need if need > 0 else need
                step = np.clip((target - (pi - pj) @ axis) / room, 0.0, 0.7)
                if step == 0:
                    continue
                for (ids, weight), direction in ((wi, up), (wj, down)):
                    current[ids] = self.radial_limit(current[ids] + weight[:, None] * (step * direction), ids)
        if self.closed:
            current[-1] = current[0]
        return current


def _swept_score(part):
    d = part.swept_bodies[0].diagnostics
    if not d.get("exact_collision_completed", False):
        return None
    score = int(d["exact_intersection_count"]), int(d.get("copper_gap_violating_facet_pairs", 0))
    if score == (0, 0) and not d.get("mesh_solid_ok", False):
        return None
    return score


def _clusters(body, pairs, radius):
    """Partition physical witnesses by both owning station arclengths, including the seam."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(pairs) <= 1:
        return []
    p = body.fabrication_path
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p.wire_points_m, axis=0), axis=1))]
    mid = (arc[1:] + arc[:-1]) / 2
    ids = np.array([r[:2] for r in pairs], dtype=int)
    coords = mid[ids]
    count = len(coords)
    both = np.concatenate((coords, coords[:, ::-1]))
    tree = cKDTree(both, boxsize=arc[-1] if p.closed else None)
    edges = tree.query_pairs(radius, p=np.inf, output_type="ndarray") % count
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    groups = [[r for r, label in zip(pairs, labels) if label == group] for group in np.unique(labels)]
    # Small independent repairs cannot collide with a far-away unmodified station.
    return sorted(groups, key=len) if len(groups) > 1 else []


def _remote_merit(body, config):
    """Continuous overlap depth and gap shortfall, with a separate all-facet final gate."""
    from .swept_clearance import SweepIntervals

    f = body.fabrication_path
    diag = body.diagnostics
    pairs = np.asarray(diag.get("copper_clearance_witness_intervals", ()), dtype=float)
    if not len(pairs):
        return 0.0
    n = len(f.cross_section_m)
    a, b = pairs[:, :2].astype(int).T
    keep = SweepIntervals(f, body.mesh.faces).separate_visits(2 * n * a, 2 * n * b)
    a, b, d = a[keep], b[keep], pairs[keep, 2]
    if not len(a):
        return 0.0
    r = body.mesh.vertices[: len(f.points_m) * n].reshape(-1, n, 3)
    blocks = np.concatenate((r, np.roll(r, -1, axis=0)), axis=1)
    directions = np.stack(
        [f.normals[a], f.normals[b], f.transverse[a], f.transverse[b], _unit(np.cross(f.tangents[a], f.tangents[b]))],
        axis=1,
    )
    av = np.einsum("iks,ijs->ijk", blocks[a], directions)
    bv = np.einsum("iks,ijs->ijk", blocks[b], directions)
    separating = np.maximum(av.min(axis=2) - bv.max(axis=2), bv.min(axis=2) - av.max(axis=2))
    separating[np.linalg.norm(directions, axis=2) < 0.5] = -np.inf
    penetration = np.maximum(0.0, -separating.max(axis=1))
    deficit = np.maximum(0.0, required_copper_gap(config) - d) + np.where(d < 1e-10, penetration, 0.0)
    return float(np.sum(deficit))


class _GapPreview:
    """Rank proposed moves on translated section rings. Every adoption still uses a full resweep."""

    def __init__(self, body, motion, wire):
        from .swept_clearance import SweepIntervals

        self.body = body
        self.path = body.fabrication_path
        self.section = len(self.path.cross_section_m)
        self.intervals = SweepIntervals(self.path, body.mesh.faces)
        self.required = required_copper_gap(motion.config)
        self.centers = self.path.points_m
        rings = len(self.centers)
        self.rings = body.mesh.vertices[: rings * self.section].reshape(rings, self.section, 3)
        ids, fractions = motion.body_stations(wire, body)
        lengths = np.linalg.norm(np.diff(wire, axis=0), axis=1)
        arc = np.r_[0.0, lengths.cumsum()]
        visits = arc[ids] + fractions * lengths[ids]
        ring_arc = self.intervals.arc
        mid = (ring_arc[1:] + ring_arc[:-1]) / 2
        if self.path.closed:
            visits = np.unwrap(visits, period=arc[-1])
            period = np.sign(visits[-1] - visits[0]) * arc[-1]
            mid = np.r_[mid[-1] - ring_arc[-1], mid, mid[0] + ring_arc[-1]]
            visits = np.r_[visits[-1] - period, visits, visits[0] + period]
            pos = np.interp(ring_arc[:-1], mid, visits) % arc[-1]
        else:
            pos = np.interp(
                ring_arc,
                mid,
                visits,
                left=0.0 if visits[-1] >= visits[0] else arc[-1],
                right=arc[-1] if visits[-1] >= visits[0] else 0.0,
            )
        self.index = np.clip(np.searchsorted(arc, pos, side="right") - 1, 0, len(lengths) - 1)
        self.fraction = np.clip((pos - arc[self.index]) / np.maximum(lengths[self.index], 1e-30), 0.0, 1.0)

    def merit(self, change):
        from .group_route_validation import GroupRouteValidationLimits, _TriangleAuditBudgetExceeded
        from .geometry_kernels import triangle_pair_distances
        from .swept_clearance import loft_pair_batches

        offset = change[self.index] * (1 - self.fraction[:, None]) + change[self.index + 1] * self.fraction[:, None]
        rings = self.rings + offset[:, None, :]
        following = np.roll(rings, -1, axis=0)
        blocks = np.concatenate((rings, following), axis=1)
        s = len(self.path.segment_owners)
        lower = blocks[:s].min(axis=1)
        upper = blocks[:s].max(axis=1)
        faces = self.body.mesh.faces[: s * 2 * self.section].reshape(s, 2 * self.section, 3)
        vertices = rings.reshape(-1, 3)
        total = 0.0
        try:
            for pair in loft_pair_batches(lower, upper, self.required + 1e-12, GroupRouteValidationLimits(), {}):
                a, b = pair.T
                ok = self.intervals.separate_visits(a * (2 * self.section), b * (2 * self.section))
                a, b = a[ok], b[ok]
                if not len(a):
                    continue
                # Evaluate only overlapping per-facet boxes in each station pair.
                ta, tb = vertices[faces[a]], vertices[faces[b]]
                la, ha = ta.min(axis=2), ta.max(axis=2)
                lb, hb = tb.min(axis=2), tb.max(axis=2)
                gap = np.maximum(0, np.maximum(la[:, :, None] - hb[:, None], lb[:, None] - ha[:, :, None]))
                row, i, j = np.nonzero(np.einsum("...k,...k->...", gap, gap) <= (self.required + 1e-12) ** 2)
                if not len(row):
                    continue
                d = np.full(len(a), np.inf)
                np.minimum.at(d, row, triangle_pair_distances(ta[row, i], tb[row, j], backend="numpy").values_m)
                total += np.maximum(0, self.required - d).sum()
                hit = d < 1e-10
                if np.any(hit):
                    aa, bb = a[hit], b[hit]
                    directions = np.stack(
                        [
                            self.path.normals[aa],
                            self.path.normals[bb],
                            self.path.transverse[aa],
                            self.path.transverse[bb],
                            _unit(np.cross(self.path.tangents[aa], self.path.tangents[bb])),
                        ],
                        axis=1,
                    )
                    av = np.einsum("iks,ijs->ijk", blocks[aa], directions)
                    bv = np.einsum("iks,ijs->ijk", blocks[bb], directions)
                    distance = np.maximum(av.min(axis=2) - bv.max(axis=2), bv.min(axis=2) - av.max(axis=2))
                    distance[np.linalg.norm(directions, axis=2) < 0.5] = -np.inf
                    total += np.maximum(0.0, -distance.max(axis=1)).sum()
        except _TriangleAuditBudgetExceeded:
            return np.inf
        return float(total)


def improve_crossing_candidate(part, config, *, max_iterations=5):
    """Return an isolated, fully rescored winding with fewer exact defects, or None.

    Partial improvements retain an unresolved status for subsequent conditioning.
    """
    from .layout import cleanup_wire_paths
    from .manufacturing import create_swept_coil_meshes

    if (
        not part.swept_bodies
        or len(part.swept_bodies) != 1
        or config.skip_normal_shift
        or part.normal_shift_provenance.get("method") != "arclength_crossing_layers"
    ):
        return None
    motion = _CutMotion(part, config)
    score = _swept_score(part)
    if score is None or score == (0, 0) or not np.any(motion.authorized) or min(motion.limit, motion.height) <= 0:
        return None
    best = part
    retained = None
    retained_quality = None
    retained_score = score
    wire = motion.points
    merit = _remote_merit(part.swept_bodies[0], config)
    sweeps = 0
    for iteration in range(max_iterations):
        diag = best.swept_bodies[0].diagnostics
        pairs = diag.get("copper_clearance_witness_intervals", ())
        if not len(pairs) or sweeps >= 2 * max_iterations + 2:
            break
        body = best.swept_bodies[0]
        groups = _clusters(body, pairs, 2 * motion.ramp)
        candidates = ([groups[-1]] + groups[:2]) if len(groups) > 3 else groups
        plans = (
            [(pairs, 1.25, False), (pairs, 0.45, False)]
            + [(g, 0.85, False) for g in candidates]
            + [(pairs, 0.2, False)]
        )
        owners = body.fabrication_path.segment_owners
        if any(_opposing_cut_legs(owners[a], owners[b]) for a, b, _ in pairs):
            plans.extend([(pairs, 0.8, True), (pairs, 0.32, True)])
        prepared = {}
        preview = (
            _GapPreview(body, motion, wire)
            if len(pairs) > 3 and len(body.fabrication_path.cross_section_m) <= 8
            else None
        )
        trials = []
        for order, (active, scale, lateral_lanes) in enumerate(plans):
            key = id(active), lateral_lanes
            if key not in prepared:
                prepared[key] = motion.constraints(wire, body, active, lateral_lanes)
            coordinates = motion.step(wire, prepared[key], scale)
            if np.max(np.linalg.norm(coordinates - wire, axis=1)) < 1e-8:
                continue
            mounted = motion.mount(coordinates)
            if mounted is None:
                continue
            rank = preview.merit(mounted[0] - wire) if preview is not None else order
            trials.append((rank, order, mounted))
        for _, _, mounted in sorted(trials, key=lambda entry: entry[:2]):
            coordinates, height, lateral, final_normals = mounted
            geometry = [
                WirePathGeometry.OFFSET_SURFACE
                if g != WirePathGeometry.FREE_3D and max(abs(height[i : i + 2])) > 1e-10
                else g
                for i, g in enumerate(motion.geometry)
            ]
            trial = copy(part)
            trial.wire_path = WirePath(coordinates, list(motion.owners), geometry)
            trial.shift_array = height / config.normal_shift_length
            trial.points_to_shift = trial.shift_array > 0
            trial.normal_shift_provenance = {
                **motion.detail,
                "points_after_m": coordinates.copy(),
                "lift_distance_m": height.copy(),
                "lift_factors": trial.shift_array.copy(),
                "lateral_displacement_m": lateral,
                "output_footprint_m": motion.origin,
                "output_normals": final_normals,
                "output_segment_geometry": [v.value for v in geometry],
                "physical_relaxation_iterations": iteration + 1,
            }
            try:
                cleanup_wire_paths([trial], config=config)
            except ValueError as exc:
                if not str(exc).startswith("Route quality checks failed"):
                    raise
                continue
            if sweeps >= 2 * max_iterations + 2:
                break
            create_swept_coil_meshes([trial], config)
            sweeps += 1
            trial_score = _swept_score(trial)
            if trial_score is None:
                continue
            if trial_score < retained_score:
                quality = _source_quality(part, motion, coordinates, config, trial_score)
                if quality is not None:
                    retained, retained_score, retained_quality = trial, trial_score, quality
                elif trial_score == (0, 0):
                    # A smoothing-induced copper pass can hide a defect in
                    # the electrical source. Try the remaining bounded moves.
                    continue
            next_merit = _remote_merit(trial.swept_bodies[0], config)
            if trial_score == (0, 0) or (round(next_merit, 9), trial_score) < (round(merit, 9), score):
                best, score, wire, merit = trial, trial_score, coordinates, next_merit
                break
        else:
            break
        if retained_score == (0, 0):
            break
    if retained is None:
        return None
    return _finish_candidate(part, retained, motion, config, retained_score, retained_quality)


def _source_quality(part, motion, wire, config, score):
    from .crossing_lifts import _proposal_contacts
    from .layout import _wire_backtrack_indices, _wire_clearance_failure_threshold

    target = _wire_clearance_failure_threshold(config)
    existing = part.normal_shift_diagnostics
    physical = float(existing["normal_shift_physical_clearance_m"])
    contacts = _proposal_contacts(wire, max(physical, target, 1e-9), config)
    if any(d + 1e-9 < target for _, _, d in contacts):
        return None
    backtracks = _wire_backtrack_indices(wire)
    positions = motion.detail["output_source_segment_indices"] + motion.detail["output_source_segment_fractions"]
    mapped = positions[backtracks]
    inherited = _wire_backtrack_indices(motion.detail["points_before_m"])
    introduced = int(
        np.count_nonzero((np.abs(np.rint(mapped) - mapped) > 1e-10) | ~np.isin(np.rint(mapped), inherited))
    )
    crossings = sum(d <= 1e-9 for _, _, d in contacts)
    already_failed = score != (0, 0)
    if crossings > (existing.get("normal_shift_centerline_intersection_count", 0) if already_failed else 0):
        return None
    if introduced > (existing.get("normal_shift_introduced_backtrack_count", 0) if already_failed else 0):
        return None
    return contacts, backtracks, crossings, introduced


def _finish_candidate(part, candidate, motion, config, score, quality):
    from .crossing_lifts import _update_group_paths, _sync_cut_spans
    from .layout import _update_inter_group_paths_after_shift, route_diagnostics

    contacts, backtracks, crossings, introduced = quality
    physical = float(part.normal_shift_diagnostics["normal_shift_physical_clearance_m"])
    result = candidate.normal_shift_provenance
    wire = result["points_after_m"]
    raw_path = WirePath(wire, list(motion.owners), list(map(WirePathGeometry, result["output_segment_geometry"])))
    candidate.connected_groups = deepcopy(part.connected_groups)
    candidate.opening_cuts_among_groups = deepcopy(part.opening_cuts_among_groups)
    cuts = [
        dict(primary=c.connection_path_v, closing=c.closing_connection_path_v, auxiliary=list(c.normal_shift_paths_v))
        for c in candidate.opening_cuts_among_groups
    ]
    local_positions = np.arange(len(wire), dtype=float)
    _update_inter_group_paths_after_shift(candidate, motion.points, wire, local_positions)
    _update_group_paths(candidate, motion.points, raw_path, local_positions, motion.owners)
    _sync_cut_spans(candidate, raw_path, cuts)
    result.update(contacts_after=contacts, backtrack_point_indices=backtracks)
    failed = score != (0, 0)
    reasons = [candidate.manufacturing_diagnostics.get("copper_clearance_failure", "")]
    if crossings:
        reasons.append("centerline_intersection")
    if introduced:
        reasons.append("centerline_backtrack")
    candidate.normal_shift_diagnostics = {
        **part.normal_shift_diagnostics,
        "physical_relaxation_iterations": result["physical_relaxation_iterations"],
        "normal_shift_status": "unresolved" if failed else "complete",
        "normal_shift_failure_reasons": ";".join(filter(None, reasons)),
        "normal_shift_unresolved": failed,
        "normal_shift_quality_failure": failed,
        "normal_shift_peak_lift_m": float(np.max(abs(result["lift_distance_m"]))),
        "normal_shift_backtrack_count": len(backtracks),
        "normal_shift_introduced_backtrack_count": introduced,
        "normal_shift_centerline_intersection_count": crossings,
        "normal_shift_unresolved_contact_count": 0,
        "normal_shift_physical_clearance_candidate_count": sum(d + 1e-9 < physical for _, _, d in contacts),
        "normal_shift_physical_clearance_status": "unresolved_swept_body" if failed else "verified_swept_body",
    }
    candidate.route_diagnostics = route_diagnostics(candidate, config)
    return candidate


def relax_crossing_candidate(part, config, *, max_iterations=5):
    """Return only an independently reswept, physically clear cut relaxation."""
    candidate = improve_crossing_candidate(part, config, max_iterations=max_iterations)
    return candidate if candidate is not None and _swept_score(candidate) == (0, 0) else None
