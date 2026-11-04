"""Bounded relief of profile reversals on attached contours and raised handoffs."""

import numpy as np

from .models import WirePathGeometry
from .wire_provenance import nearest_source_segment_indices


class _PlanarInterior:
    """Exact orthogonal projection in the interior of a convex planar support."""

    def __init__(self, mesh, projector, edges):
        self.projector = projector
        self.normal = projector.face_normals[0]
        self.origin = mesh.vertices[0]
        self.starts = mesh.vertices[edges[:, 0]]
        self.outward = np.cross(mesh.vertices[edges[:, 1]] - self.starts, self.normal)
        self.bounds = np.einsum("ij,ij->i", self.starts, self.outward)

    def project(self, points):
        projected = points - ((points - self.origin) @ self.normal)[:, None] * self.normal
        normals = np.broadcast_to(self.normal, projected.shape).copy()
        outside = np.any(projected @ self.outward.T > self.bounds, axis=1)
        if np.any(outside):
            projected[outside], normals[outside] = self.projector.project(points[outside])
        return projected, normals

    @classmethod
    def for_mesh(cls, mesh, projector):
        normals = projector.face_normals
        if not len(normals) or np.max(np.linalg.norm(normals - normals[0], axis=1)) > 1e-12:
            return projector
        if (
            abs(np.linalg.norm(normals[0]) - 1) > 1e-12
            or np.max(np.linalg.norm(projector.vertex_normals - normals[0], axis=1)) > 1e-12
        ):
            return projector
        if np.max(np.abs((mesh.vertices - mesh.vertices[0]) @ normals[0])) > 1e-12:
            return projector
        faces = mesh.faces
        boundary = projector.boundary_edges
        edges = [(a, b) for tri in faces for a, b in zip(tri, np.roll(tri, -1)) if tuple(sorted((a, b))) in boundary]
        if not edges or len(edges) > 1024:
            return projector
        plane = cls(mesh, projector, np.array(edges))
        if any(np.any(plane.outward @ p > plane.bounds + 1e-14) for p in mesh.vertices):
            return projector
        return plane


def reversed_profile_edges(vertices, section_count, *, closed):
    """Stations where a profile edge runs backwards along its centroid chord."""
    rings = vertices.reshape(-1, section_count, 3)
    if closed:
        rings = np.concatenate((rings, rings[:1]))
    edges = np.diff(rings, axis=0)
    chords = np.mean(edges, axis=1)
    forward = np.einsum("ijk,ik->ij", edges, chords)
    return np.flatnonzero(np.min(forward, axis=1) < -1e-14 * np.einsum("ij,ij->i", chords, chords))


def _short_flank_gap(rings, station, gap, closed):
    """Check the two nonadjacent flanks of one coarse hairpin using the real loft diagonals."""
    from .geometry_kernels import triangle_pair_distances
    from .manufacturing import _triangulate_sweep_laterals

    indices = np.arange(station - 2, station + 3)
    if len(rings) < 5 or (not closed and (indices[0] < 0 or indices[-1] >= len(rings))):
        return False
    local = rings[indices % len(rings)]
    count = local.shape[1]
    flat = local.reshape(-1, 3)
    faces, _ = _triangulate_sweep_laterals(flat, 5, count, closed=False, optimize_diagonals=True)
    triangles = flat[np.asarray(faces)].reshape(4, 2 * count, 3, 3)
    legs = np.diff(local.mean(axis=1), axis=0)
    lengths = np.linalg.norm(legs, axis=1)
    units = legs / np.maximum(lengths[:, None], 1e-15)
    turns = np.arccos(np.clip(np.einsum("ij,ij->i", units[1:], units[:-1]), -1, 1))
    for a, b in ((0, 2), (0, 3), (1, 3)):
        if turns[a:b].sum() < np.pi / 2:
            continue
        left, right = triangles[a], triangles[b]
        if np.any(left.min(axis=(0, 1)) > right.max(axis=(0, 1)) + gap) or np.any(
            right.min(axis=(0, 1)) > left.max(axis=(0, 1)) + gap
        ):
            continue
        # Tile the profiles so nonrectangular user-defined sections stay bounded.
        for face in left:
            distances = triangle_pair_distances(np.broadcast_to(face, right.shape), right).values_m
            if np.any(distances < gap):
                return True
    return False


def profile_fold_stations(vertices, section_count, *, closed, bend_radius=0.0, gap=0.0):
    """Locate inverted folds and returning legs too tight for the specified bend radius."""
    reverse = reversed_profile_edges(vertices, section_count, closed=closed)
    rings = vertices.reshape(-1, section_count, 3)
    centers = rings.mean(axis=1)
    previous = np.roll(centers, 1, axis=0) - centers
    following = np.roll(centers, -1, axis=0) - centers
    normal = np.cross(previous, following)
    square = np.einsum("ij,ij->i", normal, normal)
    first_sq = np.einsum("ij,ij->i", previous, previous)
    next_sq = np.einsum("ij,ij->i", following, following)
    valid = square > 1e-16 * first_sq * next_sq
    if not closed:
        valid[[0, -1]] = False
    # Radius vector from each station to its three-point circle centre.
    numerator = first_sq[:, None] * np.cross(following, normal) + next_sq[:, None] * np.cross(normal, previous)
    radius = np.divide(numerator, 2 * square[:, None], out=np.zeros_like(numerator), where=valid[:, None])
    radius_sq = np.einsum("ij,ij->i", radius, radius)
    inward = np.einsum("ijk,ik->ij", rings - centers[:, None], radius)
    risk = valid & (inward.max(axis=1) >= radius_sq)
    if bend_radius > 0 and gap > 0:
        # Long samples can straddle a tight hairpin without placing a ring
        # inside the bend. Its two returning flanks may still violate spacing.
        cosine_numerator = np.einsum("ij,ij->i", previous, following)
        hairpins = cosine_numerator > np.sqrt(0.5 * first_sq * next_sq)
        if not closed:
            hairpins[[0, -1]] = False
        near_inner_face = valid & (inward.max(axis=1) + gap * np.sqrt(radius_sq) >= radius_sq)
        for i in np.flatnonzero((hairpins | near_inner_face) & ~risk & (radius_sq < bend_radius**2)):
            risk[i] = _short_flank_gap(rings, i, gap, closed)
    return np.union1d(reverse, np.flatnonzero(risk))


def _directed_bound(points, line, step):
    # The distance to a polyline is 1-Lipschitz. Sampling every interval
    # bounds the *entire* curve, in addition to its stored vertices.
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    counts = np.maximum(1, np.ceil(lengths / step).astype(int))
    if counts.sum() > 8192 or len(line) > 8192:
        return np.inf
    samples = np.concatenate(
        [p + v * (np.arange(n) / n)[:, None] for p, v, n in zip(points[:-1], vectors, counts)] + [points[-1:]]
    )
    legs = np.diff(line, axis=0)
    squared = np.maximum(np.einsum("ij,ij->i", legs, legs), 1e-30)
    worst = 0.0
    for start in range(0, len(samples), 64):
        offset = samples[start : start + 64, None] - line[:-1]
        t = np.einsum("ijk,jk->ij", offset, legs) / squared
        delta = offset - np.clip(t, 0, 1)[:, :, None] * legs
        worst = max(worst, np.linalg.norm(delta, axis=2).min(axis=1).max())
    return worst + 0.5 * np.max(lengths / counts)


def _radii(points):
    legs = np.diff(points, axis=0)
    length = np.linalg.norm(legs, axis=1)
    area = np.linalg.norm(np.cross(legs[:-1], legs[1:]), axis=1)
    chord = np.linalg.norm(points[2:] - points[:-2], axis=1)
    radii = np.divide(chord * length[:-1] * length[1:], 2 * area, out=np.full(len(area), np.inf), where=area > 1e-20)
    reversal = np.einsum("ij,ij->i", legs[:-1], legs[1:]) < -0.999 * length[:-1] * length[1:]
    radii[reversal | (length[:-1] < 1e-12) | (length[1:] < 1e-12)] = 0
    return radii


def _clip_source(raw, start, finish):
    """Trim the nominal source to nearest points at the two attachment stations."""
    legs = np.diff(raw, axis=0)
    ends = []
    for point in (start, finish):
        t = np.einsum("ij,ij->i", point - raw[:-1], legs) / np.maximum(np.einsum("ij,ij->i", legs, legs), 1e-30)
        projections = raw[:-1] + np.clip(t, 0, 1)[:, None] * legs
        i = np.argmin(np.linalg.norm(projections - point, axis=1))
        ends.append((i, projections[i]))
    (i, p), (j, q) = ends
    return np.vstack((p, raw[i + 1 : j + 1], q)) if j >= i else np.empty((0, 3))


def _raised_handoff(owners):
    # A raised curve may round a single attachment without shortcutting the
    # other end of a connector or removing the bight between successive turns.
    keys = [
        (owner.kind, owner.group_index, owner.loop_index, owner.connector_index, owner.connector_role)
        for owner in owners
    ]
    spans = [k for i, k in enumerate(keys) if not i or keys[i - 1] != k]
    return (
        len(spans) <= 3
        and len({k for k in spans if k[0] == "contour_loop"}) <= 1
        and len({k for k in spans if k[0] == "inter_group_connector"}) <= 1
        and all(
            k[0] in {"contour_group", "contour_loop", "within_group_return", "inter_group_connector"} for k in spans
        )
    )


def _spiral_attachment(owner, group):
    return (
        group >= 0
        and owner.group_index == group
        and (
            owner.kind == "contour_loop"
            or (owner.kind == "contour_group" and owner.connector_role == "joint_spiral_join")
        )
    )


def _same_turn_attachment(owners):
    spans = [owner for index, owner in enumerate(owners) if not index or owner != owners[index - 1]]
    return (
        len(spans) <= 2
        and all(_spiral_attachment(o, spans[0].group_index) for o in spans)
        and (sum(o.kind == "contour_loop" for o in spans) <= 1)
    )


def _lift_profile(points, projector):
    support, normals = projector.project(points)
    distances = np.einsum("ij,ij->i", points - support, normals)
    return _normalized_arclength(points), distances


def _preserve_bridge(trial, original, projector, maximum_height):
    # Hermite already matches incoming/outgoing 3-D tangents. Use a clamped
    # bump if needed, instead of projecting each sample back onto the mesh
    # (which would introduce facet kinks and could flatten the bridge).
    arc, height = original
    projected, normals = projector.project(trial)
    signed = np.einsum("ij,ij->i", trial - projected, normals)
    orientation = np.sign(height[np.argmax(np.abs(height))])
    if not orientation or np.any(orientation * height < -1e-8):
        return None
    normalized = _normalized_arclength(trial)
    floor = np.interp(normalized, arc, height) * orientation
    clearance = signed * orientation
    window = np.sin(np.pi * normalized) ** 2
    upper = max(maximum_height, float(np.max(height * orientation)))
    bump = max(0.0, float(np.max((floor[1:-1] - clearance[1:-1]) / window[1:-1])))
    adjustment = bump * window
    if np.any(clearance + adjustment > upper):
        from scipy.optimize import linprog
        from scipy.special import comb

        # Shape the correction when a symmetric elevation would overshoot the
        # prescribed layer. All basis functions vanish with zero end slope.
        degrees = np.arange(5)
        u = normalized[:, None]
        basis = (window[:, None] * comb(4, degrees) * u**degrees * (1 - u) ** (4 - degrees))[1:-1]
        scaled = (np.r_[floor[1:-1] - clearance[1:-1], clearance[1:-1] - upper]) / upper
        solution = linprog(
            np.ones(10),
            A_ub=np.block([[-basis, basis], [basis, -basis]]),
            b_ub=-scaled,
            bounds=(0.0, 8.0),
            method="highs",
        )
        if not solution.success:
            return None
        adjustment[1:-1] = upper * (basis @ (solution.x[:5] - solution.x[5:]))
    candidate = trial + (adjustment * orientation)[:, None] * normals
    candidate[[0, -1]] = trial[[0, -1]]
    pr, n = projector.project(candidate)
    gap = np.einsum("ij,ij->i", candidate - pr, n) * orientation
    if np.any(gap < floor - 1e-8) or np.max(gap) > upper + 1e-8:
        return None
    return candidate


def _normalized_arclength(points):
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return arc / arc[-1]


def _tangent_curves(base, slope_a, slope_b, directions, radius, original, step):
    for gain_a, gain_b in (
        (1, 1),
        (0.8, 0.8),
        (1.2, 1.2),
        (0.6, 0.6),
        (1.4, 1.4),
        (0.7, 1.1),
        (1.1, 0.7),
        (0.8, 1.4),
        (1.4, 0.8),
    ):
        yield base + gain_a * slope_a + gain_b * slope_b
    if directions[0] @ directions[1] > -0.4:
        return
    # A simple cubic on nearly antiparallel legs collapses the inside of the
    # hairpin. Separate its legs locally while retaining both end tangents.
    common = directions[0] - directions[1]
    separation = base[-1] - base[0]
    lateral = separation - common * ((separation @ common) / (common @ common))
    norm = np.linalg.norm(lateral)
    if norm <= 1e-12:
        return
    side = lateral / norm
    advance = common / np.linalg.norm(common)
    apex = original[np.argmax(original @ advance)]

    def flank(start, finish, first_tangent, last_tangent, gain):
        length = np.linalg.norm(finish - start)
        u = np.linspace(0, 1, max(4, int(np.ceil(length / step))) + 1)[:, None]
        return (
            (1 - u) ** 2 * (1 + 2 * u) * start
            + u**2 * (3 - 2 * u) * finish
            + gain * length * (u * (1 - u) ** 2 * first_tangent - u**2 * (1 - u) * last_tangent)
        )

    for arc_radius in (radius * 1.05, radius * 1.2):
        angle = np.linspace(-np.pi / 2, np.pi / 2, max(12, int(np.ceil(np.pi * arc_radius / step))) + 1)[:, None]
        arc = arc_radius * (np.cos(angle) * advance + np.sin(angle) * side)
        for retreat in (1, 0.5, 1.5):
            bend = apex - retreat * arc_radius * advance + arc
            if (bend[0] - base[0]) @ advance < radius or (bend[-1] - base[-1]) @ advance < radius:
                continue
            for gain in (1, 0.7, 1.3):
                incoming = flank(base[0], bend[0], directions[0], advance, gain)
                outgoing = flank(bend[-1], base[-1], -advance, directions[1], gain)
                yield np.vstack((incoming[:-1], bend[:-1], outgoing))


def _fit_span(span, source, projector, *, length, radius, corridor, sample_spacing, raised, maximum_height):
    start, finish = span[1], span[-2]
    heights = _lift_profile(span[1:-1], projector) if raised else None
    incoming = start - span[0]
    outgoing = span[-1] - finish
    directions = np.array([incoming, outgoing])
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms < 1e-12):
        return None
    directions /= norms[:, None]
    count = max(8, int(np.ceil(1.5 * length / sample_spacing)))
    if count > 4096:
        return None
    t = np.linspace(0, 1, count + 1)[:, None]
    # Cubic Hermite span between unchanged adjacent source tangents.
    base = (1 - t) ** 2 * (1 + 2 * t) * start + t**2 * (3 - 2 * t) * finish
    slope_a = (t * (1 - t) ** 2 * directions[0]) * length
    slope_b = (-(t**2) * (1 - t) * directions[1]) * length
    for trial in _tangent_curves(base, slope_a, slope_b, directions, radius, span[1:-1], sample_spacing):
        if np.min(_radii(trial)) < 0.99 * radius:
            continue
        if raised:
            trial = _preserve_bridge(trial, heights, projector, maximum_height)
            if trial is None:
                continue
        else:
            trial[1:-1] = projector.project(trial[1:-1])[0]
        if np.min(_radii(np.vstack((span[0], trial, span[-1])))) < 0.99 * radius:
            continue
        # Compare against the original *routed* source, clipping its
        # first and last intervals at the two retained endpoints.
        step = min(corridor / 5, sample_spacing)
        bound = max(_directed_bound(trial, source, step), _directed_bound(source, trial, step))
        if bound <= corridor:
            return trial, bound
    return None


def relieve_folds(
    points,
    geometry,
    folds,
    part,
    projector,
    *,
    radius,
    budget,
    sample_spacing,
    maximum_height=0.0,
    raised_budget=None,
    joint_budget=0.0,
):
    """Round only folds with tangent-matched replacements in the requested corridor."""
    if not len(folds) or not part.wire_path.segment_owners or radius <= 0:
        return points, geometry, 0, 0.0
    raw = part.wire_path.v
    raw_ids = nearest_source_segment_indices(raw, points)
    owners = part.wire_path.segment_owners
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    edits = []
    attempted = set()
    worst_bound = 0.0
    for fold in folds:
        if fold >= len(raw_ids) or (edits and fold <= edits[-1][1]):
            continue
        owner = owners[raw_ids[fold]]
        kind = geometry[fold]
        raised = kind == WirePathGeometry.OFFSET_SURFACE
        join = kind == WirePathGeometry.SURFACE and joint_budget > 0 and _spiral_attachment(owner, owner.group_index)
        corridor = raised_budget if raised and raised_budget is not None else budget
        if corridor <= 0:
            continue
        if not (raised or join) and (owner.kind != "contour_loop" or kind != WirePathGeometry.SURFACE):
            continue
        station = 0.5 * (arc[fold] + arc[fold + 1])
        left_limit, right_limit = fold, fold + 1

        def admitted(index):
            target = owners[raw_ids[index]]
            return geometry[index] == kind and (
                raised or target == owner or (join and _spiral_attachment(target, owner.group_index))
            )

        while left_limit > 1 and admitted(left_limit - 1):
            left_limit -= 1
        while right_limit < len(points) - 2 and admitted(right_limit):
            right_limit += 1
        spans = []
        for scale in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
            for left, right in ((scale, scale), (scale, 0.7 * scale), (0.7 * scale, scale)):
                first, last = np.searchsorted(arc, (station - left * radius, station + right * radius))
                span = max(left_limit, first - 1), min(right_limit, last)
                if span not in spans:
                    spans.append(span)
        for first, last in spans:
            if first >= fold or last <= fold + 1 or (edits and first <= edits[-1][1]):
                continue
            if (first, last) in attempted:
                continue
            source_indices = raw_ids[first:last]
            if np.any(np.diff(source_indices) < 0):
                continue
            # Check all nominal intervals too, including those skipped by the
            # earlier sweep sampling. Never bypass a join, cut or raised span.
            lo, hi = int(source_indices[0]), int(source_indices[-1])
            source_owners = owners[lo : hi + 1]
            if any(label != kind for label in part.wire_path.segment_geometry[lo : hi + 1]):
                continue
            source_corridor = corridor
            if join and any(item.kind == "contour_group" for item in source_owners):
                if not _same_turn_attachment(source_owners):
                    continue
                source_corridor = min(corridor, joint_budget)
            elif not (_raised_handoff(source_owners) if raised else all(item == owner for item in source_owners)):
                continue
            attempted.add((first, last))
            start, finish = points[first], points[last]
            source = _clip_source(raw[max(0, lo - 1) : hi + 3], start, finish)
            if len(source) < 3:
                continue
            fit = _fit_span(
                points[first - 1 : last + 2],
                source,
                projector,
                length=arc[last] - arc[first],
                radius=radius,
                corridor=source_corridor,
                sample_spacing=sample_spacing,
                raised=raised,
                maximum_height=maximum_height,
            )
            if fit is not None:
                trial, bound = fit
                worst_bound = max(worst_bound, bound)
                edits.append((first, last, trial))
                break
    if edits:
        result, labels, cursor = [], [], 0
        for first, last, replacement in edits:
            result.extend((points[cursor:first], replacement[:-1]))
            labels.extend(geometry[cursor:first])
            labels.extend([geometry[first]] * (len(replacement) - 1))
            cursor = last
        points = np.concatenate(result + [points[cursor:]])
        geometry = labels + list(geometry[cursor:])
    return points, geometry, len(edits), worst_bound


def repair_reversed_sweep(part, points, geometry, vertices, section, config, *, closed, mounting_offset, radius):
    from .swept_clearance import required_copper_gap
    from .layout import _within_group_cut_width
    from .manufacturing import (
        _SurfaceProjector,
        _analytic_cylinder_surface_constraint,
        _build_sweep_rings,
        _sweep_arc_step,
        _transfer_path_segment_geometry,
    )

    clearance = required_copper_gap(config)
    folds = profile_fold_stations(vertices, len(section), closed=closed, bend_radius=radius, gap=clearance)
    if not len(folds) or radius <= 0:
        return None
    projector = _analytic_cylinder_surface_constraint(part, config)
    if projector is None:
        projector = _PlanarInterior.for_mesh(part.mesh, _SurfaceProjector(part.mesh))
    spacing = radius * _sweep_arc_step(config, radius)
    if config.sweep_max_segment_length > 0:
        spacing = min(spacing, config.sweep_max_segment_length)
    applied, worst = 0, 0.0
    rings = None
    raised_budget = config.sweep_corner_cluster_max_source_deviation_m
    joint_budget = 0.0
    if part.normal_shift_provenance.get("method") == "arclength_crossing_layers":
        limit = min(abs(config.normal_shift_length), config.interconnection_cut_width / 2)
        raised_budget = min(raised_budget, limit) if raised_budget > 0 else limit
        joint_budget = 0.5 * _within_group_cut_width(config)
    # Adjoining admissible windows can overlap. Re-evaluate their physical
    # rings after each set of disjoint edits, always against the same source.
    for _ in range(4):
        trial, labels, count, bound = relieve_folds(
            points,
            geometry,
            folds,
            part,
            projector,
            radius=radius,
            budget=config.sweep_corner_cluster_max_source_deviation_m,
            sample_spacing=spacing,
            maximum_height=abs(config.normal_shift_length),
            raised_budget=raised_budget,
            joint_budget=joint_budget,
        )
        if not count:
            break
        candidate = _build_sweep_rings(
            part, trial, labels, section, config, closed=closed, mounting_offset=mounting_offset
        )
        remaining = profile_fold_stations(candidate[4], len(section), closed=closed, bend_radius=radius, gap=clearance)
        if len(remaining) >= len(folds):
            break
        points = candidate[0]
        geometry = _transfer_path_segment_geometry(trial, labels, points, source_closed=closed, target_closed=closed)
        rings = candidate
        folds = remaining
        worst = max(worst, bound)
        applied += count
        if not len(folds):
            break
    if rings is None:
        return None
    return (
        rings,
        geometry,
        {
            "sweep_fold_relief_count": applied,
            "sweep_fold_relief_max_source_bound_m": float(worst),
            "sweep_fold_relief_unresolved_stations": len(folds),
        },
    )
