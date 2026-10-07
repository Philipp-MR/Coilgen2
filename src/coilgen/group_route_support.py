"""Conservative whole-body support certificates for joint routing proposals.

Convex planar domains use exact halfspaces. Centered mounting also supports
oriented triangular surfaces through bounded whole-chord support certificates
and a conservative whole-body tube bound against physical boundary edges.
Intentional centered embedding is not physical substrate-exterior clearance.
"""

from dataclasses import dataclass, field

import numpy as np

from .connector_bridge_validation import _SpatialSupport, _segment_hits_triangle


@dataclass(frozen=True)
class GroupRouteSupport:
    accepted: bool
    reasons: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)
    _domain: tuple | None = field(default=None, repr=False, compare=False)


def _planar_convex_domain(mesh, tolerance, maximum_faces):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or len(faces) > maximum_faces
        or not np.all(np.isfinite(vertices))
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        return None, "invalid_support_mesh_or_budget"
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas2 = np.linalg.norm(cross, axis=1)
    if np.any(areas2 <= np.finfo(float).tiny):
        return None, "degenerate_support_face"
    normal = cross[int(np.argmax(areas2))] / float(np.max(areas2))
    origin = triangles[0, 0]
    if np.max(np.abs((vertices - origin) @ normal)) > tolerance:
        return None, "unsupported_nonplanar_support"
    if np.any((cross @ normal) / areas2 < 1.0 - 1e-10):
        return None, "inconsistent_support_orientation"

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    signs = np.where(edges[:, 0] == ordered[:, 0], 1, -1)
    signed_counts = np.bincount(inverse, weights=signs, minlength=len(unique))
    if np.any(counts > 2) or np.any((counts == 2) & (signed_counts != 0)):
        return None, "nonmanifold_or_overlapping_support"
    boundary = edges[counts[inverse] == 1]
    if len(boundary) < 3:
        return None, "unsupported_support_boundary"
    starts, start_counts = np.unique(boundary[:, 0], return_counts=True)
    ends, end_counts = np.unique(boundary[:, 1], return_counts=True)
    if not np.array_equal(starts, ends) or np.any(start_counts != 1) or np.any(end_counts != 1):
        return None, "unsupported_support_boundary"
    following = dict(map(tuple, boundary))
    indices, visited, current = [], set(), int(boundary[0, 0])
    while current not in visited:
        indices.append(current)
        visited.add(current)
        current = int(following[current])
    if current != indices[0] or len(indices) != len(boundary):
        return None, "unsupported_holed_or_disconnected_support"
    polygon = vertices[indices]
    directions = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= tolerance):
        return None, "degenerate_support_boundary"
    unit_edges = directions / lengths[:, None]
    turns = np.cross(unit_edges, np.roll(unit_edges, -1, axis=0)) @ normal
    cosines = np.einsum("ij,ij->i", unit_edges, np.roll(unit_edges, -1, axis=0))
    # A consistently left-turning boundary with rotation index one is convex.
    # Checking the total turn rejects star boundaries that wind more than once.
    if np.any(turns < -1e-10) or abs(float(np.arctan2(turns, cosines).sum()) - 2 * np.pi) > 1e-7:
        return None, "unsupported_nonconvex_support"
    area = 0.5 * float(np.sum(np.cross(polygon - origin, np.roll(polygon, -1, axis=0) - origin), axis=0) @ normal)
    if area <= 0 or abs(float(areas2.sum()) / 2 - area) > max(tolerance * lengths.sum(), area * 1e-10):
        return None, "uncertified_support_coverage"
    # Drop exactly collinear boundary samples for inexpensive halfspace tests.
    keep = np.abs(np.roll(turns, 1)) > 1e-14
    polygon = polygon[keep]
    if len(polygon) > 2048:
        return None, "support_boundary_budget_exceeded"
    directions = np.roll(polygon, -1, axis=0) - polygon
    inward = np.cross(normal, directions / np.linalg.norm(directions, axis=1)[:, None])
    return (origin, normal, polygon, inward), ""


def _domain_margin(points, polygon, inward):
    minimum = np.inf
    for offset in range(0, len(points), 256):
        distances = np.einsum("pij,ij->pi", points[offset : offset + 256, None] - polygon, inward)
        minimum = min(minimum, float(np.min(distances)))
    return minimum


def _oriented_surface_reason(mesh, maximum_faces):
    """Require a finite coherently oriented edge-manifold support."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or len(faces) > maximum_faces
        or not np.all(np.isfinite(vertices))
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        return "invalid_support_mesh_or_budget"
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(areas <= np.finfo(float).tiny):
        return "degenerate_support_face"
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    balance = np.bincount(inverse, weights=np.where(edges[:, 0] == ordered[:, 0], 1, -1), minlength=len(unique))
    if np.any(counts > 2) or np.any((counts == 2) & (balance != 0)):
        return "nonmanifold_or_inconsistently_oriented_support"
    boundary = edges[counts[inverse] == 1]
    if len(boundary):
        starts, a = np.unique(boundary[:, 0], return_counts=True)
        ends, b = np.unique(boundary[:, 1], return_counts=True)
        if not np.array_equal(starts, ends) or np.any(a != 1) or np.any(b != 1):
            return "nonmanifold_support_boundary"
    return ""


def _loft_topology_matches(faces, ring_count, section_count, closed):
    """Every triangle must lie in one adjacent-ring tube or an end cap."""
    ring_vertices = ring_count * section_count
    rings = faces // section_count
    caps = faces >= ring_vertices
    for index in np.flatnonzero(np.any(caps, axis=1)):
        if closed:
            return False
        cap_vertices = faces[index][caps[index]]
        if len(cap_vertices) != 1 or cap_vertices[0] not in {ring_vertices, ring_vertices + 1}:
            return False
        target = 0 if cap_vertices[0] == ring_vertices else ring_count - 1
        if np.any(rings[index][~caps[index]] != target):
            return False
        rings[index][caps[index]] = target
    minimum, maximum = rings.min(axis=1), rings.max(axis=1)
    adjacent = maximum - minimum <= 1
    if closed:
        adjacent |= np.all((rings == 0) | (rings == ring_count - 1), axis=1)
    return bool(np.all(adjacent))


def _analytic_surface_chord_allowances(part, config, origins, tolerance, diagnostics):
    """Bound radial tessellation deficit only for declared analytic SURFACE chords.

    A centered body intentionally embeds its profile. Its nominal analytic
    cylinder realization is represented by straight chords, whose exact radial
    deficit is permitted only within the already configured chord-error bound.
    Raw FREE chords and raw source vertices receive no allowance.
    """
    from .manufacturing import _analytic_cylinder_surface_constraint
    from .models import WirePathGeometry

    surface = _analytic_cylinder_surface_constraint(part, config)
    if surface is None or float(config.normal_shift_length) < 0:
        return None
    error_limit = float(config.sweep_max_chord_error)
    if not np.isfinite(error_limit) or error_limit <= 0:
        return None
    offsets = np.asarray(part.mesh.vertices) - surface.center
    axial = offsets @ surface.axis
    radial = offsets - axial[:, None] * surface.axis
    radial_error = float(np.max(np.abs(np.linalg.norm(radial, axis=1) - surface.radius)))
    if radial_error > tolerance:
        return None
    triangles = np.asarray(part.mesh.vertices)[np.asarray(part.mesh.faces)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths <= np.finfo(float).tiny):
        return None
    normals = cross / lengths[:, None]
    if np.any(np.abs(normals @ surface.axis) > 1e-10):
        return None
    face_radial = radial[np.asarray(part.mesh.faces)].mean(axis=1)
    if np.any(np.einsum("ij,ij->i", normals, face_radial) <= 0):
        return None
    labels = part.fabrication_path.segment_geometry
    points = np.vstack((origins, origins[0])) if part.fabrication_path.closed else origins
    if len(labels) != len(points) - 1:
        return None
    relative = points - surface.center
    z = relative @ surface.axis
    xy = relative - z[:, None] * surface.axis
    radii = np.linalg.norm(xy, axis=1)
    nominal = np.asarray([label == WirePathGeometry.SURFACE for label in labels])
    endpoints_valid = (
        (np.abs(radii[:-1] - surface.radius) <= tolerance)
        & (np.abs(radii[1:] - surface.radius) <= tolerance)
        & surface.within_domain(points[:-1], tolerance)
        & surface.within_domain(points[1:], tolerance)
    )
    difference = xy[1:] - xy[:-1]
    squared_length = np.einsum("ij,ij->i", difference, difference)
    parameter = np.divide(
        -np.einsum("ij,ij->i", xy[:-1], difference),
        squared_length,
        out=np.zeros_like(squared_length),
        where=squared_length > 0,
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = xy[:-1] + parameter[:, None] * difference
    squared_minimum = np.einsum("ij,ij->i", closest, closest)
    # Difference of squares avoids cancellation for nanometre chord deficits.
    deficit = np.maximum(0.0, (surface.radius**2 - squared_minimum) / (surface.radius + np.sqrt(squared_minimum)))
    eligible = nominal & endpoints_valid & (deficit <= error_limit)
    allowances = np.where(eligible, deficit, 0.0)
    diagnostics.update(
        analytic_reference_tessellation_certificate="typed_nominal_surface_chords_with_exact_cylinder_radial_deficit",
        analytic_reference_radius_m=surface.radius,
        analytic_support_vertex_radial_error_m=radial_error,
        analytic_reference_chord_error_limit_m=error_limit,
        analytic_reference_surface_chord_count=int(np.count_nonzero(eligible)),
        analytic_reference_ineligible_surface_chord_count=int(np.count_nonzero(nominal & ~eligible)),
        maximum_analytic_reference_chord_deficit_m=float(allowances.max(initial=0.0)),
        analytic_reference_semantics="bounded_nominal_chord_tessellation_with_intentional_centered_embedding",
        whole_body_exterior_to_support=False,
    )
    return allowances


class _SupportScanBudgetExceeded(RuntimeError):
    """A physical support broad-phase scan did not complete its hard budget."""


class _CenteredSpatialSupport(_SpatialSupport):
    """Keep exact support queries, replacing only the initial chord AABB scan.

    Node and individual leaf-box comparisons are charged before execution.
    The exact inclusive leaf mask and increasing face order preserve the old
    clipped-chord negative-side/intersection predicate and its first hit.
    """

    def __init__(self, *args, maximum_box_work=50_000_000, **kwargs):
        from .geometry_collision import _build_segment_bvh

        super().__init__(*args, **kwargs)
        self.maximum_box_work = maximum_box_work
        self.box_work = 0
        self.attempted_box_work = 0
        self.node_box_work = 0
        self.leaf_box_work = 0
        self._crossing_cache = {}
        self.analytic_tessellation_facet_hit_count = 0
        count, leaf_size = len(self.triangles), 12
        # Reserve a conservative construction bound before allocating the tree.
        depth = int(np.ceil(np.log2(max(1.0, count / leaf_size))))
        reserve = count * (depth + 2)
        if reserve > maximum_box_work:
            self.attempted_box_work = reserve
            raise _SupportScanBudgetExceeded
        self._root = _build_segment_bvh(self.low, self.high, leaf_size)

        def construction_work(node):
            if node.leaf:
                return len(node.indices), len(node.indices)
            first_count, first_work = construction_work(node.left)
            second_count, second_work = construction_work(node.right)
            return first_count + second_count, first_work + second_work + first_count + second_count

        _, bounds_work = construction_work(self._root)
        # Initial face boxes plus face boxes contributing to each node bound.
        self.construction_box_work = count + bounds_work
        assert self.construction_box_work <= reserve
        self._reserve(self.construction_box_work)

    def _reserve(self, count):
        self.attempted_box_work = max(self.attempted_box_work, self.box_work + count)
        if self.box_work + count > self.maximum_box_work:
            raise _SupportScanBudgetExceeded
        self.box_work += count

    def _candidate_faces(self, first, last):
        low, high = np.minimum(first, last), np.maximum(first, last)
        pending, possible = [self._root], []
        while pending:
            node = pending.pop()
            self._reserve(1)
            self.node_box_work += 1
            if not (np.all(node.minimum <= high) and np.all(node.maximum >= low)):
                continue
            if not node.leaf:
                pending.extend((node.right, node.left))
                continue
            indices = node.indices
            self._reserve(len(indices))
            self.leaf_box_work += len(indices)
            mask = np.all(self.low[indices] <= high, axis=1) & np.all(self.high[indices] >= low, axis=1)
            possible.extend(indices[mask])
        return np.asarray(sorted(possible), dtype=int)

    def crosses_surface(self, segments, analytic_chord_allowances=None):
        import hashlib

        points = np.ascontiguousarray(segments)
        allowances = (
            np.zeros(len(segments)) if analytic_chord_allowances is None else np.asarray(analytic_chord_allowances)
        )
        if allowances.shape != (len(segments),) or np.any(~np.isfinite(allowances)) or np.any(allowances < 0):
            raise ValueError("Invalid analytic reference chord allowances.")
        key = (
            points.shape,
            points.dtype.str,
            hashlib.sha256(points.tobytes()).digest(),
            hashlib.sha256(allowances.tobytes()).digest(),
        )
        if key in self._crossing_cache:
            return self._crossing_cache[key]
        for (start, end), allowance in zip(segments, allowances, strict=True):
            length = float(np.linalg.norm(end - start))
            if length <= 2 * self.tolerance:
                continue
            direction = (end - start) / length
            first, last = start + self.tolerance * direction, end - self.tolerance * direction
            for index in self._candidate_faces(first, last):
                triangle = self.triangles[index]
                signed = (np.vstack((start, end)) - triangle[0]) @ self.projector.face_normals[index] * self.sign
                if float(signed.min()) >= -self.tolerance - allowance:
                    continue
                if _segment_hits_triangle(first, last, triangle):
                    if allowance > 0:
                        # The factory proved this entire typed nominal chord
                        # within its exact analytic-cylinder deficit bound.
                        # Its polyhedral facet hit is representation error;
                        # actual distance/lift/boundary predicates still run.
                        self.analytic_tessellation_facet_hit_count += 1
                        continue
                    self._crossing_cache[key] = True
                    return True
        self._crossing_cache[key] = False
        return False

    def validate(self, segments, *, maximum_evaluations=8192, analytic_chord_allowances=None):
        if analytic_chord_allowances is None:
            return super().validate(segments, maximum_evaluations=maximum_evaluations)
        if self.crosses_surface(segments, analytic_chord_allowances):
            return ["support_surface_intersection"]
        from .geometry_kernels import _point_triangle_sq

        cache = {}

        def sample(point):
            key = tuple(point)
            if key not in cache:
                cache[key] = self.sample(point)
            return cache[key]

        pending = [
            (start, end, allowance) for (start, end), allowance in zip(segments, analytic_chord_allowances, strict=True)
        ]
        while pending:
            if self.evaluations + 3 > maximum_evaluations:
                return ["support_certificate_budget_exceeded"]
            start, end, allowance = pending.pop()
            midpoint, length = (start + end) / 2, float(np.linalg.norm(end - start))
            values = [sample(point) for point in (start, midpoint, end)]
            if any(
                value[0] > self.height + self.tolerance or value[2] < -self.tolerance - allowance for value in values
            ):
                return ["lift_limit_violation"]
            if any(value[0] > self.tolerance and value[1] <= value[0] + self.tolerance for value in values):
                return ["support_domain_violation"]
            middle = values[1]
            height_upper = middle[0] + length / 2
            for face in {value[3] for value in values}:
                triangle = self.triangles[face]
                triangle_upper = np.sqrt(max(_point_triangle_sq(start, triangle), _point_triangle_sq(end, triangle)))
                height_upper = min(height_upper, triangle_upper)
            boundary_safe = middle[1] - middle[0] > length + self.tolerance
            if height_upper <= self.height + self.tolerance and boundary_safe:
                continue
            if length <= self.tolerance:
                return ["support_domain_uncertified"]
            pending.extend(((start, midpoint, allowance), (midpoint, end, allowance)))
        return []


def _general_centered_certificate(
    part, config, origins, envelope, tolerance, maximum_evaluations, diagnostics, *, analytic_surface_chords=None
):
    from .manufacturing import _SurfaceProjector
    from .models import WirePathGeometry

    maximum_lift = float(config.normal_shift_length)
    projector = _SurfaceProjector(part.mesh)
    # A cached normal array may predate a mesh edit. Physical orientation comes
    # from the coherently oriented triangles checked by the domain preflight.
    edges = projector.triangles[:, 1:] - projector.triangles[:, :1]
    face_normals = np.cross(edges[:, 0], edges[:, 1])
    projector.face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, None]
    work_limit = 50_000_000
    try:
        checker = _CenteredSpatialSupport(
            projector, abs(maximum_lift), -1.0 if maximum_lift < 0 else 1.0, tolerance, maximum_box_work=work_limit
        )
    except _SupportScanBudgetExceeded:
        return "support_face_edge_work_budget_exceeded"
    cache, query = {}, checker.sample

    def sample(point):
        key = tuple(point)
        if key not in cache:
            cache[key] = query(point)
        return cache[key]

    checker.sample = sample
    points = np.vstack((origins, origins[0])) if part.fabrication_path.closed else origins
    final_segments = np.stack((points[:-1], points[1:]), axis=1)
    source = part.wire_path
    free = np.asarray([value != WirePathGeometry.SURFACE for value in source.segment_geometry])
    raw_segments = np.stack((source.v[:-1], source.v[1:]), axis=1)[free]
    surface_indices = np.flatnonzero(~free)
    surface_vertices = source.v[np.unique(np.concatenate((surface_indices, surface_indices + 1)))]
    # Source surface chords may be refined/projected across mesh creases, but
    # their original vertices must still obey the reference support contract.
    reference_segments = np.concatenate((final_segments, raw_segments, np.repeat(surface_vertices[:, None], 2, axis=1)))
    analytic_allowances = None
    if analytic_surface_chords is not None:
        values = np.asarray(analytic_surface_chords)
        if values.shape != (len(final_segments),):
            return "invalid_analytic_reference_chord_certificate"
        analytic_allowances = np.concatenate((values, np.zeros(len(raw_segments) + len(surface_vertices))))
    # Preflight exactly the old chord-intersection predicate using charged
    # node/leaf AABB scans. All subsequent support query predicates are unchanged.
    try:
        crosses_surface = checker.crosses_surface(reference_segments, analytic_allowances)
    except _SupportScanBudgetExceeded:
        diagnostics.update(
            support_face_edge_work_limit=work_limit,
            support_initial_chord_face_work=checker.box_work,
            support_attempted_initial_chord_face_work=checker.attempted_box_work,
        )
        return "support_face_edge_work_budget_exceeded"
    initial_work = checker.box_work
    query_work = len(checker.triangles) + len(checker.boundaries)
    work_evaluations = max(0, (work_limit - initial_work) // query_work)
    effective_evaluations = min(maximum_evaluations, work_evaluations)
    diagnostics.update(
        certificate_scope="centered_reference_support_and_whole_body_boundary_tube",
        support_face_edge_work_limit=work_limit,
        support_initial_chord_face_work=initial_work,
        support_chord_scan_kind="exact_triangle_aabb_bvh",
        support_bvh_construction_box_work=checker.construction_box_work,
        support_bvh_node_box_work=checker.node_box_work,
        support_bvh_leaf_box_work=checker.leaf_box_work,
        analytic_tessellation_facet_hit_count=checker.analytic_tessellation_facet_hit_count,
        support_query_face_edge_work=query_work,
        effective_support_evaluation_limit=effective_evaluations,
        raw_surface_reference_vertex_count=len(surface_vertices),
    )
    if crosses_surface:
        diagnostics.update(reference_support_reasons=["support_surface_intersection"], reference_support_evaluations=0)
        return "support_surface_intersection"
    if work_evaluations < 3:
        return "support_face_edge_work_budget_exceeded"
    reasons = checker.validate(
        reference_segments, maximum_evaluations=effective_evaluations, analytic_chord_allowances=analytic_allowances
    )
    diagnostics.update(
        certificate_scope="centered_reference_support_and_whole_body_boundary_tube",
        centered_profile_embedding_intended=True,
        whole_body_exterior_to_support=False,
        reference_support_reasons=list(reasons),
        reference_support_evaluations=checker.evaluations,
        maximum_reference_support_distance_m=checker.maximum_distance,
        minimum_sampled_reference_signed_height_m=checker.minimum_height,
        configured_maximum_lift_m=abs(maximum_lift),
        final_reference_chord_count=len(final_segments),
        raw_free_reference_chord_count=len(raw_segments),
    )
    if reasons:
        return reasons[0]
    diagnostics["body_reference_tube_radius_m"] = envelope
    diagnostics["support_boundary_edge_count"] = len(checker.boundaries)
    if not len(checker.boundaries):
        diagnostics["body_footprint_certificate"] = "closed_support_has_no_physical_boundary"
        return ""
    # Both distances are 1-Lipschitz. At any point within radius E of an
    # origin chord, their difference is at least F(midpoint)-length-2E.
    # Positive difference excludes every physical boundary as nearest support.
    pending = list(final_segments)
    minimum_gap = np.inf
    while pending:
        if checker.evaluations + 1 > effective_evaluations:
            return "body_support_footprint_budget_exceeded"
        start, end = pending.pop()
        midpoint = 0.5 * (start + end)
        support_distance, boundary_distance, _, _ = sample(midpoint)
        gap = boundary_distance - support_distance
        minimum_gap = min(minimum_gap, gap)
        length = float(np.linalg.norm(end - start))
        if gap > length + 2 * envelope + tolerance:
            continue
        if gap <= 2 * envelope + tolerance or length <= tolerance:
            diagnostics["minimum_sampled_boundary_support_distance_gap_m"] = minimum_gap
            return "body_support_footprint_uncertified"
        pending.extend(((start, midpoint), (midpoint, end)))
    diagnostics.update(
        body_footprint_certificate="adaptive_boundary_minus_support_distance_exceeds_two_profile_radii",
        minimum_sampled_boundary_support_distance_gap_m=minimum_gap,
        support_evaluation_count=checker.evaluations,
    )
    return ""


def check_group_route_support_domain(part_mesh, config, *, tolerance_m=1e-8, maximum_support_faces=200000):
    """Fast eligibility preflight; this does not certify a candidate body."""
    diagnostics = {"certificate_scope": "support_domain_eligibility"}
    if not np.isfinite(tolerance_m) or tolerance_m <= 0 or maximum_support_faces < 1:
        return GroupRouteSupport(False, ("invalid_support_certificate_limits",), diagnostics)
    mode = str(config.conductor_mounting_mode).strip().lower().replace("-", "_")
    if mode not in {"centered", "surface_outer", "surface_inner"}:
        return GroupRouteSupport(False, ("unsupported_mounting_mode",), diagnostics)
    if not np.isfinite(config.normal_shift_length):
        return GroupRouteSupport(False, ("invalid_lift_limit",), diagnostics)
    domain, reason = _planar_convex_domain(part_mesh, tolerance_m, maximum_support_faces)
    if reason:
        # Planar exterior mounting has a full-body halfspace proof. Curved
        # reference supports are admissible only under centered semantics.
        allowed = {
            "unsupported_nonplanar_support",
            "unsupported_nonconvex_support",
            "unsupported_holed_or_disconnected_support",
            "unsupported_support_boundary",
            "support_boundary_budget_exceeded",
        }
        if mode != "centered" or reason not in allowed:
            return GroupRouteSupport(False, (reason,), diagnostics)
        reason = _oriented_surface_reason(part_mesh, maximum_support_faces)
        if reason:
            return GroupRouteSupport(False, (reason,), diagnostics)
        diagnostics.update(
            support_geometry="oriented_triangular_reference_support", centered_profile_embedding_intended=True
        )
        return GroupRouteSupport(True, (), diagnostics)
    diagnostics.update(support_geometry="oriented_convex_planar_disk", support_normal=domain[1].tolist())
    return GroupRouteSupport(True, (), diagnostics, domain)


def certify_group_route_support(
    part,
    config,
    *,
    tolerance_m: float = 1e-8,
    maximum_support_faces: int = 200000,
    maximum_body_faces: int = 2000000,
    maximum_support_evaluations: int = 200000,
) -> GroupRouteSupport:
    """Certify the final body footprint and declared mounting/lift contract.

    Centered mounting intentionally embeds part of the section in the reference
    support. It is accepted only when the actual body matches its stored section
    frames and both the original and final unmounted reference origins obey the
    configured signed lift. This is not a claim that a centered body is wholly
    outside a physical substrate. Surface-outer/inner bodies additionally obey
    the corresponding exterior plane halfspace, including triangle interiors.

    Convex planar domains have an exact halfspace proof. Other centered supports
    use adaptive whole-chord support checks and a whole-triangle tube proof of
    physical-boundary avoidance. No geometry is mutated or projected for adoption.
    """
    diagnostics = {
        "certificate_scope": "convex_planar_support_mounting_and_body_footprint",
        "maximum_body_faces": maximum_body_faces,
        "maximum_support_faces": maximum_support_faces,
        "maximum_support_evaluations": maximum_support_evaluations,
    }

    def reject(reason):
        return GroupRouteSupport(False, (reason,), diagnostics)

    if (
        not np.isfinite(tolerance_m)
        or tolerance_m <= 0
        or min(maximum_support_faces, maximum_body_faces, maximum_support_evaluations) < 1
    ):
        return reject("invalid_support_certificate_limits")
    eligibility = check_group_route_support_domain(
        part.mesh, config, tolerance_m=tolerance_m, maximum_support_faces=maximum_support_faces
    )
    if not eligibility.accepted:
        return reject(eligibility.reasons[0])
    domain = eligibility._domain
    body = part.swept_body
    if body is None or not len(body.faces) or len(body.faces) > maximum_body_faces:
        return reject("missing_swept_body_or_budget")
    body_points = np.asarray(body.vertices)
    if (
        body_points.ndim != 2
        or body_points.shape[1:] != (3,)
        or not len(body_points)
        or not np.all(np.isfinite(body_points))
        or np.any(body.faces < 0)
        or np.any(body.faces >= len(body_points))
    ):
        return reject("invalid_swept_body")
    if part.wire_path is None or len(part.wire_path.v) < 2 or not np.all(np.isfinite(part.wire_path.v)):
        return reject("missing_reference_path")
    mode = str(config.conductor_mounting_mode).strip().lower().replace("-", "_")
    if mode not in {"centered", "surface_outer", "surface_inner"}:
        return reject("unsupported_mounting_mode")
    maximum_lift = float(config.normal_shift_length)
    if not np.isfinite(maximum_lift):
        return reject("invalid_lift_limit")
    sign = -1.0 if maximum_lift < 0 else 1.0

    # Verify the stored physical trace actually describes this mesh; otherwise
    # centered embedding allowances could conceal arbitrary displaced geometry.
    trace = part.fabrication_path
    if trace is None:
        return reject("missing_final_section_frames")
    from .manufacturing import _mounting_centerline_offset, _polygon_centroid, conductor_cross_section_points

    section = conductor_cross_section_points(config)
    if section.shape != trace.cross_section_m.shape or not np.allclose(
        section, trace.cross_section_m, rtol=0, atol=tolerance_m
    ):
        return reject("section_frame_profile_mismatch")
    centroid = _polygon_centroid(section)
    if (
        not np.allclose(np.linalg.norm(trace.normals, axis=1), 1.0, rtol=0, atol=1e-8)
        or not np.allclose(np.linalg.norm(trace.transverse, axis=1), 1.0, rtol=0, atol=1e-8)
        or np.any(np.abs(np.einsum("ij,ij->i", trace.normals, trace.transverse)) > 1e-8)
    ):
        return reject("invalid_section_frames")
    section_centered = section - centroid
    rings = (
        trace.points_m[:, None]
        + section_centered[None, :, :1] * trace.transverse[:, None]
        + section_centered[None, :, 1:] * trace.normals[:, None]
    )
    expected_vertices = rings.reshape(-1, 3)
    if not trace.closed:
        expected_vertices = np.vstack((expected_vertices, rings[0].mean(axis=0), rings[-1].mean(axis=0)))
    if (
        expected_vertices.shape != body_points.shape
        or np.max(np.linalg.norm(expected_vertices - body_points, axis=1)) > tolerance_m
    ):
        return reject("body_section_frame_mismatch")
    if not _loft_topology_matches(body.faces, len(trace.points_m), len(section), trace.closed):
        return reject("body_section_loft_topology_mismatch")
    mounting_offset = _mounting_centerline_offset(config, section)
    final_origins = trace.points_m - (mounting_offset + centroid[1]) * trace.normals - centroid[0] * trace.transverse
    if domain is None:
        ring_vertex_count = len(final_origins) * len(section)
        actual_ring_offsets = body_points[:ring_vertex_count].reshape(rings.shape) - final_origins[:, None]
        envelope = float(np.linalg.norm(actual_ring_offsets, axis=2).max())
        if not trace.closed:
            envelope = max(
                envelope,
                float(np.linalg.norm(body_points[-2:] - final_origins[[0, -1]], axis=1).max()),
            )
        reason = _general_centered_certificate(
            part,
            config,
            final_origins,
            envelope,
            tolerance_m,
            maximum_support_evaluations,
            diagnostics,
            analytic_surface_chords=_analytic_surface_chord_allowances(
                part, config, final_origins, tolerance_m, diagnostics
            ),
        )
        return reject(reason) if reason else GroupRouteSupport(True, (), diagnostics)
    origin, normal, polygon, inward = domain
    reference = np.vstack((part.wire_path.v, final_origins))
    heights = (reference - origin) @ normal * sign
    diagnostics.update(
        mounting_mode=mode,
        support_normal=normal.tolist(),
        maximum_reference_lift_m=float(heights.max()),
        minimum_reference_signed_height_m=float(heights.min()),
        configured_maximum_lift_m=abs(maximum_lift),
        centered_profile_embedding_intended=mode == "centered",
    )
    if heights.min() < -tolerance_m or heights.max() > abs(maximum_lift) + tolerance_m:
        return reject("reference_support_side_or_lift_violation")
    reference_margin = _domain_margin(reference, polygon, inward)
    body_margin = _domain_margin(body_points, polygon, inward)
    diagnostics.update(reference_domain_margin_m=reference_margin, body_domain_margin_m=body_margin)
    if reference_margin < -tolerance_m:
        return reject("reference_support_domain_violation")
    if body_margin < -tolerance_m:
        return reject("body_support_domain_violation")
    body_heights = (body_points - origin) @ normal
    diagnostics.update(
        minimum_body_signed_height_m=float(body_heights.min()), maximum_body_signed_height_m=float(body_heights.max())
    )
    if mode == "surface_outer" and body_heights.min() < -tolerance_m:
        return reject("body_support_penetration")
    if mode == "surface_inner" and body_heights.max() > tolerance_m:
        return reject("body_support_penetration")
    diagnostics["whole_body_exterior_to_support_plane"] = mode != "centered"
    return GroupRouteSupport(True, (), diagnostics)
