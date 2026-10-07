"""Validate a directed group route on an isolated manufacturing preview.

This deliberately reuses the production shift, cleanup and sweep. Admission
limits bound the size of routes offered to those routines; they are not a hard
runtime or allocation bound inside the existing manufacturing implementation.
The additional exact triangle audit has an explicit broad-phase pair budget.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import pickle
from time import perf_counter

import numpy as np

from .config import CoilGenConfig
from .geometry_collision import _build_segment_bvh
from .geometry_kernels import triangle_pair_distances
from .layout import cleanup_wire_paths, shift_return_paths
from .manufacturing import conductor_cross_section_points, create_swept_coil_meshes
from .models import CoilPart, InterGroupCut, TargetField, WirePath
from .physics import biot_savart_wire_path
from .progress import DesignCancelled


@dataclass(frozen=True)
class GroupRouteValidationLimits:
    maximum_route_points: int = 50000
    maximum_section_points: int = 128
    # A 10 m trace with the standard 64-point round section has about 1.5 M
    # faces. Pair work is streamed and independently bounded below.
    maximum_sweep_faces: int = 2000000
    maximum_triangle_pairs: int = 32000000
    maximum_triangle_broadphase_pairs: int = 128000000
    maximum_target_points: int = 10000

    def __post_init__(self):
        if any(not isinstance(value, int) or value < 1 for value in vars(self).values()):
            raise ValueError("Group route validation limits must be positive integers.")


@dataclass(frozen=True)
class GroupRouteBaseline:
    """Content-keyed baseline, reusable only for identical source/config/ROI."""

    fingerprint: str
    route_field_per_amp: np.ndarray
    physical_field_per_amp: np.ndarray | None
    diagnostics: dict
    reasons: tuple[str, ...]


@dataclass
class GroupRouteValidation:
    accepted: bool
    reasons: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)
    prepared_part: CoilPart | None = None
    baseline: GroupRouteBaseline | None = None
    # Accepted manufactured part alone, on the requested target grid at 1 A.
    # Other-part background fields and operating-current scaling are excluded.
    physical_field_per_amp: np.ndarray | None = None


_ROUTE_STATE = (
    "contour_lines",
    "groups",
    "loop_groups",
    "loop_in_loop_matrix",
    "group_levels",
    "level_positions",
    "group_centers_v",
    "connected_groups",
    "opening_cuts_among_groups",
    "wire_path",
    "topology_diagnostics",
    "route_diagnostics",
    "normal_shift_diagnostics",
    "normal_shift_provenance",
    "shift_array",
    "points_to_shift",
    "wire_self_contacts",
)
_PHYSICAL_STATE = ("fabrication_path", "swept_body", "swept_bodies", "manufacturing_diagnostics")


def _fingerprint(part, config, targets, limits):
    # Content hashing, rather than identities/shapes, catches in-place edits.
    payload = (
        config,
        limits,
        targets,
        part.contour_step,
        part.mesh,
        tuple(getattr(part, name) for name in _ROUTE_STATE),
        tuple(getattr(part, name) for name in _PHYSICAL_STATE),
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def _isolated_part(part):
    result = copy(part)
    for name in _ROUTE_STATE:
        setattr(result, name, deepcopy(getattr(part, name)))
    # Share the large support arrays through read-only views, with a separate
    # mesh object and metadata so cached normals/metadata cannot alter the live part.
    result.mesh = copy(part.mesh)
    for name, value in vars(part.mesh).items():
        if isinstance(value, np.ndarray):
            value = value.view()
            value.flags.writeable = False
        else:
            value = deepcopy(value)
        setattr(result.mesh, name, value)
    result.swept_body = None
    result.swept_bodies = []
    result.fabrication_path = None
    result.manufacturing_diagnostics = {}
    result.field_by_layout = None
    return result


def _preview_config(config):
    return replace(
        config, skip_sweep=False, contour_only_sweep_mode=False, fail_on_route_quality_issues=False, progress_flag=False
    )


def _field(points, targets):
    return biot_savart_wire_path(points, targets, worker_count=1, max_chunk_memory_mb=32.0)


def _readonly(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


def _target_points(target_points_m, limits):
    points = np.asarray(target_points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("Target points must be a nonempty finite (n, 3) array in metres.")
    if len(points) > limits.maximum_target_points:
        raise ValueError("target_point_budget_exceeded")
    return points


def _admission(part, config, limits):
    if part.wire_path is None or len(part.wire_path.v) > limits.maximum_route_points:
        return "route_point_budget_exceeded"
    if len(conductor_cross_section_points(config)) > limits.maximum_section_points:
        return "section_point_budget_exceeded"
    return ""


def _incident_triangle_overlaps(first, second, shared, tolerance):
    """Detect intersection beyond the vertex/edge legitimately shared by a pair.

    For triangles sharing one vertex, their intersection lies on a common ray
    from that vertex (or in overlapping planar wedges). Shared-edge triangles
    can overlap beyond the edge only when coplanar on the same side of it.
    """
    first_shared, second_shared = shared.any(axis=2), shared.any(axis=1)
    count = first_shared.sum(axis=1)
    overlaps = count == 3
    indices = np.flatnonzero(count == 2)
    if len(indices):
        a, b = first[indices], second[indices]
        edge_points = a[first_shared[indices]].reshape(-1, 2, 3)
        origin, edge = edge_points[:, 0], edge_points[:, 1] - edge_points[:, 0]
        first_outer = a[~first_shared[indices]] - origin
        second_outer = b[~second_shared[indices]] - origin
        first_normal = np.cross(edge, first_outer)
        second_normal = np.cross(edge, second_outer)
        coplanar = np.abs(np.einsum("ij,ij->i", first_normal, second_outer)) <= (
            tolerance * np.linalg.norm(first_normal, axis=1)
        )
        overlaps[indices] = coplanar & (np.einsum("ij,ij->i", first_normal, second_normal) > 0)
    indices = np.flatnonzero(count == 1)
    if not len(indices):
        return overlaps
    a, b = first[indices], second[indices]
    origin = a[first_shared[indices]]
    a_edges = a[~first_shared[indices]].reshape(-1, 2, 3) - origin[:, None]
    b_edges = b[~second_shared[indices]].reshape(-1, 2, 3) - origin[:, None]
    a_normal, b_normal = np.cross(a_edges[:, 0], a_edges[:, 1]), np.cross(b_edges[:, 0], b_edges[:, 1])

    def ray_extent(edges, direction):
        normal = np.cross(edges[:, 0], edges[:, 1])
        denominator = np.einsum("ij,ij->i", normal, normal)
        alpha = np.einsum("ij,ij->i", np.cross(direction, edges[:, 1]), normal) / denominator
        beta = np.einsum("ij,ij->i", np.cross(edges[:, 0], direction), normal) / denominator
        rounding = 32 * np.finfo(float).eps * np.maximum(np.abs(alpha), np.abs(beta))
        inside = (alpha >= -rounding) & (beta >= -rounding) & (alpha + beta > 0)
        return np.divide(1.0, alpha + beta, out=np.zeros(len(edges)), where=inside)

    direction = np.cross(a_normal, b_normal)
    direction_length = np.linalg.norm(direction, axis=1)
    unit = np.divide(
        direction, direction_length[:, None], out=np.zeros_like(direction), where=direction_length[:, None] > 0
    )
    coplanar = np.max(np.abs(np.einsum("ij,ikj->ik", a_normal, b_edges)), axis=1) <= (
        tolerance * np.linalg.norm(a_normal, axis=1)
    )
    hit = np.zeros(len(indices), dtype=bool)
    for sign in (1.0, -1.0):
        hit |= ~coplanar & (np.minimum(ray_extent(a_edges, sign * unit), ray_extent(b_edges, sign * unit)) > tolerance)
    # Overlapping convex planar wedges contain an edge ray of either triangle.
    for edges in (a_edges, b_edges):
        for edge in range(2):
            ray = edges[:, edge] / np.linalg.norm(edges[:, edge], axis=1)[:, None]
            hit |= coplanar & (np.minimum(ray_extent(a_edges, ray), ray_extent(b_edges, ray)) > tolerance)
    overlaps[indices] = hit
    return overlaps


class _TriangleAuditBudgetExceeded(RuntimeError):
    pass


def _triangle_pair_batches(minima, maxima, tolerance, limits, diagnostics, progress_callback=None):
    """Visit every possible contact once without retaining the whole pair set.

    Raw BVH leaf products and exact-AABB near pairs have separate work limits.
    At most twelve by twelve raw pairs are materialized at one leaf product;
    exact predicates receive batches of at most 4096 pairs. Shared-simplex
    contacts remain in the stream and receive the independent overlap test.
    """
    root = _build_segment_bvh(minima, maxima)
    stack, buffered, buffered_count = [(root, root)], [], 0
    raw_count = near_count = 0
    last_report = perf_counter()
    while stack:
        if progress_callback is not None and perf_counter() - last_report >= 0.5:
            progress_callback()
            last_report = perf_counter()
        first, second = stack.pop()
        separation = np.maximum(0.0, np.maximum(first.minimum - second.maximum, second.minimum - first.maximum))
        if float(separation @ separation) > tolerance**2:
            continue
        if first is second:
            if not first.leaf:
                stack.extend(((first.right, first.right), (first.left, first.right), (first.left, first.left)))
                continue
            local_first, local_second = np.triu_indices(len(first.indices), 1)
            pairs = np.column_stack((first.indices[local_first], first.indices[local_second]))
        elif first.leaf and second.leaf:
            pairs = np.column_stack(
                (np.repeat(first.indices, len(second.indices)), np.tile(second.indices, len(first.indices)))
            )
        else:
            first_size = len(first.indices) if first.leaf else float(np.sum((first.maximum - first.minimum) ** 2))
            second_size = len(second.indices) if second.leaf else float(np.sum((second.maximum - second.minimum) ** 2))
            if not first.leaf and (second.leaf or first_size >= second_size):
                stack.extend(((first.right, second), (first.left, second)))
            else:
                stack.extend(((first, second.right), (first, second.left)))
            continue
        raw_count += len(pairs)
        diagnostics["exact_collision_broadphase_pair_count"] = raw_count
        if raw_count > limits.maximum_triangle_broadphase_pairs:
            raise _TriangleAuditBudgetExceeded("triangle_broadphase_pair_budget_exceeded")
        if not len(pairs):
            continue
        pairs.sort(axis=1)
        first_indices, second_indices = pairs.T
        separation = np.maximum(
            0.0,
            np.maximum(minima[first_indices] - maxima[second_indices], minima[second_indices] - maxima[first_indices]),
        )
        pairs = pairs[np.einsum("ij,ij->i", separation, separation) <= tolerance**2]
        near_count += len(pairs)
        diagnostics["exact_collision_near_pair_count"] = near_count
        if near_count > limits.maximum_triangle_pairs:
            raise _TriangleAuditBudgetExceeded("triangle_pair_budget_exceeded")
        if not len(pairs):
            continue
        buffered.append(pairs)
        buffered_count += len(pairs)
        if buffered_count >= 4096:
            combined = np.concatenate(buffered)
            yield combined[:4096]
            remainder = combined[4096:]
            buffered = [remainder] if len(remainder) else []
            buffered_count = len(remainder)
    if buffered_count:
        yield np.concatenate(buffered)


def _exact_collision_audit(mesh, limits, progress_callback=None):
    """All triangle pairs, excluding only their legitimate shared simplex."""
    if len(mesh.faces) > limits.maximum_sweep_faces:
        return {"exact_collision_completed": False}, "sweep_face_budget_exceeded"
    if (
        not len(mesh.faces)
        or not np.all(np.isfinite(mesh.vertices))
        or np.any(mesh.faces < 0)
        or np.any(mesh.faces >= len(mesh.vertices))
    ):
        return {"exact_collision_completed": False}, "invalid_sweep_mesh"
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(normals, axis=1) == 0):
        return {"exact_collision_completed": False}, "degenerate_sweep_triangle"
    tolerance = max(float(np.max(np.ptp(mesh.vertices, axis=0))), 1.0) * 1e-12
    minima, maxima = triangles.min(axis=1), triangles.max(axis=1)
    diagnostics = {
        "exact_collision_completed": False,
        "exact_collision_streamed": True,
        "exact_collision_batch_pair_limit": 4096,
        "exact_collision_face_limit": limits.maximum_sweep_faces,
        "exact_collision_near_pair_limit": limits.maximum_triangle_pairs,
        "exact_collision_broadphase_pair_limit": limits.maximum_triangle_broadphase_pairs,
        "exact_collision_broadphase_pair_count": 0,
        "exact_collision_near_pair_count": 0,
    }
    count = 0
    tested = 0
    incident_tested = incident_count = 0

    def report(status=None, message=""):
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "exact body collision audit",
                    "completed": tested + incident_tested,
                    "total": None,
                    "status": status,
                    "message": message
                    or (
                        f"broad-phase pairs={diagnostics['exact_collision_broadphase_pair_count']}; "
                        f"nearby pairs={diagnostics['exact_collision_near_pair_count']}"
                    ),
                }
            )

    report(message=f"building triangle search tree; faces={len(mesh.faces)}")
    try:
        for batch in _triangle_pair_batches(minima, maxima, tolerance, limits, diagnostics, report):
            first, second = batch.T
            shared = mesh.faces[first, :, None] == mesh.faces[second, None, :]
            incident = np.any(shared, axis=(1, 2))
            if np.any(incident):
                incident_tested += int(incident.sum())
                incident_count += int(
                    _incident_triangle_overlaps(
                        triangles[first[incident]], triangles[second[incident]], shared[incident], tolerance
                    ).sum()
                )
            first, second = first[~incident], second[~incident]
            if not len(first):
                report()
                continue
            result = triangle_pair_distances(triangles[first], triangles[second])
            distances = result.values_m
            diagnostics["exact_collision_distance_backend"] = result.backend
            if not np.all(np.isfinite(distances)):
                report("stopped", "nonfinite triangle distance")
                return diagnostics, "nonfinite_triangle_distance"
            count += int(np.count_nonzero(distances <= tolerance))
            tested += len(first)
            report()
    except _TriangleAuditBudgetExceeded as error:
        report("stopped", str(error))
        return diagnostics, str(error)
    diagnostics.update(
        {
            "exact_collision_completed": True,
            "exact_collision_pair_count": tested,
            "exact_incident_pair_count": incident_tested,
            "exact_incident_overlap_count": incident_count,
            "exact_intersection_count": count + incident_count,
            "exact_collision_tolerance_m": tolerance,
        }
    )
    report("finished")
    return diagnostics, ("swept_mesh_self_intersection" if count + incident_count else "")


def _sweep_preview(part, config, limits, progress_callback=None):
    reason = _admission(part, config, limits)
    if reason:
        return {}, (reason,)
    if progress_callback is not None:
        progress_callback({"phase": "manufacturing body preview", "completed": 0, "total": 1})
    create_swept_coil_meshes([part], _preview_config(config))
    if progress_callback is not None:
        progress_callback({"phase": "manufacturing body preview", "completed": 1, "total": 1, "status": "finished"})
    return _check_swept_geometry(part, limits, progress_callback)


def _check_swept_geometry(part, limits, progress_callback=None):
    """Audit existing preview geometry without assuming its diagnostics passed."""
    diagnostics = dict(part.manufacturing_diagnostics)
    reasons = tuple(filter(None, str(diagnostics.get("manufacturing_failure_reasons", "")).split(";")))
    if not diagnostics.get("manufacturing_ok", False) and not reasons:
        reasons = ("manufacturing_not_certified",)
    if part.swept_body is None or part.fabrication_path is None:
        diagnostics["exact_collision_completed"] = False
        return diagnostics, (*reasons, "missing_swept_geometry")
    if not diagnostics.get("mesh_solid_ok", False):
        diagnostics["exact_collision_completed"] = False
        return diagnostics, tuple(dict.fromkeys((*reasons, "invalid_mesh_solid")))
    if reasons:
        # A failed production gate already excludes this body. Spending the
        # exact pair budget cannot certify it or change candidate rejection.
        diagnostics.update(exact_collision_completed=False, exact_collision_skip_reason="production_geometry_failed")
        return diagnostics, reasons
    exact, reason = _exact_collision_audit(part.swept_body, limits, progress_callback)
    diagnostics.update(exact)
    # Keep acceptance evidence with the returned body rather than only in the
    # validation wrapper. Production manufacturing diagnostics remain intact.
    part.manufacturing_diagnostics.update(exact)
    return diagnostics, tuple(dict.fromkeys((*reasons, *((reason,) if reason else ()))))


def prepare_group_route_baseline(
    part: CoilPart,
    config: CoilGenConfig,
    target_points_m,
    limits: GroupRouteValidationLimits = GroupRouteValidationLimits(),
    *,
    reuse_existing_sweep: bool = False,
    audit_geometry: bool = True,
    progress_callback=None,
):
    """Prepare an already shifted/cleaned baseline without changing the source.

    Explicit reuse is for an existing production sweep of this route/config.
    It requires both the body and fabrication path, audits the copied body,
    and never silently generates missing geometry. All physical state enters
    the content fingerprint, including diagnostics and centroid coordinates.
    ``audit_geometry=False`` prepares only field references from existing
    physical geometry. It grants no incumbent certificate; the adoption caller
    must run full baseline certification before comparing objective values.
    """
    targets = _target_points(target_points_m, limits)
    if not audit_geometry and not reuse_existing_sweep:
        raise ValueError("Deferred baseline geometry audit requires an existing sweep.")
    reason = _admission(part, config, limits)
    if reason:
        return GroupRouteBaseline("not_admitted", _readonly(np.zeros_like(targets)), None, {}, (reason,))
    fingerprint = _fingerprint(part, config, targets, limits)
    preview = _isolated_part(part)
    route_field = _readonly(_field(preview.wire_path.v, targets))
    try:
        if reuse_existing_sweep:
            if part.swept_body is None or part.fabrication_path is None:
                return GroupRouteBaseline(
                    fingerprint,
                    route_field,
                    None,
                    {"baseline_sweep_reused": False},
                    ("missing_existing_swept_geometry",),
                )
            # One deepcopy retains aliases between the aggregate body and its
            # SweptBody record while isolating diagnostic updates from the source.
            if audit_geometry:
                state = deepcopy(tuple(getattr(part, name) for name in _PHYSICAL_STATE))
                for name, value in zip(_PHYSICAL_STATE, state):
                    setattr(preview, name, value)
                diagnostics, reasons = _check_swept_geometry(preview, limits, progress_callback)
            else:
                # Only read the centroid; avoid copying a dense body that no
                # field-rejected candidate will need audited.
                preview.fabrication_path = part.fabrication_path
                diagnostics = {"baseline_geometry_audit_deferred": True, "exact_collision_completed": False}
                reasons = ()
        else:
            diagnostics, reasons = _sweep_preview(preview, config, limits, progress_callback)
        diagnostics["baseline_sweep_reused"] = bool(reuse_existing_sweep)
        physical_field = (
            None
            if preview.fabrication_path is None
            else _readonly(_field(preview.fabrication_path.wire_points_m, targets))
        )
    except DesignCancelled:
        raise
    except (ValueError, RuntimeError, FloatingPointError) as error:
        diagnostics, reasons, physical_field = {"error": str(error)}, ("baseline_sweep_failed",), None
    return GroupRouteBaseline(fingerprint, route_field, physical_field, diagnostics, reasons)


def _structural_reasons(path, segment_ids, protected_segments, group_sequence, required_groups, cuts):
    points = path.v
    reasons = []
    if len(points) < 4 or not np.all(np.isfinite(points)) or not np.array_equal(points[0], points[-1]):
        return ("candidate_not_finite_closed_cycle",)
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-12):
        reasons.append("candidate_zero_length_segment")
    # Vertices are traversal occurrences, not spatial welds. A return can
    # revisit a position before lifting; geometry is checked after preparation.
    if len(segment_ids) != len(points) - 1:
        return (*reasons, "segment_provenance_length_mismatch")
    if len(path.segment_owners) != len(segment_ids):
        reasons.append("missing_explicit_segment_owners")
    counts = Counter(identifier for identifier in segment_ids if identifier is not None)
    if set(counts) != set(protected_segments) or any(value != 1 for value in counts.values()):
        reasons.append("directed_contour_coverage_mismatch")
    for index, identifier in enumerate(segment_ids):
        if identifier is None or identifier not in protected_segments:
            continue
        expected = np.asarray(protected_segments[identifier], dtype=float)
        if expected.shape != (2, 3) or not np.array_equal(points[index : index + 2], expected):
            reasons.append("protected_segment_coordinates_or_direction_changed")
            break
    sequence, required = tuple(group_sequence), tuple(required_groups)
    if not sequence or len(set(sequence)) != len(sequence) or Counter(sequence) != Counter(required):
        reasons.append("group_cycle_coverage_mismatch")
    elif len(sequence) > 1:
        expected_edges = Counter(zip(sequence, sequence[1:] + sequence[:1]))
        actual_edges = Counter((cut.group1_index, cut.group2_index) for cut in cuts)
        if actual_edges != expected_edges:
            reasons.append("group_transition_cycle_mismatch")
    return tuple(dict.fromkeys(reasons))


def _prepared_contour_reasons(path, protected_segments):
    # Normal shifting may repair replaceable edges, but it may not silently
    # move or drop a prescribed directed contour edge.
    points = path.v
    if len(points) < 4 or not np.all(np.isfinite(points)) or not np.array_equal(points[0], points[-1]):
        return ("prepared_not_finite_closed_cycle",)
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-12):
        return ("prepared_zero_length_segment",)
    edges = Counter((tuple(a), tuple(b)) for a, b in zip(points[:-1], points[1:]))
    required = Counter((tuple(a), tuple(b)) for a, b in protected_segments.values())
    if any(edges[edge] != count for edge, count in required.items()):
        return ("prepared_protected_contour_changed",)
    if len(np.unique(path.v[:-1], axis=0)) != len(path.v) - 1:
        return ("prepared_revisited_vertex",)
    return ()


def validate_group_route_candidate(
    part: CoilPart,
    candidate_path: WirePath,
    config: CoilGenConfig,
    *,
    protected_segments: Mapping[Hashable, np.ndarray],
    candidate_segment_ids: Sequence[Hashable | None],
    group_sequence: Sequence[int],
    required_groups: Sequence[int],
    candidate_cuts: Sequence[InterGroupCut] = (),
    target_points_m: np.ndarray,
    maximum_field_delta_t_per_a: float,
    maximum_longitudinal_field_delta_t_per_a: float | None = None,
    target_field: TargetField | None = None,
    current_a: float | None = None,
    other_field_per_amp: np.ndarray | None = None,
    other_physical_field_per_amp: np.ndarray | None = None,
    apply_normal_shift: bool = True,
    baseline: GroupRouteBaseline | None = None,
    limits: GroupRouteValidationLimits = GroupRouteValidationLimits(),
    progress_callback=None,
) -> GroupRouteValidation:
    """Require exact directed coverage, same-current fields and a valid body.

    ``None`` segment IDs explicitly mark replaceable routing edges. Protected
    IDs must be immutable, unique per required directed edge, including separate
    physical turns. The per-amp delta budget is absolute, with no current refit.
    A baseline failure never waives any candidate manufacturing constraint.
    Set ``apply_normal_shift=False`` for explicitly shaped transition curves;
    cleanup, protected-edge, field, manufacturing and exact collision checks
    still run. The physical-stage background can be supplied separately from
    the route-stage background; omission retains the existing shared behavior.
    """
    # Reject large candidates before copying route/group state or sweeping a baseline.
    if len(candidate_path.v) > limits.maximum_route_points:
        return GroupRouteValidation(False, ("route_point_budget_exceeded",))
    reasons = _structural_reasons(
        candidate_path, candidate_segment_ids, protected_segments, group_sequence, required_groups, candidate_cuts
    )
    if reasons:
        return GroupRouteValidation(False, reasons)
    try:
        targets = _target_points(target_points_m, limits)
    except ValueError as error:
        return GroupRouteValidation(False, ("invalid_target_points",), {"error": str(error)})
    if target_field is not None and not np.array_equal(targets, target_field.coords):
        return GroupRouteValidation(False, ("target_coordinates_mismatch",))
    if not np.isfinite(maximum_field_delta_t_per_a) or maximum_field_delta_t_per_a < 0:
        return GroupRouteValidation(False, ("invalid_field_delta_budget",))
    longitudinal_budget = maximum_longitudinal_field_delta_t_per_a
    if longitudinal_budget is None:
        if getattr(config, "joint_group_routing_max_longitudinal_field_change_rel", None) is not None:
            return GroupRouteValidation(False, ("missing_longitudinal_field_delta_budget",))
    elif (
        isinstance(longitudinal_budget, (bool, np.bool_))
        or not np.isfinite(longitudinal_budget)
        or longitudinal_budget < 0
    ):
        return GroupRouteValidation(False, ("invalid_longitudinal_field_delta_budget",))
    current = part.contour_step if current_a is None else current_a
    if current is None or not np.isfinite(current) or current == 0:
        return GroupRouteValidation(False, ("invalid_fixed_current",))
    if target_field is None and any(
        getattr(config, name) > 0
        for name in (
            "layout_target_error_std_limit_T",
            "layout_target_error_rms_limit_T",
            "suppression_field_rms_limit_T",
            "suppression_field_peak_limit_T",
            "linearity_rms_tolerance_percent",
            "linearity_peak_tolerance_percent",
            "linearity_cross_gradient_tolerance_percent",
            "gradient_strength_tolerance_percent",
        )
    ):
        return GroupRouteValidation(False, ("missing_target_field_context",))
    try:
        other = np.zeros_like(targets) if other_field_per_amp is None else np.asarray(other_field_per_amp, dtype=float)
        if other.shape != targets.shape or not np.all(np.isfinite(other)):
            return GroupRouteValidation(False, ("invalid_other_part_field",))
    except (TypeError, ValueError):
        return GroupRouteValidation(False, ("invalid_other_part_field",))
    try:
        physical_other = (
            other if other_physical_field_per_amp is None else np.asarray(other_physical_field_per_amp, dtype=float)
        )
        if physical_other.shape != targets.shape or not np.all(np.isfinite(physical_other)):
            return GroupRouteValidation(False, ("invalid_other_physical_part_field",))
    except (TypeError, ValueError):
        return GroupRouteValidation(False, ("invalid_other_physical_part_field",))
    if baseline is not None and baseline.fingerprint != _fingerprint(part, config, targets, limits):
        return GroupRouteValidation(False, ("stale_baseline",))
    preview = _isolated_part(part)
    preview.wire_path = deepcopy(candidate_path)
    preview.opening_cuts_among_groups = deepcopy(list(candidate_cuts))
    preview.route_diagnostics = {}
    preview.normal_shift_diagnostics = {}
    preview.normal_shift_provenance = {}
    preview.shift_array = preview.points_to_shift = None
    preview.wire_self_contacts = []
    reason = _admission(preview, config, limits)
    if reason:
        return GroupRouteValidation(False, (reason,))
    diagnostics = {
        "protected_segment_count": len(protected_segments),
        "group_count": len(group_sequence),
        "field_current_a": current,
        "normal_shift_applied": bool(apply_normal_shift),
    }
    try:
        preview_config = _preview_config(config)
        if apply_normal_shift:
            shift_return_paths([preview], preview_config)
        cleanup_wire_paths([preview], config=preview_config)
        reasons = _prepared_contour_reasons(preview.wire_path, protected_segments)
        if reasons:
            return GroupRouteValidation(False, reasons, diagnostics, baseline=baseline)
        reason = _admission(preview, config, limits)
        if reason:
            return GroupRouteValidation(False, (reason,), diagnostics, baseline=baseline)
        if baseline is None:
            baseline = prepare_group_route_baseline(part, config, targets, limits, progress_callback=progress_callback)
        diagnostics["baseline_manufacturing_reasons"] = baseline.reasons
        diagnostics["baseline_geometry_audit_deferred"] = baseline.diagnostics.get(
            "baseline_geometry_audit_deferred", False
        )
        if baseline.physical_field_per_amp is None:
            return GroupRouteValidation(False, ("baseline_physical_field_unavailable",), diagnostics, baseline=baseline)

        def assess(reference, candidate, stage):
            background = physical_other if stage == "physical" else other
            reference, candidate = reference + background, candidate + background
            if target_field is None:
                # With no target context only the explicit vector-change
                # criterion is available; configured target limits require it.
                delta = np.linalg.norm(candidate - reference, axis=1)
                diagnostics[f"{stage}_maximum_field_delta_t_per_a"] = float(np.max(delta))
                if not np.all(np.isfinite(delta)) or np.max(delta) > maximum_field_delta_t_per_a:
                    return ("field_delta_budget_exceeded",)
                if longitudinal_budget is not None:
                    from .group_route_fields import _maximum_longitudinal_delta

                    maximum_longitudinal_delta = _maximum_longitudinal_delta(
                        candidate - reference, config.b_0_direction
                    )
                    diagnostics[f"{stage}_maximum_pointwise_longitudinal_delta_t_per_a"] = maximum_longitudinal_delta
                    diagnostics[f"{stage}_maximum_longitudinal_field_delta_t_per_a"] = longitudinal_budget
                    if not np.isfinite(maximum_longitudinal_delta) or maximum_longitudinal_delta > longitudinal_budget:
                        return ("pointwise_longitudinal_field_delta_budget",)
                return ()
            from .group_route_fields import assess_group_route_field

            longitudinal_kwargs = (
                {} if longitudinal_budget is None else {"maximum_longitudinal_field_delta_t_per_a": longitudinal_budget}
            )
            assessment = assess_group_route_field(
                reference,
                candidate,
                target_field,
                config,
                diagnostics["field_current_a"],
                maximum_field_delta_t_per_a,
                **longitudinal_kwargs,
            )
            diagnostics.update({f"{stage}_{key}": value for key, value in assessment.metrics.items()})
            return assessment.reasons

        reasons = assess(baseline.route_field_per_amp, _field(preview.wire_path.v, targets), "route")
        if reasons:
            return GroupRouteValidation(False, tuple(reasons), diagnostics, baseline=baseline)
        sweep, reasons = _sweep_preview(preview, config, limits, progress_callback)
        diagnostics["manufacturing"] = sweep
        if reasons:
            return GroupRouteValidation(False, reasons, diagnostics, baseline=baseline)
        if not preview.fabrication_path.closed:
            return GroupRouteValidation(False, ("prepared_physical_path_not_closed",), diagnostics, baseline=baseline)
        physical_field = _field(preview.fabrication_path.wire_points_m, targets)
        reasons = assess(baseline.physical_field_per_amp, physical_field, "physical")
        if reasons:
            return GroupRouteValidation(False, tuple(reasons), diagnostics, baseline=baseline)
    except DesignCancelled:
        raise
    except (ValueError, RuntimeError, FloatingPointError) as error:
        diagnostics["error"] = str(error)
        return GroupRouteValidation(False, ("candidate_preview_failed",), diagnostics, baseline=baseline)
    diagnostics["normal_shift"] = deepcopy(preview.normal_shift_diagnostics)
    diagnostics["route"] = deepcopy(preview.route_diagnostics)
    return GroupRouteValidation(True, (), diagnostics, preview, baseline, _readonly(physical_field))
