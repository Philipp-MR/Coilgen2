"""Bounded clearance and containment checks between finished conductor bodies.

Each body still needs its own manufacturing/self-intersection certificate.
This audit compares different bodies only; a coincident vertex or edge is never
an allowed electrical connection between them. Distances are in metres.
"""

from __future__ import annotations

import numpy as np

from .geometry_collision import _SegmentBvhNode, _segment_bvh_candidate_pairs
from .geometry_kernels import triangle_pair_distances


class _AuditIncomplete(RuntimeError):
    pass


def _bounds_tree(minimum, maximum, indices):
    low, high = minimum[indices].min(axis=0), maximum[indices].max(axis=0)
    if len(indices) <= 12:
        return _SegmentBvhNode(indices, low, high)
    centers = (minimum[indices] + maximum[indices]) / 2
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    ordered = indices[np.argsort(centers[:, axis], kind="stable")]
    middle = len(ordered) // 2
    return _SegmentBvhNode(
        np.empty(0, dtype=np.int64),
        low,
        high,
        _bounds_tree(minimum, maximum, ordered[:middle]),
        _bounds_tree(minimum, maximum, ordered[middle:]),
    )


def _box_gap(first_low, first_high, second_low, second_high):
    return np.maximum(0, np.maximum(first_low - second_high, second_low - first_high))


def _components(vertices, faces):
    """Edge-connected closed shells, representative points and orientation signs.

    Two shells may share one welded vertex without sharing their material
    boundary. Joining them by vertex connectivity would incorrectly XOR their
    volumes during containment tests.
    """
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, edge_labels, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise _AuditIncomplete("interpart_containment_requires_closed_manifold_bodies")
    parent = np.arange(len(faces))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edge_faces = np.tile(np.arange(len(faces)), 3)
    adjacent = edge_faces[np.argsort(edge_labels, kind="stable")].reshape(-1, 2)
    for first_face, second_face in adjacent:
        first, second = root(first_face), root(second_face)
        if first != second:
            parent[second] = first
    labels = np.array([root(index) for index in range(len(faces))])
    components = []
    order = np.argsort(labels, kind="stable")
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    for indices in np.split(order, boundaries):
        point = vertices[faces[indices[0], 0]]
        relative = vertices[faces[indices]] - point
        terms = np.einsum("ij,ij->i", relative[:, 0], np.cross(relative[:, 1], relative[:, 2]))
        volume = float(np.sum(terms))
        if not np.isfinite(volume) or abs(volume) <= 64 * np.finfo(float).eps * float(np.sum(np.abs(terms))):
            raise _AuditIncomplete("interpart_component_volume_unresolved")
        components.append((point, indices, 1 if volume > 0 else -1))
    return components


class _Body:
    def __init__(self, mesh):
        vertices, faces = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or not np.all(np.isfinite(vertices))
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
        ):
            raise _AuditIncomplete("invalid_interpart_body")
        self.triangles = vertices[faces]
        normals = np.cross(self.triangles[:, 1] - self.triangles[:, 0], self.triangles[:, 2] - self.triangles[:, 0])
        if np.any(np.linalg.norm(normals, axis=1) == 0) or not np.all(np.isfinite(normals)):
            raise _AuditIncomplete("degenerate_interpart_triangle")
        self.minimum, self.maximum = self.triangles.min(axis=1), self.triangles.max(axis=1)
        self.tree = _bounds_tree(self.minimum, self.maximum, np.arange(len(faces)))
        components = _components(vertices, faces)
        self.component_points = np.array([point for point, _indices, _sign in components])
        self.component_trees = (
            [self.tree]
            if len(components) == 1
            else [_bounds_tree(self.minimum, self.maximum, indices) for _point, indices, _sign in components]
        )
        self.mixed_shell_orientations = len({sign for _point, _indices, sign in components}) > 1


def _cross_triangle_batches(first, second, threshold, consume):
    """Yield bounded batches from cross-tree traversal; never same-body pairs."""
    stack, pending, size = [(first.tree, second.tree)], [], 0
    while stack:
        a, b = stack.pop()
        gap = _box_gap(a.minimum, a.maximum, b.minimum, b.maximum)
        if float(gap @ gap) > threshold**2:
            continue
        if a.leaf and b.leaf:
            consume(len(a.indices) * len(b.indices), "triangle")
            ai, bi = np.repeat(a.indices, len(b.indices)), np.tile(b.indices, len(a.indices))
            delta = _box_gap(first.minimum[ai], first.maximum[ai], second.minimum[bi], second.maximum[bi])
            keep = np.einsum("ij,ij->i", delta, delta) <= threshold**2
            if np.any(keep):
                pending.append(np.column_stack((ai[keep], bi[keep])))
                size += int(keep.sum())
            if size >= 4096:
                yield np.concatenate(pending)
                pending, size = [], 0
        elif not a.leaf and (b.leaf or np.linalg.norm(a.maximum - a.minimum) >= np.linalg.norm(b.maximum - b.minimum)):
            stack.extend(((a.right, b), (a.left, b)))
        else:
            stack.extend(((a, b.right), (a, b.left)))
    if pending:
        yield np.concatenate(pending)


def _ray_boxes(point, direction, low, high, tolerance):
    # All deterministic directions below have nonzero components.
    first, last = (low - point - tolerance) / direction, (high - point + tolerance) / direction
    return float(np.min(np.maximum(first, last))) >= max(0.0, float(np.max(np.minimum(first, last))))


def _inside_body(point, body, tolerance, consume, *, tree=None):
    """Parity for ONE shell; ambiguous rays retry instead of guessing.

    Do not combine disconnected target shells here: parity would XOR their
    volumes, concealing overlap inside two nested conductor components.
    """
    tree = tree if tree is not None else body.tree
    if np.any(point < tree.minimum - tolerance) or np.any(point > tree.maximum + tolerance):
        return False
    directions = (
        (1.0, np.sqrt(2), np.sqrt(3)),
        (np.sqrt(3), -1.0, np.sqrt(2)),
        (np.sqrt(2), np.sqrt(3), -1.0),
        (1.0, -np.sqrt(5), np.sqrt(7)),
    )
    for raw in directions:
        direction = np.asarray(raw) / np.linalg.norm(raw)
        stack, intersections, ambiguous = [tree], 0, False
        while stack and not ambiguous:
            node = stack.pop()
            if not _ray_boxes(point, direction, node.minimum, node.maximum, tolerance):
                continue
            if not node.leaf:
                stack.extend((node.right, node.left))
                continue
            consume(len(node.indices), "containment")
            triangles = body.triangles[node.indices]
            first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            normals = np.cross(first, second)
            cross = np.cross(direction, second)
            determinant = np.einsum("ij,ij->i", first, cross)
            scale = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
            parallel = np.abs(determinant) <= 64 * np.finfo(float).eps * scale
            delta = point - triangles[:, 0]
            coplanar = np.abs(np.einsum("ij,ij->i", delta, normals)) <= tolerance * np.linalg.norm(normals, axis=1)
            if np.any(parallel & coplanar):
                ambiguous = True
                continue
            valid = ~parallel
            delta, first, second, cross, determinant, scale = (
                value[valid] for value in (delta, first, second, cross, determinant, scale)
            )
            if not len(determinant):
                continue
            u = np.einsum("ij,ij->i", delta, cross) / determinant
            other = np.cross(delta, first)
            v = (other @ direction) / determinant
            distance = np.einsum("ij,ij->i", second, other) / determinant
            conditioning = scale / np.abs(determinant)
            bary_tol = np.maximum(
                128 * np.finfo(float).eps * conditioning,
                tolerance / np.maximum(np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)),
            )
            possible = (distance >= -tolerance) & (u >= -bary_tol) & (v >= -bary_tol) & (u + v <= 1 + bary_tol)
            uncertain = (distance <= tolerance) | (u <= bary_tol) | (v <= bary_tol) | (u + v >= 1 - bary_tol)
            if np.any(possible & uncertain):
                ambiguous = True
                continue
            intersections += int(np.count_nonzero(possible))
        if not ambiguous:
            return bool(intersections % 2)
    raise _AuditIncomplete("interpart_containment_numerically_ambiguous")


def audit_interpart_clearance(bodies, required_clearance_m, limits):
    """Return ``(diagnostics, reason)``; empty reason certifies inter-part gaps.

    The existing maximum-sweep-face budget applies to the total input. Separate
    body-pair, cross-triangle and containment-ray work counters each use the
    maximum-triangle-pair budget. No same-body triangle comparisons consume it.
    Individual bodies must already have valid manufacturing/self-intersection
    certificates. Every connected shell is checked for containment after all
    nearby surfaces are separated. Multiple conductor shells use union semantics;
    mixed orientations could instead describe a cavity and are unsupported when
    containment needs checking. Incomplete/ambiguous checks reject.
    """
    diagnostics = {
        "interpart_clearance_completed": False,
        "interpart_body_count": len(bodies),
        "interpart_surface_distance_scope": "pairs_near_required_gap",
        "interpart_containment_completed": False,
    }
    try:
        required = float(required_clearance_m)
        if not np.isfinite(required) or required < 0:
            raise _AuditIncomplete("invalid_interpart_clearance")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (limits.maximum_sweep_faces, limits.maximum_triangle_pairs)
        ):
            raise _AuditIncomplete("invalid_interpart_audit_budget")
        face_count = sum(len(body.faces) for body in bodies)
        diagnostics.update(interpart_face_count=face_count, interpart_required_clearance_m=required)
        if face_count > limits.maximum_sweep_faces:
            raise _AuditIncomplete("interpart_face_budget_exceeded")
        if len(bodies) < 2:
            diagnostics.update(interpart_clearance_completed=True, interpart_containment_completed=True)
            return diagnostics, ""
        prepared = [_Body(body) for body in bodies]
        low, high = (
            np.array([body.tree.minimum for body in prepared]),
            np.array([body.tree.maximum for body in prepared]),
        )
        tolerance = max(
            1e-12, 64 * np.finfo(float).eps * max(float(np.max(np.abs(low))), float(np.max(np.abs(high))), required)
        )
        threshold = required + tolerance
        diagnostics["interpart_numerical_tolerance_m"] = tolerance
        body_pairs = _segment_bvh_candidate_pairs(
            low, high, threshold, leaf_size=4, max_pairs=limits.maximum_triangle_pairs
        )
        if body_pairs is None:
            raise _AuditIncomplete("interpart_body_pair_budget_exceeded")
        if len(body_pairs):
            delta = _box_gap(
                low[body_pairs[:, 0]], high[body_pairs[:, 0]], low[body_pairs[:, 1]], high[body_pairs[:, 1]]
            )
            body_pairs = body_pairs[np.einsum("ij,ij->i", delta, delta) <= threshold**2]
        diagnostics.update(
            interpart_body_pair_count=len(body_pairs),
            interpart_triangle_candidates=0,
            interpart_triangle_pairs_tested=0,
            interpart_containment_triangles_tested=0,
            interpart_containment_queries=0,
            interpart_minimum_checked_surface_distance_m=None,
        )

        def consume(count, stage):
            key = {
                "triangle": "interpart_triangle_candidates",
                "containment": "interpart_containment_triangles_tested",
                "shell_query": "interpart_containment_queries",
            }[stage]
            diagnostics[key] += count
            if diagnostics[key] > limits.maximum_triangle_pairs:
                raise _AuditIncomplete(f"interpart_{stage}_pair_budget_exceeded")

        minimum = np.inf
        for a, b in body_pairs:
            first, second = prepared[a], prepared[b]
            for pairs in _cross_triangle_batches(first, second, threshold, consume):
                distances = triangle_pair_distances(
                    first.triangles[pairs[:, 0]], second.triangles[pairs[:, 1]]
                ).values_m
                if not np.all(np.isfinite(distances)):
                    raise _AuditIncomplete("nonfinite_interpart_triangle_distance")
                diagnostics["interpart_triangle_pairs_tested"] += len(pairs)
                minimum = min(minimum, float(np.min(distances)))
                diagnostics["interpart_minimum_checked_surface_distance_m"] = minimum
                violations = (distances <= tolerance) | (distances < required - tolerance)
                if np.any(violations):
                    index = int(np.flatnonzero(violations)[0])
                    diagnostics.update(
                        interpart_witness_body_indices=[int(a), int(b)],
                        interpart_witness_triangle_indices=pairs[index].tolist(),
                        interpart_witness_surface_distance_m=float(distances[index]),
                    )
                    return diagnostics, "interpart_surface_clearance_violation"
            if first.mixed_shell_orientations or second.mixed_shell_orientations:
                raise _AuditIncomplete("interpart_cavity_shell_semantics_unsupported")
            for source, target, source_index, target_index in ((first, second, a, b), (second, first, b, a)):
                for point in source.component_points:
                    for shell_index, tree in enumerate(target.component_trees):
                        consume(1, "shell_query")
                        if _inside_body(point, target, tolerance, consume, tree=tree):
                            diagnostics.update(
                                interpart_containment_witness_body_indices=[int(source_index), int(target_index)],
                                interpart_containment_witness_target_shell=shell_index,
                                interpart_containment_witness_point_m=point.tolist(),
                            )
                            return diagnostics, "interpart_solid_containment"
        diagnostics.update(
            interpart_clearance_completed=True,
            interpart_containment_completed=True,
            interpart_certified_gap_lower_bound_m=max(0.0, required - tolerance),
        )
        return diagnostics, ""
    except (_AuditIncomplete, AttributeError, TypeError, ValueError, FloatingPointError) as error:
        diagnostics["interpart_clearance_detail"] = str(error)
        return diagnostics, str(error) if isinstance(error, _AuditIncomplete) else "invalid_interpart_audit_context"
