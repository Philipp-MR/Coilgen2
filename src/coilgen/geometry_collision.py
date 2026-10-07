"""Primitive broad-phase and accelerated mesh checks shared by sweep callers.

This module depends only on arrays and the distance kernels. Caller-specific
adjacency exclusions, reporting limits and contact tolerances remain explicit.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .geometry_kernels import triangle_pair_distances


class _SegmentBvhNode:
    __slots__ = ("indices", "minimum", "maximum", "left", "right")

    def __init__(
        self,
        indices: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
        left: "_SegmentBvhNode | None" = None,
        right: "_SegmentBvhNode | None" = None,
    ) -> None:
        self.indices = indices
        self.minimum = minimum
        self.maximum = maximum
        self.left = left
        self.right = right

    @property
    def leaf(self) -> bool:
        return self.left is None


@lru_cache(maxsize=128)
def _leaf_positions(first: int, second: int, same: bool):
    positions = np.triu_indices(first, 1) if same else tuple(np.indices((first, second)).reshape(2, -1))
    for value in positions:
        value.flags.writeable = False
    return positions


def _build_segment_bvh(starts: np.ndarray, ends: np.ndarray, leaf_size: int = 12) -> _SegmentBvhNode:
    """Build a deterministic AABB hierarchy over a nonempty segment array."""
    segment_minima = np.minimum(starts, ends)
    segment_maxima = np.maximum(starts, ends)
    centroids = 0.5 * (segment_minima + segment_maxima)

    def build(indices: np.ndarray) -> _SegmentBvhNode:
        minimum = np.min(segment_minima[indices], axis=0)
        maximum = np.max(segment_maxima[indices], axis=0)
        if len(indices) <= leaf_size:
            return _SegmentBvhNode(indices, minimum, maximum)
        axis = int(np.argmax(np.ptp(centroids[indices], axis=0)))
        order = indices[np.argsort(centroids[indices, axis], kind="stable")]
        middle = len(order) // 2
        return _SegmentBvhNode(
            np.zeros(0, dtype=np.int64), minimum, maximum, build(order[:middle]), build(order[middle:])
        )

    return build(np.arange(len(starts), dtype=np.int64))


def _segment_bvh_candidate_pairs(
    starts: np.ndarray,
    ends: np.ndarray,
    distance: float,
    *,
    leaf_size: int = 12,
    max_pairs: int | None = None,
) -> np.ndarray | None:
    """Return conservative BVH pairs, or ``None`` if the optional budget is exceeded.

    Leaf pairs still need individual AABB filtering. The budget limits raw
    leaf pairs before allocation; callers can fall back to a bounded search.
    Existing callers without a budget retain their complete candidate set.
    """

    if len(starts) < 2:
        return np.zeros((0, 2), dtype=np.int64)

    def boxes_within_distance(first: _SegmentBvhNode, second: _SegmentBvhNode) -> bool:
        separation = np.maximum(0.0, np.maximum(first.minimum - second.maximum, second.minimum - first.maximum))
        return bool(np.dot(separation, separation) <= distance * distance)

    pairs: list[np.ndarray] = []
    pair_count = 0
    exceeded_budget = False

    def reserve(count: int) -> bool:
        nonlocal exceeded_budget
        if max_pairs is not None and pair_count + count > max_pairs:
            exceeded_budget = True
            return False
        return True

    def collect(first: _SegmentBvhNode, second: _SegmentBvhNode) -> None:
        nonlocal pair_count
        if exceeded_budget or not boxes_within_distance(first, second):
            return
        if first is second:
            if first.leaf:
                if not reserve(len(first.indices) * (len(first.indices) - 1) // 2):
                    return
                local_i, local_j = _leaf_positions(len(first.indices), len(first.indices), True)
                i, j = first.indices[local_i], first.indices[local_j]
                pairs.append(np.column_stack((np.minimum(i, j), np.maximum(i, j))))
                pair_count += len(i)
                return
            collect(first.left, first.left)
            collect(first.left, first.right)
            collect(first.right, first.right)
            return
        if first.leaf and second.leaf:
            if not reserve(len(first.indices) * len(second.indices)):
                return
            local_i, local_j = _leaf_positions(len(first.indices), len(second.indices), False)
            i, j = first.indices[local_i], second.indices[local_j]
            pairs.append(np.column_stack((np.minimum(i, j), np.maximum(i, j))))
            pair_count += len(i)
            return
        first_size = (
            len(first.indices) if first.leaf else np.dot(first.maximum - first.minimum, first.maximum - first.minimum)
        )
        second_size = (
            len(second.indices)
            if second.leaf
            else np.dot(second.maximum - second.minimum, second.maximum - second.minimum)
        )
        if not first.leaf and (second.leaf or first_size >= second_size):
            collect(first.left, second)
            collect(first.right, second)
        else:
            collect(first, second.left)
            collect(first, second.right)

    root = _build_segment_bvh(starts, ends, leaf_size)
    collect(root, root)
    if exceeded_budget:
        return None
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    raw = np.concatenate(pairs, axis=0)
    if not len(raw):
        return np.zeros((0, 2), dtype=np.int64)
    order = np.lexsort((raw[:, 1], raw[:, 0]))
    ordered = raw[order]
    keep = np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)]
    return ordered[keep]


def accelerated_mesh_collision_diagnostics(mesh, *, backend="auto", maximum_reported_intersections=100):
    """Batch the same remote-triangle test used by research sweep validation."""
    triangles = mesh.vertices[mesh.faces]
    tolerance = max(float(np.max(np.ptp(mesh.vertices, axis=0))), 1.0) * 1e-12 if len(mesh.vertices) else 1e-12
    minima, maxima = triangles.min(axis=1), triangles.max(axis=1)
    pairs = _segment_bvh_candidate_pairs(minima, maxima, tolerance)
    if len(pairs):
        delta = np.maximum(
            0.0, np.maximum(minima[pairs[:, 0]] - maxima[pairs[:, 1]], minima[pairs[:, 1]] - maxima[pairs[:, 0]])
        )
        pairs = pairs[np.einsum("ij,ij->i", delta, delta) <= tolerance**2]
        adjacent = np.any(mesh.faces[pairs[:, 0], :, None] == mesh.faces[pairs[:, 1], None, :], axis=(1, 2))
        pairs = pairs[~adjacent]
    measured = triangle_pair_distances(triangles[pairs[:, 0]], triangles[pairs[:, 1]], backend=backend)
    count = int(np.count_nonzero(measured.values_m <= tolerance))
    return {
        "sweep_mesh_collision_broad_phase": "triangle_bvh",
        "sweep_mesh_collision_kernel": measured.backend,
        "sweep_mesh_collision_candidate_pair_count": len(pairs),
        "sweep_mesh_self_intersection_count": min(count, maximum_reported_intersections),
        "sweep_mesh_self_intersection_truncated": count >= maximum_reported_intersections,
        "sweep_mesh_self_intersection": count > 0,
    }
