"""Exact connector clearance against a fixed snapshot of winding contours."""

from __future__ import annotations

import math

import numpy as np

from .geometry_collision import _build_segment_bvh
from .geometry_kernels import segment_distance


class ContourClearanceIndex:
    """Reuse conservative bounds without changing the segment distance kernel.

    The index belongs to one routing/diagnostic operation. It copies contour
    geometry; callers create a new index after any contour change. Only AABBs
    farther than the best exact distance are pruned. Leaf pairs use the same
    scalar kernel and 1e-12 segment admission as legacy connector diagnostics.
    """

    def __init__(self, contours):
        starts, ends, labels = [], [], []
        for label, contour in enumerate(contours):
            points = np.asarray(contour, dtype=float)
            if len(points) < 2:
                continue
            valid = np.all(np.isfinite(points[:-1]), axis=1) & np.all(np.isfinite(points[1:]), axis=1)
            valid &= np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
            starts.append(points[:-1][valid])
            ends.append(points[1:][valid])
            labels.append(np.full(np.count_nonzero(valid), label, dtype=np.int64))
        self.starts = np.concatenate(starts) if starts else np.empty((0, 3))
        self.ends = np.concatenate(ends) if ends else np.empty((0, 3))
        self.labels = np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)
        self.root = _build_segment_bvh(self.starts, self.ends) if len(self.starts) else None
        self.scale = max(float(np.max(np.abs(self.starts), initial=0)), float(np.max(np.abs(self.ends), initial=0)))
        self.query_count = 0
        self.segment_pair_evaluations = 0
        for values in (self.starts, self.ends, self.labels):
            values.flags.writeable = False

    def distance_to_path(self, points, excluded_loops=()):
        """Return exact minimum and first contour on ties; empty sets give inf/-1."""
        self.query_count += 1
        points = np.asarray(points, dtype=float)
        if self.root is None or len(points) < 2:
            return math.inf, -1
        allowed = ~np.isin(self.labels, tuple(excluded_loops))
        if not np.any(allowed):
            return math.inf, -1
        first_allowed = int(np.min(self.labels[allowed]))
        scale = max(self.scale, float(np.max(np.abs(points), initial=0)), np.finfo(float).tiny)
        padding = 64 * np.finfo(float).eps * scale
        best, best_loop = math.inf, -1
        for start, end in zip(points[:-1], points[1:]):
            if not np.all(np.isfinite((start, end))) or np.linalg.norm(end - start) <= 1e-12:
                continue
            low, high = np.minimum(start, end), np.maximum(start, end)

            def lower_bound(node):
                separation = np.maximum(0.0, np.maximum(low - node.maximum, node.minimum - high))
                return float(np.linalg.norm(separation))

            stack = [(lower_bound(self.root), self.root)]
            while stack:
                bound, node = stack.pop()
                if bound > best + padding:
                    continue
                if node.leaf:
                    for index in node.indices[allowed[node.indices]]:
                        self.segment_pair_evaluations += 1
                        distance = segment_distance(start, end, self.starts[index], self.ends[index])
                        label = int(self.labels[index])
                        if distance < best or (distance == best and label < best_loop):
                            best, best_loop = distance, label
                    if best == 0.0 and best_loop == first_allowed:
                        return best, best_loop
                else:
                    children = [(lower_bound(child), child) for child in (node.left, node.right)]
                    children.sort(key=lambda item: item[0], reverse=True)
                    stack.extend(children)
        return best, best_loop
