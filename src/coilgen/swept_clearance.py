"""Clearance between different stretches of one manufactured conductor."""

from __future__ import annotations

import numpy as np


class SweepIntervals:
    """Map the production loft facets to the intervals of their physical winding."""

    def __init__(self, path, faces):
        self.closed = path.closed
        points = path.wire_points_m
        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        self.arc = np.r_[0.0, np.cumsum(lengths)]
        tangents = np.divide(vectors, lengths[:, None], out=np.zeros_like(vectors), where=lengths[:, None] > 0)
        # A short returning leg can face its neighbor even inside the usual
        # arclength exclusion for a straight run. Include those folded visits.
        turns = np.arccos(np.clip(np.einsum("ij,ij->i", tangents[:-1], tangents[1:]), -1.0, 1.0))
        self.turn = np.r_[0.0, np.cumsum(turns)]
        self.total_turn = self.turn[-1]
        if self.closed:
            self.total_turn += np.arccos(np.clip(tangents[-1] @ tangents[0], -1.0, 1.0))
        profile = path.cross_section_m
        self.local_length = 4 * np.max(np.linalg.norm(profile[:, None] - profile[None, :], axis=2))
        self.station = np.arange(len(faces)) // (2 * len(profile))
        caps = self.station >= len(lengths)
        if np.any(caps):
            # The open loft appends alternating front and back caps.
            self.station[caps] = np.where(np.flatnonzero(caps) % 2, len(lengths) - 1, 0)

    def separate_visits(self, first, second):
        i, j = (
            np.minimum(self.station[first], self.station[second]),
            np.maximum(self.station[first], self.station[second]),
        )
        gap = np.maximum(0.0, self.arc[j] - self.arc[i + 1])
        turn = self.turn[j] - self.turn[i]
        neighbors = j <= i + 1
        if self.closed:
            wrap = self.arc[-1] - self.arc[j + 1] + self.arc[i]
            turn = np.where(wrap < gap, self.total_turn - turn, turn)
            gap = np.minimum(wrap, gap)
            neighbors |= j - i == len(self.turn) - 1
        return ~neighbors & ((gap > self.local_length) | (turn >= np.pi / 2))


def required_copper_gap(config):
    return float(
        2 * config.sweep_insulation_thickness
        + config.sweep_manufacturing_tolerance
        + config.sweep_bend_clearance_margin
    )


def loft_pair_batches(minima, maxima, gap, limits, diagnostics):
    """Stream a binary AABB tree over consecutive loft facets, in bounded tiles."""
    from .group_route_validation import _TriangleAuditBudgetExceeded

    if len(minima) < 2:
        return

    lo, hi = [minima], [maxima]
    while len(lo[-1]) > 1:
        left, right = lo[-1][::2].copy(), hi[-1][::2].copy()
        n = len(lo[-1]) // 2
        left[:n] = np.minimum(left[:n], lo[-1][1::2])
        right[:n] = np.maximum(right[:n], hi[-1][1::2])
        lo.append(left)
        hi.append(right)
    raw = near = 0
    pending = []
    pending_count = 0

    def visit(level, a, b):
        nonlocal raw, near
        delta = np.maximum(0, np.maximum(lo[level][a] - hi[level][b], lo[level][b] - hi[level][a]))
        hit = np.einsum("ij,ij->i", delta, delta) <= gap**2
        a, b = a[hit], b[hit]
        if level == 0:
            raw += len(hit)
            near += len(a)
            diagnostics["exact_collision_broadphase_pair_count"] = raw
            diagnostics["exact_collision_near_pair_count"] = near
            if raw > limits.maximum_triangle_broadphase_pairs:
                raise _TriangleAuditBudgetExceeded("triangle_broadphase_pair_budget_exceeded")
            if near > limits.maximum_triangle_pairs:
                raise _TriangleAuditBudgetExceeded("triangle_pair_budget_exceeded")
            if len(a):
                yield np.column_stack((a, b))
            return
        for start in range(0, len(a), 4096):
            i, j = a[start : start + 4096] * 2, b[start : start + 4096] * 2
            ca = np.concatenate((i, i, i + 1, i + 1))
            cb = np.concatenate((j, j + 1, j, j + 1))
            valid = ((ca < cb) if level == 1 else (ca <= cb)) & (cb < len(lo[level - 1]))
            yield from visit(level - 1, ca[valid], cb[valid])

    root = np.array([0])
    for pairs in visit(len(lo) - 1, root, root):
        pending.append(pairs)
        pending_count += len(pairs)
        if pending_count >= 4096:
            pairs = np.concatenate(pending)
            offset = len(pairs) // 4096 * 4096
            for i in range(0, offset, 4096):
                yield pairs[i : i + 4096]
            pending = [pairs[offset:]]
            pending_count -= offset
    if pending_count:
        yield np.concatenate(pending)
