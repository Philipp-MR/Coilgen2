"""Distance batching preserves capped minima and actual spatial work budgets."""

import numpy as np
import pytest

from coilgen.geometry_kernels import segment_distances
from coilgen.joint_transition_curves import _DistanceBudgetExceeded, _SegmentDistanceIndex


def _scalar_gap(index, segments, cap, *, excluded=(), consume=None, maximum_pairs=200000):
    """Frozen scalar traversal before vectorizing its existing 16-query batches."""
    if not len(index.segments) or not len(segments):
        return cap
    first = np.asarray(segments).reshape(-1, 2, 3)
    excluded = np.asarray(tuple(excluded), dtype=np.int64)
    minimum, evaluated = float(cap), 0
    for offset in range(0, len(first), 16):
        batch = first[offset : offset + 16]
        if index.tree is None:
            neighbors, count = index._numpy_neighbors(batch, minimum, maximum_pairs - evaluated)
        else:
            centers = batch.mean(axis=1)
            radii = np.linalg.norm(np.diff(batch, axis=1)[:, 0], axis=1) / 2
            radii += index.maximum_half_length + minimum + 1e-12
            count = int(np.sum(index.tree.query_ball_point(centers, radii, return_length=True)))
        evaluated += count
        if evaluated > maximum_pairs:
            raise _DistanceBudgetExceeded("curve_distance_pair_budget_exceeded")
        if consume is not None:
            consume(count)
        if index.tree is not None:
            neighbors = index.tree.query_ball_point(centers, radii, return_sorted=True)
        for segment, indices in zip(batch, neighbors):
            indices = np.asarray(indices, dtype=np.int64)
            if len(excluded):
                indices = indices[~np.isin(indices, excluded)]
            if not len(indices):
                continue
            low, high = segment.min(axis=0), segment.max(axis=0)
            gap = np.maximum(0, np.maximum(low - index.maximum[indices], index.minimum[indices] - high))
            indices = indices[np.einsum("ij,ij->i", gap, gap) <= (minimum + 1e-12) ** 2]
            if len(indices):
                other = index.segments[indices]
                distances = segment_distances(
                    segment[0], segment[1] - segment[0], other[:, 0], other[:, 1] - other[:, 0]
                )
                minimum = min(minimum, float(np.min(distances)))
    return minimum


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("cap", [0.0, 0.003, 0.03, np.inf])
@pytest.mark.parametrize("seed", [812, 814, 891])
def test_batched_gap_matches_scalar_minimum_and_raw_budget_counts(fallback, cap, seed):
    rng = np.random.default_rng(seed)
    obstacles = rng.uniform(-0.025, 0.025, (48, 2, 3))
    queries = rng.uniform(-0.03, 0.03, (37, 2, 3))
    obstacles[0, 1] = obstacles[0, 0]
    queries[2, 1] = queries[2, 0]
    index = _SegmentDistanceIndex(obstacles)
    if fallback:
        index.tree = None
        index.fallback_tree = index._build_bounds(np.arange(len(obstacles)))
    excluded = (-1, 0, 2, 2, 14, 90)
    expected_counts, counts = [], []
    expected = _scalar_gap(index, queries, cap, excluded=excluded, consume=expected_counts.append)
    actual = index.gap(queries, cap, excluded=excluded, consume=counts.append)
    assert actual == expected
    assert counts == expected_counts


@pytest.mark.parametrize("fallback", [False, True])
def test_raw_pair_budget_exhausts_before_allocating_distance_pairs(fallback):
    obstacles = np.zeros((64, 2, 3))
    queries = np.zeros((32, 2, 3))
    index = _SegmentDistanceIndex(obstacles)
    if fallback:
        index.tree = None
        index.fallback_tree = index._build_bounds(np.arange(len(obstacles)))
    counts = []
    with pytest.raises(_DistanceBudgetExceeded, match="curve_distance_pair_budget_exceeded"):
        index.gap(queries, 0.003, excluded=range(64), consume=counts.append, maximum_pairs=100)
    assert not counts


def test_all_excluded_and_empty_neighbors_preserve_capped_gap():
    index = _SegmentDistanceIndex(np.array([[[0.0, 0, 0], [1, 0, 0]]]))
    close = np.array([[[0.0, 0, 0], [1, 0, 0]]])
    far = close + 20
    assert index.gap(close, 0.003, excluded=(0,)) == 0.003
    assert index.gap(far, 0.003) == 0.003
