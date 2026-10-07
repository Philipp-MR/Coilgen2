"""BVH pruning preserves physical distance results and has a bounded fallback."""

import numpy as np
import pytest

from coilgen import geometry_collision, layout
from coilgen.config import CoilGenConfig


def test_shared_bvh_pair_budget_preserves_complete_default_result():
    starts = np.zeros((31, 3))
    ends = np.ones_like(starts)
    expected = np.column_stack(np.triu_indices(len(starts), 1))
    np.testing.assert_array_equal(geometry_collision._segment_bvh_candidate_pairs(starts, ends, 0), expected)
    np.testing.assert_array_equal(
        geometry_collision._segment_bvh_candidate_pairs(starts, ends, 0, max_pairs=len(expected)), expected
    )
    assert geometry_collision._segment_bvh_candidate_pairs(starts, ends, 0, max_pairs=len(expected) - 1) is None
    assert geometry_collision._segment_bvh_candidate_pairs(starts, ends, 0, max_pairs=0) is None


def test_geometric_bvh_pairs_preserve_distance_and_raw_budget_boundaries():
    starts = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [5, 0, 0], [2, 1, 0]], dtype=float)
    ends = starts + [1, 0, 0]
    expected = np.array([[0, 1], [0, 2], [1, 4], [2, 4]], dtype=np.int64)
    # Singleton leaves make the broad-phase pairs exact AABB-distance pairs.
    # Index ordering differs from centroid ordering and must remain stable.
    result = geometry_collision._segment_bvh_candidate_pairs(starts, ends, 1, leaf_size=1, max_pairs=4)
    np.testing.assert_array_equal(result, expected)
    assert geometry_collision._segment_bvh_candidate_pairs(starts, ends, 1, leaf_size=1, max_pairs=3) is None
    result = geometry_collision._segment_bvh_candidate_pairs(
        starts, ends, np.nextafter(1.0, 0.0), leaf_size=1, max_pairs=0
    )
    np.testing.assert_array_equal(result, np.zeros((0, 2), dtype=np.int64))


def _route(seed, closed):
    random = np.random.default_rng(seed)
    points = 0.01 * random.normal(size=(87, 3))
    # Highly unequal segment lengths and empty segments exercise the mapping
    # from valid BVH primitives back to the original topological indices.
    points[10] = points[9]
    points[20] = [1, 0, 0]
    points[21] = [-1, 0, 0]
    points[30:34] = 0.001 * np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])
    if closed:
        points[-1] = points[0]
    return points


@pytest.mark.parametrize("seed", [7, 103, 2026])
@pytest.mark.parametrize("closed", [False, True])
def test_bvh_contacts_and_minimum_match_bounded_all_pairs(seed, closed):
    points = _route(seed, closed)
    reference = CoilGenConfig(wire_distance_method="all_pairs", wire_distance_chunk_size=7, min_wire_distance_m=0.002)
    spatial = CoilGenConfig(wire_distance_method="spatial", wire_distance_chunk_size=13, min_wire_distance_m=0.002)
    expected = sorted(layout.find_wire_segment_contacts_3d(points, 0.002, config=reference))
    measured = sorted(layout.find_wire_segment_contacts_3d(points, 0.002, config=spatial))
    np.testing.assert_allclose(measured, expected, rtol=0, atol=1e-17)
    expected_minimum = layout.find_min_wire_segment_distance_3d(points, reference)
    actual_minimum = layout.find_min_wire_segment_distance_3d(points, spatial)
    assert actual_minimum[2] == pytest.approx(expected_minimum[2], abs=1e-17)


@pytest.mark.parametrize("method", ["spatial", "auto"])
def test_bvh_budget_exhaustion_falls_back_without_losing_contacts_or_minimum(monkeypatch, method):
    points = _route(9, True)
    reference = CoilGenConfig(wire_distance_method="all_pairs", wire_distance_chunk_size=5, min_wire_distance_m=0.002)
    expected_contacts = sorted(layout.find_wire_segment_contacts_3d(points, 0.002, reference))
    expected_minimum = layout.find_min_wire_segment_distance_3d(points, reference)
    original = geometry_collision._segment_bvh_candidate_pairs
    exhausted = []

    def small_budget(starts, ends, distance, *, max_pairs):
        assert 0 < max_pairs <= 2_000_000
        result = original(starts, ends, distance, max_pairs=1)
        exhausted.append(result is None)
        return result

    monkeypatch.setattr(geometry_collision, "_segment_bvh_candidate_pairs", small_budget)
    config = CoilGenConfig(
        wire_distance_method=method,
        wire_distance_spatial_min_segments=0,
        wire_distance_chunk_size=5,
        min_wire_distance_m=0.002,
    )
    assert sorted(layout.find_wire_segment_contacts_3d(points, 0.002, config)) == expected_contacts
    assert layout.find_min_wire_segment_distance_3d(points, config) == expected_minimum
    assert exhausted == [True, True]


def test_auto_retains_filtered_candidate_fraction_limit():
    points = _route(77, False)
    starts, directions, valid, closed = layout._wire_segment_data(points)
    kwargs = dict(max_distance=np.inf, config=CoilGenConfig(wire_distance_chunk_size=8), local_exclusion=0.0)
    assert layout._wire_segment_candidate_pair_chunks(starts, directions, valid, closed, **kwargs) is not None
    assert (
        layout._wire_segment_candidate_pair_chunks(
            starts, directions, valid, closed, **kwargs, max_candidate_fraction=0.02
        )
        is None
    )
