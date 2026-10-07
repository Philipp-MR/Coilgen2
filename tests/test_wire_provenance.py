from __future__ import annotations

import builtins
from unittest.mock import patch

import numpy as np
import pytest

from coilgen.models import WirePathGeometry, WireSegmentOwner
from coilgen.wire_provenance import (
    _indexed_segment_matches,
    nearest_source_segment_indices,
    transfer_segment_geometry,
    transfer_segment_owners,
    transfer_segment_provenance,
)


def _original_mapping(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Independent copy of the pre-index objective used for regression parity."""

    vectors = np.diff(source, axis=0)
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    valid = lengths_sq > 1e-24
    indices = np.flatnonzero(valid)
    if not len(indices):
        return np.zeros(len(target) - 1, dtype=np.int64)
    starts, vectors, lengths_sq = source[:-1][valid], vectors[valid], lengths_sq[valid]
    units = vectors / np.sqrt(lengths_sq)[:, None]
    target_vectors = np.diff(target, axis=0)
    target_lengths = np.linalg.norm(target_vectors, axis=1)
    directions = np.zeros_like(target_vectors)
    target_valid = target_lengths > 1e-12
    directions[target_valid] = target_vectors[target_valid] / target_lengths[target_valid, None]
    result = []
    for start in range(0, len(target) - 1, 37):
        stop = min(start + 37, len(target) - 1)
        midpoints = 0.5 * (target[start:stop] + target[start + 1 : stop + 1])
        relative = midpoints[:, None, :] - starts[None, :, :]
        parameters = np.clip(np.einsum("csi,si->cs", relative, vectors) / lengths_sq[None, :], 0, 1)
        closest = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
        delta = midpoints[:, None, :] - closest
        distance_sq = np.einsum("csi,csi->cs", delta, delta)
        alignment = np.abs(np.einsum("ci,si->cs", directions[start:stop], units))
        scale = np.maximum(target_lengths[start:stop, None] ** 2, np.median(lengths_sq))
        result.extend(indices[np.argmin(distance_sq + 1e-4 * (1 - alignment) ** 2 * scale, axis=1)])
    return np.asarray(result)


def _indexed_mapping(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    vectors = np.diff(source, axis=0)
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    valid = lengths_sq > 1e-24
    target_vectors = np.diff(target, axis=0)
    target_lengths = np.linalg.norm(target_vectors, axis=1)
    directions = np.zeros_like(target_vectors)
    target_valid = target_lengths > 1e-12
    directions[target_valid] = target_vectors[target_valid] / target_lengths[target_valid, None]
    result = _indexed_segment_matches(
        0.5 * (target[:-1] + target[1:]),
        directions,
        target_lengths**2,
        source[:-1][valid],
        vectors[valid],
        lengths_sq[valid],
        vectors[valid] / np.sqrt(lengths_sq[valid])[:, None],
        float(np.median(lengths_sq[valid])),
    )
    assert result is not None
    return np.flatnonzero(valid)[result]


@pytest.mark.parametrize("scale,translation", [(1.0, 0.0), (1e-8, 0.0), (1e6, 0.0), (1.0, 1e10)])
def test_indexed_direction_objective_matches_exhaustive_with_degenerate_segments(scale, translation):
    rng = np.random.default_rng(581)
    source = rng.normal(size=(145, 3)).cumsum(axis=0)
    source[51] = source[50]
    source[110:113] = source[34:37]
    target = source[rng.integers(0, len(source), 121)] + rng.normal(size=(121, 3)) * 0.03
    target[42] = target[41]
    source = source * scale + translation
    target = target * scale + translation

    np.testing.assert_array_equal(_indexed_mapping(source, target), _original_mapping(source, target))


def test_indexed_search_keeps_a_long_segment_with_a_distant_midpoint():
    source = np.asarray([[-100, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    target = np.asarray([[-0.1, 0, 0], [0.1, 0, 0]], dtype=float)

    np.testing.assert_array_equal(_indexed_mapping(source, target), [0])


def test_direction_penalty_can_choose_a_segment_that_is_not_geometrically_closest():
    source = np.asarray([[0, -1, 0], [0, 1, 0], [-1, 0, 0.001], [1, 0, 0.001]], dtype=float)
    target = np.asarray([[-0.1, 0, 0], [0.1, 0, 0]], dtype=float)

    np.testing.assert_array_equal(_indexed_mapping(source, target), [2])
    np.testing.assert_array_equal(_indexed_mapping(source, target[::-1]), [2])


def test_indexed_retrace_ties_select_first_source_interval():
    source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
    target = np.asarray([[0.2, 0, 0], [0.8, 0, 0]], dtype=float)

    np.testing.assert_array_equal(_indexed_mapping(source, target), [0])


def test_large_public_mapping_matches_original_and_does_not_mutate_inputs():
    rng = np.random.default_rng(241)
    source = rng.normal(size=(1025, 3)).cumsum(axis=0)
    target = source + rng.normal(size=source.shape) * 0.015
    snapshots = source.copy(), target.copy()
    with patch("coilgen.wire_provenance._indexed_segment_matches", wraps=_indexed_segment_matches) as indexed:
        actual = nearest_source_segment_indices(source, target)

    assert indexed.call_count == 1
    np.testing.assert_array_equal(actual, _original_mapping(source, target))
    np.testing.assert_array_equal(source, snapshots[0])
    np.testing.assert_array_equal(target, snapshots[1])


def test_large_mapping_preserves_numpy_only_installation():
    rng = np.random.default_rng(392)
    source = rng.normal(size=(1025, 3)).cumsum(axis=0)
    target = source + np.asarray([0, 0, 0.01])
    original_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("optional spatial dependency absent")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=no_scipy):
        actual = nearest_source_segment_indices(source, target)

    np.testing.assert_array_equal(actual, _original_mapping(source, target))


def test_extreme_finite_coordinates_retain_exhaustive_overflow_behavior():
    source = np.column_stack((np.arange(1025), np.zeros((1025, 2))))
    target = np.full_like(source, 1e160)
    with np.errstate(over="ignore", invalid="ignore"):
        actual = nearest_source_segment_indices(source, target)
        expected = _original_mapping(source, target)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_combined_transfer_maps_exact_retraces_once_and_converts_geometry(reverse):
    source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
    target = source[::-1].copy() if reverse else source.copy()
    owners = [WireSegmentOwner("contour_loop", loop_index=index) for index in range(3)]
    geometry = ["surface", WirePathGeometry.OFFSET_SURFACE, "free_3d"]
    with patch(
        "coilgen.wire_provenance.nearest_source_segment_indices", wraps=nearest_source_segment_indices
    ) as mapping:
        transferred_owners, transferred_geometry = transfer_segment_provenance(source, owners, geometry, target)

    assert mapping.call_count == 1
    assert transferred_owners == (owners[::-1] if reverse else owners)
    assert all(owner is expected for owner, expected in zip(transferred_owners, owners[::-1] if reverse else owners))
    expected_geometry = geometry[::-1] if reverse else geometry
    assert transferred_geometry == [WirePathGeometry(value) for value in expected_geometry]
    assert all(isinstance(value, WirePathGeometry) for value in transferred_geometry)


@pytest.mark.parametrize("with_owners,with_geometry", [(False, False), (True, False), (False, True), (True, True)])
def test_combined_transfer_matches_independent_legacy_transfers(with_owners, with_geometry):
    source = np.asarray([[0, -1, 0], [0, 1, 0], [-1, 0, 0.001], [1, 0, 0.001]], dtype=float)
    target = np.asarray([[-0.1, 0, 0], [0.1, 0, 0]], dtype=float)
    owners = [WireSegmentOwner("contour_loop", loop_index=index) for index in range(3)] if with_owners else []
    geometry = ["surface", "offset_surface", "free_3d"] if with_geometry else []
    expected = (
        transfer_segment_owners(source, owners, target),
        transfer_segment_geometry(source, geometry, target),
    )
    with patch(
        "coilgen.wire_provenance.nearest_source_segment_indices", wraps=nearest_source_segment_indices
    ) as mapping:
        actual = transfer_segment_provenance(source, owners, geometry, target)

    assert actual == expected
    assert mapping.call_count == int(with_owners or with_geometry)


def test_combined_transfer_rejects_inconsistent_metadata_before_matching():
    source = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    with patch("coilgen.wire_provenance.nearest_source_segment_indices") as mapping:
        with pytest.raises(ValueError, match="one entry per source segment"):
            transfer_segment_provenance(source, [], ["surface", "free_3d"], source)
    mapping.assert_not_called()


def test_empty_target_preserves_empty_transfer_even_with_nonmatching_metadata():
    source = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    assert transfer_segment_provenance(source, [], ["surface", "free_3d"], source[:1]) == ([], [])


def test_all_degenerate_sources_keep_first_owner():
    source = np.zeros((4, 3))
    target = np.ones((3, 3))
    owners = [WireSegmentOwner("contour_loop", loop_index=index) for index in range(3)]
    assert transfer_segment_provenance(source, owners, [], target) == ([owners[0], owners[0]], [])
