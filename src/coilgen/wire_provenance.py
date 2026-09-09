from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

from .models import WirePathGeometry, WireSegmentOwner

_SegmentValue = TypeVar("_SegmentValue")


def nearest_source_segment_indices(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Map target intervals to source intervals by position and direction.

    Target segments are matched to the nearest source segment by midpoint,
    with a direction penalty to disambiguate nearby crossings. Reversing a
    segment does not change the match.
    """

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    source_segment_count = max(len(source) - 1, 0)
    target_segment_count = max(len(target) - 1, 0)
    if target_segment_count == 0:
        return np.zeros(0, dtype=np.int64)
    if source_segment_count == 0:
        raise ValueError("source_points must contain at least one segment")
    # Exact paths have an unambiguous interval map even when distinct visits
    # retrace the same coordinates. Geometric matching would lose that order.
    if np.array_equal(source, target):
        return np.arange(source_segment_count, dtype=np.int64)
    if np.array_equal(source[::-1], target):
        return np.arange(source_segment_count - 1, -1, -1, dtype=np.int64)

    source_starts = source[:-1]
    source_vectors = np.diff(source, axis=0)
    source_length_sq = np.einsum("ij,ij->i", source_vectors, source_vectors)
    valid = source_length_sq > 1e-24
    if not np.any(valid):
        return np.zeros(target_segment_count, dtype=np.int64)

    valid_indices = np.flatnonzero(valid)
    starts = source_starts[valid]
    vectors = source_vectors[valid]
    length_sq = source_length_sq[valid]
    unit_vectors = vectors / np.sqrt(length_sq)[:, None]
    source_scale_sq = float(np.median(length_sq)) if len(length_sq) else 1.0

    target_vectors = np.diff(target, axis=0)
    target_midpoints = 0.5 * (target[:-1] + target[1:])
    target_lengths = np.linalg.norm(target_vectors, axis=1)
    target_units = np.zeros_like(target_vectors)
    target_valid = target_lengths > 1e-12
    target_units[target_valid] = target_vectors[target_valid] / target_lengths[target_valid, None]

    nearest_indices = np.zeros(target_segment_count, dtype=np.int64)
    chunk_size = max(int(chunk_size), 1)
    for start_index in range(0, target_segment_count, chunk_size):
        stop_index = min(start_index + chunk_size, target_segment_count)
        midpoints = target_midpoints[start_index:stop_index]
        relative = midpoints[:, None, :] - starts[None, :, :]
        parameters = np.einsum("csi,si->cs", relative, vectors) / length_sq[None, :]
        parameters = np.clip(parameters, 0.0, 1.0)
        closest = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
        distance_sq = np.einsum("csi,csi->cs", midpoints[:, None, :] - closest, midpoints[:, None, :] - closest)

        directions = target_units[start_index:stop_index]
        alignment = np.abs(np.einsum("ci,si->cs", directions, unit_vectors))
        local_scale_sq = np.maximum(
            target_lengths[start_index:stop_index, None] ** 2,
            source_scale_sq,
        )
        scores = distance_sq + 1e-4 * (1.0 - alignment) ** 2 * local_scale_sq
        nearest = valid_indices[np.argmin(scores, axis=1)]
        nearest_indices[start_index:stop_index] = nearest
    return nearest_indices


def transfer_segment_values(
    source_points: np.ndarray,
    source_values: Sequence[_SegmentValue],
    target_points: np.ndarray,
    *,
    chunk_size: int = 256,
) -> list[_SegmentValue]:
    """Transfer categorical interval data to a geometrically modified path."""

    source_segment_count = max(len(np.asarray(source_points)) - 1, 0)
    target_segment_count = max(len(np.asarray(target_points)) - 1, 0)
    if target_segment_count == 0 or len(source_values) == 0:
        return []
    if len(source_values) != source_segment_count:
        raise ValueError("source_values must contain one entry per source segment")
    nearest = nearest_source_segment_indices(source_points, target_points, chunk_size=chunk_size)
    return [source_values[int(index)] for index in nearest]


def transfer_segment_owners(
    source_points: np.ndarray,
    source_owners: list[WireSegmentOwner],
    target_points: np.ndarray,
    *,
    chunk_size: int = 256,
) -> list[WireSegmentOwner]:
    """Transfer interval ownership to a geometrically modified centerline."""

    return transfer_segment_values(
        source_points,
        source_owners,
        target_points,
        chunk_size=chunk_size,
    )


def transfer_segment_geometry(
    source_points: np.ndarray,
    source_geometry: Sequence[WirePathGeometry | str],
    target_points: np.ndarray,
    *,
    chunk_size: int = 256,
) -> list[WirePathGeometry]:
    """Transfer explicit route geometry without inferring it from ownership."""

    transferred = transfer_segment_values(
        source_points,
        source_geometry,
        target_points,
        chunk_size=chunk_size,
    )
    return [WirePathGeometry(value) for value in transferred]


def dedupe_path_with_segment_owners(
    points: np.ndarray,
    owners: list[WireSegmentOwner],
    *,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, list[WireSegmentOwner]]:
    """Remove consecutive duplicate points while preserving segment provenance."""

    points = np.asarray(points, dtype=float)
    if owners and len(owners) != max(len(points) - 1, 0):
        raise ValueError("owners must contain one entry per source segment")
    if len(points) < 2:
        return points.copy(), []
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > float(tolerance)
    deduped = points[keep]
    if not owners:
        return deduped, []
    source_indices = np.flatnonzero(keep)[1:] - 1
    return deduped, [owners[int(index)] for index in source_indices]
