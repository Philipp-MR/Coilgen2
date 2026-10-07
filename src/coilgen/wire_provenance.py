from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

from .models import WirePathGeometry, WireSegmentOwner

_SegmentValue = TypeVar("_SegmentValue")


def _segment_match_scores(
    midpoints: np.ndarray,
    directions: np.ndarray,
    target_length_sq: np.ndarray,
    starts: np.ndarray,
    vectors: np.ndarray,
    length_sq: np.ndarray,
    unit_vectors: np.ndarray,
    source_scale_sq: float,
) -> np.ndarray:
    """The original midpoint/direction objective, for any source subset."""

    relative = midpoints[:, None, :] - starts[None, :, :]
    parameters = np.einsum("csi,si->cs", relative, vectors) / length_sq[None, :]
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
    distance_sq = np.einsum("csi,csi->cs", midpoints[:, None, :] - closest, midpoints[:, None, :] - closest)
    alignment = np.abs(np.einsum("ci,si->cs", directions, unit_vectors))
    local_scale_sq = np.maximum(target_length_sq[:, None], source_scale_sq)
    return distance_sq + 1e-4 * (1.0 - alignment) ** 2 * local_scale_sq


def _indexed_segment_matches(
    midpoints: np.ndarray,
    directions: np.ndarray,
    target_length_sq: np.ndarray,
    starts: np.ndarray,
    vectors: np.ndarray,
    length_sq: np.ndarray,
    unit_vectors: np.ndarray,
    source_scale_sq: float,
) -> np.ndarray | None:
    """Prune only sources whose distance cannot beat a known full score.

    Every segment is inside the sphere around its midpoint with radius half
    its length. The direction penalty is nonnegative, so a segment beating a
    seed's full score must have its midpoint within sqrt(score) + half_length.
    A global maximum half-length makes the tree query conservative. Keeping
    candidates in source order preserves the original first-index tie rule.
    Each query allocates at most one source-sized candidate list.
    """

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None

    centers = starts + 0.5 * vectors
    half_length = 0.5 * float(np.sqrt(np.max(length_sq)))
    coordinate_scale = max(float(np.max(np.abs(centers))), float(np.max(np.abs(midpoints))), half_length)
    # Preserve the exhaustive fallback's behavior for finite coordinates whose
    # squared distances can overflow (cKDTree reports no nearest seed there).
    if coordinate_scale > np.sqrt(np.finfo(float).max) / 8.0:
        return None
    tree = cKDTree(centers)
    seeds = np.asarray(tree.query(midpoints, k=1)[1], dtype=np.int64)
    rounding_guard = 64.0 * np.finfo(float).eps * coordinate_scale
    nearest = np.empty(len(midpoints), dtype=np.int64)
    for target_index, seed in enumerate(seeds):
        target_slice = slice(target_index, target_index + 1)
        seed_slice = slice(seed, seed + 1)
        score = float(
            _segment_match_scores(
                midpoints[target_slice],
                directions[target_slice],
                target_length_sq[target_slice],
                starts[seed_slice],
                vectors[seed_slice],
                length_sq[seed_slice],
                unit_vectors[seed_slice],
                source_scale_sq,
            )[0, 0]
        )
        radius = np.nextafter(np.sqrt(score) + half_length + rounding_guard, np.inf)
        indices = np.asarray(tree.query_ball_point(midpoints[target_index], radius, return_sorted=True), dtype=np.int64)
        scores = _segment_match_scores(
            midpoints[target_slice],
            directions[target_slice],
            target_length_sq[target_slice],
            starts[indices],
            vectors[indices],
            length_sq[indices],
            unit_vectors[indices],
            source_scale_sq,
        )
        nearest[target_index] = indices[int(np.argmin(scores[0]))]
    return nearest


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

    # Small jobs are faster as one vectorized calculation. For large paths,
    # the exact broad phase avoids the quadratic pair array without changing
    # the objective, direction handling, or stable source-index tie breaking.
    if (
        len(starts) >= 512
        and len(starts) * target_segment_count >= 1_000_000
        and np.all(np.isfinite(starts))
        and np.all(np.isfinite(length_sq))
        and np.all(np.isfinite(target_midpoints))
        and np.all(np.isfinite(target_lengths))
    ):
        indexed = _indexed_segment_matches(
            target_midpoints,
            target_units,
            target_lengths**2,
            starts,
            vectors,
            length_sq,
            unit_vectors,
            source_scale_sq,
        )
        if indexed is not None:
            return valid_indices[indexed]

    nearest_indices = np.zeros(target_segment_count, dtype=np.int64)
    chunk_size = max(int(chunk_size), 1)
    for start_index in range(0, target_segment_count, chunk_size):
        stop_index = min(start_index + chunk_size, target_segment_count)
        scores = _segment_match_scores(
            target_midpoints[start_index:stop_index],
            target_units[start_index:stop_index],
            target_lengths[start_index:stop_index] ** 2,
            starts,
            vectors,
            length_sq,
            unit_vectors,
            source_scale_sq,
        )
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


def transfer_segment_provenance(
    source_points: np.ndarray,
    source_owners: Sequence[WireSegmentOwner],
    source_geometry: Sequence[WirePathGeometry | str],
    target_points: np.ndarray,
    *,
    chunk_size: int = 256,
) -> tuple[list[WireSegmentOwner], list[WirePathGeometry]]:
    """Transfer ownership and geometry through the same interval mapping.

    Either metadata sequence may be empty. Matching a modified centerline is
    the expensive part, so compute its direction-aware map only once when
    both kinds of provenance are available.
    """

    source_segment_count = max(len(np.asarray(source_points)) - 1, 0)
    target_segment_count = max(len(np.asarray(target_points)) - 1, 0)
    if target_segment_count == 0 or (len(source_owners) == 0 and len(source_geometry) == 0):
        return [], []
    for values in (source_owners, source_geometry):
        if len(values) and len(values) != source_segment_count:
            raise ValueError("source_values must contain one entry per source segment")
    nearest = nearest_source_segment_indices(source_points, target_points, chunk_size=chunk_size)
    owners = [source_owners[int(index)] for index in nearest] if len(source_owners) else []
    geometry = [WirePathGeometry(source_geometry[int(index)]) for index in nearest] if len(source_geometry) else []
    return owners, geometry


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
