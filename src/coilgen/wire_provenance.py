from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

from .models import WirePathGeometry, WireSegmentOwner
from .progress import ProgressReporter

_SegmentValue = TypeVar("_SegmentValue")


def ordered_source_stations(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map the midpoints of a resampled winding to distinct ordered source visits.

    Both polylines describe the same complete conductor. Repeated endpoints
    indicate closure, and the closed target may start on either side of its seam.
    The result indexes the supplied source even if the target traverses it backwards.
    """
    from scipy.ndimage import median_filter
    from scipy.spatial import cKDTree

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    nearest = nearest_source_segment_indices(source, target)
    if not len(nearest):
        return nearest, np.empty(0)
    sv = np.diff(source, axis=0)
    sl = np.linalg.norm(sv, axis=1)
    sa = np.r_[0.0, sl.cumsum()]
    length = sa[-1]
    tv = np.diff(target, axis=0)
    tl = np.linalg.norm(tv, axis=1)
    ta = np.r_[0.0, tl.cumsum()]
    mid = (target[:-1] + target[1:]) / 2
    if min(length, ta[-1]) <= 1e-12:
        return nearest, np.zeros(len(nearest))

    def projection(idx):
        relative = mid[:, None, :] - source[idx]
        return np.clip(np.einsum("ijk,ijk->ij", relative, sv[idx]) / np.maximum(sl[idx] ** 2, 1e-30), 0.0, 1.0)

    t = projection(nearest[:, None])[:, 0]
    station = sa[nearest] + t * sl[nearest]
    u = (ta[:-1] + tl / 2) * length / ta[-1]
    closed = np.linalg.norm(source[0] - source[-1]) <= 1e-9 and np.linalg.norm(target[0] - target[-1]) <= 1e-9
    if closed:
        # Select direction and phase from the whole winding; an arbitrary
        # coincident point at the seam is not a reliable alignment anchor.
        def alignment(sign):
            delta = station - sign * u
            phase = np.angle(np.exp(2j * np.pi * delta / length).mean()) * length / (2 * np.pi)
            delta = (delta - phase + length / 2) % length - length / 2
            return np.median(abs(delta)), sign, phase, delta

        _, sign, phase, residual = min((alignment(1), alignment(-1)), key=lambda row: row[0])
    else:
        forward = np.linalg.norm(source[0] - target[0]) + np.linalg.norm(source[-1] - target[-1])
        backward = np.linalg.norm(source[0] - target[-1]) + np.linalg.norm(source[-1] - target[0])
        sign = 1 if forward <= backward else -1
        phase = 0.0 if sign == 1 else length
        residual = station - phase - sign * u
    if sign < 0:
        source = source[::-1]
        sv = np.diff(source, axis=0)
        sl = sl[::-1]
        sa = np.r_[0.0, sl.cumsum()]
        nearest = len(sl) - 1 - nearest
        t = 1 - t
        station = length - station
        phase = length - phase
        residual = -residual
    # Fast path retains the original geometric answer on an unambiguous,
    # correctly ordered trace, including exact retraces with identical sampling.
    directed = station + length * np.rint((u + phase - station) / length) if closed else station
    if np.all(np.diff(directed) >= -1e-12):
        return (nearest if sign > 0 else len(sl) - 1 - nearest), (t if sign > 0 else 1 - t)
    center = np.median(residual)
    delta = residual - center
    anchor = abs(delta) <= max(0.008 * length, 5 * np.median(abs(delta)), 1e-9)
    delta = np.interp(u, u[anchor], delta[anchor])
    window = min(101, 2 * (len(u) // 90) + 3)
    expected = np.maximum.accumulate(u + phase + center + median_filter(delta, size=window, mode="nearest"))
    if not closed:
        expected = np.clip(expected, 0.0, length)
    smid = sa[:-1] + sl / 2
    xyz = (source[:-1] + source[1:]) / 2
    periods = np.arange(-1, 2) * length if closed else np.array([0.0])
    valid = np.flatnonzero(sl > 1e-12)
    ids = np.tile(valid, len(periods))
    cloud = np.column_stack((xyz[ids], 0.13 * np.concatenate([smid[valid] + k for k in periods])))
    e = expected % length if closed else expected
    near = cKDTree(cloud).query(np.column_stack((mid, 0.13 * e)), k=np.arange(1, min(12, len(ids)) + 1))[1]
    at = np.clip(np.searchsorted(sa, e, side="right") - 1, 0, len(sl) - 1)
    candidates = np.column_stack((ids[near], nearest, at, at))
    fracs = projection(candidates)
    # One state at the predicted arclength ensures a feasible monotonic
    # sequence even across a removed source spike or coincident samples.
    fracs[:, -1] = np.clip((e - sa[at]) / np.maximum(sl[at], 1e-30), 0.0, 1.0)
    pos = sa[candidates] + fracs * sl[candidates]
    if closed:
        pos += length * np.rint((expected[:, None] - pos) / length)
    xyz = source[candidates] + fracs[:, :, None] * sv[candidates]
    error = np.sum((xyz - mid[:, None, :]) ** 2, axis=2) + 1e-5 * (pos - expected[:, None]) ** 2
    parents = np.zeros(candidates.shape, dtype=np.int16)
    cost = error[0].copy()
    cols = np.arange(len(cost))
    for i in range(1, len(u)):
        travel = pos[i][None, :] - pos[i - 1][:, None]
        penalty = (0.03 * (travel - (u[i] - u[i - 1]))) ** 2
        penalty[travel < -1e-12] = np.inf
        scores = cost[:, None] + penalty
        parents[i] = np.argmin(scores, axis=0)
        cost = error[i] + scores[parents[i], cols]
    chosen = np.empty(len(u), dtype=int)
    chosen[-1] = np.argmin(cost)
    for i in range(len(u) - 1, 0, -1):
        chosen[i - 1] = parents[i, chosen[i]]
    rows = np.arange(len(u))
    ix = candidates[rows, chosen]
    f = fracs[rows, chosen]
    return (ix if sign > 0 else len(sl) - 1 - ix), (f if sign > 0 else 1 - f)


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
    task = ProgressReporter(label="Path provenance").task(
        "indexed target intervals", len(midpoints), indeterminate=True
    )
    # All targets at once, chunked to bound memory. Pair scores repeat the
    # per-target arithmetic element by element; ties keep the first source in
    # sorted order, as np.argmin over the sorted ball did.
    chunk = 2048
    for begin in range(0, len(midpoints), chunk):
        task.update(begin)
        rows = np.arange(begin, min(begin + chunk, len(midpoints)))
        seed = seeds[rows]
        seed_scores = _segment_match_pair_scores(
            midpoints[rows],
            directions[rows],
            target_length_sq[rows],
            starts[seed],
            vectors[seed],
            length_sq[seed],
            unit_vectors[seed],
            source_scale_sq,
        )
        radius = np.nextafter(np.sqrt(seed_scores) + half_length + rounding_guard, np.inf)
        balls = tree.query_ball_point(midpoints[rows], radius, return_sorted=True)
        counts = np.fromiter((len(ball) for ball in balls), dtype=np.int64, count=len(balls))
        sources = np.fromiter((index for ball in balls for index in ball), dtype=np.int64, count=int(counts.sum()))
        targets = np.repeat(rows, counts)
        scores = _segment_match_pair_scores(
            midpoints[targets],
            directions[targets],
            target_length_sq[targets],
            starts[sources],
            vectors[sources],
            length_sq[sources],
            unit_vectors[sources],
            source_scale_sq,
        )
        # First pair at each row's minimum. A NaN in a row makes its minimum NaN,
        # nothing equals it, and the row falls back to the per-target search
        # (whose argmin returns the first NaN). Empty rows fall back too.
        first = np.full(len(rows), -1, dtype=np.int64)
        nonempty = np.flatnonzero(counts > 0)
        if len(nonempty):
            offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))[nonempty]
            row_minimum = np.full(len(rows), np.nan)
            row_minimum[nonempty] = np.minimum.reduceat(scores, offsets)
            owner = np.repeat(np.arange(len(rows)), counts)
            hit = np.flatnonzero(scores == row_minimum[owner])
            hit_rows, first_hit = np.unique(owner[hit], return_index=True)
            first[hit_rows] = hit[first_hit]
        usable = first >= 0
        nearest[rows[usable]] = sources[first[usable]]
        for local in np.flatnonzero(~usable):
            nearest[rows[local]] = _indexed_segment_match_one(
                int(rows[local]),
                int(seeds[rows[local]]),
                tree,
                midpoints,
                directions,
                target_length_sq,
                starts,
                vectors,
                length_sq,
                unit_vectors,
                source_scale_sq,
                half_length,
                rounding_guard,
            )
    task.done()
    return nearest


def _segment_match_pair_scores(
    midpoints: np.ndarray,
    directions: np.ndarray,
    target_length_sq: np.ndarray,
    starts: np.ndarray,
    vectors: np.ndarray,
    length_sq: np.ndarray,
    unit_vectors: np.ndarray,
    source_scale_sq: float,
) -> np.ndarray:
    """``_segment_match_scores`` for matched rows (target i against source i).

    Same expressions and einsum contractions, which give the same value per
    element in either layout.
    """

    relative = midpoints - starts
    parameters = np.einsum("pi,pi->p", relative, vectors) / length_sq
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts + parameters[:, None] * vectors
    distance_sq = np.einsum("pi,pi->p", midpoints - closest, midpoints - closest)
    alignment = np.abs(np.einsum("pi,pi->p", directions, unit_vectors))
    local_scale_sq = np.maximum(target_length_sq, source_scale_sq)
    return distance_sq + 1e-4 * (1.0 - alignment) ** 2 * local_scale_sq


def _indexed_segment_match_one(
    target_index,
    seed,
    tree,
    midpoints,
    directions,
    target_length_sq,
    starts,
    vectors,
    length_sq,
    unit_vectors,
    source_scale_sq,
    half_length,
    rounding_guard,
) -> int:
    """The former per-target search; used for rows with NaN scores or an empty ball."""

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
    return int(indices[int(np.argmin(scores[0]))])


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
    task = ProgressReporter(label="Path provenance").task("target intervals", target_segment_count, indeterminate=True)
    for start_index in range(0, target_segment_count, chunk_size):
        task.update(start_index)
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
    task.done()
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
