"""Wire segment distance, contact and broad-phase queries in physical coordinates."""

from __future__ import annotations
import math
import numpy as np
from .config import CoilGenConfig
from .geometry_kernels import segment_distance as _segment_segment_distance_3d, segment_distances
from .parallel import chunk_ranges, memory_limited_chunk_size, parallel_map, resolve_worker_count, should_parallelize


def _wire_clearance_failure_threshold(config: CoilGenConfig | None) -> float:
    if config is None:
        return 0.0
    return max(
        float(getattr(config, "min_wire_distance_m", 0.0)), float(getattr(config, "min_turn_distance", 0.0)), 0.0
    )


_COINCIDENT_VERTEX_TOLERANCE_M = 1e-9


def _merge_coincident_vertices(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop through-running vertices within 1 nm of the last kept one.

    Joins can leave sub-nanometre slivers. The segments on either side of such
    a sliver are not index-adjacent, so a through-running wire would report a
    false zero-distance "crossing". Where the wire reverses at the sliver, it is
    kept, so the two legs remain non-adjacent and a real hairpin is still found.
    A closing vertex of a closed path is always kept.
    """
    points = np.asarray(points, dtype=float)
    tolerance = _COINCIDENT_VERTEX_TOLERANCE_M
    if len(points) < 3 or np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance):
        return points, np.arange(len(points))
    closed = bool(np.linalg.norm(points[0] - points[-1]) <= 1e-8)
    kept = [0]
    for index in range(1, len(points)):
        anchor = points[kept[-1]]
        if np.linalg.norm(points[index] - anchor) > tolerance or (closed and index == len(points) - 1):
            kept.append(index)
            continue
        following = next(
            (later for later in range(index + 1, len(points)) if np.linalg.norm(points[later] - anchor) > tolerance),
            None,
        )
        if len(kept) >= 2 and following is not None:
            incoming = anchor - points[kept[-2]]
            outgoing = points[following] - anchor
            if float(incoming @ outgoing) <= 0.0:
                kept.append(index)
    kept_indices = np.asarray(kept, dtype=np.int64)
    return points[kept_indices], kept_indices


def _original_segment(kept: np.ndarray, merged_segment: int) -> int:
    """The real (last) original segment of a merged run of coincident vertices."""
    return int(kept[int(merged_segment) + 1]) - 1


def find_wire_segment_contacts_3d(
    points: np.ndarray,
    threshold: float,
    config: CoilGenConfig | None = None,
) -> list[tuple[int, int, float]]:
    """Find non-adjacent 3D wire segments closer than ``threshold``.

    Indices refer to segment starts in ``points``; coincident vertices are
    merged first so sub-nanometre slivers cannot fake a crossing.
    """

    merged, kept = _merge_coincident_vertices(points)
    return [
        (_original_segment(kept, seg_a), _original_segment(kept, seg_b), distance)
        for seg_a, seg_b, distance in _find_wire_segment_contacts_3d(merged, threshold, config)
    ]


def _find_wire_segment_contacts_3d(
    points: np.ndarray,
    threshold: float,
    config: CoilGenConfig | None = None,
) -> list[tuple[int, int, float]]:
    points = np.asarray(points, dtype=float)
    threshold = max(float(threshold), 0.0)
    n_segments = max(0, len(points) - 1)
    if n_segments < 2:
        return []
    starts, directions, valid_segments, closed = _wire_segment_data(points)
    local_exclusion = _wire_local_continuation_length(threshold, config)
    method = _wire_distance_method(config)
    if method == "spatial" or (method == "auto" and _should_attempt_spatial_wire_distance(n_segments, config)):
        contacts = _find_wire_segment_contacts_spatial(
            starts,
            directions,
            valid_segments,
            closed,
            threshold,
            config,
            local_exclusion=local_exclusion,
            max_candidate_fraction=0.02 if method == "auto" else None,
        )
        if contacts is not None:
            return contacts
    ranges = _wire_segment_chunk_ranges(n_segments, config)
    worker_count = resolve_worker_count(config) if config is not None else 1
    work_size = n_segments * n_segments
    if should_parallelize(work_size, config, worker_count=worker_count) and len(ranges) > 1:
        chunks = parallel_map(
            lambda segment_range: _wire_segment_contact_chunk(
                starts,
                directions,
                valid_segments,
                closed,
                segment_range,
                threshold,
                local_exclusion,
            ),
            ranges,
            worker_count=worker_count,
        )
        contacts: list[tuple[int, int, float]] = []
        for chunk in chunks:
            contacts.extend(chunk)
        return contacts

    contacts = []
    for segment_range in ranges:
        contacts.extend(
            _wire_segment_contact_chunk(
                starts,
                directions,
                valid_segments,
                closed,
                segment_range,
                threshold,
                local_exclusion,
            )
        )
    return contacts


def find_min_wire_segment_distance_3d(
    points: np.ndarray,
    config: CoilGenConfig | None = None,
) -> tuple[int, int, float]:
    """Return the closest non-adjacent pair of 3D wire segments.

    Indices refer to segment starts in ``points``; coincident vertices are
    merged first, as in :func:`find_wire_segment_contacts_3d`.
    """

    merged, kept = _merge_coincident_vertices(points)
    seg_a, seg_b, distance = _find_min_wire_segment_distance_3d(merged, config)
    if seg_a < 0:
        return seg_a, seg_b, distance
    return _original_segment(kept, seg_a), _original_segment(kept, seg_b), distance


def _find_min_wire_segment_distance_3d(
    points: np.ndarray,
    config: CoilGenConfig | None = None,
) -> tuple[int, int, float]:
    points = np.asarray(points, dtype=float)
    n_segments = max(0, len(points) - 1)
    if n_segments < 2:
        return -1, -1, 0.0
    starts, directions, valid_segments, closed = _wire_segment_data(points)
    local_exclusion = _wire_local_continuation_length(_wire_clearance_failure_threshold(config), config)
    method = _wire_distance_method(config)
    if method == "spatial" or (method == "auto" and _should_attempt_spatial_wire_distance(n_segments, config)):
        closest = _find_min_wire_segment_distance_spatial(
            starts,
            directions,
            valid_segments,
            closed,
            config,
            local_exclusion=local_exclusion,
            max_candidate_fraction=0.02 if method == "auto" else None,
        )
        if closest is not None:
            return closest
    ranges = _wire_segment_chunk_ranges(n_segments, config)
    worker_count = resolve_worker_count(config) if config is not None else 1
    work_size = n_segments * n_segments
    if should_parallelize(work_size, config, worker_count=worker_count) and len(ranges) > 1:
        chunk_results = parallel_map(
            lambda segment_range: _min_wire_segment_distance_chunk(
                starts,
                directions,
                valid_segments,
                closed,
                segment_range,
                local_exclusion,
            ),
            ranges,
            worker_count=worker_count,
        )
    else:
        chunk_results = [
            _min_wire_segment_distance_chunk(starts, directions, valid_segments, closed, segment_range, local_exclusion)
            for segment_range in ranges
        ]

    best_a = -1
    best_b = -1
    best_distance = math.inf
    for seg_a, seg_b, distance in chunk_results:
        if distance < best_distance:
            best_a = int(seg_a)
            best_b = int(seg_b)
            best_distance = float(distance)
    if not math.isfinite(best_distance):
        return -1, -1, 0.0
    return best_a, best_b, float(best_distance)


def _wire_segment_data(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    starts = np.asarray(points[:-1], dtype=float)
    directions = np.asarray(points[1:] - points[:-1], dtype=float)
    valid_segments = np.einsum("ij,ij->i", directions, directions) > 1e-24
    closed = bool(np.linalg.norm(points[0] - points[-1]) <= 1e-8)
    return starts, directions, valid_segments, closed


def _wire_distance_method(config: CoilGenConfig | None) -> str:
    method = str(getattr(config, "wire_distance_method", "spatial")).lower()
    if method not in {"spatial", "all_pairs"}:
        if method == "auto":
            return method
        raise ValueError("wire_distance_method must be 'spatial', 'all_pairs', or 'auto'.")
    return method


def _should_attempt_spatial_wire_distance(n_segments: int, config: CoilGenConfig | None) -> bool:
    threshold = max(int(getattr(config, "wire_distance_spatial_min_segments", 3000)), 0)
    return int(n_segments) >= threshold


def _wire_segment_chunk_ranges(n_segments: int, config: CoilGenConfig | None) -> list[tuple[int, int]]:
    configured_memory_mb = float(getattr(config, "parallel_max_chunk_memory_mb", 64.0))
    chunk_size = memory_limited_chunk_size(
        int(getattr(config, "wire_distance_chunk_size", 512)),
        n_segments,
        max_chunk_memory_mb=configured_memory_mb,
        bytes_per_pair=144,
    )
    return chunk_ranges(n_segments, chunk_size)


def _find_wire_segment_contacts_spatial(
    starts: np.ndarray,
    directions: np.ndarray,
    valid_segments: np.ndarray,
    closed: bool,
    threshold: float,
    config: CoilGenConfig | None,
    *,
    local_exclusion: float,
    max_candidate_fraction: float | None = None,
) -> list[tuple[int, int, float]] | None:
    pair_chunks = _wire_segment_candidate_pair_chunks(
        starts,
        directions,
        valid_segments,
        closed,
        max_distance=threshold,
        config=config,
        local_exclusion=local_exclusion,
        max_candidate_fraction=max_candidate_fraction,
    )
    if pair_chunks is None:
        return None
    worker_count = resolve_worker_count(config) if config is not None else 1
    total_pairs = sum(len(chunk) for chunk in pair_chunks)
    if total_pairs == 0:
        return []
    work_size = max(total_pairs, 1)
    if should_parallelize(work_size, config, worker_count=worker_count) and len(pair_chunks) > 1:
        chunks = parallel_map(
            lambda pairs: _wire_segment_contact_pair_chunk(starts, directions, pairs, threshold),
            pair_chunks,
            worker_count=worker_count,
        )
    else:
        chunks = [_wire_segment_contact_pair_chunk(starts, directions, pairs, threshold) for pairs in pair_chunks]
    contacts: list[tuple[int, int, float]] = []
    for chunk in chunks:
        contacts.extend(chunk)
    return contacts


def _find_min_wire_segment_distance_spatial(
    starts: np.ndarray,
    directions: np.ndarray,
    valid_segments: np.ndarray,
    closed: bool,
    config: CoilGenConfig | None,
    *,
    local_exclusion: float,
    max_candidate_fraction: float | None = None,
) -> tuple[int, int, float] | None:
    initial_best = _initial_wire_segment_distance_bound(starts, directions, valid_segments, closed, local_exclusion)
    if not math.isfinite(initial_best):
        return -1, -1, 0.0
    pair_chunks = _wire_segment_candidate_pair_chunks(
        starts,
        directions,
        valid_segments,
        closed,
        max_distance=initial_best,
        config=config,
        local_exclusion=local_exclusion,
        max_candidate_fraction=max_candidate_fraction,
    )
    if pair_chunks is None:
        return None
    worker_count = resolve_worker_count(config) if config is not None else 1
    total_pairs = sum(len(chunk) for chunk in pair_chunks)
    if total_pairs == 0:
        return -1, -1, 0.0
    work_size = max(total_pairs, 1)
    if should_parallelize(work_size, config, worker_count=worker_count) and len(pair_chunks) > 1:
        chunk_results = parallel_map(
            lambda pairs: _min_wire_segment_distance_pair_chunk(starts, directions, pairs),
            pair_chunks,
            worker_count=worker_count,
        )
    else:
        chunk_results = [_min_wire_segment_distance_pair_chunk(starts, directions, pairs) for pairs in pair_chunks]

    best_a = -1
    best_b = -1
    best_distance = math.inf
    for seg_a, seg_b, distance in chunk_results:
        if distance < best_distance:
            best_a = int(seg_a)
            best_b = int(seg_b)
            best_distance = float(distance)
    if not math.isfinite(best_distance):
        return -1, -1, 0.0
    return best_a, best_b, float(best_distance)


def _wire_segment_candidate_pair_chunks(
    starts: np.ndarray,
    directions: np.ndarray,
    valid_segments: np.ndarray,
    closed: bool,
    *,
    max_distance: float,
    config: CoilGenConfig | None,
    local_exclusion: float,
    max_candidate_fraction: float | None = None,
) -> list[np.ndarray] | None:
    from .geometry_collision import _segment_bvh_candidate_pairs

    n_segments = len(starts)
    if n_segments < 2 or not math.isfinite(max_distance):
        max_distance = math.inf
    mins, maxs = _wire_segment_aabbs(starts, directions)
    valid_indices = np.flatnonzero(valid_segments)
    if len(valid_indices) < 2:
        return []
    max_distance = max(float(max_distance), 0.0)
    max_distance_squared = max_distance * max_distance if math.isfinite(max_distance) else math.inf
    chunk_size = max(int(getattr(config, "wire_distance_chunk_size", 512)), 1)
    max_candidate_count = None
    if max_candidate_fraction is not None:
        all_pair_count = max((n_segments - 1) * (n_segments - 2) // 2, 0)
        max_candidate_count = max(int(float(max_candidate_fraction) * float(all_pair_count)), chunk_size)
    # Bound the raw leaf-pair list even for explicitly requested spatial
    # searches. Dense configurations use the existing memory-bounded all-pairs
    # fallback; the auto fraction also caps the final, filtered candidate set.
    raw_pair_budget = 2_000_000
    if max_candidate_count is not None:
        raw_pair_budget = min(raw_pair_budget, max(65_536, 8 * max_candidate_count))
    raw_pairs = _segment_bvh_candidate_pairs(
        mins[valid_indices], maxs[valid_indices], max_distance, max_pairs=raw_pair_budget
    )
    if raw_pairs is None:
        return None
    candidate_count = 0
    segment_length_prefix = _wire_segment_length_prefix(directions)
    segment_lengths = np.linalg.norm(directions, axis=1)
    chunks: list[np.ndarray] = []

    for start in range(0, len(raw_pairs), 65_536):
        pairs = valid_indices[raw_pairs[start : start + 65_536]]
        first, second = pairs.T
        allowed = second > first + 1
        if closed:
            allowed &= (first != 0) | (second != n_segments - 1)
        delta = np.maximum(np.maximum(mins[first] - maxs[second], mins[second] - maxs[first]), 0.0)
        bounds_squared = np.einsum("ij,ij->i", delta, delta)
        allowed &= bounds_squared <= max_distance_squared
        if local_exclusion > 0.0:
            gap = segment_length_prefix[second] - segment_length_prefix[first + 1]
            if closed:
                gap = np.minimum(
                    gap, segment_length_prefix[-1] - (segment_length_prefix[second + 1] - segment_length_prefix[first])
                )
            norm_product = segment_lengths[first] * segment_lengths[second]
            dot = np.einsum("ij,ij->i", directions[first], directions[second])
            cosine = np.divide(dot, norm_product, out=np.zeros_like(dot), where=norm_product > 1e-30)
            local = (gap <= local_exclusion) & (norm_product > 1e-30) & (cosine > -0.9)
            excluded = allowed & local
            allowed &= ~local
            # Preserve actual nonadjacent crossings even within the metric
            # continuation exclusion. AABB overlap alone is not sufficient.
            possible_zeros = np.flatnonzero(excluded & (bounds_squared <= 1e-18))
            if len(possible_zeros):
                a, b = pairs[possible_zeros].T
                allowed[possible_zeros] = segment_distances(starts[a], directions[a], starts[b], directions[b]) <= 1e-9
        pairs = pairs[allowed]
        candidate_count += len(pairs)
        if max_candidate_count is not None and candidate_count > max_candidate_count:
            return None
        if len(pairs):
            chunks.append(pairs)
    if not chunks:
        return []
    pairs = np.concatenate(chunks)
    # Retain the former sweep traversal order so tied nearest pairs and any
    # downstream contact grouping do not change with the broad-phase backend.
    axis = int(np.argmax(np.ptp(maxs[valid_indices], axis=0)))
    order = valid_indices[np.argsort(mins[valid_indices, axis], kind="stable")]
    ranks = np.empty(n_segments, dtype=np.int64)
    ranks[order] = np.arange(len(order))
    pair_ranks = ranks[pairs]
    ordering = np.lexsort((np.max(pair_ranks, axis=1), np.min(pair_ranks, axis=1)))
    pairs = pairs[ordering]
    return [pairs[offset : offset + chunk_size] for offset in range(0, len(pairs), chunk_size)]


def _wire_segment_pair_allowed(
    seg_a: int,
    seg_b: int,
    n_segments: int,
    closed: bool,
    *,
    starts: np.ndarray | None = None,
    directions: np.ndarray | None = None,
    local_exclusion: float = 0.0,
    segment_length_prefix: np.ndarray | None = None,
) -> bool:
    if seg_b <= seg_a + 1:
        return False
    if closed and seg_a == 0 and seg_b == n_segments - 1:
        return False
    if (
        directions is not None
        and local_exclusion > 0.0
        and _is_short_local_continuation(
            directions,
            seg_a,
            seg_b,
            local_exclusion,
            segment_length_prefix=segment_length_prefix,
            closed=closed,
        )
    ):
        # Metric continuation excludes ordinary neighboring bend clearance,
        # never a real self-crossing. Rescue only overlapping segment AABBs;
        # the exact kernel resolves skew segments whose boxes merely overlap.
        if starts is None:
            return False
        first, second = starts[seg_a], starts[seg_b]
        first_end, second_end = first + directions[seg_a], second + directions[seg_b]
        if (
            _aabb_distance_squared(
                np.minimum(first, first_end),
                np.maximum(first, first_end),
                np.minimum(second, second_end),
                np.maximum(second, second_end),
            )
            > 1e-18
        ):
            return False
        return bool(segment_distances(first, directions[seg_a], second, directions[seg_b]) <= 1e-9)
    return True


def _wire_segment_pairs_allowed(
    pairs: np.ndarray,
    n_segments: int,
    closed: bool,
    starts: np.ndarray,
    directions: np.ndarray,
    local_exclusion: float,
    segment_length_prefix: np.ndarray,
) -> np.ndarray:
    """``_wire_segment_pair_allowed`` for each row of ``pairs``, with the same arithmetic."""

    seg_a = pairs[:, 0]
    seg_b = pairs[:, 1]
    allowed = seg_b > seg_a + 1
    if closed:
        allowed &= ~((seg_a == 0) & (seg_b == n_segments - 1))
    if directions is None or not local_exclusion > 0.0 or not np.any(allowed):
        return allowed
    prefix = np.asarray(segment_length_prefix, dtype=float)
    rows = np.flatnonzero(allowed)
    a = seg_a[rows]
    b = seg_b[rows]
    gap = prefix[b] - prefix[a + 1]
    if closed:
        wrap = prefix[-1] - (prefix[b + 1] - prefix[a])
        gap = np.where(wrap < gap, wrap, gap)  # min(gap, wrap) keeps gap on ties
    direction_a = directions[a]
    direction_b = directions[b]
    norm_a = np.sqrt(
        direction_a[:, 0] * direction_a[:, 0]
        + direction_a[:, 1] * direction_a[:, 1]
        + direction_a[:, 2] * direction_a[:, 2]
    )
    norm_b = np.sqrt(
        direction_b[:, 0] * direction_b[:, 0]
        + direction_b[:, 1] * direction_b[:, 1]
        + direction_b[:, 2] * direction_b[:, 2]
    )
    norm_product = norm_a * norm_b
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = (
            direction_a[:, 0] * direction_b[:, 0]
            + direction_a[:, 1] * direction_b[:, 1]
            + direction_a[:, 2] * direction_b[:, 2]
        ) / norm_product
    continuation = ~(gap > float(local_exclusion)) & (norm_product > 1e-30) & (cosine > -0.9)
    # A continuation pair is rescued only when its segment boxes touch; the rare
    # touching pairs get the exact scalar test.
    first = starts[a]
    second = starts[b]
    first_end = first + direction_a
    second_end = second + direction_b
    delta = np.maximum(
        np.maximum(
            np.minimum(first, first_end) - np.maximum(second, second_end),
            np.minimum(second, second_end) - np.maximum(first, first_end),
        ),
        0.0,
    )
    box_distance_sq = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + delta[:, 2] * delta[:, 2]
    separated = continuation & (box_distance_sq > 1e-18)
    allowed[rows[separated]] = False
    touching = continuation & ~separated
    for row, pair_a, pair_b in zip(rows[touching], a[touching], b[touching]):
        allowed[row] = _wire_segment_pair_allowed(
            int(pair_a),
            int(pair_b),
            n_segments,
            closed,
            starts=starts,
            directions=directions,
            local_exclusion=local_exclusion,
            segment_length_prefix=segment_length_prefix,
        )
    return allowed


def _is_short_local_continuation(
    directions: np.ndarray,
    seg_a: int,
    seg_b: int,
    local_exclusion: float,
    *,
    segment_length_prefix: np.ndarray | None = None,
    closed: bool = False,
) -> bool:
    """Identify mesh-subdivided continuation, not a distinct hairpin arm."""

    prefix = (
        _wire_segment_length_prefix(directions)
        if segment_length_prefix is None
        else np.asarray(segment_length_prefix, dtype=float)
    )
    gap_length = float(prefix[seg_b] - prefix[seg_a + 1])
    if closed:
        wrap_gap_length = float(prefix[-1] - (prefix[seg_b + 1] - prefix[seg_a]))
        gap_length = min(gap_length, wrap_gap_length)
    if gap_length > float(local_exclusion):
        return False
    direction_a = directions[seg_a]
    direction_b = directions[seg_b]
    norm_product = float(np.linalg.norm(direction_a) * np.linalg.norm(direction_b))
    # Turns below about 155 degrees are one locally connected bend.  Near-
    # antiparallel reversals remain in the search so narrow hairpins still fail.
    return norm_product > 1e-30 and float(np.dot(direction_a, direction_b) / norm_product) > -0.9


def _wire_segment_length_prefix(directions: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.asarray(directions, dtype=float), axis=1)
    return np.concatenate((np.zeros(1, dtype=float), np.cumsum(lengths)))


def _wire_segment_aabbs(starts: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = starts + directions
    return np.minimum(starts, ends), np.maximum(starts, ends)


def _aabb_distance_squared(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    delta = np.maximum(np.maximum(min_a - max_b, min_b - max_a), 0.0)
    return float(np.dot(delta, delta))


def _initial_wire_segment_distance_bound(
    starts: np.ndarray,
    directions: np.ndarray,
    valid_segments: np.ndarray,
    closed: bool,
    local_exclusion: float,
) -> float:
    n_segments = len(starts)
    valid_indices = np.flatnonzero(valid_segments)
    if len(valid_indices) < 2:
        return math.inf
    best = math.inf
    segment_length_prefix = _wire_segment_length_prefix(directions)
    centers = starts + 0.5 * directions
    # Same candidate pairs, in the same order and the same 65536-pair chunks, as
    # the former nested loops; the pair filter runs on arrays.
    batches = []
    for axis in range(3):
        order = valid_indices[np.argsort(centers[valid_indices, axis], kind="mergesort")]
        window = min(16, max(len(order) - 1, 1))
        positions = np.arange(len(order))[:, None]
        following = positions + np.arange(1, window + 1)[None, :]
        inside = following < len(order)
        seg_i = np.broadcast_to(order[:, None], following.shape)[inside]
        seg_j = order[following[inside]]
        batches.append(np.stack((np.minimum(seg_i, seg_j), np.maximum(seg_i, seg_j)), axis=1))
    pairs = np.concatenate(batches).astype(np.int64, copy=False)
    pairs = pairs[
        _wire_segment_pairs_allowed(
            pairs, n_segments, closed, starts, directions, local_exclusion, segment_length_prefix
        )
    ]
    for begin in range(0, len(pairs), 65536):
        _seg_a, _seg_b, distance = _min_wire_segment_distance_pair_chunk(
            starts, directions, pairs[begin : begin + 65536]
        )
        if distance < best:
            best = float(distance)
    if math.isfinite(best):
        return best
    for seg_a in valid_indices:
        for seg_b in valid_indices:
            seg_a = int(seg_a)
            seg_b = int(seg_b)
            if _wire_segment_pair_allowed(
                seg_a,
                seg_b,
                n_segments,
                closed,
                starts=starts,
                directions=directions,
                local_exclusion=local_exclusion,
                segment_length_prefix=segment_length_prefix,
            ):
                return _segment_segment_distance_3d(
                    starts[seg_a],
                    starts[seg_a] + directions[seg_a],
                    starts[seg_b],
                    starts[seg_b] + directions[seg_b],
                )
    return math.inf


def _min_wire_segment_distance_pair_chunk(
    starts: np.ndarray,
    directions: np.ndarray,
    pairs: np.ndarray,
) -> tuple[int, int, float]:
    if len(pairs) == 0:
        return -1, -1, math.inf
    distances = _wire_segment_pair_index_distances(starts, directions, pairs)
    flat_ind = int(np.argmin(distances))
    distance = float(distances[flat_ind])
    if not math.isfinite(distance):
        return -1, -1, math.inf
    return int(pairs[flat_ind, 0]), int(pairs[flat_ind, 1]), distance


def _wire_segment_contact_pair_chunk(
    starts: np.ndarray,
    directions: np.ndarray,
    pairs: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    if len(pairs) == 0:
        return []
    distances = _wire_segment_pair_index_distances(starts, directions, pairs)
    contact_indices = np.flatnonzero(distances <= threshold)
    return [(int(pairs[ind, 0]), int(pairs[ind, 1]), float(distances[ind])) for ind in contact_indices]


def _wire_segment_pair_index_distances(
    starts: np.ndarray,
    directions: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    first, second = pairs[:, 0], pairs[:, 1]
    return segment_distances(starts[first], directions[first], starts[second], directions[second])


def _wire_segment_pair_distances(
    starts: np.ndarray,
    directions: np.ndarray,
    valid_segments: np.ndarray,
    closed: bool,
    segment_range: tuple[int, int],
    local_exclusion: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, stop = segment_range
    n_segments = len(starts)
    row_indices = np.arange(start, stop, dtype=np.int64)
    col_indices = np.arange(n_segments, dtype=np.int64)
    pair_mask = col_indices[None, :] >= row_indices[:, None] + 2
    if closed and start == 0:
        pair_mask[0, n_segments - 1] = False
    pair_mask &= valid_segments[row_indices, None] & valid_segments[None, :]
    if local_exclusion > 0.0:
        prefix = _wire_segment_length_prefix(directions)
        gap_lengths = prefix[col_indices][None, :] - prefix[row_indices + 1][:, None]
        if closed:
            wrap_gap_lengths = prefix[-1] - (prefix[col_indices + 1][None, :] - prefix[row_indices][:, None])
            gap_lengths = np.minimum(gap_lengths, wrap_gap_lengths)
        row_directions = directions[row_indices]
        direction_norms = np.linalg.norm(directions, axis=1)
        norm_products = direction_norms[row_indices, None] * direction_norms[None, :]
        direction_cosines = np.divide(
            row_directions @ directions.T,
            norm_products,
            out=np.full_like(norm_products, -1.0),
            where=norm_products > 1e-30,
        )
        local_pairs = (gap_lengths <= local_exclusion) & (direction_cosines > -0.9)
        excluded = pair_mask & local_pairs
        pair_mask &= ~local_pairs
        # Keep the existing bounded row chunk. AABB filtering needs only bool
        # matrices, and exact distances are evaluated only for overlapping
        # excluded pairs, not for every metric continuation.
        mins, maxs = _wire_segment_aabbs(starts, directions)
        for axis in range(3):
            excluded &= mins[row_indices, None, axis] <= maxs[None, :, axis] + 1e-9
            excluded &= mins[None, :, axis] <= maxs[row_indices, None, axis] + 1e-9
        rescue_rows, rescue_cols = np.nonzero(excluded)
        if len(rescue_rows):
            touching = (
                segment_distances(
                    starts[row_indices[rescue_rows]],
                    directions[row_indices[rescue_rows]],
                    starts[rescue_cols],
                    directions[rescue_cols],
                )
                <= 1e-9
            )
            pair_mask[rescue_rows[touching], rescue_cols[touching]] = True
    if not np.any(pair_mask):
        return row_indices, col_indices, np.full(pair_mask.shape, math.inf, dtype=float)

    pair_rows, pair_cols = np.nonzero(pair_mask)
    distances = np.full(pair_mask.shape, math.inf, dtype=float)
    distances[pair_rows, pair_cols] = segment_distances(
        starts[row_indices[pair_rows]],
        directions[row_indices[pair_rows]],
        starts[pair_cols],
        directions[pair_cols],
    )
    return row_indices, col_indices, distances


def _min_wire_segment_distance_chunk(
    starts: np.ndarray,
    directions: np.ndarray,
    valid_segments: np.ndarray,
    closed: bool,
    segment_range: tuple[int, int],
    local_exclusion: float,
) -> tuple[int, int, float]:
    row_indices, col_indices, distances = _wire_segment_pair_distances(
        starts,
        directions,
        valid_segments,
        closed,
        segment_range,
        local_exclusion,
    )
    flat_ind = int(np.argmin(distances))
    distance = float(distances.flat[flat_ind])
    if not math.isfinite(distance):
        return -1, -1, math.inf
    row, col = np.unravel_index(flat_ind, distances.shape)
    return int(row_indices[row]), int(col_indices[col]), distance


def _wire_segment_contact_chunk(
    starts: np.ndarray,
    directions: np.ndarray,
    valid_segments: np.ndarray,
    closed: bool,
    segment_range: tuple[int, int],
    threshold: float,
    local_exclusion: float,
) -> list[tuple[int, int, float]]:
    row_indices, col_indices, distances = _wire_segment_pair_distances(
        starts,
        directions,
        valid_segments,
        closed,
        segment_range,
        local_exclusion,
    )
    contact_rows, contact_cols = np.where(distances <= threshold)
    return [
        (int(row_indices[row]), int(col_indices[col]), float(distances[row, col]))
        for row, col in zip(contact_rows, contact_cols)
    ]


def _wire_local_continuation_length(clearance: float, config: CoilGenConfig | None = None) -> float:
    """Arc length within which non-reversing segments are one continuing wire.

    Two segments joined by a short third one are at most that short length
    apart, so without a floor the "minimum wire distance" collapses to the
    shortest segment whenever the configured clearance is zero. Parts of one
    continuing wire closer along the arc than the required centerline
    clearance overlap their own conductor body anyway, so that clearance is
    the floor. Near-reversals (hairpins) stay checked.
    """
    floor = 0.0
    if config is not None:
        try:
            from .manufacturing import _required_centerline_clearance, conductor_cross_section_points

            floor = _required_centerline_clearance(conductor_cross_section_points(config), config)
        except (ValueError, TypeError, AttributeError):
            floor = 0.5 * max(float(getattr(config, "conductor_cross_section_width", 0.0) or 0.0), 0.0)
    return max(2.0 * max(float(clearance), 0.0), float(floor) if np.isfinite(floor) else 0.0)
