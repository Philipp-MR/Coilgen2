from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar, cast


T = TypeVar("T")
U = TypeVar("U")


def available_cpu_count() -> int:
    """Return a conservative logical CPU count that works on all supported systems."""

    return max(int(os.cpu_count() or 1), 1)


def resolve_worker_count(config_or_count: object | None = None) -> int:
    """Resolve a worker count from CoilGenConfig or an integer.

    A value of 0 means all logical CPU cores. A value of 1 forces serial
    execution. Negative values reserve cores, e.g. -1 uses all but one.
    """

    if config_or_count is None:
        requested = 1
    elif isinstance(config_or_count, int):
        requested = int(config_or_count)
    else:
        requested = int(getattr(config_or_count, "parallel_worker_count", 1))

    cpu_count = available_cpu_count()
    if requested == 0:
        return cpu_count
    if requested < 0:
        return max(cpu_count + requested, 1)
    return max(requested, 1)


def parallel_min_work_size(config: object | None = None) -> int:
    return max(int(getattr(config, "parallel_min_work_size", 50000)), 0)


def should_parallelize(
    work_size: int | float, config: object | None = None, *, worker_count: int | None = None
) -> bool:
    workers = resolve_worker_count(config) if worker_count is None else max(int(worker_count), 1)
    return workers > 1 and float(work_size) >= float(parallel_min_work_size(config))


def chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    total = max(int(total), 0)
    chunk_size = max(int(chunk_size), 1)
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def balanced_ranges(total: int, worker_count: int, *, chunks_per_worker: int = 4) -> list[tuple[int, int]]:
    total = max(int(total), 0)
    if total == 0:
        return []
    chunk_count = min(total, max(int(worker_count), 1) * max(int(chunks_per_worker), 1))
    base = total // chunk_count
    extra = total % chunk_count
    ranges: list[tuple[int, int]] = []
    start = 0
    for chunk_ind in range(chunk_count):
        stop = start + base + (1 if chunk_ind < extra else 0)
        if stop > start:
            ranges.append((start, stop))
        start = stop
    return ranges


def parallel_map(
    function: Callable[[T], U],
    items: Sequence[T] | Iterable[T],
    *,
    worker_count: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[U]:
    item_list = list(items)
    if len(item_list) <= 1 or worker_count <= 1:
        results = []
        total = len(item_list)
        for ind, item in enumerate(item_list, start=1):
            results.append(function(item))
            if progress_callback is not None:
                progress_callback(ind, total)
        return results
    with ThreadPoolExecutor(
        max_workers=min(int(worker_count), len(item_list)), thread_name_prefix="coilgen"
    ) as executor:
        futures = {executor.submit(function, item): ind for ind, item in enumerate(item_list)}
        results: list[U | None] = [None] * len(item_list)
        completed = 0
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(item_list))
    return [cast(U, result) for result in results]


def memory_limited_chunk_size(
    requested_chunk_size: int,
    secondary_count: int,
    *,
    max_chunk_memory_mb: float,
    bytes_per_pair: int,
) -> int:
    """Return a chunk size capped by the approximate temporary array footprint."""

    requested = max(int(requested_chunk_size), 1)
    count = max(int(secondary_count), 1)
    max_bytes = max(float(max_chunk_memory_mb), 1.0) * 1024.0 * 1024.0
    limited = int(max_bytes // max(count * int(bytes_per_pair), 1))
    return max(min(requested, max(limited, 1)), 1)
