from __future__ import annotations

import math

import numpy as np

from .config import CoilGenConfig
from .models import CoilPart
from .parallel import chunk_ranges, memory_limited_chunk_size, parallel_map, resolve_worker_count, should_parallelize
from .progress import ProgressReporter

MU0 = 4.0 * math.pi * 1e-7
NEUMANN_COEFF = MU0 / (4.0 * math.pi)


def conductor_cross_section_area(config: CoilGenConfig) -> float:
    """Return the conductor cross-section area in square metres."""

    return _cross_section_area(config)


def equivalent_round_wire_radius(config: CoilGenConfig) -> float:
    """Return the round-wire radius with the same cross-section area."""

    return _equivalent_round_radius(_cross_section_area(config))


def calculate_inductance_by_coil_layout(
    parts: list[CoilPart],
    config: CoilGenConfig,
) -> tuple[list[CoilPart], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute inductance with a discretized Neumann/Biot-Savart integral.

    For thin conductors, the line-integral form is

        L = mu0/(4*pi) * integral integral dl dot dl' / |r-r'|

    The singular self contribution of each straight segment is regularized with
    a finite round-wire equivalent radius derived from the conductor cross
    section. By default the connected wire path is used. If
    ``config.inductance_source`` is ``"unconnected_contours"`` the extracted
    contour loops are used instead; ``"auto"`` selects contour loops whenever
    topology/routing postprocessing is skipped.
    """

    n_parts = len(parts)
    inductance_matrix = np.zeros((n_parts, n_parts), dtype=float)
    resistance = np.zeros(n_parts, dtype=float)
    lengths = np.zeros(n_parts, dtype=float)
    cross_sections = np.zeros(n_parts, dtype=float)
    source = _resolve_inductance_source(config)

    if config.skip_inductance_calculation:
        area = _cross_section_area(config)
        for part_ind, part in enumerate(parts):
            _starts, _ends, _centers, dl = _part_segments(part, source=source)
            lengths[part_ind] = float(np.linalg.norm(dl, axis=1).sum()) if len(dl) else 0.0
            cross_sections[part_ind] = area
            part.coil_resistance = 0.0
            part.coil_inductance = 0.0
            part.coil_length = float(lengths[part_ind])
            part.coil_cross_section = float(area)
        return parts, inductance_matrix, resistance, lengths, cross_sections

    area = _cross_section_area(config)
    equivalent_radius = _equivalent_round_radius(area)
    resistivity = _conductor_resistivity(config)
    worker_count = resolve_worker_count(config)
    reporter = ProgressReporter.from_config(config, label="Inductance")

    segment_sets = [_part_segments(part, source=source) for part in parts]
    for part_ind, part in enumerate(parts):
        starts, ends, centers, dl = segment_sets[part_ind]
        lengths[part_ind] = float(np.linalg.norm(dl, axis=1).sum()) if len(dl) else 0.0
        cross_sections[part_ind] = area
        resistance[part_ind] = resistivity * lengths[part_ind] / max(area, 1e-30)
        inductance_matrix[part_ind, part_ind] = _self_inductance(
            centers,
            dl,
            equivalent_radius,
            config,
            worker_count,
            reporter=reporter,
            progress_label=f"part {part_ind + 1}/{n_parts} self chunks",
        )

    for i in range(n_parts):
        for j in range(i + 1, n_parts):
            mutual = _mutual_inductance(
                segment_sets[i][2],
                segment_sets[i][3],
                segment_sets[j][2],
                segment_sets[j][3],
                softening_radius=equivalent_radius,
                config=config,
                worker_count=worker_count,
                reporter=reporter,
                progress_label=f"part {i + 1}-{j + 1} mutual chunks",
            )
            inductance_matrix[i, j] = mutual
            inductance_matrix[j, i] = mutual

    for part_ind, part in enumerate(parts):
        part.coil_resistance = float(resistance[part_ind])
        part.coil_inductance = float(inductance_matrix[part_ind, part_ind])
        part.coil_length = float(lengths[part_ind])
        part.coil_cross_section = float(cross_sections[part_ind])
    return parts, inductance_matrix, resistance, lengths, cross_sections


def _wire_segments(part: CoilPart) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if part.wire_path is None or len(part.wire_path.v) < 2:
        empty = np.zeros((0, 3), dtype=float)
        return empty, empty, empty, empty
    points = np.asarray(part.wire_path.v, dtype=float)
    starts = points[:-1]
    ends = points[1:]
    dl = ends - starts
    lengths = np.linalg.norm(dl, axis=1)
    keep = lengths > 1e-12
    starts = starts[keep]
    ends = ends[keep]
    dl = dl[keep]
    centers = (starts + ends) / 2.0
    return starts, ends, centers, dl


def _part_segments(part: CoilPart, *, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if source == "unconnected_contours":
        return _contour_segments(part)
    return _wire_segments(part)


def _contour_segments(part: CoilPart) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    for contour in part.contour_lines:
        points = _closed_contour_points(np.asarray(contour.v, dtype=float))
        if len(points) < 2:
            continue
        for _turn_ind in range(int(contour.turn_multiplicity)):
            starts_list.append(points[:-1])
            ends_list.append(points[1:])
    if not starts_list:
        empty = np.zeros((0, 3), dtype=float)
        return empty, empty, empty, empty
    starts = np.vstack(starts_list)
    ends = np.vstack(ends_list)
    dl = ends - starts
    lengths = np.linalg.norm(dl, axis=1)
    keep = lengths > 1e-12
    starts = starts[keep]
    ends = ends[keep]
    dl = dl[keep]
    centers = (starts + ends) / 2.0
    return starts, ends, centers, dl


def _closed_contour_points(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    if np.linalg.norm(points[0] - points[-1]) <= 1e-10:
        return points
    return np.vstack((points, points[0]))


def _resolve_inductance_source(config: CoilGenConfig) -> str:
    source = str(getattr(config, "inductance_source", "auto")).strip().lower()
    if source == "auto":
        return "unconnected_contours" if bool(config.skip_postprocessing) else "wire_path"
    if source in {"wire_path", "unconnected_contours"}:
        return source
    return "wire_path"


def _self_inductance(
    centers: np.ndarray,
    dl: np.ndarray,
    equivalent_radius: float,
    config: CoilGenConfig | None = None,
    worker_count: int = 1,
    reporter: ProgressReporter | None = None,
    progress_label: str = "self chunks",
) -> float:
    if len(dl) == 0:
        return 0.0
    off_diag = _mutual_inductance(
        centers,
        dl,
        centers,
        dl,
        ignore_diagonal=True,
        softening_radius=equivalent_radius,
        config=config,
        worker_count=worker_count,
        reporter=reporter,
        progress_label=progress_label,
    )
    segment_lengths = np.linalg.norm(dl, axis=1)
    radius = max(float(equivalent_radius), 1e-12)
    # Straight round-wire segment self term. Includes a small internal-inductance
    # correction for nonmagnetic conductors through the +0.25 term.
    safe_lengths = np.maximum(segment_lengths, radius * 1.0001)
    self_terms = (MU0 / (2.0 * math.pi)) * safe_lengths * (np.log((2.0 * safe_lengths) / radius) - 1.0 + 0.25)
    self_terms = np.maximum(self_terms, 0.0)
    self_only = float(np.sum(self_terms))
    total = float(off_diag + self_only)
    return max(total, self_only * 0.05, 0.0)


def _mutual_inductance(
    centers_a: np.ndarray,
    dl_a: np.ndarray,
    centers_b: np.ndarray,
    dl_b: np.ndarray,
    *,
    ignore_diagonal: bool = False,
    softening_radius: float = 0.0,
    chunk_size: int = 256,
    config: CoilGenConfig | None = None,
    worker_count: int = 1,
    reporter: ProgressReporter | None = None,
    progress_label: str = "mutual chunks",
) -> float:
    if len(dl_a) == 0 or len(dl_b) == 0:
        return 0.0
    configured_chunk_size = int(getattr(config, "inductance_chunk_size", chunk_size))
    configured_memory_mb = float(getattr(config, "parallel_max_chunk_memory_mb", 64.0))
    effective_chunk_size = memory_limited_chunk_size(
        configured_chunk_size,
        len(dl_b),
        max_chunk_memory_mb=configured_memory_mb,
        bytes_per_pair=40,
    )
    ranges = chunk_ranges(len(dl_a), effective_chunk_size)
    work_size = len(dl_a) * len(dl_b)
    task = reporter.task(progress_label, len(ranges)) if reporter is not None and reporter.enabled else None
    if should_parallelize(work_size, config, worker_count=worker_count) and len(ranges) > 1:
        totals = parallel_map(
            lambda segment_range: _mutual_inductance_chunk(
                centers_a,
                dl_a,
                centers_b,
                dl_b,
                segment_range,
                ignore_diagonal=ignore_diagonal,
                softening_radius=softening_radius,
            ),
            ranges,
            worker_count=worker_count,
            progress_callback=None if task is None else lambda done, total: task.update(done),
        )
        if task is not None:
            task.done()
        total = float(np.sum(totals))
    else:
        total = 0.0
        for chunk_ind, segment_range in enumerate(ranges, start=1):
            total += _mutual_inductance_chunk(
                centers_a,
                dl_a,
                centers_b,
                dl_b,
                segment_range,
                ignore_diagonal=ignore_diagonal,
                softening_radius=softening_radius,
            )
            if task is not None:
                task.update(chunk_ind)
        if task is not None:
            task.done()
    return NEUMANN_COEFF * total


def _mutual_inductance_chunk(
    centers_a: np.ndarray,
    dl_a: np.ndarray,
    centers_b: np.ndarray,
    dl_b: np.ndarray,
    segment_range: tuple[int, int],
    *,
    ignore_diagonal: bool = False,
    softening_radius: float = 0.0,
) -> float:
    start, stop = segment_range
    center_chunk = centers_a[start:stop]
    dl_chunk = dl_a[start:stop]
    r = center_chunk[:, None, :] - centers_b[None, :, :]
    distance = np.linalg.norm(r, axis=2)
    if softening_radius > 0:
        distance = np.sqrt(distance * distance + softening_radius * softening_radius)
    if ignore_diagonal and centers_a is centers_b and dl_a is dl_b:
        rows = np.arange(start, stop)
        valid = rows < len(dl_b)
        distance[np.arange(stop - start)[valid], rows[valid]] = np.inf
    dot_dl = np.einsum("ik,jk->ij", dl_chunk, dl_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = np.where(distance > 1e-15, dot_dl / distance, 0.0)
    return float(np.sum(contribution))


def _cross_section_area(config: CoilGenConfig) -> float:
    points = np.asarray(config.cross_sectional_points, dtype=float)
    if points.ndim == 2:
        if points.shape[0] == 2 and points.shape[1] != 2:
            points = points.T
        if points.shape[1] == 2 and len(points) >= 3:
            x = points[:, 0]
            y = points[:, 1]
            area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            if area > 0:
                return area
    return max(float(config.conductor_cross_section_width) * float(config.conductor_cross_section_height), 1e-30)


def _equivalent_round_radius(area: float) -> float:
    return math.sqrt(max(float(area), 1e-30) / math.pi)


def _conductor_resistivity(config: CoilGenConfig) -> float:
    # MATLAB's `specific_conductivity_conductor` default is 0.018e-6, which is
    # copper resistivity in ohm meter despite the historic variable name.
    value = float(config.specific_conductivity_conductor)
    return value if value > 0 else 1.724e-8
