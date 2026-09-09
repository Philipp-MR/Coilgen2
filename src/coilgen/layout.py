from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import CoilGenConfig
from .geometry_kernels import segment_distance as _segment_segment_distance_3d, segment_distances
from .contours import build_simple_wire_paths, contour_current_alignment, contour_evaluation_points
from .models import (
    CoilPart,
    ConnectedGroup,
    ContourLine,
    InterGroupCut,
    LoopGroup,
    WirePath,
    WirePathGeometry,
    WireSegmentOwner,
)
from .parallel import chunk_ranges, memory_limited_chunk_size, parallel_map, resolve_worker_count, should_parallelize
from .progress import ProgressReporter
from .surface_distance import SurfaceGeodesicSolver
from .surface_topology import (
    LoopTopologySignature,
    build_intrinsic_loop_inclusion_matrix,
    classify_loop_topology_signatures,
    validate_intrinsic_topology_inputs,
)
from .wire_provenance import transfer_segment_geometry, transfer_segment_owners


@dataclass
class NearestLoopPoint:
    v: np.ndarray


@dataclass
class MutualLoopDistance:
    distance: float
    near_point_a: NearestLoopPoint
    index_a: int
    near_point_b: NearestLoopPoint
    index_b: int


@dataclass
class _ContourWireSegmentData:
    starts: np.ndarray
    directions: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    segment_indices: np.ndarray
    contour_bounds_min: np.ndarray
    contour_bounds_max: np.ndarray

    @property
    def segment_count(self) -> int:
        return int(len(self.starts))


@dataclass
class LoopCut:
    v: np.ndarray
    segment_index: int


@dataclass
class LoopCutShape:
    cut_points: list[LoopCut]
    high_cut: LoopCut
    low_cut: LoopCut
    cut_width_high: float
    cut_width_low: float


@dataclass
class LoopCutPairOption:
    pair_index: int
    high_cut: LoopCut
    low_cut: LoopCut


@dataclass
class GroupConnectionCandidate:
    local_a: int
    local_b: int
    point_a_v: np.ndarray
    point_b_v: np.ndarray
    direct_distance: float
    geodesic_distance: float
    selection_distance: float
    used_geodesic_selection: bool
    used_clearance_fallback: bool = False
    loop_a_index: int = -1
    loop_b_index: int = -1


@dataclass
class GroupConnectionSearchPath:
    v: np.ndarray
    loop_indices: np.ndarray

    def __post_init__(self) -> None:
        self.v = np.asarray(self.v, dtype=float).reshape((-1, 3))
        self.loop_indices = np.asarray(self.loop_indices, dtype=np.int64).reshape(-1)
        if len(self.loop_indices) != len(self.v):
            raise ValueError("loop_indices must contain one entry per inter-group search point.")


@dataclass
class InterGroupCutAlignmentState:
    origin_v: np.ndarray
    axis_v: np.ndarray
    preferred_radial_v: np.ndarray
    distance_rel_tol: float
    candidate_factor: int
    return_corridor_aligned: bool = False


@dataclass
class ConnectorPathResult:
    path_v: np.ndarray
    used_direct_fallback: bool


@dataclass
class GroupFusionResult:
    fused: WirePath
    connection_path_v: np.ndarray
    closing_connection_path_v: np.ndarray
    cut1_v: np.ndarray
    cut2_v: np.ndarray
    selected_point1_v: np.ndarray
    selected_point2_v: np.ndarray
    direct_distance: float
    geodesic_distance: float
    connector_distance: float
    used_direct_fallback: bool
    path_length: float
    normal_shift_paths_v: tuple[np.ndarray, ...] = ()
    combined_return: bool = False
    group1_variant_name: str = ""
    group2_variant_name: str = ""
    loop1_index: int = -1
    loop2_index: int = -1


@dataclass
class GroupRouteVariant:
    name: str
    spiral_v: np.ndarray
    return_path_v: np.ndarray


@dataclass
class WithinGroupRailCandidate:
    side: str
    opened_loops: list[ContourLine]
    opening_cuts: list[np.ndarray]
    connected_group: ConnectedGroup
    score: float
    pair_indices: tuple[int, ...] = ()
    candidate_count: int = 1
    metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class _NormalShiftRoute:
    """One wire subroute that is allowed to pass above another conductor."""

    key: tuple[str, int, int]
    priority: int
    order: int
    segment_indices: tuple[int, ...]


_DIRECT_INTER_GROUP_CONNECTION_PATH_METHODS = frozenset(
    {"direct", "direct_3d", "straight", "3d", "3d_cut", "3d_cuts", "direct_3d_cut", "direct_3d_cuts"}
)
_RELAXED_GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS = frozenset(
    {"geodesic", "surface_geodesic_or_direct", "mesh_geodesic_or_direct", "relaxed_geodesic"}
)
_STRICT_GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS = frozenset(
    {"surface", "surface_geodesic", "mesh_geodesic", "strict_geodesic"}
)
_GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS = (
    _RELAXED_GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS | _STRICT_GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS
)


def find_minimal_contour_distance(
    parts: list[CoilPart],
    config: CoilGenConfig,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    """Find the closest surface-geodesic spacing between contour loops."""

    for part in parts:
        n_loops = len(part.contour_lines)
        part.contour_distance_matrix = np.zeros((n_loops, n_loops), dtype=float)
        if config.skip_calculation_min_winding_distance or n_loops < 2:
            part.minimal_contour_distance = 0.0
            part.pcb_track_width = 0.0
            continue

        distance_matrix = np.full((n_loops, n_loops), np.inf, dtype=float)
        np.fill_diagonal(distance_matrix, 0.0)
        min_distance = math.inf
        solver = SurfaceGeodesicSolver(part.mesh)
        distance_contours = [
            ContourLine(
                v=contour_evaluation_points(part.mesh, contour, config),
                potential=contour.potential,
                current_orientation=contour.current_orientation,
                turn_multiplicity=contour.turn_multiplicity,
            )
            for contour in part.contour_lines
        ]
        total_pairs = n_loops * (n_loops - 1) // 2
        task = progress.task(f"part contour-distance pairs ({n_loops} loops)", total_pairs) if progress else None
        completed_pairs = 0
        for ind_a in range(n_loops):
            for ind_b in range(ind_a + 1, n_loops):
                loop_distance = _find_min_geodesic_loop_distance(
                    solver,
                    distance_contours[ind_a],
                    distance_contours[ind_b],
                )
                distance_matrix[ind_a, ind_b] = loop_distance.distance
                distance_matrix[ind_b, ind_a] = loop_distance.distance
                min_distance = min(min_distance, loop_distance.distance)
                completed_pairs += 1
                if task is not None:
                    task.update(
                        completed_pairs,
                        message=f"loop pair {completed_pairs}/{total_pairs}; current min {min_distance:.4g} m",
                    )
        if task is not None:
            task.done()

        if not math.isfinite(min_distance):
            min_distance = 0.0
        part.contour_distance_matrix = distance_matrix
        part.minimal_contour_distance = float(min_distance)
        part.pcb_track_width = float(min_distance)
    return parts


def calculate_unconnected_contour_wire_distances(
    parts: list[CoilPart],
    config: CoilGenConfig | None = None,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    """Measure Euclidean centerline clearance in the unconnected contour set."""

    skip = bool(getattr(config, "skip_unconnected_contour_wire_distance", False)) if config is not None else False
    total_pairs = 0
    if not skip:
        total_pairs = sum(max(len(part.contour_lines) * (len(part.contour_lines) - 1) // 2, 0) for part in parts)
    task = (
        progress.task("unconnected contour wire-distance pairs", total_pairs) if progress and total_pairs > 0 else None
    )
    for part_index, part in enumerate(parts):
        diagnostics = _unconnected_contour_wire_distance_diagnostics(
            part,
            config=config,
            skip=skip,
            progress_task=task,
            part_index=part_index,
            part_count=len(parts),
        )
        part.unconnected_contour_wire_distance_diagnostics = diagnostics
        part.unconnected_contour_min_wire_distance = float(
            diagnostics.get("unconnected_contours_min_wire_distance", 0.0)
        )
    if task is not None:
        task.done()
    return parts


def _unconnected_contour_wire_distance_diagnostics(
    part: CoilPart,
    *,
    config: CoilGenConfig | None = None,
    skip: bool = False,
    progress_task: object | None = None,
    part_index: int = 0,
    part_count: int = 1,
) -> dict[str, float | int | bool]:
    contour_count = int(len(part.contour_lines))
    diagnostics: dict[str, float | int | bool] = {
        "unconnected_contours_wire_distance_skipped": bool(skip),
        "unconnected_contours_wire_distance_contour_count": contour_count,
        "unconnected_contours_wire_distance_pair_count": 0,
        "unconnected_contours_wire_distance_exact_pair_count": 0,
        "unconnected_contours_wire_distance_pruned_pair_count": 0,
        "unconnected_contours_wire_distance_exact_segment_pair_count": 0,
        "unconnected_contours_min_wire_distance": 0.0,
        "unconnected_contours_min_wire_loop_a": -1,
        "unconnected_contours_min_wire_loop_b": -1,
        "unconnected_contours_min_wire_segment_a": -1,
        "unconnected_contours_min_wire_segment_b": -1,
    }
    if skip or contour_count < 2:
        return diagnostics

    best_distance = math.inf
    best_loop_a = -1
    best_loop_b = -1
    best_seg_a = -1
    best_seg_b = -1
    pair_count = 0
    exact_pair_count = 0
    pruned_pair_count = 0
    exact_segment_pair_count = 0
    segment_data = [
        _contour_wire_segment_data(
            _ensure_closed_path_array(
                contour.v if config is None else contour_evaluation_points(part.mesh, contour, config),
                1.0e-10,
            )
        )
        for contour in part.contour_lines
    ]
    loop_pairs = _sorted_contour_wire_loop_pairs(segment_data)
    for lower_bound, loop_a, loop_b in loop_pairs:
        pair_count += 1
        if lower_bound >= best_distance:
            pruned_pair_count += 1
        else:
            exact_pair_count += 1
            seg_a, seg_b, distance, checked_segments = _contour_segment_data_min_wire_segment_distance_3d(
                segment_data[loop_a],
                segment_data[loop_b],
                upper_bound=best_distance,
            )
            exact_segment_pair_count += int(checked_segments)
            if distance < best_distance:
                best_distance = float(distance)
                best_loop_a = int(loop_a)
                best_loop_b = int(loop_b)
                best_seg_a = int(seg_a)
                best_seg_b = int(seg_b)
        if progress_task is not None:
            progress_task.advance(
                1,
                message=(
                    f"part {part_index + 1}/{max(part_count, 1)} loop pair {pair_count}/{len(loop_pairs)}; "
                    f"current min {_format_progress_distance(best_distance)}"
                ),
            )

    diagnostics["unconnected_contours_wire_distance_pair_count"] = int(pair_count)
    diagnostics["unconnected_contours_wire_distance_exact_pair_count"] = int(exact_pair_count)
    diagnostics["unconnected_contours_wire_distance_pruned_pair_count"] = int(pruned_pair_count)
    diagnostics["unconnected_contours_wire_distance_exact_segment_pair_count"] = int(exact_segment_pair_count)
    if math.isfinite(best_distance):
        diagnostics.update(
            {
                "unconnected_contours_min_wire_distance": float(best_distance),
                "unconnected_contours_min_wire_loop_a": best_loop_a,
                "unconnected_contours_min_wire_loop_b": best_loop_b,
                "unconnected_contours_min_wire_segment_a": best_seg_a,
                "unconnected_contours_min_wire_segment_b": best_seg_b,
            }
        )
    return diagnostics


def _contour_wire_segment_data(points: np.ndarray) -> _ContourWireSegmentData:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return _empty_contour_wire_segment_data()
    starts = points[:-1]
    ends = points[1:]
    directions = ends - starts
    valid = np.linalg.norm(directions, axis=1) > 1.0e-12
    if not np.any(valid):
        return _empty_contour_wire_segment_data()
    starts = starts[valid]
    ends = ends[valid]
    directions = directions[valid]
    segment_indices = np.flatnonzero(valid).astype(np.int64)
    bounds_min = np.minimum(starts, ends)
    bounds_max = np.maximum(starts, ends)
    return _ContourWireSegmentData(
        starts=starts,
        directions=directions,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        segment_indices=segment_indices,
        contour_bounds_min=np.min(bounds_min, axis=0),
        contour_bounds_max=np.max(bounds_max, axis=0),
    )


def _empty_contour_wire_segment_data() -> _ContourWireSegmentData:
    empty = np.zeros((0, 3), dtype=float)
    return _ContourWireSegmentData(
        starts=empty,
        directions=empty,
        bounds_min=empty,
        bounds_max=empty,
        segment_indices=np.zeros(0, dtype=np.int64),
        contour_bounds_min=empty,
        contour_bounds_max=empty,
    )


def _sorted_contour_wire_loop_pairs(
    segment_data: list[_ContourWireSegmentData],
) -> list[tuple[float, int, int]]:
    loop_pairs: list[tuple[float, int, int]] = []
    for loop_a, data_a in enumerate(segment_data):
        if data_a.segment_count < 1:
            continue
        for loop_b in range(loop_a + 1, len(segment_data)):
            data_b = segment_data[loop_b]
            if data_b.segment_count < 1:
                continue
            lower_bound = _axis_aligned_bounds_distance(
                data_a.contour_bounds_min,
                data_a.contour_bounds_max,
                data_b.contour_bounds_min,
                data_b.contour_bounds_max,
            )
            loop_pairs.append((lower_bound, loop_a, loop_b))
    loop_pairs.sort(key=lambda item: item[0])
    return loop_pairs


def _contour_segment_data_min_wire_segment_distance_3d(
    data_a: _ContourWireSegmentData,
    data_b: _ContourWireSegmentData,
    *,
    upper_bound: float = math.inf,
) -> tuple[int, int, float, int]:
    if data_a.segment_count < 1 or data_b.segment_count < 1:
        return -1, -1, math.inf, 0
    best_distance = float(upper_bound) if math.isfinite(float(upper_bound)) else math.inf
    best_seg_a = -1
    best_seg_b = -1
    checked_segment_pairs = 0
    starts = np.vstack((data_a.starts, data_b.starts))
    directions = np.vstack((data_a.directions, data_b.directions))
    segment_offset_b = data_a.segment_count
    block_size = max(1, min(64, data_a.segment_count))
    for block_start in range(0, data_a.segment_count, block_size):
        block_end = min(block_start + block_size, data_a.segment_count)
        lower_bounds = _axis_aligned_bounds_distance_matrix(
            data_a.bounds_min[block_start:block_end],
            data_a.bounds_max[block_start:block_end],
            data_b.bounds_min,
            data_b.bounds_max,
        )
        if math.isfinite(best_distance):
            flat_candidates = np.flatnonzero(lower_bounds.ravel() < best_distance)
        else:
            flat_candidates = np.arange(lower_bounds.size, dtype=np.int64)
        if len(flat_candidates) == 0:
            continue
        candidate_lower_bounds = lower_bounds.ravel()[flat_candidates]
        order = np.argsort(candidate_lower_bounds, kind="mergesort")
        flat_candidates = flat_candidates[order]
        candidate_lower_bounds = candidate_lower_bounds[order]
        rows, cols = np.unravel_index(flat_candidates, lower_bounds.shape)
        pair_indices = np.column_stack((rows + block_start, cols + segment_offset_b)).astype(np.int64, copy=False)
        for chunk_start in range(0, len(pair_indices), 65536):
            if math.isfinite(best_distance) and candidate_lower_bounds[chunk_start] >= best_distance:
                break
            chunk_end = min(chunk_start + 65536, len(pair_indices))
            if math.isfinite(best_distance):
                chunk_end = chunk_start + int(
                    np.searchsorted(candidate_lower_bounds[chunk_start:chunk_end], best_distance, side="left")
                )
            if chunk_end <= chunk_start:
                break
            chunk = pair_indices[chunk_start:chunk_end]
            distances = _wire_segment_pair_index_distances(starts, directions, chunk)
            checked_segment_pairs += int(len(chunk))
            if len(distances) == 0:
                continue
            local_index = int(np.argmin(distances))
            distance = float(distances[local_index])
            if distance < best_distance:
                best_distance = distance
                best_seg_a = int(data_a.segment_indices[int(chunk[local_index, 0])])
                best_seg_b = int(data_b.segment_indices[int(chunk[local_index, 1] - segment_offset_b)])
    return best_seg_a, best_seg_b, float(best_distance), checked_segment_pairs


def _axis_aligned_bounds_distance(
    bounds_min_a: np.ndarray,
    bounds_max_a: np.ndarray,
    bounds_min_b: np.ndarray,
    bounds_max_b: np.ndarray,
) -> float:
    separation = np.maximum(np.maximum(bounds_min_a - bounds_max_b, bounds_min_b - bounds_max_a), 0.0)
    return float(np.linalg.norm(separation))


def _axis_aligned_bounds_distance_matrix(
    bounds_min_a: np.ndarray,
    bounds_max_a: np.ndarray,
    bounds_min_b: np.ndarray,
    bounds_max_b: np.ndarray,
) -> np.ndarray:
    separation = np.maximum(
        np.maximum(
            bounds_min_a[:, None, :] - bounds_max_b[None, :, :],
            bounds_min_b[None, :, :] - bounds_max_a[:, None, :],
        ),
        0.0,
    )
    return np.linalg.norm(separation, axis=2)


def _format_progress_distance(distance: float) -> str:
    return f"{distance:.4g} m" if math.isfinite(float(distance)) else "n/a"


def _find_min_geodesic_loop_distance(
    solver: SurfaceGeodesicSolver,
    loop_a: ContourLine,
    loop_b: ContourLine,
) -> MutualLoopDistance:
    result = solver.distance_between_contours(loop_a, loop_b)
    return MutualLoopDistance(
        distance=float(result.distance),
        near_point_a=NearestLoopPoint(v=result.point_a),
        index_a=int(result.index_a),
        near_point_b=NearestLoopPoint(v=result.point_b),
        index_b=int(result.index_b),
    )


def find_min_mutual_loop_distance(
    loop_a: ContourLine,
    loop_b: ContourLine,
    only_point_flag: bool = False,
) -> MutualLoopDistance:
    """Calculate nearest positions and segment indices between two loops.

    This mirrors CoilGen's MATLAB postprocessing helper: by default it first
    finds the closest point on loop B to any vertex of loop A, then projects that
    point back onto loop A. ``only_point_flag`` restricts the search to vertices.
    """

    if len(loop_a.v) == 0 or len(loop_b.v) == 0:
        raise ValueError("Both loops must contain at least one point.")

    if only_point_flag or len(loop_a.v) < 2 or len(loop_b.v) < 2:
        return _nearest_vertex_pair(loop_a, loop_b)

    best_distance = math.inf
    best_b_index = 0
    best_b_v = loop_b.v[0]

    for segment_index, (start_v, end_v) in enumerate(zip(loop_b.v[:-1], loop_b.v[1:])):
        segment = end_v - start_v
        denom = float(np.dot(segment, segment))
        if denom <= 1e-30:
            t = np.zeros(len(loop_a.v), dtype=float)
        else:
            t = ((loop_a.v - start_v) @ segment) / denom
            t = np.clip(t, 0.0, 1.0)
        near_points = start_v + t[:, None] * segment
        distances = np.linalg.norm(near_points - loop_a.v, axis=1)
        vertex_index = int(np.argmin(distances))
        distance = float(distances[vertex_index])
        if distance < best_distance:
            best_distance = distance
            best_b_index = segment_index
            best_b_v = near_points[vertex_index]

    distance_a, best_a_index, best_a_v, best_a_t = _nearest_point_on_polyline(best_b_v, loop_a.v)

    return MutualLoopDistance(
        distance=float(distance_a),
        near_point_a=NearestLoopPoint(v=best_a_v),
        index_a=best_a_index,
        near_point_b=NearestLoopPoint(v=best_b_v),
        index_b=best_b_index,
    )


def find_min_loop_point_distance(point: np.ndarray, loop: np.ndarray) -> tuple[float, int]:
    """Return the closest distance and segment index from a point to a polyline."""

    distance, segment_index, _, _ = _nearest_point_on_polyline(point, _coerce_points(loop))
    return distance, segment_index


def generate_group_cutshapes(
    loop_group: LoopGroup,
    group_center: np.ndarray,
    mesh,
    cut_width: float,
    b_0_direction,
    cut_plane_definition: str = "nearest",
    cut_height_ratio: float = 0.5,
) -> list[LoopCutShape]:
    """Generate high/low rectangular opening cuts for all loops in a group."""

    if not loop_group.loops:
        return []
    group_center = np.asarray(group_center, dtype=float)
    loop_normal = calc_mean_loop_normal(loop_group)
    cut_plane_direction = _cut_plane_direction(
        loop_group,
        group_center,
        loop_normal,
        np.asarray(b_0_direction, dtype=float),
        cut_plane_definition,
    )

    shapes: list[LoopCutShape] = []
    first_pair_center: np.ndarray | None = None
    cut_direction: np.ndarray | None = None
    for loop_ind, loop in enumerate(loop_group.loops):
        cut_points = _loop_plane_cut_points(loop, cut_plane_direction, group_center)
        if len(cut_points) < 2:
            cut_points = _projection_fallback_cut_points(loop, cut_plane_direction)
        cut_points = _dedupe_loop_cuts(cut_points)
        if len(cut_points) < 2:
            raise ValueError("Could not generate two cut points for loop.")
        if len(cut_points) % 2:
            cut_points = _projection_fallback_cut_points(loop, cut_plane_direction)

        if loop_ind == 0:
            pair = _select_cut_pair_by_center(cut_points, group_center)
            cut_direction = pair[1].v - pair[0].v
            cut_direction = _normalize_vector(cut_direction, fallback=np.array([1.0, 0.0, 0.0]))
            high_cut, low_cut = _assign_high_low_cuts(pair, cut_direction)
            first_pair_center = (high_cut.v + low_cut.v) / 2.0
        else:
            pair = _select_cut_pair_by_center(cut_points, first_pair_center)
            high_cut, low_cut = _assign_high_low_cuts(pair, cut_direction)

        width_high = _calc_local_opening_gap(loop, high_cut.segment_index, cut_width)
        width_low = _calc_local_opening_gap(loop, low_cut.segment_index, cut_width)
        shapes.append(
            LoopCutShape(
                cut_points=cut_points,
                high_cut=high_cut,
                low_cut=low_cut,
                cut_width_high=width_high,
                cut_width_low=width_low,
            )
        )
    return shapes


def calc_mean_loop_normal(loop_group: LoopGroup) -> np.ndarray:
    """Estimate the mean 3D normal of all loops in a group."""

    normals = []
    last_center = np.zeros(3, dtype=float)
    for loop in loop_group.loops:
        if len(loop.v) < 3:
            continue
        center = loop.v.mean(axis=0)
        last_center = center
        loop_vecs = loop.v[1:] - loop.v[:-1]
        center_vecs = loop.v[:-1] - center
        normal = np.mean(np.cross(loop_vecs, center_vecs), axis=0)
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normals.append(normal / norm)
    if not normals:
        return np.array([0.0, 0.0, 1.0])
    loop_normal = _normalize_vector(np.mean(normals, axis=0), fallback=np.array([0.0, 0.0, 1.0]))
    if np.dot(loop_normal, last_center) < 0:
        loop_normal *= -1.0
    return loop_normal


def contour_orientation_diagnostics(part: CoilPart) -> list[dict[str, float | int | bool]]:
    """Return stream-current orientation diagnostics for extracted contour loops."""

    rows: list[dict[str, float | int | bool]] = []
    face_normals = _mesh_face_normals(part.mesh)
    for contour_ind, contour in enumerate(part.contour_lines):
        area_vector = _polyline_area_vector(contour.v)
        valid_faces = np.zeros(0, dtype=np.int64)
        if contour.segment_faces is not None and len(contour.segment_faces):
            valid_faces = np.asarray(contour.segment_faces, dtype=np.int64)
            valid_faces = valid_faces[(valid_faces >= 0) & (valid_faces < len(face_normals))]
        elif contour.surface_faces is not None and len(contour.surface_faces):
            valid_faces = np.asarray(contour.surface_faces, dtype=np.int64)
            valid_faces = valid_faces[(valid_faces >= 0) & (valid_faces < len(face_normals))]
        if len(valid_faces):
            raw_mean_normal = np.mean(face_normals[valid_faces], axis=0)
            surface_normal_coherence = float(np.linalg.norm(raw_mean_normal))
            mean_normal = _normalize_vector(raw_mean_normal, fallback=area_vector)
        else:
            surface_normal_coherence = 0.0
            mean_normal = _normalize_vector(area_vector, fallback=np.array([0.0, 0.0, 1.0]))
        signed_area = 0.5 * float(np.dot(area_vector, mean_normal))
        area_norm = float(np.linalg.norm(area_vector))
        area_normal_alignment = abs(float(np.dot(area_vector / area_norm, mean_normal))) if area_norm > 1e-15 else 0.0
        chirality_defined = bool(
            abs(signed_area) > 1e-15 and surface_normal_coherence > 1e-8 and area_normal_alignment > 1e-8
        )
        orientation_sign = 0 if not chirality_defined else 1 if signed_area > 0 else -1
        current_sign = (
            0 if abs(float(contour.current_orientation)) <= 1e-15 else 1 if contour.current_orientation > 0 else -1
        )
        stream_alignment, stream_alignment_scale = contour_current_alignment(part, contour)
        stream_alignment_normalized = (
            stream_alignment / stream_alignment_scale if stream_alignment_scale > 1e-30 else 0.0
        )
        stream_sign = 0 if abs(stream_alignment_normalized) <= 1e-10 else 1 if stream_alignment_normalized > 0 else -1
        rows.append(
            {
                "contour_index": contour_ind,
                "point_count": len(contour.v),
                "potential": float(contour.potential),
                "signed_area": signed_area,
                "absolute_area": abs(signed_area),
                "surface_normal_coherence": surface_normal_coherence,
                "area_normal_alignment": area_normal_alignment,
                "orientation_sign": orientation_sign,
                "current_orientation_sign": current_sign,
                "stream_current_alignment": float(stream_alignment),
                "stream_current_alignment_normalized": float(stream_alignment_normalized),
                "stream_current_orientation_sign": stream_sign,
                "chirality_defined": chirality_defined,
                "current_orientation_consistent": bool(
                    current_sign != 0 and stream_sign != 0 and current_sign == stream_sign
                ),
            }
        )
    return rows


def topological_loop_grouping(
    parts: list[CoilPart],
    config: CoilGenConfig | None = None,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    """Group contour loops by intrinsic surface topology.

    Contractible/nested loops use the MATLAB CoilGen containment semantics:
    loops with identical enclosing loops form a parallel level, and loop groups
    are made exclusive by subtracting nested subgroup loops.  In ``auto`` or
    ``parallel_surface`` mode, boundary-associated essential loops are grouped
    by surface boundary partition and oriented loop direction instead of 3D
    enclosure.  The topology tests are computed on the 3D surface, not from
    flattened polygons.
    """

    if isinstance(config, ProgressReporter) and progress is None:
        progress = config
        config = None

    part_task = progress.task("topology parts", len(parts)) if progress else None
    for part_ind, part in enumerate(parts):
        loops = part.contour_lines
        n_loops = len(loops)
        part.groups = []
        part.loop_groups = []
        part.group_levels = []
        part.level_positions = []
        part.loop_in_loop_matrix = np.zeros((n_loops, n_loops), dtype=bool)
        part.topology_diagnostics = {}
        if n_loops == 0:
            if part_task is not None:
                part_task.advance(1, message=f"part {part_ind + 1}/{len(parts)} has no contours")
            continue

        if progress is not None:
            progress.report(
                0.0,
                f"part {part_ind + 1}/{len(parts)}: validating {n_loops} contour loop(s) for intrinsic topology",
                force=True,
            )
        validation = validate_intrinsic_topology_inputs(part, check_separating=True, progress=progress)
        part.topology_diagnostics = validation.as_dict()
        validation.raise_for_errors()

        if progress is not None:
            progress.report(
                0.0,
                f"part {part_ind + 1}/{len(parts)}: building {n_loops} x {n_loops} loop inclusion matrix",
                force=True,
            )
        contains = build_intrinsic_loop_inclusion_matrix(part, validate=False, progress=progress)
        part.loop_in_loop_matrix = contains.copy()

        method = _topological_grouping_method(config)
        contains_groups = _containment_grouping_from_matrix(contains)
        signatures: list[LoopTopologySignature] = []
        use_parallel_surface = False
        if method != "containment":
            signatures = classify_loop_topology_signatures(part)
            part.topology_diagnostics.update(_loop_topology_signature_diagnostics(signatures))
            use_parallel_surface = method == "parallel_surface" or any(
                signature.boundary_associated for signature in signatures
            )

        if use_parallel_surface:
            grouping = _surface_parallel_loop_grouping(part, contains_groups, signatures)
            part.topology_diagnostics.update(grouping["diagnostics"])
        else:
            grouping = contains_groups
            part.topology_diagnostics["topological_grouping_method"] = "containment"

        part.loop_groups = [list(group) for group in grouping["loop_groups"]]
        part.group_levels = [list(level) for level in grouping["group_levels"]]
        part.level_positions = [list(positions) for positions in grouping["level_positions"]]

        higher_loops = [tuple(np.where(contains[loop_ind, :])[0].tolist()) for loop_ind in range(n_loops)]
        group_outer_loop_indices: list[list[int]] = []
        for loop_indices in part.loop_groups:
            ordered = sorted(loop_indices, key=lambda idx: len(higher_loops[idx]), reverse=True)
            outer_loop_indices = _topologically_outer_loop_indices(ordered, contains)
            group_outer_loop_indices.append(list(outer_loop_indices))
            part.groups.append(
                LoopGroup(
                    loops=[loops[idx] for idx in ordered],
                    loop_indices=ordered,
                    outer_loop_indices=outer_loop_indices,
                )
            )
        part.topology_diagnostics["group_outer_loop_indices"] = group_outer_loop_indices
        if part_task is not None:
            part_task.advance(
                1,
                message=(
                    f"part {part_ind + 1}/{len(parts)}: {n_loops} loops -> "
                    f"{len(part.groups)} group(s), {len(part.group_levels)} level(s)"
                ),
            )
    if part_task is not None:
        part_task.done()
    return parts


def calculate_group_centers(parts: list[CoilPart]) -> list[CoilPart]:
    for part in parts:
        if not part.groups:
            part.group_centers_v = np.zeros((0, 3), dtype=float)
            continue
        centers_v = []
        for group in part.groups:
            all_v = np.vstack([loop.v for loop in group.loops])
            centers_v.append(all_v.mean(axis=0))
        part.group_centers_v = np.asarray(centers_v, dtype=float)
    return parts


def interconnect_within_groups(
    parts: list[CoilPart],
    config: CoilGenConfig,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    use_geodesic_connections = _use_geodesic_inter_group_paths(config)
    require_surface_connections = _strict_geodesic_inter_group_paths(config)
    total_groups = sum(len(part.groups) for part in parts)
    task = progress.task("within-group routing", total_groups) if progress else None
    completed_groups = 0
    for part_ind, part in enumerate(parts):
        part.connected_groups = []
        geodesic_solver = SurfaceGeodesicSolver(part.mesh) if use_geodesic_connections else None
        if not part.groups:
            continue
        for group_ind, group in enumerate(part.groups):
            loop_count = len(group.loops)
            if task is not None:
                task.update(
                    completed_groups,
                    message=(
                        f"part {part_ind + 1}/{len(parts)} group {group_ind + 1}/{len(part.groups)}; "
                        f"loops={loop_count}; method={config.inter_group_connection_path_method}"
                    ),
                    force=completed_groups == 0,
                )
            if len(group.loops) == 1:
                loop = group.loops[0]
                connected = _connected_group_from_single_loop(loop)
                connected.outer_loop_indices = tuple(group.outer_loop_indices)
                group.opened_loops = [loop]
                group.opening_cuts_v = []
                part.connected_groups.append(connected)
                completed_groups += 1
                if task is not None:
                    task.update(completed_groups, message=f"group {group_ind + 1}/{len(part.groups)} is a single loop")
                continue

            opened_loops: list[ContourLine] = []
            opening_cuts: list[np.ndarray] = []
            group_center_v = (
                part.group_centers_v[group_ind]
                if part.group_centers_v is not None
                else np.vstack([loop.v for loop in group.loops]).mean(axis=0)
            )
            try:
                rectangle_cuts = generate_group_cutshapes(
                    group,
                    group_center_v,
                    part.mesh,
                    _within_group_cut_width(config),
                    config.b_0_direction,
                    cut_plane_definition="nearest",
                    cut_height_ratio=0.5,
                )
            except ValueError:
                for loop in group.loops:
                    cut_center = _select_loop_cut_center_3d(loop, group_center_v, config)
                    opened, cut_points = open_loop_with_3d_sphere(loop, cut_center, _within_group_cut_width(config))
                    opened_loops.append(opened)
                    opening_cuts.append(cut_points)
                group.opened_loops = opened_loops
                group.opening_cuts_v = opening_cuts
                group.within_group_cut_side = "fallback"
                group.within_group_cut_scores = {}
                connected = _connect_opened_loops(
                    opened_loops,
                    geodesic_solver,
                    allow_direct_fallback=not require_surface_connections,
                    corner_radius=0.0,
                    corner_step_deg=float(config.sweep_bend_arc_angle_deg),
                )
                connected.outer_loop_indices = tuple(group.outer_loop_indices)
                if require_surface_connections and len(connected.v) < 2:
                    raise ValueError(
                        "Unable to build a valid surface-geodesic within-group connector. "
                        "The route was not completed because a straight 3D connector would leave the support surface."
                    )
                group.within_group_cut_score = _within_group_return_path_score(connected.return_path_v)
                group.within_group_cut_pair_indices = ()
                group.within_group_cut_metrics = _within_group_return_path_metrics(connected.return_path_v)
                part.connected_groups.append(connected)
                completed_groups += 1
                if task is not None:
                    task.update(
                        completed_groups,
                        message=f"group {group_ind + 1}/{len(part.groups)} used fallback cut generation",
                    )
                continue
            rail = _select_within_group_cut_rail(
                group,
                rectangle_cuts,
                config,
                group_ind,
                geodesic_solver=geodesic_solver,
                allow_direct_fallback=not require_surface_connections,
            )
            if require_surface_connections and (not math.isfinite(rail.score) or len(rail.connected_group.v) < 2):
                raise ValueError(
                    "Unable to build a valid surface-geodesic within-group connector. "
                    "The route was not completed because a straight 3D connector would leave the support surface."
                )
            group.opened_loops = rail.opened_loops
            group.opening_cuts_v = rail.opening_cuts
            group.within_group_cut_side = rail.side
            group.within_group_cut_score = rail.score
            group.within_group_cut_scores = _within_group_cut_rail_scores(
                group,
                rectangle_cuts,
                config,
                geodesic_solver=geodesic_solver,
                allow_direct_fallback=not require_surface_connections,
            )
            group.within_group_cut_pair_indices = rail.pair_indices
            group.within_group_cut_metrics = dict(rail.metrics or {})
            rail.connected_group.outer_loop_indices = tuple(group.outer_loop_indices)
            part.connected_groups.append(rail.connected_group)
            completed_groups += 1
            if task is not None:
                task.update(
                    completed_groups,
                    message=(
                        f"group {group_ind + 1}/{len(part.groups)} selected {rail.side} rail; "
                        f"score={rail.score:.4g}; candidates={rail.candidate_count}"
                    ),
                )
    if task is not None:
        task.done()
    return parts


def interconnect_among_groups(
    parts: list[CoilPart],
    config: CoilGenConfig | None = None,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    """Create one wire path per coil part by fusing connected loop groups.

    The processing follows CoilGen's level hierarchy: deeper inner levels are
    fused first, parallel groups in a level are connected through their enclosing
    host group when one exists, and existing return-path points are ignored while
    choosing new cut locations.
    """

    config = CoilGenConfig() if config is None else config
    use_geodesic_connections = _use_geodesic_inter_group_paths(config)
    require_surface_connections = _strict_geodesic_inter_group_paths(config)
    total_connections = sum(max(len(part.connected_groups) - 1, 0) for part in parts)
    task = progress.task("inter-group routing", total_connections) if progress else None
    completed_connections = 0
    for part_ind, part in enumerate(parts):
        part.opening_cuts_among_groups = []
        if not part.connected_groups:
            build_simple_wire_paths([part])
            continue
        if len(part.connected_groups) == 1:
            group = part.connected_groups[0]
            part.wire_path = _rotate_wire_path_to_max_z(WirePath(v=group.v.copy()))
            continue
        if not part.group_levels:
            if task is not None:
                task.update(
                    completed_connections,
                    message=(
                        f"part {part_ind + 1}/{len(parts)} has no topology levels; "
                        f"greedy fuse {len(part.connected_groups)} group(s)"
                    ),
                    force=True,
                )
            geodesic_solver = SurfaceGeodesicSolver(part.mesh) if use_geodesic_connections else None
            part.wire_path = _greedy_fuse_connected_groups(
                part.connected_groups,
                geodesic_solver,
                allow_direct_fallback=not require_surface_connections,
            )
            completed_connections += max(len(part.connected_groups) - 1, 0)
            if task is not None:
                task.update(completed_connections, message=f"part {part_ind + 1}/{len(parts)} greedy fused")
            continue

        connected_group_buffer = [_copy_connected_group(group) for group in part.connected_groups]
        terminal_fusion_eligible = [True] * len(connected_group_buffer)
        geodesic_solver = SurfaceGeodesicSolver(part.mesh) if use_geodesic_connections else None
        all_return_points_v = _stack_return_points_v(connected_group_buffer)
        alignment_state = _inter_group_cut_alignment_state(part, config)
        hierarchy = [len(positions) for positions in part.level_positions]
        for hierarchy_rank in range(max(hierarchy), -1, -1):
            level_indices = [level_ind for level_ind, rank in enumerate(hierarchy) if rank == hierarchy_rank]
            for level_ind in level_indices:
                groups_to_connect = list(part.group_levels[level_ind])
                is_enclosing = [False] * len(groups_to_connect)
                if part.level_positions[level_ind]:
                    groups_to_connect.append(part.level_positions[level_ind][-1])
                    is_enclosing.append(True)

                while len(groups_to_connect) > 1:
                    if task is not None:
                        task.update(
                            completed_connections,
                            message=(
                                f"part {part_ind + 1}/{len(parts)} level {level_ind + 1}/{len(part.group_levels)} "
                                f"rank={hierarchy_rank}; selecting connector among {len(groups_to_connect)} group(s); "
                                f"candidate_count={_inter_group_candidate_count(config)}; "
                                f"method={config.inter_group_connection_path_method}"
                            ),
                            force=completed_connections == 0,
                        )
                    connection = _select_group_connection(
                        part,
                        connected_group_buffer,
                        groups_to_connect,
                        all_return_points_v,
                        config,
                        geodesic_solver,
                        alignment_state=alignment_state,
                    )
                    if connection is None:
                        if require_surface_connections:
                            raise ValueError(
                                "Unable to find a valid surface-geodesic inter-group connector. "
                                "The route was not completed because a straight 3D connector would leave the support surface."
                            )
                        break
                    local_a = connection.local_a
                    local_b = connection.local_b
                    point_a_v = connection.point_a_v
                    point_b_v = connection.point_b_v
                    loop_a_index = connection.loop_a_index
                    loop_b_index = connection.loop_b_index
                    group_a_index = groups_to_connect[local_a]
                    group_b_index = groups_to_connect[local_b]
                    opened_a, cut_a = _open_connected_group_with_3d_sphere(
                        connected_group_buffer[group_a_index],
                        point_a_v,
                        _inter_group_cut_width(config),
                    )
                    opened_b, cut_b = _open_connected_group_with_3d_sphere(
                        connected_group_buffer[group_b_index],
                        point_b_v,
                        _inter_group_cut_width(config),
                    )
                    (
                        fused,
                        connection_path_v,
                        closing_connection_path_v,
                        used_direct_fallback,
                    ) = _fuse_opened_wire_paths_with_connector_pair_diagnostics(
                        opened_a,
                        opened_b,
                        geodesic_solver,
                        allow_direct_fallback=not require_surface_connections,
                    )
                    if len(connection_path_v) < 2:
                        raise ValueError(
                            "Unable to fuse connected groups with a surface-geodesic connector. "
                            "The route was not completed because a straight 3D connector would leave the support surface."
                        )
                    combined_return = None
                    if terminal_fusion_eligible[group_a_index] and terminal_fusion_eligible[group_b_index]:
                        combined_return = _combined_return_group_fusion_candidate(
                            part,
                            connected_group_buffer[group_a_index],
                            connected_group_buffer[group_b_index],
                            group_a_index,
                            group_b_index,
                            config,
                            geodesic_solver,
                            allow_direct_fallback=not require_surface_connections,
                            alignment_state=alignment_state,
                        )
                    if combined_return is not None and not _combined_return_is_shorter(
                        combined_return,
                        fused,
                    ):
                        combined_return = None
                    if combined_return is not None:
                        fused = combined_return.fused
                        connection_path_v = combined_return.connection_path_v
                        closing_connection_path_v = combined_return.closing_connection_path_v
                        cut_a = combined_return.cut1_v
                        cut_b = combined_return.cut2_v
                        point_a_v = combined_return.selected_point1_v
                        point_b_v = combined_return.selected_point2_v
                        loop_a_index = combined_return.loop1_index
                        loop_b_index = combined_return.loop2_index
                        used_direct_fallback = combined_return.used_direct_fallback
                    connection_method = (
                        "geodesic" if geodesic_solver is not None and not used_direct_fallback else "direct_3d"
                    )
                    connector_distance = _path_length(connection_path_v)
                    geodesic_distance = (
                        combined_return.geodesic_distance
                        if combined_return is not None
                        else connection.geodesic_distance
                    )
                    if connection_method == "geodesic" and not math.isfinite(geodesic_distance):
                        geodesic_distance = connector_distance
                    part.opening_cuts_among_groups.append(
                        InterGroupCut(
                            cut1_v=cut_a,
                            cut2_v=cut_b,
                            connection_path_v=connection_path_v,
                            closing_connection_path_v=closing_connection_path_v,
                            group1_index=group_a_index,
                            group2_index=group_b_index,
                            loop1_index=loop_a_index,
                            loop2_index=loop_b_index,
                            selected_point1_v=point_a_v.reshape(1, 3),
                            selected_point2_v=point_b_v.reshape(1, 3),
                            direct_distance=(
                                combined_return.direct_distance
                                if combined_return is not None
                                else connection.direct_distance
                            ),
                            geodesic_distance=geodesic_distance,
                            connector_distance=connector_distance,
                            connection_method=connection_method,
                            used_direct_fallback=used_direct_fallback,
                            used_clearance_fallback=(
                                False if combined_return is not None else connection.used_clearance_fallback
                            ),
                            is_return_connection=combined_return is not None,
                            normal_shift_paths_v=(
                                list(combined_return.normal_shift_paths_v) if combined_return is not None else []
                            ),
                            group1_variant_name=(
                                combined_return.group1_variant_name if combined_return is not None else ""
                            ),
                            group2_variant_name=(
                                combined_return.group2_variant_name if combined_return is not None else ""
                            ),
                        )
                    )

                    if is_enclosing[local_a]:
                        target_local = local_a
                        remove_local = local_b
                    else:
                        target_local = local_b
                        remove_local = local_a
                    target_group_index = groups_to_connect[target_local]
                    merged_outer_loop_indices = tuple(
                        dict.fromkeys(
                            (
                                *_connected_group_outer_loop_indices(
                                    part,
                                    connected_group_buffer[group_a_index],
                                    group_a_index,
                                ),
                                *_connected_group_outer_loop_indices(
                                    part,
                                    connected_group_buffer[group_b_index],
                                    group_b_index,
                                ),
                            )
                        )
                    )
                    connected_group_buffer[target_group_index] = _replace_connected_group_path(
                        connected_group_buffer[target_group_index],
                        fused,
                        outer_loop_indices=merged_outer_loop_indices,
                    )
                    terminal_fusion_eligible[target_group_index] = False
                    del groups_to_connect[remove_local]
                    del is_enclosing[remove_local]
                    completed_connections += 1
                    if task is not None:
                        task.update(
                            completed_connections,
                            message=(
                                f"connected groups {group_a_index}->{group_b_index}; "
                                f"method={connection_method}; "
                                f"route={'combined_return' if combined_return is not None else 'opened_cut'}; "
                                f"connector={connector_distance:.4g} m; "
                                f"remaining in level={len(groups_to_connect)}"
                            ),
                        )

        largest_index = int(np.argmax([len(group.v) for group in connected_group_buffer]))
        largest = connected_group_buffer[largest_index]
        part.wire_path = _rotate_wire_path_to_max_z(WirePath(v=largest.v.copy()))
    for part in parts:
        _assign_wire_path_segment_owners(part)
    if task is not None:
        task.done()
    return parts


def _assign_wire_path_segment_owners(part: CoilPart, tolerance: float = 1e-9) -> None:
    """Label final route intervals and their explicit geometric contract."""

    if part.wire_path is None or len(part.wire_path.v) < 2:
        return

    candidate_owners: dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[WireSegmentOwner]] = {}

    def add_polyline(points_v: np.ndarray, owner: WireSegmentOwner) -> None:
        points = np.asarray(points_v, dtype=float)
        for start_v, end_v in zip(points[:-1], points[1:]):
            if np.linalg.norm(end_v - start_v) <= 1e-12:
                continue
            candidate_owners.setdefault(_segment_geometry_key(start_v, end_v, tolerance), []).append(owner)

    for group_index, group in enumerate(part.connected_groups):
        add_polyline(group.v, WireSegmentOwner(kind="contour_group", group_index=group_index))
        if group_index < len(part.groups):
            loop_group = part.groups[group_index]
            for local_loop_index, opened_loop in enumerate(loop_group.opened_loops):
                loop_index = (
                    int(loop_group.loop_indices[local_loop_index])
                    if local_loop_index < len(loop_group.loop_indices)
                    else int(local_loop_index)
                )
                add_polyline(
                    opened_loop.v,
                    WireSegmentOwner(
                        kind="contour_loop",
                        group_index=group_index,
                        loop_index=loop_index,
                    ),
                )
    for connector_index, cut in enumerate(part.opening_cuts_among_groups):
        common = {
            "kind": "inter_group_connector",
            "connector_index": connector_index,
            "group1_index": cut.group1_index,
            "group2_index": cut.group2_index,
        }
        add_polyline(cut.connection_path_v, WireSegmentOwner(**common, connector_role="primary"))
        add_polyline(cut.closing_connection_path_v, WireSegmentOwner(**common, connector_role="closing"))

    owners: list[WireSegmentOwner | None] = []
    for start_v, end_v in zip(part.wire_path.v[:-1], part.wire_path.v[1:]):
        matches = candidate_owners.get(_segment_geometry_key(start_v, end_v, tolerance), [])
        connector_matches = [owner for owner in matches if owner.kind == "inter_group_connector"]
        loop_matches = [owner for owner in matches if owner.kind == "contour_loop"]
        owners.append((connector_matches or loop_matches or matches or [None])[0])

    next_connector_index = len(part.opening_cuts_among_groups)
    unknown_runs = _false_runs([owner is not None for owner in owners])
    for run_start, run_stop in unknown_runs:
        previous = owners[run_start - 1] if run_start > 0 else None
        following = owners[run_stop] if run_stop < len(owners) else None
        previous_group = _owner_group_index(previous)
        following_group = _owner_group_index(following)
        if previous_group >= 0 and following_group >= 0 and previous_group != following_group:
            owner = WireSegmentOwner(
                kind="inter_group_connector",
                connector_index=next_connector_index,
                group1_index=previous_group,
                group2_index=following_group,
                connector_role="bridge",
            )
            next_connector_index += 1
        elif previous is not None and previous == following:
            owner = previous
        elif previous is not None:
            owner = previous
        elif following is not None:
            owner = following
        else:
            owner = WireSegmentOwner(kind="contour_group", group_index=0)
        owners[run_start:run_stop] = [owner] * (run_stop - run_start)

    final_owners = [owner for owner in owners if owner is not None]
    part.wire_path = WirePath(
        v=part.wire_path.v.copy(),
        segment_owners=final_owners,
        segment_geometry=[_wire_segment_geometry(part, owner) for owner in final_owners],
    )


def _wire_segment_geometry(part: CoilPart, owner: WireSegmentOwner) -> WirePathGeometry:
    if owner.kind in {"contour_group", "contour_loop"}:
        return WirePathGeometry.SURFACE
    if owner.kind != "inter_group_connector":
        return WirePathGeometry.FREE_3D
    connector_index = int(owner.connector_index)
    if not 0 <= connector_index < len(part.opening_cuts_among_groups):
        return WirePathGeometry.FREE_3D
    connector = part.opening_cuts_among_groups[connector_index]
    if connector.connection_method == "geodesic" and not connector.used_direct_fallback:
        return WirePathGeometry.SURFACE
    return WirePathGeometry.FREE_3D


def _segment_geometry_key(
    start_v: np.ndarray,
    end_v: np.ndarray,
    tolerance: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    scale = 1.0 / max(float(tolerance), 1e-15)
    start = tuple(int(value) for value in np.rint(np.asarray(start_v, dtype=float) * scale))
    end = tuple(int(value) for value in np.rint(np.asarray(end_v, dtype=float) * scale))
    return (start, end) if start <= end else (end, start)


def _owner_group_index(owner: WireSegmentOwner | None) -> int:
    if owner is None:
        return -1
    if owner.kind in {"contour_group", "contour_loop"}:
        return owner.group_index
    return -1


def _false_runs(mask: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = -1
    for index, value in enumerate([*mask, True]):
        if not value and start < 0:
            start = index
        elif value and start >= 0:
            runs.append((start, index))
            start = -1
    return runs


def _combined_return_is_shorter(
    combined_return: GroupFusionResult,
    opened_cut_route: WirePath,
) -> bool:
    """Use a shared return corridor only when it reduces conductor length.

    Both candidates already contain the complete closed path through the two
    groups.  Comparing complete centerlines prevents a locally valid return
    terminal pair from overriding a much more compact opened-cut connection.
    Ties favor the ordinary route because it has fewer connector sections.
    """

    combined_length = float(combined_return.path_length)
    opened_cut_length = _path_length(opened_cut_route.v)
    if not math.isfinite(combined_length) or not math.isfinite(opened_cut_length):
        return False
    comparison_tolerance = max(1.0e-12, 1.0e-9 * opened_cut_length)
    return combined_length < opened_cut_length - comparison_tolerance


def _use_geodesic_inter_group_paths(config: CoilGenConfig) -> bool:
    method = str(config.inter_group_connection_path_method).strip().lower()
    if method in _DIRECT_INTER_GROUP_CONNECTION_PATH_METHODS:
        return False
    if method in _GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS:
        return True
    valid = sorted(_DIRECT_INTER_GROUP_CONNECTION_PATH_METHODS | _GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS)
    raise ValueError(
        "inter_group_connection_path_method must be one of "
        f"{', '.join(valid)}; got {config.inter_group_connection_path_method!r}."
    )


def _strict_geodesic_inter_group_paths(config: CoilGenConfig) -> bool:
    method = str(config.inter_group_connection_path_method).strip().lower()
    if method in _STRICT_GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS:
        return True
    if method in _DIRECT_INTER_GROUP_CONNECTION_PATH_METHODS | _RELAXED_GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS:
        return False
    valid = sorted(_DIRECT_INTER_GROUP_CONNECTION_PATH_METHODS | _GEODESIC_INTER_GROUP_CONNECTION_PATH_METHODS)
    raise ValueError(
        "inter_group_connection_path_method must be one of "
        f"{', '.join(valid)}; got {config.inter_group_connection_path_method!r}."
    )


def _inter_group_cut_alignment_state(
    part: CoilPart,
    config: CoilGenConfig | None,
) -> InterGroupCutAlignmentState | None:
    if config is None:
        return None
    mode = str(getattr(config, "inter_group_cut_alignment", "auto")).strip().lower()
    if mode in {"off", "none", "disabled", "false", "0"}:
        return None
    forced = mode in {"on", "true", "1", "cylinder", "cylindrical", "cylinder_seam", "cylindrical_seam", "seam"}
    if not forced and mode != "auto":
        return None
    if not forced and not bool(getattr(config, "surface_is_cylinder_flag", False)):
        return None

    axis_result = _mesh_boundary_axis(part.mesh)
    if axis_result is None:
        if not forced:
            return None
        origin_v = np.asarray(part.mesh.vertices, dtype=float).mean(axis=0)
        axis_v = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        origin_v, axis_v = axis_result

    return_corridor = _preferred_return_corridor_radial(part, origin_v, axis_v)
    if return_corridor is None:
        preferred_radial_v = _preferred_cylindrical_cut_radial(axis_v)
        return_corridor_aligned = False
    else:
        preferred_radial_v = return_corridor
        return_corridor_aligned = True
    return InterGroupCutAlignmentState(
        origin_v=np.asarray(origin_v, dtype=float),
        axis_v=_normalize_vector(axis_v, fallback=np.array([0.0, 0.0, 1.0])),
        preferred_radial_v=preferred_radial_v,
        distance_rel_tol=max(float(getattr(config, "inter_group_cut_alignment_distance_rel_tol", 0.0)), 0.0),
        candidate_factor=max(int(getattr(config, "inter_group_cut_alignment_candidate_factor", 1)), 1),
        return_corridor_aligned=return_corridor_aligned,
    )


def _mesh_boundary_axis(mesh) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    centers: list[np.ndarray] = []
    for boundary in getattr(mesh, "boundaries", []) or []:
        boundary_nodes = np.unique(np.asarray(boundary, dtype=np.int64))
        boundary_nodes = boundary_nodes[(boundary_nodes >= 0) & (boundary_nodes < len(vertices))]
        if len(boundary_nodes):
            centers.append(vertices[boundary_nodes].mean(axis=0))
    if len(centers) < 2:
        return None

    best_pair: tuple[int, int] | None = None
    best_distance = -math.inf
    for ind_a, center_a in enumerate(centers):
        for ind_b in range(ind_a + 1, len(centers)):
            distance = float(np.linalg.norm(center_a - centers[ind_b]))
            if distance > best_distance:
                best_distance = distance
                best_pair = (ind_a, ind_b)
    if best_pair is None or best_distance <= 1e-12:
        return None
    center_a = centers[best_pair[0]]
    center_b = centers[best_pair[1]]
    axis_v = _normalize_vector(center_b - center_a, fallback=np.array([0.0, 0.0, 1.0]))
    origin_v = (center_a + center_b) / 2.0
    return origin_v, axis_v


def _preferred_cylindrical_cut_radial(axis_v: np.ndarray) -> np.ndarray:
    axis_v = _normalize_vector(axis_v, fallback=np.array([0.0, 0.0, 1.0]))
    for candidate in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        projected = candidate - axis_v * float(np.dot(candidate, axis_v))
        if np.linalg.norm(projected) > 1e-12:
            return _normalize_vector(projected, fallback=np.array([1.0, 0.0, 0.0]))
    return _orthogonal_vector(axis_v)


def _preferred_return_corridor_radial(
    part: CoilPart,
    origin_v: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray | None:
    """Infer a cylindrical return corridor from existing within-group rails.

    Return rails on opposite sides of the same cylinder diameter represent the
    same manufacturable cut plane, so the average is taken after orienting all
    rail vectors to a common half-plane.  If the rails are not coherent, the
    caller falls back to the historical fixed cylindrical preference.
    """

    axis_v = _normalize_vector(axis_v, fallback=np.array([0.0, 0.0, 1.0]))
    rails: list[tuple[np.ndarray, float]] = []
    for connected in getattr(part, "connected_groups", []) or []:
        return_path_v = np.asarray(getattr(connected, "return_path_v", np.zeros((0, 3))), dtype=float)
        if return_path_v.ndim != 2 or return_path_v.shape[1] != 3 or len(return_path_v) < 2:
            continue
        return_path_v = return_path_v[np.all(np.isfinite(return_path_v), axis=1)]
        if len(return_path_v) < 2:
            continue
        rel = return_path_v - np.asarray(origin_v, dtype=float)
        radial = rel - np.outer(rel @ axis_v, axis_v)
        radial_norm = np.linalg.norm(radial, axis=1)
        valid = radial_norm > 1e-12
        if not np.any(valid):
            continue
        radial_units = radial[valid] / radial_norm[valid, None]
        rail_radial = np.mean(radial_units, axis=0)
        rail_strength = float(np.linalg.norm(rail_radial))
        if rail_strength < 0.75:
            continue
        weight = max(_path_length(return_path_v), float(np.count_nonzero(valid)))
        rails.append((rail_radial / rail_strength, weight))
    if not rails:
        return None

    reference = max(rails, key=lambda item: item[1])[0]
    weighted = np.zeros(3, dtype=float)
    total_weight = 0.0
    for rail_radial, weight in rails:
        oriented = rail_radial if float(np.dot(rail_radial, reference)) >= 0.0 else -rail_radial
        weighted += float(weight) * oriented
        total_weight += float(weight)
    if total_weight <= 0.0:
        return None
    preferred = weighted / total_weight
    if np.linalg.norm(preferred) < 0.5:
        return None
    return _normalize_vector(preferred, fallback=reference)


def _outer_group_connection_search_path(
    part: CoilPart,
    group: ConnectedGroup,
    group_index: int,
    all_return_points_v: np.ndarray,
    *,
    boundary_threshold: int,
) -> GroupConnectionSearchPath:
    """Return routed samples belonging to the group's exterior contour frontier."""

    route = _remove_return_neighbors_from_group(group, all_return_points_v, boundary_threshold)
    outer_loop_indices = _connected_group_outer_loop_indices(part, group, group_index)
    if len(route.v) == 0 or not outer_loop_indices:
        return GroupConnectionSearchPath(v=np.zeros((0, 3), dtype=float), loop_indices=np.zeros(0, dtype=np.int64))

    contour_points: list[np.ndarray] = []
    point_loop_indices: list[np.ndarray] = []
    for loop_index in outer_loop_indices:
        if not (0 <= loop_index < len(part.contour_lines)):
            continue
        points = np.asarray(part.contour_lines[loop_index].v, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            continue
        contour_points.append(points)
        point_loop_indices.append(np.full(len(points), loop_index, dtype=np.int64))
    if not contour_points:
        return GroupConnectionSearchPath(v=np.zeros((0, 3), dtype=float), loop_indices=np.zeros(0, dtype=np.int64))

    reference_v = np.vstack(contour_points)
    reference_loop_indices = np.concatenate(point_loop_indices)
    nearest_distance, nearest_index = _nearest_reference_points(route.v, reference_v)
    scale = max(float(np.linalg.norm(np.ptp(part.mesh.vertices, axis=0))), 1.0)
    membership_tolerance = max(1.0e-9, 1.0e-10 * scale)
    keep = np.isfinite(nearest_distance) & (nearest_distance <= membership_tolerance)
    return GroupConnectionSearchPath(
        v=route.v[keep].copy(),
        loop_indices=reference_loop_indices[nearest_index[keep]].copy(),
    )


def _connected_group_outer_loop_indices(
    part: CoilPart,
    group: ConnectedGroup,
    group_index: int,
) -> tuple[int, ...]:
    indices = tuple(
        dict.fromkeys(int(index) for index in group.outer_loop_indices if 0 <= int(index) < len(part.contour_lines))
    )
    if indices:
        return indices
    if not (0 <= int(group_index) < len(part.groups)):
        return ()
    loop_group = part.groups[int(group_index)]
    indices = tuple(
        dict.fromkeys(
            int(index) for index in loop_group.outer_loop_indices if 0 <= int(index) < len(part.contour_lines)
        )
    )
    if indices:
        return indices
    return _topologically_outer_loop_indices(loop_group.loop_indices, part.loop_in_loop_matrix)


def _nearest_reference_points(points_v: np.ndarray, reference_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_v = np.asarray(points_v, dtype=float)
    reference_v = np.asarray(reference_v, dtype=float)
    if len(points_v) == 0 or len(reference_v) == 0:
        return np.full(len(points_v), math.inf, dtype=float), np.zeros(len(points_v), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree

        distance, index = cKDTree(reference_v).query(points_v, k=1)
        return np.asarray(distance, dtype=float), np.asarray(index, dtype=np.int64)
    except ImportError:
        distance = np.full(len(points_v), math.inf, dtype=float)
        index = np.zeros(len(points_v), dtype=np.int64)
        for start in range(0, len(points_v), 256):
            stop = min(start + 256, len(points_v))
            distance_sq = np.sum((points_v[start:stop, None, :] - reference_v[None, :, :]) ** 2, axis=2)
            local_index = np.argmin(distance_sq, axis=1)
            index[start:stop] = local_index
            distance[start:stop] = np.sqrt(distance_sq[np.arange(stop - start), local_index])
        return distance, index


def _select_group_connection(
    part: CoilPart,
    connected_groups: list[ConnectedGroup],
    groups_to_connect: list[int],
    all_return_points_v: np.ndarray,
    config: CoilGenConfig,
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    *,
    alignment_state: InterGroupCutAlignmentState | None = None,
) -> GroupConnectionCandidate | None:
    if alignment_state is None:
        alignment_state = _inter_group_cut_alignment_state(part, config)
    search_groups = [
        _outer_group_connection_search_path(
            part,
            connected_groups[group_ind],
            group_ind,
            all_return_points_v,
            boundary_threshold=2,
        )
        for group_ind in groups_to_connect
    ]
    candidates: list[GroupConnectionCandidate] = []
    base_candidate_count = _inter_group_candidate_count(config)
    candidate_count = _inter_group_alignment_candidate_count(base_candidate_count, alignment_state)
    if geodesic_solver is not None and alignment_state is None:
        for candidate in _group_connection_direct_shortlist(search_groups, candidate_count):
            if _group_connection_candidate_is_valid(
                part,
                connected_groups,
                groups_to_connect,
                candidate,
                config,
                geodesic_solver,
            ):
                return candidate

    for local_a, group_a in enumerate(search_groups):
        for local_b, group_b in enumerate(search_groups):
            if local_a == local_b:
                continue
            candidates.extend(
                _group_connection_candidates(
                    local_a,
                    local_b,
                    group_a.v,
                    group_b.v,
                    group_a.loop_indices,
                    group_b.loop_indices,
                    candidate_count,
                    geodesic_solver,
                    alignment_state,
                )
            )
    candidates = _sort_group_connection_candidates(candidates, alignment_state)
    for candidate in candidates:
        if _group_connection_candidate_is_valid(
            part,
            connected_groups,
            groups_to_connect,
            candidate,
            config,
            geodesic_solver,
        ):
            return candidate
    if _strict_geodesic_inter_group_paths(config):
        return None
    if candidates:
        fallback = candidates[0]
        fallback.used_clearance_fallback = True
        return fallback
    return None


def _group_connection_direct_shortlist(
    groups: list[GroupConnectionSearchPath],
    candidate_count: int,
) -> list[GroupConnectionCandidate]:
    """Cheap first pass for geodesic routing.

    Surface-geodesic validation is expensive because it runs graph searches.
    This shortlist ranks group pairs by direct endpoint distance and lets the
    validator compute geodesics only for the most plausible connector cuts.  If
    none passes, the caller falls back to the exhaustive geodesic candidate
    builder below.
    """

    candidates: list[GroupConnectionCandidate] = []
    for local_a, group_a in enumerate(groups):
        for local_b in range(local_a + 1, len(groups)):
            group_b = groups[local_b]
            if len(group_a.v) == 0 or len(group_b.v) == 0:
                continue
            direct_distance, point_a_ind, point_b_ind = _nearest_point_pair(group_a.v, group_b.v)
            if not math.isfinite(direct_distance) or direct_distance <= 1e-12:
                continue
            candidates.append(
                GroupConnectionCandidate(
                    local_a=local_a,
                    local_b=local_b,
                    point_a_v=group_a.v[point_a_ind].copy(),
                    point_b_v=group_b.v[point_b_ind].copy(),
                    direct_distance=float(direct_distance),
                    geodesic_distance=math.inf,
                    selection_distance=float(direct_distance),
                    used_geodesic_selection=False,
                    loop_a_index=int(group_a.loop_indices[point_a_ind]),
                    loop_b_index=int(group_b.loop_indices[point_b_ind]),
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.selection_distance)[: max(int(candidate_count), 1)]


def _group_connection_candidates(
    local_a: int,
    local_b: int,
    points_a: np.ndarray,
    points_b: np.ndarray,
    loop_indices_a: np.ndarray,
    loop_indices_b: np.ndarray,
    candidate_count: int,
    geodesic_solver: SurfaceGeodesicSolver | None,
    alignment_state: InterGroupCutAlignmentState | None = None,
) -> list[GroupConnectionCandidate]:
    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)
    loop_indices_a = np.asarray(loop_indices_a, dtype=np.int64).reshape(-1)
    loop_indices_b = np.asarray(loop_indices_b, dtype=np.int64).reshape(-1)
    if len(points_a) == 0 or len(points_b) == 0:
        return []
    if len(loop_indices_a) != len(points_a) or len(loop_indices_b) != len(points_b):
        raise ValueError("Inter-group search loop provenance must match the candidate point arrays.")

    index_pairs: dict[tuple[int, int], None] = {}
    for _direct_distance, point_a_ind, point_b_ind in _nearest_point_pairs(points_a, points_b, candidate_count):
        index_pairs[(point_a_ind, point_b_ind)] = None
    for _direct_distance, point_a_ind, point_b_ind in _distributed_nearest_point_pairs(
        points_a,
        points_b,
        candidate_count,
    ):
        index_pairs[(point_a_ind, point_b_ind)] = None
    if alignment_state is not None:
        for _direct_distance, point_a_ind, point_b_ind in _aligned_point_pairs(
            points_a,
            points_b,
            alignment_state,
            candidate_count,
        ):
            index_pairs[(point_a_ind, point_b_ind)] = None
    if geodesic_solver is not None:
        geodesic = geodesic_solver.distance_between_point_sets(points_a, points_b)
        if math.isfinite(geodesic.distance):
            index_pairs[(int(geodesic.index_a), int(geodesic.index_b))] = None

    candidates: list[GroupConnectionCandidate] = []
    for point_a_ind, point_b_ind in index_pairs:
        point_a_v = points_a[point_a_ind]
        point_b_v = points_b[point_b_ind]
        direct_distance = float(np.linalg.norm(point_a_v - point_b_v))
        if direct_distance <= 1e-12 or not math.isfinite(direct_distance):
            continue
        geodesic_distance = math.inf
        used_geodesic_selection = False
        selection_distance = direct_distance
        if geodesic_solver is not None:
            geodesic = geodesic_solver.distance_between_points(point_a_v, point_b_v)
            geodesic_distance = float(geodesic.distance)
            if math.isfinite(geodesic_distance):
                selection_distance = geodesic_distance
                used_geodesic_selection = True
        candidates.append(
            GroupConnectionCandidate(
                local_a=local_a,
                local_b=local_b,
                point_a_v=point_a_v.copy(),
                point_b_v=point_b_v.copy(),
                direct_distance=direct_distance,
                geodesic_distance=geodesic_distance,
                selection_distance=float(selection_distance),
                used_geodesic_selection=used_geodesic_selection,
                loop_a_index=int(loop_indices_a[point_a_ind]),
                loop_b_index=int(loop_indices_b[point_b_ind]),
            )
        )
    return candidates


def _aligned_point_pairs(
    points_a: np.ndarray,
    points_b: np.ndarray,
    alignment_state: InterGroupCutAlignmentState,
    count: int,
) -> list[tuple[float, int, int]]:
    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)
    if len(points_a) == 0 or len(points_b) == 0:
        return []
    count = max(int(count), 1)
    shortlist_count_a = min(len(points_a), max(count, 8))
    shortlist_count_b = min(len(points_b), max(count, 8))
    errors_a = _point_alignment_errors(points_a, alignment_state)
    errors_b = _point_alignment_errors(points_b, alignment_state)
    finite_a = np.flatnonzero(np.isfinite(errors_a))
    finite_b = np.flatnonzero(np.isfinite(errors_b))
    if len(finite_a) == 0 or len(finite_b) == 0:
        return []

    selected_a = finite_a[np.argsort(errors_a[finite_a], kind="stable")[:shortlist_count_a]]
    selected_b = finite_b[np.argsort(errors_b[finite_b], kind="stable")[:shortlist_count_b]]
    pairs = _nearest_point_pairs(points_a[selected_a], points_b[selected_b], count)
    return [(distance, int(selected_a[local_a]), int(selected_b[local_b])) for distance, local_a, local_b in pairs]


def _point_alignment_errors(
    points: np.ndarray,
    alignment_state: InterGroupCutAlignmentState,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    rel = points - alignment_state.origin_v
    radial = rel - np.outer(rel @ alignment_state.axis_v, alignment_state.axis_v)
    radial_norm = np.linalg.norm(radial, axis=1)
    errors = np.full(len(points), math.inf, dtype=float)
    valid = radial_norm > 1e-12
    if not np.any(valid):
        return errors
    radial_unit = radial[valid] / radial_norm[valid, None]
    dots = np.clip(radial_unit @ alignment_state.preferred_radial_v, -1.0, 1.0)
    errors[valid] = 1.0 - dots
    return errors


def _inter_group_alignment_candidate_count(
    base_candidate_count: int,
    alignment_state: InterGroupCutAlignmentState | None,
) -> int:
    count = max(int(base_candidate_count), 1)
    if alignment_state is None:
        return count
    return max(count, count * max(int(alignment_state.candidate_factor), 1))


def _sort_group_connection_candidates(
    candidates: list[GroupConnectionCandidate],
    alignment_state: InterGroupCutAlignmentState | None,
) -> list[GroupConnectionCandidate]:
    if not candidates:
        return []
    if alignment_state is None:
        return sorted(candidates, key=lambda candidate: candidate.selection_distance)
    finite_distances = [
        float(candidate.selection_distance)
        for candidate in candidates
        if math.isfinite(float(candidate.selection_distance))
    ]
    if not finite_distances:
        return sorted(candidates, key=lambda candidate: candidate.selection_distance)
    best_distance = min(finite_distances)
    return sorted(
        candidates,
        key=lambda candidate: _aligned_group_connection_sort_key(candidate, alignment_state, best_distance),
    )


def _aligned_group_connection_sort_key(
    candidate: GroupConnectionCandidate,
    alignment_state: InterGroupCutAlignmentState,
    best_distance: float,
) -> tuple[float, float, float, int, int]:
    distance = float(candidate.selection_distance)
    if not math.isfinite(distance):
        return (2.0, math.inf, math.inf, int(candidate.local_a), int(candidate.local_b))
    tolerance = max(float(alignment_state.distance_rel_tol) * max(abs(best_distance), 1e-12), 1e-12)
    if alignment_state.return_corridor_aligned:
        tolerance = max(tolerance, 0.25 * max(abs(best_distance), 1e-12))
    alignment_error = _group_connection_alignment_error(candidate, alignment_state)
    if distance <= best_distance + tolerance:
        return (0.0, alignment_error, distance, int(candidate.local_a), int(candidate.local_b))
    return (1.0, distance, alignment_error, int(candidate.local_a), int(candidate.local_b))


def _group_connection_alignment_error(
    candidate: GroupConnectionCandidate,
    alignment_state: InterGroupCutAlignmentState,
) -> float:
    midpoint_v = (np.asarray(candidate.point_a_v, dtype=float) + np.asarray(candidate.point_b_v, dtype=float)) / 2.0
    radial_v = midpoint_v - alignment_state.origin_v
    radial_v = radial_v - alignment_state.axis_v * float(np.dot(radial_v, alignment_state.axis_v))
    radial_norm = float(np.linalg.norm(radial_v))
    if radial_norm <= 1e-12:
        return math.inf
    radial_v = radial_v / radial_norm
    dot = float(np.clip(np.dot(radial_v, alignment_state.preferred_radial_v), -1.0, 1.0))
    return 1.0 - dot


def _group_connection_candidate_is_valid(
    part: CoilPart,
    connected_groups: list[ConnectedGroup],
    groups_to_connect: list[int],
    candidate: GroupConnectionCandidate,
    config: CoilGenConfig,
    geodesic_solver: SurfaceGeodesicSolver | None,
) -> bool:
    group_a_index = groups_to_connect[candidate.local_a]
    group_b_index = groups_to_connect[candidate.local_b]
    outer_a = _connected_group_outer_loop_indices(part, connected_groups[group_a_index], group_a_index)
    outer_b = _connected_group_outer_loop_indices(part, connected_groups[group_b_index], group_b_index)
    if candidate.loop_a_index not in outer_a or candidate.loop_b_index not in outer_b:
        return False
    endpoint_loop_indices = {int(candidate.loop_a_index), int(candidate.loop_b_index)}
    allow_direct_fallback = not _strict_geodesic_inter_group_paths(config)
    reject_suspicious_length = allow_direct_fallback
    point_connector = _inter_group_connector_result(
        candidate.point_a_v,
        candidate.point_b_v,
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    if len(point_connector.path_v) < 2:
        return False
    point_connector_distance = _path_length(point_connector.path_v)
    if reject_suspicious_length and _connector_ratio(
        point_connector_distance, candidate.direct_distance, candidate.geodesic_distance
    ) > _suspicious_connector_ratio(config):
        return False
    if _connector_path_clearance_to_unrelated_contours(
        part,
        point_connector.path_v,
        endpoint_loop_indices,
    ) < _connector_clearance_threshold(config):
        return False

    opened_a, cut_a = _open_connected_group_with_3d_sphere(
        connected_groups[group_a_index],
        candidate.point_a_v,
        _inter_group_cut_width(config),
    )
    opened_b, cut_b = _open_connected_group_with_3d_sphere(
        connected_groups[group_b_index],
        candidate.point_b_v,
        _inter_group_cut_width(config),
    )
    opening_threshold = _opening_cut_span_threshold(config)
    span_a = _opening_cut_span(cut_a, opening_threshold)
    span_b = _opening_cut_span(cut_b, opening_threshold)
    if not math.isfinite(span_a) or span_a <= opening_threshold:
        return False
    if not math.isfinite(span_b) or span_b <= opening_threshold:
        return False

    _fused, connection_path_v, closing_connection_path_v, _used_direct_fallback = (
        _fuse_opened_wire_paths_with_connector_pair_diagnostics(
            opened_a,
            opened_b,
            geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        )
    )
    if min(len(connection_path_v), len(closing_connection_path_v)) < 2:
        return False
    clearance_threshold = _connector_clearance_threshold(config)
    for connector_path_v in (connection_path_v, closing_connection_path_v):
        connector_distance = _path_length(connector_path_v)
        if reject_suspicious_length and _connector_ratio(
            connector_distance, candidate.direct_distance, candidate.geodesic_distance
        ) > _suspicious_connector_ratio(config):
            return False
        if (
            _connector_path_clearance_to_unrelated_contours(
                part,
                connector_path_v,
                endpoint_loop_indices,
            )
            < clearance_threshold
        ):
            return False
    return True


def _remove_return_neighbors_from_group(
    group: ConnectedGroup,
    points_to_remove_v: np.ndarray,
    boundary_threshold: int,
) -> WirePath:
    if len(group.v) == 0 or len(points_to_remove_v) == 0:
        return WirePath(v=group.v.copy())

    points = np.asarray(points_to_remove_v, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return WirePath(v=group.v.copy())

    remove = np.zeros(len(group.v), dtype=bool)
    for point in points:
        matches = np.where(np.linalg.norm(group.v - point, axis=1) <= 1e-10)[0]
        for match in matches:
            for offset in range(-boundary_threshold, boundary_threshold + 1):
                remove[(int(match) + offset) % len(remove)] = True

    if np.count_nonzero(~remove) < 2:
        return WirePath(v=group.v.copy())
    return WirePath(v=group.v[~remove].copy())


def _nearest_point_pair(points_a: np.ndarray, points_b: np.ndarray) -> tuple[float, int, int]:
    pairs = _nearest_point_pairs(points_a, points_b, 1)
    if not pairs:
        return math.inf, 0, 0
    return pairs[0]


def _nearest_point_pairs(points_a: np.ndarray, points_b: np.ndarray, count: int) -> list[tuple[float, int, int]]:
    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)
    if len(points_a) == 0 or len(points_b) == 0:
        return []

    count = max(int(count), 1)
    candidates: list[tuple[float, int, int]] = []
    chunk_size = max(1, min(1024, len(points_a)))
    for start in range(0, len(points_a), chunk_size):
        stop = min(start + chunk_size, len(points_a))
        chunk = points_a[start:stop]
        distances = np.linalg.norm(chunk[:, None, :] - points_b[None, :, :], axis=2)
        finite_flat = np.where(np.isfinite(distances.ravel()))[0]
        if len(finite_flat) == 0:
            continue
        local_count = min(count, len(finite_flat))
        finite_distances = distances.ravel()[finite_flat]
        selected = finite_flat[np.argpartition(finite_distances, local_count - 1)[:local_count]]
        for flat_index in selected:
            local_a, point_b_ind = np.unravel_index(int(flat_index), distances.shape)
            candidates.append((float(distances[local_a, point_b_ind]), start + int(local_a), int(point_b_ind)))
        candidates = sorted(candidates, key=lambda item: item[0])[:count]
    return candidates


def _distributed_nearest_point_pairs(
    points_a: np.ndarray,
    points_b: np.ndarray,
    count: int,
) -> list[tuple[float, int, int]]:
    """Sample both outer contours so valid far-side cuts remain discoverable."""

    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)
    if len(points_a) == 0 or len(points_b) == 0:
        return []
    sample_count_a = min(len(points_a), max(2 * int(count), 8))
    sample_count_b = min(len(points_b), max(2 * int(count), 8))
    sampled_a = np.unique(np.linspace(0, len(points_a) - 1, sample_count_a, dtype=np.int64))
    sampled_b = np.unique(np.linspace(0, len(points_b) - 1, sample_count_b, dtype=np.int64))
    candidates: dict[tuple[int, int], float] = {}
    for point_a_index in sampled_a:
        distances = np.linalg.norm(points_b - points_a[int(point_a_index)], axis=1)
        point_b_index = int(np.argmin(distances))
        candidates[(int(point_a_index), point_b_index)] = float(distances[point_b_index])
    for point_b_index in sampled_b:
        distances = np.linalg.norm(points_a - points_b[int(point_b_index)], axis=1)
        point_a_index = int(np.argmin(distances))
        key = (point_a_index, int(point_b_index))
        candidates[key] = min(candidates.get(key, math.inf), float(distances[point_a_index]))
    return sorted(
        ((distance, point_a_index, point_b_index) for (point_a_index, point_b_index), distance in candidates.items()),
        key=lambda item: item[0],
    )


def _open_connected_group_with_3d_sphere(
    group: ConnectedGroup,
    sphere_center: np.ndarray,
    sphere_diameter: float,
) -> tuple[WirePath, np.ndarray]:
    if len(group.v) < 2:
        return WirePath(v=group.v.copy()), np.zeros((0, 3), dtype=float)
    temporary_loop = ContourLine(v=group.v, potential=0.0)
    opened, cut_points = open_loop_with_3d_sphere(temporary_loop, sphere_center, sphere_diameter)
    return WirePath(v=opened.v), cut_points


def _fuse_opened_wire_paths_with_connector(
    path_a: WirePath,
    path_b: WirePath,
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    *,
    allow_direct_fallback: bool = True,
) -> tuple[WirePath, np.ndarray]:
    fused, connector_v, _closing_connector_v, _used_direct_fallback = (
        _fuse_opened_wire_paths_with_connector_pair_diagnostics(
            path_a,
            path_b,
            geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        )
    )
    return fused, connector_v


def _fuse_opened_wire_paths_with_connector_pair_diagnostics(
    path_a: WirePath,
    path_b: WirePath,
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    *,
    allow_direct_fallback: bool = True,
) -> tuple[WirePath, np.ndarray, np.ndarray, bool]:
    """Fuse two opened paths and retain both bridges between their four terminals."""

    empty = np.zeros((0, 3), dtype=float)
    if len(path_a.v) == 0:
        return WirePath(v=path_b.v.copy()), empty, empty, False
    if len(path_b.v) == 0:
        return WirePath(v=path_a.v.copy()), empty, empty, False

    option_ab_v, connector_ab, closing_ab = _fused_path_candidate(
        path_a,
        path_b,
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    option_ba_v, connector_ba, closing_ba = _fused_path_candidate(
        path_b,
        path_a,
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    if len(option_ab_v) == 0 and len(option_ba_v) == 0:
        return WirePath(v=empty), empty, empty, True
    if len(option_ab_v) == 0:
        return (
            WirePath(v=option_ba_v),
            connector_ba.path_v,
            closing_ba.path_v,
            connector_ba.used_direct_fallback or closing_ba.used_direct_fallback,
        )
    if len(option_ba_v) == 0:
        return (
            WirePath(v=option_ab_v),
            connector_ab.path_v,
            closing_ab.path_v,
            connector_ab.used_direct_fallback or closing_ab.used_direct_fallback,
        )
    if _path_length(option_ab_v) <= _path_length(option_ba_v):
        return (
            WirePath(v=option_ab_v),
            connector_ab.path_v,
            closing_ab.path_v,
            connector_ab.used_direct_fallback or closing_ab.used_direct_fallback,
        )
    return (
        WirePath(v=option_ba_v),
        connector_ba.path_v,
        closing_ba.path_v,
        connector_ba.used_direct_fallback or closing_ba.used_direct_fallback,
    )


def _fused_path_candidate(
    first: WirePath,
    second: WirePath,
    geodesic_solver: SurfaceGeodesicSolver | None,
    *,
    allow_direct_fallback: bool = True,
) -> tuple[np.ndarray, ConnectorPathResult, ConnectorPathResult]:
    first_v = first.v
    second_v = second.v
    connector = _inter_group_connector_result(
        first_v[-1],
        second_v[0],
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    if len(connector.path_v) < 2:
        return (
            np.zeros((0, 3), dtype=float),
            connector,
            ConnectorPathResult(
                path_v=np.zeros((0, 3), dtype=float), used_direct_fallback=connector.used_direct_fallback
            ),
        )
    closing_connector = _inter_group_connector_result(
        second_v[-1],
        first_v[0],
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    if len(closing_connector.path_v) < 2:
        return np.zeros((0, 3), dtype=float), connector, closing_connector
    return (
        _stack_path_sections((first_v, connector.path_v, second_v, closing_connector.path_v, first_v[:1])),
        connector,
        closing_connector,
    )


def _combined_return_group_fusion_candidate(
    part: CoilPart,
    group1: ConnectedGroup,
    group2: ConnectedGroup,
    group1_index: int,
    group2_index: int,
    config: CoilGenConfig,
    geodesic_solver: SurfaceGeodesicSolver | None,
    *,
    allow_direct_fallback: bool,
    alignment_state: InterGroupCutAlignmentState | None = None,
) -> GroupFusionResult | None:
    """Try to fuse two atomic groups by sharing the return corridor.

    The ordinary inter-group route opens two already closed group paths.  When
    both groups still expose their original spiral and return terminals, a
    cleaner current path is often available: traverse both spiral sections
    first, then use the second group's return rail, jump along the return
    corridor, and finish on the first group's return rail.  This is kept as an
    internal opportunistic candidate because it only makes sense when the
    terminal geometry is already favorable.
    """

    group1_variants = _group_route_variants(group1)
    group2_variants = _group_route_variants(group2)
    if not group1_variants or not group2_variants:
        return None

    group1_loop_indices = _loop_indices_for_groups(part, (group1_index,))
    group2_loop_indices = _loop_indices_for_groups(part, (group2_index,))
    group1_outer_loop_indices = set(_connected_group_outer_loop_indices(part, group1, group1_index))
    group2_outer_loop_indices = set(_connected_group_outer_loop_indices(part, group2, group2_index))
    if not group1_outer_loop_indices or not group2_outer_loop_indices:
        return None
    candidates: list[GroupFusionResult | None] = []
    for group1_variant in group1_variants:
        for group2_variant in group2_variants:
            candidates.append(
                _combined_return_group_fusion_for_order(
                    part,
                    group1_variant,
                    group2_variant,
                    0,
                    1,
                    config,
                    geodesic_solver,
                    allow_direct_fallback=allow_direct_fallback,
                    first_group_loop_indices=group1_loop_indices,
                    second_group_loop_indices=group2_loop_indices,
                    first_outer_loop_indices=group1_outer_loop_indices,
                    second_outer_loop_indices=group2_outer_loop_indices,
                    alignment_state=alignment_state,
                )
            )
            candidates.append(
                _combined_return_group_fusion_for_order(
                    part,
                    group2_variant,
                    group1_variant,
                    1,
                    0,
                    config,
                    geodesic_solver,
                    allow_direct_fallback=allow_direct_fallback,
                    first_group_loop_indices=group2_loop_indices,
                    second_group_loop_indices=group1_loop_indices,
                    first_outer_loop_indices=group2_outer_loop_indices,
                    second_outer_loop_indices=group1_outer_loop_indices,
                    alignment_state=alignment_state,
                )
            )
    valid_candidates = [candidate for candidate in candidates if candidate is not None]
    if not valid_candidates:
        return None
    valid_candidates.sort(key=lambda candidate: (candidate.path_length, candidate.connector_distance))
    return valid_candidates[0]


def _combined_return_group_fusion_for_order(
    part: CoilPart,
    first_variant: GroupRouteVariant,
    second_variant: GroupRouteVariant,
    first_input_index: int,
    second_input_index: int,
    config: CoilGenConfig,
    geodesic_solver: SurfaceGeodesicSolver | None,
    *,
    allow_direct_fallback: bool,
    first_group_loop_indices: set[int],
    second_group_loop_indices: set[int],
    first_outer_loop_indices: set[int],
    second_outer_loop_indices: set[int],
    alignment_state: InterGroupCutAlignmentState | None,
) -> GroupFusionResult | None:
    spiral_first = _terminal_path_array(first_variant.spiral_v)
    spiral_second = _terminal_path_array(second_variant.spiral_v)
    return_first = _terminal_path_array(first_variant.return_path_v)
    return_second = _terminal_path_array(second_variant.return_path_v)
    if min(len(spiral_first), len(spiral_second), len(return_first), len(return_second)) < 2:
        return None
    if not _return_terminals_share_side(return_first[0], return_second[-1], alignment_state):
        return None

    spiral_first_end_loop = _loop_index_at_point(part, spiral_first[-1], first_group_loop_indices)
    spiral_second_start_loop = _loop_index_at_point(part, spiral_second[0], second_group_loop_indices)
    return_second_end_loop = _loop_index_at_point(part, return_second[-1], second_group_loop_indices)
    return_first_start_loop = _loop_index_at_point(part, return_first[0], first_group_loop_indices)
    if (
        spiral_first_end_loop not in first_outer_loop_indices
        or spiral_second_start_loop not in second_outer_loop_indices
        or return_second_end_loop not in second_outer_loop_indices
        or return_first_start_loop not in first_outer_loop_indices
    ):
        return None

    spiral_second_end_loop = _loop_index_at_point(part, spiral_second[-1], second_group_loop_indices)
    return_second_start_loop = _loop_index_at_point(part, return_second[0], second_group_loop_indices)
    return_first_end_loop = _loop_index_at_point(part, return_first[-1], first_group_loop_indices)
    spiral_first_start_loop = _loop_index_at_point(part, spiral_first[0], first_group_loop_indices)
    if (
        min(
            spiral_second_end_loop,
            return_second_start_loop,
            return_first_end_loop,
            spiral_first_start_loop,
        )
        < 0
    ):
        return None

    spiral_connector = _inter_group_connector_result(
        spiral_first[-1],
        spiral_second[0],
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    second_return_connector = _inter_group_connector_result(
        spiral_second[-1],
        return_second[0],
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    return_connector = _inter_group_connector_result(
        return_second[-1],
        return_first[0],
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    closing_connector = _inter_group_connector_result(
        return_first[-1],
        spiral_first[0],
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    if (
        min(
            len(spiral_connector.path_v),
            len(second_return_connector.path_v),
            len(return_connector.path_v),
            len(closing_connector.path_v),
        )
        < 2
    ):
        return None

    if not _combined_return_connector_is_valid(
        part,
        spiral_connector.path_v,
        spiral_first[-1],
        spiral_second[0],
        config,
        {spiral_first_end_loop, spiral_second_start_loop},
        geodesic_solver,
        spiral_connector.used_direct_fallback,
    ):
        return None
    if not _combined_return_connector_is_valid(
        part,
        second_return_connector.path_v,
        spiral_second[-1],
        return_second[0],
        config,
        {spiral_second_end_loop, return_second_start_loop},
        geodesic_solver,
        second_return_connector.used_direct_fallback,
    ):
        return None
    if not _combined_return_connector_is_valid(
        part,
        return_connector.path_v,
        return_second[-1],
        return_first[0],
        config,
        {return_second_end_loop, return_first_start_loop},
        geodesic_solver,
        return_connector.used_direct_fallback,
    ):
        return None
    if not _combined_return_connector_is_valid(
        part,
        closing_connector.path_v,
        return_first[-1],
        spiral_first[0],
        config,
        {return_first_end_loop, spiral_first_start_loop},
        geodesic_solver,
        closing_connector.used_direct_fallback,
    ):
        return None

    fused_v = _stack_path_sections(
        (
            spiral_first,
            spiral_connector.path_v,
            spiral_second,
            second_return_connector.path_v,
            return_second,
            return_connector.path_v,
            return_first,
            closing_connector.path_v,
            spiral_first[:1],
        )
    )
    if len(fused_v) < 2 or not np.all(np.isfinite(fused_v)):
        return None
    selected_points = {
        first_input_index: return_first[0].copy(),
        second_input_index: return_second[-1].copy(),
    }
    cuts = {
        first_input_index: _return_terminal_cut_shape(return_first, 0),
        second_input_index: _return_terminal_cut_shape(return_second, -1),
    }
    variant_names = {
        first_input_index: first_variant.name,
        second_input_index: second_variant.name,
    }
    selected_loop_indices = {
        first_input_index: return_first_start_loop,
        second_input_index: return_second_end_loop,
    }
    direct_distance = float(np.linalg.norm(return_second[-1] - return_first[0]))
    connector_distance = _path_length(return_connector.path_v)
    if direct_distance <= 1e-12 or not math.isfinite(connector_distance):
        return None
    geodesic_distance = (
        connector_distance if geodesic_solver is not None and not return_connector.used_direct_fallback else math.inf
    )
    return GroupFusionResult(
        fused=WirePath(v=fused_v),
        connection_path_v=return_connector.path_v.copy(),
        closing_connection_path_v=spiral_connector.path_v.copy(),
        cut1_v=cuts[0],
        cut2_v=cuts[1],
        selected_point1_v=selected_points[0],
        selected_point2_v=selected_points[1],
        direct_distance=direct_distance,
        geodesic_distance=geodesic_distance,
        connector_distance=connector_distance,
        used_direct_fallback=(
            spiral_connector.used_direct_fallback
            or second_return_connector.used_direct_fallback
            or return_connector.used_direct_fallback
            or closing_connector.used_direct_fallback
        ),
        path_length=_path_length(fused_v),
        normal_shift_paths_v=(
            spiral_connector.path_v.copy(),
            second_return_connector.path_v.copy(),
            return_connector.path_v.copy(),
            closing_connector.path_v.copy(),
        ),
        combined_return=True,
        group1_variant_name=variant_names[0],
        group2_variant_name=variant_names[1],
        loop1_index=selected_loop_indices[0],
        loop2_index=selected_loop_indices[1],
    )


def _group_route_variants(group: ConnectedGroup) -> list[GroupRouteVariant]:
    return_path_v = _terminal_path_array(group.return_path_v)
    if len(return_path_v) < 2:
        return []
    raw_variants = (
        ("spiral_in", group.spiral_in_v, return_path_v),
        ("spiral_out", group.spiral_out_v, return_path_v[::-1].copy()),
    )
    variants: list[GroupRouteVariant] = []
    signatures: set[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = set()
    for name, spiral_v, variant_return_v in raw_variants:
        spiral_v = _terminal_path_array(spiral_v)
        variant_return_v = _terminal_path_array(variant_return_v)
        if len(spiral_v) < 2 or len(variant_return_v) < 2:
            continue
        signature = (
            tuple(np.round(spiral_v[0], 12)),
            tuple(np.round(spiral_v[-1], 12)),
            tuple(np.round(variant_return_v[0], 12)),
            tuple(np.round(variant_return_v[-1], 12)),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        variants.append(
            GroupRouteVariant(
                name=name,
                spiral_v=spiral_v,
                return_path_v=variant_return_v,
            )
        )
    return variants


def _terminal_path_array(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.zeros((0, 3), dtype=float)
    return _clean_path_array(points, 1e-12)


def _return_terminal_cut_shape(return_path_v: np.ndarray, endpoint_index: int) -> np.ndarray:
    return_path_v = _terminal_path_array(return_path_v)
    if len(return_path_v) < 2:
        return np.zeros((0, 3), dtype=float)
    if endpoint_index == 0:
        return return_path_v[:2].copy()
    return return_path_v[-2:].copy()


def _return_terminals_share_side(
    point_a_v: np.ndarray,
    point_b_v: np.ndarray,
    alignment_state: InterGroupCutAlignmentState | None,
) -> bool:
    if alignment_state is None:
        return True
    radial_a = _point_radial_unit(point_a_v, alignment_state)
    radial_b = _point_radial_unit(point_b_v, alignment_state)
    if radial_a is None or radial_b is None:
        return False
    return float(np.dot(radial_a, radial_b)) >= 0.5


def _point_radial_unit(
    point_v: np.ndarray,
    alignment_state: InterGroupCutAlignmentState,
) -> np.ndarray | None:
    radial_v = np.asarray(point_v, dtype=float) - alignment_state.origin_v
    radial_v = radial_v - alignment_state.axis_v * float(np.dot(radial_v, alignment_state.axis_v))
    radial_norm = float(np.linalg.norm(radial_v))
    if radial_norm <= 1e-12:
        return None
    return radial_v / radial_norm


def _combined_return_connector_is_valid(
    part: CoilPart,
    connector_path_v: np.ndarray,
    start_v: np.ndarray,
    end_v: np.ndarray,
    config: CoilGenConfig,
    excluded_loop_indices: set[int],
    geodesic_solver: SurfaceGeodesicSolver | None,
    used_direct_fallback: bool,
) -> bool:
    direct_distance = float(np.linalg.norm(np.asarray(start_v, dtype=float) - np.asarray(end_v, dtype=float)))
    connector_distance = _path_length(connector_path_v)
    if not math.isfinite(connector_distance):
        return False
    if direct_distance <= 1e-12:
        return connector_distance <= 1e-10
    geodesic_distance = connector_distance if geodesic_solver is not None and not used_direct_fallback else math.inf
    if _connector_ratio(connector_distance, direct_distance, geodesic_distance) > _suspicious_connector_ratio(config):
        return False
    return _connector_path_clearance_to_unrelated_contours(
        part,
        connector_path_v,
        excluded_loop_indices,
    ) >= _connector_clearance_threshold(config)


def _inter_group_connector_path(
    start_v: np.ndarray,
    end_v: np.ndarray,
    geodesic_solver: SurfaceGeodesicSolver | None,
    *,
    allow_direct_fallback: bool = True,
) -> np.ndarray:
    return _inter_group_connector_result(
        start_v,
        end_v,
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    ).path_v


def _inter_group_connector_result(
    start_v: np.ndarray,
    end_v: np.ndarray,
    geodesic_solver: SurfaceGeodesicSolver | None,
    *,
    allow_direct_fallback: bool = True,
) -> ConnectorPathResult:
    direct_v = np.vstack((np.asarray(start_v, dtype=float).reshape(1, 3), np.asarray(end_v, dtype=float).reshape(1, 3)))
    if geodesic_solver is None:
        if allow_direct_fallback:
            return ConnectorPathResult(path_v=direct_v, used_direct_fallback=False)
        return ConnectorPathResult(path_v=np.zeros((0, 3), dtype=float), used_direct_fallback=True)

    result = geodesic_solver.distance_between_points(start_v, end_v)
    if not math.isfinite(result.distance) or len(result.path_v) < 2:
        if allow_direct_fallback:
            return ConnectorPathResult(path_v=direct_v, used_direct_fallback=True)
        return ConnectorPathResult(path_v=np.zeros((0, 3), dtype=float), used_direct_fallback=True)
    return ConnectorPathResult(path_v=result.path_v.copy(), used_direct_fallback=False)


def _stack_path_sections(sections: tuple[np.ndarray, ...], tolerance: float = 1e-10) -> np.ndarray:
    path_v = np.zeros((0, 3), dtype=float)
    for section in sections:
        section_v = np.asarray(section, dtype=float)
        if len(section_v) == 0:
            continue
        if len(path_v) == 0:
            path_v = section_v.copy()
            continue
        append_v = section_v[1:] if np.linalg.norm(path_v[-1] - section_v[0]) <= tolerance else section_v
        if len(append_v):
            path_v = np.vstack((path_v, append_v))
    return path_v


def _copy_connected_group(group: ConnectedGroup) -> ConnectedGroup:
    return ConnectedGroup(
        v=group.v.copy(),
        return_path_v=group.return_path_v.copy(),
        spiral_in_v=group.spiral_in_v.copy(),
        spiral_out_v=group.spiral_out_v.copy(),
        outer_loop_indices=tuple(group.outer_loop_indices),
    )


def _replace_connected_group_path(
    group: ConnectedGroup,
    path: WirePath,
    *,
    outer_loop_indices: tuple[int, ...] | None = None,
) -> ConnectedGroup:
    return ConnectedGroup(
        v=path.v.copy(),
        return_path_v=group.return_path_v.copy(),
        spiral_in_v=group.spiral_in_v.copy(),
        spiral_out_v=group.spiral_out_v.copy(),
        outer_loop_indices=(
            tuple(group.outer_loop_indices) if outer_loop_indices is None else tuple(outer_loop_indices)
        ),
    )


def _stack_return_points_v(groups: list[ConnectedGroup]) -> np.ndarray:
    arrays = [group.return_path_v for group in groups if len(group.return_path_v)]
    if not arrays:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(arrays)


def _greedy_fuse_connected_groups(
    groups: list[ConnectedGroup],
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    *,
    allow_direct_fallback: bool = True,
) -> WirePath:
    remaining = [_copy_connected_group(group) for group in groups]
    start_ind = int(np.argmax([len(group.v) for group in remaining]))
    first = remaining.pop(start_ind)
    v_path = first.v.copy()
    while remaining:
        current = v_path[-1]
        best = None
        for group_ind, group in enumerate(remaining):
            v_option = group.v
            distances = np.linalg.norm(v_option - current, axis=1)
            point_ind = int(np.argmin(distances))
            if geodesic_solver is None:
                score = float(distances[point_ind])
            else:
                score = float(geodesic_solver.distance_between_points(current, v_option[point_ind]).distance)
                if not math.isfinite(score):
                    if not allow_direct_fallback:
                        continue
                    score = float(distances[point_ind])
            if best is None or score < best[0]:
                best = (score, group_ind, point_ind, v_option)
        if best is None:
            raise ValueError(
                "Unable to greedily fuse connected groups with a surface-geodesic connector. "
                "The route was not completed because a straight 3D connector would leave the support surface."
            )
        _, group_ind, point_ind, v_option = best
        remaining.pop(group_ind)
        v_rot = np.vstack((v_option[point_ind:], v_option[: point_ind + 1]))
        connector_v = _inter_group_connector_path(
            v_path[-1],
            v_rot[0],
            geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        )
        if len(connector_v) < 2:
            raise ValueError(
                "Unable to greedily fuse connected groups with a surface-geodesic connector. "
                "The route was not completed because a straight 3D connector would leave the support surface."
            )
        v_path = _stack_path_sections((v_path, connector_v, v_rot))
    if not _is_closed_path_array(v_path, 1e-10):
        closing_connector_v = _inter_group_connector_path(
            v_path[-1],
            v_path[0],
            geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        )
        if len(closing_connector_v) < 2:
            raise ValueError(
                "Unable to close the greedily fused route with a surface-geodesic connector. "
                "The route was not completed because a straight 3D connector would leave the support surface."
            )
        v_path = _stack_path_sections((v_path, closing_connector_v))
    return _rotate_wire_path_to_max_z(WirePath(v=v_path))


def _rotate_wire_path_to_max_z(path: WirePath) -> WirePath:
    if len(path.v) == 0:
        return path
    was_closed = _is_closed_path_array(path.v, 1e-10)
    points = path.v[:-1] if was_closed else path.v
    if len(points) == 0:
        return WirePath(v=path.v.copy())
    start_ind = int(np.argmax(points[:, 2]))
    rotated = np.roll(points, -start_ind, axis=0)
    if was_closed:
        rotated = np.vstack((rotated, rotated[0]))
    owners = (
        list(np.roll(np.asarray(path.segment_owners, dtype=object), -start_ind))
        if was_closed and path.segment_owners
        else transfer_segment_owners(path.v, path.segment_owners, rotated)
    )
    geometry = (
        list(np.roll(np.asarray(path.segment_geometry, dtype=object), -start_ind))
        if was_closed and path.segment_geometry
        else transfer_segment_geometry(path.v, path.segment_geometry, rotated)
    )
    return WirePath(v=rotated, segment_owners=owners, segment_geometry=geometry)


def _cut_plane_direction(
    loop_group: LoopGroup,
    group_center: np.ndarray,
    loop_normal: np.ndarray,
    b_0_direction: np.ndarray,
    cut_plane_definition: str,
) -> np.ndarray:
    b0 = _normalize_vector(b_0_direction, fallback=np.array([0.0, 0.0, 1.0]))
    loop_normal = _normalize_vector(loop_normal, fallback=np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(np.cross(b0, loop_normal)) < 0.01:
        if cut_plane_definition == "nearest":
            first_loop = loop_group.loops[0]
            nearest_ind = int(np.argmin(np.linalg.norm(first_loop.v - group_center, axis=1)))
            alternative_cut_direction = _normalize_vector(
                first_loop.v[nearest_ind] - group_center,
                fallback=_orthogonal_vector(loop_normal),
            )
            return _normalize_vector(
                np.cross(alternative_cut_direction, loop_normal), fallback=_orthogonal_vector(loop_normal)
            )

        axes = np.eye(3)
        axis = axes[int(np.argmin(np.abs(axes @ b0)))]
        alternative_center_normal = _normalize_vector(np.cross(b0, axis), fallback=_orthogonal_vector(b0))
        return _normalize_vector(np.cross(b0, alternative_center_normal), fallback=_orthogonal_vector(loop_normal))
    return _normalize_vector(np.cross(b0, loop_normal), fallback=_orthogonal_vector(loop_normal))


def _loop_plane_cut_points(loop: ContourLine, plane_normal: np.ndarray, plane_pos: np.ndarray) -> list[LoopCut]:
    cut_points: list[LoopCut] = []
    for segment_index, (point_a, point_b) in enumerate(zip(loop.v[:-1], loop.v[1:])):
        cut_point, cut_flag, t = _plane_line_intersection(plane_normal, plane_pos, point_a, point_b)
        if cut_flag != 1:
            continue
        cut_points.append(LoopCut(v=cut_point, segment_index=segment_index))
    return cut_points


def _plane_line_intersection(
    plane_normal: np.ndarray,
    plane_pos: np.ndarray,
    point_0: np.ndarray,
    point_1: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    line_vec = point_1 - point_0
    diff_vec = point_0 - plane_pos
    denominator = float(np.dot(plane_normal, line_vec))
    numerator = -float(np.dot(plane_normal, diff_vec))
    if abs(denominator) < 1e-7:
        return np.zeros(3, dtype=float), 2 if numerator == 0 else 0, 0.0
    t = numerator / denominator
    if t < -1e-7 or t > 1.0 + 1e-7:
        return point_0 + t * line_vec, 3, float(t)
    t = float(np.clip(t, 0.0, 1.0))
    return point_0 + t * line_vec, 1, t


def _projection_fallback_cut_points(loop: ContourLine, cut_plane_direction: np.ndarray) -> list[LoopCut]:
    projections = loop.v @ cut_plane_direction
    min_ind = int(np.argmin(projections))
    max_ind = int(np.argmax(projections))
    if min_ind == max_ind and len(loop.v) > 1:
        max_ind = (min_ind + max(1, len(loop.v) // 2)) % len(loop.v)
    return [_vertex_loop_cut(loop, min_ind), _vertex_loop_cut(loop, max_ind)]


def _vertex_loop_cut(loop: ContourLine, vertex_index: int) -> LoopCut:
    vertex_index = int(vertex_index) % len(loop.v)
    segment_index = min(vertex_index, max(len(loop.v) - 2, 0))
    return LoopCut(v=loop.v[vertex_index].copy(), segment_index=segment_index)


def _dedupe_loop_cuts(cut_points: list[LoopCut]) -> list[LoopCut]:
    deduped: list[LoopCut] = []
    for cut in cut_points:
        previous_point = None if not deduped else deduped[-1].v
        first_point = None if not deduped else deduped[0].v
        if previous_point is not None and np.linalg.norm(cut.v - previous_point) <= 1e-10:
            continue
        if first_point is not None and np.linalg.norm(cut.v - first_point) <= 1e-10 and len(cut_points) > 1:
            continue
        deduped.append(cut)
    return deduped


def _select_cut_pair_by_center(cut_points: list[LoopCut], center: np.ndarray | None) -> tuple[LoopCut, LoopCut]:
    if len(cut_points) < 2:
        raise ValueError("At least two cut points are required.")
    if center is None:
        return cut_points[0], cut_points[1]
    pair_count = max(1, len(cut_points) // 2)
    pairs = [
        (cut_points[2 * pair_ind], cut_points[min(2 * pair_ind + 1, len(cut_points) - 1)])
        for pair_ind in range(pair_count)
    ]
    return min(pairs, key=lambda pair: np.linalg.norm(pair[0].v - center) + np.linalg.norm(pair[1].v - center))


def _assign_high_low_cuts(pair: tuple[LoopCut, LoopCut], cut_direction: np.ndarray) -> tuple[LoopCut, LoopCut]:
    projections = [float(np.dot(pair[0].v, cut_direction)), float(np.dot(pair[1].v, cut_direction))]
    high_ind = int(np.argmin(projections))
    low_ind = 1 - high_ind
    return pair[high_ind], pair[low_ind]


def _calc_local_opening_gap(loop: ContourLine, segment_index: int, opening_gap: float) -> float:
    return float(opening_gap)


def _select_within_group_cut_rail(
    group: LoopGroup,
    cut_infos: list[LoopCutShape],
    config: CoilGenConfig,
    group_ind: int,
    *,
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    allow_direct_fallback: bool = True,
) -> WithinGroupRailCandidate:
    forced_side = _force_cut_selection_for_group(config, group_ind)
    if forced_side is not None:
        return _build_within_group_cut_rail(
            group,
            cut_infos,
            forced_side,
            config,
            geodesic_solver=geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        )

    candidates = [
        _build_within_group_cut_rail(
            group,
            cut_infos,
            "high",
            config,
            geodesic_solver=geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        ),
        _build_within_group_cut_rail(
            group,
            cut_infos,
            "low",
            config,
            geodesic_solver=geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        ),
    ]
    return _select_preferred_within_group_rail_candidate(candidates)


def _select_preferred_within_group_rail_candidate(
    candidates: list[WithinGroupRailCandidate],
) -> WithinGroupRailCandidate:
    if not candidates:
        raise ValueError("At least one within-group rail candidate is required.")
    finite = [candidate for candidate in candidates if math.isfinite(float(candidate.score))]
    if not finite:
        return min(candidates, key=lambda candidate: candidate.score)
    best = min(finite, key=lambda candidate: candidate.score)
    high = next((candidate for candidate in finite if candidate.side == "high"), None)
    if high is not None and _within_group_rail_scores_are_tied(high.score, best.score):
        return high
    return best


def _within_group_rail_scores_are_tied(first: float, second: float) -> bool:
    first = float(first)
    second = float(second)
    if not math.isfinite(first) or not math.isfinite(second):
        return False
    tolerance = max(1.0e-10, 1.0e-6 * max(abs(first), abs(second), 1.0e-12))
    return abs(first - second) <= tolerance


def _within_group_cut_rail_scores(
    group: LoopGroup,
    cut_infos: list[LoopCutShape],
    config: CoilGenConfig,
    *,
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    allow_direct_fallback: bool = True,
) -> dict[str, float]:
    high = _build_within_group_cut_rail(
        group,
        cut_infos,
        "high",
        config,
        geodesic_solver=geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    low = _build_within_group_cut_rail(
        group,
        cut_infos,
        "low",
        config,
        geodesic_solver=geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    return {
        "high": high.score,
        "low": low.score,
        "high_candidate_count": float(high.candidate_count),
        "low_candidate_count": float(low.candidate_count),
    }


def _build_within_group_cut_rail(
    group: LoopGroup,
    cut_infos: list[LoopCutShape],
    side: str,
    config: CoilGenConfig,
    *,
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    allow_direct_fallback: bool = True,
) -> WithinGroupRailCandidate:
    candidates = _build_within_group_cut_rail_candidates(
        group,
        cut_infos,
        side,
        config,
        geodesic_solver=geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
    )
    return min(candidates, key=lambda candidate: candidate.score)


def _build_within_group_cut_rail_candidates(
    group: LoopGroup,
    cut_infos: list[LoopCutShape],
    side: str,
    config: CoilGenConfig,
    *,
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    allow_direct_fallback: bool = True,
) -> list[WithinGroupRailCandidate]:
    if not group.loops or not cut_infos:
        return [
            WithinGroupRailCandidate(
                side=str(side),
                opened_loops=[],
                opening_cuts=[],
                connected_group=ConnectedGroup(
                    v=np.zeros((0, 3), dtype=float),
                    return_path_v=np.zeros((0, 3), dtype=float),
                    spiral_in_v=np.zeros((0, 3), dtype=float),
                    spiral_out_v=np.zeros((0, 3), dtype=float),
                ),
                score=math.inf,
                pair_indices=(),
                candidate_count=0,
            )
        ]

    side = "low" if str(side).lower() == "low" else "high"
    cut_direction = _within_group_cut_direction(cut_infos)
    option_lists = [_loop_cut_pair_options(cut_info, cut_direction, config) for cut_info in cut_infos]
    beam_width = _within_group_candidate_count(config)

    beams: list[tuple[float, list[LoopCutPairOption], list[np.ndarray]]] = [(0.0, [], [])]
    for options in option_lists:
        next_beams: list[tuple[float, list[LoopCutPairOption], list[np.ndarray]]] = []
        for _score, sequence, points in beams:
            for option in options:
                selected_cut = _loop_cut_pair_option_cut(option, side)
                next_points = [*points, selected_cut.v]
                score = _within_group_cut_center_score(np.asarray(next_points, dtype=float))
                next_beams.append((score, [*sequence, option], next_points))
        next_beams.sort(key=lambda beam: (beam[0], sum(option.pair_index for option in beam[1])))
        beams = next_beams[:beam_width]

    baseline = [_preferred_loop_cut_pair_option(options) for options in option_lists]
    if not any(_same_pair_sequence(sequence, baseline) for _score, sequence, _points in beams):
        baseline_points = [_loop_cut_pair_option_cut(option, side).v for option in baseline]
        beams.append(
            (_within_group_cut_center_score(np.asarray(baseline_points, dtype=float)), baseline, baseline_points)
        )

    candidates = [
        _build_within_group_cut_rail_from_options(
            group,
            sequence,
            side,
            config,
            candidate_count=len(beams),
            geodesic_solver=geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        )
        for _score, sequence, _points in beams
    ]
    return candidates


def _build_within_group_cut_rail_from_options(
    group: LoopGroup,
    options: list[LoopCutPairOption],
    side: str,
    config: CoilGenConfig,
    *,
    candidate_count: int,
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    allow_direct_fallback: bool = True,
) -> WithinGroupRailCandidate:
    opened_loops: list[ContourLine] = []
    opening_cuts: list[np.ndarray] = []
    degenerate_count = 0
    opening_threshold = _opening_cut_span_threshold(config)
    for loop, option in zip(group.loops, options):
        selected_cut = _loop_cut_pair_option_cut(option, side)
        opened, cut_points = open_loop_with_3d_sphere(loop, selected_cut.v, _within_group_cut_width(config))
        opened_loops.append(opened)
        opening_cuts.append(cut_points)
        span = _opening_cut_span(cut_points, opening_threshold)
        if not math.isfinite(span) or span <= opening_threshold:
            degenerate_count += 1
    connected = _connect_opened_loops(
        opened_loops,
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
        corner_radius=0.0,
        corner_step_deg=float(config.sweep_bend_arc_angle_deg),
    )
    metrics = _within_group_return_path_metrics(connected.return_path_v)
    score = metrics["score"]
    if len(opened_loops) > 1 and len(connected.v) < 2:
        score = math.inf
        metrics["surface_connection_failed"] = 1.0
        metrics["score"] = float(score)
    if degenerate_count:
        score += 1e9 * degenerate_count
        metrics["degenerate_opening_count"] = float(degenerate_count)
        metrics["score"] = float(score)
    return WithinGroupRailCandidate(
        side=str(side),
        opened_loops=opened_loops,
        opening_cuts=opening_cuts,
        connected_group=connected,
        score=float(score),
        pair_indices=tuple(option.pair_index for option in options),
        candidate_count=int(candidate_count),
        metrics=metrics,
    )


def _within_group_cut_direction(cut_infos: list[LoopCutShape]) -> np.ndarray:
    first = cut_infos[0]
    return _normalize_vector(first.low_cut.v - first.high_cut.v, fallback=np.array([1.0, 0.0, 0.0]))


def _loop_cut_pair_options(
    cut_info: LoopCutShape,
    cut_direction: np.ndarray,
    config: CoilGenConfig,
) -> list[LoopCutPairOption]:
    pair_count = max(1, len(cut_info.cut_points) // 2)
    options: list[LoopCutPairOption] = []
    for pair_index in range(pair_count):
        cut_a = cut_info.cut_points[2 * pair_index]
        cut_b = cut_info.cut_points[min(2 * pair_index + 1, len(cut_info.cut_points) - 1)]
        high_cut, low_cut = _assign_high_low_cuts((cut_a, cut_b), cut_direction)
        options.append(LoopCutPairOption(pair_index=pair_index, high_cut=high_cut, low_cut=low_cut))

    preferred = LoopCutPairOption(pair_index=0, high_cut=cut_info.high_cut, low_cut=cut_info.low_cut)
    if not any(_same_loop_cut_pair_option(option, preferred) for option in options):
        options.append(preferred)

    preferred_center = 0.5 * (cut_info.high_cut.v + cut_info.low_cut.v)
    options.sort(
        key=lambda option: (
            np.linalg.norm(0.5 * (option.high_cut.v + option.low_cut.v) - preferred_center),
            np.linalg.norm(option.high_cut.v - cut_info.high_cut.v)
            + np.linalg.norm(option.low_cut.v - cut_info.low_cut.v),
        )
    )
    return options[: _within_group_candidate_count(config)]


def _preferred_loop_cut_pair_option(options: list[LoopCutPairOption]) -> LoopCutPairOption:
    return options[0]


def _loop_cut_pair_option_cut(option: LoopCutPairOption, side: str) -> LoopCut:
    return option.low_cut if side == "low" else option.high_cut


def _same_loop_cut_pair_option(first: LoopCutPairOption, second: LoopCutPairOption) -> bool:
    return bool(
        np.linalg.norm(first.high_cut.v - second.high_cut.v) <= 1e-10
        and np.linalg.norm(first.low_cut.v - second.low_cut.v) <= 1e-10
    )


def _same_pair_sequence(first: list[LoopCutPairOption], second: list[LoopCutPairOption]) -> bool:
    return len(first) == len(second) and all(_same_loop_cut_pair_option(a, b) for a, b in zip(first, second))


def _within_group_candidate_count(config: CoilGenConfig) -> int:
    return max(int(config.within_group_cut_candidate_count), 1)


def _within_group_cut_center_score(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return _within_group_return_path_score(points)


def _within_group_return_path_score(return_path_v: np.ndarray) -> float:
    return _within_group_return_path_metrics(return_path_v)["score"]


def _within_group_return_path_metrics(return_path_v: np.ndarray) -> dict[str, float]:
    points = np.asarray(return_path_v, dtype=float)
    if len(points) < 2:
        return {
            "score": 0.0,
            "path_length": 0.0,
            "endpoint_distance": 0.0,
            "excess_length": 0.0,
            "turn_penalty": 0.0,
            "line_deviation": 0.0,
            "mean_segment_length": 0.0,
        }
    path_length = _path_length(points)
    endpoint_distance = float(np.linalg.norm(points[-1] - points[0]))
    if not math.isfinite(path_length) or endpoint_distance <= 1e-15:
        return {
            "score": math.inf,
            "path_length": float(path_length),
            "endpoint_distance": float(endpoint_distance),
            "excess_length": math.inf,
            "turn_penalty": math.inf,
            "line_deviation": math.inf,
            "mean_segment_length": 0.0,
        }
    excess_length = max(path_length - endpoint_distance, 0.0)
    turn_penalty = _return_path_turn_penalty(points)
    line_deviation = _return_path_line_deviation(points)
    mean_segment_length = path_length / max(len(points) - 1, 1)
    score = path_length + 3.0 * excess_length + mean_segment_length * turn_penalty + 2.0 * line_deviation
    return {
        "score": float(score),
        "path_length": float(path_length),
        "endpoint_distance": float(endpoint_distance),
        "excess_length": float(excess_length),
        "turn_penalty": float(turn_penalty),
        "line_deviation": float(line_deviation),
        "mean_segment_length": float(mean_segment_length),
    }


def _return_path_turn_penalty(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    penalty = 0.0
    for prev_point, point, next_point in zip(points[:-2], points[1:-1], points[2:]):
        incoming = point - prev_point
        outgoing = next_point - point
        incoming_norm = np.linalg.norm(incoming)
        outgoing_norm = np.linalg.norm(outgoing)
        if incoming_norm <= 1e-15 or outgoing_norm <= 1e-15:
            continue
        dot = float(np.clip(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm), -1.0, 1.0))
        penalty += 1.0 - dot
    return float(penalty)


def _return_path_line_deviation(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    line = points[-1] - points[0]
    line_length = np.linalg.norm(line)
    if line_length <= 1e-15:
        return math.inf
    line_unit = line / line_length
    rel = points[1:-1] - points[0]
    projected = np.outer(rel @ line_unit, line_unit)
    distances = np.linalg.norm(rel - projected, axis=1)
    return float(np.sqrt(np.mean(distances * distances))) if len(distances) else 0.0


def _force_cut_selection_for_group(config: CoilGenConfig, group_ind: int) -> str | None:
    values = [str(value).lower() for value in config.force_cut_selection]
    if not values:
        return None
    if len(values) == 1:
        return values[0] if values[0] in {"high", "low"} else None
    if group_ind < len(values) and values[group_ind] in {"high", "low"}:
        return values[group_ind]
    return None


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-15:
        fallback = np.asarray(fallback, dtype=float)
        fallback_norm = np.linalg.norm(fallback)
        return fallback / fallback_norm if fallback_norm > 1e-15 else np.array([1.0, 0.0, 0.0])
    return vector / norm


def _mesh_face_normals(mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-15
    out = np.zeros_like(normals)
    out[valid] = normals[valid] / norms[valid, None]
    return out


def _polyline_area_vector(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.zeros(3, dtype=float)
    return np.sum(np.cross(points[:-1], points[1:]), axis=0)


def _orthogonal_vector(vector: np.ndarray) -> np.ndarray:
    vector = _normalize_vector(vector, fallback=np.array([0.0, 0.0, 1.0]))
    helper = np.array([1.0, 0.0, 0.0]) if abs(vector[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _normalize_vector(np.cross(vector, helper), fallback=np.array([0.0, 1.0, 0.0]))


def shift_return_paths(parts: list[CoilPart], config: CoilGenConfig) -> list[CoilPart]:
    """Resolve wire crossings with one local normal-direction overpass.

    Long within-group returns have precedence over contour-to-contour routing.
    Inter-group connectors are lifted only where they cross another conductor.
    The paired underpass is explicitly held on the support surface.
    """

    for part in parts:
        if part.wire_path is None:
            continue
        n_points = len(part.wire_path.v)
        if config.skip_normal_shift or n_points == 0:
            part.shift_array = np.zeros(n_points, dtype=float)
            part.points_to_shift = np.zeros(n_points, dtype=bool)
            continue

        base_shift_levels = np.zeros(n_points, dtype=float)
        overpass_points = np.zeros(n_points, dtype=bool)
        underpass_points = np.zeros(n_points, dtype=bool)
        unshifted_wire_v = part.wire_path.v.copy()
        contact_threshold = _wire_contact_threshold(config)
        transition_length = abs(float(config.normal_shift_length)) * max(
            float(config.normal_sweep_smooth_factor),
            0.0,
        )
        _mark_within_group_return_levels(part, base_shift_levels)
        routes_by_segment = _normal_shift_routes_by_segment(part)
        routes_by_segment = _extend_normal_shift_routes_by_arclength(
            routes_by_segment,
            unshifted_wire_v,
            contact_threshold,
        )
        contacts = find_wire_segment_contacts_3d(unshifted_wire_v, contact_threshold, config=config)
        crossing_events = _crossing_event_segment_selections(
            contacts,
            routes_by_segment,
            unshifted_wire_v,
            max(contact_threshold, transition_length),
        )
        for event_overpasses, event_underpasses in crossing_events:
            for segment_index in event_overpasses:
                _mark_segment_points(overpass_points, segment_index)
            for segment_index in event_underpasses:
                _mark_segment_points(underpass_points, segment_index)

        shift_array = _crossing_shift_profile(
            base_shift_levels,
            overpass_points,
            underpass_points,
            unshifted_wire_v,
            transition_length,
            config.normal_shift_smooth_factors,
        )
        normals = _smooth_wire_path_normals(
            wire_path_normals(part),
            max(int(config.sweep_surface_normal_smoothing_iterations), 0),
        )
        shift_length = float(config.normal_shift_length)
        shifted_wire_v = unshifted_wire_v + normals * (shift_array[:, None] * shift_length)
        part.wire_self_contacts = contacts
        if np.any(shift_array > 0):
            shifted_geometry = [
                (
                    WirePathGeometry.FREE_3D
                    if geometry == WirePathGeometry.FREE_3D
                    else WirePathGeometry.OFFSET_SURFACE
                    if max(abs(float(shift_array[index])), abs(float(shift_array[index + 1]))) > 1e-12
                    else WirePathGeometry.SURFACE
                )
                for index, geometry in enumerate(part.wire_path.segment_geometry)
            ]
            shifted_path = WirePath(
                v=shifted_wire_v,
                segment_owners=list(part.wire_path.segment_owners),
                segment_geometry=shifted_geometry,
            )
            part.wire_path = shifted_path
            _update_inter_group_paths_after_shift(part, unshifted_wire_v, shifted_wire_v)
            shifted_wire_v, shift_array = _repair_shifted_wire_reversal_spurs(
                shifted_wire_v,
                shift_array,
                config,
            )
            repaired_owners = transfer_segment_owners(
                shifted_path.v,
                shifted_path.segment_owners,
                shifted_wire_v,
            )
            repaired_geometry = transfer_segment_geometry(
                shifted_path.v,
                shifted_path.segment_geometry,
                shifted_wire_v,
            )
            part.wire_path = WirePath(
                v=shifted_wire_v,
                segment_owners=repaired_owners,
                segment_geometry=repaired_geometry,
            )
        points_to_shift = shift_array > 0.0
        part.shift_array = shift_array
        part.points_to_shift = points_to_shift
    return parts


def cleanup_wire_paths(
    parts: list[CoilPart],
    tolerance: float | CoilGenConfig = 1e-10,
    config: CoilGenConfig | None = None,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    """Remove degenerate wire-path points, close paths, and store route diagnostics."""

    if isinstance(tolerance, CoilGenConfig) and config is None:
        config = tolerance
        tolerance = 1e-10
    tolerance = float(tolerance)
    task = progress.task("route cleanup diagnostics", len(parts)) if progress else None
    for part_ind, part in enumerate(parts):
        if task is not None:
            point_count = 0 if part.wire_path is None else len(part.wire_path.v)
            task.update(
                part_ind,
                message=(
                    f"part {part_ind + 1}/{len(parts)}; wire points={point_count}; "
                    f"inter-group cuts={len(part.opening_cuts_among_groups)}"
                ),
                force=part_ind == 0,
            )
        if part.wire_path is None:
            part.route_diagnostics = route_diagnostics(part, config)
            _raise_on_bad_route_quality(part, config)
            if task is not None:
                task.advance(1, message=f"part {part_ind + 1}/{len(parts)} has no connected wire path")
            continue
        source_path = part.wire_path
        cleaned = _clean_path_array(source_path.v, tolerance)
        cleaned = _ensure_closed_path_array(cleaned, tolerance)
        cleaned_owners = transfer_segment_owners(source_path.v, source_path.segment_owners, cleaned)
        cleaned_geometry = transfer_segment_geometry(source_path.v, source_path.segment_geometry, cleaned)
        part.wire_path = WirePath(
            v=cleaned,
            segment_owners=cleaned_owners,
            segment_geometry=cleaned_geometry,
        )
        if part.shift_array is not None and len(part.shift_array) == len(source_path.v):
            part.shift_array = _transfer_point_scalars(source_path.v, part.shift_array, cleaned)
            part.points_to_shift = part.shift_array > 0.0
        part.route_diagnostics = route_diagnostics(part, config)
        _raise_on_bad_route_quality(part, config)
        if task is not None:
            task.advance(
                1,
                message=(
                    f"part {part_ind + 1}/{len(parts)} cleaned; "
                    f"route status={part.route_diagnostics.get('route_quality_status', 'unknown')}"
                ),
            )
    if task is not None:
        task.done()
    return parts


def route_diagnostics(part: CoilPart, config: CoilGenConfig | None = None) -> dict[str, float | int | bool | str]:
    inter_group = _inter_group_route_diagnostics(part, config)
    route_quality = _empty_route_quality_diagnostics(config)
    opening_cuts = _opening_cut_route_diagnostics(part, config=config)
    quality = _route_quality_summary(route_quality, opening_cuts, inter_group)
    if part.wire_path is None or len(part.wire_path.v) < 2:
        return {
            "point_count": 0,
            "segment_count": 0,
            "length": 0.0,
            "min_segment_length": 0.0,
            "closed_v": False,
            "self_contact_count": int(len(part.wire_self_contacts)),
            "shifted_point_count": int(np.count_nonzero(part.points_to_shift))
            if part.points_to_shift is not None
            else 0,
            "max_normal_shift_factor": float(np.max(part.shift_array))
            if part.shift_array is not None and len(part.shift_array)
            else 0.0,
            **route_quality,
            **opening_cuts,
            **inter_group,
            **quality,
        }
    segment_lengths = np.linalg.norm(np.diff(part.wire_path.v, axis=0), axis=1)
    positive_lengths = segment_lengths[segment_lengths > 1e-15]
    min_seg_a, min_seg_b, min_wire_distance = find_min_wire_segment_distance_3d(part.wire_path.v, config=config)
    route_quality = _route_quality_diagnostics(part, min_seg_a, min_seg_b, min_wire_distance, config)
    quality = _route_quality_summary(route_quality, opening_cuts, inter_group)
    return {
        "point_count": int(len(part.wire_path.v)),
        "segment_count": int(max(len(part.wire_path.v) - 1, 0)),
        "length": float(segment_lengths.sum()),
        "min_segment_length": float(np.min(positive_lengths)) if len(positive_lengths) else 0.0,
        "closed_v": bool(np.linalg.norm(part.wire_path.v[0] - part.wire_path.v[-1]) <= 1e-8),
        "self_contact_count": int(len(part.wire_self_contacts)),
        "shifted_point_count": int(np.count_nonzero(part.points_to_shift)) if part.points_to_shift is not None else 0,
        "max_normal_shift_factor": float(np.max(part.shift_array))
        if part.shift_array is not None and len(part.shift_array)
        else 0.0,
        **route_quality,
        **opening_cuts,
        **inter_group,
        **quality,
    }


def _empty_route_quality_diagnostics(config: CoilGenConfig | None = None) -> dict[str, float | int]:
    return {
        "min_wire_wire_distance": 0.0,
        "min_wire_wire_segment_a": -1,
        "min_wire_wire_segment_b": -1,
        "wire_clearance_failure_count": 0,
        "wire_clearance_failure_threshold": float(_wire_clearance_failure_threshold(config)),
        "wire_clearance_warning_count": 0,
        "wire_clearance_warning_threshold": float(_wire_clearance_warning_threshold(config)),
        "min_inter_group_connector_to_contour_distance": 0.0,
        "min_inter_group_connector_to_contour_cut_index": -1,
        "min_inter_group_connector_to_contour_loop_index": -1,
        "inter_group_connector_clearance_violation_count": 0,
    }


def _route_quality_diagnostics(
    part: CoilPart,
    min_seg_a: int,
    min_seg_b: int,
    min_wire_distance: float,
    config: CoilGenConfig | None,
) -> dict[str, float | int]:
    connector_distance, cut_index, loop_index = _min_connector_interior_to_contour_distance(part)
    clearance_threshold = _connector_clearance_threshold(config)
    wire_failure_threshold = _wire_clearance_failure_threshold(config)
    wire_warning_threshold = _wire_clearance_warning_threshold(config)
    valid_wire_pair = min_seg_a >= 0 and min_seg_b >= 0
    wire_failure = valid_wire_pair and wire_failure_threshold > 0.0 and min_wire_distance < wire_failure_threshold
    return {
        "min_wire_wire_distance": float(min_wire_distance),
        "min_wire_wire_segment_a": int(min_seg_a),
        "min_wire_wire_segment_b": int(min_seg_b),
        "wire_clearance_failure_count": int(wire_failure),
        "wire_clearance_failure_threshold": float(wire_failure_threshold),
        "wire_clearance_warning_count": int(valid_wire_pair and min_wire_distance < wire_warning_threshold),
        "wire_clearance_warning_threshold": float(wire_warning_threshold),
        "min_inter_group_connector_to_contour_distance": float(connector_distance),
        "min_inter_group_connector_to_contour_cut_index": int(cut_index),
        "min_inter_group_connector_to_contour_loop_index": int(loop_index),
        "inter_group_connector_clearance_violation_count": int(
            _connector_clearance_violation_count(part, clearance_threshold)
        ),
    }


def _inter_group_route_diagnostics(part: CoilPart, config: CoilGenConfig | None) -> dict[str, float | int | bool]:
    cuts = part.opening_cuts_among_groups
    connector_paths = [path_v for cut in cuts for path_v in _inter_group_connection_paths(cut)]
    strict_surface = config is not None and _strict_geodesic_inter_group_paths(config)
    direct_distances = [float(cut.direct_distance) for cut in cuts if math.isfinite(cut.direct_distance)]
    geodesic_distances = [float(cut.geodesic_distance) for cut in cuts if math.isfinite(cut.geodesic_distance)]
    connector_distances = [_path_length(path_v) for path_v in connector_paths if len(path_v) >= 2]
    connector_point_counts = [len(path_v) for path_v in connector_paths]
    connector_ratios = [_inter_group_connector_ratio(cut) for cut in cuts]
    finite_ratios = [ratio for ratio in connector_ratios if math.isfinite(ratio)]
    suspicious_ratio = _suspicious_connector_ratio(config)
    forbidden_direct_count = int(
        strict_surface and sum(cut.connection_method != "geodesic" or cut.used_direct_fallback for cut in cuts)
    )
    return {
        "inter_group_cut_count": int(len(cuts)),
        "inter_group_geodesic_connector_count": int(sum(cut.connection_method == "geodesic" for cut in cuts)),
        "inter_group_direct_connector_count": int(sum(cut.connection_method == "direct_3d" for cut in cuts)),
        "inter_group_direct_fallback_count": int(sum(cut.used_direct_fallback for cut in cuts)),
        "surface_geodesic_connector_failure_count": forbidden_direct_count,
        "inter_group_clearance_fallback_count": int(sum(cut.used_clearance_fallback for cut in cuts)),
        "inter_group_connector_point_count": int(sum(connector_point_counts)),
        "inter_group_connector_path_count": int(len(connector_paths)),
        "inter_group_closing_connector_point_count": int(sum(len(cut.closing_connection_path_v) for cut in cuts)),
        "suspicious_inter_group_connector_count": int(sum(ratio > suspicious_ratio for ratio in connector_ratios)),
        "suspicious_inter_group_connector_ratio_threshold": float(suspicious_ratio),
        "max_inter_group_connector_ratio": max(finite_ratios) if finite_ratios else 0.0,
        "max_inter_group_direct_distance": max(direct_distances) if direct_distances else 0.0,
        "max_inter_group_geodesic_distance": max(geodesic_distances) if geodesic_distances else 0.0,
        "max_inter_group_connector_distance": max(connector_distances) if connector_distances else 0.0,
    }


def _opening_cut_route_diagnostics(
    part: CoilPart,
    tolerance: float | None = None,
    config: CoilGenConfig | None = None,
) -> dict[str, float | int]:
    if tolerance is None:
        tolerance = _opening_cut_span_threshold(config)
    within_spans = [_opening_cut_span(cut_v, tolerance) for group in part.groups for cut_v in group.opening_cuts_v]
    inter_spans = [
        _opening_cut_span(cut_v, tolerance)
        for cut in part.opening_cuts_among_groups
        for cut_v in (cut.cut1_v, cut.cut2_v)
    ]
    all_spans = within_spans + inter_spans
    finite_spans = [span for span in all_spans if math.isfinite(span)]
    finite_within_spans = [span for span in within_spans if math.isfinite(span)]
    finite_inter_spans = [span for span in inter_spans if math.isfinite(span)]
    within_group_cut_scores = [
        float(group.within_group_cut_score)
        for group in part.groups
        if math.isfinite(float(group.within_group_cut_score))
    ]
    within_group_candidate_counts = [
        float(value)
        for group in part.groups
        for key, value in group.within_group_cut_scores.items()
        if key.endswith("_candidate_count") and math.isfinite(float(value))
    ]
    within_group_line_deviations = [
        float(group.within_group_cut_metrics.get("line_deviation", 0.0))
        for group in part.groups
        if math.isfinite(float(group.within_group_cut_metrics.get("line_deviation", 0.0)))
    ]
    within_group_excess_lengths = [
        float(group.within_group_cut_metrics.get("excess_length", 0.0))
        for group in part.groups
        if math.isfinite(float(group.within_group_cut_metrics.get("excess_length", 0.0)))
    ]
    return {
        "opening_cut_count": int(len(all_spans)),
        "degenerate_opening_cut_count": int(sum(not math.isfinite(span) or span <= tolerance for span in all_spans)),
        "within_group_opening_cut_count": int(len(within_spans)),
        "within_group_degenerate_opening_cut_count": int(
            sum(not math.isfinite(span) or span <= tolerance for span in within_spans)
        ),
        "inter_group_opening_cut_count": int(len(inter_spans)),
        "inter_group_degenerate_opening_cut_count": int(
            sum(not math.isfinite(span) or span <= tolerance for span in inter_spans)
        ),
        "within_group_high_cut_group_count": int(sum(group.within_group_cut_side == "high" for group in part.groups)),
        "within_group_low_cut_group_count": int(sum(group.within_group_cut_side == "low" for group in part.groups)),
        "within_group_fallback_cut_group_count": int(
            sum(group.within_group_cut_side == "fallback" for group in part.groups)
        ),
        "max_within_group_cut_score": max(within_group_cut_scores) if within_group_cut_scores else 0.0,
        "max_within_group_cut_candidate_count": max(within_group_candidate_counts)
        if within_group_candidate_counts
        else 0.0,
        "max_within_group_return_line_deviation": max(within_group_line_deviations)
        if within_group_line_deviations
        else 0.0,
        "max_within_group_return_excess_length": max(within_group_excess_lengths)
        if within_group_excess_lengths
        else 0.0,
        "requested_within_group_cut_width": _within_group_cut_width(config) if config is not None else 0.0,
        "min_within_group_opening_cut_span": min(finite_within_spans) if finite_within_spans else 0.0,
        "max_within_group_opening_cut_span": max(finite_within_spans) if finite_within_spans else 0.0,
        "requested_inter_group_cut_width": _inter_group_cut_width(config) if config is not None else 0.0,
        "min_inter_group_opening_cut_span": min(finite_inter_spans) if finite_inter_spans else 0.0,
        "max_inter_group_opening_cut_span": max(finite_inter_spans) if finite_inter_spans else 0.0,
        "min_opening_cut_span": min(finite_spans) if finite_spans else 0.0,
        "max_opening_cut_span": max(finite_spans) if finite_spans else 0.0,
        "min_opening_cut_span_threshold": float(tolerance),
    }


def _opening_cut_span(cut_v: np.ndarray, tolerance: float) -> float:
    cut_v = np.asarray(cut_v, dtype=float)
    if cut_v.ndim != 2 or cut_v.shape[1] != 3 or len(cut_v) < 2:
        return math.inf
    cut_v = cut_v[np.all(np.isfinite(cut_v), axis=1)]
    if len(cut_v) < 2:
        return math.inf
    return float(np.linalg.norm(cut_v[0] - cut_v[1]))


def _inter_group_connector_ratio(cut: InterGroupCut) -> float:
    return max(
        (
            _connector_ratio(_path_length(path_v), cut.direct_distance, cut.geodesic_distance)
            for path_v in _inter_group_connection_paths(cut)
        ),
        default=0.0,
    )


def _connector_ratio(connector_distance: float, direct_distance: float, geodesic_distance: float) -> float:
    expected = geodesic_distance if math.isfinite(geodesic_distance) and geodesic_distance > 1e-15 else direct_distance
    if not math.isfinite(expected) or expected <= 1e-15:
        return 0.0
    return float(connector_distance / expected)


def _min_connector_interior_to_contour_distance(part: CoilPart) -> tuple[float, int, int]:
    best_distance = math.inf
    best_cut = -1
    best_loop = -1
    if not part.contour_lines:
        return 0.0, -1, -1
    for cut_ind, cut in enumerate(part.opening_cuts_among_groups):
        excluded = _inter_group_endpoint_loop_indices(part, cut)
        for path_v in _inter_group_connection_paths(cut):
            distance, loop_ind = _connector_path_clearance_to_unrelated_contours(
                part, path_v, excluded, return_loop=True
            )
            if distance < best_distance:
                best_distance = distance
                best_cut = cut_ind
                best_loop = loop_ind
    if not math.isfinite(best_distance):
        return 0.0, -1, -1
    return float(best_distance), best_cut, best_loop


def _connector_clearance_violation_count(part: CoilPart, clearance_threshold: float) -> int:
    if clearance_threshold <= 0.0:
        return 0
    count = 0
    if not part.contour_lines:
        return 0
    for cut in part.opening_cuts_among_groups:
        excluded = _inter_group_endpoint_loop_indices(part, cut)
        count += sum(
            _connector_path_clearance_to_unrelated_contours(part, path_v, excluded) < clearance_threshold
            for path_v in _inter_group_connection_paths(cut)
        )
    return count


def _inter_group_endpoint_loop_indices(part: CoilPart, cut: InterGroupCut) -> set[int]:
    selected = {
        int(loop_index)
        for loop_index in (cut.loop1_index, cut.loop2_index)
        if 0 <= int(loop_index) < len(part.contour_lines)
    }
    if selected:
        return selected
    return _loop_indices_for_groups(part, (cut.group1_index, cut.group2_index))


def _inter_group_connection_paths(cut: InterGroupCut, *, include_auxiliary: bool = False) -> list[np.ndarray]:
    """Return the distinct routed bridges represented by an inter-group cut."""

    candidates = [cut.connection_path_v, cut.closing_connection_path_v]
    if include_auxiliary:
        candidates.extend(cut.normal_shift_paths_v)
    return _distinct_connection_paths(candidates)


def _route_quality_summary(
    route_quality: dict[str, float | int],
    opening_cuts: dict[str, float | int],
    inter_group: dict[str, float | int | bool],
) -> dict[str, bool | int | str]:
    clearance_violations = int(route_quality.get("inter_group_connector_clearance_violation_count", 0))
    wire_clearance_failures = int(route_quality.get("wire_clearance_failure_count", 0))
    degenerate_cuts = int(opening_cuts.get("degenerate_opening_cut_count", 0))
    suspicious_connectors = int(inter_group.get("suspicious_inter_group_connector_count", 0))
    surface_connector_failures = int(inter_group.get("surface_geodesic_connector_failure_count", 0))
    failures = clearance_violations + wire_clearance_failures + degenerate_cuts + surface_connector_failures
    topology_failures = degenerate_cuts + surface_connector_failures
    warnings = (
        int(route_quality.get("wire_clearance_warning_count", 0))
        + int(inter_group.get("inter_group_clearance_fallback_count", 0))
        + int(inter_group.get("inter_group_direct_fallback_count", 0))
        + suspicious_connectors
    )
    failure_reasons: list[str] = []
    if wire_clearance_failures:
        failure_reasons.append("wire_clearance_target")
    if clearance_violations:
        failure_reasons.append("connector_clearance_target")
    if degenerate_cuts:
        failure_reasons.append("degenerate_opening_cut")
    if surface_connector_failures:
        failure_reasons.append("surface_geodesic_connector")
    status = "fail" if failures else "warn" if warnings else "pass"
    return {
        "route_quality_ok": bool(failures == 0),
        "route_quality_failure_count": int(failures),
        "route_quality_failure_reasons": ";".join(failure_reasons),
        "route_quality_warning_count": int(warnings),
        "route_quality_status": status,
        "route_topology_ok": bool(topology_failures == 0),
        "route_topology_failure_count": int(topology_failures),
    }


def _raise_on_bad_route_quality(part: CoilPart, config: CoilGenConfig | None) -> None:
    if config is None or not bool(config.fail_on_route_quality_issues):
        return
    diagnostics = part.route_diagnostics
    if diagnostics.get("route_quality_ok", True):
        return
    raise ValueError(
        "Route quality checks failed: "
        f"wire_clearance_failure_count={diagnostics.get('wire_clearance_failure_count', 0)}, "
        f"degenerate_opening_cut_count={diagnostics.get('degenerate_opening_cut_count', 0)}, "
        f"suspicious_inter_group_connector_count={diagnostics.get('suspicious_inter_group_connector_count', 0)}, "
        f"surface_geodesic_connector_failure_count={diagnostics.get('surface_geodesic_connector_failure_count', 0)}, "
        "inter_group_connector_clearance_violation_count="
        f"{diagnostics.get('inter_group_connector_clearance_violation_count', 0)}."
    )


def _suspicious_connector_ratio(config: CoilGenConfig | None) -> float:
    if config is None:
        return 2.0
    return max(float(config.suspicious_inter_group_connector_ratio), 1.0)


def _connector_clearance_threshold(config: CoilGenConfig | None) -> float:
    if config is None:
        return 0.0
    return max(float(config.min_inter_group_connector_clearance), 0.0)


def _wire_clearance_warning_threshold(config: CoilGenConfig | None) -> float:
    if config is None:
        return 0.0
    return max(float(config.conductor_cross_section_width), 0.0)


def _wire_clearance_failure_threshold(config: CoilGenConfig | None) -> float:
    if config is None:
        return 0.0
    return max(
        float(getattr(config, "min_wire_distance_m", 0.0)), float(getattr(config, "min_turn_distance", 0.0)), 0.0
    )


def _inter_group_candidate_count(config: CoilGenConfig | None) -> int:
    if config is None:
        return 1
    return max(int(config.inter_group_connection_candidate_count), 1)


def _within_group_cut_width(config: CoilGenConfig) -> float:
    override = getattr(config, "within_group_cut_width", None)
    return float(config.interconnection_cut_width if override is None else override)


def _inter_group_cut_width(config: CoilGenConfig) -> float:
    override = getattr(config, "inter_group_cut_width", None)
    return float(config.interconnection_cut_width if override is None else override)


def _opening_cut_span_threshold(config: CoilGenConfig | None) -> float:
    if config is None:
        return 1e-10
    return max(float(config.min_opening_cut_span), 0.0)


def _loop_indices_for_groups(part: CoilPart, group_indices: tuple[int, ...]) -> set[int]:
    loop_indices: set[int] = set()
    for group_ind in group_indices:
        if 0 <= int(group_ind) < len(part.groups):
            loop_indices.update(int(loop_ind) for loop_ind in part.groups[int(group_ind)].loop_indices)
    return loop_indices


def _loop_index_at_point(
    part: CoilPart,
    point_v: np.ndarray,
    candidate_loop_indices: set[int] | tuple[int, ...],
) -> int:
    point_v = np.asarray(point_v, dtype=float).reshape(3)
    best_distance = math.inf
    best_loop_index = -1
    for loop_index in candidate_loop_indices:
        if not (0 <= int(loop_index) < len(part.contour_lines)):
            continue
        contour_v = np.asarray(part.contour_lines[int(loop_index)].v, dtype=float)
        if len(contour_v) < 2:
            continue
        distance = _point_polyline_distance_3d(point_v, contour_v)
        if distance < best_distance:
            best_distance = distance
            best_loop_index = int(loop_index)
    scale = max(float(np.linalg.norm(np.ptp(part.mesh.vertices, axis=0))), 1.0)
    return best_loop_index if best_distance <= max(1.0e-8, 1.0e-9 * scale) else -1


def _connector_path_clearance_to_unrelated_contours(
    part: CoilPart,
    connector_path_v: np.ndarray,
    excluded_loop_indices: set[int],
    *,
    return_loop: bool = False,
) -> float | tuple[float, int]:
    connector_path_v = np.asarray(connector_path_v, dtype=float)
    if len(connector_path_v) < 2:
        result: float | tuple[float, int] = (math.inf, -1) if return_loop else math.inf
        return result

    best_distance = math.inf
    best_loop = -1
    for loop_ind, contour in enumerate(part.contour_lines):
        if loop_ind in excluded_loop_indices or len(contour.v) < 2:
            continue
        distance = _polyline_polyline_distance_3d(connector_path_v, contour.v)
        if distance < best_distance:
            best_distance = distance
            best_loop = loop_ind
    if return_loop:
        return float(best_distance), best_loop
    return float(best_distance)


def _polyline_polyline_distance_3d(polyline_a: np.ndarray, polyline_b: np.ndarray) -> float:
    polyline_a = np.asarray(polyline_a, dtype=float)
    polyline_b = np.asarray(polyline_b, dtype=float)
    if len(polyline_a) == 0 or len(polyline_b) == 0:
        return math.inf
    if len(polyline_a) == 1:
        return _point_polyline_distance_3d(polyline_a[0], polyline_b)
    if len(polyline_b) == 1:
        return _point_polyline_distance_3d(polyline_b[0], polyline_a)

    best_distance = math.inf
    for start_a, end_a in zip(polyline_a[:-1], polyline_a[1:]):
        if np.linalg.norm(end_a - start_a) <= 1e-12:
            continue
        for start_b, end_b in zip(polyline_b[:-1], polyline_b[1:]):
            if np.linalg.norm(end_b - start_b) <= 1e-12:
                continue
            distance = _segment_segment_distance_3d(start_a, end_a, start_b, end_b)
            if distance < best_distance:
                best_distance = distance
    return float(best_distance)


def _clean_path_array(v: np.ndarray, tolerance: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.zeros((0, 3), dtype=float)
    finite = np.all(np.isfinite(v), axis=1)
    v = v[finite]
    if len(v) < 2:
        return v.copy()
    keep = np.ones(len(v), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(v, axis=0), axis=1) > tolerance
    return _remove_immediate_retraces(v[keep], tolerance)


def _transfer_point_scalars(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Transfer point data when cleanup retains a subset of route vertices."""

    source = np.asarray(source_points, dtype=float)
    values = np.asarray(source_values, dtype=float).reshape(-1)
    target = np.asarray(target_points, dtype=float)
    if len(source) != len(values):
        raise ValueError("source_values must contain one value per source point")
    if len(target) == 0:
        return np.zeros(0, dtype=float)
    try:
        from scipy.spatial import cKDTree

        _distance, indices = cKDTree(source).query(target, k=1)
        return values[np.asarray(indices, dtype=np.int64)]
    except ImportError:
        output = np.empty(len(target), dtype=float)
        for start in range(0, len(target), 256):
            stop = min(start + 256, len(target))
            distance_sq = np.sum((target[start:stop, None, :] - source[None, :, :]) ** 2, axis=2)
            output[start:stop] = values[np.argmin(distance_sq, axis=1)]
        return output


def _remove_immediate_retraces(v: np.ndarray, tolerance: float) -> np.ndarray:
    """Collapse A-B-A spurs, which retrace a conductor segment exactly."""

    reduced: list[np.ndarray] = []
    for point in np.asarray(v, dtype=float):
        if len(reduced) >= 2 and np.linalg.norm(point - reduced[-2]) <= tolerance:
            reduced.pop()
            continue
        reduced.append(point)
    return np.asarray(reduced, dtype=float).reshape((-1, 3))


def _is_closed_path_array(v: np.ndarray, tolerance: float) -> bool:
    v = np.asarray(v, dtype=float)
    return bool(len(v) > 1 and np.linalg.norm(v[0] - v[-1]) <= float(tolerance))


def _ensure_closed_path_array(v: np.ndarray, tolerance: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return v.copy()
    if _is_closed_path_array(v, tolerance):
        closed = v.copy()
        closed[-1] = closed[0]
        return closed
    return np.vstack((v, v[0]))


def find_wire_segment_contacts_3d(
    points: np.ndarray,
    threshold: float,
    config: CoilGenConfig | None = None,
) -> list[tuple[int, int, float]]:
    """Find non-adjacent 3D wire segments closer than ``threshold``."""

    points = np.asarray(points, dtype=float)
    threshold = max(float(threshold), 0.0)
    n_segments = max(0, len(points) - 1)
    if n_segments < 2:
        return []
    starts, directions, valid_segments, closed = _wire_segment_data(points)
    local_exclusion = _wire_local_continuation_length(threshold)
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
    """Return the closest non-adjacent pair of 3D wire segments."""

    points = np.asarray(points, dtype=float)
    n_segments = max(0, len(points) - 1)
    if n_segments < 2:
        return -1, -1, 0.0
    starts, directions, valid_segments, closed = _wire_segment_data(points)
    local_exclusion = _wire_local_continuation_length(_wire_clearance_failure_threshold(config))
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
    n_segments = len(starts)
    if n_segments < 2 or not math.isfinite(max_distance):
        max_distance = math.inf
    mins, maxs = _wire_segment_aabbs(starts, directions)
    valid_indices = np.flatnonzero(valid_segments)
    if len(valid_indices) < 2:
        return []
    span = np.ptp(maxs[valid_indices], axis=0)
    axis = int(np.argmax(span))
    order = valid_indices[np.argsort(mins[valid_indices, axis], kind="mergesort")]
    max_distance = max(float(max_distance), 0.0)
    max_distance_squared = max_distance * max_distance if math.isfinite(max_distance) else math.inf
    chunk_size = max(int(getattr(config, "wire_distance_chunk_size", 512)), 1)
    max_candidate_count = None
    if max_candidate_fraction is not None:
        all_pair_count = max((n_segments - 1) * (n_segments - 2) // 2, 0)
        max_candidate_count = max(int(float(max_candidate_fraction) * float(all_pair_count)), chunk_size)
    candidate_count = 0
    segment_length_prefix = _wire_segment_length_prefix(directions)
    chunks: list[np.ndarray] = []
    current: list[tuple[int, int]] = []

    for order_pos, seg_i in enumerate(order):
        limit = maxs[seg_i, axis] + max_distance
        next_pos = order_pos + 1
        while next_pos < len(order):
            seg_j = int(order[next_pos])
            if mins[seg_j, axis] > limit:
                break
            seg_a = int(min(seg_i, seg_j))
            seg_b = int(max(seg_i, seg_j))
            if _wire_segment_pair_allowed(
                seg_a,
                seg_b,
                n_segments,
                closed,
                directions=directions,
                local_exclusion=local_exclusion,
                segment_length_prefix=segment_length_prefix,
            ):
                if _aabb_distance_squared(mins[seg_a], maxs[seg_a], mins[seg_b], maxs[seg_b]) <= max_distance_squared:
                    candidate_count += 1
                    if max_candidate_count is not None and candidate_count > max_candidate_count:
                        return None
                    current.append((seg_a, seg_b))
                    if len(current) >= chunk_size:
                        chunks.append(np.asarray(current, dtype=np.int64))
                        current = []
            next_pos += 1
    if current:
        chunks.append(np.asarray(current, dtype=np.int64))
    return chunks


def _wire_segment_pair_allowed(
    seg_a: int,
    seg_b: int,
    n_segments: int,
    closed: bool,
    *,
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
        return False
    return True


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
    candidate_pairs: list[tuple[int, int]] = []

    def flush_candidates() -> None:
        nonlocal best, candidate_pairs
        if not candidate_pairs:
            return
        _seg_a, _seg_b, distance = _min_wire_segment_distance_pair_chunk(
            starts,
            directions,
            np.asarray(candidate_pairs, dtype=np.int64),
        )
        if distance < best:
            best = float(distance)
        candidate_pairs = []

    for axis in range(3):
        order = valid_indices[np.argsort(centers[valid_indices, axis], kind="mergesort")]
        window = min(16, max(len(order) - 1, 1))
        for pos, seg_i in enumerate(order):
            for next_pos in range(pos + 1, min(pos + 1 + window, len(order))):
                seg_j = int(order[next_pos])
                seg_a = int(min(seg_i, seg_j))
                seg_b = int(max(seg_i, seg_j))
                if not _wire_segment_pair_allowed(
                    seg_a,
                    seg_b,
                    n_segments,
                    closed,
                    directions=directions,
                    local_exclusion=local_exclusion,
                    segment_length_prefix=segment_length_prefix,
                ):
                    continue
                candidate_pairs.append((seg_a, seg_b))
                if len(candidate_pairs) >= 65536:
                    flush_candidates()
    flush_candidates()
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
        pair_mask &= ~((gap_lengths <= local_exclusion) & (direction_cosines > -0.9))
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


def _wire_contact_threshold(config: CoilGenConfig) -> float:
    conductor_half_width = 0.5 * abs(float(config.conductor_cross_section_width))
    shift_resolution = 0.1 * abs(float(config.normal_shift_length))
    return max(
        conductor_half_width,
        shift_resolution,
        _wire_clearance_failure_threshold(config),
        1e-9,
    )


def _wire_local_continuation_length(clearance: float) -> float:
    return 2.0 * max(float(clearance), 0.0)


def _point_polyline_distance_3d(point: np.ndarray, polyline: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) == 0:
        return math.inf
    if len(polyline) == 1:
        return float(np.linalg.norm(point - polyline[0]))
    best_distance = math.inf
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-30:
            distance = float(np.linalg.norm(point - start))
        else:
            t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
            distance = float(np.linalg.norm(point - (start + t * segment)))
        if distance < best_distance:
            best_distance = distance
    return best_distance


def _normal_shift_routes_by_segment(part: CoilPart) -> list[list[_NormalShiftRoute]]:
    """Classify the final wire intervals that may serve as crossing overpasses."""

    if part.wire_path is None:
        return []
    wire_v = part.wire_path.v
    routes_by_segment: list[list[_NormalShiftRoute]] = [[] for _ in range(max(len(wire_v) - 1, 0))]
    candidates: list[tuple[tuple[str, int, int], int, np.ndarray]] = []
    for group_index, group in enumerate(part.connected_groups):
        candidates.append((("within_group_return", group_index, 0), 2, group.return_path_v))
    for cut_index, cut in enumerate(part.opening_cuts_among_groups):
        paths = [cut.connection_path_v, cut.closing_connection_path_v, *cut.normal_shift_paths_v]
        for role_index, path_v in enumerate(_distinct_connection_paths(paths)):
            candidates.append((("inter_group_connector", cut_index, role_index), 1, path_v))

    for order, (key, priority, path_v) in enumerate(candidates):
        segment_indices = _reference_path_wire_segment_indices(path_v, wire_v)
        if not segment_indices:
            continue
        route = _NormalShiftRoute(
            key=key,
            priority=int(priority),
            order=int(order),
            segment_indices=segment_indices,
        )
        for segment_index in segment_indices:
            if 0 <= segment_index < len(routes_by_segment):
                routes_by_segment[segment_index].append(route)
    return routes_by_segment


def _extend_normal_shift_routes_by_arclength(
    routes_by_segment: list[list[_NormalShiftRoute]],
    wire_path_v: np.ndarray,
    maximum_distance: float,
) -> list[list[_NormalShiftRoute]]:
    """Carry route identity through short connector approaches.

    One geometric crossing usually produces several clearance contacts. Some
    contacts touch the stored connector itself while others touch its immediately
    adjacent transition segment. Giving those neighbors the nearest route identity
    prevents contradictory over/under votes inside the same crossing.
    """

    routes = [list(segment_routes) for segment_routes in routes_by_segment]
    points = np.asarray(wire_path_v, dtype=float)
    segment_count = max(len(points) - 1, 0)
    maximum_distance = max(float(maximum_distance), 0.0)
    if len(routes) != segment_count or segment_count == 0 or maximum_distance <= 0.0:
        return routes

    preferred = [_preferred_normal_shift_route(routes, index) for index in range(segment_count)]
    source_indices = np.flatnonzero([route is not None for route in preferred])
    if len(source_indices) == 0:
        return routes

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    segment_starts = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
    segment_midpoints = segment_starts + 0.5 * segment_lengths
    source_positions = segment_midpoints[source_indices]
    closed = len(points) > 2 and np.linalg.norm(points[0] - points[-1]) <= 1e-10
    total_length = float(np.sum(segment_lengths))

    for segment_index in np.flatnonzero([route is None for route in preferred]):
        position = float(segment_midpoints[segment_index])
        insertion = int(np.searchsorted(source_positions, position))
        candidate_offsets = (-1, 0)
        candidates: list[tuple[float, _NormalShiftRoute]] = []
        for offset in candidate_offsets:
            source_position_index = insertion + offset
            if closed:
                source_position_index %= len(source_indices)
            elif not 0 <= source_position_index < len(source_indices):
                continue
            source_index = int(source_indices[source_position_index])
            route = preferred[source_index]
            if route is None:
                continue
            distance = abs(position - float(segment_midpoints[source_index]))
            if closed and total_length > 0.0:
                distance = min(distance, total_length - distance)
            if distance <= maximum_distance:
                candidates.append((float(distance), route))
        if candidates:
            _, route = min(
                candidates,
                key=lambda item: (item[0], -item[1].priority, -item[1].order),
            )
            routes[segment_index] = [route]
    return routes


def _mark_within_group_return_levels(part: CoilPart, shift_levels: np.ndarray) -> None:
    """Lift the long return rail, leaving ordinary contour links on the support."""

    if part.wire_path is None:
        return
    for group in part.connected_groups:
        _mark_reference_wire_path(part.wire_path.v, group.return_path_v, shift_levels, level=1.0)


def _select_crossing_overpass(
    seg_a: int,
    seg_b: int,
    routes_by_segment: list[list[_NormalShiftRoute]],
) -> tuple[int, int] | None:
    """Choose one overpass using route role, then stable route precedence."""

    route_a = _preferred_normal_shift_route(routes_by_segment, seg_a)
    route_b = _preferred_normal_shift_route(routes_by_segment, seg_b)
    if route_a is None and route_b is None:
        return None
    if route_a is not None and route_b is None:
        return int(seg_a), int(seg_b)
    if route_b is not None and route_a is None:
        return int(seg_b), int(seg_a)
    assert route_a is not None and route_b is not None
    if route_a.key == route_b.key:
        selected = max(int(seg_a), int(seg_b))
    else:
        precedence_a = (route_a.priority, route_a.order)
        precedence_b = (route_b.priority, route_b.order)
        selected = int(seg_a) if precedence_a > precedence_b else int(seg_b)
    other = int(seg_b) if selected == int(seg_a) else int(seg_a)
    return selected, other


def _crossing_event_segment_selections(
    contacts: list[tuple[int, int, float]],
    routes_by_segment: list[list[_NormalShiftRoute]],
    wire_path_v: np.ndarray,
    cluster_arclength: float,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Resolve each physical crossing once across all nearby contact pairs."""

    if not contacts:
        return []
    points = np.asarray(wire_path_v, dtype=float)
    segment_count = max(len(points) - 1, 0)
    if len(routes_by_segment) != segment_count:
        return []
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    starts = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
    midpoints = starts + 0.5 * segment_lengths
    total_length = float(np.sum(segment_lengths))
    closed = len(points) > 2 and np.linalg.norm(points[0] - points[-1]) <= 1e-10
    cluster_arclength = max(float(cluster_arclength), 0.0)

    parent = np.arange(len(contacts), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    def arclength_distance(first_segment: int, second_segment: int) -> float:
        distance = abs(float(midpoints[first_segment]) - float(midpoints[second_segment]))
        if closed and total_length > 0.0:
            distance = min(distance, total_length - distance)
        return distance

    for first_index, (first_a, first_b, _first_distance) in enumerate(contacts):
        if not (0 <= first_a < segment_count and 0 <= first_b < segment_count):
            continue
        for second_index in range(first_index + 1, len(contacts)):
            second_a, second_b, _second_distance = contacts[second_index]
            if not (0 <= second_a < segment_count and 0 <= second_b < segment_count):
                continue
            if (
                arclength_distance(first_a, second_a) <= cluster_arclength
                and arclength_distance(first_b, second_b) <= cluster_arclength
            ):
                union(first_index, second_index)

    grouped: dict[int, list[tuple[int, int, float]]] = {}
    for index, contact in enumerate(contacts):
        grouped.setdefault(find(index), []).append(contact)

    selections: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for event_contacts in grouped.values():
        branch_a = tuple(sorted({int(contact[0]) for contact in event_contacts}))
        branch_b = tuple(sorted({int(contact[1]) for contact in event_contacts}))
        route_a = _representative_crossing_branch_route(routes_by_segment, branch_a)
        route_b = _representative_crossing_branch_route(routes_by_segment, branch_b)
        if route_a is None and route_b is None:
            continue
        if route_a is not None and route_b is None:
            selections.append((branch_a, branch_b))
            continue
        if route_b is not None and route_a is None:
            selections.append((branch_b, branch_a))
            continue
        assert route_a is not None and route_b is not None
        if route_a.key == route_b.key:
            branch_a_wins = float(np.median(branch_a)) > float(np.median(branch_b))
        else:
            branch_a_wins = (route_a.priority, route_a.order) > (route_b.priority, route_b.order)
        selections.append((branch_a, branch_b) if branch_a_wins else (branch_b, branch_a))
    return selections


def _representative_crossing_branch_route(
    routes_by_segment: list[list[_NormalShiftRoute]],
    segment_indices: tuple[int, ...],
) -> _NormalShiftRoute | None:
    counts: dict[tuple[str, int, int], tuple[int, _NormalShiftRoute]] = {}
    for segment_index in segment_indices:
        route = _preferred_normal_shift_route(routes_by_segment, segment_index)
        if route is None:
            continue
        count, _ = counts.get(route.key, (0, route))
        counts[route.key] = (count + 1, route)
    if not counts:
        return None
    return max(
        counts.values(),
        key=lambda item: (item[0], item[1].priority, item[1].order),
    )[1]


def _preferred_normal_shift_route(
    routes_by_segment: list[list[_NormalShiftRoute]],
    segment_index: int,
) -> _NormalShiftRoute | None:
    if not 0 <= int(segment_index) < len(routes_by_segment):
        return None
    routes = routes_by_segment[int(segment_index)]
    return max(routes, key=lambda route: (route.priority, route.order)) if routes else None


def _mark_segment_points(mask: np.ndarray, segment_index: int) -> None:
    if 0 <= segment_index < len(mask):
        mask[segment_index] = True
    if 0 <= segment_index + 1 < len(mask):
        mask[segment_index + 1] = True


def _crossing_shift_profile(
    base_shift_levels: np.ndarray,
    overpass_points: np.ndarray,
    underpass_points: np.ndarray,
    wire_path_v: np.ndarray,
    transition_length: float,
    smooth_factors,
) -> np.ndarray:
    """Build a continuous one-layer crossing profile.

    Base return lifts and explicit overpasses are combined into one smooth lift
    field. Underpasses then suppress that complete field over a smooth arclength
    window. This keeps an underpass authoritative without reintroducing isolated
    ``1 -> 0 -> 1`` samples.
    """

    base_levels = np.asarray(base_shift_levels, dtype=float).reshape(-1)
    overpass = np.asarray(overpass_points, dtype=bool).reshape(-1)
    underpass = np.asarray(underpass_points, dtype=bool).reshape(-1)
    points = np.asarray(wire_path_v, dtype=float)
    if not (len(base_levels) == len(overpass) == len(underpass) == len(points)):
        raise ValueError("Crossing shift masks must match the wire-path point count.")

    def smooth(levels: np.ndarray) -> np.ndarray:
        return (
            _smooth_shift_levels_by_arclength(levels, points, transition_length)
            if transition_length > 0.0
            else _smooth_shift_levels(levels, smooth_factors)
        )

    base_profile = smooth(base_levels)
    overpass_profile = smooth(overpass.astype(float))
    underpass_profile = smooth(underpass.astype(float))
    lift_profile = np.maximum(base_profile, overpass_profile)
    profile = lift_profile * (1.0 - np.clip(underpass_profile, 0.0, 1.0))
    return np.clip(profile, 0.0, 1.0)


def smooth_shift_mask(points_to_shift: np.ndarray, smooth_factors) -> np.ndarray:
    mask = np.asarray(points_to_shift, dtype=float)
    if len(mask) == 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=float)
    factors = [int(round(float(value))) for value in smooth_factors]
    plateau = max(factors[0] if len(factors) > 0 else 2, 1)
    ramp = max(factors[1] if len(factors) > 1 else 3, 1)
    kernel = np.concatenate((np.arange(1, ramp + 1), np.full(plateau, ramp), np.arange(ramp - 1, 0, -1)))
    kernel = kernel / max(float(ramp), 1.0)
    shift = np.convolve(mask, kernel, mode="same")
    if len(shift) != len(mask):
        start = max((len(shift) - len(mask)) // 2, 0)
        shift = shift[start : start + len(mask)]
    return np.clip(shift, 0.0, 1.0)


def wire_path_normals(part: CoilPart) -> np.ndarray:
    mesh = part.mesh
    if mesh.vertex_normals is None:
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(part.wire_path.v), 1))
        return normals
    vertex_normals = mesh.vertex_normals.copy()
    center = mesh.vertices.mean(axis=0)
    if np.mean(np.sum(vertex_normals * (mesh.vertices - center), axis=1)) < 0:
        vertex_normals *= -1.0
    normals = np.zeros_like(part.wire_path.v)
    for point_ind, point in enumerate(part.wire_path.v):
        nearest = int(np.argmin(np.linalg.norm(mesh.vertices - point, axis=1)))
        normal = vertex_normals[nearest]
        norm = np.linalg.norm(normal)
        normals[point_ind] = normal / norm if norm > 1e-15 else np.array([0.0, 0.0, 1.0])
    return normals


def _smooth_wire_path_normals(normals: np.ndarray, iterations: int) -> np.ndarray:
    """Regularize lift directions along the wire, including mirrored-mesh seams."""

    out = np.asarray(normals, dtype=float).copy()
    if len(out) == 0:
        return out
    lengths = np.linalg.norm(out, axis=1)
    valid = lengths > 1e-15
    out[valid] /= lengths[valid, None]
    for index in range(1, len(out)):
        if np.dot(out[index], out[index - 1]) < 0.0:
            out[index] *= -1.0
    closed = len(out) > 2 and np.linalg.norm(out[0] - out[-1]) <= 1e-8
    unique = out[:-1].copy() if closed else out.copy()
    for _ in range(max(int(iterations), 0)):
        if len(unique) < 3:
            break
        smoothed = unique.copy()
        if closed:
            smoothed = 0.25 * np.roll(unique, 1, axis=0) + 0.5 * unique + 0.25 * np.roll(unique, -1, axis=0)
        else:
            smoothed[1:-1] = 0.25 * unique[:-2] + 0.5 * unique[1:-1] + 0.25 * unique[2:]
        lengths = np.linalg.norm(smoothed, axis=1)
        smoothed[lengths > 1e-15] /= lengths[lengths > 1e-15, None]
        unique = smoothed
    return np.vstack((unique, unique[:1])) if closed else unique


def _distinct_connection_paths(candidates: list[np.ndarray] | tuple[np.ndarray, ...]) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    for candidate in candidates:
        path_v = np.asarray(candidate, dtype=float)
        if path_v.ndim != 2 or path_v.shape[1:] != (3,) or len(path_v) < 2:
            continue
        if any(
            path_v.shape == existing.shape
            and (
                np.allclose(path_v, existing, atol=1e-12, rtol=0.0)
                or np.allclose(path_v, existing[::-1], atol=1e-12, rtol=0.0)
            )
            for existing in paths
        ):
            continue
        paths.append(path_v)
    return paths


def _smooth_shift_levels(shift_levels: np.ndarray, smooth_factors) -> np.ndarray:
    """Smooth integer routing layers without blending away their separation."""

    levels = np.asarray(shift_levels, dtype=float)
    max_level = int(math.ceil(float(np.max(levels)))) if len(levels) else 0
    smoothed = np.zeros_like(levels)
    for level in range(1, max_level + 1):
        smoothed += smooth_shift_mask(levels >= float(level), smooth_factors)
    return smoothed


def _smooth_shift_levels_by_arclength(
    shift_levels: np.ndarray,
    wire_path_v: np.ndarray,
    transition_length: float,
) -> np.ndarray:
    """Taper routing lifts by physical distance while preserving full layer height."""

    levels = np.asarray(shift_levels, dtype=float).reshape(-1)
    points = np.asarray(wire_path_v, dtype=float)
    if len(levels) == 0 or len(points) != len(levels) or not np.any(levels > 0.0):
        return np.zeros_like(levels)
    closed = len(points) > 2 and np.linalg.norm(points[0] - points[-1]) <= 1e-10
    unique_count = len(points) - 1 if closed else len(points)
    unique_points = points[:unique_count]
    unique_levels = levels[:unique_count]
    if unique_count < 2:
        return levels.copy()

    segment_lengths = np.linalg.norm(np.diff(unique_points, axis=0), axis=1)
    positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(positions[-1])
    if closed:
        closing_length = float(np.linalg.norm(unique_points[-1] - unique_points[0]))
        total_length += closing_length
    if total_length <= 1e-15 or transition_length <= 0.0:
        return levels.copy()

    smoothed = np.zeros(unique_count, dtype=float)
    max_level = int(math.ceil(float(np.max(unique_levels))))
    for level in range(1, max_level + 1):
        active = unique_levels >= float(level)
        if not np.any(active):
            continue
        profile = np.ones(unique_count, dtype=float) if np.all(active) else np.zeros(unique_count, dtype=float)
        if not np.all(active):
            active_positions = positions[active]
            distances = np.abs(positions[:, None] - active_positions[None, :])
            if closed:
                distances = np.minimum(distances, total_length - distances)
            nearest = np.min(distances, axis=1)
            normalized = np.clip(nearest / float(transition_length), 0.0, 1.0)
            profile = 0.5 * (1.0 + np.cos(np.pi * normalized))
            profile[nearest >= transition_length] = 0.0
            profile[active] = 1.0
        smoothed += profile

    if not closed:
        return smoothed
    return np.concatenate((smoothed, smoothed[:1]))


def _update_inter_group_paths_after_shift(
    part: CoilPart,
    unshifted_wire_v: np.ndarray,
    shifted_wire_v: np.ndarray,
) -> None:
    """Keep stored connector geometry synchronized with the shifted final wire."""

    for cut in part.opening_cuts_among_groups:
        cut.connection_path_v = _map_reference_path_to_shifted_wire(
            cut.connection_path_v, unshifted_wire_v, shifted_wire_v
        )
        cut.closing_connection_path_v = _map_reference_path_to_shifted_wire(
            cut.closing_connection_path_v, unshifted_wire_v, shifted_wire_v
        )
        cut.normal_shift_paths_v = [
            _map_reference_path_to_shifted_wire(path_v, unshifted_wire_v, shifted_wire_v)
            for path_v in cut.normal_shift_paths_v
        ]


def _map_reference_path_to_shifted_wire(
    reference_path_v: np.ndarray,
    unshifted_wire_v: np.ndarray,
    shifted_wire_v: np.ndarray,
    tolerance: float = 1e-8,
) -> np.ndarray:
    reference = np.asarray(reference_path_v, dtype=float)
    before = np.asarray(unshifted_wire_v, dtype=float)
    after = np.asarray(shifted_wire_v, dtype=float)
    if len(reference) == 0 or len(before) != len(after) or len(before) == 0:
        return reference.copy()

    closed = len(before) > 1 and np.linalg.norm(before[0] - before[-1]) <= tolerance
    searchable = before[:-1] if closed else before
    shifted_searchable = after[:-1] if closed else after
    indices = _reference_path_wire_indices(reference, before, tolerance=tolerance)
    if indices is not None:
        return shifted_searchable[indices].copy()

    nearest = [int(np.argmin(np.linalg.norm(searchable - point, axis=1))) for point in reference]
    errors = [float(np.linalg.norm(searchable[index] - point)) for index, point in zip(nearest, reference)]
    if errors and max(errors) <= tolerance:
        return shifted_searchable[np.asarray(nearest, dtype=np.int64)].copy()
    return reference.copy()


def _reference_path_wire_indices(
    reference_path_v: np.ndarray,
    wire_path_v: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray | None:
    """Match one contiguous occurrence of a reference path in a self-touching wire."""

    reference = np.asarray(reference_path_v, dtype=float)
    wire = np.asarray(wire_path_v, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[1:] != (3,)
        or len(reference) < 2
        or wire.ndim != 2
        or wire.shape[1:] != (3,)
        or len(wire) < len(reference)
    ):
        return None
    closed = len(wire) > 1 and np.linalg.norm(wire[0] - wire[-1]) <= tolerance
    searchable = wire[:-1] if closed else wire
    if len(reference) > len(searchable):
        return None
    starts = np.flatnonzero(np.linalg.norm(searchable - reference[0], axis=1) <= tolerance)
    offsets = np.arange(len(reference), dtype=np.int64)
    best_indices: np.ndarray | None = None
    best_error = math.inf
    for start in starts:
        for direction in (1, -1):
            indices = int(start) + direction * offsets
            if closed:
                indices %= len(searchable)
            elif np.any(indices < 0) or np.any(indices >= len(searchable)):
                continue
            error = float(np.max(np.linalg.norm(searchable[indices] - reference, axis=1)))
            if error < best_error:
                best_error = error
                best_indices = indices.copy()
    return best_indices if best_indices is not None and best_error <= tolerance else None


def _reference_path_wire_segment_indices(
    reference_path_v: np.ndarray,
    wire_path_v: np.ndarray,
) -> tuple[int, ...]:
    """Return final-wire segments belonging to one stored routed subpath."""

    reference = np.asarray(reference_path_v, dtype=float)
    wire = np.asarray(wire_path_v, dtype=float)
    segment_count = max(len(wire) - 1, 0)
    if (
        reference.ndim != 2
        or reference.shape[1:] != (3,)
        or len(reference) == 0
        or wire.ndim != 2
        or wire.shape[1:] != (3,)
        or segment_count == 0
    ):
        return ()
    if len(reference) == 1:
        nearest = int(np.argmin(np.linalg.norm(wire - reference[0], axis=1)))
        return (min(nearest, segment_count - 1),)

    point_indices = _reference_path_wire_indices(reference, wire)
    if point_indices is None:
        point_indices = _reference_path_wire_interval_indices(reference, wire)
    if point_indices is None or len(point_indices) < 2:
        return ()

    closed = len(wire) > 2 and np.linalg.norm(wire[0] - wire[-1]) <= 1e-8
    unique_count = len(wire) - 1 if closed else len(wire)
    segments: list[int] = []
    for first, second in zip(point_indices[:-1], point_indices[1:]):
        first = int(first)
        second = int(second)
        if second == first + 1:
            segment_index = first
        elif first == second + 1:
            segment_index = second
        elif closed and first == unique_count - 1 and second == 0:
            segment_index = unique_count - 1
        elif closed and second == unique_count - 1 and first == 0:
            segment_index = unique_count - 1
        else:
            continue
        if 0 <= segment_index < segment_count and segment_index not in segments:
            segments.append(segment_index)
    return tuple(segments)


def _mark_reference_wire_path(
    wire_path_v: np.ndarray,
    reference_path_v: np.ndarray,
    shift_levels: np.ndarray,
    *,
    level: float = 1.0,
) -> None:
    indices = _reference_path_wire_indices(reference_path_v, wire_path_v)
    if indices is not None:
        shift_levels[indices] = np.maximum(shift_levels[indices], float(level))
        return
    indices = _reference_path_wire_interval_indices(reference_path_v, wire_path_v)
    if indices is not None:
        shift_levels[indices] = np.maximum(shift_levels[indices], float(level))
        return
    _mark_nearest_wire_points(wire_path_v, reference_path_v, shift_levels, level=level)


def _reference_path_wire_interval_indices(
    reference_path_v: np.ndarray,
    wire_path_v: np.ndarray,
) -> np.ndarray | None:
    """Match a routed subpath after connector or fillet samples were inserted."""

    reference = _remove_duplicate_consecutive_points(np.asarray(reference_path_v, dtype=float))
    wire = np.asarray(wire_path_v, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[1:] != (3,)
        or len(reference) < 2
        or wire.ndim != 2
        or wire.shape[1:] != (3,)
        or len(wire) < 2
    ):
        return None
    closed = len(wire) > 2 and np.linalg.norm(wire[0] - wire[-1]) <= 1e-8
    searchable = wire[:-1] if closed else wire
    count = len(searchable)
    start = int(np.argmin(np.linalg.norm(searchable - reference[0], axis=1)))
    end = int(np.argmin(np.linalg.norm(searchable - reference[-1], axis=1)))
    reference_length = _path_length(reference)
    if reference_length <= 1e-12:
        return None

    candidates: list[tuple[float, np.ndarray]] = []
    for direction in (1, -1):
        if closed:
            steps = ((end - start) % count) if direction > 0 else ((start - end) % count)
            indices = (start + direction * np.arange(steps + 1, dtype=np.int64)) % count
        else:
            if (direction > 0 and end < start) or (direction < 0 and end > start):
                continue
            indices = np.arange(start, end + direction, direction, dtype=np.int64)
        candidate_length = _path_length(searchable[indices])
        length_error = abs(candidate_length - reference_length) / reference_length
        endpoint_error = (
            np.linalg.norm(searchable[start] - reference[0]) + np.linalg.norm(searchable[end] - reference[-1])
        ) / reference_length
        candidates.append((float(length_error + endpoint_error), indices))
    if not candidates:
        return None
    score, indices = min(candidates, key=lambda item: item[0])
    return indices if score <= 0.5 else None


def _mark_nearest_wire_points(
    wire_path_v: np.ndarray,
    reference_points_v: np.ndarray,
    shift_levels: np.ndarray,
    *,
    level: float = 1.0,
) -> None:
    wire_path_v = np.asarray(wire_path_v, dtype=float)
    reference_points_v = np.asarray(reference_points_v, dtype=float)
    if len(wire_path_v) == 0 or reference_points_v.ndim != 2 or reference_points_v.shape[1] != 3:
        return
    for point in reference_points_v:
        if not np.all(np.isfinite(point)):
            continue
        distances = np.linalg.norm(wire_path_v - point, axis=1)
        nearest = int(np.argmin(distances))
        shift_levels[nearest] = max(float(shift_levels[nearest]), float(level))


def _select_loop_cut_center_3d(
    loop: ContourLine,
    group_center_v: np.ndarray,
    config: CoilGenConfig,
) -> np.ndarray:
    rel = loop.v - np.asarray(group_center_v, dtype=float)
    distances = np.linalg.norm(rel, axis=1)
    if len(config.force_cut_selection) and str(config.force_cut_selection[0]).lower() == "low":
        idx = int(np.argmin(distances))
    else:
        idx = int(np.argmax(distances))
    return loop.v[idx]


def open_loop_with_3d_sphere(
    loop: ContourLine,
    sphere_center: np.ndarray,
    sphere_diameter: float,
) -> tuple[ContourLine, np.ndarray]:
    """Open a loop to the requested endpoint chord, compensating local curvature."""

    target_span = max(float(sphere_diameter), 1e-12)
    lower_radius = 0.5 * target_span
    best = _open_loop_with_3d_sphere_radius(loop, sphere_center, lower_radius)
    best_span = _opening_cut_span(best[1], 1e-12)
    if math.isfinite(best_span) and best_span >= target_span * (1.0 - 1e-6):
        return best

    upper_radius = lower_radius
    bracketed = False
    for _ in range(10):
        upper_radius *= 1.5
        candidate = _open_loop_with_3d_sphere_radius(loop, sphere_center, upper_radius)
        span = _opening_cut_span(candidate[1], 1e-12)
        if math.isfinite(span) and span > best_span:
            best, best_span = candidate, span
        if math.isfinite(span) and span >= target_span:
            best = candidate
            bracketed = True
            break
    if not bracketed:
        return best

    for _ in range(18):
        radius = 0.5 * (lower_radius + upper_radius)
        candidate = _open_loop_with_3d_sphere_radius(loop, sphere_center, radius)
        span = _opening_cut_span(candidate[1], 1e-12)
        if math.isfinite(span) and span >= target_span:
            upper_radius = radius
            best = candidate
        else:
            lower_radius = radius
    return best


def _open_loop_with_3d_sphere_radius(
    loop: ContourLine,
    sphere_center: np.ndarray,
    radius: float,
) -> tuple[ContourLine, np.ndarray]:
    v = _remove_closing_endpoint(_remove_duplicate_consecutive_points(loop.v))
    if len(v) <= 3:
        return loop, _cut_points_v(loop.v)
    v, near_ind = _insert_nearest_point(v, np.asarray(sphere_center, dtype=float))
    radius = max(float(radius), 1e-12)
    inside = np.linalg.norm(v - sphere_center, axis=1) < radius
    if not inside.any() or inside.all():
        return _open_loop_by_nearest_window_from_points(loop, v, near_ind)

    spans = _inside_spans_cyclic(inside)
    if not spans:
        return _open_loop_by_nearest_window_from_points(loop, v, near_ind)
    n_points = len(v)
    containing_spans = [span for span in spans if _cyclic_span_contains(span, near_ind, n_points)]
    span_options = containing_spans if containing_spans else spans
    start, end = min(
        span_options,
        key=lambda span: np.mean(
            np.linalg.norm(v[_cyclic_indices(span[0], span[1], n_points)] - sphere_center, axis=1)
        ),
    )

    before = (start - 1) % n_points
    after = (end + 1) % n_points
    cut_a_v = _sphere_crossing(
        v[before],
        v[start],
        sphere_center,
        radius,
    )
    cut_b_v = _sphere_crossing(
        v[end],
        v[after],
        sphere_center,
        radius,
    )

    keep_indices = _cyclic_indices(after, before, n_points)
    if not keep_indices:
        return _open_loop_by_nearest_window_from_points(loop, v, near_ind)
    opened_v = np.vstack((cut_b_v, v[keep_indices], cut_a_v))
    opened_v = _remove_duplicate_consecutive_points(opened_v)
    cut_points = np.vstack((cut_b_v, cut_a_v))

    opened = ContourLine(
        v=opened_v,
        potential=loop.potential,
        current_orientation=loop.current_orientation,
        turn_multiplicity=loop.turn_multiplicity,
    )
    if _opening_cut_span(cut_points, 1e-10) <= 1e-10 or len(opened.v) < 2:
        return _open_loop_by_nearest_window_from_points(loop, v, near_ind)
    return opened, cut_points


def _connected_group_from_single_loop(loop: ContourLine) -> ConnectedGroup:
    empty_v = np.zeros((0, 3), dtype=float)
    return ConnectedGroup(
        v=loop.v.copy(),
        return_path_v=empty_v,
        spiral_in_v=loop.v.copy(),
        spiral_out_v=loop.v.copy(),
    )


def _connect_opened_loops(
    opened_loops: list[ContourLine],
    geodesic_solver: SurfaceGeodesicSolver | None = None,
    *,
    allow_direct_fallback: bool = True,
    corner_radius: float = 0.0,
    corner_step_deg: float = 8.0,
) -> ConnectedGroup:
    if not opened_loops:
        empty_v = np.zeros((0, 3), dtype=float)
        return ConnectedGroup(v=empty_v, return_path_v=empty_v, spiral_in_v=empty_v, spiral_out_v=empty_v)

    if geodesic_solver is None:
        spiral_in_v = np.vstack([loop.v for loop in opened_loops])
        spiral_out_v = np.vstack([loop.v for loop in reversed(opened_loops)])
        turnaround_loop = opened_loops[-1]
        return_path_v = np.vstack(
            (
                turnaround_loop.v[[-1, 0]],
                *[loop.v[0].reshape(1, 3) for loop in reversed(opened_loops[:-1])],
            )
        )
        v = np.vstack((spiral_in_v, return_path_v, spiral_in_v[0]))
        return ConnectedGroup(
            v=v,
            return_path_v=return_path_v,
            spiral_in_v=spiral_in_v,
            spiral_out_v=spiral_out_v,
        )

    loop_sections = [loop.v for loop in opened_loops if len(loop.v)]
    turnaround_loop = opened_loops[-1]
    # The last loop is the inward-spiral turnaround, not necessarily the
    # topological outer loop. Keep its opening chord as an explicit bridge so
    # the subsequent normal lift applies to the whole crossing. A surface
    # geodesic here walks back through the nearest mesh vertex and creates a
    # narrow V-shaped zipper.
    return_sections = [turnaround_loop.v[[-1, 0]]]
    return_sections.extend(loop.v[0].reshape(1, 3) for loop in reversed(opened_loops[:-1]) if len(loop.v))
    spiral_in_v = _connect_path_sections_with_connectors(
        loop_sections,
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
        corner_radius=corner_radius,
        corner_step_deg=corner_step_deg,
    )
    return_path_v = _connect_path_sections_with_connectors(
        return_sections,
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
        corner_radius=corner_radius,
        corner_step_deg=corner_step_deg,
    )
    if len(spiral_in_v) == 0 or (len(return_sections) > 1 and len(return_path_v) == 0):
        empty_v = np.zeros((0, 3), dtype=float)
        return ConnectedGroup(v=empty_v, return_path_v=empty_v, spiral_in_v=empty_v, spiral_out_v=empty_v)
    v = _connect_path_sections_with_connectors(
        [spiral_in_v, return_path_v, spiral_in_v[:1]],
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
        corner_radius=corner_radius,
        corner_step_deg=corner_step_deg,
    )
    if len(v) == 0:
        empty_v = np.zeros((0, 3), dtype=float)
        return ConnectedGroup(v=empty_v, return_path_v=empty_v, spiral_in_v=empty_v, spiral_out_v=empty_v)
    spiral_out_v = _connect_path_sections_with_connectors(
        [loop.v for loop in reversed(opened_loops) if len(loop.v)],
        geodesic_solver,
        allow_direct_fallback=allow_direct_fallback,
        corner_radius=corner_radius,
        corner_step_deg=corner_step_deg,
    )
    return ConnectedGroup(
        v=v,
        return_path_v=return_path_v,
        spiral_in_v=spiral_in_v,
        spiral_out_v=spiral_out_v,
    )


def _connect_path_sections_with_connectors(
    sections: list[np.ndarray],
    geodesic_solver: SurfaceGeodesicSolver | None,
    *,
    allow_direct_fallback: bool,
    corner_radius: float = 0.0,
    corner_step_deg: float = 8.0,
) -> np.ndarray:
    finite_sections = []
    for section in sections:
        section_v = np.asarray(section, dtype=float)
        if section_v.ndim != 2 or section_v.shape[1] != 3:
            continue
        section_v = section_v[np.all(np.isfinite(section_v), axis=1)]
        if len(section_v):
            finite_sections.append(section_v)
    if not finite_sections:
        return np.zeros((0, 3), dtype=float)

    path_v = finite_sections[0].copy()
    for section_v in finite_sections[1:]:
        if len(path_v) == 0:
            return np.zeros((0, 3), dtype=float)
        connector_v = _inter_group_connector_path(
            path_v[-1],
            section_v[0],
            geodesic_solver,
            allow_direct_fallback=allow_direct_fallback,
        )
        if len(connector_v) < 2:
            return np.zeros((0, 3), dtype=float)
        path_v = _join_path_sections_with_fillets(
            path_v,
            connector_v,
            section_v,
            radius=max(float(corner_radius), 0.0),
            angle_step_deg=max(float(corner_step_deg), 0.25),
        )
    return path_v


def _join_path_sections_with_fillets(
    before_v: np.ndarray,
    connector_v: np.ndarray,
    after_v: np.ndarray,
    *,
    radius: float,
    angle_step_deg: float,
) -> np.ndarray:
    path = _fillet_path_join(before_v, connector_v, radius=radius, angle_step_deg=angle_step_deg)
    return _fillet_path_join(path, after_v, radius=radius, angle_step_deg=angle_step_deg)


def _repair_shifted_wire_reversal_spurs(
    points_v: np.ndarray,
    shift_array: np.ndarray,
    config: CoilGenConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse short offset-curve cusps without altering conductor topology."""

    points = np.asarray(points_v, dtype=float).copy()
    shifts = np.asarray(shift_array, dtype=float).reshape(-1).copy()
    if len(points) < 5 or len(points) != len(shifts):
        return points, shifts
    closed = np.linalg.norm(points[0] - points[-1]) <= 1e-8
    max_span = max(
        _within_group_cut_width(config),
        _inter_group_cut_width(config),
    )
    if max_span <= 0.0:
        return points, shifts

    _seg_a, _seg_b, clearance = find_min_wire_segment_distance_3d(points, config)
    min_turn_cosine = math.cos(math.radians(155.0))
    index = 1
    while index + 2 < len(points):
        if closed and index + 2 == len(points) - 1:
            break
        first = points[index] - points[index - 1]
        reversal = points[index + 1] - points[index]
        third = points[index + 2] - points[index + 1]
        lengths = np.array(
            [np.linalg.norm(first), np.linalg.norm(reversal), np.linalg.norm(third)],
            dtype=float,
        )
        if np.min(lengths) <= 1e-12:
            index += 1
            continue
        unit_first, unit_reversal, unit_third = first / lengths[0], reversal / lengths[1], third / lengths[2]
        shortcut = points[index + 2] - points[index - 1]
        shortcut_length = float(np.linalg.norm(shortcut))
        shifted = bool(np.max(shifts[index - 1 : index + 3]) > 0.0)
        is_retraced_spur = (
            shifted
            and shortcut_length <= max_span
            and float(np.sum(lengths)) <= 4.0 * max_span
            and float(np.dot(unit_first, unit_reversal)) < min_turn_cosine
            and float(np.dot(unit_reversal, unit_third)) < min_turn_cosine
            and float(np.dot(unit_first, unit_third)) > 0.5
        )
        if not is_retraced_spur or shortcut_length <= 1e-12:
            index += 1
            continue
        shortcut_unit = shortcut / shortcut_length
        if float(np.dot(shortcut_unit, unit_first)) <= 0.0 or float(np.dot(shortcut_unit, unit_third)) <= 0.0:
            index += 1
            continue

        candidate_points = np.delete(points, [index, index + 1], axis=0)
        candidate_shifts = np.delete(shifts, [index, index + 1])
        _seg_a, _seg_b, candidate_clearance = find_min_wire_segment_distance_3d(candidate_points, config)
        if candidate_clearance + 1e-9 < clearance:
            index += 1
            continue
        points, shifts, clearance = candidate_points, candidate_shifts, float(candidate_clearance)
        index = max(index - 2, 1)
    if closed and len(points):
        points[-1] = points[0]
        shifts[-1] = shifts[0]
    return points, shifts


def _fillet_path_join(
    before_v: np.ndarray,
    after_v: np.ndarray,
    *,
    radius: float,
    angle_step_deg: float,
) -> np.ndarray:
    before = _remove_duplicate_consecutive_points(np.asarray(before_v, dtype=float))
    after = _remove_duplicate_consecutive_points(np.asarray(after_v, dtype=float))
    if radius <= 0.0 or len(before) < 2 or len(after) < 2:
        return _stack_path_sections((before, after))
    if np.linalg.norm(before[-1] - after[0]) > 1e-9:
        return _stack_path_sections((before, after))

    corner = 0.5 * (before[-1] + after[0])
    incoming = corner - before[-2]
    outgoing = after[1] - corner
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if incoming_length <= 1e-15 or outgoing_length <= 1e-15:
        return _stack_path_sections((before, after))
    incoming /= incoming_length
    outgoing /= outgoing_length
    turn_angle = float(np.arccos(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))
    if turn_angle <= np.deg2rad(1.0) or turn_angle >= np.pi - 1e-6:
        return _stack_path_sections((before, after))

    tangent_distance = radius * math.tan(0.5 * turn_angle)
    tangent_distance = min(
        tangent_distance,
        0.45 * _path_length(before),
        0.45 * _path_length(after),
    )
    if tangent_distance <= 1e-12:
        return _stack_path_sections((before, after))
    before_trimmed = _trim_polyline_end(before, tangent_distance)
    after_trimmed = _trim_polyline_start(after, tangent_distance)
    start = before_trimmed[-1]
    end = after_trimmed[0]
    start_tangent = start - before_trimmed[-2]
    end_tangent = after_trimmed[1] - end
    start_tangent /= max(float(np.linalg.norm(start_tangent)), 1e-15)
    end_tangent /= max(float(np.linalg.norm(end_tangent)), 1e-15)
    effective_angle = float(np.arccos(np.clip(np.dot(start_tangent, end_tangent), -1.0, 1.0)))
    effective_radius = tangent_distance / max(math.tan(0.5 * effective_angle), 1e-15)
    handle_length = 4.0 / 3.0 * effective_radius * math.tan(0.25 * effective_angle)
    control_1 = start + handle_length * start_tangent
    control_2 = end - handle_length * end_tangent
    count = max(int(np.ceil(np.degrees(effective_angle) / max(angle_step_deg, 0.25))), 2)
    t = np.linspace(0.0, 1.0, count + 1)[:, None]
    curve = (
        (1.0 - t) ** 3 * start + 3.0 * (1.0 - t) ** 2 * t * control_1 + 3.0 * (1.0 - t) * t**2 * control_2 + t**3 * end
    )
    return _stack_path_sections((before_trimmed, curve, after_trimmed))


def _trim_polyline_end(points: np.ndarray, distance: float) -> np.ndarray:
    remaining = float(distance)
    for index in range(len(points) - 2, -1, -1):
        segment = points[index] - points[index + 1]
        length = float(np.linalg.norm(segment))
        if length <= 1e-15:
            continue
        if remaining <= length:
            tangent = points[index + 1] + segment * (remaining / length)
            return _stack_path_sections((points[: index + 1], tangent.reshape(1, 3)))
        remaining -= length
    return points[:1].copy()


def _trim_polyline_start(points: np.ndarray, distance: float) -> np.ndarray:
    remaining = float(distance)
    for index in range(len(points) - 1):
        segment = points[index + 1] - points[index]
        length = float(np.linalg.norm(segment))
        if length <= 1e-15:
            continue
        if remaining <= length:
            tangent = points[index] + segment * (remaining / length)
            return _stack_path_sections((tangent.reshape(1, 3), points[index + 1 :]))
        remaining -= length
    return points[-1:].copy()


def _remove_duplicate_consecutive_points(v: np.ndarray) -> np.ndarray:
    if len(v) < 2:
        return v.copy()
    keep = np.ones(len(v), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(v, axis=0), axis=1) > 1e-10
    return v[keep]


def _remove_closing_endpoint(v: np.ndarray) -> np.ndarray:
    if len(v) > 1 and np.linalg.norm(v[0] - v[-1]) <= 1e-10:
        return v[:-1].copy()
    return v.copy()


def _insert_nearest_point(v: np.ndarray, target_point: np.ndarray) -> tuple[np.ndarray, int]:
    v = _remove_closing_endpoint(v)
    if len(v) < 2:
        return v.copy(), 0
    starts = v
    ends = np.vstack((v[1:], v[0]))
    seg = ends - starts
    denom = np.sum(seg * seg, axis=1)
    denom[denom == 0] = 1.0
    t = np.sum((target_point - starts) * seg, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    near_v = starts + seg * t[:, None]
    near_ind = int(np.argmin(np.linalg.norm(near_v - target_point, axis=1)))
    insert_at = near_ind + 1
    if insert_at >= len(v):
        return np.vstack((v, near_v[near_ind])), len(v)
    return np.insert(v, insert_at, near_v[near_ind], axis=0), insert_at


def _cyclic_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 0:
        return []
    start = int(start) % count
    end = int(end) % count
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, count)) + list(range(0, end + 1))


def _cyclic_span_contains(span: tuple[int, int], index: int, count: int) -> bool:
    if count <= 0:
        return False
    start, end = int(span[0]) % count, int(span[1]) % count
    index = int(index) % count
    if start <= end:
        return start <= index <= end
    return index >= start or index <= end


def _inside_spans(inside: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = None
    for idx, value in enumerate(inside):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(inside) - 1))
    return spans


def _inside_spans_cyclic(inside: np.ndarray) -> list[tuple[int, int]]:
    inside = np.asarray(inside, dtype=bool)
    spans = _inside_spans(inside)
    if len(spans) > 1 and spans[0][0] == 0 and spans[-1][1] == len(inside) - 1:
        merged = (spans[-1][0], spans[0][1])
        spans = [merged, *spans[1:-1]]
    return spans


def _sphere_crossing(
    p0_v: np.ndarray,
    p1_v: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    d0 = np.linalg.norm(p0_v - center)
    d1 = np.linalg.norm(p1_v - center)
    denom = d1 - d0
    t = 0.0 if abs(denom) < 1e-15 else np.clip((radius - d0) / denom, 0.0, 1.0)
    return p0_v + t * (p1_v - p0_v)


def _open_loop_by_nearest_window_from_points(
    loop: ContourLine,
    points: np.ndarray,
    near_ind: int,
    tolerance: float = 1e-10,
) -> tuple[ContourLine, np.ndarray]:
    v = np.asarray(points, dtype=float).copy()
    if len(v) <= 3:
        return loop, _cut_points_v(v)
    v = _remove_duplicate_consecutive_points(v)
    if len(v) <= 3:
        return loop, _cut_points_v(v)
    if np.linalg.norm(v[0] - v[-1]) <= 1e-10:
        v = v[:-1]
    start = near_ind % len(v)
    max_window = max(1, min(len(v) // 2 - 1, 8))
    for window in range(1, max_window + 1):
        after = (start + window) % len(v)
        before = (start - window) % len(v)
        keep_indices = _cyclic_indices(after, before, len(v))
        if len(keep_indices) < 2:
            continue
        opened_v = v[keep_indices]
        cut_points = _cut_points_v(opened_v)
        if _opening_cut_span(cut_points, tolerance) > tolerance:
            return (
                ContourLine(
                    v=opened_v,
                    potential=loop.potential,
                    current_orientation=loop.current_orientation,
                    turn_multiplicity=loop.turn_multiplicity,
                ),
                cut_points,
            )
    opened_v = np.vstack((v[start + 1 :], v[:start]))
    if len(opened_v) < 2:
        opened_v = v
    return ContourLine(
        v=opened_v,
        potential=loop.potential,
        current_orientation=loop.current_orientation,
        turn_multiplicity=loop.turn_multiplicity,
    ), _cut_points_v(opened_v)


def _cut_points_v(path: np.ndarray) -> np.ndarray:
    if len(path) < 2:
        return np.zeros((0, 3), dtype=float)
    return np.vstack((path[0], path[-1]))


def _nearest_vertex_pair(loop_a: ContourLine, loop_b: ContourLine) -> MutualLoopDistance:
    distances = np.linalg.norm(loop_a.v[None, :, :] - loop_b.v[:, None, :], axis=2)
    flat_index = int(np.argmin(distances))
    index_b, index_a = np.unravel_index(flat_index, distances.shape)
    return MutualLoopDistance(
        distance=float(distances[index_b, index_a]),
        near_point_a=NearestLoopPoint(v=loop_a.v[index_a].copy()),
        index_a=int(index_a),
        near_point_b=NearestLoopPoint(v=loop_b.v[index_b].copy()),
        index_b=int(index_b),
    )


def _nearest_point_on_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[float, int, np.ndarray, float]:
    point = np.asarray(point, dtype=float).reshape(-1)
    polyline = _coerce_points(polyline, expected_dims=len(point))
    if len(polyline) == 0:
        raise ValueError("Polyline must contain at least one point.")
    if len(polyline) == 1:
        return float(np.linalg.norm(point - polyline[0])), 0, polyline[0].copy(), 0.0

    starts = polyline[:-1]
    ends = polyline[1:]
    segments = ends - starts
    denom = np.sum(segments * segments, axis=1)
    t = np.zeros(len(segments), dtype=float)
    valid = denom > 1e-30
    t[valid] = np.sum((point - starts[valid]) * segments[valid], axis=1) / denom[valid]
    t = np.clip(t, 0.0, 1.0)
    near_points = starts + t[:, None] * segments
    distances = np.linalg.norm(near_points - point, axis=1)
    segment_index = int(np.argmin(distances))
    return float(distances[segment_index]), segment_index, near_points[segment_index].copy(), float(t[segment_index])


def _coerce_points(points: np.ndarray, expected_dims: int | None = None) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D point array, got shape {array.shape}.")
    if expected_dims is not None and array.shape[1] != expected_dims:
        if array.shape[0] == expected_dims:
            array = array.T
        else:
            raise ValueError(f"Expected points with dimension {expected_dims}, got shape {array.shape}.")
    return array


def _parallel_loop_levels(lower_keys: list[tuple[int, ...]]) -> list[list[int]]:
    levels: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for key in lower_keys:
        if key in seen:
            continue
        seen.add(key)
        levels.append([loop_ind for loop_ind, loop_key in enumerate(lower_keys) if loop_key == key])
    return levels


def _exclusive_loop_groups(overlapping_groups: list[list[int]]) -> list[list[int]]:
    exclusive_groups: list[list[int]] = []
    group_sets = [set(group) for group in overlapping_groups]
    for group_ind, group in enumerate(overlapping_groups):
        loops_to_remove: set[int] = set()
        for other_ind, other_set in enumerate(group_sets):
            if other_ind == group_ind:
                continue
            if other_set and other_set.issubset(group_sets[group_ind]):
                loops_to_remove.update(other_set)
        exclusive = [loop_ind for loop_ind in group if loop_ind not in loops_to_remove]
        if exclusive and exclusive not in exclusive_groups:
            exclusive_groups.append(exclusive)
    return exclusive_groups


def _topological_grouping_method(config: CoilGenConfig | None) -> str:
    if config is None:
        return "containment"
    method = str(getattr(config, "topological_grouping_method", "auto")).strip().lower().replace("-", "_")
    if method in {"legacy", "nested", "loop_in_loop"}:
        return "containment"
    if method in {"parallel", "parallel_surface", "surface_parallel", "boundary_aware"}:
        return "parallel_surface"
    return "auto"


def _topologically_outer_loop_indices(
    loop_indices: list[int] | tuple[int, ...],
    contains: np.ndarray,
) -> tuple[int, ...]:
    """Return the exterior frontier of a loop group.

    ``contains[outer, inner]`` is true when ``outer`` intrinsically encloses
    ``inner`` on the support surface.  A loop is therefore on the exterior
    frontier when no other loop in the same group contains it.  A conventional
    nested winding group has one such loop.  Homologically parallel contours on
    an open surface can have several incomparable exterior-frontier loops; all
    are retained because choosing one by extraction order would not be
    geometrically meaningful.
    """

    indices = tuple(dict.fromkeys(int(index) for index in loop_indices))
    if len(indices) <= 1:
        return indices
    matrix = np.asarray(contains, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or max(indices) >= matrix.shape[0]:
        return indices[:1]
    group_matrix = matrix[np.ix_(indices, indices)]
    enclosing_count = np.count_nonzero(group_matrix, axis=0)
    minimum = int(np.min(enclosing_count))
    return tuple(index for index, count in zip(indices, enclosing_count) if int(count) == minimum)


def _containment_grouping_from_matrix(contains: np.ndarray) -> dict[str, object]:
    n_loops = int(contains.shape[0])
    higher_loops = [tuple(np.where(contains[loop_ind, :])[0].tolist()) for loop_ind in range(n_loops)]
    lower_loops = [tuple(np.where(contains[:, loop_ind])[0].tolist()) for loop_ind in range(n_loops)]
    lower_keys = [lower if lower else (-1,) for lower in lower_loops]
    group_levels_by_loop = _parallel_loop_levels(lower_keys)

    is_global_top_loop = [len(higher_loops[loop_ind]) == n_loops - 1 for loop_ind in range(n_loops)]
    group_levels_by_loop = [level for level in group_levels_by_loop if len(level) != 1 or is_global_top_loop[level[0]]]
    if not group_levels_by_loop:
        group_levels_by_loop = [[loop_ind] for loop_ind in range(n_loops)]

    overlapping_groups = [
        sorted(set([loop_ind, *higher_loops[loop_ind]])) for level in group_levels_by_loop for loop_ind in level
    ]
    loop_groups = _exclusive_loop_groups(overlapping_groups)
    loop_groups = sorted(loop_groups, key=len, reverse=True)

    loop_to_group: dict[int, int] = {}
    for group_ind, group in enumerate(loop_groups):
        for loop_ind in group:
            loop_to_group[int(loop_ind)] = group_ind

    group_levels: list[list[int]] = []
    for level in group_levels_by_loop:
        group_level = sorted({loop_to_group[loop_ind] for loop_ind in level if loop_ind in loop_to_group})
        if group_level and group_level not in group_levels:
            group_levels.append(group_level)
    group_levels = sorted(group_levels, key=min)

    level_positions: list[list[int]] = []
    for group_level in group_levels:
        loops_in_level = sorted({loop_ind for group_ind in group_level for loop_ind in loop_groups[group_ind]})
        enclosing_loops = sorted({outer for loop_ind in loops_in_level for outer in np.where(contains[:, loop_ind])[0]})
        positions = [
            group_ind
            for group_ind, group in enumerate(loop_groups)
            if group_ind not in group_level and any(loop_ind in group for loop_ind in enclosing_loops)
        ]
        level_positions.append(positions)

    rank_of_group = np.zeros(len(loop_groups), dtype=int)
    for group_ind in range(len(loop_groups)):
        for level_ind, group_level in enumerate(group_levels):
            if group_ind in group_level:
                rank_of_group[group_ind] = len(level_positions[level_ind])
                break
    for level_ind, positions in enumerate(level_positions):
        level_positions[level_ind] = sorted(positions, key=lambda group_ind: rank_of_group[group_ind])

    return {
        "loop_groups": [list(group) for group in loop_groups],
        "group_levels": group_levels,
        "level_positions": level_positions,
    }


def _surface_parallel_loop_grouping(
    part: CoilPart,
    containment_grouping: dict[str, object],
    signatures: list[LoopTopologySignature],
) -> dict[str, object]:
    signature_by_loop = {signature.loop_index: signature for signature in signatures}
    boundary_loop_indices = {
        signature.loop_index for signature in signatures if signature.boundary_associated and not signature.ambiguous
    }
    if not boundary_loop_indices:
        grouping = dict(containment_grouping)
        grouping["diagnostics"] = {"topological_grouping_method": "containment"}
        return grouping

    grouped: dict[tuple[tuple[tuple[int, ...], ...], int], list[int]] = {}
    group_key_labels: dict[tuple[tuple[tuple[int, ...], ...], int], str] = {}
    for loop_ind in sorted(boundary_loop_indices):
        signature = signature_by_loop[loop_ind]
        orientation = _surface_parallel_orientation_sign(part, part.contour_lines[loop_ind], signature)
        key = (signature.boundary_signature, orientation)
        grouped.setdefault(key, []).append(loop_ind)
        group_key_labels[key] = f"boundary={signature.boundary_signature},orientation={orientation}"

    loop_groups: list[list[int]] = []
    split_reasons: list[str] = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), item[1])):
        for group in _split_boundary_group_by_potential_blockers(part, grouped[key], boundary_loop_indices):
            if len(group) != len(grouped[key]):
                split_reasons.append(f"{group_key_labels[key]} split by incompatible intermediate contour potential")
            loop_groups.append(group)

    for group in containment_grouping.get("loop_groups", []):
        residual = [int(loop_ind) for loop_ind in group if int(loop_ind) not in boundary_loop_indices]
        if residual and residual not in loop_groups:
            loop_groups.append(residual)

    assigned = {loop_ind for group in loop_groups for loop_ind in group}
    for loop_ind in range(len(part.contour_lines)):
        if loop_ind not in assigned:
            loop_groups.append([loop_ind])

    loop_groups, axial_outlier_repairs = _repair_surface_parallel_axial_outliers(
        part,
        loop_groups,
        boundary_loop_indices,
    )
    loop_groups = sorted(loop_groups, key=lambda group: (-len(group), min(group)))
    group_levels = [list(range(len(loop_groups)))] if loop_groups else []
    level_positions = [[] for _ in group_levels]
    return {
        "loop_groups": loop_groups,
        "group_levels": group_levels,
        "level_positions": level_positions,
        "diagnostics": {
            "topological_grouping_method": "parallel_surface",
            "boundary_associated_loop_count": len(boundary_loop_indices),
            "parallel_surface_group_count": len(loop_groups),
            "parallel_surface_group_loop_indices": [list(group) for group in loop_groups],
            "parallel_surface_split_reasons": split_reasons,
            "parallel_surface_axial_outlier_repairs": axial_outlier_repairs,
        },
    }


def _repair_surface_parallel_axial_outliers(
    part: CoilPart,
    loop_groups: list[list[int]],
    boundary_loop_indices: set[int],
) -> tuple[list[list[int]], list[str]]:
    axis_result = _mesh_boundary_axis(part.mesh)
    if axis_result is None or len(loop_groups) < 2:
        return loop_groups, []
    origin_v, axis_v = axis_result
    axis_v = _normalize_vector(axis_v, fallback=np.array([0.0, 0.0, 1.0]))

    axial_by_loop: dict[int, float] = {}
    for loop_ind in boundary_loop_indices:
        if 0 <= int(loop_ind) < len(part.contour_lines):
            contour_v = np.asarray(part.contour_lines[int(loop_ind)].v, dtype=float)
            if len(contour_v):
                axial_by_loop[int(loop_ind)] = float(np.dot(np.mean(contour_v, axis=0) - origin_v, axis_v))
    if len(axial_by_loop) < 4:
        return loop_groups, []

    axial_values = np.asarray(list(axial_by_loop.values()), dtype=float)
    span = float(np.ptp(axial_values))
    tolerance = max(1e-8, 0.02 * span)

    group_infos: list[dict[str, object]] = []
    for group_ind, group in enumerate(loop_groups):
        boundary_members = [int(loop_ind) for loop_ind in group if int(loop_ind) in axial_by_loop]
        if len(boundary_members) < 3:
            continue
        signed_members = [
            (loop_ind, -1 if axial_by_loop[loop_ind] < -tolerance else 1 if axial_by_loop[loop_ind] > tolerance else 0)
            for loop_ind in boundary_members
        ]
        nonzero = [(loop_ind, sign) for loop_ind, sign in signed_members if sign != 0]
        if len(nonzero) < 3:
            continue
        counts = {
            -1: sum(1 for _loop_ind, sign in nonzero if sign < 0),
            1: sum(1 for _loop_ind, sign in nonzero if sign > 0),
        }
        dominant_sign = -1 if counts[-1] >= counts[1] else 1
        outliers = [loop_ind for loop_ind, sign in nonzero if sign != dominant_sign]
        max_outliers = max(1, int(math.floor(0.15 * len(nonzero))))
        if len(outliers) > max_outliers:
            continue
        group_infos.append(
            {
                "group_ind": group_ind,
                "dominant_sign": dominant_sign,
                "member_count": len(nonzero),
                "outliers": outliers,
            }
        )

    sign_to_group: dict[int, int] = {}
    for sign in (-1, 1):
        candidates = [info for info in group_infos if int(info["dominant_sign"]) == sign]
        if candidates:
            selected = max(candidates, key=lambda info: int(info["member_count"]))
            sign_to_group[sign] = int(selected["group_ind"])
    if -1 not in sign_to_group or 1 not in sign_to_group:
        return loop_groups, []

    repaired = [list(group) for group in loop_groups]
    repairs: list[str] = []
    for info in group_infos:
        source_group = int(info["group_ind"])
        for loop_ind in list(info["outliers"]):
            target_sign = -1 if axial_by_loop[int(loop_ind)] < 0.0 else 1
            target_group = sign_to_group.get(target_sign)
            if target_group is None or target_group == source_group:
                continue
            if loop_ind in repaired[source_group]:
                repaired[source_group].remove(loop_ind)
            if loop_ind not in repaired[target_group]:
                repaired[target_group].append(loop_ind)
                repaired[target_group].sort()
            repairs.append(f"L{loop_ind}:G{source_group}->G{target_group}")

    if not repairs:
        return loop_groups, []
    repaired = [group for group in repaired if group]
    return repaired, repairs


def _split_boundary_group_by_potential_blockers(
    part: CoilPart,
    loop_indices: list[int],
    boundary_loop_indices: set[int],
) -> list[list[int]]:
    ordered = sorted(loop_indices, key=lambda loop_ind: float(part.contour_lines[loop_ind].potential))
    if len(ordered) <= 1:
        return [ordered]
    blocker_potentials = sorted(
        float(contour.potential)
        for loop_ind, contour in enumerate(part.contour_lines)
        if loop_ind not in boundary_loop_indices and np.isfinite(float(contour.potential))
    )
    if not blocker_potentials:
        return [ordered]

    groups: list[list[int]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        low = min(float(part.contour_lines[previous].potential), float(part.contour_lines[current].potential))
        high = max(float(part.contour_lines[previous].potential), float(part.contour_lines[current].potential))
        blocked = any(low < potential < high for potential in blocker_potentials)
        if blocked:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def _surface_parallel_orientation_sign(
    part: CoilPart,
    contour: ContourLine,
    signature: LoopTopologySignature,
) -> int:
    area_vector = _polyline_area_vector(contour.v)
    reference = _surface_parallel_reference_vector(part, contour, signature)
    signed = float(np.dot(area_vector, reference))
    scale = float(np.linalg.norm(area_vector) * np.linalg.norm(reference))
    if scale > 1e-30 and abs(signed) > 1e-10 * scale:
        return 1 if signed > 0.0 else -1
    return _loop_geometric_orientation_sign(part, contour)


def _surface_parallel_reference_vector(
    part: CoilPart, contour: ContourLine, signature: LoopTopologySignature
) -> np.ndarray:
    if len(signature.component_centroids) >= 2:
        ordered = sorted(
            zip(signature.component_boundary_sets, signature.component_areas, signature.component_centroids),
            key=lambda item: (item[0], item[1]),
        )
        start = np.asarray(ordered[0][2], dtype=float)
        end = np.asarray(ordered[-1][2], dtype=float)
        vector = end - start
        if np.linalg.norm(vector) > 1e-15:
            return vector
    vertices = np.asarray(part.mesh.vertices, dtype=float)
    center = np.mean(vertices, axis=0) if len(vertices) else np.zeros(3, dtype=float)
    contour_center = np.mean(contour.v, axis=0) if len(contour.v) else center
    vector = contour_center - center
    if np.linalg.norm(vector) > 1e-15:
        return vector
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _loop_geometric_orientation_sign(part: CoilPart, contour: ContourLine) -> int:
    area_vector = _polyline_area_vector(contour.v)
    face_normals = _mesh_face_normals(part.mesh)
    if contour.segment_faces is not None and len(contour.segment_faces):
        valid_faces = np.asarray(contour.segment_faces, dtype=np.int64)
        valid_faces = valid_faces[(valid_faces >= 0) & (valid_faces < len(face_normals))]
        normal = (
            _normalize_vector(np.mean(face_normals[valid_faces], axis=0), fallback=np.array([0.0, 0.0, 1.0]))
            if len(valid_faces)
            else _normalize_vector(area_vector, fallback=np.array([0.0, 0.0, 1.0]))
        )
    else:
        normal = _normalize_vector(area_vector, fallback=np.array([0.0, 0.0, 1.0]))
    signed = float(np.dot(area_vector, normal))
    if abs(signed) <= 1e-15:
        return 1
    return 1 if signed > 0.0 else -1


def _loop_topology_signature_diagnostics(signatures: list[LoopTopologySignature]) -> dict[str, object]:
    return {
        "loop_topology_signature_count": len(signatures),
        "loop_topology_signatures": [
            {
                "loop_index": signature.loop_index,
                "kind": signature.kind,
                "separating": signature.separating,
                "contractible_candidate": signature.contractible_candidate,
                "boundary_associated": signature.boundary_associated,
                "ambiguous": signature.ambiguous,
                "boundary_signature": [list(boundary_set) for boundary_set in signature.boundary_signature],
                "component_boundary_sets": [list(boundary_set) for boundary_set in signature.component_boundary_sets],
                "component_areas": list(signature.component_areas),
            }
            for signature in signatures
        ],
    }


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
