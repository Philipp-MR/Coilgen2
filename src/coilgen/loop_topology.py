"""Intrinsic loop inclusion, grouping and orientation; independent of path routing."""

from __future__ import annotations
import math
import numpy as np
from .config import CoilGenConfig
from .models import CoilPart, ContourLine, LoopGroup
from .progress import ProgressReporter, progress_scope
from .regular_rings import regular_ring_order, ring_support_axis
from .routing_objective import routing_selection_policy
from .surface_topology import (
    LoopTopologySignature,
    build_intrinsic_loop_inclusion_matrix,
    classify_loop_topology_signatures,
    validate_intrinsic_topology_inputs,
)


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

    weights = np.asarray([max(len(part.contour_lines) ** 2, 1) for part in parts], dtype=float)
    boundaries = np.r_[0, np.cumsum(weights) / max(weights.sum(), 1) * 100]
    for part_ind, part in enumerate(parts):
        start, end = boundaries[part_ind : part_ind + 2]
        loops = part.contour_lines
        n_loops = len(loops)
        part.groups = []
        part.loop_groups = []
        part.group_levels = []
        part.level_positions = []
        part.loop_in_loop_matrix = np.zeros((n_loops, n_loops), dtype=bool)
        part.topology_diagnostics = {}
        if n_loops == 0:
            if progress is not None:
                progress.report(end, f"part {part_ind + 1}/{len(parts)} has no contours", force=True)
            continue

        if progress is not None:
            progress.report(
                None,
                f"part {part_ind + 1}/{len(parts)}: validating {n_loops} contour loop(s) for intrinsic topology",
                force=True,
            )
        with progress_scope(start, start + 0.4 * (end - start), "Validate intrinsic contour topology"):
            validation = validate_intrinsic_topology_inputs(part, check_separating=True, progress=progress)
        part.topology_diagnostics = validation.as_dict()
        validation.raise_for_errors()

        if progress is not None:
            progress.report(
                None,
                f"part {part_ind + 1}/{len(parts)}: building {n_loops} x {n_loops} loop inclusion matrix",
                force=True,
            )
        with progress_scope(start + 0.4 * (end - start), start + 0.9 * (end - start), "Build loop inclusion matrix"):
            contains = build_intrinsic_loop_inclusion_matrix(part, validate=False, progress=progress)
        part.loop_in_loop_matrix = contains.copy()

        if progress is not None:
            progress.report(None, f"Classify contour groups on support {part_ind + 1}/{len(parts)}", force=True)
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
        ring_order_diagnostics: list[dict[str, object]] = []
        regular_order = routing_selection_policy(config) == "regular"
        ring_axis = ring_support_axis(part.mesh) if regular_order else None
        for group_index, loop_indices in enumerate(part.loop_groups):
            ordered = sorted(loop_indices, key=lambda idx: len(higher_loops[idx]), reverse=True)
            if regular_order:
                ordered, order_diagnostics = regular_ring_order(part, ordered, signatures, contains, ring_axis)
                ring_order_diagnostics.append({"group_index": group_index, **order_diagnostics})
                part.loop_groups[group_index] = list(ordered)
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
        if regular_order:
            part.topology_diagnostics["regular_ring_traversal"] = ring_order_diagnostics
        if progress is not None:
            progress.report(
                end,
                message=(
                    f"part {part_ind + 1}/{len(parts)}: {n_loops} loops -> "
                    f"{len(part.groups)} group(s), {len(part.group_levels)} level(s)"
                ),
            )
    return parts


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
                "component_euler_characteristics": list(signature.component_euler_characteristics),
            }
            for signature in signatures
        ],
    }
