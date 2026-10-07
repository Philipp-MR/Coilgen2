"""Join grouped support parts into one directed-routing problem.

This only assembles inputs. A disjoint mesh union creates no support in the
space between its components. Cross-support transfers must pass the unchanged
support, lift, finite-conductor and field checks before a circuit is accepted.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from .models import CoilPart, ContourSparsificationResult, Mesh


def _concatenate_optional(objects, name):
    values = [getattr(item, name) for item in objects]
    return np.concatenate(values, axis=0) if all(value is not None for value in values) else None


def _offset_contour(contour, vertex_offset, face_offset):
    result = deepcopy(contour)
    for name, offset in (
        ("surface_faces", face_offset),
        ("segment_faces", face_offset),
        ("surface_edge_vertices", vertex_offset),
    ):
        indices = getattr(result, name)
        if indices is not None:
            setattr(result, name, np.where(indices >= 0, indices + offset, indices))
    return result


def assemble_routing_parts(parts: list[CoilPart], *, primary_surface_index: int = 0) -> list[CoilPart]:
    """Return one isolated routing part after grouping and before interconnection.

    Every source contour and its traversal direction is retained. Mesh, face,
    contour and group indices use contiguous disjoint-union offsets; their
    source surface ranges are recorded in ``topology_diagnostics['routing_assembly']``.
    Sources must prescribe one finite nonzero series current (``contour_step``).
    An independently powered or unequal-current assembly is not a series coil.

    The original primary surface supplies the level-list/reporting convention.
    Source FEM operators are not presented as an operator for the merged mesh;
    already evaluated stream functions and face current densities are retained.
    Source diagnostic and sparsification histories keep their local indices in
    the recorded source summaries. No original object is modified.
    """
    if len(parts) <= 1:
        return parts
    if not 0 <= primary_surface_index < len(parts):
        raise ValueError("The original primary surface index is outside the assembly.")
    current = parts[primary_surface_index].contour_step
    if current is None or not np.isfinite(current) or current == 0:
        raise ValueError("A single routing circuit requires a finite nonzero contour_step.")
    if any(
        part.contour_step is None
        or not np.isfinite(part.contour_step)
        or not np.isclose(part.contour_step, current, rtol=1e-12, atol=0.0)
        for part in parts
    ):
        raise ValueError("A single routing circuit requires equal contour_step on every source surface.")
    if any(part.wire_path is not None or part.connected_groups or part.fabrication_path is not None for part in parts):
        raise ValueError("Assemble grouped sources before routing or sweeping their conductors.")

    vertex_offsets = np.cumsum([0, *(part.mesh.n_vertices for part in parts)])
    face_offsets = np.cumsum([0, *(part.mesh.n_faces for part in parts)])
    contour_offsets = np.cumsum([0, *(len(part.contour_lines) for part in parts)])
    group_offsets = np.cumsum([0, *(len(part.groups) for part in parts)])
    mesh = Mesh(
        vertices=np.concatenate([part.mesh.vertices for part in parts]),
        faces=np.concatenate([part.mesh.faces + offset for part, offset in zip(parts, vertex_offsets)]),
        vertex_normals=_concatenate_optional([part.mesh for part in parts], "vertex_normals"),
        face_normals=_concatenate_optional([part.mesh for part in parts], "face_normals"),
        boundaries=[
            boundary.copy() + offset for part, offset in zip(parts, vertex_offsets) for boundary in part.mesh.boundaries
        ],
        stream_function=_concatenate_optional([part.mesh for part in parts], "stream_function"),
        metadata={"combined_from": [deepcopy(part.mesh.metadata) for part in parts]},
    )
    primary = parts[primary_surface_index]
    result = CoilPart(
        mesh=mesh,
        stream_function=_concatenate_optional(parts, "stream_function"),
        current_density=_concatenate_optional(parts, "current_density"),
        potential_level_list=deepcopy(primary.potential_level_list),
        contour_step=float(current),
        contour_level_diagnostics=deepcopy(primary.contour_level_diagnostics),
        loop_significance=_concatenate_optional(parts, "loop_significance"),
    )
    centers, roots, sources = [], [], []
    has_sparsification = any(part.contour_sparsification is not None for part in parts)
    sparsification = ContourSparsificationResult() if has_sparsification else None
    for index, part in enumerate(parts):
        vertex_offset, face_offset = int(vertex_offsets[index]), int(face_offsets[index])
        contour_offset, group_offset = int(contour_offsets[index]), int(group_offsets[index])
        assigned = [loop_index for group in part.groups for loop_index in group.loop_indices]
        if sorted(assigned) != list(range(len(part.contour_lines))):
            raise ValueError(f"Source surface {index} does not assign every contour to exactly one group.")
        contours = [_offset_contour(loop, vertex_offset, face_offset) for loop in part.contour_lines]
        result.contour_lines.extend(contours)
        for local_index, group in enumerate(part.groups):
            if len(group.loops) != len(group.loop_indices) or any(
                not np.array_equal(loop.v, part.contour_lines[loop_index].v)
                for loop, loop_index in zip(group.loops, group.loop_indices)
            ):
                raise ValueError(f"Source surface {index} has inconsistent directed group membership.")
            copied = deepcopy(group)
            copied.loops = [contours[loop_index] for loop_index in group.loop_indices]
            copied.loop_indices = [loop_index + contour_offset for loop_index in group.loop_indices]
            copied.outer_loop_indices = tuple(loop_index + contour_offset for loop_index in group.outer_loop_indices)
            copied.opened_loops = [_offset_contour(loop, vertex_offset, face_offset) for loop in group.opened_loops]
            result.groups.append(copied)
            result.loop_groups.append(copied.loop_indices.copy())
            if part.group_centers_v is not None and np.shape(part.group_centers_v) == (len(part.groups), 3):
                centers.append(part.group_centers_v[local_index].copy())
            else:
                centers.append(np.concatenate([loop.v for loop in copied.loops]).mean(axis=0))
        if part.group_levels:
            if len(part.group_levels) != len(part.level_positions):
                raise ValueError(f"Source surface {index} has inconsistent group hierarchy.")
            for level, positions in zip(part.group_levels, part.level_positions):
                if any(not 0 <= group < len(part.groups) for group in (*level, *positions)):
                    raise ValueError(f"Source surface {index} has invalid group hierarchy indices.")
                mapped = [group + group_offset for group in level]
                if positions:
                    result.group_levels.append(mapped)
                    result.level_positions.append([group + group_offset for group in positions])
                else:
                    roots.extend(mapped)
        else:
            roots.extend(range(group_offset, group_offset + len(part.groups)))
        source = {
            "source_surface_index": index,
            "vertex_range": [vertex_offset, int(vertex_offsets[index + 1])],
            "face_range": [face_offset, int(face_offsets[index + 1])],
            "contour_range": [contour_offset, int(contour_offsets[index + 1])],
            "group_range": [group_offset, int(group_offsets[index + 1])],
            "contour_step_a": float(part.contour_step),
            "potential_level_list": None if part.potential_level_list is None else part.potential_level_list.tolist(),
            "contour_level_diagnostics": deepcopy(part.contour_level_diagnostics),
            "topology_diagnostics": deepcopy(part.topology_diagnostics),
            "contour_distance_diagnostics": deepcopy(part.unconnected_contour_wire_distance_diagnostics),
        }
        if sparsification is not None:
            original = part.contour_sparsification
            source_loops = original.source_contours if original is not None else part.contour_lines
            source_offset = len(sparsification.source_contours)
            sparsification.source_contours.extend(
                _offset_contour(loop, vertex_offset, face_offset) for loop in source_loops
            )
            membership = (
                original.final_source_contour_indices
                if original is not None
                else [(loop_index,) for loop_index in range(len(part.contour_lines))]
            )
            if len(membership) != len(part.contour_lines):
                raise ValueError(f"Source surface {index} has incomplete sparsification provenance.")
            if any(
                not members or any(not 0 <= member < len(source_loops) for member in members) for members in membership
            ):
                raise ValueError(f"Source surface {index} has invalid sparsification source indices.")
            sparsification.final_source_contour_indices.extend(
                tuple(member + source_offset for member in members) for members in membership
            )
            if original is not None:
                sparsification.realized_turn_contours.extend(
                    _offset_contour(loop, vertex_offset, face_offset) for loop in original.realized_turn_contours
                )
                # Historical representative indices refer to intermediate local
                # contour sets. Retain their source namespace instead of guessing.
                source["sparsification_merge_records"] = [asdict(record) for record in original.merge_records]
                source["sparsification_diagnostics"] = deepcopy(original.diagnostics)
            source["sparsification_source_range"] = [source_offset, len(sparsification.source_contours)]
        sources.append(source)
    if roots:
        result.group_levels.insert(0, list(dict.fromkeys(roots)))
        result.level_positions.insert(0, [])
    result.group_centers_v = np.asarray(centers, dtype=float).reshape(-1, 3)
    if all(part.loop_in_loop_matrix is not None for part in parts):
        result.loop_in_loop_matrix = np.zeros((len(result.contour_lines), len(result.contour_lines)), dtype=bool)
        for index, part in enumerate(parts):
            start, end = contour_offsets[index : index + 2]
            if np.shape(part.loop_in_loop_matrix) != (end - start, end - start):
                raise ValueError(f"Source surface {index} has an inconsistent containment matrix.")
            result.loop_in_loop_matrix[start:end, start:end] = part.loop_in_loop_matrix
    assembly = {
        "source_part_count": len(parts),
        "original_primary_surface_index": int(primary_surface_index),
        "sources": sources,
        "group_source_surface_indices": [index for index, part in enumerate(parts) for _ in part.groups],
        "contour_source_surface_indices": [index for index, part in enumerate(parts) for _ in part.contour_lines],
        "single_circuit_required": True,
        "support_policy": "unchanged_disjoint_union_no_gap_support_invented",
        "physical_feasibility_proven": False,
    }
    mesh.metadata["routing_assembly"] = {
        "source_part_count": len(parts),
        "sources": [
            {name: source[name] for name in ("source_surface_index", "vertex_range", "face_range")}
            for source in sources
        ],
    }
    result.topology_diagnostics = {"topological_grouping_method": "source_part_assembly", "routing_assembly": assembly}
    if sparsification is not None:
        sparsification.diagnostics = {"source_part_histories": "topology_diagnostics.routing_assembly.sources"}
        result.contour_sparsification = sparsification
    result.combined_loop_length = sum(part.combined_loop_length for part in parts)
    # These source-only minima do not claim to include new cross-support pairs.
    result.minimal_contour_distance = float("nan")
    result.unconnected_contour_min_wire_distance = float("nan")
    result.unconnected_contour_wire_distance_diagnostics = {"reason": "assembly_cross_support_distances_not_evaluated"}
    return [result]


def validate_routing_assembly(parts: list[CoilPart], config) -> list[CoilPart]:
    """Fail explicitly unless the assembled conductor is one certified body.

    Run after joint routing and its production sweep. Existing manufacturing
    flags alone are insufficient: repeat the exact swept-triangle audit and
    independent whole-body support certificate on the final geometry. An
    adopted joint candidate must also carry its fixed-current field evidence.
    Baselines receive no implicit field-acceptance claim; final field reporting
    and configured field gates still apply in the ordinary evaluation stage.
    """
    assemblies = [part for part in parts if "routing_assembly" in part.topology_diagnostics]
    if not assemblies:
        return parts
    if len(parts) != 1 or len(assemblies) != 1:
        raise ValueError("Single-circuit routing assembly failed: multiple independent conductor parts remain.")
    part = assemblies[0]
    reasons = []
    path = part.wire_path
    if path is None or len(path.v) < 4 or not np.all(np.isfinite(path.v)):
        reasons.append("missing_finite_closed_route")
    elif not np.array_equal(path.v[0], path.v[-1]):
        reasons.append("raw_route_is_not_closed")
    elif np.any(np.linalg.norm(np.diff(path.v, axis=0), axis=1) <= 1e-12):
        reasons.append("raw_route_has_degenerate_intervals")
    trace = part.fabrication_path
    if trace is None or not trace.closed or not np.all(np.isfinite(trace.points_m)):
        reasons.append("missing_closed_fabrication_path")
    if part.swept_body is None:
        reasons.append("missing_swept_body")
    diagnostics = {"single_circuit_certified": False, "field_acceptance_checked": False}
    if not reasons:
        from .group_route_support import certify_group_route_support
        from .group_route_validation import GroupRouteValidationLimits, _check_swept_geometry

        try:
            body, body_reasons = _check_swept_geometry(part, GroupRouteValidationLimits())
            diagnostics["manufacturing"] = body
            reasons.extend(body_reasons)
            support = certify_group_route_support(part, config)
            diagnostics["support"] = support.diagnostics
            reasons.extend(support.reasons)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            reasons.append("assembly_physical_validation_failed")
            diagnostics["detail"] = str(error)
    selected = part.joint_route_diagnostics
    if selected.get("status") == "accepted":
        attempts = selected.get("attempts", ())
        index = selected.get("selected_attempt_index", -1)
        attempt = attempts[index] if isinstance(index, int) and 0 <= index < len(attempts) else {}
        validation = attempt.get("validation", {})
        current = validation.get("field_current_a")
        valid_current = (
            current is not None
            and part.contour_step is not None
            and np.isclose(current, part.contour_step, rtol=1e-12, atol=0.0)
        )
        if (
            not attempt.get("selected")
            or not attempt.get("eligible_for_adoption")
            or attempt.get("reasons")
            or not validation.get("route_checked")
            or not validation.get("physical_checked")
            or not valid_current
            or validation.get("route_current_refitted", True)
            or validation.get("physical_current_refitted", True)
        ):
            reasons.append("selected_joint_route_lacks_fixed_current_acceptance_evidence")
        else:
            diagnostics["field_acceptance_checked"] = True
    diagnostics["reasons"] = list(dict.fromkeys(reasons))
    diagnostics["single_circuit_certified"] = not reasons
    part.route_diagnostics["routing_assembly_validation"] = diagnostics
    part.topology_diagnostics["routing_assembly"]["physical_feasibility_proven"] = not reasons
    if reasons:
        raise ValueError("Single-circuit routing assembly failed: " + "; ".join(diagnostics["reasons"]) + ".")
    return parts
