"""Revision-bound group choices and explicit electrical circuit partitioning.

The default path does not call partitioning. A circuit is a single-path routing
context with original substrate geometry and directed source contours. No
fictional wire joins, new support surfaces, or fitted currents are introduced.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
import json
import re

import numpy as np

from .models import CoilPart, Mesh, RoutedCircuit

SCHEMA = "coilgen.group-routing/1"
SPIRAL_ORDERS = ("auto", "spiral_in", "spiral_out")
CUT_SIDES = ("auto", "high", "low")


def _hash_array(digest, array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())


def group_snapshot(parts):
    """Describe the exact pre-assembly grouping, in source support coordinates."""
    digest = sha256()
    rows = []
    for support, part in enumerate(parts):
        _hash_array(digest, part.mesh.vertices, "<f8")
        _hash_array(digest, part.mesh.faces, "<i8")
        digest.update(repr(None if part.contour_step is None else float(part.contour_step)).encode("ascii"))
        for loop in part.contour_lines:
            _hash_array(digest, loop.v, "<f8")
            digest.update(
                repr((float(loop.potential), float(loop.current_orientation), int(loop.turn_multiplicity))).encode(
                    "ascii"
                )
            )
        digest.update(json.dumps([group.loop_indices for group in part.groups]).encode("ascii"))
        for index, group in enumerate(part.groups):
            levels = [i for i, level in enumerate(part.group_levels) if index in level]
            parents = [part.level_positions[i] for i in levels if i < len(part.level_positions)]
            points = np.concatenate([loop.v for loop in group.loops])
            rows.append(
                {
                    "key": f"s{support}:g{index}",
                    "label": f"S{support + 1} / G{index + 1}",
                    "support_index": support,
                    "group_index": index,
                    "loop_indices": list(group.loop_indices),
                    "contour_count": len(group.loops),
                    "level_indices": levels,
                    "parent_groups": parents,
                    "center_m": np.mean(points, axis=0).tolist(),
                }
            )
    return {"schema": SCHEMA, "fingerprint": digest.hexdigest(), "groups": rows}


def validate_group_plan(plan, snapshot=None):
    if plan is None:
        return
    if not isinstance(plan, dict) or set(plan) != {"schema", "fingerprint", "mode", "groups"}:
        raise ValueError("A group plan requires schema, fingerprint, mode and groups.")
    if (
        plan["schema"] != SCHEMA
        or not isinstance(plan["fingerprint"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", plan["fingerprint"])
    ):
        raise ValueError("The group plan has an unsupported schema or fingerprint.")
    if not isinstance(plan["mode"], str) or plan["mode"] not in {"existing", "custom", "per_group"}:
        raise ValueError("Group circuit mode must be existing, custom or per_group.")
    if not isinstance(plan["groups"], (list, tuple)) or not plan["groups"]:
        raise ValueError("A group plan must contain the complete group assignment.")
    keys = []
    for row in plan["groups"]:
        if not isinstance(row, dict) or set(row) != {"key", "circuit", "spiral", "cut"}:
            raise ValueError("Each group requires key, circuit, spiral and cut.")
        if not isinstance(row["key"], str) or not re.fullmatch(r"s\d+:g\d+", row["key"]):
            raise ValueError("Group keys must refer to an original support and group.")
        if not isinstance(row["circuit"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", row["circuit"]):
            raise ValueError("Circuit names require 1–40 letters, digits, underscores or hyphens.")
        if (
            not isinstance(row["spiral"], str)
            or not isinstance(row["cut"], str)
            or row["spiral"] not in SPIRAL_ORDERS
            or row["cut"] not in CUT_SIDES
        ):
            raise ValueError("Unsupported group spiral order or cut side.")
        keys.append(row["key"])
    if len(keys) != len(set(keys)):
        raise ValueError("A group can belong to exactly one circuit.")
    if snapshot is not None:
        if plan["fingerprint"] != snapshot["fingerprint"]:
            raise ValueError("Group geometry or membership changed. Prepare groups again and review assignments.")
        if set(keys) != {row["key"] for row in snapshot["groups"]}:
            raise ValueError("Every source group must be assigned exactly once; groups cannot be omitted.")


def default_group_plan(snapshot, config):
    """Opening an editor is not a model edit, including legacy cut-side choices."""
    rows = []
    force = tuple(config.force_cut_selection)
    for global_index, row in enumerate(snapshot["groups"]):
        index = row["group_index"] if config.routing_circuit_scope == "per_part" else global_index
        side = force[0] if len(force) == 1 else force[index] if index < len(force) else "auto"
        side = "auto" if side == "none" else side
        circuit = f"S{row['support_index'] + 1}" if config.routing_circuit_scope == "per_part" else "C1"
        rows.append({"key": row["key"], "circuit": circuit, "spiral": "auto", "cut": side})
    return {"schema": SCHEMA, "fingerprint": snapshot["fingerprint"], "mode": "existing", "groups": rows}


def _subset(part, group_indices, keys):
    """Retain directed contour membership without duplicating its numerical field."""
    selected = [part.groups[index] for index in group_indices]
    loop_indices = sorted({index for group in selected for index in group.loop_indices})
    if sum(len(group.loop_indices) for group in selected) != len(loop_indices):
        raise ValueError("Overlapping source group membership is unsupported.")
    remap = {old: new for new, old in enumerate(loop_indices)}
    contours = [deepcopy(part.contour_lines[index]) for index in loop_indices]
    groups = []
    for group in selected:
        copy = deepcopy(group)
        copy.loop_indices = [remap[index] for index in group.loop_indices]
        copy.loops = [contours[index] for index in copy.loop_indices]
        copy.outer_loop_indices = tuple(remap[index] for index in group.outer_loop_indices)
        groups.append(copy)
    group_remap = {old: new for new, old in enumerate(group_indices)}
    levels, positions = [], []
    for index, level in enumerate(part.group_levels):
        members = [group_remap[group] for group in level if group in group_remap]
        if members:
            levels.append(members)
            parents = part.level_positions[index] if index < len(part.level_positions) else []
            positions.append([group_remap[parent] for parent in parents if parent in group_remap])
    result = CoilPart(
        mesh=part.mesh,
        stream_function=part.stream_function,
        # Source FEM operators/densities belong to PipelineState.source_parts.
        # Repeating them for each circuit would multiply sheet fields and archives.
        contour_step=part.contour_step,
        potential_level_list=part.potential_level_list,
        contour_lines=contours,
        groups=groups,
        loop_groups=[list(group.loop_indices) for group in groups],
        group_levels=levels,
        level_positions=positions,
        loop_in_loop_matrix=None
        if part.loop_in_loop_matrix is None
        else part.loop_in_loop_matrix[np.ix_(loop_indices, loop_indices)].copy(),
        group_centers_v=np.asarray([np.concatenate([loop.v for loop in group.loops]).mean(axis=0) for group in groups]),
        topology_diagnostics={"source_group_keys": list(keys), "group_subset": True},
    )
    return result


def prepare_group_circuits(parts, config):
    """Partition only an explicitly requested plan; default objects stay intact."""
    plan = config.routing_group_plan
    snapshot = group_snapshot(parts)
    validate_group_plan(plan, snapshot)
    if plan is None:
        return parts
    assignments = {row["key"]: row for row in plan["groups"]}
    if plan["mode"] == "existing":
        for support, part in enumerate(parts):
            part.topology_diagnostics["group_editor_choices"] = [
                assignments[f"s{support}:g{index}"] for index in range(len(part.groups))
            ]
        return parts
    by_circuit = {}
    for row in snapshot["groups"]:
        key = row["key"]
        circuit = key.replace(":", "_") if plan["mode"] == "per_group" else assignments[key]["circuit"]
        by_circuit.setdefault(circuit, {}).setdefault(row["support_index"], []).append(row)
    from .routing_assembly import assemble_routing_parts

    result = []
    for circuit_id, sources in by_circuit.items():
        contexts, keys = [], []
        for support, rows in sources.items():
            source_keys = [row["key"] for row in rows]
            context = _subset(parts[support], [row["group_index"] for row in rows], source_keys)
            contexts.append(context)
            keys.extend(source_keys)
        merged = assemble_routing_parts(contexts)[0] if len(contexts) > 1 else contexts[0]
        values = {field.name: getattr(merged, field.name) for field in fields(CoilPart)}
        context = RoutedCircuit(
            **values, circuit_id=circuit_id, source_group_keys=tuple(keys), source_support_indices=tuple(sources)
        )
        context.topology_diagnostics["group_editor_choices"] = [assignments[key] for key in keys]
        result.append(context)
    currents = [part.contour_step for part in result]
    if not currents or any(
        current is None
        or not np.isfinite(current)
        or current == 0
        or not np.isclose(current, currents[0], rtol=1e-12, atol=0)
        for current in currents
    ):
        raise ValueError("Group circuits require one common finite nonzero contour current.")
    return result


def circuit_config(part, config):
    choices = part.topology_diagnostics.get("group_editor_choices")
    if choices is None:
        if config.routing_group_plan is not None:
            raise ValueError("Bind the group plan to its prepared source grouping before routing.")
        return config
    return replace(config, routing_group_plan=None, force_cut_selection=tuple(row["cut"] for row in choices))


def apply_spiral_choices(part):
    """Choose traversal order without reversing any source contour segment."""
    choices = part.topology_diagnostics.get("group_editor_choices", ())
    from .layout import _group_route_variants

    for group, choice in zip(part.connected_groups, choices):
        order = choice["spiral"]
        group.routing_spiral_order = order
        if order == "auto" or len(group.return_path_v) < 2:
            continue
        variants = _group_route_variants(group)
        if not variants:
            raise ValueError(f"The requested spiral order is unavailable for {choice['key']}.")
        variant = variants[0]
        group.v = np.vstack((variant.spiral_v, variant.return_path_v[1:]))
        if not np.allclose(group.v[0], group.v[-1], rtol=0, atol=1e-12):
            group.v = np.vstack((group.v, group.v[:1]))


def circuit_metadata(part):
    if not isinstance(part, RoutedCircuit):
        return {}
    return {
        "circuit_id": part.circuit_id,
        "source_group_keys": list(part.source_group_keys),
        "source_support_indices": list(part.source_support_indices),
    }


def validate_directed_circuits(parts):
    """Check protected source intervals before resampling or physical realization."""
    from collections import Counter

    for part in parts:
        points = None if part.wire_path is None else part.wire_path.v
        if (
            points is None
            or len(points) < 3
            or not np.all(np.isfinite(points))
            or not np.allclose(points[0], points[-1], rtol=0, atol=1e-12)
        ):
            raise ValueError("Every requested group circuit must be a finite closed directed path.")
        edges = Counter((tuple(a), tuple(b)) for a, b in zip(points[:-1], points[1:]))
        openings = []
        for cut in part.opening_cuts_among_groups:
            for points_name, center_name in (("cut1_v", "selected_point1_v"), ("cut2_v", "selected_point2_v")):
                cut_points = np.asarray(getattr(cut, points_name))
                centers = np.asarray(getattr(cut, center_name))
                if len(cut_points) and len(centers):
                    center = centers[0]
                    radius = float(np.max(np.linalg.norm(cut_points - center, axis=1)))
                    openings.append((center, radius))
        checked = 0
        for group in part.groups:
            for loop in group.opened_loops:
                retained = 0
                for a, b in zip(loop.v[:-1], loop.v[1:]):
                    length = np.linalg.norm(b - a)
                    if length <= 1e-12:
                        continue
                    if edges[(tuple(a), tuple(b))] == 1:
                        retained += 1
                        checked += 1
                        continue
                    # Inter-group openings can legitimately clip a previously
                    # opened contour. Only their recorded cut balls are exempt.
                    in_opening = any(
                        np.linalg.norm(
                            a + np.clip(np.dot(center - a, b - a) / (length * length), 0, 1) * (b - a) - center
                        )
                        <= radius + 1e-12
                        for center, radius in openings
                    )
                    if not in_opening:
                        raise ValueError("A protected directed contour interval is missing, reversed or duplicated.")
                if not retained:
                    raise ValueError("An entire source contour disappeared during circuit connection.")
        part.route_diagnostics["group_directed_interval_check"] = {
            "checked_intervals": checked,
            "cut_scope": "outside_recorded_inter_group_opening_balls",
            "all_source_contours_represented": True,
        }
    return parts


def review_circuit_clearance(parts, config):
    """Use the existing exact body-pair audit across independently routed circuits."""
    from .research.fabrication import FabricationRecord, assembly_clearance

    records = []
    for index, part in enumerate(parts, 1):
        for body_index, body in enumerate(part.swept_bodies):
            if body.fabrication_path is None:
                return {"checked": False, "reason": "missing_physical_circuit_geometry"}
            records.append(
                FabricationRecord(
                    f"circuit{index}_body{body_index}",
                    index,
                    body.contour_index,
                    body.fabrication_path,
                    body.mesh,
                    body.diagnostics,
                )
            )
    if len(records) != len(parts):
        return {"checked": False, "reason": "not_every_circuit_has_one_swept_body"}
    report = assembly_clearance(
        records,
        insulation_thickness_m=config.sweep_insulation_thickness,
        tolerance_m=config.sweep_manufacturing_tolerance,
        margin_m=config.sweep_bend_clearance_margin,
    )
    if not report["passed"]:
        for part in parts:
            for diagnostics in [part.manufacturing_diagnostics, *(body.diagnostics for body in part.swept_bodies)]:
                diagnostics["manufacturing_ok"] = False
                previous = diagnostics.get("manufacturing_failure_reasons", "")
                diagnostics["manufacturing_failure_reasons"] = (
                    str(previous) + ";group_circuit_assembly_clearance"
                ).strip(";")
    return {"checked": True, **report}


def source_support_specs(parts):
    """One former per physical source, even when several circuits share it.

    Group unique source slabs by their first owning numerical context. Each
    slab is still cut by all conductor paths in the established former builder.
    """
    from .geometry import combine_meshes

    covered = set()
    specs = []
    for index, part in enumerate(parts):
        source_ids = getattr(part, "source_support_indices", ())
        if not source_ids:
            specs.append((index, part.mesh, ()))
            continue
        sources = part.topology_diagnostics.get("routing_assembly", {}).get("sources", [])
        meshes, owned = [], []
        for local, source_id in enumerate(source_ids):
            if source_id in covered:
                continue
            if len(source_ids) == 1:
                mesh = part.mesh
            else:
                source = sources[local]
                start, stop = source["vertex_range"]
                first, last = source["face_range"]
                mesh = Mesh(
                    vertices=part.mesh.vertices[start:stop],
                    faces=part.mesh.faces[first:last] - start,
                    vertex_normals=None if part.mesh.vertex_normals is None else part.mesh.vertex_normals[start:stop],
                    metadata=deepcopy(part.mesh.metadata.get("combined_from", [{}] * len(source_ids))[local]),
                )
            meshes.append(mesh)
            owned.append(source_id)
            covered.add(source_id)
        if meshes:
            specs.append((index, meshes[0] if len(meshes) == 1 else combine_meshes(meshes), tuple(owned)))
    return specs
