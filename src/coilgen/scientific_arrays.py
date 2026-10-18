"""Display/export arrays with their original scientific representation and SI units."""

from __future__ import annotations

import numpy as np


def path_provenance_arrays(prefix, path) -> dict:
    """Retain typed interval ownership alongside its exact saved coordinates."""
    arrays = {}
    owners = tuple(getattr(path, "segment_owners", ()))
    for key in ("kind", "group_index", "loop_index", "connector_index", "connector_role"):
        if owners:
            arrays[f"{prefix}_owner_{key}"] = np.asarray([getattr(owner, key) for owner in owners])
    geometry = tuple(getattr(path, "segment_geometry", ()))
    if geometry:
        arrays[f"{prefix}_segment_geometry"] = np.asarray([value.value for value in geometry])
    return arrays


def fabrication_frame_arrays(prefix, path) -> dict:
    """Saved frames define the physical profile, including asymmetric sections."""
    arrays = path_provenance_arrays(prefix, path)
    for key in ("points_m", "normals", "transverse", "cross_section_m"):
        arrays[f"{prefix}_frame_{key}"] = getattr(path, key)
    arrays[f"{prefix}_frame_closed"] = np.asarray(path.closed)
    return arrays


def routing_evidence(parts) -> dict:
    """Small adoption/current descriptors; geometry remains in scientific arrays."""
    keys = (
        "status",
        "reason",
        "selected_route_status",
        "selected_manufacturing_failure_reasons",
        "selection_policy",
        "candidate_count",
        "accepted_improvement_found_within_budget",
        "infeasibility_proven",
    )
    return {
        "scope": "recorded_routes_and_typed_interval_ownership",
        "parts": [
            {
                "part_index": index,
                "current_A": part.contour_step,
                "loop_groups": [list(group.loop_indices) for group in part.groups],
                "joint_routing": {
                    key: part.joint_route_diagnostics[key] for key in keys if key in part.joint_route_diagnostics
                },
            }
            for index, part in enumerate(parts)
        ],
    }


def scientific_arrays(parts, target, mesh=None, sheet_field=None, *, include_routing=False) -> dict:
    """Select recorded geometry and fields; never construct routes or fit currents."""
    if mesh is None:
        vertices = np.vstack([part.mesh.vertices for part in parts])
        offsets = np.cumsum([0] + [part.mesh.n_vertices for part in parts[:-1]])
        faces = np.vstack([part.mesh.faces + offset for part, offset in zip(parts, offsets)])
    else:
        vertices, faces = mesh.vertices, mesh.faces
    arrays = {
        "vertices_m": vertices,
        "faces": faces,
        "target_points_m": target.coords,
        "target_field_T": target.b,
        "target_suppressed": target.suppressed,
        "target_weights": target.weights,
    }
    if mesh is not None and mesh.stream_function is not None:
        arrays["stream_function_A"] = mesh.stream_function
    if sheet_field is not None:
        arrays["sheet_field_T"] = sheet_field
    densities = [getattr(part, "current_density", None) for part in parts]
    if densities and all(
        value is not None and np.shape(value) == (part.mesh.n_faces, 3) for part, value in zip(parts, densities)
    ):
        values = np.vstack(densities)
        if np.all(np.isfinite(values)):
            arrays["sheet_current_density_A_per_m"] = values
            arrays["sheet_current_density_points_m"] = np.vstack(
                [part.mesh.vertices[part.mesh.faces].mean(axis=1) for part in parts]
            )
    for part_index, part in enumerate(parts):
        if include_routing:
            arrays[f"support_{part_index}_vertices_m"] = part.mesh.vertices
            arrays[f"support_{part_index}_faces"] = part.mesh.faces
        for contour_index, contour in enumerate(part.contour_lines):
            arrays[f"contour_{part_index}_{contour_index}_points_m"] = contour.v
        if part.wire_path is not None:
            arrays[f"route_{part_index}_points_m"] = part.wire_path.v
            if include_routing:
                arrays.update(path_provenance_arrays(f"route_{part_index}", part.wire_path))
        if part.fabrication_path is not None:
            arrays[f"fabrication_{part_index}_points_m"] = part.fabrication_path.wire_points_m
            if include_routing:
                arrays.update(fabrication_frame_arrays(f"fabrication_{part_index}", part.fabrication_path))
        for body_index, body in enumerate(part.swept_bodies):
            if body.fabrication_path is not None:
                arrays[f"fabrication_{part_index}_{body_index}_points_m"] = body.fabrication_path.wire_points_m
                if include_routing:
                    arrays.update(
                        fabrication_frame_arrays(f"fabrication_{part_index}_{body_index}", body.fabrication_path)
                    )
        # Preserve computed triangular boundaries instead of reconstructing solids.
        bodies = [body.mesh for body in part.swept_bodies]
        if not bodies and part.swept_body is not None:
            bodies = [part.swept_body]
        for body_index, body_mesh in enumerate(bodies):
            arrays[f"swept_{part_index}_{body_index}_vertices_m"] = body_mesh.vertices
            arrays[f"swept_{part_index}_{body_index}_faces"] = body_mesh.faces
    return arrays
