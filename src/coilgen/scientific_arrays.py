"""Display/export arrays with their original scientific representation and SI units."""

from __future__ import annotations

import numpy as np


def scientific_arrays(parts, target, mesh=None, sheet_field=None) -> dict:
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
        for contour_index, contour in enumerate(part.contour_lines):
            arrays[f"contour_{part_index}_{contour_index}_points_m"] = contour.v
        if part.wire_path is not None:
            arrays[f"route_{part_index}_points_m"] = part.wire_path.v
        if part.fabrication_path is not None:
            arrays[f"fabrication_{part_index}_points_m"] = part.fabrication_path.wire_points_m
        for body_index, body in enumerate(part.swept_bodies):
            if body.fabrication_path is not None:
                arrays[f"fabrication_{part_index}_{body_index}_points_m"] = body.fabrication_path.wire_points_m
        # Preserve computed triangular boundaries instead of reconstructing solids.
        bodies = [body.mesh for body in part.swept_bodies]
        if not bodies and part.swept_body is not None:
            bodies = [part.swept_body]
        for body_index, body_mesh in enumerate(bodies):
            arrays[f"swept_{part_index}_{body_index}_vertices_m"] = body_mesh.vertices
            arrays[f"swept_{part_index}_{body_index}_faces"] = body_mesh.faces
    return arrays
