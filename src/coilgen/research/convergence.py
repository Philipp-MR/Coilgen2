"""Bounded integration checks for an unchanged piecewise-linear current sheet."""

from dataclasses import replace

import numpy as np

from ..basis import calculate_basis_functions, calculate_one_ring_by_mesh
from ..field_components import field_component_values
from ..geometry import compute_face_normals, compute_vertex_normals
from ..geometry_kernels import _point_triangle_sq_numpy
from ..models import CoilPart, Mesh, TargetField
from ..physics import calculate_sensitivity_matrix
from ..progress import ProgressReporter


def _subdivide_sheet(mesh, stream):
    """Split triangles in their original planes and interpolate the same P1 field."""
    vertices, values = list(mesh.vertices), list(stream)
    midpoints, faces = {}, []

    def midpoint(a, b):
        edge = tuple(sorted((int(a), int(b))))
        if edge not in midpoints:
            midpoints[edge] = len(vertices)
            vertices.append((mesh.vertices[edge[0]] + mesh.vertices[edge[1]]) / 2)
            values.append((stream[edge[0]] + stream[edge[1]]) / 2)
        return midpoints[edge]

    for a, b, c in mesh.faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return Mesh(np.asarray(vertices), np.asarray(faces), metadata=dict(mesh.metadata)), np.asarray(values)


def _integer_sequence(values, name, minimum):
    values = tuple(values)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum for value in values
    ):
        raise ValueError(f"{name} must contain integers >= {minimum}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate settings.")
    return tuple(sorted(int(value) for value in values))


def _reject_on_sheet_samples(mesh, points):
    from scipy.spatial import cKDTree

    triangles = mesh.vertices[mesh.faces]
    centres = triangles.mean(axis=1)
    radius = float(np.max(np.linalg.norm(triangles - centres[:, None], axis=2)))
    tolerance = 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(points)), np.max(np.abs(mesh.vertices)))
    tree = cKDTree(centres)
    for start in range(0, len(points), 128):
        samples = points[start : start + 128]
        candidates = tree.query_ball_point(samples, radius + tolerance)
        counts = np.asarray([len(ids) for ids in candidates])
        if not counts.sum():
            continue
        if counts.sum() > 1_000_000:
            raise ValueError("Sheet proximity workspace exceeds its budget.")
        ids = np.concatenate([np.asarray(ids, int) for ids in candidates])
        repeated = np.repeat(samples, counts, axis=0)
        if np.any(_point_triangle_sq_numpy(repeated, triangles[ids]) <= tolerance**2):
            raise ValueError("Integration samples must lie off the singular current-sheet surface.")


def check_sheet_integration(
    result,
    points_m,
    *,
    relative_tolerance,
    absolute_tolerance_T=0.0,
    orders=(2, 4),
    subdivisions=(0, 1),
    max_operator_bytes=256 * 1024**2,
    max_integration_work=100_000_000,
):
    """Compare quadrature and triangle subdivisions without solving or refitting.

    The reference uses the highest requested order and subdivision. Acceptance
    describes only agreement within this tested integration family. Subdivision
    interpolates the identical P1 stream function on the identical faceted
    support; it does not establish convergence of the optimized FEM solution.
    Explicit tolerances are scalar RMS [T] for scalar fields, or magnitude RMS
    [T] for vector fields. All evaluations use the same supplied SI/world grid.
    Samples on the singular zero-thickness sheet are rejected; no principal
    value or one-sided surface-field convention is implied.
    """
    orders = _integer_sequence(orders, "orders", 1)
    subdivisions = _integer_sequence(subdivisions, "subdivisions", 0)
    if len(orders) * len(subdivisions) < 2:
        raise ValueError("Integration comparison requires at least two settings.")
    for value, name in ((relative_tolerance, "relative_tolerance"), (absolute_tolerance_T, "absolute_tolerance_T")):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar.")
    for value, name in ((max_operator_bytes, "max_operator_bytes"), (max_integration_work, "max_integration_work")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("points_m must contain finite nonempty (n,3) world coordinates in metres.")
    if not result.coil_parts:
        raise ValueError("Integration comparison requires at least one current sheet.")
    face_count = sum(part.mesh.n_faces for part in result.coil_parts)
    if any(not part.mesh.n_faces for part in result.coil_parts):
        raise ValueError("Every current sheet requires a nonempty triangle mesh.")
    # Reject extreme levels before exponentiation, even for arbitrary-size
    # Python integers supplied by a caller.
    if subdivisions[-1] > int(max_integration_work).bit_length() // 2:
        raise ValueError("Requested triangle subdivision exceeds the integration work budget.")
    work = sum(len(points) * face_count * 4**level * order**2 for level in subdivisions for order in orders)
    if work > max_integration_work:
        raise ValueError(f"Requested integration work {work} exceeds budget {max_integration_work}.")
    for part in result.coil_parts:
        stream = np.asarray(part.stream_function)
        if stream.shape != (part.mesh.n_vertices,) or not np.all(np.isfinite(stream)):
            raise ValueError("Every current sheet requires a finite stream function matching its vertices.")
    # All part operators coexist. Bound their combined allocation, including
    # conservative midpoint growth, before building any basis or field matrix.
    upper_vertices = sum(part.mesh.n_vertices for part in result.coil_parts) + face_count * (4 ** subdivisions[-1] - 1)
    if 3 * len(points) * upper_vertices * 8 > max_operator_bytes:
        raise ValueError("Combined sensitivity matrices exceed max_operator_bytes.")
    for part in result.coil_parts:
        _reject_on_sheet_samples(part.mesh, points)
    config = result.input_data
    target = TargetField(
        points, np.zeros_like(points), np.ones(len(points)), np.zeros(len(points), bool), np.ones(len(points), int)
    )
    reporter = ProgressReporter.from_config(config, label="Sheet integration comparison")
    comparisons = []
    for level in subdivisions:
        parts = []
        for original in result.coil_parts:
            mesh, stream = original.mesh.copy(), np.array(original.stream_function, copy=True)
            for _ in range(level):
                mesh, stream = _subdivide_sheet(mesh, stream)
            mesh.face_normals, mesh.vertex_normals = compute_face_normals(mesh), compute_vertex_normals(mesh)
            parts.append(CoilPart(mesh=mesh, stream_function=stream))
        calculate_one_ring_by_mesh(parts)
        calculate_basis_functions(parts)
        for order in orders:
            reporter.report(None, f"Quadrature order {order}, triangle subdivision {level}", force=True)
            calculate_sensitivity_matrix(parts, target, replace(config, gauss_order=order))
            field = sum(np.einsum("ctv,v->tc", part.sensitivity_matrix, part.stream_function) for part in parts)
            selected = np.asarray(
                field_component_values(field, config.target_field_component, b0_direction=config.b_0_direction)
            ).reshape(len(points), -1)
            if not np.all(np.isfinite(selected)):
                raise ValueError("Integration comparison produced nonfinite active fields.")
            comparisons.append(
                {
                    "quadrature_order": order,
                    "triangle_subdivision": level,
                    "field_T": field,
                    "_selected": selected,
                    "vertex_count": sum(p.mesh.n_vertices for p in parts),
                    "triangle_count": sum(p.mesh.n_faces for p in parts),
                }
            )
    reference = comparisons[-1]["_selected"]
    reference_rms = float(np.sqrt(np.mean(np.sum(reference**2, axis=1))))
    limit = absolute_tolerance_T + relative_tolerance * reference_rms
    for row in comparisons:
        difference = row.pop("_selected") - reference
        row["difference_rms_T"] = float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))
        row["difference_peak_T"] = float(np.max(np.linalg.norm(difference, axis=1)))
        row["relative_difference"] = row["difference_rms_T"] / reference_rms if reference_rms else None
        row["passed"] = row["difference_rms_T"] <= limit
    reporter.finish("Unchanged-sheet integration comparison complete")
    return {
        "available": True,
        "passed": all(row["passed"] for row in comparisons[:-1]),
        "comparisons": comparisons,
        "reference_rms_T": reference_rms,
        "agreement_limit_T": limit,
        "reference_quadrature_order": orders[-1],
        "reference_triangle_subdivision": subdivisions[-1],
        "points_m": np.array(points, copy=True),
        "current_refitted": False,
        "independent_quadrature_check": len(orders) > 1,
        "independent_mesh_solution_check": False,
        "scope": "Same piecewise-linear sheet current, faceted support and world sample grid; integration agreement only. No optimization-mesh convergence or hardware qualification.",
    }
