from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .config import CoilGenConfig
from .geometry import compute_face_normals, compute_vertex_normals, rotation_matrix_by_vector
from .models import CoilGenResult, CoilPart, Mesh, WirePathGeometry
from .wire_provenance import transfer_segment_geometry

_DEFAULT_MESH_AREA_TOLERANCE_M2 = 1e-12


def create_swept_coil_meshes(parts: list[CoilPart], config: CoilGenConfig) -> list[CoilPart]:
    """Create triangulated conductor boundaries for visualization and mesh export.

    This backend deliberately does not claim to create an analytic CAD solid.
    Manufacturing readiness only means that the triangle boundary and configured
    route constraints passed their checks; robust support subtraction belongs to
    the separate analytic-solid backend.
    """

    if config.skip_sweep:
        for part in parts:
            part.swept_body = None
            part.manufacturing_diagnostics = {"swept_body_created": False, "manufacturing_ok": True}
        return parts

    cross_section = conductor_cross_section_points(config)
    for part in parts:
        if part.wire_path is None or len(part.wire_path.v) < 2:
            part.swept_body = None
            part.manufacturing_diagnostics = {"swept_body_created": False, "manufacturing_ok": False}
            continue
        part.swept_body, sweep_diagnostics = _sweep_wire_path_with_diagnostics(part, cross_section, config)
        part.manufacturing_diagnostics = mesh_manufacturing_diagnostics(part.swept_body)
        part.manufacturing_diagnostics.update(sweep_diagnostics)
        part.manufacturing_diagnostics.update(
            {
                "sweep_geometry_backend": "triangular_mesh",
                "sweep_geometry_role": "visualization_mesh_export",
                "analytic_solid_created": False,
            }
        )
        _finalize_manufacturing_diagnostics(part)
    return parts


def create_swept_coil_bodies(parts: list[CoilPart], config: CoilGenConfig) -> list[CoilPart]:
    """Compatibility wrapper for :func:`create_swept_coil_meshes`."""

    return create_swept_coil_meshes(parts, config)


def manufacturing_diagnostics_rows(result: CoilGenResult) -> list[dict[str, float | int | bool | str]]:
    """Return one manufacturing diagnostic row per coil part."""

    rows: list[dict[str, float | int | bool | str]] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        row: dict[str, float | int | bool | str] = {"part_index": part_ind}
        if part.swept_body is not None:
            if not part.manufacturing_diagnostics:
                part.manufacturing_diagnostics = mesh_manufacturing_diagnostics(part.swept_body)
            _finalize_manufacturing_diagnostics(part)
            row.update(part.manufacturing_diagnostics)
        elif part.manufacturing_diagnostics:
            row.update(part.manufacturing_diagnostics)
        else:
            row.update({"swept_body_created": False, "manufacturing_ok": False})
        rows.append(row)
    return rows


def save_manufacturing_diagnostics(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil_body",
) -> list[Path]:
    """Save swept-body manufacturing diagnostics as JSON and CSV."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = manufacturing_diagnostics_rows(result)
    json_path = output / f"{prefix}_manufacturing_diagnostics.json"
    csv_path = output / f"{prefix}_manufacturing_diagnostics.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_rows_csv(csv_path, rows)
    return [json_path, csv_path]


def wire_trace_rows(result: CoilGenResult) -> list[dict[str, int | float | str]]:
    """Return routed wire centerline points and interval geometry in SI units."""

    rows: list[dict[str, int | float | str]] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        if part.wire_path is None:
            continue
        points = np.asarray(part.wire_path.v, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            continue
        segment_lengths = np.zeros(len(points), dtype=float)
        if len(points) > 1:
            segment_lengths[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_lengths = np.cumsum(segment_lengths)
        for point_ind, (point, segment_length, cumulative_length) in enumerate(
            zip(points, segment_lengths, cumulative_lengths),
        ):
            segment_geometry = (
                part.wire_path.segment_geometry[point_ind - 1].value
                if point_ind > 0 and point_ind - 1 < len(part.wire_path.segment_geometry)
                else ""
            )
            rows.append(
                {
                    "part_index": int(part_ind),
                    "point_index": int(point_ind + 1),
                    "x_m": float(point[0]),
                    "y_m": float(point[1]),
                    "z_m": float(point[2]),
                    "segment_length_from_previous_m": float(segment_length),
                    "cumulative_length_m": float(cumulative_length),
                    "segment_geometry_from_previous": segment_geometry,
                }
            )
    return rows


def save_wire_trace_as_csv(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
) -> list[Path]:
    """Save routed wire centerlines using the configured CSV schema."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = getattr(result, "input_data", None)
    csv_format = str(getattr(config, "wire_trace_csv_format", "detailed")).strip().lower()
    if csv_format == "xyz_no_header":
        return _save_wire_paths_as_xyz_csv(result, output, prefix=prefix)

    path = output / f"{prefix}_wire_trace.csv"
    fieldnames = [
        "part_index",
        "point_index",
        "x_m",
        "y_m",
        "z_m",
        "segment_length_from_previous_m",
        "cumulative_length_m",
        "segment_geometry_from_previous",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wire_trace_rows(result))
    return [path]


def _save_wire_paths_as_xyz_csv(result: CoilGenResult, output: Path, *, prefix: str) -> list[Path]:
    """Write one headerless ordered X,Y,Z file per connected routed part."""

    routed_parts: list[tuple[int, np.ndarray]] = []
    for part_index, part in enumerate(result.coil_parts, start=1):
        if part.wire_path is None:
            continue
        points = np.asarray(part.wire_path.v, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            continue
        if not np.all(np.isfinite(points)):
            raise ValueError(f"Wire path for coil part {part_index} contains non-finite coordinates.")
        routed_parts.append((part_index, points))

    config = getattr(result, "input_data", None)
    unit = str(getattr(config, "wire_trace_csv_length_unit", "m")).strip().lower()
    scales = {"m": 1.0, "cm": 100.0, "mm": 1000.0}
    if unit not in scales:
        raise ValueError("wire_trace_csv_length_unit must be m, cm, or mm.")
    decimal_places = int(getattr(config, "wire_trace_csv_decimal_places", 9))
    if decimal_places < 0 or decimal_places > 15:
        raise ValueError("wire_trace_csv_decimal_places must lie in [0, 15].")
    coordinate_scale = scales[unit]
    formatter = f"{{:.{decimal_places}f}}".format

    files: list[Path] = []
    multiple = len(routed_parts) > 1
    for part_index, points in routed_parts:
        suffix = f"_part{part_index:03d}" if multiple else ""
        path = output / f"{prefix}_wire_path{suffix}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=",", lineterminator="\n")
            for point in points:
                writer.writerow(formatter(coordinate_scale * float(value)) for value in point)
        files.append(path)
    return files


def mesh_manufacturing_diagnostics(
    mesh: Mesh, tolerance: float = _DEFAULT_MESH_AREA_TOLERANCE_M2
) -> dict[str, float | int | bool]:
    """Check whether a triangular shell is a consistently oriented solid boundary.

    ``tolerance`` is an area tolerance in square metres. Its square root is
    used as the corresponding edge-length tolerance. This preserves the legacy
    face-area interpretation while making short-edge rejection dimensionally
    consistent.
    """

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    nonfinite_vertices = int(np.count_nonzero(~np.all(np.isfinite(vertices), axis=1))) if len(vertices) else 0
    invalid_face_indices = int(np.count_nonzero((faces < 0) | (faces >= len(vertices)))) if len(faces) else 0
    if invalid_face_indices == 0:
        valid_faces = faces
    else:
        valid_faces = faces[np.all((faces >= 0) & (faces < len(vertices)), axis=1)]
    areas = _face_areas(vertices, valid_faces) if len(valid_faces) else np.zeros(0, dtype=float)
    finite_areas = areas[np.isfinite(areas)]
    area_tolerance = max(float(tolerance), 0.0)
    edge_tolerance = float(np.sqrt(area_tolerance))
    volume_tolerance = edge_tolerance**3
    degenerate_faces = int(np.count_nonzero(~np.isfinite(areas) | (areas <= area_tolerance)))
    edge_lengths = _mesh_edge_lengths(vertices, valid_faces) if len(valid_faces) else np.zeros(0, dtype=float)
    finite_edge_lengths = edge_lengths[np.isfinite(edge_lengths)]
    short_edges = int(np.count_nonzero(~np.isfinite(edge_lengths) | (edge_lengths <= edge_tolerance)))
    open_edges, nonmanifold_edges, misoriented_edges = _edge_topology_counts(valid_faces)
    signed_volume = _signed_mesh_volume(vertices, valid_faces) if len(valid_faces) else 0.0
    positive_volume = bool(np.isfinite(signed_volume) and signed_volume > volume_tolerance)
    manufacturing_ok = (
        nonfinite_vertices == 0
        and invalid_face_indices == 0
        and degenerate_faces == 0
        and short_edges == 0
        and open_edges == 0
        and nonmanifold_edges == 0
        and misoriented_edges == 0
        and positive_volume
    )
    return {
        "swept_body_created": bool(len(vertices) > 0 and len(faces) > 0),
        "manufacturing_ok": bool(manufacturing_ok),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "nonfinite_vertex_count": int(nonfinite_vertices),
        "invalid_face_index_count": int(invalid_face_indices),
        "degenerate_face_count": int(degenerate_faces),
        "short_edge_count": int(short_edges),
        "open_edge_count": int(open_edges),
        "nonmanifold_edge_count": int(nonmanifold_edges),
        "misoriented_edge_count": int(misoriented_edges),
        "watertight": bool(open_edges == 0 and nonmanifold_edges == 0 and misoriented_edges == 0 and len(faces) > 0),
        "signed_volume_m3": float(signed_volume),
        "absolute_volume_m3": float(abs(signed_volume)) if np.isfinite(signed_volume) else 0.0,
        "positive_volume": bool(positive_volume),
        "edge_length_tolerance_m": float(edge_tolerance),
        "min_face_area": float(np.min(finite_areas)) if len(finite_areas) else 0.0,
        "max_face_area": float(np.max(finite_areas)) if len(finite_areas) else 0.0,
        "min_edge_length_m": float(np.min(finite_edge_lengths)) if len(finite_edge_lengths) else 0.0,
        "max_edge_length_m": float(np.max(finite_edge_lengths)) if len(finite_edge_lengths) else 0.0,
    }


def _finalize_manufacturing_diagnostics(part: CoilPart) -> None:
    """Combine mesh topology and physical sweep checks into export readiness."""

    diagnostics = part.manufacturing_diagnostics
    mesh_geometry_ok = bool(diagnostics.get("mesh_solid_ok", diagnostics.get("manufacturing_ok", False)))
    route_status = str(part.route_diagnostics.get("route_quality_status", "")).strip().lower()
    route_topology_reasons = _manufacturing_route_topology_failure_reasons(part)
    route_design_target_failure = _route_design_target_failure(part)
    reasons: list[str] = []
    if not mesh_geometry_ok:
        reasons.append("invalid_mesh_solid")
    if route_topology_reasons:
        reasons.append("route_topology_failure")
    if bool(diagnostics.get("sweep_route_clearance_violation", False)):
        reasons.append("route_clearance_violation")
    if bool(diagnostics.get("sweep_capsule_self_intersection", False)):
        reasons.append("sweep_self_intersection")
    if bool(diagnostics.get("sweep_mesh_self_intersection", False)):
        reasons.append("swept_mesh_self_intersection")
    if bool(diagnostics.get("sweep_self_intersection_skipped", False)):
        reasons.append("self_intersection_check_skipped")
    if bool(diagnostics.get("sweep_min_bend_radius_violation", False)):
        reasons.append("minimum_bend_radius_violation")
    if bool(diagnostics.get("sweep_curvature_envelope_violation", False)):
        reasons.append("curvature_envelope_violation")
    if bool(diagnostics.get("sweep_min_segment_length_violation", False)):
        reasons.append("minimum_segment_length_violation")
    if bool(diagnostics.get("sweep_local_fold_violation", False)):
        reasons.append("folded_sweep_quad")
    if bool(diagnostics.get("sweep_triangle_quality_violation", False)):
        reasons.append("poor_sweep_triangle_quality")
    if int(diagnostics.get("sweep_adaptive_depth_limit_count", 0)) > 0:
        reasons.append("adaptive_sampling_depth_limit")
    if int(diagnostics.get("sweep_adaptive_unresolved_count", 0)) > 0:
        reasons.append("unresolved_adaptive_sampling")
    if int(diagnostics.get("sweep_surface_projection_domain_fallback_count", 0)) > 0:
        reasons.append("unresolved_surface_trace")
    if bool(diagnostics.get("sweep_support_penetration_violation", False)):
        reasons.append("support_surface_penetration")
    if bool(diagnostics.get("sweep_centerline_support_domain_violation", False)):
        reasons.append("centerline_outside_support_domain")
    diagnostics["route_design_quality_status"] = route_status or "not_checked"
    diagnostics["route_design_target_failure"] = bool(route_design_target_failure)
    diagnostics["manufacturing_route_topology_failure"] = bool(route_topology_reasons)
    diagnostics["manufacturing_route_topology_failure_reasons"] = ";".join(route_topology_reasons)
    diagnostics["mesh_solid_ok"] = mesh_geometry_ok
    diagnostics["manufacturing_failure_reasons"] = ";".join(reasons)
    diagnostics["manufacturing_ok"] = bool(not reasons)


def _route_design_target_failure(part: CoilPart) -> bool:
    diagnostics = part.route_diagnostics
    return bool(
        int(diagnostics.get("wire_clearance_failure_count", 0)) > 0
        or int(diagnostics.get("inter_group_connector_clearance_violation_count", 0)) > 0
    )


def _manufacturing_route_topology_failure_reasons(part: CoilPart) -> list[str]:
    """Return route defects that cannot be cleared by a valid swept envelope.

    Requested wire/connector spacing is a design constraint and is deliberately
    excluded here. Its physical consequence is checked against the actual
    cross-section envelope by :func:`_physical_clearance_diagnostics`.
    """

    diagnostics = part.route_diagnostics
    reasons: list[str] = []
    degenerate_cuts = int(diagnostics.get("degenerate_opening_cut_count", 0))
    surface_connector_failures = int(diagnostics.get("surface_geodesic_connector_failure_count", 0))
    if degenerate_cuts > 0:
        reasons.append("degenerate_opening_cut")
    if surface_connector_failures > 0:
        reasons.append("surface_geodesic_connector")

    route_status = str(diagnostics.get("route_quality_status", "")).strip().lower()
    known_failures = (
        int(diagnostics.get("wire_clearance_failure_count", 0))
        + int(diagnostics.get("inter_group_connector_clearance_violation_count", 0))
        + degenerate_cuts
        + surface_connector_failures
    )
    reported_failures = int(diagnostics.get("route_quality_failure_count", known_failures))
    declared_reasons = {
        reason for reason in str(diagnostics.get("route_quality_failure_reasons", "")).split(";") if reason
    }
    known_reason_names = {
        "wire_clearance_target",
        "connector_clearance_target",
        "degenerate_opening_cut",
        "surface_geodesic_connector",
    }
    if route_status == "fail" and (
        (known_failures == 0 and not declared_reasons)
        or reported_failures > known_failures
        or any(reason not in known_reason_names for reason in declared_reasons)
    ):
        reasons.append("unspecified_route_failure")
    return reasons


def _required_centerline_clearance(cross_section: np.ndarray, config: CoilGenConfig) -> float:
    return float(
        2.0 * _cross_section_envelope_radius(cross_section)
        + 2.0 * float(config.sweep_insulation_thickness)
        + float(config.sweep_manufacturing_tolerance)
        + float(config.sweep_bend_clearance_margin)
    )


def _route_physical_clearance_precheck_failure(
    part: CoilPart,
    cross_section: np.ndarray,
    config: CoilGenConfig,
) -> bool:
    diagnostics = part.route_diagnostics
    segment_a = int(diagnostics.get("min_wire_wire_segment_a", -1))
    segment_b = int(diagnostics.get("min_wire_wire_segment_b", -1))
    if segment_a < 0 or segment_b < 0 or diagnostics.get("min_wire_wire_distance") is None:
        return False
    return bool(float(diagnostics["min_wire_wire_distance"]) < _required_centerline_clearance(cross_section, config))


def _manufacturing_route_precheck_reasons(
    part: CoilPart,
    cross_section: np.ndarray,
    config: CoilGenConfig,
) -> list[str]:
    reasons = _manufacturing_route_topology_failure_reasons(part)
    if _route_physical_clearance_precheck_failure(part, cross_section, config):
        reasons.append("route_clearance_violation")
    return reasons


def conductor_cross_section_points(config: CoilGenConfig) -> np.ndarray:
    """Return a validated convex conductor profile in physical local axes.

    Profile coordinate 0 is the in-surface direction transverse to the wire
    and coordinate 1 is the support-surface normal direction. The default
    ``((0, 0),)`` value is a sentinel requesting the configured rectangle.
    """

    configured = np.asarray(config.cross_sectional_points, dtype=float)
    if configured.shape == (1, 2) and np.all(configured == 0.0):
        width = max(float(config.conductor_cross_section_width), 1e-12)
        height = max(float(config.conductor_cross_section_height), 1e-12)
        configured = np.asarray(
            [
                [-width / 2.0, -height / 2.0],
                [width / 2.0, -height / 2.0],
                [width / 2.0, height / 2.0],
                [-width / 2.0, height / 2.0],
            ],
            dtype=float,
        )
    return _validated_cross_section(configured)


def sweep_wire_path(part: CoilPart, cross_section: np.ndarray, config: CoilGenConfig | None = None) -> Mesh:
    """Sweep ``cross_section`` along a coil part wire path and return a mesh."""

    mesh, _diagnostics = _sweep_wire_path_with_diagnostics(
        part, cross_section, CoilGenConfig() if config is None else config
    )
    return mesh


def _sweep_wire_path_with_diagnostics(
    part: CoilPart,
    cross_section: np.ndarray,
    config: CoilGenConfig,
) -> tuple[Mesh, dict[str, float | int | bool | str]]:
    source_points = np.asarray(part.wire_path.v, dtype=float)
    raw_points = _dedupe_path(source_points)
    closed_path = _is_closed_path(raw_points)
    raw_geometry = _transfer_path_segment_geometry(
        source_points,
        part.wire_path.segment_geometry,
        raw_points,
        source_closed=_is_closed_path(source_points),
        target_closed=closed_path,
    )
    if closed_path:
        raw_points = raw_points[:-1]
    cross_section = _validated_cross_section(cross_section)
    route_precheck_reasons = _manufacturing_route_precheck_reasons(part, cross_section, config)
    route_precheck_failed = bool(route_precheck_reasons)
    points, segment_geometry, path_diagnostics = _prepare_sweep_centerline(
        part,
        raw_points,
        raw_geometry,
        cross_section,
        config,
        closed=closed_path,
    )
    path_diagnostics["manufacturing_route_precheck_failure"] = route_precheck_failed
    path_diagnostics["manufacturing_route_precheck_failure_reasons"] = ";".join(route_precheck_reasons)
    minimum_point_count = 3 if closed_path else 2
    if len(points) < minimum_point_count:
        return Mesh(vertices=np.zeros((0, 3), dtype=float), faces=np.zeros((0, 3), dtype=np.int64)), path_diagnostics

    mounting_offset = _mounting_centerline_offset(config, cross_section)
    points, tangents, normals, binormals, vertices, ring_repair_diagnostics = _build_sweep_rings(
        part,
        points,
        segment_geometry,
        cross_section,
        config,
        closed=closed_path,
        mounting_offset=mounting_offset,
    )
    path_diagnostics.update(ring_repair_diagnostics)
    path_diagnostics["sweep_output_point_count"] = int(len(points))
    ring_count = len(points)
    section_count = len(cross_section)

    faces, loft_diagnostics = _triangulate_sweep_laterals(
        vertices,
        ring_count,
        section_count,
        closed=closed_path,
        optimize_diagonals=True,
    )
    loft_diagnostics["sweep_diagonal_optimization_skip_reason"] = ""

    if not closed_path:
        # A cap-centre fan avoids the severe sliver triangles produced by using
        # one polygon corner as the fan origin, especially for round profiles.
        start_base = 0
        end_base = (ring_count - 1) * section_count
        start_center = len(vertices)
        end_center = start_center + 1
        rings = vertices.reshape((ring_count, section_count, 3))
        vertices = np.vstack((vertices, np.mean(rings[0], axis=0), np.mean(rings[-1], axis=0)))
        for section_ind in range(section_count):
            section_next = (section_ind + 1) % section_count
            faces.append([start_center, start_base + section_ind, start_base + section_next])
            faces.append([end_center, end_base + section_next, end_base + section_ind])

    mesh = Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))
    diagnostics = {
        **path_diagnostics,
        "sweep_ring_count": int(ring_count),
        "sweep_closed_path": bool(closed_path),
        "sweep_end_cap_count": 0 if closed_path else 2,
        "sweep_section_point_count": int(section_count),
        "sweep_cross_section_axis_0": "surface_transverse",
        "sweep_cross_section_axis_1": "surface_normal",
        "conductor_mounting_mode": _normalised_text(config.conductor_mounting_mode),
        "sweep_centerline_offset_m": float(mounting_offset),
        "sweep_frame_method_requested": _normalised_text(config.sweep_frame_method),
        "sweep_frame_method": _effective_sweep_frame_method(config, segment_geometry),
    }
    diagnostics.update(loft_diagnostics)
    diagnostics.update(_swept_mesh_quality_diagnostics(mesh, config))
    diagnostics.update(
        _sweep_section_conservation_diagnostics(
            mesh,
            vertices[: ring_count * section_count],
            points,
            normals,
            binormals,
            cross_section,
            mounting_offset,
            closed=closed_path,
        )
    )
    diagnostics.update(
        _support_conformity_diagnostics(
            part,
            mesh,
            points,
            config,
        )
    )
    diagnostics.update(
        _physical_clearance_diagnostics(
            part,
            points,
            cross_section,
            config,
            closed=closed_path,
        )
    )
    diagnostics.update(_final_mesh_collision_diagnostics(part, mesh, diagnostics))
    return mesh, diagnostics


def _sweep_section_conservation_diagnostics(
    mesh: Mesh,
    ring_vertices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    cross_section: np.ndarray,
    mounting_offset: float,
    *,
    closed: bool,
) -> dict[str, float]:
    """Measure discrete profile-area and swept-volume conservation."""

    nominal_area = _polygon_area(np.asarray(cross_section, dtype=float))
    ring_count = len(points)
    section_count = len(cross_section)
    rings = np.asarray(ring_vertices, dtype=float).reshape((ring_count, section_count, 3))
    ring_area_vectors = 0.5 * np.sum(
        np.cross(rings, np.roll(rings, -1, axis=1)),
        axis=1,
    )
    ring_areas = np.linalg.norm(ring_area_vectors, axis=1)
    area_relative_errors = (
        np.abs(ring_areas - nominal_area) / nominal_area
        if nominal_area > 0.0
        else np.zeros(len(ring_areas), dtype=float)
    )
    profile_centroid = _polygon_centroid(np.asarray(cross_section, dtype=float))
    section_centers = points + (mounting_offset + profile_centroid[1]) * normals + profile_centroid[0] * binormals
    centerline_length = float(np.sum(_centerline_segment_lengths(section_centers, closed=closed)))
    expected_volume = nominal_area * centerline_length
    actual_volume = abs(_signed_mesh_volume(mesh.vertices, mesh.faces)) if mesh.n_faces else 0.0
    volume_relative_error = abs(actual_volume - expected_volume) / expected_volume if expected_volume > 0.0 else 0.0
    return {
        "sweep_nominal_cross_section_area_m2": float(nominal_area),
        "sweep_min_ring_cross_section_area_m2": float(np.min(ring_areas)) if len(ring_areas) else 0.0,
        "sweep_max_ring_cross_section_area_m2": float(np.max(ring_areas)) if len(ring_areas) else 0.0,
        "sweep_ring_cross_section_area_max_relative_error": (
            float(np.max(area_relative_errors)) if len(area_relative_errors) else 0.0
        ),
        "sweep_profile_centroid_path_length_m": float(centerline_length),
        "sweep_expected_profile_volume_m3": float(expected_volume),
        "sweep_profile_volume_relative_error": float(volume_relative_error),
    }


def _final_mesh_collision_diagnostics(
    part: CoilPart,
    mesh: Mesh,
    diagnostics: dict[str, float | int | bool | str],
) -> dict[str, float | int | bool | str]:
    """Select an exact or conservative final collision proof by problem size."""

    exact_triangle_face_limit = 20_000
    upstream_failure = (
        bool(_manufacturing_route_topology_failure_reasons(part))
        or bool(diagnostics.get("sweep_route_clearance_violation", False))
        or bool(diagnostics.get("sweep_capsule_self_intersection", False))
        or bool(diagnostics.get("sweep_local_fold_violation", False))
        or bool(diagnostics.get("sweep_triangle_quality_violation", False))
        or bool(diagnostics.get("sweep_support_penetration_violation", False))
    )
    if mesh.n_faces <= exact_triangle_face_limit:
        return _swept_mesh_self_intersection_diagnostics(mesh)
    if upstream_failure:
        return {
            "sweep_mesh_collision_broad_phase": "skipped_after_upstream_failure",
            "sweep_mesh_collision_candidate_pair_count": 0,
            "sweep_mesh_self_intersection_count": 0,
            "sweep_mesh_self_intersection_truncated": False,
            "sweep_mesh_self_intersection": False,
            "sweep_mesh_collision_check_skipped": True,
        }
    capsule_completed = _normalised_text(
        diagnostics.get("sweep_self_intersection_check", "")
    ) == "capsule_fast" and not bool(diagnostics.get("sweep_self_intersection_skipped", True))
    if capsule_completed:
        # Every loft triangle lies inside the convex hull of the endpoint
        # cross-section envelope spheres, i.e. its centerline segment capsule.
        # A passed remote-capsule check plus passed local fold/quality checks is
        # therefore a conservative non-intersection certificate for large
        # sweeps without an O(face-count) Python triangle traversal.
        return {
            "sweep_mesh_collision_broad_phase": "capsule_envelope_certificate",
            "sweep_mesh_collision_candidate_pair_count": int(diagnostics.get("sweep_capsule_candidate_pair_count", 0)),
            "sweep_mesh_self_intersection_count": 0,
            "sweep_mesh_self_intersection_truncated": False,
            "sweep_mesh_self_intersection": False,
            "sweep_mesh_collision_check_skipped": False,
        }
    return {
        "sweep_mesh_collision_broad_phase": "not_certified",
        "sweep_mesh_collision_candidate_pair_count": 0,
        "sweep_mesh_self_intersection_count": 0,
        "sweep_mesh_self_intersection_truncated": False,
        "sweep_mesh_self_intersection": False,
        "sweep_mesh_collision_check_skipped": True,
    }


def _triangulate_sweep_laterals(
    vertices: np.ndarray,
    ring_count: int,
    section_count: int,
    *,
    closed: bool,
    optimize_diagonals: bool,
) -> tuple[list[list[int]], dict[str, int | float | bool]]:
    """Triangulate loft quads with vectorized fold/aspect optimization."""

    segment_count = ring_count if closed else ring_count - 1
    ring_indices = np.repeat(np.arange(segment_count, dtype=np.int64), section_count)
    section_indices = np.tile(np.arange(section_count, dtype=np.int64), segment_count)
    following_rings = (ring_indices + 1) % ring_count
    following_sections = (section_indices + 1) % section_count
    a = ring_indices * section_count + section_indices
    b = ring_indices * section_count + following_sections
    c = following_rings * section_count + section_indices
    d = following_rings * section_count + following_sections
    primary = np.stack(
        (
            np.column_stack((a, c, d)),
            np.column_stack((a, d, b)),
        ),
        axis=1,
    )
    primary_folds, primary_aspects, primary_dots = _quad_triangulation_metrics(vertices, primary)
    alternate_count = 0
    selected = primary
    selected_folds = primary_folds
    selected_dots = primary_dots
    if optimize_diagonals:
        alternate = np.stack(
            (
                np.column_stack((a, c, b)),
                np.column_stack((b, c, d)),
            ),
            axis=1,
        )
        alternate_folds, alternate_aspects, alternate_dots = _quad_triangulation_metrics(vertices, alternate)
        use_alternate = (alternate_folds < primary_folds) | (
            (alternate_folds == primary_folds) & (alternate_aspects < primary_aspects)
        )
        selected = np.where(use_alternate[:, None, None], alternate, primary)
        selected_folds = np.where(use_alternate, alternate_folds, primary_folds)
        selected_dots = np.where(use_alternate, alternate_dots, primary_dots)
        alternate_count = int(np.count_nonzero(use_alternate))
    folded_quad_count = int(np.count_nonzero(selected_folds))
    worst_quad_normal_dot = float(np.min(selected_dots)) if len(selected_dots) else 1.0
    return selected.reshape((-1, 3)).tolist(), {
        "sweep_alternate_quad_diagonal_count": alternate_count,
        "sweep_folded_quad_count": int(folded_quad_count),
        "sweep_worst_quad_normal_dot": float(worst_quad_normal_dot),
        "sweep_local_fold_violation": bool(folded_quad_count > 0),
        "sweep_diagonal_optimization_skipped": bool(not optimize_diagonals),
    }


def _quad_triangulation_metrics(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fold flags, worst altitude aspects, and normal dots per quad."""

    coordinates = vertices[np.asarray(triangles, dtype=np.int64)]
    normals = np.cross(
        coordinates[:, :, 1] - coordinates[:, :, 0],
        coordinates[:, :, 2] - coordinates[:, :, 0],
    )
    normal_lengths = np.linalg.norm(normals, axis=2)
    unit_normals = np.divide(
        normals,
        normal_lengths[:, :, None],
        out=np.zeros_like(normals),
        where=normal_lengths[:, :, None] > 1e-30,
    )
    normal_dots = np.einsum("ij,ij->i", unit_normals[:, 0], unit_normals[:, 1])
    folded = (~np.isfinite(normal_dots)) | (normal_dots <= 0.0) | np.any(normal_lengths <= 1e-30, axis=1)
    edge_lengths = np.stack(
        (
            np.linalg.norm(coordinates[:, :, 1] - coordinates[:, :, 0], axis=2),
            np.linalg.norm(coordinates[:, :, 2] - coordinates[:, :, 1], axis=2),
            np.linalg.norm(coordinates[:, :, 0] - coordinates[:, :, 2], axis=2),
        ),
        axis=2,
    )
    maximum_edges = np.max(edge_lengths, axis=2)
    altitude_aspects = np.divide(
        maximum_edges**2,
        normal_lengths,
        out=np.full_like(maximum_edges, np.inf),
        where=normal_lengths > 1e-30,
    )
    return folded.astype(np.int8), np.max(altitude_aspects, axis=1), normal_dots


def _swept_mesh_quality_diagnostics(
    mesh: Mesh,
    config: CoilGenConfig,
) -> dict[str, float | int | bool]:
    triangles = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    minimum_edges = np.min(edge_lengths, axis=1)
    maximum_edges = np.max(edge_lengths, axis=1)
    edge_ratios = np.divide(
        maximum_edges,
        minimum_edges,
        out=np.full(len(triangles), np.inf),
        where=minimum_edges > 1e-30,
    )
    area_twice = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    altitude_ratios = np.divide(
        maximum_edges**2,
        area_twice,
        out=np.full(len(triangles), np.inf),
        where=area_twice > 1e-30,
    )
    edge_ratio = float(np.max(edge_ratios)) if len(edge_ratios) else 0.0
    altitude_ratio = float(np.max(altitude_ratios)) if len(altitude_ratios) else 0.0
    edge_limit = float(config.mesh_quality_max_triangle_edge_ratio)
    altitude_limit = float(config.mesh_quality_max_altitude_aspect_ratio)
    return {
        "sweep_triangle_edge_ratio_max": edge_ratio,
        "sweep_triangle_altitude_aspect_ratio_max": altitude_ratio,
        "sweep_triangle_edge_ratio_limit": edge_limit,
        "sweep_triangle_altitude_aspect_ratio_limit": altitude_limit,
        "sweep_triangle_quality_violation": bool(edge_ratio > edge_limit or altitude_ratio > altitude_limit),
    }


def _swept_mesh_self_intersection_diagnostics(
    mesh: Mesh,
    *,
    maximum_reported_intersections: int = 100,
) -> dict[str, float | int | bool | str]:
    """Check non-adjacent swept triangles using a BVH broad phase."""

    triangles = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
    if len(triangles) < 2:
        return {
            "sweep_mesh_collision_broad_phase": "triangle_bvh",
            "sweep_mesh_collision_candidate_pair_count": 0,
            "sweep_mesh_self_intersection_count": 0,
            "sweep_mesh_self_intersection_truncated": False,
            "sweep_mesh_self_intersection": False,
        }
    minima = np.min(triangles, axis=1)
    maxima = np.max(triangles, axis=1)
    centroids = 0.5 * (minima + maxima)
    scale = max(float(np.max(maxima) - np.min(minima)), 1.0)
    tolerance = 1e-12 * scale

    def build(indices: np.ndarray) -> _SegmentBvhNode:
        minimum = np.min(minima[indices], axis=0)
        maximum = np.max(maxima[indices], axis=0)
        if len(indices) <= 12:
            return _SegmentBvhNode(indices, minimum, maximum)
        spread = np.ptp(centroids[indices], axis=0)
        axis = int(np.argmax(spread))
        order = indices[np.argsort(centroids[indices, axis], kind="stable")]
        middle = len(order) // 2
        return _SegmentBvhNode(
            np.zeros(0, dtype=np.int64),
            minimum,
            maximum,
            build(order[:middle]),
            build(order[middle:]),
        )

    def boxes_overlap(first: _SegmentBvhNode, second: _SegmentBvhNode) -> bool:
        return bool(
            np.all(first.maximum + tolerance >= second.minimum) and np.all(second.maximum + tolerance >= first.minimum)
        )

    candidate_count = 0
    intersection_count = 0
    truncated = False
    faces = np.asarray(mesh.faces, dtype=np.int64)

    def test_pair(first_index: int, second_index: int) -> None:
        nonlocal candidate_count, intersection_count, truncated
        if truncated or np.intersect1d(faces[first_index], faces[second_index], assume_unique=True).size:
            return
        if np.any(maxima[first_index] + tolerance < minima[second_index]) or np.any(
            maxima[second_index] + tolerance < minima[first_index]
        ):
            return
        candidate_count += 1
        if _triangle_pair_intersects_robust(
            triangles[first_index],
            triangles[second_index],
            tolerance,
        ):
            intersection_count += 1
            truncated = intersection_count >= maximum_reported_intersections

    def collect(first: _SegmentBvhNode, second: _SegmentBvhNode) -> None:
        if truncated or not boxes_overlap(first, second):
            return
        if first is second:
            if first.leaf:
                for local_first in range(len(first.indices)):
                    for local_second in range(local_first + 1, len(first.indices)):
                        test_pair(int(first.indices[local_first]), int(first.indices[local_second]))
                return
            collect(first.left, first.left)
            collect(first.left, first.right)
            collect(first.right, first.right)
            return
        if first.leaf and second.leaf:
            for first_index in first.indices:
                for second_index in second.indices:
                    test_pair(int(first_index), int(second_index))
            return
        first_extent = float(np.dot(first.maximum - first.minimum, first.maximum - first.minimum))
        second_extent = float(np.dot(second.maximum - second.minimum, second.maximum - second.minimum))
        if not first.leaf and (second.leaf or first_extent >= second_extent):
            collect(first.left, second)
            collect(first.right, second)
        else:
            collect(first, second.left)
            collect(first, second.right)

    root = build(np.arange(len(triangles), dtype=np.int64))
    collect(root, root)
    return {
        "sweep_mesh_collision_broad_phase": "triangle_bvh",
        "sweep_mesh_collision_candidate_pair_count": int(candidate_count),
        "sweep_mesh_self_intersection_count": int(intersection_count),
        "sweep_mesh_self_intersection_truncated": bool(truncated),
        "sweep_mesh_self_intersection": bool(intersection_count > 0),
    }


def _triangle_pair_intersects_robust(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> bool:
    first_normal = np.cross(first[1] - first[0], first[2] - first[0])
    second_normal = np.cross(second[1] - second[0], second[2] - second[0])
    first_length = float(np.linalg.norm(first_normal))
    second_length = float(np.linalg.norm(second_normal))
    if first_length <= tolerance or second_length <= tolerance:
        return False
    unit_first = first_normal / first_length
    unit_second = second_normal / second_length
    parallel = float(np.linalg.norm(np.cross(unit_first, unit_second))) <= 1e-10
    coplanar = parallel and abs(float(np.dot(unit_first, second[0] - first[0]))) <= tolerance
    if coplanar:
        drop_axis = int(np.argmax(np.abs(unit_first)))
        return _coplanar_triangles_intersect(
            np.delete(first, drop_axis, axis=1),
            np.delete(second, drop_axis, axis=1),
            tolerance,
        )
    for triangle_a, triangle_b in ((first, second), (second, first)):
        for edge_index in range(3):
            if _segment_triangle_intersects(
                triangle_a[edge_index],
                triangle_a[(edge_index + 1) % 3],
                triangle_b,
                tolerance,
            ):
                return True
    return False


def _segment_triangle_intersects(
    start: np.ndarray,
    end: np.ndarray,
    triangle: np.ndarray,
    tolerance: float,
) -> bool:
    direction = end - start
    edge_first = triangle[1] - triangle[0]
    edge_second = triangle[2] - triangle[0]
    cross_direction = np.cross(direction, edge_second)
    determinant = float(np.dot(edge_first, cross_direction))
    if abs(determinant) <= tolerance:
        return False
    inverse = 1.0 / determinant
    offset = start - triangle[0]
    first_coordinate = inverse * float(np.dot(offset, cross_direction))
    if first_coordinate < -tolerance or first_coordinate > 1.0 + tolerance:
        return False
    cross_offset = np.cross(offset, edge_first)
    second_coordinate = inverse * float(np.dot(direction, cross_offset))
    if second_coordinate < -tolerance or first_coordinate + second_coordinate > 1.0 + tolerance:
        return False
    segment_coordinate = inverse * float(np.dot(edge_second, cross_offset))
    return bool(-tolerance <= segment_coordinate <= 1.0 + tolerance)


def _coplanar_triangles_intersect(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> bool:
    for triangle_a, triangle_b in ((first, second), (second, first)):
        for edge_a in range(3):
            for edge_b in range(3):
                if _segments_intersect_or_touch_2d(
                    triangle_a[edge_a],
                    triangle_a[(edge_a + 1) % 3],
                    triangle_b[edge_b],
                    triangle_b[(edge_b + 1) % 3],
                    tolerance,
                ):
                    return True
        if _point_in_triangle_2d(triangle_a[0], triangle_b, tolerance):
            return True
    return False


def _segments_intersect_or_touch_2d(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> bool:
    def orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        edge = end - start
        offset = point - start
        return float(edge[0] * offset[1] - edge[1] * offset[0])

    values = (
        orientation(first_start, first_end, second_start),
        orientation(first_start, first_end, second_end),
        orientation(second_start, second_end, first_start),
        orientation(second_start, second_end, first_end),
    )
    if values[0] * values[1] < -(tolerance**2) and values[2] * values[3] < -(tolerance**2):
        return True
    for value, point, start, end in (
        (values[0], second_start, first_start, first_end),
        (values[1], second_end, first_start, first_end),
        (values[2], first_start, second_start, second_end),
        (values[3], first_end, second_start, second_end),
    ):
        if (
            abs(value) <= tolerance
            and np.all(point >= np.minimum(start, end) - tolerance)
            and np.all(point <= np.maximum(start, end) + tolerance)
        ):
            return True
    return False


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, tolerance: float) -> bool:
    signs = []
    for edge_index in range(3):
        start = triangle[edge_index]
        end = triangle[(edge_index + 1) % 3]
        edge = end - start
        offset = point - start
        signs.append(float(edge[0] * offset[1] - edge[1] * offset[0]))
    return bool(all(value >= -tolerance for value in signs) or all(value <= tolerance for value in signs))


def _support_conformity_diagnostics(
    part: CoilPart,
    swept_mesh: Mesh,
    centerline: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, float | int | bool | str]:
    """Check the swept body against the intended side of the support surface."""

    mode = _normalised_text(config.conductor_mounting_mode)
    tolerance = max(
        float(config.sweep_manufacturing_tolerance),
        float(config.sweep_max_chord_error),
        1e-8,
    )
    analytic_surface = _analytic_cylinder_surface_constraint(part, config)
    if analytic_surface is not None:
        projected_centerline, _centerline_normals = analytic_surface.project(centerline)
        axial = (np.asarray(centerline, dtype=float) - analytic_surface.center) @ analytic_surface.axis
        domain_residuals = np.maximum(
            np.maximum(analytic_surface.axial_min - axial, axial - analytic_surface.axial_max),
            0.0,
        )
    else:
        projector = _SurfaceProjector(part.mesh)
        (
            projected_centerline,
            _centerline_normals,
            _centerline_face_normals,
            centerline_face_indices,
            centerline_barycentric,
        ) = projector.project_with_details(centerline)
        domain_residuals = _support_domain_residuals(
            projector,
            centerline,
            projected_centerline,
            centerline_face_indices,
            centerline_barycentric,
        )
    maximum_domain_residual = float(np.max(domain_residuals)) if len(domain_residuals) else 0.0
    diagnostics: dict[str, float | int | bool | str] = {
        "sweep_support_conformity_mode": mode,
        "sweep_support_conformity_skipped_due_route_precheck_failure": False,
        "sweep_support_conformity_tolerance_m": float(tolerance),
        "sweep_centerline_max_support_domain_residual_m": maximum_domain_residual,
        "sweep_centerline_support_domain_violation": bool(maximum_domain_residual > tolerance),
        "sweep_support_penetration_count": 0,
        "sweep_support_max_penetration_m": 0.0,
        "sweep_support_penetration_violation": False,
    }
    if mode == "centered" or swept_mesh.n_vertices == 0:
        return diagnostics

    support_projector = analytic_surface if analytic_surface is not None else projector
    projected_vertices, support_normals = support_projector.project(swept_mesh.vertices)
    signed_distances = np.einsum("ij,ij->i", swept_mesh.vertices - projected_vertices, support_normals)
    penetration = -signed_distances if mode == "surface_outer" else signed_distances
    violation = penetration > tolerance
    diagnostics.update(
        {
            "sweep_support_penetration_count": int(np.count_nonzero(violation)),
            "sweep_support_max_penetration_m": float(max(np.max(penetration), 0.0)),
            "sweep_support_penetration_violation": bool(np.any(violation)),
        }
    )
    return diagnostics


def _support_domain_residuals(
    projector: "_SurfaceProjector",
    points: np.ndarray,
    projected: np.ndarray,
    face_indices: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    """Measure only displacement across true support-boundary edges."""

    residuals = np.zeros(len(points), dtype=float)
    barycentric_tolerance = 1e-10
    for point_index, face_index in enumerate(face_indices):
        face = projector.faces[int(face_index)]
        displacement = points[point_index] - projected[point_index]
        for opposite_vertex in np.flatnonzero(barycentric[point_index] <= barycentric_tolerance):
            edge_local = [index for index in range(3) if index != int(opposite_vertex)]
            start_index = int(face[edge_local[0]])
            end_index = int(face[edge_local[1]])
            edge_key = (min(start_index, end_index), max(start_index, end_index))
            if edge_key not in projector.boundary_edges:
                continue
            edge = projector.vertices[end_index] - projector.vertices[start_index]
            edge_length = float(np.linalg.norm(edge))
            if edge_length <= 1e-30:
                continue
            conormal = np.cross(projector.face_normals[int(face_index)], edge / edge_length)
            conormal_length = float(np.linalg.norm(conormal))
            if conormal_length <= 1e-30:
                continue
            residuals[point_index] = max(
                residuals[point_index],
                abs(float(np.dot(displacement, conormal / conormal_length))),
            )
    return residuals


def _build_sweep_rings(
    part: CoilPart,
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    cross_section: np.ndarray,
    config: CoilGenConfig,
    *,
    closed: bool,
    mounting_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Build framed rings and remove only collinear, numerically short samples.

    Folded rings are a route/frame failure and are deliberately not repaired by
    deleting centerline samples. Deleting fold-adjacent rings changes the wire
    route and previously amplified the raised-connector projection failure.
    """

    points = np.asarray(points, dtype=float).copy()
    segment_geometry = [WirePathGeometry(value) for value in segment_geometry]
    section_count = len(cross_section)
    edge_tolerance = float(np.sqrt(_DEFAULT_MESH_AREA_TOLERANCE_M2))
    profile_edges = np.roll(cross_section, -1, axis=0) - cross_section
    minimum_profile_edge = float(np.min(np.linalg.norm(profile_edges, axis=1)))
    quality_safety_factor = 1.25
    quality_edge_tolerance = max(
        edge_tolerance,
        quality_safety_factor
        * minimum_profile_edge
        / max(float(config.mesh_quality_max_triangle_edge_ratio), 1.0),
        quality_safety_factor
        * minimum_profile_edge
        / max(float(config.mesh_quality_max_altitude_aspect_ratio), 1.0),
    )
    removed_count = 0
    quality_coarsening_removed_count = 0
    minimum_point_count = 3 if closed else 2
    max_iterations = 8
    maximum_quality_removals = max(int(np.ceil(0.02 * len(points))), 0)

    for _iteration in range(max_iterations + 1):
        tangents = _path_tangents(points, closed=closed)
        normals, binormals = _path_frames(
            part,
            points,
            tangents,
            segment_geometry,
            config,
            closed=closed,
        )
        vertices = _sweep_ring_vertices(points, normals, binormals, cross_section, mounting_offset)
        lateral_lengths = _sweep_lateral_edge_lengths(vertices, len(points), section_count, closed=closed)
        short_segment_indices = np.flatnonzero(np.any(lateral_lengths <= edge_tolerance, axis=1))
        quality_segment_indices = np.flatnonzero(np.any(lateral_lengths < quality_edge_tolerance, axis=1))
        (
            folded_quad_count,
            folded_segment_indices,
            poor_quad_count,
            poor_segment_indices,
        ) = _sweep_fold_and_quality_segments(
            vertices,
            len(points),
            section_count,
            float(config.mesh_quality_max_altitude_aspect_ratio),
            closed=closed,
        )
        if (
            (len(quality_segment_indices) == 0 and folded_quad_count == 0 and poor_quad_count == 0)
            or _iteration == max_iterations
        ):
            return (
                points,
                tangents,
                normals,
                binormals,
                vertices,
                _sweep_ring_repair_diagnostics(
                    lateral_lengths,
                    edge_tolerance,
                    removed_count,
                    quality_coarsening_removed_count,
                    quality_edge_tolerance,
                ),
            )

        removals = _safe_short_ring_point_removals(
            points,
            short_segment_indices,
            config,
            closed=closed,
            minimum_point_count=minimum_point_count,
        )
        remaining_quality_removals = maximum_quality_removals - quality_coarsening_removed_count
        quality_removals = _safe_fold_coarsening_removals(
            points,
            segment_geometry,
            np.unique(np.concatenate((folded_segment_indices, poor_segment_indices, quality_segment_indices))),
            config,
            closed=closed,
            minimum_point_count=minimum_point_count,
            excluded=removals,
            maximum_removals=remaining_quality_removals,
        )
        removals.update(quality_removals)
        if not removals:
            return (
                points,
                tangents,
                normals,
                binormals,
                vertices,
                _sweep_ring_repair_diagnostics(
                    lateral_lengths,
                    edge_tolerance,
                    removed_count,
                    quality_coarsening_removed_count,
                    quality_edge_tolerance,
                ),
            )
        keep = np.ones(len(points), dtype=bool)
        keep[list(removals)] = False
        next_points = points[keep]
        segment_geometry = _transfer_path_segment_geometry(
            points,
            segment_geometry,
            next_points,
            source_closed=closed,
            target_closed=closed,
        )
        trial_tangents = _path_tangents(next_points, closed=closed)
        trial_normals, trial_binormals = _path_frames(
            part,
            next_points,
            trial_tangents,
            segment_geometry,
            config,
            closed=closed,
        )
        trial_vertices = _sweep_ring_vertices(
            next_points,
            trial_normals,
            trial_binormals,
            cross_section,
            mounting_offset,
        )
        trial_lateral_lengths = _sweep_lateral_edge_lengths(
            trial_vertices,
            len(next_points),
            section_count,
            closed=closed,
        )
        trial_short_count = int(
            np.count_nonzero(~np.isfinite(trial_lateral_lengths) | (trial_lateral_lengths <= edge_tolerance))
        )
        trial_quality_count = int(
            np.count_nonzero(~np.isfinite(trial_lateral_lengths) | (trial_lateral_lengths < quality_edge_tolerance))
        )
        current_short_count = int(np.count_nonzero(~np.isfinite(lateral_lengths) | (lateral_lengths <= edge_tolerance)))
        current_quality_count = int(
            np.count_nonzero(~np.isfinite(lateral_lengths) | (lateral_lengths < quality_edge_tolerance))
        )
        (
            trial_folded_quad_count,
            _trial_folded_segments,
            trial_poor_quad_count,
            _trial_poor_segments,
        ) = _sweep_fold_and_quality_segments(
            trial_vertices,
            len(next_points),
            section_count,
            float(config.mesh_quality_max_altitude_aspect_ratio),
            closed=closed,
        )
        improves_quality = (
            trial_folded_quad_count <= folded_quad_count
            and trial_poor_quad_count <= poor_quad_count
            and trial_quality_count <= current_quality_count
            and trial_short_count <= current_short_count
            and (
                trial_folded_quad_count < folded_quad_count
                or trial_poor_quad_count < poor_quad_count
                or trial_quality_count < current_quality_count
                or trial_short_count < current_short_count
            )
        )
        if not improves_quality:
            return (
                points,
                tangents,
                normals,
                binormals,
                vertices,
                _sweep_ring_repair_diagnostics(
                    lateral_lengths,
                    edge_tolerance,
                    removed_count,
                    quality_coarsening_removed_count,
                    quality_edge_tolerance,
                ),
            )
        points = next_points
        removed_count += len(removals) - len(quality_removals)
        quality_coarsening_removed_count += len(quality_removals)

    raise RuntimeError("Unreachable sweep-ring repair state.")


def _sweep_ring_repair_diagnostics(
    lateral_lengths: np.ndarray,
    edge_tolerance: float,
    removed_count: int,
    quality_coarsening_removed_count: int,
    quality_edge_tolerance: float,
) -> dict[str, float | int | bool]:
    finite_lengths = lateral_lengths[np.isfinite(lateral_lengths)]
    return {
        "sweep_degenerate_ring_points_removed": int(removed_count),
        "sweep_fold_repair_points_removed": 0,
        "sweep_quality_coarsening_points_removed": int(quality_coarsening_removed_count),
        "sweep_ring_points_removed": int(removed_count + quality_coarsening_removed_count),
        "sweep_quality_coarsening_max_fraction": 0.02,
        "sweep_quality_lateral_edge_target_m": float(quality_edge_tolerance),
        "sweep_unrepaired_short_lateral_edge_count": int(
            np.count_nonzero(~np.isfinite(lateral_lengths) | (lateral_lengths <= edge_tolerance))
        ),
        "sweep_min_lateral_edge_length_m": float(np.min(finite_lengths)) if len(finite_lengths) else 0.0,
        "sweep_ring_edge_tolerance_m": float(edge_tolerance),
    }


def _sweep_fold_and_quality_segments(
    vertices: np.ndarray,
    ring_count: int,
    section_count: int,
    maximum_altitude_aspect: float,
    *,
    closed: bool,
) -> tuple[int, np.ndarray, int, np.ndarray]:
    faces, diagnostics = _triangulate_sweep_laterals(
        vertices,
        ring_count,
        section_count,
        closed=closed,
        optimize_diagonals=True,
    )
    triangle_pairs = np.asarray(faces, dtype=np.int64).reshape((-1, 2, 3))
    folded, aspects, _dots = _quad_triangulation_metrics(vertices, triangle_pairs)
    poor = (~np.isfinite(aspects)) | (aspects > max(float(maximum_altitude_aspect), 0.0))
    return (
        int(diagnostics["sweep_folded_quad_count"]),
        np.unique(np.flatnonzero(folded) // section_count),
        int(np.count_nonzero(poor)),
        np.unique(np.flatnonzero(poor) // section_count),
    )


def _safe_fold_coarsening_removals(
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    folded_segment_indices: np.ndarray,
    config: CoilGenConfig,
    *,
    closed: bool,
    minimum_point_count: int,
    excluded: set[int],
    maximum_removals: int,
) -> set[int]:
    """Propose bounded point removals that preserve the route within tolerance."""

    if maximum_removals <= 0 or len(folded_segment_indices) == 0:
        return set()
    tolerance = max(
        float(config.sweep_max_chord_error),
        float(config.sweep_manufacturing_tolerance),
        float(config.sweep_collinearity_tolerance),
        0.0,
    )
    if tolerance <= 0.0:
        return set()
    max_segment_length = max(float(config.sweep_max_segment_length), 0.0)
    scored: list[tuple[float, int]] = []
    for segment_index in folded_segment_indices:
        for candidate in (int(segment_index), (int(segment_index) + 1) % len(points)):
            if not closed and candidate in {0, len(points) - 1}:
                continue
            incoming_geometry = segment_geometry[(candidate - 1) % len(segment_geometry)]
            outgoing_geometry = segment_geometry[candidate % len(segment_geometry)]
            if incoming_geometry != outgoing_geometry and WirePathGeometry.SURFACE in {
                incoming_geometry,
                outgoing_geometry,
            }:
                continue
            previous = (candidate - 1) % len(points)
            following = (candidate + 1) % len(points)
            replacement = points[following] - points[previous]
            replacement_length = float(np.linalg.norm(replacement))
            if replacement_length <= 1e-15:
                continue
            if max_segment_length > 0.0 and replacement_length > max_segment_length:
                continue
            projection = float(
                np.clip(
                    np.dot(points[candidate] - points[previous], replacement) / replacement_length**2,
                    0.0,
                    1.0,
                )
            )
            deviation = float(np.linalg.norm(points[candidate] - (points[previous] + projection * replacement)))
            if deviation <= tolerance:
                scored.append((deviation, candidate))

    removals: set[int] = set()
    unavailable = set(excluded)
    for _deviation, candidate in sorted(scored):
        if candidate in unavailable or candidate in removals:
            continue
        neighbours = {(candidate - 1) % len(points), (candidate + 1) % len(points)}
        if neighbours & (unavailable | removals):
            continue
        if len(points) - len(unavailable) - len(removals) - 1 < minimum_point_count:
            break
        removals.add(candidate)
        if len(removals) >= maximum_removals:
            break
    return removals


def _sweep_ring_vertices(
    points: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    cross_section: np.ndarray,
    mounting_offset: float,
) -> np.ndarray:
    centers = points + mounting_offset * normals
    rings = (
        centers[:, None, :]
        + cross_section[None, :, 0, None] * binormals[:, None, :]
        + cross_section[None, :, 1, None] * normals[:, None, :]
    )
    return rings.reshape((-1, 3))


def _sweep_lateral_edge_lengths(
    vertices: np.ndarray,
    ring_count: int,
    section_count: int,
    *,
    closed: bool,
) -> np.ndarray:
    rings = np.asarray(vertices, dtype=float).reshape((ring_count, section_count, 3))
    if closed:
        following = np.roll(rings, -1, axis=0)
        return np.linalg.norm(following - rings, axis=2)
    return np.linalg.norm(rings[1:] - rings[:-1], axis=2)


def _safe_short_ring_point_removals(
    points: np.ndarray,
    short_segment_indices: np.ndarray,
    config: CoilGenConfig,
    *,
    closed: bool,
    minimum_point_count: int,
) -> set[int]:
    if len(points) <= minimum_point_count:
        return set()
    tolerance = max(float(config.sweep_collinearity_tolerance), 1e-12)
    angle_tolerance = max(float(config.sweep_tangent_angle_tolerance_deg), 0.0)
    max_segment_length = max(float(config.sweep_max_segment_length), 0.0)
    removals: set[int] = set()

    for segment_index in short_segment_indices:
        current = int(segment_index)
        following = (current + 1) % len(points)
        candidates = (following, current)
        scored: list[tuple[float, float, int]] = []
        for candidate in candidates:
            if not closed and candidate in {0, len(points) - 1}:
                continue
            previous = (candidate - 1) % len(points)
            next_index = (candidate + 1) % len(points)
            replacement = points[next_index] - points[previous]
            replacement_length = float(np.linalg.norm(replacement))
            if replacement_length <= 1e-15:
                continue
            if max_segment_length > 0.0 and replacement_length > max_segment_length:
                continue
            projection = float(np.dot(points[candidate] - points[previous], replacement) / replacement_length**2)
            if projection < 0.0 or projection > 1.0:
                continue
            deviation = float(np.linalg.norm(points[candidate] - (points[previous] + projection * replacement)))
            turn_angle = _turn_angle_deg(points[previous], points[candidate], points[next_index])
            if deviation <= tolerance and turn_angle <= angle_tolerance:
                scored.append((deviation, turn_angle, candidate))
        if not scored:
            continue
        candidate = min(scored)[2]
        if candidate in removals:
            continue
        if (candidate - 1) % len(points) in removals or (candidate + 1) % len(points) in removals:
            continue
        if len(points) - len(removals) - 1 >= minimum_point_count:
            removals.add(candidate)
    return removals


def _prepare_sweep_centerline(
    part: CoilPart,
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    cross_section: np.ndarray,
    config: CoilGenConfig,
    *,
    closed: bool,
) -> tuple[np.ndarray, list[WirePathGeometry], dict[str, float | int | bool | str]]:
    diagnostics: dict[str, float | int | bool | str] = {
        "sweep_input_point_count": int(len(points)),
        "sweep_path_simplification": _normalised_text(config.sweep_path_simplification),
        "sweep_points_removed": 0,
        "sweep_bend_mode": _normalised_text(config.sweep_bend_mode),
        "sweep_bend_points_inserted": 0,
        "sweep_min_turn_angle_deg": 0.0,
        "sweep_max_turn_angle_deg": 0.0,
        "sweep_min_effective_bend_radius_m": 0.0,
        "sweep_closed_path": bool(closed),
        "sweep_pre_bend_simplification_skipped": False,
    }
    if len(points) < 2:
        diagnostics["sweep_output_point_count"] = int(len(points))
        return points, segment_geometry, diagnostics

    turn_angles = _polyline_turn_angles_deg(points, closed=closed)
    if len(turn_angles):
        diagnostics["sweep_min_turn_angle_deg"] = float(np.min(turn_angles))
        diagnostics["sweep_max_turn_angle_deg"] = float(np.max(turn_angles))

    if _normalised_text(config.sweep_bend_mode) == "fillet":
        # The fillet resolver owns a scale-aware, provenance-preserving
        # analysis skeleton. Running the legacy ring simplifier first makes
        # trim feasibility depend on which near-corner sample survived.
        simplified = np.asarray(points, dtype=float)
        simplified_geometry = [WirePathGeometry(value) for value in segment_geometry]
        diagnostics["sweep_pre_bend_simplification_skipped"] = True
    else:
        simplified = _simplify_sweep_centerline(
            part,
            points,
            segment_geometry,
            config,
            closed=closed,
        )
        simplified_geometry = _transfer_path_segment_geometry(
            points,
            segment_geometry,
            simplified,
            source_closed=closed,
            target_closed=closed,
        )
    diagnostics["sweep_points_removed"] = int(len(points) - len(simplified))
    bent, bent_geometry, bend_diagnostics = _resolve_sweep_centerline_bends(
        part,
        simplified,
        simplified_geometry,
        cross_section,
        config,
        closed=closed,
    )
    diagnostics.update(bend_diagnostics)
    resampled, resampled_geometry, sampling_diagnostics = _surface_adaptive_resample(
        part,
        bent,
        bent_geometry,
        cross_section,
        config,
        closed=closed,
    )
    diagnostics.update(sampling_diagnostics)
    diagnostics["sweep_adaptive_sampling_skipped_due_route_precheck_failure"] = False
    resampled_radii = _polyline_circumradii(
        resampled,
        closed=closed,
        minimum_turn_angle_deg=max(float(config.sweep_tangent_angle_tolerance_deg), 1e-6),
    )
    diagnostics["sweep_min_resampled_bend_radius_m"] = float(np.min(resampled_radii)) if len(resampled_radii) else 0.0
    required_bend_radius = float(diagnostics.get("sweep_required_bend_radius_m", 0.0))
    if (
        _normalised_text(config.sweep_bend_mode) == "fillet"
        and required_bend_radius > 0.0
        and len(resampled_radii)
        and float(np.min(resampled_radii)) < 0.99 * required_bend_radius
    ):
        diagnostics["sweep_min_bend_radius_violation"] = True
    envelope_radius = _cross_section_envelope_radius(cross_section)
    minimum_resampled_radius = float(np.min(resampled_radii)) if len(resampled_radii) else np.inf
    curvature_envelope_ratio = (
        float(envelope_radius / minimum_resampled_radius)
        if np.isfinite(minimum_resampled_radius) and minimum_resampled_radius > 0.0
        else 0.0
    )
    diagnostics["sweep_max_curvature_envelope_ratio"] = curvature_envelope_ratio
    diagnostics["sweep_curvature_envelope_limit"] = float(config.sweep_max_curvature_envelope_ratio)
    diagnostics["sweep_curvature_envelope_violation"] = bool(
        curvature_envelope_ratio >= float(config.sweep_max_curvature_envelope_ratio)
    )
    diagnostics["sweep_output_point_count"] = int(len(resampled))
    return resampled, resampled_geometry, diagnostics


def _polyline_circumradii(
    points: np.ndarray,
    *,
    closed: bool,
    minimum_turn_angle_deg: float,
) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(0, dtype=float)
    radii: list[float] = []
    indices = range(len(points)) if closed else range(1, len(points) - 1)
    for index in indices:
        previous = points[(index - 1) % len(points)]
        current = points[index]
        following = points[(index + 1) % len(points)]
        if _turn_angle_deg(previous, current, following) <= minimum_turn_angle_deg:
            continue
        first = float(np.linalg.norm(current - previous))
        second = float(np.linalg.norm(following - current))
        third = float(np.linalg.norm(following - previous))
        area_twice = float(np.linalg.norm(np.cross(current - previous, following - previous)))
        if area_twice <= 1e-30:
            continue
        radii.append(first * second * third / (2.0 * area_twice))
    return np.asarray(radii, dtype=float)


def _surface_adaptive_resample(
    part: CoilPart,
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    cross_section: np.ndarray,
    config: CoilGenConfig,
    *,
    closed: bool,
) -> tuple[np.ndarray, list[WirePathGeometry], dict[str, float | int | bool]]:
    """Refine a typed route without changing offset or free-space geometry."""

    original_lengths = _centerline_segment_lengths(points, closed=closed)
    diagnostics: dict[str, float | int | bool] = {
        "sweep_adaptive_points_inserted": 0,
        "sweep_adaptive_max_input_segment_length_m": (
            float(np.max(original_lengths)) if len(original_lengths) else 0.0
        ),
        "sweep_adaptive_max_output_segment_length_m": (
            float(np.max(original_lengths)) if len(original_lengths) else 0.0
        ),
        "sweep_adaptive_max_chord_correction_m": 0.0,
        "sweep_adaptive_max_frame_rotation_deg": 0.0,
        "sweep_adaptive_depth_limit_count": 0,
        "sweep_adaptive_unresolved_count": 0,
        "sweep_surface_projection_domain_fallback_count": 0,
        "sweep_adaptive_surface_projection_backend": "triangular_mesh",
    }
    if len(points) < 2 or part.mesh.n_faces == 0:
        return points, segment_geometry, diagnostics

    expected_segment_count = len(points) if closed else len(points) - 1
    if len(segment_geometry) != expected_segment_count:
        raise ValueError(
            "segment_geometry must contain one entry per sweep-centerline segment "
            f"({len(segment_geometry)} != {expected_segment_count})"
        )
    segment_geometry = [WirePathGeometry(value) for value in segment_geometry]
    surface_segments = np.asarray(
        [value == WirePathGeometry.SURFACE for value in segment_geometry],
        dtype=bool,
    )
    diagnostics["sweep_adaptive_surface_segment_count"] = int(np.count_nonzero(surface_segments))
    diagnostics["sweep_adaptive_offset_surface_segment_count"] = int(
        sum(value == WirePathGeometry.OFFSET_SURFACE for value in segment_geometry)
    )
    diagnostics["sweep_adaptive_free_3d_segment_count"] = int(
        sum(value == WirePathGeometry.FREE_3D for value in segment_geometry)
    )
    diagnostics["sweep_adaptive_linear_segment_count"] = int(len(surface_segments) - np.count_nonzero(surface_segments))

    profile_edges = np.roll(cross_section, -1, axis=0) - cross_section
    minimum_profile_edge = float(np.min(np.linalg.norm(profile_edges, axis=1)))
    quality_segment_length = 0.8 * float(config.mesh_quality_max_triangle_edge_ratio) * minimum_profile_edge
    configured_segment_length = max(float(config.sweep_max_segment_length), 0.0)
    max_segment_length = (
        min(configured_segment_length, quality_segment_length)
        if configured_segment_length > 0.0
        else quality_segment_length
    )
    max_chord_error = max(float(config.sweep_max_chord_error), 0.0)
    max_frame_rotation = max(float(config.sweep_max_frame_rotation_deg), 0.0)
    minimum_refinement_length = max(0.1 * minimum_profile_edge, 1e-6)
    maximum_output_points = max(20 * len(points), 1000)
    if max_segment_length == 0.0 and max_chord_error == 0.0 and max_frame_rotation >= 180.0:
        return points, segment_geometry, diagnostics

    analytic_surface = _analytic_cylinder_surface_constraint(part, config)
    projector = None if analytic_surface is not None else _SurfaceProjector(part.mesh)
    diagnostics["sweep_adaptive_surface_projection_backend"] = (
        "analytic_cylinder" if analytic_surface is not None else "triangular_mesh"
    )
    support_projector = analytic_surface if analytic_surface is not None else projector
    if support_projector is None:
        return points, segment_geometry, diagnostics
    _projected, support_normals = support_projector.project(points)
    surface_tolerance = max(0.1 * max_chord_error, 1e-9)
    samples = [support_normals[index] for index in range(len(points))]
    output: list[np.ndarray] = [points[0].copy()]
    output_geometry: list[WirePathGeometry] = []
    maximum_depth = 14
    max_correction_seen = 0.0
    max_rotation_seen = 0.0
    depth_limit_count = 0
    unresolved_count = 0
    domain_fallback_count = 0

    def append_segment(
        start: np.ndarray,
        start_normal: np.ndarray,
        end: np.ndarray,
        end_normal: np.ndarray,
        depth: int,
        geometry: WirePathGeometry,
    ) -> None:
        nonlocal max_correction_seen
        nonlocal max_rotation_seen
        nonlocal depth_limit_count
        nonlocal unresolved_count
        nonlocal domain_fallback_count
        midpoint_linear = 0.5 * (start + end)
        follows_support = geometry == WirePathGeometry.SURFACE
        midpoint_normal = start_normal + end_normal
        midpoint_normal_length = float(np.linalg.norm(midpoint_normal))
        midpoint_normal = (
            midpoint_normal / midpoint_normal_length if midpoint_normal_length > 1e-15 else start_normal.copy()
        )
        midpoint_on_support_domain = True
        if follows_support:
            if analytic_surface is not None:
                analytic_midpoint, analytic_normal = analytic_surface.geodesic_midpoint(start, end)
                midpoint_projected = analytic_midpoint[None, :]
                projected_normal = analytic_normal[None, :]
                midpoint_on_support_domain = bool(
                    analytic_surface.within_domain(midpoint_projected, surface_tolerance)[0]
                )
            else:
                assert projector is not None
                midpoint_projected, projected_normal, midpoint_face_normal = projector.project_with_face_normals(
                    midpoint_linear[None, :]
                )
                midpoint_residual = midpoint_linear - midpoint_projected[0]
                midpoint_face_offset = float(np.dot(midpoint_residual, midpoint_face_normal[0]))
                midpoint_tangential_residual = midpoint_residual - midpoint_face_offset * midpoint_face_normal[0]
                midpoint_on_support_domain = bool(np.linalg.norm(midpoint_tangential_residual) <= surface_tolerance)
            midpoint_normal = projected_normal[0]
            if midpoint_on_support_domain:
                midpoint = midpoint_projected[0]
            else:
                # A closest-point projection at a support boundary is not a
                # valid continuation of the route. Preserve the authoritative
                # input chord and make the unresolved surface trace explicit.
                midpoint = midpoint_linear
                domain_fallback_count += 1
        else:
            midpoint = midpoint_linear
        chord_correction = float(np.linalg.norm(midpoint - midpoint_linear))
        frame_rotation = _vector_angle_deg(start_normal, end_normal)
        split_frame_rotation = max(
            _vector_angle_deg(start_normal, midpoint_normal),
            _vector_angle_deg(midpoint_normal, end_normal),
        )
        max_correction_seen = max(max_correction_seen, chord_correction)
        max_rotation_seen = max(max_rotation_seen, frame_rotation)
        frame_refinement_is_resolving = split_frame_rotation < 0.8 * frame_rotation
        segment_length = float(np.linalg.norm(end - start))
        length_needs_refinement = max_segment_length > 0.0 and segment_length > max_segment_length
        chord_needs_refinement = follows_support and max_chord_error > 0.0 and chord_correction > max_chord_error
        frame_needs_refinement = (
            follows_support
            and max_frame_rotation > 0.0
            and frame_rotation > max_frame_rotation
            and frame_refinement_is_resolving
        )
        needs_refinement = length_needs_refinement or chord_needs_refinement or frame_needs_refinement
        if (
            needs_refinement
            and not length_needs_refinement
            and (segment_length <= minimum_refinement_length or len(output) >= maximum_output_points)
        ):
            unresolved_count += 1
            output.append(end.copy())
            output_geometry.append(geometry)
            return
        if not needs_refinement:
            output.append(end.copy())
            output_geometry.append(geometry)
            return
        if depth >= maximum_depth:
            depth_limit_count += 1
            output.append(end.copy())
            output_geometry.append(geometry)
            return
        append_segment(start, start_normal, midpoint, midpoint_normal, depth + 1, geometry)
        append_segment(midpoint, midpoint_normal, end, end_normal, depth + 1, geometry)

    segment_count = len(points) if closed else len(points) - 1
    for segment_index in range(segment_count):
        following = (segment_index + 1) % len(points)
        append_segment(
            points[segment_index],
            samples[segment_index],
            points[following],
            samples[following],
            0,
            segment_geometry[segment_index],
        )
    if closed:
        output.pop()
    sampled_points = np.asarray(output, dtype=float)
    resampled = _dedupe_path(sampled_points)
    resampled_geometry = _transfer_path_segment_geometry(
        sampled_points,
        output_geometry,
        resampled,
        source_closed=closed,
        target_closed=closed,
    )
    output_lengths = _centerline_segment_lengths(resampled, closed=closed)
    diagnostics.update(
        {
            "sweep_effective_max_segment_length_m": float(max_segment_length),
            "sweep_adaptive_points_inserted": int(max(len(resampled) - len(points), 0)),
            "sweep_adaptive_max_output_segment_length_m": (
                float(np.max(output_lengths)) if len(output_lengths) else 0.0
            ),
            "sweep_adaptive_max_chord_correction_m": float(max_correction_seen),
            "sweep_adaptive_max_frame_rotation_deg": float(max_rotation_seen),
            "sweep_adaptive_depth_limit_count": int(depth_limit_count),
            "sweep_adaptive_unresolved_count": int(unresolved_count),
            "sweep_surface_projection_domain_fallback_count": int(domain_fallback_count),
        }
    )
    return resampled, resampled_geometry, diagnostics


def _simplify_sweep_centerline(
    part: CoilPart,
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    config: CoilGenConfig,
    *,
    closed: bool,
) -> np.ndarray:
    mode = _normalised_text(config.sweep_path_simplification)
    minimum_count = 3 if closed else 2
    if mode == "off" or len(points) <= minimum_count:
        return points

    normals = _nearest_surface_normals(part, points, config)
    tangent_tolerance = float(config.sweep_tangent_angle_tolerance_deg)
    frame_tolerance = float(config.sweep_frame_angle_tolerance_deg)
    collinearity_tolerance = float(config.sweep_collinearity_tolerance)
    if mode == "exact":
        tangent_tolerance = min(tangent_tolerance, 0.5)
        frame_tolerance = min(frame_tolerance, 0.5)
        collinearity_tolerance = min(collinearity_tolerance, 1e-8)

    keep = np.ones(len(points), dtype=bool)
    max_segment_length = float(config.sweep_max_segment_length)
    indices = range(len(points)) if closed else range(1, len(points) - 1)
    for ind in indices:
        previous_ind = (ind - 1) % len(points)
        following_ind = (ind + 1) % len(points)
        incoming_geometry = segment_geometry[previous_ind]
        outgoing_geometry = segment_geometry[ind]
        if incoming_geometry != outgoing_geometry:
            continue
        previous = points[previous_ind]
        current = points[ind]
        following = points[following_ind]
        combined = following - previous
        combined_length = float(np.linalg.norm(combined))
        if combined_length <= 1e-15:
            continue
        if max_segment_length > 0.0 and combined_length > max_segment_length:
            continue
        turn_angle = _turn_angle_deg(previous, current, following)
        if turn_angle > tangent_tolerance:
            continue
        distance_to_line = float(np.linalg.norm(np.cross(current - previous, combined)) / combined_length)
        if distance_to_line > collinearity_tolerance:
            continue
        normal_angle = _vector_angle_deg(normals[previous_ind], normals[following_ind])
        if normal_angle > frame_tolerance:
            continue
        keep[ind] = False
    simplified = points[keep]
    return simplified if len(simplified) >= minimum_count else points


@dataclass(frozen=True, slots=True)
class _SweepCornerCandidate:
    index: int
    turn_angle_rad: float
    discrete_radius_m: float
    window_radius_m: float
    tangent_distance_m: float
    incoming_geometry: WirePathGeometry
    outgoing_geometry: WirePathGeometry
    blend_geometry: WirePathGeometry
    detection_reasons: tuple[str, ...]
    required_previous_support_index: int
    required_following_support_index: int


@dataclass(frozen=True, slots=True)
class _SweepFilletBuildResult:
    points: np.ndarray
    segment_geometry: list[WirePathGeometry]
    inserted_count: int
    maximum_projection_m: float
    resolved_indices: frozenset[int]
    failed_indices: frozenset[int]
    surface_resolved_count: int
    surface_validation_failure_count: int
    maximum_radius_inflation: float


@dataclass(frozen=True, slots=True)
class _SweepCornerPlan:
    candidate: _SweepCornerCandidate
    previous_support_index: int
    following_support_index: int
    maximum_support_deviation_m: float


def _resolve_sweep_centerline_bends(
    part: CoilPart,
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    cross_section: np.ndarray,
    config: CoilGenConfig,
    *,
    closed: bool,
) -> tuple[np.ndarray, list[WirePathGeometry], dict[str, float | int | bool | str]]:
    """Resolve scale-relevant polyline corners before constructing sweep rings.

    Collinear samples are removed only from the private bend-analysis spine.
    This makes available tangent length independent of route tessellation. A
    requested minimum radius is never reduced: interacting bends that cannot
    satisfy their shared trim budget remain visualization miters and block
    manufacturing readiness.
    """

    source_points = np.asarray(points, dtype=float)
    source_geometry = [WirePathGeometry(value) for value in segment_geometry]
    analytic_surface = _analytic_cylinder_surface_constraint(part, config)
    envelope_radius = _cross_section_envelope_radius(cross_section)
    bend_mode = _normalised_text(config.sweep_bend_mode)
    configured_radius = max(float(config.sweep_min_bend_radius), 0.0)
    curvature_limit = float(config.sweep_max_curvature_envelope_ratio)
    automatic_radius = envelope_radius / curvature_limit if curvature_limit > 0.0 else 0.0
    required_radius = max(configured_radius, automatic_radius) if bend_mode == "fillet" else configured_radius
    diagnostics: dict[str, float | int | bool | str] = {
        "sweep_bend_points_inserted": 0,
        "sweep_min_effective_bend_radius_m": 0.0,
        "sweep_required_bend_radius_m": float(required_radius),
        "sweep_bend_radius_source": (
            "configured_and_profile_envelope"
            if configured_radius > 0.0 and bend_mode == "fillet"
            else "profile_envelope"
            if bend_mode == "fillet"
            else "configured"
        ),
        "sweep_bend_radius_reduced_count": 0,
        "sweep_bend_infeasible_count": 0,
        "sweep_bend_trim_conflict_count": 0,
        "sweep_min_bend_radius_violation": False,
        "sweep_corner_candidate_count": 0,
        "sweep_corner_angle_candidate_count": 0,
        "sweep_corner_radius_candidate_count": 0,
        "sweep_corner_window_candidate_count": 0,
        "sweep_corner_resolved_count": 0,
        "sweep_corner_geometry_transition_count": 0,
        "sweep_surface_corner_requires_intrinsic_reroute_count": 0,
        "sweep_corner_max_angle_deg": 0.0,
        "sweep_corner_min_discrete_radius_m": 0.0,
        "sweep_corner_min_window_radius_m": 0.0,
        "sweep_corner_max_discrete_curvature_envelope_ratio": 0.0,
        "sweep_undersized_sampled_curve_count": 0,
        "sweep_bend_analysis_points_removed": 0,
        "sweep_surface_bend_projection_max_m": 0.0,
        "sweep_surface_fillet_backend": ("analytic_cylinder" if analytic_surface is not None else "triangular_mesh"),
        "sweep_analytic_surface_retraction_point_count": 0,
        "sweep_analytic_surface_retraction_max_m": 0.0,
        "sweep_surface_corner_eligible_count": 0,
        "sweep_surface_corner_resolved_count": 0,
        "sweep_surface_corner_validation_failure_count": 0,
        "sweep_surface_fillet_radius_inflation_max": 1.0,
        "sweep_bend_multisegment_support_count": 0,
        "sweep_bend_support_max_span_segments": 1,
        "sweep_bend_support_max_deviation_m": 0.0,
        "sweep_bend_support_unavailable_count": 0,
        "sweep_bend_overlap_conflict_count": 0,
        "sweep_bend_window_envelope_count": 0,
        "sweep_bend_window_envelope_fallback_count": 0,
    }
    if analytic_surface is not None:
        surface_points = _surface_path_point_mask(source_geometry, len(source_points), closed=closed)
        if np.any(surface_points):
            projected, _normals = analytic_surface.project(source_points[surface_points])
            corrections = np.linalg.norm(projected - source_points[surface_points], axis=1)
            source_points = source_points.copy()
            source_points[surface_points] = projected
            diagnostics["sweep_analytic_surface_retraction_point_count"] = int(np.count_nonzero(surface_points))
            diagnostics["sweep_analytic_surface_retraction_max_m"] = (
                float(np.max(corrections)) if len(corrections) else 0.0
            )
    minimum_point_count = 3 if closed else 2
    if len(source_points) < max(minimum_point_count, 3):
        return source_points, source_geometry, diagnostics

    planar_support = _mesh_is_planar(
        part.mesh,
        max(float(config.sweep_max_chord_error), float(config.sweep_collinearity_tolerance), 1e-9),
    )
    analysis_points, analysis_geometry = _bend_analysis_skeleton(
        source_points,
        source_geometry,
        config,
        closed=closed,
        simplify_surface=bool(planar_support or analytic_surface is not None),
    )
    diagnostics["sweep_bend_analysis_points_removed"] = int(len(source_points) - len(analysis_points))
    analysis_closed = closed
    if closed:
        analysis_points, analysis_geometry = _open_closed_bend_analysis_path(
            part,
            analysis_points,
            analysis_geometry,
            analytic_surface=analytic_surface,
        )
        analysis_closed = False
    candidates = _sweep_corner_candidates(
        analysis_points,
        analysis_geometry,
        required_radius,
        config,
        closed=analysis_closed,
        analytic_surface=analytic_surface,
    )
    analysis_radii = _polyline_circumradii(
        analysis_points,
        closed=analysis_closed,
        minimum_turn_angle_deg=max(float(config.sweep_tangent_angle_tolerance_deg), 1e-6),
    )
    diagnostics["sweep_undersized_sampled_curve_count"] = int(
        np.count_nonzero(analysis_radii < 0.99 * required_radius) if required_radius > 0.0 else 0
    )
    diagnostics["sweep_corner_candidate_count"] = int(len(candidates))
    diagnostics["sweep_corner_angle_candidate_count"] = int(
        sum("angle" in candidate.detection_reasons for candidate in candidates)
    )
    diagnostics["sweep_corner_radius_candidate_count"] = int(
        sum("minimum_radius" in candidate.detection_reasons for candidate in candidates)
    )
    diagnostics["sweep_corner_window_candidate_count"] = int(
        sum("window_radius" in candidate.detection_reasons for candidate in candidates)
    )
    diagnostics["sweep_corner_geometry_transition_count"] = int(
        sum(candidate.incoming_geometry != candidate.outgoing_geometry for candidate in candidates)
    )
    if candidates:
        diagnostics["sweep_corner_max_angle_deg"] = float(
            max(np.degrees(candidate.turn_angle_rad) for candidate in candidates)
        )
        finite_radii = [
            candidate.discrete_radius_m
            for candidate in candidates
            if np.isfinite(candidate.discrete_radius_m) and candidate.discrete_radius_m > 0.0
        ]
        if finite_radii:
            minimum_radius = float(min(finite_radii))
            diagnostics["sweep_corner_min_discrete_radius_m"] = minimum_radius
            diagnostics["sweep_corner_max_discrete_curvature_envelope_ratio"] = float(envelope_radius / minimum_radius)
        finite_window_radii = [
            candidate.window_radius_m
            for candidate in candidates
            if np.isfinite(candidate.window_radius_m) and candidate.window_radius_m > 0.0
        ]
        if finite_window_radii:
            diagnostics["sweep_corner_min_window_radius_m"] = float(min(finite_window_radii))

    if not candidates or required_radius <= 0.0:
        return source_points, source_geometry, diagnostics
    if bend_mode != "fillet":
        diagnostics["sweep_bend_infeasible_count"] = int(len(candidates))
        diagnostics["sweep_min_bend_radius_violation"] = True
        return source_points, source_geometry, diagnostics

    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.blend_geometry != WirePathGeometry.SURFACE or planar_support or analytic_surface is not None
    ]
    unsupported_surface_indices = {
        candidate.index
        for candidate in candidates
        if candidate.blend_geometry == WirePathGeometry.SURFACE and not planar_support and analytic_surface is None
    }
    surface_eligible_count = sum(
        candidate.blend_geometry == WirePathGeometry.SURFACE for candidate in eligible_candidates
    )
    diagnostics["sweep_surface_corner_eligible_count"] = int(surface_eligible_count)
    trim_candidates: list[_SweepCornerCandidate] = []
    for candidate in eligible_candidates:
        if analytic_surface is not None and candidate.blend_geometry == WirePathGeometry.SURFACE:
            candidate = _candidate_with_analytic_surface_tangents(
                analysis_points,
                candidate,
                required_radius,
                analytic_surface,
            )
        trim_candidates.append(candidate)
    corner_plans, conflicts, support_diagnostics = _plan_open_sweep_corner_intervals(
        analysis_points,
        analysis_geometry,
        trim_candidates,
        required_radius,
        config,
        analytic_surface=analytic_surface,
    )
    diagnostics.update(support_diagnostics)
    trim_candidates = [
        corner_plans[candidate.index].candidate if candidate.index in corner_plans else candidate
        for candidate in trim_candidates
    ]
    intrinsically_infeasible = {
        candidate.index
        for candidate in trim_candidates
        if candidate.turn_angle_rad >= np.pi - 1e-6 or candidate.tangent_distance_m <= 1e-15
    }
    infeasible_indices = conflicts | intrinsically_infeasible | unsupported_surface_indices
    resolved = {
        candidate.index: candidate
        for candidate in trim_candidates
        if candidate.index not in infeasible_indices and candidate.index in corner_plans
    }
    if resolved:
        build = _build_radius_controlled_fillet_path(
            part,
            analysis_points,
            analysis_geometry,
            resolved,
            required_radius,
            config,
            closed=False,
            corner_plans={index: corner_plans[index] for index in resolved},
        )
    else:
        build = _SweepFilletBuildResult(
            points=source_points,
            segment_geometry=source_geometry,
            inserted_count=0,
            maximum_projection_m=0.0,
            resolved_indices=frozenset(),
            failed_indices=frozenset(),
            surface_resolved_count=0,
            surface_validation_failure_count=0,
            maximum_radius_inflation=1.0,
        )
    bent = build.points
    bent_geometry = build.segment_geometry
    if closed and len(bent) > 1 and np.linalg.norm(bent[-1] - bent[0]) <= 1e-10:
        bent = bent[:-1]
    validation_failures = set(build.failed_indices)
    final_infeasible_indices = infeasible_indices | validation_failures
    expected_geometry_count = len(bent) if closed else max(len(bent) - 1, 0)
    if len(bent_geometry) != expected_geometry_count:
        raise RuntimeError(
            "Radius-controlled bend resolver produced inconsistent segment provenance "
            f"({len(bent_geometry)} != {expected_geometry_count})."
        )

    diagnostics["sweep_bend_points_inserted"] = int(build.inserted_count)
    diagnostics["sweep_min_effective_bend_radius_m"] = float(required_radius) if build.resolved_indices else 0.0
    diagnostics["sweep_corner_resolved_count"] = int(len(build.resolved_indices))
    diagnostics["sweep_bend_trim_conflict_count"] = int(len(conflicts))
    diagnostics["sweep_bend_infeasible_count"] = int(len(final_infeasible_indices))
    diagnostics["sweep_min_bend_radius_violation"] = bool(final_infeasible_indices)
    diagnostics["sweep_surface_bend_projection_max_m"] = float(build.maximum_projection_m)
    diagnostics["sweep_surface_corner_resolved_count"] = int(build.surface_resolved_count)
    diagnostics["sweep_surface_corner_validation_failure_count"] = int(build.surface_validation_failure_count)
    diagnostics["sweep_surface_corner_requires_intrinsic_reroute_count"] = int(
        len(unsupported_surface_indices) + build.surface_validation_failure_count
    )
    diagnostics["sweep_surface_fillet_radius_inflation_max"] = float(build.maximum_radius_inflation)
    return bent, bent_geometry, diagnostics


def _candidate_with_analytic_surface_tangents(
    points: np.ndarray,
    candidate: _SweepCornerCandidate,
    required_radius: float,
    surface: _AnalyticCylinderSurface,
) -> _SweepCornerCandidate:
    previous = points[(candidate.index - 1) % len(points)]
    corner = points[candidate.index]
    following = points[(candidate.index + 1) % len(points)]
    local = surface.log_map(corner, np.vstack((previous, following)))
    incoming = -local[0]
    outgoing = local[1]
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if incoming_length <= 1e-15 or outgoing_length <= 1e-15:
        return replace(candidate, tangent_distance_m=0.0)
    turn_angle = float(
        np.arccos(
            np.clip(
                np.dot(incoming / incoming_length, outgoing / outgoing_length),
                -1.0,
                1.0,
            )
        )
    )
    return replace(
        candidate,
        turn_angle_rad=turn_angle,
        tangent_distance_m=1.05 * required_radius * np.tan(turn_angle / 2.0),
    )


def _bend_analysis_skeleton(
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    config: CoilGenConfig,
    *,
    closed: bool,
    simplify_surface: bool = False,
) -> tuple[np.ndarray, list[WirePathGeometry]]:
    """Build a scale-aware polyline for bend decisions.

    Ramer-Douglas-Peucker reduction is bounded by the configured support-chord
    error and never crosses a geometry-contract transition. The downstream
    adaptive sampler restores longitudinal resolution after bend construction.
    """

    if len(points) < 3:
        return np.asarray(points, dtype=float), [WirePathGeometry(value) for value in segment_geometry]
    tolerance = max(
        float(config.sweep_max_chord_error),
        float(config.sweep_collinearity_tolerance),
        0.0,
    )
    if tolerance <= 0.0:
        return np.asarray(points, dtype=float), [WirePathGeometry(value) for value in segment_geometry]
    geometry = [WirePathGeometry(value) for value in segment_geometry]
    mandatory = {0}
    mandatory.update(
        index
        for index in range(1, len(points) if closed else len(points) - 1)
        if geometry[index - 1] != geometry[index % len(geometry)]
    )
    keep: set[int] = set(mandatory)
    if closed:
        if len(mandatory) == 1:
            mandatory.add(len(points) // 2)
            keep.update(mandatory)
        ordered_mandatory = sorted(mandatory)
        for position, start in enumerate(ordered_mandatory):
            stop = ordered_mandatory[(position + 1) % len(ordered_mandatory)]
            sequence = (
                list(range(start, stop + 1)) if stop > start else [*range(start, len(points)), *range(0, stop + 1)]
            )
            if geometry[start % len(geometry)] == WirePathGeometry.SURFACE and not simplify_surface:
                keep.update(sequence)
                continue
            local_points = np.asarray(points, dtype=float)[sequence]
            local_keep = _rdp_path_indices(local_points, 0, len(local_points) - 1, tolerance)
            keep.update(sequence[index] for index in local_keep)
        if len(keep) < 3:
            keep.update({len(points) // 3, 2 * len(points) // 3})
    else:
        mandatory.add(len(points) - 1)
        keep.add(len(points) - 1)
        ordered_mandatory = sorted(mandatory)
        for start, stop in zip(ordered_mandatory[:-1], ordered_mandatory[1:]):
            if geometry[start] == WirePathGeometry.SURFACE and not simplify_surface:
                keep.update(range(start, stop + 1))
                continue
            keep.update(_rdp_path_indices(points, start, stop, tolerance))
    indices = np.asarray(sorted(keep), dtype=np.int64)
    skeleton = np.asarray(points, dtype=float)[indices]
    skeleton_geometry = _transfer_path_segment_geometry(
        np.asarray(points, dtype=float),
        geometry,
        skeleton,
        source_closed=closed,
        target_closed=closed,
    )
    return skeleton, skeleton_geometry


def _open_closed_bend_analysis_path(
    part: CoilPart,
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    *,
    analytic_surface: _AnalyticCylinderSurface | None,
) -> tuple[np.ndarray, list[WirePathGeometry]]:
    """Split a closed analysis path inside its longest segment.

    The duplicated seam converts cyclic interval planning into the same
    arclength problem as an open route without placing the seam on a corner.
    """

    points = np.asarray(points, dtype=float)
    geometry = [WirePathGeometry(value) for value in segment_geometry]
    if len(points) < 3 or len(geometry) != len(points):
        raise ValueError("Closed bend analysis requires one segment type per point.")
    lengths = np.zeros(len(points), dtype=float)
    for index, value in enumerate(geometry):
        following = (index + 1) % len(points)
        if value == WirePathGeometry.SURFACE and analytic_surface is not None:
            local = analytic_surface.log_map(points[index], points[following][None, :])
            lengths[index] = float(np.linalg.norm(local[0]))
        else:
            lengths[index] = float(np.linalg.norm(points[following] - points[index]))
    seam_segment = int(np.argmax(lengths))
    following = (seam_segment + 1) % len(points)
    if geometry[seam_segment] == WirePathGeometry.SURFACE:
        if analytic_surface is not None:
            midpoint, _normal = analytic_surface.geodesic_midpoint(
                points[seam_segment],
                points[following],
            )
        else:
            linear_midpoint = 0.5 * (points[seam_segment] + points[following])
            midpoint = _SurfaceProjector(part.mesh).project(linear_midpoint[None, :])[0][0]
    else:
        midpoint = 0.5 * (points[seam_segment] + points[following])

    order = [(following + offset) % len(points) for offset in range(len(points))]
    opened_points = np.vstack((midpoint, points[order], midpoint))
    opened_geometry = [
        geometry[seam_segment],
        *(geometry[index] for index in order[:-1]),
        geometry[seam_segment],
    ]
    return opened_points, opened_geometry


def _mesh_is_planar(mesh: Mesh, tolerance: float) -> bool:
    if mesh.n_vertices < 3 or mesh.n_faces == 0:
        return False
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for face in faces:
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length <= 1e-15:
            continue
        normal /= length
        distances = np.abs((vertices - a) @ normal)
        return bool(float(np.max(distances)) <= max(float(tolerance), 1e-12))
    return False


def _rdp_path_indices(
    points: np.ndarray,
    start: int,
    stop: int,
    tolerance: float,
) -> set[int]:
    keep = {int(start), int(stop)}
    stack: list[tuple[int, int]] = [(int(start), int(stop))]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        interior = np.asarray(points[first + 1 : last], dtype=float)
        distances = _point_segment_distances(interior, points[first], points[last])
        if len(distances) == 0:
            continue
        local_index = int(np.argmax(distances))
        if float(distances[local_index]) <= tolerance:
            continue
        split = first + 1 + local_index
        keep.add(split)
        stack.append((first, split))
        stack.append((split, last))
    return keep


def _point_segment_distances(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-30:
        return np.linalg.norm(points - start, axis=1)
    parameters = np.clip((points - start) @ segment / length_sq, 0.0, 1.0)
    closest = np.asarray(start, dtype=float) + parameters[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def _sweep_corner_candidates(
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    required_radius: float,
    config: CoilGenConfig,
    *,
    closed: bool,
    analytic_surface: _AnalyticCylinderSurface | None = None,
) -> list[_SweepCornerCandidate]:
    """Return every scale-relevant bend represented by the polyline.

    A local tangent jump remains useful for exact CAD-style vertices, but it is
    not sampling invariant: the same physical bend may arrive as one large
    angle or several moderate angles. Minimum-radius checks therefore promote
    every locally undersized sample, while a fixed-arclength tangent window
    catches curvature distributed over several samples. Window detections are
    reduced to one representative per periodic event so dense sampling does
    not create an arbitrary number of fillet requests.
    """

    indices = list(range(len(points)) if closed else range(1, len(points) - 1))
    detection_angle = max(
        float(config.sweep_corner_detection_angle_deg),
        float(config.sweep_tangent_angle_tolerance_deg),
    )
    tangent_tolerance = max(float(config.sweep_tangent_angle_tolerance_deg), 0.0)
    turn_angles = {
        index: _turn_angle_deg(
            points[(index - 1) % len(points)],
            points[index],
            points[(index + 1) % len(points)],
        )
        for index in indices
    }
    if analytic_surface is not None:
        for index in indices:
            incoming_geometry = segment_geometry[(index - 1) % len(segment_geometry)]
            outgoing_geometry = segment_geometry[index % len(segment_geometry)]
            if incoming_geometry == WirePathGeometry.SURFACE and outgoing_geometry == WirePathGeometry.SURFACE:
                turn_angles[index] = _analytic_surface_turn_angle_deg(
                    points[(index - 1) % len(points)],
                    points[index],
                    points[(index + 1) % len(points)],
                    analytic_surface,
                )
    discrete_radii = {
        index: _corner_circumradius(
            points[(index - 1) % len(points)],
            points[index],
            points[(index + 1) % len(points)],
        )
        for index in indices
    }
    window_radii = _windowed_polyline_bend_radii(
        points,
        required_radius,
        closed=closed,
    )
    reasons_by_index: dict[int, set[str]] = {}
    window_support_bounds: dict[int, tuple[int, int]] = {}
    radius_threshold = 0.99 * required_radius

    for index in indices:
        turn_angle_deg = turn_angles[index]
        if turn_angle_deg <= tangent_tolerance:
            continue
        previous_index = (index - 1) % len(points)
        following_index = (index + 1) % len(points)
        neighbour_angle = max(
            (turn_angles[neighbour] for neighbour in (previous_index, following_index) if neighbour in turn_angles),
            default=0.0,
        )
        strong_tangent_jump = turn_angle_deg >= 2.0 * detection_angle
        isolated_angle_spike = bool(
            turn_angle_deg >= detection_angle and turn_angle_deg >= 2.0 * max(neighbour_angle, tangent_tolerance)
        )
        if strong_tangent_jump or isolated_angle_spike:
            reasons_by_index.setdefault(index, set()).add("angle")
        if required_radius > 0.0 and np.isfinite(discrete_radii[index]) and discrete_radii[index] < radius_threshold:
            reasons_by_index.setdefault(index, set()).add("minimum_radius")

    if required_radius > 0.0 and len(window_radii) == len(points):
        window_mask = np.zeros(len(points), dtype=bool)
        for index in indices:
            window_mask[index] = bool(
                turn_angles[index] > tangent_tolerance
                and np.isfinite(window_radii[index])
                and window_radii[index] < radius_threshold
            )
        for component in _periodic_true_components(
            window_mask,
            closed=closed,
            link_lengths=_centerline_segment_lengths(points, closed=closed),
            maximum_link_length=required_radius,
        ):
            existing = [index for index in component if index in reasons_by_index]
            pool = existing if existing else component
            representative = min(
                pool,
                key=lambda index: (
                    float(window_radii[index]),
                    -float(turn_angles.get(index, 0.0)),
                    int(index),
                ),
            )
            reasons_by_index.setdefault(representative, set()).add("window_radius")
            if not closed:
                previous_support = max(int(component[0]) - 1, 0)
                following_support = min(int(component[-1]) + 1, len(points) - 1)
                margin_start = max(previous_support - 1, 0)
                margin_stop = min(following_support + 1, len(segment_geometry))
                envelope_geometry = {
                    WirePathGeometry(segment_geometry[segment_index])
                    for segment_index in range(margin_start, margin_stop)
                }
                if envelope_geometry == {WirePathGeometry.SURFACE}:
                    window_support_bounds[representative] = (
                        previous_support,
                        following_support,
                    )

    candidates: list[_SweepCornerCandidate] = []
    for index in sorted(reasons_by_index):
        previous = points[(index - 1) % len(points)]
        corner = points[index]
        following = points[(index + 1) % len(points)]
        incoming = corner - previous
        outgoing = following - corner
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length <= 1e-15 or outgoing_length <= 1e-15:
            continue
        incoming_geometry = segment_geometry[(index - 1) % len(segment_geometry)]
        outgoing_geometry = segment_geometry[index % len(segment_geometry)]
        if (
            analytic_surface is not None
            and incoming_geometry == WirePathGeometry.SURFACE
            and outgoing_geometry == WirePathGeometry.SURFACE
        ):
            turn_angle = np.deg2rad(turn_angles[index])
        else:
            incoming_unit = incoming / incoming_length
            outgoing_unit = outgoing / outgoing_length
            turn_angle = float(np.arccos(np.clip(np.dot(incoming_unit, outgoing_unit), -1.0, 1.0)))
        if float(np.degrees(turn_angle)) <= tangent_tolerance:
            continue
        tangent_distance = (
            required_radius * np.tan(turn_angle / 2.0)
            if required_radius > 0.0 and turn_angle < np.pi - 1e-6
            else np.inf
        )
        candidates.append(
            _SweepCornerCandidate(
                index=int(index),
                turn_angle_rad=turn_angle,
                discrete_radius_m=float(discrete_radii[index]),
                window_radius_m=float(window_radii[index]),
                tangent_distance_m=float(tangent_distance),
                incoming_geometry=incoming_geometry,
                outgoing_geometry=outgoing_geometry,
                blend_geometry=_blend_segment_geometry(incoming_geometry, outgoing_geometry),
                detection_reasons=tuple(sorted(reasons_by_index[index])),
                required_previous_support_index=window_support_bounds.get(
                    index,
                    (max(index - 1, 0), min(index + 1, len(points) - 1)),
                )[0],
                required_following_support_index=window_support_bounds.get(
                    index,
                    (max(index - 1, 0), min(index + 1, len(points) - 1)),
                )[1],
            )
        )
    return candidates


def _analytic_surface_turn_angle_deg(
    previous: np.ndarray,
    corner: np.ndarray,
    following: np.ndarray,
    surface: _AnalyticCylinderSurface,
) -> float:
    local = surface.log_map(
        np.asarray(corner, dtype=float),
        np.vstack((previous, following)),
    )
    incoming = -local[0]
    outgoing = local[1]
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if incoming_length <= 1e-15 or outgoing_length <= 1e-15:
        return 0.0
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(incoming, outgoing) / (incoming_length * outgoing_length),
                    -1.0,
                    1.0,
                )
            )
        )
    )


def _windowed_polyline_bend_radii(
    points: np.ndarray,
    required_radius: float,
    *,
    closed: bool,
) -> np.ndarray:
    """Estimate bend radius from tangent change over physical arclength.

    For a circular arc, the directions of the two half-window chords differ by
    ``window / radius``. The estimate is consequently independent of how many
    vertices sample that arc. Two window sizes cover concentrated and slightly
    distributed bends without treating ordinary support curvature as a corner.
    """

    points = np.asarray(points, dtype=float)
    radii = np.full(len(points), np.inf, dtype=float)
    required_radius = max(float(required_radius), 0.0)
    if len(points) < 3 or required_radius <= 0.0:
        return radii
    segment_lengths = _centerline_segment_lengths(points, closed=closed)
    if len(segment_lengths) == 0:
        return radii
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-15:
        return radii
    maximum_window = 0.25 * total_length if closed else 0.5 * total_length
    windows = sorted(
        {
            min(scale * required_radius, maximum_window)
            for scale in (0.25, 0.5, 1.0)
            if min(scale * required_radius, maximum_window) > 1e-15
        }
    )
    indices = range(len(points)) if closed else range(1, len(points) - 1)
    for index in indices:
        center_distance = float(cumulative[index])
        for window in windows:
            if not closed and (center_distance < window or total_length - center_distance < window):
                continue
            previous = _point_at_polyline_arclength(
                points,
                cumulative,
                center_distance - window,
                closed=closed,
            )
            following = _point_at_polyline_arclength(
                points,
                cumulative,
                center_distance + window,
                closed=closed,
            )
            incoming = points[index] - previous
            outgoing = following - points[index]
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length <= 1e-15 or outgoing_length <= 1e-15:
                continue
            angle = float(
                np.arccos(
                    np.clip(
                        np.dot(incoming, outgoing) / (incoming_length * outgoing_length),
                        -1.0,
                        1.0,
                    )
                )
            )
            if angle <= 1e-12:
                continue
            radii[index] = min(radii[index], float(window / angle))
    return radii


def _point_at_polyline_arclength(
    points: np.ndarray,
    cumulative: np.ndarray,
    distance: float,
    *,
    closed: bool,
) -> np.ndarray:
    total_length = float(cumulative[-1])
    if closed:
        distance = float(distance % total_length)
        segment_count = len(points)
    else:
        distance = float(np.clip(distance, 0.0, total_length))
        if distance >= total_length:
            return np.asarray(points[-1], dtype=float)
        segment_count = len(points) - 1
    segment_index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    segment_index = min(max(segment_index, 0), segment_count - 1)
    segment_length = float(cumulative[segment_index + 1] - cumulative[segment_index])
    if segment_length <= 1e-15:
        return np.asarray(points[segment_index % len(points)], dtype=float)
    fraction = (distance - float(cumulative[segment_index])) / segment_length
    start = np.asarray(points[segment_index % len(points)], dtype=float)
    end = np.asarray(points[(segment_index + 1) % len(points)], dtype=float)
    return start + float(fraction) * (end - start)


def _periodic_true_components(
    mask: np.ndarray,
    *,
    closed: bool,
    link_lengths: np.ndarray | None = None,
    maximum_link_length: float = np.inf,
) -> list[list[int]]:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int).tolist()
    if not indices:
        return []
    lengths = None if link_lengths is None else np.asarray(link_lengths, dtype=float)

    def connected(previous: int, following: int) -> bool:
        if following != previous + 1:
            return False
        return bool(
            lengths is None or previous >= len(lengths) or float(lengths[previous]) <= float(maximum_link_length)
        )

    components: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        if connected(components[-1][-1], index):
            components[-1].append(index)
        else:
            components.append([index])
    if (
        closed
        and len(components) > 1
        and components[0][0] == 0
        and components[-1][-1] == len(mask) - 1
        and (lengths is None or len(lengths) < len(mask) or float(lengths[-1]) <= float(maximum_link_length))
    ):
        components[0] = [*components[-1], *components[0]]
        components.pop()
    return components


def _corner_circumradius(previous: np.ndarray, current: np.ndarray, following: np.ndarray) -> float:
    first = float(np.linalg.norm(current - previous))
    second = float(np.linalg.norm(following - current))
    third = float(np.linalg.norm(following - previous))
    area_twice = float(np.linalg.norm(np.cross(current - previous, following - previous)))
    if min(first, second, third) <= 1e-15 or area_twice <= 1e-30:
        return np.inf
    return float(first * second * third / (2.0 * area_twice))


def _blend_segment_geometry(
    incoming: WirePathGeometry,
    outgoing: WirePathGeometry,
) -> WirePathGeometry:
    values = {WirePathGeometry(incoming), WirePathGeometry(outgoing)}
    if WirePathGeometry.FREE_3D in values:
        return WirePathGeometry.FREE_3D
    if WirePathGeometry.OFFSET_SURFACE in values:
        return WirePathGeometry.OFFSET_SURFACE
    return WirePathGeometry.SURFACE


def _plan_open_sweep_corner_intervals(
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    candidates: list[_SweepCornerCandidate],
    required_radius: float,
    config: CoilGenConfig,
    *,
    analytic_surface: _AnalyticCylinderSurface | None,
) -> tuple[
    dict[int, _SweepCornerPlan],
    set[int],
    dict[str, float | int],
]:
    """Plan fillet support over physical path intervals, not single mesh segments."""

    candidate_indices = {candidate.index for candidate in candidates}
    maximum_deviation = max(
        2.0 * float(config.sweep_max_chord_error),
        0.1 * required_radius,
        10.0 * float(config.sweep_collinearity_tolerance),
        1e-9,
    )
    minimum_remainder = max(
        3.0 * float(config.sweep_max_chord_error),
        0.06 * required_radius,
        10.0 * float(config.sweep_collinearity_tolerance),
        1e-9,
    )
    plans: dict[int, _SweepCornerPlan] = {}
    unavailable: set[int] = set()
    window_envelope_fallbacks: set[int] = set()

    for source_candidate in candidates:
        requires_window_envelope = bool(
            source_candidate.required_previous_support_index < source_candidate.index - 1
            or source_candidate.required_following_support_index > source_candidate.index + 1
        )
        feasible: list[
            tuple[
                tuple[int, float, int, int],
                _SweepCornerCandidate,
                tuple[int, float, int, float],
                tuple[int, float, int, float],
            ]
        ] = []
        for enforce_window_extent in ((True, False) if requires_window_envelope else (True,)):
            previous_options = _open_corner_support_options(
                points,
                segment_geometry,
                source_candidate,
                side=-1,
                candidate_indices=candidate_indices,
                maximum_deviation=maximum_deviation,
                analytic_surface=analytic_surface,
                enforce_window_extent=enforce_window_extent,
            )
            following_options = _open_corner_support_options(
                points,
                segment_geometry,
                source_candidate,
                side=1,
                candidate_indices=candidate_indices,
                maximum_deviation=maximum_deviation,
                analytic_surface=analytic_surface,
                enforce_window_extent=enforce_window_extent,
            )
            for previous_support in previous_options:
                for following_support in following_options:
                    candidate = _candidate_with_support_points(
                        points,
                        source_candidate,
                        previous_support[0],
                        following_support[0],
                        required_radius,
                        analytic_surface=analytic_surface,
                    )
                    if (
                        previous_support[3] < candidate.tangent_distance_m + minimum_remainder
                        or following_support[3] < candidate.tangent_distance_m + minimum_remainder
                    ):
                        continue
                    score = (
                        previous_support[2] + following_support[2],
                        max(previous_support[1], following_support[1]),
                        max(previous_support[2], following_support[2]),
                        previous_support[2],
                    )
                    feasible.append((score, candidate, previous_support, following_support))
            if feasible:
                break
            if enforce_window_extent and requires_window_envelope:
                window_envelope_fallbacks.add(source_candidate.index)
        if not feasible:
            unavailable.add(source_candidate.index)
            continue
        _score, candidate, previous_support, following_support = min(
            feasible,
            key=lambda value: value[0],
        )
        plans[candidate.index] = _SweepCornerPlan(
            candidate=candidate,
            previous_support_index=previous_support[0],
            following_support_index=following_support[0],
            maximum_support_deviation_m=max(previous_support[1], following_support[1]),
        )

    overlap_conflicts: set[int] = set()
    ordered_plans = sorted(plans.values(), key=lambda plan: plan.candidate.index)
    for first, second in zip(ordered_plans[:-1], ordered_plans[1:]):
        if first.following_support_index < second.previous_support_index:
            continue
        available = _typed_path_arclength(
            points,
            segment_geometry,
            first.candidate.index,
            second.candidate.index,
            analytic_surface=analytic_surface,
        )
        consumed = first.candidate.tangent_distance_m + second.candidate.tangent_distance_m
        if consumed > max(available - minimum_remainder, 0.0):
            overlap_conflicts.update((first.candidate.index, second.candidate.index))

    conflicts = unavailable | overlap_conflicts
    multisegment_plans = [
        plan
        for plan in plans.values()
        if plan.candidate.index - plan.previous_support_index > 1
        or plan.following_support_index - plan.candidate.index > 1
    ]
    maximum_span = max(
        (
            max(
                plan.candidate.index - plan.previous_support_index,
                plan.following_support_index - plan.candidate.index,
            )
            for plan in plans.values()
        ),
        default=1,
    )
    diagnostics: dict[str, float | int] = {
        "sweep_bend_multisegment_support_count": int(len(multisegment_plans)),
        "sweep_bend_support_max_span_segments": int(maximum_span),
        "sweep_bend_support_max_deviation_m": float(
            max((plan.maximum_support_deviation_m for plan in plans.values()), default=0.0)
        ),
        "sweep_bend_support_unavailable_count": int(len(unavailable)),
        "sweep_bend_overlap_conflict_count": int(len(overlap_conflicts)),
        "sweep_bend_window_envelope_count": int(
            sum(
                plan.previous_support_index < plan.candidate.index - 1
                or plan.following_support_index > plan.candidate.index + 1
                for plan in plans.values()
                if "window_radius" in plan.candidate.detection_reasons
            )
        ),
        "sweep_bend_window_envelope_fallback_count": int(len(window_envelope_fallbacks)),
    }
    return plans, conflicts, diagnostics


def _open_corner_support_options(
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    candidate: _SweepCornerCandidate,
    *,
    side: int,
    candidate_indices: set[int],
    maximum_deviation: float,
    analytic_surface: _AnalyticCylinderSurface | None,
    enforce_window_extent: bool = True,
) -> list[tuple[int, float, int, float]]:
    corner_index = candidate.index
    leg_geometry = candidate.incoming_geometry if side < 0 else candidate.outgoing_geometry
    maximum_step = corner_index if side < 0 else len(points) - corner_index - 1
    options: list[tuple[int, float, int, float]] = []
    for step in range(1, maximum_step + 1):
        support_index = corner_index + side * step
        crossed_geometry = _corner_leg_segment_geometry(
            segment_geometry,
            corner_index,
            step,
            side=side,
        )
        if not _corner_support_geometry_is_compatible(
            crossed_geometry,
            leg_geometry=leg_geometry,
            blend_geometry=candidate.blend_geometry,
        ):
            break
        path_indices = (
            list(range(corner_index, support_index - 1, -1))
            if side < 0
            else list(range(corner_index, support_index + 1))
        )
        coordinates = _corner_leg_coordinates(
            points,
            path_indices,
            leg_geometry,
            analytic_surface=analytic_surface,
        )
        support_vector = coordinates[-1]
        support_distance = float(np.linalg.norm(support_vector))
        if support_distance <= 1e-15:
            continue
        deviations = _point_segment_distances(
            coordinates,
            np.zeros(coordinates.shape[1], dtype=float),
            support_vector,
        )
        deviation = float(np.max(deviations)) if len(deviations) else 0.0
        if deviation > maximum_deviation:
            break
        required_support_index = candidate.required_previous_support_index if side < 0 else candidate.required_following_support_index
        has_enveloped_window = not enforce_window_extent or (
            support_index <= required_support_index if side < 0 else support_index >= required_support_index
        )
        if has_enveloped_window:
            options.append((support_index, deviation, step, support_distance))
        if support_index in candidate_indices and support_index != corner_index:
            break
    return options


def _corner_leg_segment_geometry(
    segment_geometry: list[WirePathGeometry],
    corner_index: int,
    step: int,
    *,
    side: int,
) -> list[WirePathGeometry]:
    if side < 0:
        indices = (corner_index - offset for offset in range(1, step + 1))
    else:
        indices = (corner_index + offset - 1 for offset in range(1, step + 1))
    return [WirePathGeometry(segment_geometry[index]) for index in indices]


def _corner_support_geometry_is_compatible(
    crossed_geometry: list[WirePathGeometry],
    *,
    leg_geometry: WirePathGeometry,
    blend_geometry: WirePathGeometry,
) -> bool:
    """Allow tangent support to enter one more-constrained route interval.

    Connector builders place their FREE_3D/OFFSET_SURFACE provenance boundary
    at a sampled point, not at a curvature event. Requiring all tangent support
    to stay on one side of that point makes otherwise mirrored bends depend on
    the final sub-segment length. A spatial blend may therefore borrow bounded
    support from the immediately adjacent, more-constrained interval. It may
    not cross a second transition or relax back into a less-constrained
    contract; the existing chord-deviation and corner-overlap checks still
    bound the geometric replacement.
    """

    if not crossed_geometry or crossed_geometry[0] != WirePathGeometry(leg_geometry):
        return False
    geometry_rank = {
        WirePathGeometry.SURFACE: 0,
        WirePathGeometry.OFFSET_SURFACE: 1,
        WirePathGeometry.FREE_3D: 2,
    }
    blend_rank = geometry_rank[WirePathGeometry(blend_geometry)]
    ranks = [geometry_rank[value] for value in crossed_geometry]
    if any(rank > blend_rank for rank in ranks):
        return False
    transition_count = sum(
        previous != following
        for previous, following in zip(crossed_geometry[:-1], crossed_geometry[1:])
    )
    return bool(
        transition_count <= 1
        and all(following <= previous for previous, following in zip(ranks[:-1], ranks[1:]))
    )


def _corner_leg_coordinates(
    points: np.ndarray,
    indices: list[int],
    geometry: WirePathGeometry,
    *,
    analytic_surface: _AnalyticCylinderSurface | None,
) -> np.ndarray:
    corner = np.asarray(points[indices[0]], dtype=float)
    samples = np.asarray(points, dtype=float)[indices]
    if geometry == WirePathGeometry.SURFACE and analytic_surface is not None:
        return analytic_surface.log_map(corner, samples)
    return samples - corner


def _candidate_with_support_points(
    points: np.ndarray,
    candidate: _SweepCornerCandidate,
    previous_support_index: int,
    following_support_index: int,
    required_radius: float,
    *,
    analytic_surface: _AnalyticCylinderSurface | None,
) -> _SweepCornerCandidate:
    corner = np.asarray(points[candidate.index], dtype=float)
    supports = np.asarray(
        [points[previous_support_index], points[following_support_index]],
        dtype=float,
    )
    if candidate.blend_geometry == WirePathGeometry.SURFACE and analytic_surface is not None:
        local = analytic_surface.log_map(corner, supports)
        incoming = -local[0]
        outgoing = local[1]
        reserve = 1.05
    else:
        incoming = corner - supports[0]
        outgoing = supports[1] - corner
        reserve = 1.0
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if incoming_length <= 1e-15 or outgoing_length <= 1e-15:
        return replace(candidate, tangent_distance_m=0.0)
    turn_angle = float(
        np.arccos(
            np.clip(
                np.dot(incoming, outgoing) / (incoming_length * outgoing_length),
                -1.0,
                1.0,
            )
        )
    )
    tangent_distance = reserve * required_radius * np.tan(turn_angle / 2.0) if turn_angle < np.pi - 1e-6 else np.inf
    return replace(
        candidate,
        turn_angle_rad=turn_angle,
        tangent_distance_m=float(tangent_distance),
    )


def _typed_path_arclength(
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    start_index: int,
    stop_index: int,
    *,
    analytic_surface: _AnalyticCylinderSurface | None,
) -> float:
    if stop_index <= start_index:
        return 0.0
    length = 0.0
    for index in range(start_index, stop_index):
        geometry = WirePathGeometry(segment_geometry[index])
        if geometry == WirePathGeometry.SURFACE and analytic_surface is not None:
            local = analytic_surface.log_map(points[index], points[index + 1][None, :])
            length += float(np.linalg.norm(local[0]))
        else:
            length += float(np.linalg.norm(points[index + 1] - points[index]))
    return length


def _build_radius_controlled_fillet_path(
    part: CoilPart,
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    resolved: dict[int, _SweepCornerCandidate],
    radius: float,
    config: CoilGenConfig,
    *,
    closed: bool,
    corner_plans: dict[int, _SweepCornerPlan] | None = None,
) -> _SweepFilletBuildResult:
    if not resolved:
        return _SweepFilletBuildResult(
            points=points,
            segment_geometry=segment_geometry,
            inserted_count=0,
            maximum_projection_m=0.0,
            resolved_indices=frozenset(),
            failed_indices=frozenset(),
            surface_resolved_count=0,
            surface_validation_failure_count=0,
            maximum_radius_inflation=1.0,
        )
    arc_step = np.deg2rad(max(float(config.sweep_bend_arc_angle_deg), 0.25))
    analytic_surface = _analytic_cylinder_surface_constraint(part, config)
    projector = _SurfaceProjector(part.mesh) if analytic_surface is None and part.mesh.n_faces > 0 else None
    arcs: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray], WirePathGeometry]] = {}
    inserted_count = 0
    maximum_projection = 0.0
    maximum_radius_inflation = 1.0
    failed_indices: set[int] = set()
    surface_resolved_count = 0
    surface_validation_failure_count = 0
    for index, candidate in resolved.items():
        plan = corner_plans.get(index) if corner_plans is not None else None
        previous_index = plan.previous_support_index if plan is not None else (index - 1) % len(points)
        following_index = plan.following_support_index if plan is not None else (index + 1) % len(points)
        previous = points[previous_index]
        corner = points[index]
        following = points[following_index]
        if candidate.blend_geometry == WirePathGeometry.SURFACE and analytic_surface is not None:
            validated = _validated_analytic_cylinder_fillet(
                previous,
                corner,
                following,
                candidate,
                radius,
                analytic_surface,
                config,
            )
            if validated is None:
                failed_indices.add(index)
                surface_validation_failure_count += 1
                continue
            start, end, arc, projection, radius_inflation = validated
            maximum_projection = max(maximum_projection, projection)
            maximum_radius_inflation = max(maximum_radius_inflation, radius_inflation)
            surface_resolved_count += 1
            arcs[index] = (start, end, arc, candidate.blend_geometry)
            inserted_count += max(len(arc) - 2, 0)
            continue
        incoming = corner - previous
        outgoing = following - corner
        incoming_unit = incoming / max(float(np.linalg.norm(incoming)), 1e-15)
        outgoing_unit = outgoing / max(float(np.linalg.norm(outgoing)), 1e-15)
        tangent_distance = candidate.tangent_distance_m
        start = corner - incoming_unit * tangent_distance
        end = corner + outgoing_unit * tangent_distance
        center_direction = outgoing_unit - incoming_unit
        center_norm = float(np.linalg.norm(center_direction))
        if center_norm <= 1e-15:
            continue
        center = corner + center_direction / center_norm * (radius / max(np.cos(candidate.turn_angle_rad / 2.0), 1e-15))
        arc = _circular_arc_points(start, end, center, arc_step)
        if candidate.blend_geometry == WirePathGeometry.SURFACE and projector is not None and len(arc) > 2:
            interior = np.asarray(arc[1:-1], dtype=float)
            projected, _normals = projector.project(interior)
            maximum_projection = max(
                maximum_projection,
                float(np.max(np.linalg.norm(projected - interior, axis=1))),
            )
            arc[1:-1] = [point for point in projected]
        arcs[index] = (start, end, arc, candidate.blend_geometry)
        inserted_count += max(len(arc) - 2, 0)

    if not arcs:
        return _SweepFilletBuildResult(
            points=points,
            segment_geometry=segment_geometry,
            inserted_count=0,
            maximum_projection_m=maximum_projection,
            resolved_indices=frozenset(),
            failed_indices=frozenset(failed_indices),
            surface_resolved_count=0,
            surface_validation_failure_count=surface_validation_failure_count,
            maximum_radius_inflation=maximum_radius_inflation,
        )
    if not closed and corner_plans is not None:
        output_points, output_geometry = _assemble_open_planned_fillet_path(
            points,
            segment_geometry,
            resolved,
            arcs,
            corner_plans,
        )
        return _SweepFilletBuildResult(
            points=output_points,
            segment_geometry=output_geometry,
            inserted_count=inserted_count,
            maximum_projection_m=maximum_projection,
            resolved_indices=frozenset(arcs),
            failed_indices=frozenset(failed_indices),
            surface_resolved_count=surface_resolved_count,
            surface_validation_failure_count=surface_validation_failure_count,
            maximum_radius_inflation=maximum_radius_inflation,
        )
    first_point = arcs[0][1] if closed and 0 in arcs else points[0]
    output_points: list[np.ndarray] = [np.asarray(first_point, dtype=float)]
    output_geometry: list[WirePathGeometry] = []
    segment_count = len(points) if closed else len(points) - 1
    for segment_index in range(segment_count):
        following_index = (segment_index + 1) % len(points)
        segment_end = arcs[following_index][0] if following_index in arcs else points[following_index]
        _append_typed_path_point(
            output_points,
            output_geometry,
            segment_end,
            segment_geometry[segment_index],
        )
        if following_index in arcs:
            _start, _end, arc, blend_geometry = arcs[following_index]
            for arc_point in arc[1:]:
                _append_typed_path_point(
                    output_points,
                    output_geometry,
                    arc_point,
                    blend_geometry,
                )

    if closed and len(output_points) > 1 and np.linalg.norm(output_points[-1] - output_points[0]) <= 1e-12:
        output_points.pop()
    return _SweepFilletBuildResult(
        points=np.asarray(output_points, dtype=float),
        segment_geometry=output_geometry,
        inserted_count=inserted_count,
        maximum_projection_m=maximum_projection,
        resolved_indices=frozenset(arcs),
        failed_indices=frozenset(failed_indices),
        surface_resolved_count=surface_resolved_count,
        surface_validation_failure_count=surface_validation_failure_count,
        maximum_radius_inflation=maximum_radius_inflation,
    )


def _assemble_open_planned_fillet_path(
    points: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    resolved: dict[int, _SweepCornerCandidate],
    arcs: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray], WirePathGeometry]],
    corner_plans: dict[int, _SweepCornerPlan],
) -> tuple[np.ndarray, list[WirePathGeometry]]:
    removed_indices: set[int] = set()
    for index in arcs:
        plan = corner_plans[index]
        removed_indices.update(range(plan.previous_support_index + 1, index))
        removed_indices.update(range(index + 1, plan.following_support_index))

    output_points: list[np.ndarray] = [np.asarray(points[0], dtype=float)]
    output_geometry: list[WirePathGeometry] = []
    for index in range(1, len(points)):
        if index in arcs:
            start, _end, arc, blend_geometry = arcs[index]
            _append_typed_path_point(
                output_points,
                output_geometry,
                start,
                resolved[index].incoming_geometry,
            )
            for arc_point in arc[1:]:
                _append_typed_path_point(
                    output_points,
                    output_geometry,
                    arc_point,
                    blend_geometry,
                )
            continue
        if index in removed_indices:
            continue
        _append_typed_path_point(
            output_points,
            output_geometry,
            points[index],
            segment_geometry[index - 1],
        )
    return np.asarray(output_points, dtype=float), output_geometry


def _validated_analytic_cylinder_fillet(
    previous: np.ndarray,
    corner: np.ndarray,
    following: np.ndarray,
    candidate: _SweepCornerCandidate,
    required_radius: float,
    surface: _AnalyticCylinderSurface,
    config: CoilGenConfig,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float, float] | None:
    """Construct an intrinsic cylinder fillet and validate it in physical 3-D."""

    local_neighbours = surface.log_map(corner, np.vstack((previous, following)))
    previous_local = local_neighbours[0]
    following_local = local_neighbours[1]
    incoming = -previous_local
    outgoing = following_local
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if incoming_length <= 1e-15 or outgoing_length <= 1e-15:
        return None
    incoming_unit = incoming / incoming_length
    outgoing_unit = outgoing / outgoing_length
    center_direction = outgoing_unit - incoming_unit
    center_norm = float(np.linalg.norm(center_direction))
    half_angle_cosine = float(np.cos(candidate.turn_angle_rad / 2.0))
    if center_norm <= 1e-15 or half_angle_cosine <= 1e-15:
        return None

    output_step = np.deg2rad(max(float(config.sweep_bend_arc_angle_deg), 0.25))
    validation_step = min(output_step, np.deg2rad(0.25))
    tangency_tolerance_deg = max(float(config.sweep_tangent_angle_tolerance_deg), 0.75)
    projection_limit = max(
        0.25 * required_radius,
        10.0 * float(config.sweep_max_chord_error),
        1e-6,
    )
    for inflation in (1.0, 1.0025, 1.005, 1.01, 1.02, 1.05):
        trial_radius = required_radius * inflation
        tangent_distance = trial_radius * np.tan(candidate.turn_angle_rad / 2.0)
        if tangent_distance >= min(incoming_length, outgoing_length) - 1e-9:
            continue
        start_local = -incoming_unit * tangent_distance
        end_local = outgoing_unit * tangent_distance
        center_local = center_direction / center_norm * (trial_radius / half_angle_cosine)
        validation_local = np.asarray(
            _circular_arc_points(
                np.append(start_local, 0.0),
                np.append(end_local, 0.0),
                np.append(center_local, 0.0),
                validation_step,
            ),
            dtype=float,
        )[:, :2]
        validation_arc = surface.exp_map(corner, validation_local)
        if not np.all(np.isfinite(validation_arc)):
            continue
        if not np.all(surface.within_domain(validation_arc, 1e-9)):
            continue
        _projected_corner, corner_normals = surface.project(corner[None, :])
        corner_normal = corner_normals[0]
        circumferential_direction = np.cross(surface.axis, corner_normal)
        tangent_plane_arc = (
            corner
            + validation_local[:, 0, None] * circumferential_direction
            + validation_local[:, 1, None] * surface.axis
        )
        projection = float(np.max(np.linalg.norm(validation_arc - tangent_plane_arc, axis=1)))
        if projection > projection_limit:
            continue
        radii = _polyline_circumradii(
            validation_arc,
            closed=False,
            minimum_turn_angle_deg=1e-7,
        )
        if not len(radii) or float(np.min(radii)) < 0.999 * required_radius:
            continue

        probe_distance = min(max(1e-5 * trial_radius, 1e-9), 0.1 * tangent_distance)
        incoming_join = previous_local - validation_local[0]
        outgoing_join = following_local - validation_local[-1]
        incoming_join /= max(float(np.linalg.norm(incoming_join)), 1e-15)
        outgoing_join /= max(float(np.linalg.norm(outgoing_join)), 1e-15)
        incoming_probe = surface.exp_map(
            corner,
            (validation_local[0] + probe_distance * incoming_join)[None, :],
        )
        outgoing_probe = surface.exp_map(
            corner,
            (validation_local[-1] + probe_distance * outgoing_join)[None, :],
        )
        incoming_tangent_error = _vector_angle_deg(
            validation_arc[0] - incoming_probe[0],
            validation_arc[1] - validation_arc[0],
        )
        outgoing_tangent_error = _vector_angle_deg(
            validation_arc[-1] - validation_arc[-2],
            outgoing_probe[0] - validation_arc[-1],
        )
        if max(incoming_tangent_error, outgoing_tangent_error) > tangency_tolerance_deg:
            continue

        output_local = np.asarray(
            _circular_arc_points(
                np.append(start_local, 0.0),
                np.append(end_local, 0.0),
                np.append(center_local, 0.0),
                output_step,
            ),
            dtype=float,
        )[:, :2]
        output_arc = surface.exp_map(corner, output_local)
        if not np.all(surface.within_domain(output_arc, 1e-9)):
            continue
        output_tangent_plane = (
            corner + output_local[:, 0, None] * circumferential_direction + output_local[:, 1, None] * surface.axis
        )
        output_projection = float(np.max(np.linalg.norm(output_arc - output_tangent_plane, axis=1)))
        return (
            output_arc[0],
            output_arc[-1],
            [point for point in output_arc],
            max(projection, output_projection),
            float(inflation),
        )
    return None


def _surface_path_point_mask(
    segment_geometry: list[WirePathGeometry],
    point_count: int,
    *,
    closed: bool,
) -> np.ndarray:
    mask = np.zeros(point_count, dtype=bool)
    for segment_index, geometry in enumerate(segment_geometry):
        if WirePathGeometry(geometry) != WirePathGeometry.SURFACE:
            continue
        start = segment_index % point_count
        following = (segment_index + 1) % point_count
        mask[start] = True
        mask[following] = True
    if not closed and len(segment_geometry) != max(point_count - 1, 0):
        raise ValueError("Open surface path provenance does not match its point count.")
    if closed and len(segment_geometry) != point_count:
        raise ValueError("Closed surface path provenance does not match its point count.")
    return mask


def _append_typed_path_point(
    points: list[np.ndarray],
    segment_geometry: list[WirePathGeometry],
    point: np.ndarray,
    geometry: WirePathGeometry,
) -> None:
    point = np.asarray(point, dtype=float)
    if np.linalg.norm(point - points[-1]) <= 1e-12:
        return
    points.append(point)
    segment_geometry.append(WirePathGeometry(geometry))


def _circular_arc_points(
    start: np.ndarray, end: np.ndarray, center: np.ndarray, max_angle_step_rad: float
) -> list[np.ndarray]:
    r0 = start - center
    r1 = end - center
    radius = float(np.linalg.norm(r0))
    if radius <= 1e-15:
        return [start, end]
    r0_unit = r0 / radius
    r1_unit = r1 / max(float(np.linalg.norm(r1)), 1e-15)
    axis = np.cross(r0_unit, r1_unit)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-15:
        return [start, end]
    axis = axis / axis_norm
    angle = float(np.arccos(np.clip(np.dot(r0_unit, r1_unit), -1.0, 1.0)))
    count = max(int(np.ceil(angle / max(max_angle_step_rad, 1e-6))), 1)
    points: list[np.ndarray] = []
    for ind in range(count + 1):
        theta = angle * ind / count
        rotated = (
            r0_unit * np.cos(theta)
            + np.cross(axis, r0_unit) * np.sin(theta)
            + axis * np.dot(axis, r0_unit) * (1.0 - np.cos(theta))
        )
        points.append(center + rotated * radius)
    return points


def _mounting_centerline_offset(config: CoilGenConfig, cross_section: np.ndarray) -> float:
    mode = _normalised_text(config.conductor_mounting_mode)
    if mode == "centered":
        return 0.0
    normal_coordinates = np.asarray(cross_section, dtype=float)[:, 1]
    if mode == "surface_outer":
        return -float(np.min(normal_coordinates))
    if mode == "surface_inner":
        return -float(np.max(normal_coordinates))
    return 0.0


def _physical_clearance_diagnostics(
    part: CoilPart,
    points: np.ndarray,
    cross_section: np.ndarray,
    config: CoilGenConfig,
    *,
    closed: bool,
) -> dict[str, float | int | bool | str]:
    envelope_radius = _cross_section_envelope_radius(cross_section)
    envelope_diameter = 2.0 * envelope_radius
    required_clearance = _required_centerline_clearance(cross_section, config)
    diagnostics: dict[str, float | int | bool | str] = {
        "sweep_conductor_envelope_diameter_m": float(envelope_diameter),
        "sweep_required_centerline_clearance_m": float(required_clearance),
    }
    segment_lengths = _centerline_segment_lengths(points, closed=closed)
    minimum_segment_length = float(np.min(segment_lengths)) if len(segment_lengths) else 0.0
    minimum_allowed_segment_length = max(
        1e-3 * envelope_diameter,
        10.0 * float(config.sweep_collinearity_tolerance),
        1e-9,
    )
    diagnostics.update(
        {
            "sweep_min_segment_length_m": minimum_segment_length,
            "sweep_min_allowed_segment_length_m": float(minimum_allowed_segment_length),
            "sweep_min_segment_length_violation": bool(
                len(segment_lengths) > 0 and minimum_segment_length < minimum_allowed_segment_length
            ),
        }
    )
    route_min = part.route_diagnostics.get("min_wire_wire_distance")
    route_segment_a = int(part.route_diagnostics.get("min_wire_wire_segment_a", -1))
    route_segment_b = int(part.route_diagnostics.get("min_wire_wire_segment_b", -1))
    if route_min is not None and route_segment_a >= 0 and route_segment_b >= 0:
        margin = float(route_min) - required_clearance
        diagnostics["sweep_route_clearance_margin_m"] = float(margin)
        diagnostics["sweep_route_clearance_violation"] = bool(margin < 0.0)

    mode = _normalised_text(config.sweep_self_intersection_check)
    diagnostics["sweep_self_intersection_check"] = mode
    if mode == "capsule_fast":
        segment_count = len(points) if closed else max(len(points) - 1, 0)
        diagnostics["sweep_self_intersection_segment_count"] = int(segment_count)
        diagnostics["sweep_self_intersection_skipped"] = False
        diagnostics["sweep_self_intersection_skipped_due_route_precheck_failure"] = False
        capsule = _capsule_clearance_diagnostics(
            points,
            required_clearance,
            envelope_diameter,
            closed=closed,
        )
        diagnostics.update(capsule)
    else:
        diagnostics["sweep_self_intersection_skipped"] = True
    return diagnostics


def save_mesh_as_stl(mesh: Mesh, path: str | Path, *, solid_name: str = "coil_body") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = compute_face_normals(mesh) if len(mesh.faces) else np.zeros((0, 3), dtype=float)
    lines = [f"solid {solid_name}"]
    for normal, face in zip(normals, mesh.faces):
        lines.append(f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}")
        lines.append("    outer loop")
        for vertex in mesh.vertices[face]:
            lines.append(f"      vertex {vertex[0]:.9e} {vertex[1]:.9e} {vertex[2]:.9e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def save_swept_bodies_as_stl(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil_body",
    min_clearance: float | None = None,
    fail_on_violations: bool = True,
) -> list[Path]:
    """Export each swept conductor mesh and its current-carrying surface.

    STL carries only triangulated boundaries, not analytic volume semantics.
    One pair is written for every coil part that has a swept mesh. Strict export
    readiness is enabled by default; bypass it only for visualization/debugging.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    swept_parts = [
        (part_ind, part) for part_ind, part in enumerate(result.coil_parts, start=1) if part.swept_body is not None
    ]
    for part_ind, part in swept_parts:
        _check_export_readiness(part, part_ind, min_clearance, fail_on_violations, format_name="STL")
    for part_ind, part in swept_parts:
        saved.append(
            save_mesh_as_stl(
                part.swept_body, output / f"{prefix}_part{part_ind}.stl", solid_name=f"{prefix}_part{part_ind}"
            )
        )
        surface_name = f"{prefix}_part{part_ind}_current_carrying_surface"
        saved.append(
            save_mesh_as_stl(
                part.mesh,
                output / f"{surface_name}.stl",
                solid_name=surface_name,
            )
        )
    return saved


def save_mesh_as_step(
    mesh: Mesh,
    path: str | Path,
    *,
    solid_name: str = "coil_body",
    surface_mode: str = "advanced_brep",
) -> Path:
    """Save a swept conductor mesh as a triangle-derived STEP BREP.

    The default ``advanced_brep`` mode wraps every triangle in a planar
    ``ADVANCED_FACE``. It is a syntactically valid BREP for interchange, but it
    is still faceted and is not the analytic sweep intended for robust Boolean
    support subtraction. ``faceted_brep`` is a compatibility fallback.
    Coordinates are stored in SI metres.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    _validate_mesh_for_step(vertices, faces)
    mode = _normalize_step_surface_mode(surface_mode)
    writer = _StepWriter()

    product_shape, context = _write_step_product_header(writer, solid_name)
    if mode == "advanced_brep":
        representation = _write_advanced_brep_shape_representation(writer, vertices, faces, solid_name, context)
        description = "CoilGen swept conductor advanced BREP; planar surfaces; units=m"
        schema = "AUTOMOTIVE_DESIGN"
    else:
        representation = _write_faceted_brep_shape_representation(writer, vertices, faces, solid_name, context)
        description = "CoilGen swept conductor faceted BREP; units=m"
        schema = "AUTOMOTIVE_DESIGN_CC2"
    writer.add(f"SHAPE_DEFINITION_REPRESENTATION({product_shape},{representation})")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    text = "\n".join(
        [
            "ISO-10303-21;",
            "HEADER;",
            f"FILE_DESCRIPTION(('{description}'),'2;1');",
            f"FILE_NAME('{_step_string(path.name)}','{timestamp}',('CoilGen'),('CoilGen'),'CoilGen','CoilGen','');",
            f"FILE_SCHEMA(('{schema}'));",
            "ENDSEC;",
            "DATA;",
            *writer.lines,
            "ENDSEC;",
            "END-ISO-10303-21;",
            "",
        ]
    )
    path.write_text(text, encoding="ascii")
    return path


def _write_step_product_header(writer: "_StepWriter", solid_name: str) -> tuple[str, str]:
    application_context = writer.add("APPLICATION_CONTEXT('mechanical design')")
    writer.add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{application_context})"
    )
    product_context = writer.add(f"PRODUCT_CONTEXT('',{application_context},'mechanical')")
    product = writer.add(f"PRODUCT('{_step_string(solid_name)}','{_step_string(solid_name)}','',({product_context}))")
    product_formation = writer.add(f"PRODUCT_DEFINITION_FORMATION('','',{product})")
    product_definition_context = writer.add(
        f"PRODUCT_DEFINITION_CONTEXT('part definition',{application_context},'design')"
    )
    product_definition = writer.add(f"PRODUCT_DEFINITION('design','',{product_formation},{product_definition_context})")
    product_shape = writer.add(f"PRODUCT_DEFINITION_SHAPE('','',{product_definition})")
    length_unit = writer.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.))")
    plane_angle_unit = writer.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))")
    solid_angle_unit = writer.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())")
    uncertainty = writer.add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-9),{length_unit},'distance_accuracy_value','')"
    )
    context = writer.add(
        (
            "(GEOMETRIC_REPRESENTATION_CONTEXT(3) "
            f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({uncertainty})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({length_unit},{plane_angle_unit},{solid_angle_unit})) "
            "REPRESENTATION_CONTEXT('','3D'))"
        )
    )
    return product_shape, context


def _write_faceted_brep_shape_representation(
    writer: "_StepWriter",
    vertices: np.ndarray,
    faces: np.ndarray,
    solid_name: str,
    context: str,
) -> str:
    point_refs = [
        writer.add(f"CARTESIAN_POINT('',({_step_float(vertex[0])},{_step_float(vertex[1])},{_step_float(vertex[2])}))")
        for vertex in vertices
    ]
    face_refs = []
    for face in faces:
        loop_points = ",".join(point_refs[int(index)] for index in face)
        poly_loop = writer.add(f"POLY_LOOP('',({loop_points}))")
        bound = writer.add(f"FACE_OUTER_BOUND('',{poly_loop},.T.)")
        face_ref = writer.add(f"FACE('',({bound}))")
        face_refs.append(face_ref)
    shell = writer.add(f"CLOSED_SHELL('',({','.join(face_refs)}))")
    brep = writer.add(f"FACETED_BREP('{_step_string(solid_name)}',{shell})")
    representation = writer.add(f"SHAPE_REPRESENTATION('',({brep}),{context})")
    return representation


def _write_advanced_brep_shape_representation(
    writer: "_StepWriter",
    vertices: np.ndarray,
    faces: np.ndarray,
    solid_name: str,
    context: str,
) -> str:
    point_refs = [
        writer.add(f"CARTESIAN_POINT('',({_step_float(vertex[0])},{_step_float(vertex[1])},{_step_float(vertex[2])}))")
        for vertex in vertices
    ]
    vertex_refs = [writer.add(f"VERTEX_POINT('',{point_ref})") for point_ref in point_refs]

    edge_refs: dict[tuple[int, int], tuple[str, bool]] = {}
    for face in faces:
        for start, end in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            key = tuple(sorted((start, end)))
            if key in edge_refs:
                continue
            low, high = key
            delta = vertices[high] - vertices[low]
            length = float(np.linalg.norm(delta))
            if length <= 0.0:
                raise ValueError("STEP export found a zero-length mesh edge.")
            direction = writer.add(_step_direction(delta / length))
            vector = writer.add(f"VECTOR('',{direction},{_step_float(length)})")
            line = writer.add(f"LINE('',{point_refs[low]},{vector})")
            edge = writer.add(f"EDGE_CURVE('',{vertex_refs[low]},{vertex_refs[high]},{line},.T.)")
            edge_refs[key] = (edge, True)

    advanced_face_refs = []
    for face in faces:
        face_indices = [int(index) for index in face]
        oriented_edges = []
        for start, end in (
            (face_indices[0], face_indices[1]),
            (face_indices[1], face_indices[2]),
            (face_indices[2], face_indices[0]),
        ):
            edge, _ = edge_refs[tuple(sorted((start, end)))]
            orientation = ".T." if start < end else ".F."
            oriented_edges.append(writer.add(f"ORIENTED_EDGE('',*,*,{edge},{orientation})"))
        edge_loop = writer.add(f"EDGE_LOOP('',({','.join(oriented_edges)}))")
        bound = writer.add(f"FACE_OUTER_BOUND('',{edge_loop},.T.)")
        plane = _write_face_plane(writer, vertices[face_indices], point_refs[face_indices[0]])
        advanced_face_refs.append(writer.add(f"ADVANCED_FACE('',({bound}),{plane},.T.)"))

    shell = writer.add(f"CLOSED_SHELL('',({','.join(advanced_face_refs)}))")
    brep = writer.add(f"MANIFOLD_SOLID_BREP('{_step_string(solid_name)}',{shell})")
    representation = writer.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',({brep}),{context})")
    return representation


def save_swept_bodies_as_step(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil_body",
    surface_mode: str = "advanced_brep",
    min_clearance: float | None = None,
    fail_on_violations: bool = True,
) -> list[Path]:
    """Save swept triangle meshes as faceted STEP solids.

    This is a mesh/interchange export, not the analytic CAD manufacturing
    backend. Export readiness checks remain strict by default.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    swept_parts = [
        (part_ind, part) for part_ind, part in enumerate(result.coil_parts, start=1) if part.swept_body is not None
    ]
    for part_ind, part in swept_parts:
        _check_export_readiness(part, part_ind, min_clearance, fail_on_violations, format_name="STEP")
    for part_ind, part in swept_parts:
        saved.append(
            save_mesh_as_step(
                part.swept_body,
                output / f"{prefix}_part{part_ind}.step",
                solid_name=f"{prefix}_part{part_ind}",
                surface_mode=surface_mode,
            )
        )
    return saved


def _check_export_readiness(
    part: CoilPart,
    part_ind: int,
    min_clearance: float | None,
    fail_on_violations: bool,
    *,
    format_name: str = "STL",
) -> None:
    if part.swept_body is None:
        return
    diagnostics = part.manufacturing_diagnostics or mesh_manufacturing_diagnostics(part.swept_body)
    part.manufacturing_diagnostics = diagnostics
    _finalize_manufacturing_diagnostics(part)
    diagnostics = part.manufacturing_diagnostics
    clearance = None if min_clearance is None else max(float(min_clearance), 0.0)
    clearance_violation = False
    if clearance is not None and clearance > 0.0:
        min_wire_distance = float(part.route_diagnostics.get("min_wire_wire_distance", 0.0))
        seg_a = int(part.route_diagnostics.get("min_wire_wire_segment_a", -1))
        seg_b = int(part.route_diagnostics.get("min_wire_wire_segment_b", -1))
        clearance_violation = seg_a >= 0 and seg_b >= 0 and min_wire_distance < clearance
        diagnostics["export_min_clearance"] = float(clearance)
        diagnostics["export_clearance_violation"] = bool(clearance_violation)
        diagnostics["stl_min_clearance"] = float(clearance)
        diagnostics["stl_clearance_violation"] = bool(clearance_violation)
    if fail_on_violations and (not bool(diagnostics.get("manufacturing_ok", False)) or clearance_violation):
        reasons = str(diagnostics.get("manufacturing_failure_reasons", "")).strip() or "unspecified"
        raise ValueError(
            f"Part {part_ind} is not ready for {format_name} export: "
            f"manufacturing_ok={diagnostics.get('manufacturing_ok', False)}, "
            f"failure_reasons={reasons}, "
            f"export_clearance_violation={diagnostics.get('export_clearance_violation', False)}, "
            f"short_edges={diagnostics.get('short_edge_count', 0)}, "
            f"open_edges={diagnostics.get('open_edge_count', 0)}, "
            f"misoriented_edges={diagnostics.get('misoriented_edge_count', 0)}, "
            f"signed_volume_m3={diagnostics.get('signed_volume_m3', 0.0)}."
        )


def _validate_mesh_for_step(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("STEP export requires mesh vertices with shape (n, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("STEP export requires triangular mesh faces with shape (m, 3).")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("STEP export requires a non-empty mesh.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("STEP export requires finite vertex coordinates.")
    if np.any((faces < 0) | (faces >= len(vertices))):
        raise ValueError("STEP export found face indices outside the vertex array.")


def _normalize_step_surface_mode(surface_mode: str) -> str:
    mode = str(surface_mode).strip().lower().replace("-", "_")
    aliases = {
        "advanced": "advanced_brep",
        "advanced_brep": "advanced_brep",
        "parametric": "advanced_brep",
        "plane_faces": "advanced_brep",
        "faceted": "faceted_brep",
        "faceted_brep": "faceted_brep",
        "mesh": "faceted_brep",
    }
    if mode not in aliases:
        raise ValueError("STEP surface_mode must be 'advanced_brep' or 'faceted_brep'.")
    return aliases[mode]


def _write_face_plane(writer: "_StepWriter", triangle: np.ndarray, origin_point_ref: str) -> str:
    edge_u = triangle[1] - triangle[0]
    edge_v = triangle[2] - triangle[0]
    normal = np.cross(edge_u, edge_v)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        raise ValueError("STEP export found a degenerate mesh face.")
    normal = normal / normal_norm
    ref_direction = edge_u / max(float(np.linalg.norm(edge_u)), 1e-15)
    ref_direction = ref_direction - normal * np.dot(ref_direction, normal)
    ref_norm = float(np.linalg.norm(ref_direction))
    if ref_norm <= 0.0:
        ref_direction = edge_v - normal * np.dot(edge_v, normal)
        ref_norm = float(np.linalg.norm(ref_direction))
    if ref_norm <= 0.0:
        raise ValueError("STEP export could not define a planar face coordinate frame.")
    ref_direction = ref_direction / ref_norm
    normal_ref = writer.add(_step_direction(normal))
    ref_direction_ref = writer.add(_step_direction(ref_direction))
    axis = writer.add(f"AXIS2_PLACEMENT_3D('',{origin_point_ref},{normal_ref},{ref_direction_ref})")
    return writer.add(f"PLANE('',{axis})")


class _StepWriter:
    def __init__(self) -> None:
        self._next_id = 1
        self.lines: list[str] = []

    def add(self, entity: str) -> str:
        ref = f"#{self._next_id}"
        self._next_id += 1
        self.lines.append(f"{ref}={entity};")
        return ref


def _step_float(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("STEP export requires finite numeric values.")
    text = f"{number:.12g}"
    if "e" in text:
        text = text.replace("e", "E")
    if "." not in text and "E" not in text:
        text = f"{text}."
    return text


def _step_direction(vector: np.ndarray) -> str:
    values = np.asarray(vector, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("STEP export requires finite 3D direction vectors.")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("STEP export requires non-zero direction vectors.")
    values = values / norm
    return f"DIRECTION('',({_step_float(values[0])},{_step_float(values[1])},{_step_float(values[2])}))"


def _step_string(value: str) -> str:
    return str(value).replace("'", "''")


def _write_rows_csv(path: Path, rows: list[dict[str, float | int | bool | str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["part_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=float)
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)


def _mesh_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=float)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)


def _signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    triangles = vertices[faces]
    return float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def _edge_topology_counts(faces: np.ndarray) -> tuple[int, int, int]:
    edge_counts: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            start = int(a)
            end = int(b)
            edge = (min(start, end), max(start, end))
            state = edge_counts.setdefault(edge, [0, 0])
            state[0] += 1
            state[1] += 1 if start < end else -1
    open_edges = sum(1 for count, _orientation in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count, _orientation in edge_counts.values() if count > 2)
    misoriented_edges = sum(1 for count, orientation in edge_counts.values() if count == 2 and orientation != 0)
    return int(open_edges), int(nonmanifold_edges), int(misoriented_edges)


def _dedupe_path(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
    return points[keep]


def _transfer_path_segment_geometry(
    source_points: np.ndarray,
    source_geometry: list[WirePathGeometry | str],
    target_points: np.ndarray,
    *,
    source_closed: bool,
    target_closed: bool,
) -> list[WirePathGeometry]:
    """Transfer typed interval geometry across a local centerline edit."""

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    source_transfer = source if not source_closed or _is_closed_path(source) else np.vstack((source, source[0]))
    target_transfer = target if not target_closed or _is_closed_path(target) else np.vstack((target, target[0]))
    return transfer_segment_geometry(
        source_transfer,
        source_geometry,
        target_transfer,
    )


def _is_closed_path(points: np.ndarray, tolerance: float = 1e-9) -> bool:
    return bool(len(points) > 2 and np.linalg.norm(points[0] - points[-1]) <= float(tolerance))


def _centerline_segment_lengths(points: np.ndarray, *, closed: bool) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(0, dtype=float)
    following = np.roll(points, -1, axis=0) if closed else points[1:]
    current = points if closed else points[:-1]
    return np.linalg.norm(following - current, axis=1)


def _path_tangents(points: np.ndarray, *, closed: bool) -> np.ndarray:
    tangents = np.zeros_like(points)
    if closed:
        tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    else:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        tangents[1:-1] = points[2:] - points[:-2]
    lengths = np.linalg.norm(tangents, axis=1)
    for ind in np.flatnonzero(lengths <= 1e-15):
        following = points[(ind + 1) % len(points)] - points[ind]
        previous = points[ind] - points[(ind - 1) % len(points)]
        tangents[ind] = following if np.linalg.norm(following) > 1e-15 else previous
    lengths = np.linalg.norm(tangents, axis=1)
    tangents[lengths > 0] /= lengths[lengths > 0, None]
    return tangents


def _path_frames(
    part: CoilPart,
    points: np.ndarray,
    tangents: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    config: CoilGenConfig,
    *,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    method = _effective_sweep_frame_method(config, segment_geometry)
    if method == "surface_normal":
        normals, binormals = _surface_normal_path_frames(part, points, tangents, config)
    else:
        normals, binormals = _parallel_transport_path_frames(
            part,
            points,
            tangents,
            segment_geometry,
            config,
            closed=closed,
        )
    if closed:
        normals, binormals = _enforce_periodic_path_frames(points, tangents, normals)
    return normals, binormals


def _effective_sweep_frame_method(
    config: CoilGenConfig,
    segment_geometry: list[WirePathGeometry],
) -> str:
    requested = _normalised_text(config.sweep_frame_method)
    has_spatial_segment = any(WirePathGeometry(value) != WirePathGeometry.SURFACE for value in segment_geometry)
    if requested == "surface_normal" and not has_spatial_segment:
        return requested
    return "surface_constrained_parallel_transport"


def _surface_normal_path_frames(
    part: CoilPart,
    points: np.ndarray,
    tangents: np.ndarray,
    config: CoilGenConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_normals = _nearest_surface_normals(part, points, config)
    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)
    previous_normal: np.ndarray | None = None

    for ind, tangent in enumerate(tangents):
        normal = mesh_normals[ind] - tangent * np.dot(mesh_normals[ind], tangent)
        if np.linalg.norm(normal) < 1e-10:
            normal = _fallback_normal(tangent, previous_normal)
        normal = normal / max(np.linalg.norm(normal), 1e-15)
        if previous_normal is not None and np.dot(normal, previous_normal) < 0:
            normal *= -1.0
        binormal = np.cross(tangent, normal)
        if np.linalg.norm(binormal) < 1e-10:
            normal = _fallback_normal(tangent, previous_normal)
            binormal = np.cross(tangent, normal)
        binormal = binormal / max(np.linalg.norm(binormal), 1e-15)
        normal = np.cross(binormal, tangent)
        normal = normal / max(np.linalg.norm(normal), 1e-15)
        normals[ind] = normal
        binormals[ind] = binormal
        previous_normal = normal
    return normals, binormals


def _parallel_transport_path_frames(
    part: CoilPart,
    points: np.ndarray,
    tangents: np.ndarray,
    segment_geometry: list[WirePathGeometry],
    config: CoilGenConfig,
    *,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_normals = _smooth_path_normals(
        _nearest_surface_normals(part, points, config),
        max(int(config.sweep_surface_normal_smoothing_iterations), 0),
        closed=closed,
    )
    strength = float(np.clip(config.sweep_surface_normal_constraint_strength, 0.0, 1.0))
    guide_weights = _surface_guide_weights(segment_geometry, len(points), closed=closed)
    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)
    normal = mesh_normals[0] - tangents[0] * np.dot(mesh_normals[0], tangents[0])
    if np.linalg.norm(normal) < 1e-10:
        normal = _fallback_normal(tangents[0], None)
    normal = normal / max(np.linalg.norm(normal), 1e-15)
    normals[0] = normal
    binormals[0] = _normalised_cross(tangents[0], normal)

    for ind in range(1, len(points)):
        transported = _rotate_between_vectors(normals[ind - 1], tangents[ind - 1], tangents[ind])
        surface = mesh_normals[ind] - tangents[ind] * np.dot(mesh_normals[ind], tangents[ind])
        if np.linalg.norm(surface) < 1e-10:
            surface = transported
        surface = surface / max(np.linalg.norm(surface), 1e-15)
        if np.dot(surface, transported) < 0.0:
            surface *= -1.0
        local_strength = strength * guide_weights[ind]
        normal = (1.0 - local_strength) * transported + local_strength * surface
        if np.linalg.norm(normal) < 1e-10:
            normal = transported
        normal = normal / max(np.linalg.norm(normal), 1e-15)
        binormal = _normalised_cross(tangents[ind], normal)
        normal = _normalised_cross(binormal, tangents[ind])
        normals[ind] = normal
        binormals[ind] = binormal
    return normals, binormals


def _surface_guide_weights(
    segment_geometry: list[WirePathGeometry],
    point_count: int,
    *,
    closed: bool,
) -> np.ndarray:
    """Blend the surface-normal guide out across offset and free-space spans."""

    surface = np.asarray(
        [WirePathGeometry(value) == WirePathGeometry.SURFACE for value in segment_geometry],
        dtype=float,
    )
    expected = point_count if closed else max(point_count - 1, 0)
    if len(surface) != expected:
        raise ValueError(
            f"segment_geometry must contain one entry per framed centerline segment ({len(surface)} != {expected})"
        )
    if point_count == 0:
        return np.zeros(0, dtype=float)
    if closed:
        return 0.5 * (surface + np.roll(surface, 1))
    weights = np.zeros(point_count, dtype=float)
    if len(surface):
        weights[0] = surface[0]
        weights[-1] = surface[-1]
    if point_count > 2:
        weights[1:-1] = 0.5 * (surface[:-1] + surface[1:])
    return weights


@dataclass(frozen=True, slots=True)
class _AnalyticCylinderSurface:
    """Parameterization-free closest-point constraint for a finite cylinder."""

    center: np.ndarray
    axis: np.ndarray
    radius: float
    axial_min: float
    axial_max: float

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        offsets = points - self.center
        axial = offsets @ self.axis
        radial = offsets - axial[:, None] * self.axis
        lengths = np.linalg.norm(radial, axis=1)
        invalid = lengths <= 1e-15
        if np.any(invalid):
            fallback = _fallback_normal(self.axis, None)
            radial[invalid] = fallback
            lengths[invalid] = 1.0
        normals = radial / lengths[:, None]
        projected = self.center + axial[:, None] * self.axis + self.radius * normals
        return projected, normals

    def within_domain(self, points: np.ndarray, tolerance: float) -> np.ndarray:
        axial = (np.asarray(points, dtype=float) - self.center) @ self.axis
        return (axial >= self.axial_min - tolerance) & (axial <= self.axial_max + tolerance)

    def log_map(self, origin: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Map nearby cylinder points to exact local geodesic coordinates."""

        origin = np.asarray(origin, dtype=float)
        points = np.asarray(points, dtype=float)
        _origin_projected, origin_normals = self.project(origin[None, :])
        _points_projected, point_normals = self.project(points)
        origin_normal = origin_normals[0]
        sine = np.einsum(
            "j,ij->i",
            self.axis,
            np.cross(np.broadcast_to(origin_normal, point_normals.shape), point_normals),
        )
        cosine = point_normals @ origin_normal
        circumferential = self.radius * np.arctan2(sine, cosine)
        axial = (points - origin) @ self.axis
        return np.column_stack((circumferential, axial))

    def exp_map(self, origin: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """Map local cylinder geodesic coordinates back into physical 3-D."""

        origin = np.asarray(origin, dtype=float)
        coordinates = np.asarray(coordinates, dtype=float)
        _origin_projected, origin_normals = self.project(origin[None, :])
        origin_normal = origin_normals[0]
        angles = coordinates[:, 0] / self.radius
        circumferential_direction = np.cross(self.axis, origin_normal)
        normals = np.cos(angles)[:, None] * origin_normal + np.sin(angles)[:, None] * circumferential_direction
        origin_axial = float(np.dot(origin - self.center, self.axis))
        return self.center + (origin_axial + coordinates[:, 1])[:, None] * self.axis + self.radius * normals

    def geodesic_midpoint(self, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = self.log_map(start, np.asarray(end, dtype=float)[None, :])
        midpoint = self.exp_map(start, 0.5 * coordinates)
        projected, normals = self.project(midpoint)
        return projected[0], normals[0]


def _analytic_cylinder_surface_constraint(
    part: CoilPart,
    config: CoilGenConfig,
) -> _AnalyticCylinderSurface | None:
    """Return a verified analytic cylinder matching the generated support mesh."""

    if not bool(config.surface_is_cylinder_flag):
        return None
    if _normalised_text(config.coil_mesh_file) != "create cylinder mesh":
        return None
    parameters = tuple(config.cylinder_mesh_parameter_list)
    if len(parameters) < 8 or part.mesh.n_vertices < 3:
        return None
    radius = float(parameters[1])
    if not np.isfinite(radius) or radius <= 0.0:
        return None
    cylinder_axis = rotation_matrix_by_vector(
        np.asarray(parameters[4:7], dtype=float),
        float(parameters[7]),
    ) @ np.array([0.0, 0.0, 1.0])
    generated_axis = np.asarray(config.generated_mesh_rotation_axis, dtype=float)
    axis = (
        rotation_matrix_by_vector(
            generated_axis,
            float(config.generated_mesh_rotation_angle),
        )
        @ cylinder_axis
    )
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-15:
        return None
    axis = axis / axis_length
    vertices = np.asarray(part.mesh.vertices, dtype=float)
    center = np.mean(vertices, axis=0)
    offsets = vertices - center
    axial = offsets @ axis
    radial = offsets - axial[:, None] * axis
    radial_lengths = np.linalg.norm(radial, axis=1)
    verification_tolerance = max(0.01 * radius, 1e-8)
    if (
        not np.all(np.isfinite(radial_lengths))
        or float(np.max(np.abs(radial_lengths - radius))) > verification_tolerance
    ):
        return None
    return _AnalyticCylinderSurface(
        center=center,
        axis=axis,
        radius=radius,
        axial_min=float(np.min(axial)),
        axial_max=float(np.max(axial)),
    )


def _nearest_surface_normals(
    part: CoilPart,
    points: np.ndarray,
    config: CoilGenConfig,
) -> np.ndarray:
    analytic_surface = _analytic_cylinder_surface_constraint(part, config)
    if analytic_surface is not None:
        _projected, normals = analytic_surface.project(points)
        return normals
    if part.mesh.n_faces == 0:
        return np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
    _projected, normals = _SurfaceProjector(part.mesh).project(points)
    return normals


class _SurfaceProjector:
    """Closest-triangle projection with barycentrically interpolated normals."""

    def __init__(self, mesh: Mesh):
        self.vertices = np.asarray(mesh.vertices, dtype=float)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.triangles = self.vertices[self.faces]
        self.face_normals = (
            np.asarray(mesh.face_normals, dtype=float)
            if mesh.face_normals is not None and len(mesh.face_normals) == len(self.faces)
            else compute_face_normals(mesh)
        )
        self.vertex_normals = (
            np.asarray(mesh.vertex_normals, dtype=float)
            if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(self.vertices)
            else compute_vertex_normals(mesh)
        )
        self.vertex_faces: list[list[int]] = [[] for _ in range(len(self.vertices))]
        edge_counts: dict[tuple[int, int], int] = {}
        for face_index, face in enumerate(self.faces):
            for vertex_index in face:
                self.vertex_faces[int(vertex_index)].append(face_index)
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = (min(int(start), int(end)), max(int(start), int(end)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        self.boundary_edges = {edge for edge, count in edge_counts.items() if count == 1}
        self.vertex_tree = None
        self.centroid_tree = None
        try:
            from scipy.spatial import cKDTree

            self.vertex_tree = cKDTree(self.vertices)
            self.centroid_tree = cKDTree(np.mean(self.triangles, axis=1))
        except ImportError:
            pass

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected, normals, _face_normals = self.project_with_face_normals(points)
        return projected, normals

    def project_with_face_normals(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        projected, normals, closest_face_normals, _face_indices, _barycentric = self.project_with_details(points)
        return projected, normals, closest_face_normals

    def project_with_details(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        projected = np.zeros_like(points)
        normals = np.zeros_like(points)
        closest_face_normals = np.zeros_like(points)
        closest_face_indices = np.full(len(points), -1, dtype=np.int64)
        closest_barycentric = np.zeros((len(points), 3), dtype=float)
        vertex_candidates = None
        centroid_candidates = None
        if self.vertex_tree is not None and len(points) > 1:
            vertex_count = min(8, len(self.vertices))
            face_count = min(8, len(self.faces))
            _distances, vertex_candidates = self.vertex_tree.query(points, k=vertex_count)
            _distances, centroid_candidates = self.centroid_tree.query(points, k=face_count)
            if vertex_count == 1:
                vertex_candidates = np.asarray(vertex_candidates)[:, None]
            if face_count == 1:
                centroid_candidates = np.asarray(centroid_candidates)[:, None]
        for point_index, point in enumerate(points):
            if vertex_candidates is None:
                face_index, closest, barycentric = self._closest_face(point)
            else:
                candidate_faces = {int(index) for index in centroid_candidates[point_index]}
                for vertex_index in vertex_candidates[point_index]:
                    candidate_faces.update(self.vertex_faces[int(vertex_index)])
                face_index, closest, barycentric = self._closest_face_from_candidates(point, candidate_faces)
            normal = barycentric @ self.vertex_normals[self.faces[face_index]]
            normal_length = float(np.linalg.norm(normal))
            if normal_length <= 1e-15:
                normal = self.face_normals[face_index]
                normal_length = float(np.linalg.norm(normal))
            projected[point_index] = closest
            normals[point_index] = normal / normal_length if normal_length > 1e-15 else np.array([0.0, 0.0, 1.0])
            closest_face_normals[point_index] = self.face_normals[face_index]
            closest_face_indices[point_index] = face_index
            closest_barycentric[point_index] = barycentric
        return projected, normals, closest_face_normals, closest_face_indices, closest_barycentric

    def _closest_face(self, point: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        candidate_faces: set[int] = set()
        vertex_count = min(8, len(self.vertices))
        face_count = min(8, len(self.faces))
        if self.vertex_tree is not None:
            _distances, vertex_indices = self.vertex_tree.query(point, k=vertex_count)
            for vertex_index in np.atleast_1d(vertex_indices):
                candidate_faces.update(self.vertex_faces[int(vertex_index)])
            _distances, face_indices = self.centroid_tree.query(point, k=face_count)
            candidate_faces.update(int(index) for index in np.atleast_1d(face_indices))
        else:
            vertex_distances = np.linalg.norm(self.vertices - point, axis=1)
            nearest_vertices = np.argpartition(vertex_distances, vertex_count - 1)[:vertex_count]
            for vertex_index in nearest_vertices:
                candidate_faces.update(self.vertex_faces[int(vertex_index)])
            centroid_distances = np.linalg.norm(np.mean(self.triangles, axis=1) - point, axis=1)
            nearest_faces = np.argpartition(centroid_distances, face_count - 1)[:face_count]
            candidate_faces.update(int(index) for index in nearest_faces)
        if not candidate_faces:
            candidate_faces.add(0)
        return self._closest_face_from_candidates(point, candidate_faces)

    def _closest_face_from_candidates(
        self,
        point: np.ndarray,
        candidate_faces: set[int],
    ) -> tuple[int, np.ndarray, np.ndarray]:
        best_face = -1
        best_point = np.zeros(3, dtype=float)
        best_barycentric = np.zeros(3, dtype=float)
        best_distance_sq = np.inf
        for face_index in candidate_faces:
            closest, barycentric = _closest_point_and_barycentric(point, self.triangles[face_index])
            distance_sq = float(np.dot(point - closest, point - closest))
            if distance_sq < best_distance_sq:
                best_face = int(face_index)
                best_point = closest
                best_barycentric = barycentric
                best_distance_sq = distance_sq
        return best_face, best_point, best_barycentric


def _closest_point_and_barycentric(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Euclidean closest triangle point and its barycentric weights."""

    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0])

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])

    denominator = va + vb + vc
    if abs(denominator) <= 1e-30:
        return a, np.array([1.0, 0.0, 0.0])
    inverse = 1.0 / denominator
    v = vb * inverse
    w = vc * inverse
    return a + ab * v + ac * w, np.array([1.0 - v - w, v, w])


def _fallback_normal(tangent: np.ndarray, previous_normal: np.ndarray | None) -> np.ndarray:
    if previous_normal is not None:
        normal = previous_normal - tangent * np.dot(previous_normal, tangent)
        if np.linalg.norm(normal) > 1e-10:
            return normal
    axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, tangent)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    return axis - tangent * np.dot(axis, tangent)


def _smooth_path_normals(normals: np.ndarray, iterations: int, *, closed: bool = False) -> np.ndarray:
    out = np.asarray(normals, dtype=float).copy()
    for _ in range(max(int(iterations), 0)):
        if len(out) <= 2:
            break
        if closed:
            smoothed = 0.25 * np.roll(out, 1, axis=0) + 0.5 * out + 0.25 * np.roll(out, -1, axis=0)
        else:
            smoothed = out.copy()
            smoothed[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        lengths = np.linalg.norm(smoothed, axis=1)
        smoothed[lengths > 0] /= lengths[lengths > 0, None]
        out = smoothed
    return out


def _enforce_periodic_path_frames(
    points: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distribute closed-loop frame holonomy so the final seam is periodic."""

    if len(points) < 3:
        binormals = np.asarray([_normalised_cross(tangent, normal) for tangent, normal in zip(tangents, normals)])
        return normals, binormals
    closing_normal = _rotate_between_vectors(normals[-1], tangents[-1], tangents[0])
    residual_angle = float(
        np.arctan2(
            np.dot(tangents[0], np.cross(closing_normal, normals[0])),
            np.dot(closing_normal, normals[0]),
        )
    )
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    denominator = float(cumulative[-1])
    fractions = cumulative / denominator if denominator > 1e-15 else np.zeros(len(points), dtype=float)
    corrected_normals = np.zeros_like(normals)
    corrected_binormals = np.zeros_like(normals)
    for ind, fraction in enumerate(fractions):
        normal = _rotate_about_axis(normals[ind], tangents[ind], residual_angle * float(fraction))
        binormal = _normalised_cross(tangents[ind], normal)
        corrected_binormals[ind] = binormal
        corrected_normals[ind] = _normalised_cross(binormal, tangents[ind])
    return corrected_normals, corrected_binormals


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-15)
    vector = np.asarray(vector, dtype=float)
    rotated = (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )
    return rotated / max(float(np.linalg.norm(rotated)), 1e-15)


def _normalised_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cross = np.cross(a, b)
    length = float(np.linalg.norm(cross))
    if length <= 1e-15:
        return _fallback_normal(np.asarray(a, dtype=float), None)
    return cross / length


def _rotate_between_vectors(vector: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.asarray(source, dtype=float)
    target_norm = np.asarray(target, dtype=float)
    source_norm = source_norm / max(float(np.linalg.norm(source_norm)), 1e-15)
    target_norm = target_norm / max(float(np.linalg.norm(target_norm)), 1e-15)
    axis = np.cross(source_norm, target_norm)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(source_norm, target_norm), -1.0, 1.0))
    if axis_norm <= 1e-15:
        if dot > 0.0:
            return np.asarray(vector, dtype=float)
        axis = _fallback_normal(source_norm, None)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-15)
        angle = np.pi
    else:
        axis = axis / axis_norm
        angle = float(np.arccos(dot))
    vec = np.asarray(vector, dtype=float)
    rotated = (
        vec * np.cos(angle) + np.cross(axis, vec) * np.sin(angle) + axis * np.dot(axis, vec) * (1.0 - np.cos(angle))
    )
    return rotated / max(float(np.linalg.norm(rotated)), 1e-15)


def _polyline_turn_angles_deg(points: np.ndarray, *, closed: bool = False) -> np.ndarray:
    if len(points) <= 2:
        return np.zeros(0, dtype=float)
    indices = range(len(points)) if closed else range(1, len(points) - 1)
    return np.asarray(
        [
            _turn_angle_deg(points[(ind - 1) % len(points)], points[ind], points[(ind + 1) % len(points)])
            for ind in indices
        ],
        dtype=float,
    )


def _turn_angle_deg(previous: np.ndarray, current: np.ndarray, following: np.ndarray) -> float:
    a = current - previous
    b = following - current
    a_len = float(np.linalg.norm(a))
    b_len = float(np.linalg.norm(b))
    if a_len <= 1e-15 or b_len <= 1e-15:
        return 0.0
    return _vector_angle_deg(a / a_len, b / b_len)


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_len = float(np.linalg.norm(a))
    b_len = float(np.linalg.norm(b))
    if a_len <= 1e-15 or b_len <= 1e-15:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (a_len * b_len), -1.0, 1.0))))


def _cross_section_envelope_radius(cross_section: np.ndarray) -> float:
    points = np.asarray(cross_section, dtype=float)
    if len(points) == 0:
        return 0.0
    return float(np.max(np.linalg.norm(points, axis=1)))


def _capsule_clearance_diagnostics(
    points: np.ndarray,
    required_clearance: float,
    conductor_diameter: float,
    *,
    closed: bool,
) -> dict[str, float | int | bool | str]:
    starts = np.asarray(points, dtype=float)
    ends = np.roll(starts, -1, axis=0) if closed else starts[1:]
    starts = starts if closed else starts[:-1]
    segment_count = len(starts)
    broad_phase_distance = max(float(required_clearance), float(conductor_diameter), 0.0)
    candidates = _segment_bvh_candidate_pairs(starts, ends, broad_phase_distance)
    segment_lengths = np.linalg.norm(ends - starts, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    # A finely sampled smooth tube necessarily has overlapping capsules for
    # several neighbours on either side. Exclude a four-diameter arclength
    # neighbourhood; bend-radius checks govern this local geometry, while the
    # BVH remains responsible for topologically remote route collisions.
    local_exclusion_length = 4.0 * broad_phase_distance
    min_distance = np.inf
    min_i = -1
    min_j = -1
    violation_count = 0
    intersection_count = 0
    evaluated_count = 0
    for i, j in candidates:
        i = int(i)
        j = int(j)
        if j - i <= 1 or (closed and i == 0 and j == segment_count - 1):
            continue
        # Measure only the route lying *between* the two segments. Including
        # half of each tested segment incorrectly classifies segments i and
        # i+2 as remote when either neighbouring segment is long, even though
        # only one short continuation segment separates them.
        arclength_separation = max(float(cumulative[j] - cumulative[i + 1]), 0.0)
        if closed:
            wrap_separation = max(float(total_length - (cumulative[j + 1] - cumulative[i])), 0.0)
            arclength_separation = min(arclength_separation, wrap_separation)
        if arclength_separation <= local_exclusion_length:
            # Neighbouring capsules along the same physical tube overlap by
            # construction. Local folding is governed by the curvature/radius
            # constraint; collision checks are for topologically remote segments.
            continue
        evaluated_count += 1
        distance = _segment_distance(starts[i], ends[i], starts[j], ends[j])
        if distance < min_distance:
            min_distance = distance
            min_i = i
            min_j = j
        if distance < required_clearance:
            violation_count += 1
        if distance < conductor_diameter:
            intersection_count += 1
    finite_min = float(min_distance) if np.isfinite(min_distance) else 0.0
    return {
        "sweep_capsule_broad_phase": "bvh",
        "sweep_capsule_candidate_pair_count": int(len(candidates)),
        "sweep_capsule_evaluated_pair_count": int(evaluated_count),
        "sweep_capsule_min_centerline_distance_m": finite_min,
        "sweep_capsule_min_segment_a": int(min_i),
        "sweep_capsule_min_segment_b": int(min_j),
        "sweep_capsule_clearance_violation_count": int(violation_count),
        "sweep_capsule_self_intersection_count": int(intersection_count),
        "sweep_capsule_clearance_violation": bool(violation_count > 0),
        "sweep_capsule_self_intersection": bool(intersection_count > 0),
    }


class _SegmentBvhNode:
    __slots__ = ("indices", "minimum", "maximum", "left", "right")

    def __init__(
        self,
        indices: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
        left: "_SegmentBvhNode | None" = None,
        right: "_SegmentBvhNode | None" = None,
    ) -> None:
        self.indices = indices
        self.minimum = minimum
        self.maximum = maximum
        self.left = left
        self.right = right

    @property
    def leaf(self) -> bool:
        return self.left is None


def _segment_bvh_candidate_pairs(
    starts: np.ndarray,
    ends: np.ndarray,
    distance: float,
    *,
    leaf_size: int = 12,
) -> np.ndarray:
    """Return segment pairs whose AABBs are within ``distance`` using a BVH."""

    if len(starts) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    segment_minima = np.minimum(starts, ends)
    segment_maxima = np.maximum(starts, ends)
    centroids = 0.5 * (segment_minima + segment_maxima)

    def build(indices: np.ndarray) -> _SegmentBvhNode:
        minimum = np.min(segment_minima[indices], axis=0)
        maximum = np.max(segment_maxima[indices], axis=0)
        if len(indices) <= leaf_size:
            return _SegmentBvhNode(indices, minimum, maximum)
        spread = np.ptp(centroids[indices], axis=0)
        axis = int(np.argmax(spread))
        order = indices[np.argsort(centroids[indices, axis], kind="stable")]
        middle = len(order) // 2
        return _SegmentBvhNode(
            np.zeros(0, dtype=np.int64),
            minimum,
            maximum,
            build(order[:middle]),
            build(order[middle:]),
        )

    def boxes_within_distance(first: _SegmentBvhNode, second: _SegmentBvhNode) -> bool:
        separation = np.maximum(0.0, np.maximum(first.minimum - second.maximum, second.minimum - first.maximum))
        return bool(np.dot(separation, separation) <= distance * distance)

    pairs: list[tuple[int, int]] = []

    def collect(first: _SegmentBvhNode, second: _SegmentBvhNode) -> None:
        if not boxes_within_distance(first, second):
            return
        if first is second:
            if first.leaf:
                for local_i in range(len(first.indices)):
                    for local_j in range(local_i + 1, len(first.indices)):
                        i = int(first.indices[local_i])
                        j = int(first.indices[local_j])
                        pairs.append((min(i, j), max(i, j)))
                return
            collect(first.left, first.left)
            collect(first.left, first.right)
            collect(first.right, first.right)
            return
        if first.leaf and second.leaf:
            for i in first.indices:
                for j in second.indices:
                    start = int(min(i, j))
                    end = int(max(i, j))
                    if start != end:
                        pairs.append((start, end))
            return
        first_size = (
            len(first.indices) if first.leaf else np.dot(first.maximum - first.minimum, first.maximum - first.minimum)
        )
        second_size = (
            len(second.indices)
            if second.leaf
            else np.dot(second.maximum - second.minimum, second.maximum - second.minimum)
        )
        if not first.leaf and (second.leaf or first_size >= second_size):
            collect(first.left, second)
            collect(first.right, second)
        else:
            collect(first, second.left)
            collect(first, second.right)

    root = build(np.arange(len(starts), dtype=np.int64))
    collect(root, root)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.asarray(pairs, dtype=np.int64), axis=0)


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    # Closest distance between two 3D segments; adapted from the standard
    # clamped two-line solution and kept local to avoid geometry dependencies.
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-15
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = 0.0 if denom <= eps else np.clip((b * f - c * e) / denom, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    closest_1 = p1 + d1 * s
    closest_2 = p2 + d2 * t
    return float(np.linalg.norm(closest_1 - closest_2))


def _normalised_text(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _polygon_area(points: np.ndarray) -> float:
    return abs(_polygon_signed_area(points))


def _polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    cross_sum = float(np.sum(cross))
    if abs(cross_sum) <= 1e-30:
        return np.mean(points, axis=0)
    return np.asarray(
        [
            np.sum((points[:, 0] + following[:, 0]) * cross) / (3.0 * cross_sum),
            np.sum((points[:, 1] + following[:, 1]) * cross) / (3.0 * cross_sum),
        ],
        dtype=float,
    )


def _orient_cross_section_counterclockwise(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points if _polygon_signed_area(points) > 0.0 else points[::-1].copy()


def _validated_cross_section(points: np.ndarray) -> np.ndarray:
    """Validate the convex profile assumptions used by the loft and cap fan."""

    try:
        profile = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Conductor cross-section points must be numeric.") from exc
    if profile.ndim != 2:
        raise ValueError("Conductor cross-section points must have shape (n, 2) or (2, n).")
    if profile.shape[0] == 2 and profile.shape[1] != 2:
        profile = profile.T
    if profile.shape[1:] != (2,) or len(profile) < 3:
        raise ValueError("Conductor cross-section requires at least three 2D points.")
    if not np.all(np.isfinite(profile)):
        raise ValueError("Conductor cross-section contains non-finite coordinates.")
    if np.array_equal(profile[0], profile[-1]):
        profile = profile[:-1]
    if len(profile) < 3:
        raise ValueError("Conductor cross-section requires at least three distinct points.")

    scale = max(float(np.ptp(profile[:, 0])), float(np.ptp(profile[:, 1])), 1e-15)
    length_tolerance = max(1e-12 * scale, 1e-15)
    edges = np.roll(profile, -1, axis=0) - profile
    if np.any(np.linalg.norm(edges, axis=1) <= length_tolerance):
        raise ValueError("Conductor cross-section contains a zero-length or repeated edge.")

    if _polygon_has_self_intersections(profile, length_tolerance):
        raise ValueError("Conductor cross-section must be a simple, non-self-intersecting polygon.")
    area = _polygon_area(profile)
    if area <= max(1e-12 * scale**2, 1e-30):
        raise ValueError("Conductor cross-section has zero or numerically negligible area.")

    following_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following_edges[:, 1] - edges[:, 1] * following_edges[:, 0]
    turn_tolerance = max(1e-12 * scale**2, 1e-30)
    if np.any(np.abs(turns) <= turn_tolerance) or not (
        np.all(turns > turn_tolerance) or np.all(turns < -turn_tolerance)
    ):
        raise ValueError(
            "Conductor cross-section must be strictly convex; concave or collinear profiles "
            "require a general polygon triangulator."
        )
    return _orient_cross_section_counterclockwise(profile)


def _polygon_has_self_intersections(points: np.ndarray, tolerance: float) -> bool:
    count = len(points)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first == second or first_next == second or second_next == first:
                continue
            if _segments_intersect_2d(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
                tolerance,
            ):
                return True
    return False


def _segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float,
) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        pq = q - p
        pr = r - p
        return float(pq[0] * pr[1] - pq[1] * pr[0])

    ab_c = orientation(a, b, c)
    ab_d = orientation(a, b, d)
    cd_a = orientation(c, d, a)
    cd_b = orientation(c, d, b)
    return bool(
        ((ab_c > tolerance and ab_d < -tolerance) or (ab_c < -tolerance and ab_d > tolerance))
        and ((cd_a > tolerance and cd_b < -tolerance) or (cd_a < -tolerance and cd_b > tolerance))
    )
