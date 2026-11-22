"""Manufacturing artifact export and export-readiness checks; no path construction."""

from __future__ import annotations
from .mesh_inspection import mesh_manufacturing_diagnostics
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from .attribution import AUTHOR_CREDIT, SOFTWARE_NAME
from .conductor_geometry import conductor_trace, trace_segment_geometry
from .geometry import compute_face_normals
from .models import CoilGenResult, CoilPart, Mesh, SweptBody


_WIRE_TRACE_STEP_MIN_SEGMENT_LENGTH_M = 1e-12


class SweptBodyExportError(ValueError):
    """A swept body failed manufacturing readiness checks for export."""


def manufacturing_diagnostics_rows(result: CoilGenResult) -> list[dict[str, float | int | bool | str]]:
    """Return one diagnostic row per independently exported swept body."""

    rows: list[dict[str, float | int | bool | str]] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        bodies = _part_swept_bodies(part)
        if bodies:
            for body_index, body in enumerate(bodies, start=1):
                diagnostics = _swept_body_diagnostics(part, body)
                row: dict[str, float | int | bool | str] = {
                    "part_index": part_ind,
                    "body_index": body_index,
                    "sweep_source": "contour" if body.contour_index is not None else "wire_path",
                    "contour_index": int(body.contour_index or 0),
                    "turn_multiplicity": int(body.turn_multiplicity),
                }
                row.update(diagnostics)
                rows.append(row)
        elif part.manufacturing_diagnostics:
            row = {"part_index": part_ind}
            row.update(part.manufacturing_diagnostics)
            rows.append(row)
        else:
            row = {"part_index": part_ind}
            row.update({"swept_body_created": False, "manufacturing_ok": False})
            rows.append(row)
    return rows


def _part_swept_bodies(part: CoilPart) -> list[SweptBody]:
    swept_bodies = list(getattr(part, "swept_bodies", ()))
    if swept_bodies:
        return swept_bodies
    if part.swept_body is None:
        return []
    return [SweptBody(mesh=part.swept_body, diagnostics=part.manufacturing_diagnostics)]


def _swept_body_diagnostics(part: CoilPart, body: SweptBody) -> dict[str, float | int | bool | str]:
    diagnostics = dict(body.diagnostics) if body.diagnostics else mesh_manufacturing_diagnostics(body.mesh)
    proxy = replace(
        part,
        swept_body=body.mesh,
        swept_bodies=[],
        manufacturing_diagnostics=diagnostics,
    )
    _finalize_manufacturing_diagnostics(proxy)
    body.diagnostics = dict(proxy.manufacturing_diagnostics)
    if body.contour_index is None:
        part.manufacturing_diagnostics = dict(body.diagnostics)
    return body.diagnostics


def save_manufacturing_diagnostics(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil_body",
) -> list[Path]:
    """Save manufacturing checks, routing decisions and lift provenance."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = manufacturing_diagnostics_rows(result)
    json_path = output / f"{prefix}_manufacturing_diagnostics.json"
    csv_path = output / f"{prefix}_manufacturing_diagnostics.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_rows_csv(csv_path, rows)
    files = [json_path, csv_path]
    for part_index, part in enumerate(result.coil_parts, 1):
        joint_routing = getattr(part, "joint_route_diagnostics", {})
        if joint_routing:
            path = output / f"{prefix}_part{part_index:03d}_joint_routing.json"
            path.write_text(
                json.dumps(
                    {"schema_version": 1, "part_index": part_index, "diagnostics": joint_routing},
                    indent=2,
                    default=_provenance_json_value,
                ),
                encoding="utf-8",
            )
            files.append(path)
        provenance = getattr(part, "normal_shift_provenance", {})
        if not provenance:
            continue
        arrays = {key: value for key, value in provenance.items() if isinstance(value, np.ndarray)}
        metadata = {
            "schema_version": 1,
            "part_index": part_index,
            "length_unit": "m",
            "diagnostics": part.normal_shift_diagnostics,
            **{key: value for key, value in provenance.items() if key not in arrays},
        }
        # Numeric arrays plus JSON metadata load with allow_pickle=False. Keep
        # the pre-cleanup point indices independent of the final wire CSV.
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, default=_provenance_json_value))
        path = output / f"{prefix}_part{part_index:03d}_normal_shift_provenance.npz"
        np.savez_compressed(path, **arrays)
        files.append(path)
    return files


def _provenance_json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported normal-shift provenance value: {type(value).__name__}")


def wire_trace_rows(result: CoilGenResult, *, source: str = "auto") -> list[dict[str, int | float | str]]:
    """Return final physical centroid points and interval geometry in SI units."""

    rows: list[dict[str, int | float | str]] = []
    circuit_columns = any(getattr(part, "circuit_id", "") for part in result.coil_parts)
    for part_ind, part in enumerate(result.coil_parts, start=1):
        points, kind = conductor_trace(part, source=source)
        if points is None:
            continue
        geometry = trace_segment_geometry(part, kind)
        segment_lengths = np.zeros(len(points), dtype=float)
        if len(points) > 1:
            segment_lengths[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_lengths = np.cumsum(segment_lengths)
        for point_ind, (point, segment_length, cumulative_length) in enumerate(
            zip(points, segment_lengths, cumulative_lengths),
        ):
            segment_geometry = geometry[point_ind - 1].value if point_ind > 0 and point_ind - 1 < len(geometry) else ""
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
                    **(
                        {
                            "circuit_id": getattr(part, "circuit_id", ""),
                            "source_group_keys": ";".join(getattr(part, "source_group_keys", ())),
                            "source_support_indices": ";".join(map(str, getattr(part, "source_support_indices", ()))),
                        }
                        if circuit_columns
                        else {}
                    ),
                }
            )
    return rows


def save_wire_trace_as_csv(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
    source: str = "auto",
) -> list[Path]:
    """Save routed wire centerlines using the configured CSV schema."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = getattr(result, "input_data", None)
    csv_format = str(getattr(config, "wire_trace_csv_format", "detailed")).strip().lower()
    if csv_format == "xyz_no_header":
        return _save_wire_paths_as_xyz_csv(result, output, prefix=prefix, source=source)

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
    if any(getattr(part, "circuit_id", "") for part in result.coil_parts):
        fieldnames += ["circuit_id", "source_group_keys", "source_support_indices"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wire_trace_rows(result, source=source))
    return [path]


def save_wire_trace_artifacts(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
    source: str = "auto",
) -> list[Path]:
    """Save contour STEP wires plus routed-path CSV/STEP geometry when available."""

    return [
        *save_wire_trace_as_csv(result, output_dir, prefix=prefix, source=source),
        *save_contour_traces_as_step(result, output_dir, prefix=prefix),
        *save_wire_trace_as_step(result, output_dir, prefix=prefix, source=source),
    ]


def save_contour_traces_as_step(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
) -> list[Path]:
    """Save all unswept contour XYZ sequences as independent STEP curve paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contour_parts: list[tuple[int, list[tuple[str, np.ndarray]]]] = []
    for part_index, part in enumerate(result.coil_parts, start=1):
        traces = [
            (f"{prefix}_part{part_index:03d}_contour{contour_index:04d}", np.asarray(contour.v, dtype=float))
            for contour_index, contour in enumerate(part.contour_lines, start=1)
            if len(contour.v) >= 2
        ]
        if traces:
            contour_parts.append((part_index, traces))

    multiple = len(contour_parts) > 1
    files: list[Path] = []
    for part_index, traces in contour_parts:
        suffix = f"_part{part_index:03d}" if multiple else ""
        path = output / f"{prefix}_contour_traces{suffix}.step"
        product_name = f"{prefix}_contour_traces{suffix}"
        files.append(
            _save_xyz_polylines_as_step(
                traces,
                path,
                product_name=product_name,
                part_index=part_index,
                description="CoilGen contour centerline curve paths; piecewise-linear; units=m",
            )
        )
    return files


def save_wire_trace_as_step(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
    source: str = "auto",
) -> list[Path]:
    """Save each routed XYZ centerline as a piecewise-linear STEP curve path.

    Coordinates remain in CoilGen's internal SI metres. Consecutive points are
    joined by straight STEP ``TRIMMED_CURVE`` segments; no interpolation,
    smoothing, sweep, surface, or solid is introduced by this export.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    routed_parts = _routed_wire_parts(result, source=source)
    multiple = len(routed_parts) > 1
    files: list[Path] = []
    for part_index, points in routed_parts:
        suffix = f"_part{part_index:03d}" if multiple else ""
        path = output / f"{prefix}_wire_path{suffix}.step"
        wire_name = f"{prefix}_wire_path{suffix}"
        files.append(_save_xyz_polyline_as_step(points, path, wire_name=wire_name, part_index=part_index))
    return files


def _save_wire_paths_as_xyz_csv(
    result: CoilGenResult, output: Path, *, prefix: str, source: str = "auto"
) -> list[Path]:
    """Write one headerless ordered X,Y,Z file per connected routed part."""

    routed_parts = _routed_wire_parts(result, source=source)

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


def _routed_wire_parts(result: CoilGenResult, *, source: str = "auto") -> list[tuple[int, np.ndarray]]:
    routed_parts: list[tuple[int, np.ndarray]] = []
    for part_index, part in enumerate(result.coil_parts, start=1):
        points, _kind = conductor_trace(part, source=source)
        if points is None:
            continue
        if not np.all(np.isfinite(points)):
            raise ValueError(f"Wire path for coil part {part_index} contains non-finite coordinates.")
        routed_parts.append((part_index, points))
    return routed_parts


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
    # A bounding capsule can overlap even when oriented rectangular copper
    # meets the required insulation gap. Retain the raw screens for review.
    exact = diagnostics.get("exact_collision_completed", False)
    copper_failure = diagnostics.get("copper_clearance_failure", "")
    if copper_failure and copper_failure != "swept_mesh_self_intersection":
        reasons.append(copper_failure)
    if not exact and bool(diagnostics.get("sweep_route_clearance_violation", False)):
        reasons.append("route_clearance_violation")
    if not exact and bool(diagnostics.get("sweep_capsule_clearance_violation", False)):
        reasons.append("sweep_clearance_violation")
    if not exact and bool(diagnostics.get("sweep_capsule_self_intersection", False)):
        reasons.append("sweep_self_intersection")
    if bool(diagnostics.get("sweep_mesh_self_intersection", False)):
        reasons.append("swept_mesh_self_intersection")
    if not exact and bool(diagnostics.get("sweep_self_intersection_skipped", False)):
        reasons.append("self_intersection_check_skipped")
    if bool(diagnostics.get("sweep_min_bend_radius_violation", False)) or bool(
        diagnostics.get("sweep_final_min_bend_radius_violation", False)
    ):
        reasons.append("minimum_bend_radius_violation")
    # Ring repair can remove near-redundant samples. The mounted centroid is
    # the manufactured path; retain the precursor flag for diagnostics only.
    curvature_violation = diagnostics.get(
        "sweep_final_curvature_envelope_violation",
        diagnostics.get("sweep_curvature_envelope_violation", False),
    )
    if bool(curvature_violation):
        reasons.append("curvature_envelope_violation")
    if bool(diagnostics.get("sweep_final_collinear_reversal", False)):
        reasons.append("collinear_path_reversal")
    if bool(diagnostics.get("sweep_final_zero_length_interval", False)):
        reasons.append("zero_length_physical_interval")
    if bool(diagnostics.get("sweep_min_segment_length_violation", False)):
        reasons.append("minimum_segment_length_violation")
    if bool(diagnostics.get("sweep_local_fold_violation", False)):
        reasons.append("folded_sweep_quad")
    if bool(diagnostics.get("sweep_profile_jacobian_violation", False)):
        reasons.append("local_profile_inversion")
    if bool(diagnostics.get("sweep_final_frame_rotation_violation", False)):
        reasons.append("unresolved_frame_sampling")
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


def save_mesh_as_stl(mesh: Mesh, path: str | Path, *, solid_name: str = "coil_body") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = compute_face_normals(mesh) if len(mesh.faces) else np.zeros((0, 3), dtype=float)
    with path.open("w", encoding="ascii") as output:
        output.write(f"solid {solid_name}\n")
        for normal, face in zip(normals, mesh.faces):
            output.write(f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}\n")
            output.write("    outer loop\n")
            for vertex in mesh.vertices[face]:
                # Round-trip binary64 coordinates: rounding nearly coincident
                # stations after validation can change the exported topology.
                output.write(f"      vertex {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
            output.write("    endloop\n  endfacet\n")
        output.write(f"endsolid {solid_name}")
    return path


def _save_xyz_polyline_as_step(
    points: np.ndarray,
    path: str | Path,
    *,
    wire_name: str,
    part_index: int,
) -> Path:
    """Write one ordered XYZ sequence as a geometrically bounded STEP wireframe."""

    return _save_xyz_polylines_as_step(
        [(wire_name, points)],
        path,
        product_name=wire_name,
        part_index=part_index,
        description="CoilGen routed centerline curve path; piecewise-linear; units=m",
    )


def _save_xyz_polylines_as_step(
    traces: list[tuple[str, np.ndarray]],
    path: str | Path,
    *,
    product_name: str,
    part_index: int,
    description: str,
) -> Path:
    """Write ordered XYZ sequences as OpenCascade-compatible STEP curve sets."""

    if not traces:
        raise ValueError("STEP wire export requires at least one XYZ trace.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = _StepWriter()
    product_shape, context = _write_step_product_header(writer, product_name)
    origin = writer.add("CARTESIAN_POINT('',(0.,0.,0.))")
    axis_z = writer.add("DIRECTION('',(0.,0.,1.))")
    axis_x = writer.add("DIRECTION('',(1.,0.,0.))")
    placement = writer.add(f"AXIS2_PLACEMENT_3D('',{origin},{axis_z},{axis_x})")
    curve_set_refs: list[str] = []

    for trace_name, points in traces:
        vertices, closed = _prepare_step_wire_vertices(points, part_index=part_index)
        if closed:
            vertices = np.vstack((vertices, vertices[0]))
        segment_refs: list[str] = []
        for start, end in zip(vertices[:-1], vertices[1:]):
            delta = end - start
            length = float(np.linalg.norm(delta))
            line_origin = writer.add(_step_cartesian_point(start))
            direction = writer.add(_step_direction(delta / length))
            vector = writer.add(f"VECTOR('',{direction},1.)")
            line = writer.add(f"LINE('',{line_origin},{vector})")
            trim_start = writer.add(_step_cartesian_point(start))
            trim_end = writer.add(_step_cartesian_point(end))
            segment_refs.append(
                writer.add(
                    "TRIMMED_CURVE('',"
                    f"{line},({trim_start},PARAMETER_VALUE(0.)),"
                    f"({trim_end},PARAMETER_VALUE({_step_float(length)})),.T.,.PARAMETER.)"
                )
            )
        curve_set_refs.append(
            writer.add(f"GEOMETRIC_CURVE_SET('{_step_string(trace_name)}',({','.join(segment_refs)}))")
        )

    representation_items = ",".join((placement, *curve_set_refs))
    representation = writer.add(f"SHAPE_REPRESENTATION('',({representation_items}),{context})")
    writer.add(f"SHAPE_DEFINITION_REPRESENTATION({product_shape},{representation})")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    text = "\n".join(
        [
            "ISO-10303-21;",
            "HEADER;",
            f"FILE_DESCRIPTION(('{_step_string(description)}'),'2;1');",
            f"FILE_NAME('{_step_string(path.name)}','{timestamp}',('CoilGen'),('CoilGen'),"
            f"'{_step_string(f'{SOFTWARE_NAME} - {AUTHOR_CREDIT}')}','{_step_string(SOFTWARE_NAME)}','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",
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


def _prepare_step_wire_vertices(points: np.ndarray, *, part_index: int) -> tuple[np.ndarray, bool]:
    vertices = np.asarray(points, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"XYZ trace for coil part {part_index} must have shape (n, 3).")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"XYZ trace for coil part {part_index} contains non-finite coordinates.")
    if len(vertices) < 2:
        raise ValueError(f"XYZ trace for coil part {part_index} requires at least two points for STEP export.")

    kept = [0]
    for index in range(1, len(vertices)):
        if float(np.linalg.norm(vertices[index] - vertices[kept[-1]])) > _WIRE_TRACE_STEP_MIN_SEGMENT_LENGTH_M:
            kept.append(index)
    vertices = vertices[kept]
    if len(vertices) < 2:
        raise ValueError(f"XYZ trace for coil part {part_index} has no nonzero segments for STEP export.")

    closed = bool(
        len(vertices) > 2 and float(np.linalg.norm(vertices[-1] - vertices[0])) <= _WIRE_TRACE_STEP_MIN_SEGMENT_LENGTH_M
    )
    if closed:
        vertices = vertices[:-1]
    if len(vertices) < 2:
        raise ValueError(f"XYZ trace for coil part {part_index} has no nonzero segments for STEP export.")
    return vertices, closed


def save_swept_bodies_as_stl(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil_body",
    min_clearance: float | None = None,
    fail_on_violations: bool = False,
) -> list[Path]:
    """Export each swept conductor mesh and its current-carrying surface.

    STL carries only triangulated boundaries, not analytic volume semantics.
    Each contour-mode body is written separately.  The current-carrying support
    surface is written once per coil part, followed by its grooved support body
    (``*_grooved_support.stl``) when ``support_body_mode`` produced one.
    Manufacturing diagnostics do not prevent STL serialization by default. Pass
    ``fail_on_violations=True`` to explicitly require manufacturing readiness
    (including the grooved support checks) before writing any STL.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    swept_parts = [
        (part_ind, part, _part_swept_bodies(part))
        for part_ind, part in enumerate(result.coil_parts, start=1)
        if _part_swept_bodies(part)
    ]
    for part_ind, part, bodies in swept_parts:
        for body in bodies:
            _check_swept_body_export_readiness(
                part,
                body,
                part_ind,
                min_clearance,
                fail_on_violations,
                format_name="STL",
            )
        support_body = getattr(part, "support_body", None)
        if fail_on_violations and support_body is not None and not support_body.diagnostics.get("support_body_ok"):
            raise SweptBodyExportError(
                f"Grooved support body for coil part {part_ind} failed its checks: "
                f"{support_body.diagnostics.get('support_body_failure_reasons', 'unknown')}"
            )
    for part_ind, part, bodies in swept_parts:
        for body in bodies:
            body_name = _swept_body_name(prefix, part_ind, body)
            saved.append(
                save_mesh_as_stl(
                    body.mesh,
                    output / f"{body_name}.stl",
                    solid_name=body_name,
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
        support_body = getattr(part, "support_body", None)
        if support_body is not None and len(support_body.mesh.faces):
            # Printable former: support thickened into a solid minus the conductor grooves.
            support_name = f"{prefix}_part{part_ind}_grooved_support"
            saved.append(save_mesh_as_stl(support_body.mesh, output / f"{support_name}.stl", solid_name=support_name))
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
            f"FILE_NAME('{_step_string(path.name)}','{timestamp}',('CoilGen'),('CoilGen'),"
            f"'{_step_string(f'{SOFTWARE_NAME} - {AUTHOR_CREDIT}')}','{_step_string(SOFTWARE_NAME)}','');",
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
        (part_ind, part, _part_swept_bodies(part))
        for part_ind, part in enumerate(result.coil_parts, start=1)
        if _part_swept_bodies(part)
    ]
    for part_ind, part, bodies in swept_parts:
        for body in bodies:
            _check_swept_body_export_readiness(
                part,
                body,
                part_ind,
                min_clearance,
                fail_on_violations,
                format_name="STEP",
            )
    for part_ind, _part, bodies in swept_parts:
        for body in bodies:
            body_name = _swept_body_name(prefix, part_ind, body)
            saved.append(
                save_mesh_as_step(
                    body.mesh,
                    output / f"{body_name}.step",
                    solid_name=body_name,
                    surface_mode=surface_mode,
                )
            )
    return saved


def _swept_body_name(prefix: str, part_index: int, body: SweptBody) -> str:
    name = f"{prefix}_part{part_index}"
    if body.contour_index is not None:
        name += f"_contour{body.contour_index:04d}"
    return name


def _check_swept_body_export_readiness(
    part: CoilPart,
    body: SweptBody,
    part_ind: int,
    min_clearance: float | None,
    fail_on_violations: bool,
    *,
    format_name: str,
) -> None:
    diagnostics = _swept_body_diagnostics(part, body)
    clearance = None if min_clearance is None else max(float(min_clearance), 0.0)
    clearance_violation = False
    if body.contour_index is None and clearance is not None and clearance > 0.0:
        min_wire_distance = float(part.route_diagnostics.get("min_wire_wire_distance", 0.0))
        seg_a = int(part.route_diagnostics.get("min_wire_wire_segment_a", -1))
        seg_b = int(part.route_diagnostics.get("min_wire_wire_segment_b", -1))
        clearance_violation = seg_a >= 0 and seg_b >= 0 and min_wire_distance < clearance
        diagnostics["export_min_clearance"] = float(clearance)
        diagnostics["export_clearance_violation"] = bool(clearance_violation)
        diagnostics["stl_min_clearance"] = float(clearance)
        diagnostics["stl_clearance_violation"] = bool(clearance_violation)
    if not fail_on_violations or (bool(diagnostics.get("manufacturing_ok", False)) and not clearance_violation):
        return
    reasons = str(diagnostics.get("manufacturing_failure_reasons", "")).strip() or "unspecified"
    contour_label = "" if body.contour_index is None else f" contour {body.contour_index}"
    raise SweptBodyExportError(
        f"Part {part_ind}{contour_label} is not ready for {format_name} export: "
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


def _step_cartesian_point(point: np.ndarray) -> str:
    values = np.asarray(point, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("STEP export requires finite 3D Cartesian points.")
    return f"CARTESIAN_POINT('',({_step_float(values[0])},{_step_float(values[1])},{_step_float(values[2])}))"


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
