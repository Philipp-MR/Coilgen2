"""Strict fabrication review of the actual swept geometry and electrical path.

Thin-wire fields use the physical profile centroid after filleting and mounting.
Closed contour representatives are independent turns, never an implied series
connection. An open winding has a partial field/inductance until its external
return is supplied. Mesh geometry and winding/circuit readiness are separate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from ..field_components import component_sample_values, field_component_values
from ..inductance import calculate_inductance_by_coil_layout
from ..manufacturing import (
    _polygon_area,
    _polygon_centroid,
    _save_xyz_polyline_as_step,
    _segment_bvh_candidate_pairs,
    create_swept_coil_meshes,
    create_swept_contour_body,
    save_mesh_as_step,
    save_mesh_as_stl,
)
from ..models import CoilPart, FabricationPath, Mesh, WirePath
from ..physics import biot_savart_wire_path
from ..progress import ProgressReporter
from .kernels import geometry_acceleration, segment_pair_distances, triangle_pair_distances


def _candidate_pairs(first, second, distance):
    """Cross-set AABB candidates, filtered again at primitive-box resolution."""
    if not len(first) or not len(second):
        return np.empty((0, 2), dtype=np.int64)
    coordinates = np.concatenate((first, second))
    minima, maxima = coordinates.min(axis=1), coordinates.max(axis=1)
    pairs = _segment_bvh_candidate_pairs(minima, maxima, distance)
    if not len(pairs):
        return pairs
    pairs = pairs[(pairs[:, 0] < len(first)) & (pairs[:, 1] >= len(first))]
    delta = np.maximum(
        0.0, np.maximum(minima[pairs[:, 0]] - maxima[pairs[:, 1]], minima[pairs[:, 1]] - maxima[pairs[:, 0]])
    )
    pairs = pairs[np.einsum("ij,ij->i", delta, delta) <= distance**2]
    pairs[:, 1] -= len(first)
    return pairs


def accelerated_mesh_collision_diagnostics(mesh, *, backend="auto", maximum_reported_intersections=100):
    """Batch the same remote-triangle test used by research sweep validation."""
    triangles = mesh.vertices[mesh.faces]
    tolerance = max(float(np.max(np.ptp(mesh.vertices, axis=0))), 1.0) * 1e-12 if len(mesh.vertices) else 1e-12
    minima, maxima = triangles.min(axis=1), triangles.max(axis=1)
    pairs = _segment_bvh_candidate_pairs(minima, maxima, tolerance)
    if len(pairs):
        delta = np.maximum(
            0.0, np.maximum(minima[pairs[:, 0]] - maxima[pairs[:, 1]], minima[pairs[:, 1]] - maxima[pairs[:, 0]])
        )
        pairs = pairs[np.einsum("ij,ij->i", delta, delta) <= tolerance**2]
        adjacent = np.any(mesh.faces[pairs[:, 0], :, None] == mesh.faces[pairs[:, 1], None, :], axis=(1, 2))
        pairs = pairs[~adjacent]
    measured = triangle_pair_distances(triangles[pairs[:, 0]], triangles[pairs[:, 1]], backend=backend)
    count = int(np.count_nonzero(measured.values_m <= tolerance))
    return {
        "sweep_mesh_collision_broad_phase": "triangle_bvh",
        "sweep_mesh_collision_kernel": measured.backend,
        "sweep_mesh_collision_candidate_pair_count": len(pairs),
        "sweep_mesh_self_intersection_count": min(count, maximum_reported_intersections),
        "sweep_mesh_self_intersection_truncated": count >= maximum_reported_intersections,
        "sweep_mesh_self_intersection": count > 0,
    }


def _inside_closed_mesh(point, mesh):
    vectors = mesh.vertices[mesh.faces] - point
    lengths = np.linalg.norm(vectors, axis=2)
    numerator = np.einsum("ij,ij->i", vectors[:, 0], np.cross(vectors[:, 1], vectors[:, 2]))
    denominator = lengths.prod(axis=1)
    denominator += np.einsum("ij,ij->i", vectors[:, 0], vectors[:, 1]) * lengths[:, 2]
    denominator += np.einsum("ij,ij->i", vectors[:, 1], vectors[:, 2]) * lengths[:, 0]
    denominator += np.einsum("ij,ij->i", vectors[:, 2], vectors[:, 0]) * lengths[:, 1]
    winding = np.sum(2.0 * np.arctan2(numerator, denominator)) / (4.0 * np.pi)
    return bool(abs(winding) > 0.5)


@dataclass
class FabricationRecord:
    body_id: str
    part_index: int
    contour_index: int | None
    path: FabricationPath
    mesh: Mesh
    diagnostics: dict
    turn_multiplicity: int = 1

    @property
    def points_m(self):
        return self.path.wire_points_m


def assembly_clearance(records, *, insulation_thickness_m=0.0, tolerance_m=0.0, margin_m=0.0, backend="auto"):
    """Check every independent body pair, including different coil parts.

    Circumscribed capsules only screen candidates. Overlapping capsules are not
    reported as a physical collision without mesh intersection or containment.
    Insulation is an isotropic offset of the faceted conductor boundary.
    """
    requested = 2.0 * float(insulation_thickness_m) + float(tolerance_m) + float(margin_m)
    if not np.isfinite(requested) or min(insulation_thickness_m, tolerance_m, margin_m) < 0.0:
        raise ValueError("Assembly insulation, tolerance and margin must be finite non-negative lengths.")
    epsilon = 1e-10
    rows = []
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            row = {
                "first_body": first.body_id,
                "second_body": second.body_id,
                "required_surface_gap_m": requested,
                "conservative_envelope_overlap": False,
                "confirmed_conductor_intersection": False,
                "confirmed_clearance_violation": False,
                "minimum_surface_distance_m": None,
                "checked": True,
                "status": "clear",
            }
            first_points, second_points = first.points_m, second.points_m
            first_segments = np.stack((first_points[:-1], first_points[1:]), axis=1)
            second_segments = np.stack((second_points[:-1], second_points[1:]), axis=1)
            radii = [
                float(
                    np.max(
                        np.linalg.norm(
                            record.path.cross_section_m - _polygon_centroid(record.path.cross_section_m), axis=1
                        )
                    )
                )
                for record in (first, second)
            ]
            envelope = sum(radii) + requested
            candidates = _candidate_pairs(first_segments, second_segments, envelope + epsilon)
            if not len(candidates):
                row["certificate"] = "disjoint_centroid_capsules"
                rows.append(row)
                continue
            distances = segment_pair_distances(
                first_segments[candidates[:, 0]], second_segments[candidates[:, 1]], backend=backend
            )
            row["kernel_backend"] = distances.backend
            row["minimum_candidate_centerline_distance_m"] = float(np.min(distances.values_m))
            row["conservative_envelope_overlap"] = bool(np.any(distances.values_m < envelope))
            if not row["conservative_envelope_overlap"]:
                row["certificate"] = "disjoint_centroid_capsules"
                rows.append(row)
                continue
            first_triangles, second_triangles = (
                first.mesh.vertices[first.mesh.faces],
                second.mesh.vertices[second.mesh.faces],
            )
            candidates = _candidate_pairs(first_triangles, second_triangles, requested + epsilon)
            gap = np.inf
            if len(candidates):
                measured = triangle_pair_distances(
                    first_triangles[candidates[:, 0]], second_triangles[candidates[:, 1]], backend=backend
                )
                gap = float(np.min(measured.values_m))
                row["kernel_backend"] = measured.backend
                row["minimum_surface_distance_m"] = gap
            intersection = gap <= epsilon
            containment = (
                False
                if intersection
                else (
                    _inside_closed_mesh(first.mesh.vertices[0], second.mesh)
                    or _inside_closed_mesh(second.mesh.vertices[0], first.mesh)
                )
            )
            row.update(
                {
                    "confirmed_conductor_intersection": intersection or containment,
                    "confirmed_clearance_violation": intersection or containment or gap < requested - epsilon,
                    "containment_detected": containment,
                    "triangle_candidate_count": len(candidates),
                    "certificate": "faceted_surface_distance_and_containment",
                    "clearance_lower_bound_m": None if np.isfinite(gap) else requested,
                }
            )
            if row["confirmed_conductor_intersection"]:
                row["status"] = "conductor_intersection"
            elif row["confirmed_clearance_violation"]:
                row["status"] = "insulation_or_margin_violation"
            rows.append(row)
    return {
        "checked_body_pairs": len(rows),
        "passed": all(not row["confirmed_clearance_violation"] for row in rows),
        "distance_acceptance_tolerance_m": epsilon,
        "confirmed_intersection_pairs": sum(row["confirmed_conductor_intersection"] for row in rows),
        "confirmed_clearance_violation_pairs": sum(row["confirmed_clearance_violation"] for row in rows),
        "pairs": rows,
        "geometry_model": "faceted_conductor_plus_isotropic_insulation",
    }


def winding_instructions(record, return_path_m=None, *, checkpoint_spacing_m=0.05):
    """Group work by turns/connectors and retain the exact sampled route.

    Checkpoints are spaced by physical arclength and include sharp bends and
    ownership transitions. Sampling density does not create extra operations.
    Layer assignment and feed design remain explicit, unresolved build choices.
    """
    if not np.isfinite(checkpoint_spacing_m) or checkpoint_spacing_m <= 0.0:
        raise ValueError("Checkpoint spacing must be a finite positive length.")
    points = record.points_m
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-12):
        raise ValueError(f"{record.body_id} has a zero-length winding interval.")
    role = "open_winding"
    if record.path.closed:
        role = "closed_independent_turn" if record.path.source_kind == "contour" else "closed_winding"
    circuit_complete = bool(record.path.closed)
    electrical_points = points.copy()
    terminal_assumption = "Closed mathematical turn; feed break and drive connection are not designed."
    if not record.path.closed:
        terminal_assumption = "Start and finish terminals are exposed; external supply/return geometry is missing."
    if return_path_m is not None:
        if record.path.closed:
            raise ValueError("A return must attach to an open winding; closed turns need an explicit feed break first.")
        return_points = np.array(return_path_m, dtype=float, copy=True)
        if (
            return_points.ndim != 2
            or return_points.shape[1] != 3
            or len(return_points) < 2
            or not np.all(np.isfinite(return_points))
        ):
            raise ValueError("Return geometry requires at least two finite XYZ points.")
        if np.linalg.norm(return_points[0] - points[-1]) > 1e-9 or np.linalg.norm(return_points[-1] - points[0]) > 1e-9:
            raise ValueError("Return geometry must run continuously from the winding finish back to its start.")
        return_points[0], return_points[-1] = points[-1], points[0]
        if np.any(np.linalg.norm(np.diff(return_points, axis=0), axis=1) <= 1e-12):
            raise ValueError("Return geometry contains a zero-length interval.")
        electrical_points = np.vstack((points, return_points[1:]))
        role, circuit_complete = "winding_with_explicit_return", True
        terminal_assumption = (
            "External return is modeled electrically; its conductor envelope is not swept/clearance-certified."
        )
    winding_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    lengths = np.linalg.norm(np.diff(electrical_points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    winding_count = len(points) - 1
    roles = []
    for index in range(len(lengths)):
        owner = record.path.segment_owners[index] if index < len(record.path.segment_owners) else None
        geometry = (
            record.path.segment_geometry[index].value if index < len(record.path.segment_geometry) else "unspecified"
        )
        segment_role = "external_return" if index >= winding_count else "winding"
        kind = (
            owner.kind if owner is not None else ("contour_loop" if record.path.source_kind == "contour" else "winding")
        )
        group = owner.group_index if owner is not None else -1
        loop = owner.loop_index if owner is not None else (record.contour_index - 1 if record.contour_index else -1)
        connector = owner.connector_index if owner is not None else -1
        connector_role = owner.connector_role if owner is not None else ""
        roles.append((segment_role, kind, group, loop, connector, connector_role, geometry))
    boundaries = [0, *[i for i in range(1, len(roles)) if roles[i] != roles[i - 1]], len(lengths)]
    station_indices = set(boundaries)
    if len(lengths) > 1:
        units = np.diff(electrical_points, axis=0) / lengths[:, None]
        angles = np.arccos(np.clip(np.einsum("ij,ij->i", units[:-1], units[1:]), -1.0, 1.0))
        station_indices.update((np.flatnonzero(angles >= np.pi / 4.0) + 1).tolist())
    station_distances = sorted(
        set(cumulative[list(station_indices)]).union(
            np.arange(checkpoint_spacing_m, cumulative[-1], checkpoint_spacing_m).tolist()
        )
    )
    markers_by_distance = {distance: index + 1 for index, distance in enumerate(station_distances)}
    if circuit_complete:
        markers_by_distance[cumulative[-1]] = 1
    markers = {sample: markers_by_distance[cumulative[sample]] for sample in boundaries}
    checkpoints = []
    for distance in station_distances:
        index = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(lengths) - 1)
        fraction = float((distance - cumulative[index]) / lengths[index])
        position = electrical_points[index] + fraction * (electrical_points[index + 1] - electrical_points[index])
        checkpoints.append(
            {
                "marker": markers_by_distance[distance],
                "sample_index": index,
                "segment_fraction": fraction,
                "arclength_m": float(distance),
                "xyz_m": position.tolist(),
                "operation_boundary": distance in cumulative[boundaries],
            }
        )
    steps = []
    for number, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        segment_role, kind, group, loop, connector, connector_role, geometry = roles[start]
        if segment_role == "external_return":
            operation, instruction = "route_return", "Route the external return"
        elif kind == "inter_group_connector":
            operation, instruction = (
                "interconnect",
                f"Route connector {connector + 1} ({connector_role or 'connection'})",
            )
        elif kind == "contour_loop":
            operation, instruction = "wind_turn", f"Wind turn {loop + 1}" if loop >= 0 else "Wind the independent turn"
        elif kind == "contour_group":
            operation, instruction = "wind_group", f"Wind contour group {group + 1}"
        else:
            operation, instruction = "follow_winding", "Follow the winding route"
        steps.append(
            {
                "step_number": number,
                "body_id": record.body_id,
                "operation": operation,
                "segment_role": segment_role,
                "group_index": group,
                "loop_index": loop,
                "connector_index": connector,
                "geometry": geometry,
                "layer": None,
                "from_marker": markers[start],
                "to_marker": markers[stop],
                "start_sample_index": start,
                "end_sample_index": stop,
                "start_xyz_m": electrical_points[start].tolist(),
                "end_xyz_m": electrical_points[stop].tolist(),
                "interval_length_m": float(cumulative[stop] - cumulative[start]),
                "cumulative_length_m": float(cumulative[stop]),
                "instruction": f"{instruction} along the exported route from station {markers[start]} to {markers[stop]}.",
            }
        )
    return {
        "body_id": record.body_id,
        "circuit_role": role,
        "circuit_complete": circuit_complete,
        "drive_feed_designed": False,
        "terminal_assumption": terminal_assumption,
        "start_terminal_xyz_m": points[0].tolist(),
        "finish_terminal_xyz_m": points[-1].tolist(),
        "winding_length_m": winding_length,
        "electrical_path_length_m": float(lengths.sum()),
        "electrical_points_m": electrical_points,
        "checkpoint_spacing_m": checkpoint_spacing_m,
        "checkpoints": checkpoints,
        "sampled_interval_count": len(lengths),
        "operation_count": len(steps),
        "layer_assignment_designed": False,
        "steps": steps,
    }


@dataclass
class FabricationReport:
    paths: list[FabricationRecord]
    assembly_clearance: dict
    winding_steps: list[dict]
    field_metrics: dict
    electrical_metrics: dict
    failure_reasons: list[str]
    artifacts: list[str]
    strict: bool

    @property
    def ready(self):
        return not self.failure_reasons

    def summary(self):
        return {
            "ready": self.ready,
            "strict": self.strict,
            "failure_reasons": self.failure_reasons,
            "assembly_clearance": self.assembly_clearance,
            "field_metrics": self.field_metrics,
            "electrical_metrics": self.electrical_metrics,
            "artifacts": self.artifacts,
            "paths": [
                {
                    "body_id": record.body_id,
                    "part_index": record.part_index,
                    "contour_index": record.contour_index,
                    "closed": record.path.closed,
                    "source_kind": record.path.source_kind,
                    "point_count": len(record.points_m),
                    "length_m": record.path.length_m,
                    "turn_multiplicity": record.turn_multiplicity,
                    "diagnostics": record.diagnostics,
                }
                for record in self.paths
            ],
            "winding": [
                {key: value for key, value in item.items() if key != "electrical_points_m"}
                for item in self.winding_steps
            ],
        }


def compile_fabrication(
    result,
    output_dir=None,
    *,
    strict=True,
    kernel_backend="auto",
    leads=None,
    surface_realization="mesh",
    checkpoint_spacing_m=0.05,
):
    """Build inspectable fabrication artifacts; strict failures gate solid export.

    ``leads`` maps body IDs to explicit finish-to-start return polylines. Those
    paths participate in field/electrical evaluation but require their own
    conductor clearance review. The original result's EM solution is untouched.

    The headless compiler preserves the finite-element support by default.
    ``surface_realization='analytic'`` explicitly projects supported cylinders
    onto their ideal surface; ``None`` uses the legacy configuration setting.
    Both policies re-evaluate fields on the final mounted profile centroids.
    """
    config = replace(result.input_data, skip_sweep=False)
    if surface_realization is not None:
        if surface_realization not in {"mesh", "analytic"}:
            raise ValueError("surface_realization must be 'mesh', 'analytic', or None.")
        config = replace(config, sweep_surface_realization=surface_realization)
    reporter = ProgressReporter.from_config(config, label="Fabrication")
    reporter.report(0.0, "Sweeping physical conductor paths")
    records, reasons = [], []
    with geometry_acceleration(kernel_backend):
        for part_index, part in enumerate(result.coil_parts, start=1):
            reporter.report(
                35.0 * (part_index - 1) / max(len(result.coil_parts), 1),
                f"Sweeping coil part {part_index}/{len(result.coil_parts)}",
            )
            if part.wire_path is not None and len(part.wire_path.v) >= 2:
                local = replace(part, swept_body=None, swept_bodies=[], fabrication_path=None)
                create_swept_coil_meshes([local], replace(config, contour_only_sweep_mode=False))
                bodies = local.swept_bodies
            else:
                bodies = [
                    create_swept_contour_body(
                        part, contour, config, part_index=part_index, contour_index=index, validate=False
                    )
                    for index, contour in enumerate(part.contour_lines, start=1)
                ]
            for body_index, body in enumerate(bodies, start=1):
                body_id = f"part{part_index:03d}_body{body_index:04d}"
                if body.fabrication_path is None:
                    reasons.append(f"{body_id}:missing_fabrication_path")
                    continue
                records.append(
                    FabricationRecord(
                        body_id,
                        part_index,
                        body.contour_index,
                        body.fabrication_path,
                        body.mesh,
                        dict(body.diagnostics),
                        body.turn_multiplicity,
                    )
                )
                if not body.diagnostics.get("manufacturing_ok", False):
                    reasons.append(
                        f"{body_id}:{body.diagnostics.get('manufacturing_failure_reasons', 'invalid_sweep')}"
                    )
                if body.turn_multiplicity != 1:
                    reasons.append(f"{body_id}:weighted_turns_not_physically_expanded")
    if not records:
        reasons.append("no_fabrication_paths")
    reporter.report(40.0, "Checking insulation and conductor clearance across every body pair")
    assembly = assembly_clearance(
        records,
        insulation_thickness_m=config.sweep_insulation_thickness,
        tolerance_m=config.sweep_manufacturing_tolerance,
        margin_m=config.sweep_bend_clearance_margin,
        backend=kernel_backend,
    )
    if not assembly["passed"]:
        reasons.append("assembly_clearance_violation")
    leads = leads or {}
    unknown_leads = set(leads) - {record.body_id for record in records}
    if unknown_leads:
        raise ValueError(f"Lead geometry references unknown fabrication bodies: {sorted(unknown_leads)}")
    plans = [
        winding_instructions(record, leads.get(record.body_id), checkpoint_spacing_m=checkpoint_spacing_m)
        for record in records
    ]
    field = np.zeros_like(result.target_field.b, dtype=float)
    source_field = np.zeros_like(field)
    electrical = []
    for record, plan in zip(records, plans):
        reporter.report(
            60.0 + 30.0 * len(electrical) / max(len(records), 1),
            f"Evaluating physical field and electrical path for {record.body_id}",
        )
        points = plan["electrical_points_m"]
        contour_step = result.coil_parts[record.part_index - 1].contour_step
        if contour_step is None:
            raise ValueError("Fabrication field evaluation requires an explicit physical contour current.")
        current = float(contour_step)
        if not np.isfinite(current) or current == 0.0:
            raise ValueError("Fabrication field evaluation requires finite nonzero physical contour currents.")
        field += current * biot_savart_wire_path(points, result.target_field.coords, worker_count=1)
        source_field += current * biot_savart_wire_path(
            record.path.source_points_m, result.target_field.coords, worker_count=1
        )
        area = _polygon_area(record.path.cross_section_m)
        proxy = CoilPart(mesh=result.coil_parts[record.part_index - 1].mesh, wire_path=WirePath(v=points))
        circuit_config = replace(
            config,
            skip_inductance_calculation=False,
            skip_postprocessing=False,
            inductance_source="wire_path",
            cross_sectional_points=record.path.cross_section_m,
            parallel_worker_count=1,
        )
        _, inductance, resistance, lengths, _ = calculate_inductance_by_coil_layout([proxy], circuit_config)
        electrical.append(
            {
                "body_id": record.body_id,
                "current_A": current,
                "cross_section_area_m2": area,
                "length_m": float(lengths[0]),
                "resistance_ohm": float(resistance[0]),
                "joule_power_W": float(current**2 * resistance[0]),
                "inductance_H": float(inductance[0, 0]),
                "inductance_kind": "closed_path_self" if plan["circuit_complete"] else "open_path_partial",
                "circuit_complete": plan["circuit_complete"],
                "mutual_inductance_included": False,
            }
        )
        if not plan["circuit_complete"]:
            reasons.append(f"{record.body_id}:external_return_not_modeled")
        if record.body_id in leads:
            reasons.append(f"{record.body_id}:return_conductor_clearance_not_certified")
    mask = ~np.asarray(result.target_field.suppressed, dtype=bool)
    component = config.target_field_component
    target = field_component_values(result.target_field.b[mask], component, b0_direction=config.b_0_direction)
    actual = field_component_values(field[mask], component, b0_direction=config.b_0_direction)
    source = field_component_values(source_field[mask], component, b0_direction=config.b_0_direction)
    denominator = float(np.linalg.norm(target))
    weights = component_sample_values(result.target_field.weights[mask], component)
    weight_sum = float(np.sum(weights))
    weighted_rms = float(np.sqrt(np.sum(weights * (actual - target) ** 2) / weight_sum)) if weight_sum else None
    weighted_std = (
        float(
            np.sqrt(
                np.sum(weights * ((actual - target) - np.sum(weights * (actual - target)) / weight_sum) ** 2)
                / weight_sum
            )
        )
        if weight_sum
        else None
    )
    layout_budget_checks = {}
    for name, measured in (("rms", weighted_rms), ("std", weighted_std)):
        limit = float(getattr(config, f"layout_target_error_{name}_limit_T", 0.0))
        if limit > 0.0:
            accepted = measured is not None and measured <= limit + config.layout_target_error_acceptance_absolute_T
            layout_budget_checks[name] = {
                "requested_limit_T": limit,
                "measured_T": measured,
                "acceptance_absolute_T": config.layout_target_error_acceptance_absolute_T,
                "passed": accepted,
            }
            if not accepted:
                reasons.append(f"fabrication_field_{name}_budget_exceeded")
    metrics = {
        "current_source": "original_contour_step_no_refit",
        "surface_realization": config.sweep_surface_realization,
        "field_geometry": "fabrication_profile_centroid_thin_wire",
        "turn_model": "one_physical_path_per_body_weighted_representatives_not_expanded",
        "dsv_weighted_rms_residual_T": weighted_rms,
        "dsv_weighted_std_residual_T": weighted_std,
        "layout_budget_checks": layout_budget_checks,
        "dsv_rms_residual_T": float(np.sqrt(np.mean((actual - target) ** 2))) if len(target) else None,
        "dsv_relative_residual_percent": 100.0 * float(np.linalg.norm(actual - target)) / denominator
        if denominator
        else None,
        "dsv_source_to_fabrication_delta_rms_T": float(np.sqrt(np.mean((actual - source) ** 2)))
        if len(target)
        else None,
        "circuit_complete": all(plan["circuit_complete"] for plan in plans),
        "finite_cross_section_field_integrated": False,
    }
    report = FabricationReport(
        records,
        assembly,
        plans,
        metrics,
        {
            "bodies": electrical,
            "total_joule_power_W": sum(item["joule_power_W"] for item in electrical),
            "circuit_mutual_inductance_included": False,
        },
        reasons,
        [],
        strict,
    )
    if output_dir is not None:
        reporter.report(95.0, "Writing centroid paths, winding instructions and qualified solid artifacts")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for record in records:
            path = output / f"{record.body_id}_fabrication_centroid.csv"
            np.savetxt(path, record.points_m, delimiter=",", header="x_m,y_m,z_m", comments="", fmt="%.17g")
            report.artifacts.append(str(path.resolve()))
            frames_path = output / f"{record.body_id}_fabrication_frames.npz"
            np.savez_compressed(
                frames_path,
                points_m=record.path.points_m,
                tangents=record.path.tangents,
                normals=record.path.normals,
                transverse=record.path.transverse,
                cross_section_m=record.path.cross_section_m,
                source_points_m=record.path.source_points_m,
                closed=record.path.closed,
                source_kind=record.path.source_kind,
                segment_geometry=np.asarray([value.value for value in record.path.segment_geometry]),
                owner_kind=np.asarray([owner.kind for owner in record.path.segment_owners]),
                owner_group_index=np.asarray(
                    [owner.group_index for owner in record.path.segment_owners], dtype=np.int64
                ),
                owner_loop_index=np.asarray([owner.loop_index for owner in record.path.segment_owners], dtype=np.int64),
                owner_connector_index=np.asarray(
                    [owner.connector_index for owner in record.path.segment_owners], dtype=np.int64
                ),
                owner_connector_role=np.asarray([owner.connector_role for owner in record.path.segment_owners]),
            )
            report.artifacts.append(str(frames_path.resolve()))
            step = output / f"{record.body_id}_fabrication_centroid.step"
            _save_xyz_polyline_as_step(record.points_m, step, wire_name=record.body_id, part_index=record.part_index)
            report.artifacts.append(str(step.resolve()))
            if report.ready or not strict:
                stl = output / f"{record.body_id}_fabrication.stl"
                save_mesh_as_stl(record.mesh, stl, solid_name=record.body_id)
                report.artifacts.append(str(stl.resolve()))
                solid_step = output / f"{record.body_id}_fabrication_faceted.step"
                save_mesh_as_step(record.mesh, solid_step, solid_name=record.body_id)
                report.artifacts.append(str(solid_step.resolve()))
        for plan in plans:
            if plan["body_id"] not in leads:
                continue
            electrical_path = output / f"{plan['body_id']}_electrical_path_with_return.csv"
            np.savetxt(
                electrical_path,
                plan["electrical_points_m"],
                delimiter=",",
                header="x_m,y_m,z_m",
                comments="",
                fmt="%.17g",
            )
            report.artifacts.append(str(electrical_path.resolve()))
        field_path = output / "fabrication_fields.npz"
        np.savez_compressed(
            field_path,
            target_coordinates_m=result.target_field.coords,
            source_field_T=source_field,
            fabrication_field_T=field,
            target_field_T=result.target_field.b,
        )
        report.artifacts.append(str(field_path.resolve()))
        instructions = output / "winding_steps.csv"
        rows = [step for plan in plans for step in plan["steps"]]
        with instructions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["step_number", "body_id"])
            writer.writeheader()
            writer.writerows(rows)
        report.artifacts.append(str(instructions.resolve()))
        summary_path = output / "fabrication_report.json"
        report.artifacts.append(str(summary_path.resolve()))
        summary_path.write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
    reporter.finish("Fabrication review complete")
    return report
