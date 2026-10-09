from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .attribution import software_attribution
from .basis import current_density_from_basis
from .config import CoilGenConfig
from .design_context import EvaluationFeature, normalize_design_context
from .field_maps import (
    display_field_unit_1a,
    field_error_scalar_maps,
    field_relative_error_scalar_maps,
    field_scalar_maps,
    format_label_with_unit,
    gradient_component_scalar_maps,
    gradient_relative_error_scalar_maps,
    interpolated_slice_mosaic,
    shim_field_scalar_maps,
    shim_homogeneity_stats,
    target_domain_mask,
)
from .models import CoilGenResult, CoilPart, ContourLine
from .regularization_diagnostics import normalized_linear_trace, normalized_log_trace, regularization_scan_arrays
from .reporting import result_summary_dict


COORDINATE_UNIT = "m"
STREAM_FUNCTION_UNIT = "A"
CURRENT_DENSITY_UNIT = "A/m"
RELATIVE_ERROR_UNIT = "%"


@dataclass(slots=True)
class SvgStyle:
    width: int = 900
    height: int = 720
    margin: int = 36
    background: str = "#ffffff"
    mesh_stroke: str = "#d4d8dd"
    contour_stroke: str = "#1f6feb"
    wire_stroke: str = "#111827"
    shifted_stroke: str = "#d73a49"
    text_color: str = "#111827"


def save_result_visualizations(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
    style: SvgStyle | None = None,
) -> list[Path]:
    """Save a compact SVG visualization set for every coil part."""

    style = style or SvgStyle()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        part_prefix = f"{prefix}_part{part_ind}"
        _append_optional(
            saved,
            lambda part=part: save_wire_projection(
                part, output / f"{part_prefix}_wire_xy.svg", axes=(0, 1), style=style
            ),
        )
        _append_optional(
            saved,
            lambda part=part: save_wire_projection(
                part, output / f"{part_prefix}_wire_xz.svg", axes=(0, 2), style=style
            ),
        )
        if part.contour_lines and part.groups:
            _append_optional(
                saved,
                lambda part=part: save_contour_group_projection(
                    part, output / f"{part_prefix}_contour_groups_xy.svg", axes=(0, 1), style=style
                ),
            )
            _append_optional(
                saved,
                lambda part=part: save_contour_group_projection(
                    part, output / f"{part_prefix}_contour_groups_xz.svg", axes=(0, 2), style=style
                ),
            )
            _append_optional(
                saved,
                lambda part=part: save_contour_group_projection(
                    part, output / f"{part_prefix}_contour_groups_yz.svg", axes=(1, 2), style=style
                ),
            )
            _append_optional(
                saved,
                lambda part=part: save_contour_group_isometric(
                    part, output / f"{part_prefix}_contour_groups_iso.svg", style=style
                ),
            )
    saved.append(save_field_comparison(result, output / f"{prefix}_field_comparison.svg", style=style))
    return saved


def save_plot_gallery(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
    style: SvgStyle | None = None,
    include_diagnostics: bool = True,
    include_pcb: bool = True,
) -> list[Path]:
    """Save the full dependency-free inspection plot set for a CoilGen result."""

    style = style or SvgStyle()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    design_context = normalize_design_context(getattr(result.input_data, "design_context", "mr_gradient"))
    saved.extend(save_result_visualizations(result, output, prefix=prefix, style=style))

    for part_ind, part in enumerate(result.coil_parts, start=1):
        part_prefix = f"{prefix}_part{part_ind}"
        _append_optional(
            saved, lambda part=part: save_wire_isometric(part, output / f"{part_prefix}_wire_iso.svg", style=style)
        )
        _append_optional(
            saved,
            lambda part=part: save_wire_projection(
                part, output / f"{part_prefix}_wire_yz.svg", axes=(1, 2), style=style
            ),
        )
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            _append_optional(
                saved,
                lambda part=part, axes=axes, suffix=suffix: save_contours_projection(
                    part,
                    output / f"{part_prefix}_contours_{suffix}.svg",
                    axes=axes,
                    style=style,
                ),
            )
        _append_optional(
            saved,
            lambda part=part: save_contours_isometric(part, output / f"{part_prefix}_contours_iso.svg", style=style),
        )
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            _append_optional(
                saved,
                lambda part=part, axes=axes, suffix=suffix: save_stream_function_projection(
                    part,
                    output / f"{part_prefix}_stream_function_{suffix}.svg",
                    axes=axes,
                    style=style,
                ),
            )
        _append_optional(
            saved,
            lambda part=part: save_stream_function_isometric(
                part, output / f"{part_prefix}_stream_function_iso.svg", style=style
            ),
        )
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            _append_optional(
                saved,
                lambda part=part, axes=axes, suffix=suffix: save_current_density_projection(
                    part,
                    output / f"{part_prefix}_current_density_{suffix}.svg",
                    axes=axes,
                    style=style,
                ),
            )
        _append_optional(
            saved,
            lambda part=part: save_current_density_isometric(
                part, output / f"{part_prefix}_current_density_iso.svg", style=style
            ),
        )
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            _append_optional(
                saved,
                lambda part=part, axes=axes, suffix=suffix: save_interconnect_debug_projection(
                    part,
                    output / f"{part_prefix}_interconnect_debug_{suffix}.svg",
                    axes=axes,
                    style=style,
                ),
            )
        _append_optional(
            saved,
            lambda part=part: save_interconnect_debug_isometric(
                part, output / f"{part_prefix}_interconnect_debug_iso.svg", style=style
            ),
        )
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            _append_optional(
                saved,
                lambda part=part, axes=axes, suffix=suffix: save_swept_body_projection(
                    part,
                    output / f"{part_prefix}_swept_body_{suffix}.svg",
                    axes=axes,
                    style=style,
                ),
            )
        _append_optional(
            saved,
            lambda part=part: save_swept_body_isometric(
                part, output / f"{part_prefix}_swept_body_iso.svg", style=style
            ),
        )

    _append_optional(saved, lambda: save_within_group_cut_choices(result, output, prefix=prefix, style=style))
    for field_source in ("target", "layout", "stream", "error"):
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            _append_optional(
                saved,
                lambda field_source=field_source, axes=axes, suffix=suffix: save_target_field_map(
                    result,
                    output / f"{prefix}_target_field_{field_source}_{suffix}.svg",
                    source=field_source,
                    axes=axes,
                    style=style,
                ),
            )
        _append_optional(
            saved,
            lambda field_source=field_source: save_target_field_map_isometric(
                result,
                output / f"{prefix}_target_field_{field_source}_iso.svg",
                source=field_source,
                style=style,
            ),
        )
    if design_context.enabled(EvaluationFeature.GRADIENT_FIT):
        for gradient_source in ("target", "layout", "stream", "error"):
            for component in ("magnitude", "dbz_dx", "dbz_dy", "dbz_dz"):
                _append_optional(
                    saved,
                    lambda gradient_source=gradient_source, component=component: save_gradient_map_isometric(
                        result,
                        output / f"{prefix}_gradient_{gradient_source}_{component}_iso.svg",
                        source=gradient_source,
                        component=component,
                        style=style,
                    ),
                )
        for gradient_error_source in ("layout_vs_target", "stream_vs_target", "layout_vs_stream"):
            for component in ("magnitude", "dbz_dx", "dbz_dy", "dbz_dz"):
                _append_optional(
                    saved,
                    lambda gradient_error_source=gradient_error_source, component=component: (
                        save_gradient_relative_error_map_isometric(
                            result,
                            output / f"{prefix}_relative_gradient_error_{gradient_error_source}_{component}_iso.svg",
                            source=gradient_error_source,
                            component=component,
                            style=style,
                        )
                    ),
                )
    for error_source in ("layout_vs_target", "stream_vs_target", "layout_vs_stream"):
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            _append_optional(
                saved,
                lambda error_source=error_source, axes=axes, suffix=suffix: save_field_error_map(
                    result,
                    output / f"{prefix}_error_map_{error_source}_{suffix}.svg",
                    source=error_source,
                    axes=axes,
                    style=style,
                ),
            )
        _append_optional(
            saved,
            lambda error_source=error_source: save_field_error_map_isometric(
                result,
                output / f"{prefix}_error_map_{error_source}_iso.svg",
                source=error_source,
                style=style,
            ),
        )
    for error_source in ("layout_vs_target", "stream_vs_target", "layout_vs_stream"):
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            _append_optional(
                saved,
                lambda error_source=error_source, axes=axes, suffix=suffix: save_field_relative_error_map(
                    result,
                    output / f"{prefix}_relative_error_map_{error_source}_{suffix}.svg",
                    source=error_source,
                    axes=axes,
                    style=style,
                ),
            )
        _append_optional(
            saved,
            lambda error_source=error_source: save_field_relative_error_map_isometric(
                result,
                output / f"{prefix}_relative_error_map_{error_source}_iso.svg",
                source=error_source,
                style=style,
            ),
        )
    if design_context.enabled(EvaluationFeature.SHIM_HOMOGENEITY):
        _append_optional(
            saved,
            lambda: save_shim_homogeneity_mosaic(
                result,
                output / f"{prefix}_shim_homogeneity_before_z_mosaic.svg",
                source="before",
                axis=2,
                style=style,
            ),
        )
        _append_optional(
            saved,
            lambda: save_shim_pointcloud_before_after(
                result,
                output / f"{prefix}_shim_pointcloud_before_after_iso.svg",
                source="after",
                domain="dsv",
                color_limit_abs=_shim_reference_color_limit_abs(result, source="after", domain="dsv"),
                style=style,
            ),
        )
        _append_optional(
            saved,
            lambda: save_shim_homogeneity_mosaic(
                result,
                output / f"{prefix}_shim_homogeneity_after_z_mosaic.svg",
                source="after",
                axis=2,
                color_limit_abs=_shim_reference_color_limit_abs(result, source="after", domain="dsv"),
                style=style,
            ),
        )
        _append_optional(
            saved,
            lambda: save_shim_homogeneity_mosaic(
                result,
                output / f"{prefix}_shim_homogeneity_improvement_z_mosaic.svg",
                source="improvement",
                axis=2,
                color_limit_abs=_shim_reference_color_limit_abs(result, source="improvement", domain="dsv"),
                style=style,
            ),
        )
        _append_optional(
            saved,
            lambda: save_shim_homogeneity_mosaic(
                result,
                output / f"{prefix}_shim_homogeneity_layout_z_mosaic.svg",
                source="after",
                axis=2,
                color_limit_abs=_shim_reference_color_limit_abs(result, source="after", domain="dsv"),
                style=style,
            ),
        )
    _append_optional(saved, lambda: save_error_metric_bars(result, output / f"{prefix}_error_metrics.svg", style=style))
    _append_optional(
        saved, lambda: save_performance_summary(result, output / f"{prefix}_performance_summary.svg", style=style)
    )
    _append_optional(
        saved, lambda: save_l_curve_regularization(result, output / f"{prefix}_regularization_l_curve.svg", style=style)
    )

    if include_diagnostics:
        saved.extend(save_inter_group_diagnostics(result, output, prefix=prefix))
        saved.extend(save_route_diagnostics(result, output, prefix=prefix))
        saved.extend(save_topology_diagnostics(result, output, prefix=prefix))
        saved.extend(save_within_group_cut_diagnostics(result, output, prefix=prefix))
        saved.extend(save_field_error_diagnostics(result, output, prefix=prefix))
        saved.extend(save_contour_orientation_diagnostics(result, output, prefix=prefix))

    if include_pcb and any(part.pcb_layout is not None for part in result.coil_parts):
        from .pcb import save_pcb_centerlines_as_csv, save_pcb_layouts_as_svg

        saved.extend(save_pcb_layouts_as_svg(result, output, prefix=f"{prefix}_pcb"))
        saved.extend(save_pcb_centerlines_as_csv(result, output, prefix=f"{prefix}_pcb"))

    return _deduplicate_paths(saved)


def inter_group_diagnostics_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    """Return one row per inter-group connection cut."""

    rows: list[dict[str, Any]] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        outer_loop_indices = {int(loop_index) for group in part.groups for loop_index in group.outer_loop_indices}
        for cut_ind, cut in enumerate(part.opening_cuts_among_groups):
            row = {
                "part_index": part_ind,
                "cut_index": cut_ind,
                "group1_index": cut.group1_index,
                "group2_index": cut.group2_index,
                "loop1_index": cut.loop1_index,
                "loop2_index": cut.loop2_index,
                "outer_contour_only": bool(
                    cut.loop1_index in outer_loop_indices and cut.loop2_index in outer_loop_indices
                ),
                "connection_method": cut.connection_method,
                "used_direct_fallback": cut.used_direct_fallback,
                "used_clearance_fallback": cut.used_clearance_fallback,
                "direct_distance": cut.direct_distance,
                "geodesic_distance": cut.geodesic_distance,
                "connector_distance": cut.connector_distance,
                "connector_point_count": len(cut.connection_path_v),
                "closing_connector_point_count": len(cut.closing_connection_path_v),
                "closing_connector_distance": _polyline_length(cut.closing_connection_path_v),
                "selected_point1_x": _point_component(cut.selected_point1_v, 0),
                "selected_point1_y": _point_component(cut.selected_point1_v, 1),
                "selected_point1_z": _point_component(cut.selected_point1_v, 2),
                "selected_point2_x": _point_component(cut.selected_point2_v, 0),
                "selected_point2_y": _point_component(cut.selected_point2_v, 1),
                "selected_point2_z": _point_component(cut.selected_point2_v, 2),
                "cut1_center_x": _center_component(cut.cut1_v, 0),
                "cut1_center_y": _center_component(cut.cut1_v, 1),
                "cut1_center_z": _center_component(cut.cut1_v, 2),
                "cut2_center_x": _center_component(cut.cut2_v, 0),
                "cut2_center_y": _center_component(cut.cut2_v, 1),
                "cut2_center_z": _center_component(cut.cut2_v, 2),
            }
            rows.append(row)
    return rows


def save_inter_group_diagnostics(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
) -> list[Path]:
    """Save inter-group routing diagnostics as JSON and CSV."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = inter_group_diagnostics_rows(result)
    json_path = output / f"{prefix}_inter_group_diagnostics.json"
    csv_path = output / f"{prefix}_inter_group_diagnostics.csv"
    _write(json_path, json.dumps(rows, indent=2))
    _write_diagnostics_csv(csv_path, rows)
    return [json_path, csv_path]


def _polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) >= 2 else 0.0


def route_diagnostics_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    """Return one row per coil part with route diagnostics."""

    rows: list[dict[str, Any]] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        row: dict[str, Any] = {
            "part_index": part_ind,
            "contour_count": len(part.contour_lines),
            "nominal_contour_geometry_count": part.nominal_contour_count,
            "physical_turn_count": part.physical_turn_count,
            "group_count": len(part.groups),
            "wire_point_count": len(part.wire_path.v) if part.wire_path is not None else 0,
        }
        row.update(part.route_diagnostics)
        rows.append(row)
    return rows


def save_route_diagnostics(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
) -> list[Path]:
    """Save per-part route diagnostics as JSON and CSV."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = route_diagnostics_rows(result)
    json_path = output / f"{prefix}_route_diagnostics.json"
    csv_path = output / f"{prefix}_route_diagnostics.csv"
    _write(json_path, json.dumps(rows, indent=2))
    _write_diagnostics_csv(csv_path, rows)
    return [json_path, csv_path]


def topology_diagnostics_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    """Return one row per topological loop group."""

    rows: list[dict[str, Any]] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        parent_by_group = _topology_parent_by_group(part)
        children_by_group = {
            group_ind: sorted(child for child, parent in parent_by_group.items() if parent == group_ind)
            for group_ind in range(len(part.groups))
        }
        sibling_by_group = _topology_siblings_by_group(part)
        level_by_group = _topology_level_by_group(part)
        depth_by_group = _topology_depth_by_group(parent_by_group)
        contains = part.loop_in_loop_matrix
        for group_ind, group in enumerate(part.groups):
            group_loop_set = set(group.loop_indices)
            contained_by_groups: list[int] = []
            contains_groups: list[int] = []
            if contains is not None:
                for other_ind, other in enumerate(part.groups):
                    if other_ind == group_ind:
                        continue
                    other_loop_set = set(other.loop_indices)
                    if any(bool(contains[outer, inner]) for outer in group_loop_set for inner in other_loop_set):
                        contains_groups.append(other_ind)
                    if any(bool(contains[outer, inner]) for outer in other_loop_set for inner in group_loop_set):
                        contained_by_groups.append(other_ind)
            rows.append(
                {
                    "part_index": part_ind,
                    "group_index": group_ind,
                    "parent_group_index": parent_by_group.get(group_ind, -1),
                    "child_group_indices": _join_ints(children_by_group.get(group_ind, [])),
                    "sibling_group_indices": _join_ints(sibling_by_group.get(group_ind, [])),
                    "topology_level_index": level_by_group.get(group_ind, -1),
                    "nesting_depth": depth_by_group.get(group_ind, 0),
                    "loop_indices": _join_ints(group.loop_indices),
                    "loop_count": len(group.loop_indices),
                    "contour_point_count": int(sum(len(loop.v) for loop in group.loops)),
                    "contains_group_indices": _join_ints(sorted(contains_groups)),
                    "contained_by_group_indices": _join_ints(sorted(contained_by_groups)),
                }
            )
    return rows


def save_topology_diagnostics(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
) -> list[Path]:
    """Save topological loop-group tree diagnostics as JSON and CSV."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = topology_diagnostics_rows(result)
    json_path = output / f"{prefix}_topology_diagnostics.json"
    csv_path = output / f"{prefix}_topology_diagnostics.csv"
    _write(json_path, json.dumps(rows, indent=2))
    _write_diagnostics_csv(csv_path, rows)
    return [json_path, csv_path]


def within_group_cut_diagnostics_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    """Return one row per loop group with selected within-group cut metrics."""

    rows: list[dict[str, Any]] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        for group_ind, group in enumerate(part.groups):
            metrics = dict(group.within_group_cut_metrics)
            row: dict[str, Any] = {
                "part_index": part_ind,
                "group_index": group_ind,
                "loop_indices": _join_ints(group.loop_indices),
                "loop_count": len(group.loop_indices),
                "selected_side": group.within_group_cut_side,
                "selected_score": group.within_group_cut_score,
                "selected_pair_indices": _join_ints(group.within_group_cut_pair_indices),
                "high_score": group.within_group_cut_scores.get("high", 0.0),
                "low_score": group.within_group_cut_scores.get("low", 0.0),
                "high_candidate_count": group.within_group_cut_scores.get("high_candidate_count", 0.0),
                "low_candidate_count": group.within_group_cut_scores.get("low_candidate_count", 0.0),
            }
            for key in (
                "path_length",
                "endpoint_distance",
                "excess_length",
                "turn_penalty",
                "line_deviation",
                "mean_segment_length",
                "degenerate_opening_count",
            ):
                row[key] = metrics.get(key, 0.0)
            rows.append(row)
    return rows


def save_within_group_cut_diagnostics(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
) -> list[Path]:
    """Save selected within-group cut metrics as JSON and CSV."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = within_group_cut_diagnostics_rows(result)
    json_path = output / f"{prefix}_within_group_cut_diagnostics.json"
    csv_path = output / f"{prefix}_within_group_cut_diagnostics.csv"
    _write(json_path, json.dumps(rows, indent=2))
    _write_diagnostics_csv(csv_path, rows)
    return [json_path, csv_path]


def field_error_diagnostics_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    """Return compact scalar field/performance diagnostics."""

    rows: list[dict[str, Any]] = []
    for key, value in result.error_vals.items():
        rows.append({"source": "field_error", "domain": "dsv", "metric": key, "value": float(value)})
    summary = result_summary_dict(result)
    for key, value in summary.items():
        if "_suppressed_" not in str(key):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        rows.append({"source": "suppression_fringe", "domain": "suppression", "metric": str(key), "value": numeric})
    if isinstance(result.layout_gradient, dict):
        for key, value in result.layout_gradient.items():
            rows.append({"source": "layout_gradient", "domain": "dsv", "metric": key, "value": float(value)})
    rows.append(
        {
            "source": "performance",
            "domain": "-",
            "metric": "needed_current_layout",
            "value": float(result.needed_current_layout),
        }
    )
    rows.append(
        {"source": "performance", "domain": "-", "metric": "potential_step", "value": float(result.potential_step)}
    )
    rows.append({"source": "performance", "domain": "-", "metric": "num_levels", "value": float(result.num_levels)})
    if result.resistance is not None and len(result.resistance):
        rows.append(
            {
                "source": "electrical",
                "domain": "-",
                "metric": "same_current_total_resistance",
                "value": float(np.sum(result.resistance)),
            }
        )
    if result.inductance_matrix is not None and np.size(result.inductance_matrix):
        matrix = np.asarray(result.inductance_matrix, dtype=float)
        rows.append(
            {
                "source": "electrical",
                "domain": "-",
                "metric": "same_current_total_inductance",
                "value": float(np.sum(matrix)),
            }
        )
        for row_ind in range(matrix.shape[0]):
            for col_ind in range(row_ind + 1, matrix.shape[1]):
                rows.append(
                    {
                        "source": "electrical",
                        "domain": "-",
                        "metric": f"mutual_inductance_part{row_ind + 1}_part{col_ind + 1}",
                        "value": float(matrix[row_ind, col_ind]),
                    }
                )
    for part_ind, part in enumerate(result.coil_parts, start=1):
        rows.append(
            {"source": f"part{part_ind}", "domain": "-", "metric": "coil_length", "value": float(part.coil_length)}
        )
        rows.append(
            {
                "source": f"part{part_ind}",
                "domain": "-",
                "metric": "coil_resistance",
                "value": float(part.coil_resistance),
            }
        )
        rows.append(
            {
                "source": f"part{part_ind}",
                "domain": "-",
                "metric": "coil_inductance",
                "value": float(part.coil_inductance),
            }
        )
    return rows


def contour_orientation_diagnostics_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    """Return one row per contour with chirality/current-orientation diagnostics."""

    from .layout import contour_orientation_diagnostics

    rows: list[dict[str, Any]] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        for row in contour_orientation_diagnostics(part):
            output_row = {"part_index": part_ind}
            output_row.update(row)
            rows.append(output_row)
    return rows


def save_contour_orientation_diagnostics(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
) -> list[Path]:
    """Save contour chirality/current-orientation diagnostics as JSON and CSV."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = contour_orientation_diagnostics_rows(result)
    json_path = output / f"{prefix}_contour_orientation_diagnostics.json"
    csv_path = output / f"{prefix}_contour_orientation_diagnostics.csv"
    _write(json_path, json.dumps(rows, indent=2))
    _write_diagnostics_csv(csv_path, rows)
    return [json_path, csv_path]


def save_field_error_diagnostics(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
) -> list[Path]:
    """Save field-error, gradient, and scalar performance diagnostics."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = field_error_diagnostics_rows(result)
    json_path = output / f"{prefix}_field_error_diagnostics.json"
    csv_path = output / f"{prefix}_field_error_diagnostics.csv"
    _write(json_path, json.dumps(rows, indent=2))
    _write_diagnostics_csv(csv_path, rows)
    return [json_path, csv_path]


def save_wire_isometric(
    part: CoilPart,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
) -> Path:
    """Save the connected wire path in a fixed isometric view."""

    style = style or SvgStyle()
    _require_wire(part)
    mesh_points = _isometric_projection(part.mesh.vertices)
    wire_points = _isometric_projection(part.wire_path.v)
    scene = _SvgScene.from_points(
        _stack_nonempty([mesh_points, wire_points]),
        style,
        title=_svg_quantity_title("Connected Coil Wire Path", "Isometric View", COORDINATE_UNIT),
    )
    scene.mesh(mesh_points, part.mesh.faces, stroke=style.mesh_stroke, stroke_width=0.35)
    scene.polyline(wire_points, stroke=style.wire_stroke, stroke_width=2.0)
    if part.shift_array is not None and np.any(part.shift_array > 0):
        _draw_shift_overlay(scene, wire_points, part.shift_array, style)
    scene.legend(
        [("connected wire path", style.wire_stroke), ("normal-offset return path section", style.shifted_stroke)]
    )
    _write(path, scene.finish())
    return Path(path)


def save_wire_projection(
    part: CoilPart,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
) -> Path:
    style = style or SvgStyle()
    _require_wire(part)
    mesh_points = part.mesh.vertices[:, axes]
    wire_points = part.wire_path.v[:, axes]
    scene = _SvgScene.from_points(
        _stack_nonempty([mesh_points, wire_points]),
        style,
        title=_svg_quantity_title("Connected Coil Wire Path", _projection_title(axes), COORDINATE_UNIT),
    )
    scene.mesh(mesh_points, part.mesh.faces, stroke=style.mesh_stroke, stroke_width=0.35)
    scene.polyline(wire_points, stroke=style.wire_stroke, stroke_width=2.0)
    scene.axis_labels(*_coordinate_axis_labels(axes))
    if part.shift_array is not None and np.any(part.shift_array > 0):
        _draw_shift_overlay(scene, wire_points, part.shift_array, style)
    _write(path, scene.finish())
    return Path(path)


def save_contours_projection(
    part: CoilPart,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
    label_contours: bool = False,
) -> Path:
    """Save raw extracted contours before topological grouping."""

    return _save_contours_view(
        part,
        path,
        title=_svg_quantity_title("Raw Stream-Function Contours", _projection_title(axes), COORDINATE_UNIT),
        project=lambda points: np.asarray(points, dtype=float)[:, axes],
        style=style,
        label_contours=label_contours,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_contours_isometric(
    part: CoilPart,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
    label_contours: bool = False,
) -> Path:
    """Save raw extracted contours in a fixed isometric view."""

    return _save_contours_view(
        part,
        path,
        title=_svg_quantity_title("Raw Stream-Function Contours", "Isometric View", COORDINATE_UNIT),
        project=_isometric_projection,
        style=style,
        label_contours=label_contours,
        axis_labels=None,
    )


def save_contour_group_projection(
    part: CoilPart,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
    label_groups: bool = True,
) -> Path:
    """Save unconnected contours colored by topological loop group."""

    return _save_contour_group_view(
        part,
        path,
        title=f"Contour Groups {_projection_code(axes)} - Topological Loop Groups [{COORDINATE_UNIT}]",
        project=lambda points: np.asarray(points, dtype=float)[:, axes],
        style=style,
        label_groups=label_groups,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_contour_group_isometric(
    part: CoilPart,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
    label_groups: bool = True,
) -> Path:
    """Save unconnected contours colored by group in a fixed isometric view."""

    return _save_contour_group_view(
        part,
        path,
        title=f"Contour Groups ISO - Topological Loop Groups [{COORDINATE_UNIT}]",
        project=_isometric_projection,
        style=style,
        label_groups=label_groups,
        axis_labels=None,
    )


def save_stream_function_projection(
    part: CoilPart,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
) -> Path:
    """Save the optimized stream function as a colored surface projection."""

    return _save_face_scalar_view(
        part,
        _face_average_stream_function(part),
        path,
        title=_svg_quantity_title(
            "Optimized Stream Function on Coil Surface", _projection_title(axes), STREAM_FUNCTION_UNIT
        ),
        project=lambda points: np.asarray(points, dtype=float)[:, axes],
        style=style,
        scalar_label=format_label_with_unit("stream function, robust 99% display", STREAM_FUNCTION_UNIT),
        axis_labels=_coordinate_axis_labels(axes),
        display_percentile=99.0,
        nonnegative=False,
    )


def save_stream_function_isometric(
    part: CoilPart,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
) -> Path:
    """Save the optimized stream function in a fixed isometric view."""

    return _save_face_scalar_view(
        part,
        _face_average_stream_function(part),
        path,
        title=_svg_quantity_title("Optimized Stream Function on Coil Surface", "Isometric View", STREAM_FUNCTION_UNIT),
        project=_isometric_projection,
        style=style,
        scalar_label=format_label_with_unit("stream function, robust 99% display", STREAM_FUNCTION_UNIT),
        axis_labels=None,
        display_percentile=99.0,
        nonnegative=False,
    )


def save_current_density_projection(
    part: CoilPart,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
) -> Path:
    """Save the current-density magnitude per surface face."""

    return _save_face_scalar_view(
        part,
        _current_density_magnitude(part),
        path,
        title=_svg_quantity_title(
            "Surface Current-Density Magnitude |J|", _projection_title(axes), CURRENT_DENSITY_UNIT
        ),
        project=lambda points: np.asarray(points, dtype=float)[:, axes],
        style=style,
        scalar_label=format_label_with_unit("|J|, robust 99.5% display", CURRENT_DENSITY_UNIT),
        axis_labels=_coordinate_axis_labels(axes),
        display_percentile=99.5,
        nonnegative=True,
    )


def save_current_density_isometric(
    part: CoilPart,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
) -> Path:
    """Save the current-density magnitude in a fixed isometric view."""

    return _save_face_scalar_view(
        part,
        _current_density_magnitude(part),
        path,
        title=_svg_quantity_title("Surface Current-Density Magnitude |J|", "Isometric View", CURRENT_DENSITY_UNIT),
        project=_isometric_projection,
        style=style,
        scalar_label=format_label_with_unit("|J|, robust 99.5% display", CURRENT_DENSITY_UNIT),
        axis_labels=None,
        display_percentile=99.5,
        nonnegative=True,
    )


def save_target_field_map(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "target",
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
) -> Path:
    """Save a target-point scalar map for target, layout, stream, or layout-error Bz."""

    scalar_map = _target_field_map(result, source)
    points = result.target_field.coords[:, axes]
    return _save_point_scalar_view(
        points,
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("Target-Point Bz Field Map", scalar_map, _projection_title(axes)),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_target_field_map_isometric(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "target",
    style: SvgStyle | None = None,
) -> Path:
    """Save a target-point scalar map in a fixed isometric view."""

    scalar_map = _target_field_map(result, source)
    return _save_point_scalar_view(
        _isometric_projection(result.target_field.coords),
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("Target-Point Bz Field Map", scalar_map, "Isometric View"),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=None,
    )


def save_shim_pointcloud_before_after(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "after",
    domain: str = "dsv",
    color_limit_abs: float | None = None,
    style: SvgStyle | None = None,
) -> Path:
    """Save a before/after shim point-cloud comparison in a fixed isometric view."""

    style = style or SvgStyle()
    coords = np.asarray(result.target_field.coords, dtype=float)
    maps = shim_field_scalar_maps(result)
    source_key = _shim_source_key(source)
    entries = (
        ("before", maps["before_inhomogeneity"]),
        ("after", maps[source_key]),
    )
    domain_mask = target_domain_mask(result, domain)
    projected = _isometric_projection(coords)
    finite_masks = [
        domain_mask
        & np.isfinite(np.asarray(scalar_map.values, dtype=float).reshape(-1))
        & np.all(np.isfinite(projected), axis=1)
        for _, scalar_map in entries
    ]
    if not any(np.any(mask) for mask in finite_masks):
        raise ValueError("Shim point-cloud comparison has no finite samples.")
    combined_mask = np.logical_or.reduce(finite_masks)
    if color_limit_abs is None:
        color_limit_abs = _shim_reference_color_limit_abs(result, source=source, domain=domain)
    if color_limit_abs is None:
        color_limit_abs = max(
            float(np.nanmax(np.abs(np.asarray(scalar_map.values, dtype=float).reshape(-1)[mask])))
            for (_, scalar_map), mask in zip(entries, finite_masks)
            if np.any(mask)
        )
    limit = max(float(color_limit_abs), 1e-30)

    x_range = _finite_range(projected[combined_mask, 0])
    y_range = _finite_range(projected[combined_mask, 1])
    gap = 28.0
    top = 92.0
    footer_h = 104.0
    panel_w = (style.width - 2.0 * style.margin - gap) / 2.0
    panel_h = max(120.0, style.height - top - footer_h)
    boxes = (
        (float(style.margin), top, panel_w, panel_h),
        (float(style.margin) + panel_w + gap, top, panel_w, panel_h),
    )
    point_radius = 3.2 if int(np.count_nonzero(combined_mask)) <= 3000 else 2.3

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{style.width}" height="{style.height}" viewBox="0 0 {style.width} {style.height}">',
        f'<rect width="100%" height="100%" fill="{style.background}"/>',
        f'<text x="{style.margin}" y="{style.margin - 12}" font-family="Arial, sans-serif" font-size="17" fill="{style.text_color}">Shim Target-Point B0 Inhomogeneity: Before / After</text>',
        (
            f'<text x="{style.margin}" y="{style.margin + 12}" font-family="Arial, sans-serif" font-size="12" fill="{style.text_color}">'
            f"Domain: {escape(str(domain))}; fixed isometric projection; shared range +/-{limit:.4g} {escape(entries[0][1].unit)}</text>"
        ),
    ]
    for (source_name, scalar_map), finite, box in zip(entries, finite_masks, boxes):
        values = np.asarray(scalar_map.values, dtype=float).reshape(-1)
        stats = shim_homogeneity_stats(result, source=source_name if source_name == "before" else source, domain=domain)
        title_x = box[0] + box[2] / 2.0
        elements.append(
            f'<text x="{title_x:.3f}" y="{top - 24:.3f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="{style.text_color}">{escape(scalar_map.label_with_unit)}</text>'
        )
        metric_label = _shim_abs_metric_label(scalar_map.key)
        elements.append(
            f'<text x="{title_x:.3f}" y="{top - 8:.3f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{style.text_color}">'
            f"RMS {float(stats['rms']):.4g}, max |{escape(metric_label)}| {float(stats['max_abs']):.4g} {escape(str(stats['unit']))}</text>"
        )
        elements.append(
            f'<rect x="{box[0]:.3f}" y="{box[1]:.3f}" width="{box[2]:.3f}" height="{box[3]:.3f}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8"/>'
        )
        mapped = _map_points_to_svg_box(projected[finite], box, x_range=x_range, y_range=y_range, pad=12.0)
        for point, value in zip(mapped, values[finite]):
            color = _color_ramp(_normalize(float(value), -limit, limit))
            elements.append(
                f'<circle cx="{point[0]:.3f}" cy="{point[1]:.3f}" r="{point_radius:g}" fill="{color}" stroke="#ffffff" stroke-width="0.2" opacity="0.9"/>'
            )
    elements.extend(
        _svg_diverging_colorbar(
            style,
            y=style.height - 58.0,
            vmin=-limit,
            vmax=limit,
            label=format_label_with_unit("B0 inhomogeneity / residual", entries[0][1].unit),
        )
    )
    elements.append("</svg>")
    _write(path, "\n".join(elements))
    return Path(path)


def save_gradient_map(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "layout",
    component: str = "magnitude",
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
    neighbor_count: int = 12,
) -> Path:
    """Save a target-point gradient component map."""

    maps = gradient_component_scalar_maps(result, component=component, neighbor_count=neighbor_count)
    scalar_map = maps[str(source)]
    return _save_point_scalar_view(
        result.target_field.coords[:, axes],
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("DSV Target-Point Bz Gradient Map", scalar_map, _projection_title(axes)),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_gradient_map_isometric(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "layout",
    component: str = "magnitude",
    style: SvgStyle | None = None,
    neighbor_count: int = 12,
) -> Path:
    """Save a target-point gradient component map in a fixed isometric view."""

    maps = gradient_component_scalar_maps(result, component=component, neighbor_count=neighbor_count)
    scalar_map = maps[str(source)]
    return _save_point_scalar_view(
        _isometric_projection(result.target_field.coords),
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("DSV Target-Point Bz Gradient Map", scalar_map, "Isometric View"),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=None,
    )


def save_gradient_relative_error_map(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "layout_vs_target",
    component: str = "magnitude",
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
    neighbor_count: int = 12,
) -> Path:
    """Save a relative-percent target-point gradient-error component map."""

    maps = gradient_relative_error_scalar_maps(result, component=component, neighbor_count=neighbor_count)
    scalar_map = maps[str(source)]
    return _save_point_scalar_view(
        result.target_field.coords[:, axes],
        scalar_map.values,
        path,
        title=_scalar_map_svg_title(
            "DSV Target-Point Relative Bz Gradient Error Map", scalar_map, _projection_title(axes)
        ),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_gradient_relative_error_map_isometric(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "layout_vs_target",
    component: str = "magnitude",
    style: SvgStyle | None = None,
    neighbor_count: int = 12,
) -> Path:
    """Save a relative-percent target-point gradient-error component map in a fixed isometric view."""

    maps = gradient_relative_error_scalar_maps(result, component=component, neighbor_count=neighbor_count)
    scalar_map = maps[str(source)]
    return _save_point_scalar_view(
        _isometric_projection(result.target_field.coords),
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("DSV Target-Point Relative Bz Gradient Error Map", scalar_map, "Isometric View"),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=None,
    )


def save_field_error_map(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "layout_vs_target",
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
) -> Path:
    """Save a target-point field-error map."""

    scalar_map = field_error_scalar_maps(result)[str(source)]
    return _save_point_scalar_view(
        result.target_field.coords[:, axes],
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("DSV Target-Point Absolute Bz Error Map", scalar_map, _projection_title(axes)),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_field_error_map_isometric(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "layout_vs_target",
    style: SvgStyle | None = None,
) -> Path:
    """Save a target-point field-error map in a fixed isometric view."""

    scalar_map = field_error_scalar_maps(result)[str(source)]
    return _save_point_scalar_view(
        _isometric_projection(result.target_field.coords),
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("DSV Target-Point Absolute Bz Error Map", scalar_map, "Isometric View"),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=None,
    )


def save_field_relative_error_map(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "layout_vs_target",
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
) -> Path:
    """Save a relative-percent target-point Bz error map."""

    scalar_map = field_relative_error_scalar_maps(result)[str(source)]
    return _save_point_scalar_view(
        result.target_field.coords[:, axes],
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("DSV Target-Point Relative Bz Error Map", scalar_map, _projection_title(axes)),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_field_relative_error_map_isometric(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "layout_vs_target",
    style: SvgStyle | None = None,
) -> Path:
    """Save a relative-percent target-point Bz error map in a fixed isometric view."""

    scalar_map = field_relative_error_scalar_maps(result)[str(source)]
    return _save_point_scalar_view(
        _isometric_projection(result.target_field.coords),
        scalar_map.values,
        path,
        title=_scalar_map_svg_title("DSV Target-Point Relative Bz Error Map", scalar_map, "Isometric View"),
        style=style,
        scalar_label=scalar_map.label_with_unit,
        axis_labels=None,
    )


def save_shim_homogeneity_mosaic(
    result: CoilGenResult,
    path: str | Path,
    *,
    source: str = "after",
    axis: int = 2,
    slice_count: int = 12,
    domain: str = "dsv",
    grid_size: int = 44,
    min_points_per_slice: int = 24,
    slab_thickness_m: float | None = None,
    color_limit_abs: float | None = None,
    style: SvgStyle | None = None,
) -> Path:
    """Save filled before/after shimming inhomogeneity slices."""

    style = style or SvgStyle(width=1120, height=820)
    axis = int(axis)
    coords = np.asarray(result.target_field.coords, dtype=float)
    maps = shim_field_scalar_maps(result)
    source_key = _shim_source_key(source)
    scalar_map = maps[source_key]
    values = np.asarray(scalar_map.values, dtype=float).reshape(-1)
    if len(values) != len(coords):
        raise ValueError("Shim residual values must match target coordinate count.")
    domain_mask = target_domain_mask(result, domain)
    finite = domain_mask & np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    if not np.any(finite):
        raise ValueError("Shim homogeneity mosaic has no finite samples.")

    slices = interpolated_slice_mosaic(
        coords,
        values,
        axis=axis,
        slice_count=slice_count,
        sample_mask=finite,
        grid_size=grid_size,
        min_points=min_points_per_slice,
        slab_thickness=slab_thickness_m,
    )
    if not slices:
        raise ValueError("Shim homogeneity mosaic could not interpolate any slices.")
    if color_limit_abs is None:
        color_limit_abs = _shim_reference_color_limit_abs(result, source=source, domain=domain)
    limit = (
        max(float(color_limit_abs), 1e-30)
        if color_limit_abs is not None
        else max(float(np.nanmax(np.abs(values[finite]))), 1e-30)
    )
    stats = shim_homogeneity_stats(result, source=source, domain=domain)
    metric_label = _shim_abs_metric_label(source_key)

    columns = int(np.ceil(np.sqrt(len(slices))))
    rows = int(np.ceil(len(slices) / columns))
    header_h = 86
    footer_h = 76
    gutter = 16
    tile_w = (style.width - 2 * style.margin - gutter * (columns - 1)) / columns
    tile_h = (style.height - header_h - footer_h - style.margin - gutter * (rows - 1)) / rows
    tile_size = max(20.0, min(tile_w, tile_h))
    mosaic_w = columns * tile_size + (columns - 1) * gutter
    left = (style.width - mosaic_w) / 2
    top = header_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{style.width}" height="{style.height}" viewBox="0 0 {style.width} {style.height}">',
        f'<rect width="100%" height="100%" fill="{style.background}"/>',
        f'<text x="{style.margin}" y="30" font-family="Arial, sans-serif" font-size="17" fill="{style.text_color}">B0 Inhomogeneity Filled Z-Slice Mosaic: {escape(scalar_map.label_with_unit)}</text>',
        (
            f'<text x="{style.margin}" y="55" font-family="Arial, sans-serif" font-size="12" fill="{style.text_color}">'
            f"Domain: {escape(str(domain))}; interpolated {escape(axis_name(axis))}-slabs; "
            f"RMS {float(stats['rms']):.4g}, p95 |{escape(metric_label)}| {float(stats['p95_abs']):.4g}, "
            f"max |{escape(metric_label)}| {float(stats['max_abs']):.4g} {escape(str(stats['unit']))}</text>"
        ),
    ]

    for index, interpolated in enumerate(slices):
        row = index // columns
        col = index % columns
        x0 = left + col * (tile_size + gutter)
        y0 = top + row * (tile_size + gutter)
        mask = interpolated.mask & finite
        elements.append(
            f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{tile_size:.3f}" height="{tile_size:.3f}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x0 + 6:.3f}" y="{y0 + 15:.3f}" font-family="Arial, sans-serif" font-size="10" fill="#475569">'
            f"{escape(axis_name(axis))}={interpolated.value:.4g} m; n={int(np.count_nonzero(mask))}</text>"
        )
        plot_box = (x0 + 12.0, y0 + 22.0, tile_size - 24.0, tile_size - 34.0)
        elements.append(
            f'<rect x="{plot_box[0]:.3f}" y="{plot_box[1]:.3f}" width="{plot_box[2]:.3f}" height="{plot_box[3]:.3f}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.7"/>'
        )
        grid = np.asarray(interpolated.values, dtype=float)
        valid_grid = np.isfinite(grid)
        if np.any(valid_grid):
            rows_grid, cols_grid = grid.shape
            cell_w = plot_box[2] / max(cols_grid, 1)
            cell_h = plot_box[3] / max(rows_grid, 1)
            for iy in range(rows_grid):
                py = plot_box[1] + (rows_grid - iy - 1) * cell_h
                for ix in range(cols_grid):
                    value = grid[iy, ix]
                    if not np.isfinite(value):
                        continue
                    px = plot_box[0] + ix * cell_w
                    color = _color_ramp(_normalize(float(value), -limit, limit))
                    elements.append(
                        f'<rect x="{px:.3f}" y="{py:.3f}" width="{cell_w + 0.05:.3f}" height="{cell_h + 0.05:.3f}" fill="{color}" stroke="none"/>'
                    )
        else:
            elements.append(
                f'<text x="{x0 + tile_size / 2:.3f}" y="{y0 + tile_size / 2:.3f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#64748b">No samples</text>'
            )
    elements.extend(
        _svg_diverging_colorbar(
            style,
            y=style.height - footer_h + 24,
            vmin=-limit,
            vmax=limit,
            label=scalar_map.label_with_unit,
        )
    )
    elements.append("</svg>")
    _write(path, "\n".join(elements))
    return Path(path)


def save_error_metric_bars(
    result: CoilGenResult,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
) -> Path:
    """Save a compact bar plot of field-error metrics."""

    rows = [(key, float(value)) for key, value in result.error_vals.items() if np.isfinite(float(value))]
    if not rows:
        raise ValueError("Result has no finite field-error metrics.")
    return _save_horizontal_bar_chart(rows, path, title="Coil Design DSV Field-Error Metrics [%]", style=style)


def save_performance_summary(
    result: CoilGenResult,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
) -> Path:
    """Save scalar layout/performance values as a compact bar summary."""

    rows: list[tuple[str, float]] = [
        ("needed_current_layout", float(result.needed_current_layout)),
        ("potential_step", float(result.potential_step)),
        ("num_levels", float(result.num_levels)),
    ]
    if result.coil_lengths is not None and len(result.coil_lengths):
        rows.append(("coil_length_total", float(np.sum(result.coil_lengths))))
    if result.resistance is not None and len(result.resistance):
        rows.append(("coil_resistance_total", float(np.sum(result.resistance))))
    if result.inductance_matrix is not None and np.size(result.inductance_matrix):
        rows.append(
            ("coil_inductance_same_current_total", float(np.sum(np.asarray(result.inductance_matrix, dtype=float))))
        )
    for part_ind, part in enumerate(result.coil_parts, start=1):
        rows.extend(
            [
                (f"part{part_ind}_coil_length", float(part.coil_length)),
                (f"part{part_ind}_coil_resistance", float(part.coil_resistance)),
                (f"part{part_ind}_coil_inductance", float(part.coil_inductance)),
                (f"part{part_ind}_minimal_contour_distance", float(part.minimal_contour_distance)),
            ]
        )
    rows = [(label, value) for label, value in rows if np.isfinite(value)]
    if not rows:
        raise ValueError("Result has no finite performance metrics.")
    return _save_horizontal_bar_chart(
        rows, path, title="Coil Design Electrical and Geometric Performance Summary", style=style
    )


def save_swept_body_projection(
    part: CoilPart,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
) -> Path:
    """Save the swept conductor body mesh in a coordinate projection."""

    _require_swept_body(part)
    return _save_mesh_view(
        part.swept_body,
        path,
        title=_svg_quantity_title("Swept Conductor Body Mesh", _projection_title(axes), COORDINATE_UNIT),
        project=lambda points: np.asarray(points, dtype=float)[:, axes],
        style=style,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_swept_body_isometric(
    part: CoilPart,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
) -> Path:
    """Save the swept conductor body mesh in a fixed isometric view."""

    _require_swept_body(part)
    return _save_mesh_view(
        part.swept_body,
        path,
        title=_svg_quantity_title("Swept Conductor Body Mesh", "Isometric View", COORDINATE_UNIT),
        project=_isometric_projection,
        style=style,
        axis_labels=None,
    )


def save_interconnect_debug_projection(
    part: CoilPart,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
    label_points: bool = True,
) -> Path:
    """Save contours with the cut and interconnection markers used by routing."""

    return _save_interconnect_debug_view(
        part,
        path,
        title=f"Interconnect Debug {_projection_code(axes)} - Contour Cuts and Group Connectors [{COORDINATE_UNIT}]",
        project=lambda points: np.asarray(points, dtype=float)[:, axes],
        style=style,
        label_points=label_points,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_interconnect_debug_isometric(
    part: CoilPart,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
    label_points: bool = True,
) -> Path:
    """Save contours with routing cut markers in a fixed isometric view."""

    return _save_interconnect_debug_view(
        part,
        path,
        title=f"Interconnect Debug ISO - Contour Cuts and Group Connectors [{COORDINATE_UNIT}]",
        project=_isometric_projection,
        style=style,
        label_points=label_points,
        axis_labels=None,
    )


def save_within_group_cut_choices(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "coil",
    style: SvgStyle | None = None,
) -> list[Path]:
    """Save high/low within-group cut rail choice plots for every coil part."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = result.input_data if isinstance(result.input_data, CoilGenConfig) else CoilGenConfig()
    saved: list[Path] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        part_prefix = f"{prefix}_part{part_ind}"
        for suffix, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            saved.append(
                save_within_group_cut_choice_projection(
                    part,
                    config,
                    output / f"{part_prefix}_within_group_cut_choices_{suffix}.svg",
                    axes=axes,
                    style=style,
                )
            )
        saved.append(
            save_within_group_cut_choice_isometric(
                part,
                config,
                output / f"{part_prefix}_within_group_cut_choices_iso.svg",
                style=style,
            )
        )
    return saved


def save_within_group_cut_choice_projection(
    part: CoilPart,
    config: CoilGenConfig,
    path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    style: SvgStyle | None = None,
    label_groups: bool = True,
) -> Path:
    """Save high/low within-group cut rails in a coordinate projection."""

    return _save_within_group_cut_choice_view(
        part,
        config,
        path,
        title=_svg_quantity_title(
            "Within-Group Opening-Cut Candidate Selection", _projection_title(axes), COORDINATE_UNIT
        ),
        project=lambda points: np.asarray(points, dtype=float)[:, axes],
        style=style,
        label_groups=label_groups,
        axis_labels=_coordinate_axis_labels(axes),
    )


def save_within_group_cut_choice_isometric(
    part: CoilPart,
    config: CoilGenConfig,
    path: str | Path,
    *,
    style: SvgStyle | None = None,
    label_groups: bool = True,
) -> Path:
    """Save high/low within-group cut rails in a fixed isometric view."""

    return _save_within_group_cut_choice_view(
        part,
        config,
        path,
        title=_svg_quantity_title("Within-Group Opening-Cut Candidate Selection", "Isometric View", COORDINATE_UNIT),
        project=_isometric_projection,
        style=style,
        label_groups=label_groups,
        axis_labels=None,
    )


def _save_contours_view(
    part: CoilPart,
    path: str | Path,
    *,
    title: str,
    project: Callable[[np.ndarray], np.ndarray],
    style: SvgStyle | None,
    label_contours: bool,
    axis_labels: tuple[str, str] | None,
) -> Path:
    style = style or SvgStyle()
    _require_contours(part)
    mesh_points = project(part.mesh.vertices)
    contour_points = [project(_closed_for_display(loop.v)) for loop in part.contour_lines]
    source_points = (
        [project(_closed_for_display(loop.v)) for loop in part.contour_sparsification.source_contours]
        if part.contour_sparsification is not None
        else []
    )
    scene = _SvgScene.from_points(
        _stack_nonempty([mesh_points, *source_points, *contour_points]),
        style,
        title=title,
    )
    scene.mesh(mesh_points, part.mesh.faces, stroke=style.mesh_stroke, stroke_width=0.25)
    if axis_labels is not None:
        scene.axis_labels(*axis_labels)

    potentials = [float(loop.potential) for loop in part.contour_lines]
    vmin, vmax = _finite_min_max(potentials)
    for points in source_points:
        scene.polyline(
            points,
            stroke="#6b7280",
            stroke_width=0.9,
            opacity=0.35,
            stroke_dasharray="4 4",
        )
    for contour_ind, loop in enumerate(part.contour_lines):
        color = _color_ramp(_normalize(float(loop.potential), vmin, vmax))
        points = project(_closed_for_display(loop.v))
        width = 1.9 * np.sqrt(max(int(loop.turn_multiplicity), 1))
        scene.polyline(points, stroke=color, stroke_width=width, opacity=0.92)
        if label_contours and len(points):
            multiplier = f" x{loop.turn_multiplicity}" if loop.turn_multiplicity > 1 else ""
            scene.text_at(points.mean(axis=0), f"L{contour_ind}{multiplier}", fill=color, font_size=9)
    scalar_label = format_label_with_unit("stream function", STREAM_FUNCTION_UNIT)
    scene.legend(
        [
            (f"{scalar_label} min {vmin:.3g}", _color_ramp(0.0)),
            (f"{scalar_label} max {vmax:.3g}", _color_ramp(1.0)),
            (f"{part.nominal_contour_count} geometries / {part.physical_turn_count} turns", "#111827"),
        ]
    )
    _write(path, scene.finish())
    return Path(path)


def _save_contour_group_view(
    part: CoilPart,
    path: str | Path,
    *,
    title: str,
    project: Callable[[np.ndarray], np.ndarray],
    style: SvgStyle | None,
    label_groups: bool,
    axis_labels: tuple[str, str] | None,
) -> Path:
    style = style or SvgStyle()
    _require_grouped_contours(part)
    entries = _grouped_contour_entries(part)
    mesh_points = project(part.mesh.vertices)
    contour_points = [project(_closed_for_display(loop.v)) for _, _, loop in entries]
    scene = _SvgScene.from_points(_stack_nonempty([mesh_points, *contour_points]), style, title=title)
    scene.mesh(mesh_points, part.mesh.faces, stroke=style.mesh_stroke, stroke_width=0.3)
    if axis_labels is not None:
        scene.axis_labels(*axis_labels)

    legend: list[tuple[str, str]] = []
    seen_groups: set[int | None] = set()
    group_points: dict[int, list[np.ndarray]] = {}
    for group_ind, loop_ind, loop in entries:
        color = _group_color(group_ind)
        scene.polyline(project(_closed_for_display(loop.v)), stroke=color, stroke_width=2.1, opacity=0.95)
        if group_ind is not None:
            group_points.setdefault(group_ind, []).append(project(loop.v))
        if group_ind not in seen_groups:
            seen_groups.add(group_ind)
            legend.append((_group_label(part, group_ind), color))

    if label_groups:
        for group_ind, arrays in group_points.items():
            center = np.vstack(arrays).mean(axis=0)
            scene.text_at(center, f"G{group_ind}", fill=_group_color(group_ind), font_size=11)

    scene.legend(legend)
    _write(path, scene.finish())
    return Path(path)


def _save_within_group_cut_choice_view(
    part: CoilPart,
    config: CoilGenConfig,
    path: str | Path,
    *,
    title: str,
    project: Callable[[np.ndarray], np.ndarray],
    style: SvgStyle | None,
    label_groups: bool,
    axis_labels: tuple[str, str] | None,
) -> Path:
    from .layout import (
        _build_within_group_cut_rail,
        _inter_group_cut_alignment_state,
        _within_group_cut_width,
        generate_group_cutshapes,
    )

    style = style or SvgStyle()
    _require_grouped_contours(part)
    entries = _grouped_contour_entries(part)
    mesh_points = project(part.mesh.vertices)
    contour_points = [project(_closed_for_display(loop.v)) for _, _, loop in entries]
    rail_entries = []
    rail_points: list[np.ndarray] = []
    align_axial_rings = _inter_group_cut_alignment_state(part, config) is not None

    for group_ind, group in enumerate(part.groups):
        if len(group.loops) <= 1:
            continue
        group_center_v = (
            part.group_centers_v[group_ind]
            if part.group_centers_v is not None and group_ind < len(part.group_centers_v)
            else np.vstack([loop.v for loop in group.loops]).mean(axis=0)
        )
        try:
            cut_infos = generate_group_cutshapes(
                group,
                group_center_v,
                part.mesh,
                _within_group_cut_width(config),
                config.b_0_direction,
                cut_plane_definition="nearest",
                cut_height_ratio=0.5,
                align_axial_rings=align_axial_rings,
            )
            high = _build_within_group_cut_rail(group, cut_infos, "high", config)
            low = _build_within_group_cut_rail(group, cut_infos, "low", config)
        except ValueError:
            continue
        rail_entries.append((group_ind, high, low))
        for rail in (high, low):
            rail_points.append(project(rail.connected_group.return_path_v))
            for cut_v in rail.opening_cuts:
                rail_points.append(project(cut_v))

    scene = _SvgScene.from_points(_stack_nonempty([mesh_points, *contour_points, *rail_points]), style, title=title)
    scene.mesh(mesh_points, part.mesh.faces, stroke=style.mesh_stroke, stroke_width=0.25)
    if axis_labels is not None:
        scene.axis_labels(*axis_labels)

    for group_ind, _, loop in entries:
        scene.polyline(
            project(_closed_for_display(loop.v)), stroke=_group_color(group_ind), stroke_width=1.4, opacity=0.32
        )

    for group_ind, high, low in rail_entries:
        selected_side = part.groups[group_ind].within_group_cut_side or ("high" if high.score <= low.score else "low")
        _draw_cut_choice_rail(scene, high, project, side="high", selected=selected_side == "high")
        _draw_cut_choice_rail(scene, low, project, side="low", selected=selected_side == "low")
        if label_groups:
            label_points = (
                high.connected_group.return_path_v if selected_side == "high" else low.connected_group.return_path_v
            )
            scores = f"H {high.score:.3g} / L {low.score:.3g}"
            scene.text_at(
                project(label_points).mean(axis=0),
                f"G{group_ind} {selected_side} {scores}",
                fill="#111827",
                font_size=10,
            )

    scene.legend(
        [
            ("selected rail", "#16a34a"),
            ("high candidate", "#2563eb"),
            ("low candidate", "#dc2626"),
            ("contours", "#6b7280"),
        ]
    )
    _write(path, scene.finish())
    return Path(path)


def _draw_cut_choice_rail(
    scene: "_SvgScene",
    rail,
    project: Callable[[np.ndarray], np.ndarray],
    *,
    side: str,
    selected: bool,
) -> None:
    candidate_color = "#2563eb" if side == "high" else "#dc2626"
    stroke = "#16a34a" if selected else candidate_color
    width = 3.4 if selected else 1.8
    opacity = 0.95 if selected else 0.45
    dash = None if selected else "6 5"
    return_path = project(rail.connected_group.return_path_v)
    if len(return_path) >= 2:
        scene.polyline(return_path, stroke=stroke, stroke_width=width, opacity=opacity, stroke_dasharray=dash)
    for cut_v in rail.opening_cuts:
        cut = project(cut_v)
        if len(cut) == 0:
            continue
        scene.polyline(cut, stroke=stroke, stroke_width=width * 0.85, opacity=opacity, stroke_dasharray=dash)
        for point in cut:
            scene.circle_at(
                point, radius=3.6 if selected else 2.8, fill="#ffffff", stroke=stroke, stroke_width=1.3, opacity=opacity
            )


def _save_interconnect_debug_view(
    part: CoilPart,
    path: str | Path,
    *,
    title: str,
    project: Callable[[np.ndarray], np.ndarray],
    style: SvgStyle | None,
    label_points: bool,
    axis_labels: tuple[str, str] | None,
) -> Path:
    style = style or SvgStyle()
    _require_grouped_contours(part)
    _require_interconnect_data(part)
    entries = _grouped_contour_entries(part)
    mesh_points = project(part.mesh.vertices)
    contour_points = [project(_closed_for_display(loop.v)) for _, _, loop in entries]
    marker_points = _interconnect_debug_point_arrays(part, project)
    scene = _SvgScene.from_points(_stack_nonempty([mesh_points, *contour_points, *marker_points]), style, title=title)
    scene.mesh(mesh_points, part.mesh.faces, stroke=style.mesh_stroke, stroke_width=0.25)
    if axis_labels is not None:
        scene.axis_labels(*axis_labels)

    for group_ind, _, loop in entries:
        scene.polyline(
            project(_closed_for_display(loop.v)), stroke=_group_color(group_ind), stroke_width=1.5, opacity=0.45
        )

    for group_ind, group in enumerate(part.groups):
        for local_ind, cuts_v in enumerate(group.opening_cuts_v):
            cuts = project(cuts_v)
            if len(cuts) == 0:
                continue
            scene.polyline(cuts, stroke="#f97316", stroke_width=2.2, opacity=0.95)
            for point in cuts:
                scene.circle_at(point, radius=3.7, fill="#ffffff", stroke="#f97316", stroke_width=1.5)
            if label_points:
                loop_label = group.loop_indices[local_ind] if local_ind < len(group.loop_indices) else local_ind
                scene.text_at(np.mean(cuts, axis=0), f"G{group_ind}L{loop_label}", fill="#c2410c", font_size=9)

    for group_ind, connected in enumerate(part.connected_groups):
        return_path = project(connected.return_path_v)
        if len(return_path) == 0:
            continue
        scene.polyline(return_path, stroke="#111827", stroke_width=1.7, opacity=0.95, stroke_dasharray="5 4")
        for point in return_path:
            scene.circle_at(point, radius=3.2, fill="#111827", stroke="#ffffff", stroke_width=1.0)
        if label_points:
            scene.text_at(np.mean(return_path, axis=0), f"R{group_ind}", fill="#111827", font_size=10)

    for cut_ind, cut in enumerate(part.opening_cuts_among_groups):
        cut1 = project(cut.cut1_v)
        cut2 = project(cut.cut2_v)
        connector = project(cut.connection_path_v)
        closing_connector = project(cut.closing_connection_path_v)
        for cut_points in (cut1, cut2):
            if len(cut_points) == 0:
                continue
            scene.polyline(cut_points, stroke="#a855f7", stroke_width=2.4, opacity=0.95)
            for point in cut_points:
                scene.circle_at(point, radius=4.0, fill="#ffffff", stroke="#7e22ce", stroke_width=1.5)
        if len(connector):
            dash = "3 4" if len(connector) <= 2 else None
            scene.polyline(connector, stroke="#7e22ce", stroke_width=2.0, opacity=0.8, stroke_dasharray=dash)
        if len(closing_connector):
            dash = "3 4" if len(closing_connector) <= 2 else None
            scene.polyline(
                closing_connector,
                stroke="#c026d3",
                stroke_width=2.0,
                opacity=0.8,
                stroke_dasharray=dash,
            )
        if len(cut1) and len(cut2):
            centers = np.vstack((np.mean(cut1, axis=0), np.mean(cut2, axis=0)))
            scene.polyline(centers, stroke="#7e22ce", stroke_width=1.8, opacity=0.8, stroke_dasharray="3 4")
            if label_points:
                scene.text_at(np.mean(centers, axis=0), f"A{cut_ind}", fill="#7e22ce", font_size=10)

    scene.legend(
        [
            ("contours", "#1f77b4"),
            ("within-group opening cuts", "#f97316"),
            ("group return path", "#111827"),
            ("inter-group opening cuts", "#a855f7"),
        ]
    )
    _write(path, scene.finish())
    return Path(path)


def _save_face_scalar_view(
    part: CoilPart,
    values: np.ndarray,
    path: str | Path,
    *,
    title: str,
    project: Callable[[np.ndarray], np.ndarray],
    style: SvgStyle | None,
    scalar_label: str,
    axis_labels: tuple[str, str] | None,
    display_percentile: float = 100.0,
    nonnegative: bool = False,
) -> Path:
    style = style or SvgStyle()
    values = np.asarray(values, dtype=float)
    if len(values) != len(part.mesh.faces):
        raise ValueError("Face scalar values must match mesh face count.")
    mesh_points = project(part.mesh.vertices)
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        raise ValueError(f"{scalar_label} has no finite values.")
    vmin, vmax = _finite_display_range(finite_values, percentile=display_percentile, nonnegative=nonnegative)
    scene = _SvgScene.from_points(mesh_points, style, title=title)
    if axis_labels is not None:
        scene.axis_labels(*axis_labels)
    for face, value in zip(part.mesh.faces, values):
        if not np.isfinite(value):
            continue
        color = _color_ramp(_normalize(float(value), vmin, vmax))
        scene.polygon(mesh_points[face], fill=color, stroke="#ffffff", stroke_width=0.15, opacity=0.86)
    scene.mesh(mesh_points, part.mesh.faces, stroke=style.mesh_stroke, stroke_width=0.18)
    _draw_scalar_legend(scene, scalar_label, vmin, vmax)
    _write(path, scene.finish())
    return Path(path)


def _save_point_scalar_view(
    points: np.ndarray,
    values: np.ndarray,
    path: str | Path,
    *,
    title: str,
    style: SvgStyle | None,
    scalar_label: str,
    axis_labels: tuple[str, str] | None,
) -> Path:
    style = style or SvgStyle()
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(points) != len(values):
        raise ValueError("Point scalar values must match point count.")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"{scalar_label} has no finite values.")
    vmin, vmax = _finite_min_max(values[finite])
    scene = _SvgScene.from_points(points[finite], style, title=title)
    if axis_labels is not None:
        scene.axis_labels(*axis_labels)
    for point, value in zip(points[finite], values[finite]):
        color = _color_ramp(_normalize(float(value), vmin, vmax))
        scene.circle_at(point, radius=4.0, fill=color, stroke="#111827", stroke_width=0.35, opacity=0.9)
    _draw_scalar_legend(scene, scalar_label, vmin, vmax)
    _write(path, scene.finish())
    return Path(path)


def _save_mesh_view(
    mesh,
    path: str | Path,
    *,
    title: str,
    project: Callable[[np.ndarray], np.ndarray],
    style: SvgStyle | None,
    axis_labels: tuple[str, str] | None,
) -> Path:
    style = style or SvgStyle()
    points = project(mesh.vertices)
    scene = _SvgScene.from_points(points, style, title=title)
    if axis_labels is not None:
        scene.axis_labels(*axis_labels)
    for face in mesh.faces:
        scene.polygon(points[face], fill="#eef2ff", stroke="#c7d2fe", stroke_width=0.22, opacity=0.58)
    scene.mesh(points, mesh.faces, stroke="#64748b", stroke_width=0.18)
    _write(path, scene.finish())
    return Path(path)


def _save_horizontal_bar_chart(
    rows: Sequence[tuple[str, float]],
    path: str | Path,
    *,
    title: str,
    style: SvgStyle | None,
) -> Path:
    style = style or SvgStyle()
    rows = [(label, float(value)) for label, value in rows if np.isfinite(float(value))]
    if not rows:
        raise ValueError(f"{title} has no finite rows.")
    width = style.width
    height = max(style.height, 96 + 30 * len(rows))
    label_x = style.margin
    bar_x = min(300, max(210, width // 3))
    bar_w = width - bar_x - style.margin
    top = style.margin + 34
    max_abs = max(max(abs(value) for _, value in rows), 1e-30)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{style.background}"/>',
        f'<text x="{style.margin}" y="{style.margin - 12}" font-family="Arial, sans-serif" font-size="16" fill="{style.text_color}">{escape(title)}</text>',
    ]
    for row_ind, (label, value) in enumerate(rows):
        y = top + row_ind * 30
        length = max(1.0, abs(value) / max_abs * bar_w)
        color = _color_ramp(0.5 + 0.5 * np.sign(value) * min(abs(value) / max_abs, 1.0))
        display_label = _metric_label_with_unit(label)
        elements.append(
            f'<text x="{label_x}" y="{y + 14}" font-family="Arial, sans-serif" font-size="12" fill="{style.text_color}">{escape(_short_metric_label(display_label))}</text>'
        )
        elements.append(f'<rect x="{bar_x}" y="{y}" width="{length:.3f}" height="18" fill="{color}" opacity="0.88"/>')
        elements.append(
            f'<text x="{bar_x + length + 8:.3f}" y="{y + 14}" font-family="Arial, sans-serif" font-size="12" fill="{style.text_color}">{value:.4g}</text>'
        )
    elements.append("</svg>")
    _write(path, "\n".join(elements))
    return Path(path)


def save_field_comparison(result: CoilGenResult, path: str | Path, *, style: SvgStyle | None = None) -> Path:
    style = style or SvgStyle()
    fields = field_scalar_maps(result)
    target = fields["target"].values
    layout = fields["layout"].values
    stream = fields["stream"].values
    x = np.arange(len(target), dtype=float)
    points = _stack_nonempty(
        [
            np.column_stack((x, target)),
            np.column_stack((x, layout)),
            np.column_stack((x, stream)),
        ]
    )
    unit = display_field_unit_1a(result)
    scene = _SvgScene.from_points(
        points, style, title=format_label_with_unit("Target-Point Bz Field Comparison per Ampere", unit)
    )
    scene.axes()
    scene.axis_labels("target point index [-]", format_label_with_unit("Bz", unit))
    scene.polyline(np.column_stack((x, target)), stroke="#111827", stroke_width=2.0)
    scene.polyline(np.column_stack((x, layout)), stroke="#d73a49", stroke_width=1.8)
    scene.polyline(np.column_stack((x, stream)), stroke="#1f6feb", stroke_width=1.8)
    scene.legend(
        [("requested target", "#111827"), ("connected wire field", "#d73a49"), ("stream-function field", "#1f6feb")]
    )
    _write(path, scene.finish())
    return Path(path)


def save_l_curve_regularization(result: CoilGenResult, path: str | Path, *, style: SvgStyle | None = None) -> Path:
    style = style or SvgStyle()
    scan = regularization_scan_arrays(result)
    valid = scan["valid"]
    selected_index = scan["selected_index"]
    factors = scan["factors"]
    residuals = scan["residuals"]
    regularization_norms = scan["regularization_norms"]
    curvature = scan["curvature"]
    diagnostics = result.regularization_diagnostics or {}
    selected_word = "Fallback" if bool(diagnostics.get("fallback", False)) else "Selected"

    width = style.width
    height = max(520, style.height // 2 + 160)
    left_box = (style.margin + 52, 84, width // 2 - style.margin - 86, height - 180)
    right_box = (width // 2 + 48, 84, width // 2 - style.margin - 86, height - 180)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{style.background}"/>',
        f'<text x="{style.margin}" y="30" font-family="Arial, sans-serif" font-size="16" fill="{style.text_color}">Stream-Function Regularization L-Curve Scan</text>',
        f'<text x="{style.margin}" y="52" font-family="Arial, sans-serif" font-size="12" fill="{style.text_color}">{selected_word} factor: {factors[selected_index]:.4g}; selected lambda: {result.regularization_lambda:.4g}</text>',
    ]

    left_x = np.log10(residuals[valid])
    left_y = np.log10(regularization_norms[valid])
    left_points = _map_to_box(left_x, left_y, left_box)
    selected_left = _map_to_box(
        np.asarray([np.log10(residuals[selected_index])]),
        np.asarray([np.log10(regularization_norms[selected_index])]),
        left_box,
        x_range=(float(np.min(left_x)), float(np.max(left_x))),
        y_range=(float(np.min(left_y)), float(np.max(left_y))),
    )[0]
    elements.extend(_svg_axes(left_box, "log10 ||A x - b||2 [T/A]", "log10 ||R x||2 [Ohm A]", style))
    elements.append(_svg_polyline(left_points, "#1f6feb", 2.0))
    elements.append(_svg_marker(selected_left, "#d73a49"))
    elements.append(_svg_panel_title(left_box, "L-curve: residual vs resistance seminorm", style))

    right_x = np.log10(factors[valid])
    x_range = (float(np.min(right_x)), float(np.max(right_x)))
    right_residual = normalized_log_trace(residuals[valid])
    right_resistance = normalized_log_trace(regularization_norms[valid])
    elements.extend(_svg_axes(right_box, "log10 regularization factor [-]", "normalized value [-]", style))
    elements.append(
        _svg_polyline(
            _map_to_box(right_x, right_residual, right_box, x_range=x_range, y_range=(0.0, 1.0)), "#1f6feb", 2.0
        )
    )
    elements.append(
        _svg_polyline(
            _map_to_box(right_x, right_resistance, right_box, x_range=x_range, y_range=(0.0, 1.0)), "#f97316", 2.0
        )
    )
    finite_curvature = np.isfinite(curvature) & (curvature >= 0.0) & valid
    if np.any(finite_curvature):
        curve_x = np.log10(factors[finite_curvature])
        curve_y = normalized_linear_trace(curvature[finite_curvature])
        elements.append(
            _svg_polyline(_map_to_box(curve_x, curve_y, right_box, x_range=x_range, y_range=(0.0, 1.0)), "#16a34a", 2.0)
        )
    selected_right = _map_to_box(
        np.asarray([np.log10(factors[selected_index])]),
        np.asarray([0.5]),
        right_box,
        x_range=x_range,
        y_range=(0.0, 1.0),
    )[0]
    x0, y0, _, h = right_box
    elements.append(
        f'<line x1="{selected_right[0]:.3f}" y1="{y0:.3f}" x2="{selected_right[0]:.3f}" y2="{y0 + h:.3f}" stroke="#d73a49" stroke-width="1.4" stroke-dasharray="5 4"/>'
    )
    elements.append(_svg_marker(selected_right, "#d73a49"))
    elements.append(_svg_panel_title(right_box, "1D scan over regularization factor", style))
    elements.extend(
        [
            _svg_legend(width // 2 + 58, height - 64, "log residual", "#1f6feb", style),
            _svg_legend(width // 2 + 180, height - 64, "log resistance seminorm", "#f97316", style),
            _svg_legend(width // 2 + 370, height - 64, "curvature", "#16a34a", style),
        ]
    )
    elements.append("</svg>")
    _write(path, "\n".join(elements))
    return Path(path)


def _map_to_box(
    x_values: np.ndarray,
    y_values: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x0, y0, width, height = box
    if x_range is None:
        x_range = (float(np.min(x)), float(np.max(x)))
    if y_range is None:
        y_range = (float(np.min(y)), float(np.max(y)))
    x_span = max(float(x_range[1] - x_range[0]), 1e-30)
    y_span = max(float(y_range[1] - y_range[0]), 1e-30)
    px = x0 + (x - x_range[0]) / x_span * width
    py = y0 + height - (y - y_range[0]) / y_span * height
    return np.column_stack((px, py))


def _svg_axes(box: tuple[float, float, float, float], x_label: str, y_label: str, style: SvgStyle) -> list[str]:
    x0, y0, width, height = box
    return [
        f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{width:.3f}" height="{height:.3f}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>',
        f'<line x1="{x0:.3f}" y1="{y0 + height:.3f}" x2="{x0 + width:.3f}" y2="{y0 + height:.3f}" stroke="#334155" stroke-width="1"/>',
        f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x0:.3f}" y2="{y0 + height:.3f}" stroke="#334155" stroke-width="1"/>',
        f'<text x="{x0 + width / 2:.3f}" y="{y0 + height + 42:.3f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{style.text_color}">{escape(x_label)}</text>',
        f'<text x="{x0 - 44:.3f}" y="{y0 + height / 2:.3f}" text-anchor="middle" transform="rotate(-90 {x0 - 44:.3f} {y0 + height / 2:.3f})" font-family="Arial, sans-serif" font-size="11" fill="{style.text_color}">{escape(y_label)}</text>',
    ]


def _svg_panel_title(box: tuple[float, float, float, float], title: str, style: SvgStyle) -> str:
    x0, y0, width, _ = box
    return f'<text x="{x0 + width / 2:.3f}" y="{y0 - 14:.3f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="{style.text_color}">{escape(title)}</text>'


def _svg_polyline(points: np.ndarray, color: str, stroke_width: float) -> str:
    point_text = " ".join(f"{x:.3f},{y:.3f}" for x, y in np.asarray(points, dtype=float))
    return f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{stroke_width:.3f}" stroke-linecap="round" stroke-linejoin="round"/>'


def _svg_marker(point: np.ndarray, color: str) -> str:
    return (
        f'<circle cx="{point[0]:.3f}" cy="{point[1]:.3f}" r="5.5" fill="{color}" stroke="#111827" stroke-width="1.2"/>'
    )


def _svg_legend(x: float, y: float, label: str, color: str, style: SvgStyle) -> str:
    return (
        f'<line x1="{x:.3f}" y1="{y:.3f}" x2="{x + 22:.3f}" y2="{y:.3f}" stroke="{color}" stroke-width="2.2"/>'
        f'<text x="{x + 28:.3f}" y="{y + 4:.3f}" font-family="Arial, sans-serif" font-size="11" fill="{style.text_color}">{escape(label)}</text>'
    )


def axis_name(axis: int) -> str:
    return ("x", "y", "z")[axis]


def _coordinate_axis_label(axis: int) -> str:
    return format_label_with_unit(axis_name(axis), COORDINATE_UNIT)


def _coordinate_axis_labels(axes: tuple[int, int]) -> tuple[str, str]:
    return _coordinate_axis_label(axes[0]), _coordinate_axis_label(axes[1])


def _projection_code(axes: tuple[int, int]) -> str:
    return f"{axis_name(axes[0]).upper()}{axis_name(axes[1]).upper()}"


def _projection_title(axes: tuple[int, int]) -> str:
    return f"{_projection_code(axes)} Projection"


def _svg_quantity_title(quantity: str, view: str, unit: str | None = None) -> str:
    return format_label_with_unit(f"{quantity} - {view}", unit)


def _scalar_map_svg_title(context: str, scalar_map: Any, view: str) -> str:
    return f"{context}: {scalar_map.label_with_unit} - {view}"


def _metric_label_with_unit(label: str) -> str:
    unit = _metric_unit(label)
    return format_label_with_unit(label.replace("_", " "), unit)


def _metric_unit(label: str) -> str:
    key = label.lower()
    if "rel_error" in key or "error" in key:
        return RELATIVE_ERROR_UNIT
    if "current" in key or "potential_step" in key:
        return STREAM_FUNCTION_UNIT
    if "length" in key or "distance" in key:
        return COORDINATE_UNIT
    if "resistance" in key:
        return "Ohm"
    if "inductance" in key:
        return "H"
    if "cross_section" in key:
        return "m^2"
    if "num_levels" in key:
        return "-"
    return ""


def _append_optional(saved: list[Path], factory: Callable[[], Path | list[Path]]) -> None:
    try:
        value = factory()
    except ValueError:
        return
    if isinstance(value, list):
        saved.extend(value)
    else:
        saved.append(value)


def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    saved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = Path(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        saved.append(normalized)
    return saved


def _isometric_projection(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    screen_x = (points[:, 0] - points[:, 1]) / np.sqrt(2.0)
    screen_y = (points[:, 0] + points[:, 1] + 2.0 * points[:, 2]) / np.sqrt(6.0)
    return np.column_stack((screen_x, screen_y))


def _draw_shift_overlay(scene: "_SvgScene", points: np.ndarray, shift_array: np.ndarray, style: SvgStyle) -> None:
    shifted = np.asarray(shift_array, dtype=float) > 0
    if len(points) < 2:
        return
    start = None
    for ind, value in enumerate(shifted):
        if value and start is None:
            start = ind
        if start is not None and (not value or ind == len(shifted) - 1):
            stop = ind if value and ind == len(shifted) - 1 else ind - 1
            if stop > start:
                scene.polyline(points[start : stop + 1], stroke=style.shifted_stroke, stroke_width=3.2, opacity=0.85)
            start = None


def _point_component(points: np.ndarray, axis: int) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1 and len(points) >= 3:
        return float(points[axis])
    if points.ndim == 2 and len(points) > 0 and points.shape[1] >= 3:
        return float(points[0, axis])
    return 0.0


def _center_component(points: np.ndarray, axis: int) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim == 2 and len(points) > 0 and points.shape[1] >= 3:
        return float(np.mean(points[:, axis]))
    if points.ndim == 1 and len(points) >= 3:
        return float(points[axis])
    return 0.0


def _topology_parent_by_group(part: CoilPart) -> dict[int, int]:
    parents = {group_ind: -1 for group_ind in range(len(part.groups))}
    for group_level, positions in zip(part.group_levels, part.level_positions):
        parent = int(positions[-1]) if positions else -1
        for group_ind in group_level:
            parents[int(group_ind)] = parent
    return parents


def _topology_siblings_by_group(part: CoilPart) -> dict[int, list[int]]:
    siblings = {group_ind: [] for group_ind in range(len(part.groups))}
    for group_level in part.group_levels:
        level = [int(group_ind) for group_ind in group_level]
        for group_ind in level:
            siblings[group_ind] = [other for other in level if other != group_ind]
    return siblings


def _topology_level_by_group(part: CoilPart) -> dict[int, int]:
    levels: dict[int, int] = {}
    for level_ind, group_level in enumerate(part.group_levels):
        for group_ind in group_level:
            levels[int(group_ind)] = int(level_ind)
    return levels


def _topology_depth_by_group(parent_by_group: dict[int, int]) -> dict[int, int]:
    depths: dict[int, int] = {}
    for group_ind in parent_by_group:
        depth = 0
        seen: set[int] = set()
        parent = parent_by_group.get(group_ind, -1)
        while parent >= 0 and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = parent_by_group.get(parent, -1)
        depths[group_ind] = depth
    return depths


def _join_ints(values: Iterable[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def _write_diagnostics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0].keys())
        if rows
        else [
            "part_index",
            "cut_index",
            "group1_index",
            "group2_index",
            "loop1_index",
            "loop2_index",
            "outer_contour_only",
            "connection_method",
            "used_direct_fallback",
            "used_clearance_fallback",
            "direct_distance",
            "geodesic_distance",
            "connector_distance",
            "connector_point_count",
            "selected_point1_x",
            "selected_point1_y",
            "selected_point1_z",
            "selected_point2_x",
            "selected_point2_y",
            "selected_point2_z",
            "cut1_center_x",
            "cut1_center_y",
            "cut1_center_z",
            "cut2_center_x",
            "cut2_center_y",
            "cut2_center_z",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_wire(part: CoilPart) -> None:
    if part.wire_path is None:
        raise ValueError("Part has no wire path.")


def _require_contours(part: CoilPart) -> None:
    if not part.contour_lines:
        raise ValueError("Part has no contour lines.")


def _require_grouped_contours(part: CoilPart) -> None:
    if not part.contour_lines:
        raise ValueError("Part has no contour lines.")
    if not part.groups:
        raise ValueError("Part has contours but no topological groups; run topological_loop_grouping first.")


def _require_interconnect_data(part: CoilPart) -> None:
    if not part.connected_groups:
        raise ValueError("Part has no connected groups; run interconnect_within_groups first.")


def _require_swept_body(part: CoilPart) -> None:
    if part.swept_body is None:
        raise ValueError("Part has no swept conductor body; run with skip_sweep=False first.")


def _face_average_stream_function(part: CoilPart) -> np.ndarray:
    if part.stream_function is None:
        raise ValueError("Part has no stream function.")
    stream = np.asarray(part.stream_function, dtype=float)
    return np.mean(stream[part.mesh.faces], axis=1)


def _current_density_magnitude(part: CoilPart) -> np.ndarray:
    current_density = part.current_density
    if current_density is None and part.current_density_mat is not None and part.stream_function is not None:
        current_density = np.tensordot(part.stream_function, part.current_density_mat, axes=(0, 0))
    if current_density is None:
        current_density = current_density_from_basis(part)
    if current_density is None:
        raise ValueError("Part has no current density.")
    current_density = np.asarray(current_density, dtype=float)
    if current_density.shape != (len(part.mesh.faces), 3):
        raise ValueError("Current density must have shape (n_faces, 3).")
    return np.linalg.norm(current_density, axis=1)


def _target_field_map(result: CoilGenResult, source: str):
    source_key = source.lower().replace("-", "_")
    maps = field_scalar_maps(result)
    if source_key not in maps:
        raise ValueError(f"Unknown target-field map source: {source!r}.")
    return maps[source_key]


def _draw_scalar_legend(scene: "_SvgScene", label: str, vmin: float, vmax: float) -> None:
    scene.legend([(f"{label} min {vmin:.3g}", _color_ramp(0.0)), (f"{label} max {vmax:.3g}", _color_ramp(1.0))])


def _finite_min_max(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        raise ValueError("No finite values.")
    return float(np.min(finite)), float(np.max(finite))


def _finite_display_range(values: Iterable[float], *, percentile: float, nonnegative: bool) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        raise ValueError("No finite values.")
    percentile = float(percentile)
    if percentile < 100.0:
        percentile = min(max(percentile, 1.0), 100.0)
        if nonnegative:
            low = 0.0 if float(np.nanmin(finite)) >= 0.0 else float(np.nanpercentile(finite, 100.0 - percentile))
            high = float(np.nanpercentile(finite, percentile))
        else:
            tail = 0.5 * (100.0 - percentile)
            low, high = (float(value) for value in np.nanpercentile(finite, [tail, 100.0 - tail]))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return float(low), float(high)
    return _finite_min_max(finite)


def _finite_range(values: Iterable[float]) -> tuple[float, float]:
    low, high = _finite_min_max(values)
    if np.isclose(low, high, rtol=1e-12, atol=1e-15):
        delta = max(abs(low), 1.0) * 0.05
        return low - delta, high + delta
    return low, high


def _shim_abs_metric_label(source_key: str) -> str:
    key = str(source_key)
    if key in {"layout_improvement_before_minus_after", "stream_improvement_before_minus_after"}:
        return "improvement"
    if key == "before_inhomogeneity":
        return "inhomogeneity"
    if key in {"target", "layout", "stream"}:
        return "field"
    return "residual"


def _shim_reference_color_limit_abs(result: CoilGenResult, *, source: str, domain: str) -> float | None:
    source_key = _shim_source_key(source)
    if source_key not in {
        "layout_remaining_inhomogeneity",
        "stream_remaining_inhomogeneity",
        "layout_improvement_before_minus_after",
        "stream_improvement_before_minus_after",
    }:
        return None
    coords = np.asarray(result.target_field.coords, dtype=float)
    values = np.asarray(shim_field_scalar_maps(result)["before_inhomogeneity"].values, dtype=float).reshape(-1)
    finite = target_domain_mask(result, domain) & np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    if not np.any(finite):
        raise ValueError("Shim homogeneity mosaic has no finite before-shimming samples for color scaling.")
    return max(float(np.nanmax(np.abs(values[finite]))), 1e-30)


def _shim_source_key(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_")
    if key in {
        "before",
        "pre",
        "pre_shim",
        "pre_shimming",
        "unshimmed",
        "fieldmap",
        "inhomogeneity",
        "before_inhomogeneity",
    }:
        return "before_inhomogeneity"
    if key in {
        "after",
        "post",
        "post_shim",
        "post_shimming",
        "remaining",
        "remaining_inhomogeneity",
        "layout",
        "connected",
        "connected_wire",
        "wire",
        "wire_layout",
        "coil",
        "coilfield",
        "layout_remaining",
        "layout_remaining_inhomogeneity",
    }:
        return "layout_remaining_inhomogeneity"
    if key in {"after_stream", "stream_remaining", "stream_remaining_inhomogeneity"}:
        return "stream_remaining_inhomogeneity"
    if key in {"layout_residual", "target_minus_layout"}:
        return "layout_residual"
    if key in {"stream", "stream_function", "sf"}:
        return "stream_remaining_inhomogeneity"
    if key in {"stream_residual", "target_minus_stream"}:
        return "stream_residual"
    if key in {
        "improvement",
        "delta",
        "delta_improvement",
        "before_minus_after",
        "layout_improvement",
        "wire_improvement",
        "connected_improvement",
        "layout_improvement_before_minus_after",
    }:
        return "layout_improvement_before_minus_after"
    if key in {"stream_improvement", "before_minus_after_stream", "stream_improvement_before_minus_after"}:
        return "stream_improvement_before_minus_after"
    raise ValueError("source must be 'before', 'after', 'layout'/'coil', 'stream', or 'improvement'.")


def _map_points_to_svg_box(
    points: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    pad: float = 0.0,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    x0, y0, width, height = box
    xmin, xmax = x_range
    ymin, ymax = y_range
    x_span = max(float(xmax - xmin), 1e-30)
    y_span = max(float(ymax - ymin), 1e-30)
    draw_w = max(width - 2.0 * pad, 1.0)
    draw_h = max(height - 2.0 * pad, 1.0)
    x = x0 + pad + (points[:, 0] - xmin) / x_span * draw_w
    y = y0 + height - pad - (points[:, 1] - ymin) / y_span * draw_h
    return np.column_stack((x, y))


def _svg_diverging_colorbar(
    style: SvgStyle,
    *,
    y: float,
    vmin: float,
    vmax: float,
    label: str,
) -> list[str]:
    x0 = style.margin + 120
    width = style.width - 2 * style.margin - 240
    height = 14.0
    steps = 72
    elements = [
        f'<text x="{style.margin}" y="{y + 11:.3f}" font-family="Arial, sans-serif" font-size="12" fill="{style.text_color}">{escape(label)}</text>'
    ]
    for index in range(steps):
        x = x0 + width * index / steps
        segment_w = width / steps + 0.8
        color = _color_ramp(index / max(steps - 1, 1))
        elements.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{segment_w:.3f}" height="{height:.3f}" fill="{color}"/>')
    elements.append(
        f'<rect x="{x0:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" fill="none" stroke="#334155" stroke-width="0.8"/>'
    )
    for value, anchor, dx in ((vmin, "start", 0.0), (0.0, "middle", width / 2), (vmax, "end", width)):
        elements.append(
            f'<text x="{x0 + dx:.3f}" y="{y + height + 17:.3f}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="11" fill="{style.text_color}">{value:.4g}</text>'
        )
    return elements


def _short_metric_label(label: str) -> str:
    text = label.replace("_", " ")
    if len(text) <= 42:
        return text
    return f"{text[:39]}..."


def _grouped_contour_entries(part: CoilPart) -> list[tuple[int | None, int, ContourLine]]:
    entries: list[tuple[int | None, int, ContourLine]] = []
    used_indices: set[int] = set()
    for group_ind, group in enumerate(part.groups):
        for local_ind, loop in enumerate(group.loops):
            loop_ind = (
                group.loop_indices[local_ind]
                if local_ind < len(group.loop_indices)
                else _loop_identity_index(part, loop)
            )
            if loop_ind >= 0:
                used_indices.add(loop_ind)
            entries.append((group_ind, loop_ind, loop))
    for loop_ind, loop in enumerate(part.contour_lines):
        if loop_ind not in used_indices:
            entries.append((None, loop_ind, loop))
    return entries


def _loop_identity_index(part: CoilPart, loop: ContourLine) -> int:
    for loop_ind, candidate in enumerate(part.contour_lines):
        if candidate is loop:
            return loop_ind
    return -1


def _closed_for_display(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    if np.linalg.norm(points[0] - points[-1]) <= 1e-10:
        return points
    return np.vstack((points, points[0]))


def _group_label(part: CoilPart, group_ind: int | None) -> str:
    if group_ind is None:
        return "ungrouped"
    loops = part.groups[group_ind].loop_indices if group_ind < len(part.groups) else []
    if len(loops) == 1:
        return f"G{group_ind}: L{loops[0]}"
    return f"G{group_ind}: {len(loops)} loops"


def _group_color(group_ind: int | None) -> str:
    if group_ind is None:
        return "#6b7280"
    palette = (
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#0f766e",
        "#be123c",
    )
    return palette[group_ind % len(palette)]


def _interconnect_debug_point_arrays(part: CoilPart, project: Callable[[np.ndarray], np.ndarray]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for group in part.groups:
        for cuts_v in group.opening_cuts_v:
            arrays.append(project(cuts_v))
    for connected in part.connected_groups:
        arrays.append(project(connected.return_path_v))
    for cut in part.opening_cuts_among_groups:
        arrays.append(project(cut.cut1_v))
        arrays.append(project(cut.cut2_v))
        arrays.append(project(cut.connection_path_v))
        arrays.append(project(cut.closing_connection_path_v))
    return arrays


def _stack_nonempty(arrays: Iterable[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(array, dtype=float) for array in arrays if array is not None and len(array)]
    if not valid:
        return np.zeros((2, 2), dtype=float)
    return np.vstack(valid)


def _normalize(value: float, vmin: float, vmax: float) -> float:
    span = vmax - vmin
    if abs(span) < 1e-30:
        return 0.5
    return float(np.clip((value - vmin) / span, 0.0, 1.0))


def _color_ramp(t: float) -> str:
    # Blue -> pale -> red, intentionally simple and printable.
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        q = t / 0.5
        r = int(45 + q * (235 - 45))
        g = int(105 + q * (238 - 105))
        b = int(196 + q * (245 - 196))
    else:
        q = (t - 0.5) / 0.5
        r = int(235 + q * (203 - 235))
        g = int(238 + q * (64 - 238))
        b = int(245 + q * (72 - 245))
    return f"#{r:02x}{g:02x}{b:02x}"


def _write(path: str | Path, text: str) -> None:
    if text.startswith("<svg "):
        # Scope credits to the generating software, preserving design authorship.
        end = text.index(">") + 1
        metadata = escape(json.dumps(software_attribution(), ensure_ascii=False))
        text = f'{text[:end]}\n<metadata id="coilgen-software">{metadata}</metadata>{text[end:]}'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


class _SvgScene:
    def __init__(self, bounds: tuple[float, float, float, float], style: SvgStyle, title: str):
        self.bounds = bounds
        self.style = style
        self.elements: list[str] = []
        self.title = title

    @classmethod
    def from_points(cls, points: np.ndarray, style: SvgStyle, title: str) -> "_SvgScene":
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            bounds = (-1.0, 1.0, -1.0, 1.0)
        else:
            xmin = float(np.min(points[:, 0]))
            xmax = float(np.max(points[:, 0]))
            ymin = float(np.min(points[:, 1]))
            ymax = float(np.max(points[:, 1]))
            if abs(xmax - xmin) < 1e-12:
                xmin -= 1.0
                xmax += 1.0
            if abs(ymax - ymin) < 1e-12:
                ymin -= 1.0
                ymax += 1.0
            pad_x = (xmax - xmin) * 0.04
            pad_y = (ymax - ymin) * 0.04
            bounds = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)
        return cls(bounds, style, title)

    def finish(self) -> str:
        style = self.style
        header = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{style.width}" height="{style.height}" viewBox="0 0 {style.width} {style.height}">',
            f'<rect width="100%" height="100%" fill="{style.background}"/>',
            f'<text x="{style.margin}" y="{style.margin - 12}" font-family="Arial, sans-serif" font-size="16" fill="{style.text_color}">{escape(self.title)}</text>',
        ]
        footer = ["</svg>"]
        return "\n".join(header + self.elements + footer)

    def transform(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        xmin, xmax, ymin, ymax = self.bounds
        style = self.style
        draw_w = style.width - 2 * style.margin
        draw_h = style.height - 2 * style.margin
        scale = min(draw_w / (xmax - xmin), draw_h / (ymax - ymin))
        used_w = (xmax - xmin) * scale
        used_h = (ymax - ymin) * scale
        offset_x = style.margin + (draw_w - used_w) / 2
        offset_y = style.margin + (draw_h - used_h) / 2
        x = offset_x + (points[:, 0] - xmin) * scale
        y = style.height - (offset_y + (points[:, 1] - ymin) * scale)
        return np.column_stack((x, y))

    def polyline(
        self,
        points: np.ndarray,
        *,
        stroke: str,
        stroke_width: float,
        opacity: float = 1.0,
        stroke_dasharray: str | None = None,
    ) -> None:
        if points is None or len(points) < 2:
            return
        mapped = self.transform(points)
        path = " ".join(f"{x:.3f},{y:.3f}" for x, y in mapped)
        dash = "" if stroke_dasharray is None else f' stroke-dasharray="{escape(stroke_dasharray)}"'
        self.elements.append(
            f'<polyline points="{path}" fill="none" stroke="{stroke}" stroke-width="{stroke_width:g}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity:g}"{dash}/>'
        )

    def polygon(
        self,
        points: np.ndarray,
        *,
        fill: str,
        stroke: str,
        stroke_width: float,
        opacity: float = 1.0,
    ) -> None:
        if points is None or len(points) < 3:
            return
        mapped = self.transform(points)
        path = " ".join(f"{x:.3f},{y:.3f}" for x, y in mapped)
        self.elements.append(
            f'<polygon points="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:g}" opacity="{opacity:g}"/>'
        )

    def text_at(self, point: np.ndarray, text: str, *, fill: str, font_size: int) -> None:
        point = np.asarray(point, dtype=float).reshape(1, 2)
        x, y = self.transform(point)[0]
        self.elements.append(
            f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial, sans-serif" font-size="{font_size}" font-weight="700" '
            f'text-anchor="middle" fill="{fill}" stroke="{self.style.background}" stroke-width="3" paint-order="stroke">{escape(text)}</text>'
        )

    def circle_at(
        self,
        point: np.ndarray,
        *,
        radius: float,
        fill: str,
        stroke: str,
        stroke_width: float,
        opacity: float = 1.0,
    ) -> None:
        point = np.asarray(point, dtype=float).reshape(1, 2)
        x, y = self.transform(point)[0]
        self.elements.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:g}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:g}" opacity="{opacity:g}"/>'
        )

    def mesh(
        self,
        points: np.ndarray,
        faces: np.ndarray,
        *,
        stroke: str,
        stroke_width: float,
    ) -> None:
        if points is None or faces is None or len(points) == 0 or len(faces) == 0:
            return
        for face in faces:
            polygon = np.vstack((points[face], points[face[0]]))
            self.polyline(polygon, stroke=stroke, stroke_width=stroke_width, opacity=0.85)

    def axes(self) -> None:
        xmin, xmax, ymin, ymax = self.bounds
        zero_x = np.array([[xmin, 0.0], [xmax, 0.0]])
        zero_y = np.array([[0.0, ymin], [0.0, ymax]])
        if ymin <= 0 <= ymax:
            self.polyline(zero_x, stroke="#e5e7eb", stroke_width=1.0)
        if xmin <= 0 <= xmax:
            self.polyline(zero_y, stroke="#e5e7eb", stroke_width=1.0)

    def axis_labels(self, x_label: str, y_label: str) -> None:
        x = self.style.width / 2
        y = self.style.height - 8
        self.elements.append(
            f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial, sans-serif" font-size="12" '
            f'text-anchor="middle" fill="{self.style.text_color}">{escape(x_label)}</text>'
        )
        self.elements.append(
            f'<text x="14" y="{self.style.height / 2:.3f}" font-family="Arial, sans-serif" font-size="12" '
            f'text-anchor="middle" fill="{self.style.text_color}" transform="rotate(-90 14 {self.style.height / 2:.3f})">{escape(y_label)}</text>'
        )

    def legend(self, items: Sequence[tuple[str, str]]) -> None:
        if not items:
            return
        x = self.style.margin
        item_width = 120
        columns = max(1, (self.style.width - 2 * self.style.margin) // item_width)
        rows = (len(items) + columns - 1) // columns
        first_y = self.style.height - self.style.margin - (rows - 1) * 18 + 16
        for ind, (label, color) in enumerate(items):
            row = ind // columns
            col = ind % columns
            dx = col * item_width
            y = first_y + row * 18
            self.elements.append(
                f'<line x1="{x + dx}" y1="{y}" x2="{x + dx + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>'
            )
            self.elements.append(
                f'<text x="{x + dx + 32}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12" fill="{self.style.text_color}">{escape(label)}</text>'
            )
