from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from coilgen import (
    CoilDesignRunArtifacts,
    CoilProject,
    CoilProjectRun,
    CylinderCoilDesign,
    result_quality_status,
    result_summary_dict,
)
from projects._shared.example_utils import print_result_summary, print_saved_files
from projects._shared.paths import project_output_root


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
OUTPUT_ROOT = project_output_root("standard_cylindrical_3axis_gradient")
AXES = ("x", "y", "z")
STANDARD_ACTIVE_PLOT_KEYS = (
    "layout_field_overview",
    "wire_sequence",
    "contour_current_orientation",
    "performance_table",
)


@dataclass(frozen=True, slots=True)
class AxisRun:
    axis: str
    design: CylinderCoilDesign
    artifacts: CoilDesignRunArtifacts
    files: tuple[Path, ...] = field(default_factory=tuple)
    figures: tuple[Any, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class GradientSystemRun:
    output_dir: Path
    axis_runs: tuple[AxisRun, ...]
    summary_rows: tuple[dict[str, Any], ...]
    files: tuple[Path, ...] = field(default_factory=tuple)
    figures: tuple[Any, ...] = field(default_factory=tuple)
    project_run: CoilProjectRun | None = None

    @property
    def results(self) -> dict[str, Any]:
        return {axis_run.axis: axis_run.artifacts.result for axis_run in self.axis_runs}


def build_axis_design(axis: str, parameters: Mapping[str, Any]) -> CylinderCoilDesign:
    axis = _normalize_axis(axis)
    geometry = _section(parameters, "geometry")
    target = _section(parameters, "target")
    optimization = _section(parameters, "optimization")
    layout = _section(parameters, "layout")
    conductor = _section(parameters, "conductor")
    numerics = _section(parameters, "numerics")
    system = _section(parameters, "system")
    run = _section(parameters, "run")
    output = _section(parameters, "output")

    name = f"{_safe_stem(output.get('name', 'standard_cylindrical_3axis_gradient'))}_g{axis}"
    return CylinderCoilDesign(
        name=name,
        target_bz_expression=axis,
        target_scale=float(target["gradient_strength_T_per_m"]),
        target_region_radius_m=float(target["dsv_radius_m"]),
        target_region_resolution=int(target["dsv_resolution"]),
        contour_discretization_method=str(layout.get("contour_discretization_method", "level_count")),
        contour_levels=int(layout.get("contour_levels", 8)),
        min_turn_spacing_m=_layout_min_wire_distance_m(layout),
        max_current_A=float(layout.get("max_current_A", layout.get("max_wire_current_A", 0.0))),
        min_wire_distance_m=_layout_min_wire_distance_m(layout),
        surface_current_density_limit_A_per_m=float(layout.get("surface_current_density_limit_A_per_m", 0.0)),
        engineering_contour_step_safety_factor=float(layout.get("engineering_contour_step_safety_factor", 1.0)),
        engineering_contour_max_levels=int(layout.get("engineering_contour_max_levels", 0)),
        potential_offset_factor=float(layout.get("potential_offset_factor", 0.25)),
        min_contour_points=int(layout.get("min_contour_points", 4)),
        conductor_width_m=float(conductor["width_m"]),
        conductor_height_m=float(conductor["height_m"]),
        conductor_resistivity_ohm_m=float(conductor["resistivity_ohm_m"]),
        main_field_strength_T=float(system["main_field_strength_T"]),
        b0_direction=tuple(float(value) for value in system["b0_direction"]),
        optimization_method=str(optimization["method"]),
        regularization_method=str(optimization["regularization_method"]),
        regularization_factor=float(optimization["regularization_factor"]),
        l_curve_candidate_count=int(optimization["l_curve_candidate_count"]),
        l_curve_smoothing_window=int(optimization["l_curve_smoothing_window"]),
        target_error_tolerance_percent=float(optimization["target_error_tolerance_percent"]),
        target_error_tolerance_absolute_T=float(optimization["target_error_tolerance_absolute_T"]),
        target_error_constraint_candidate_count=int(optimization["target_error_constraint_candidate_count"]),
        radius_m=float(geometry["radius_m"]),
        length_m=float(geometry["length_m"]),
        angular_divisions=int(geometry["angular_divisions"]),
        length_divisions=int(geometry["length_divisions"]),
        rotation_axis=tuple(float(value) for value in geometry["rotation_axis"]),
        rotation_angle_rad=float(geometry["rotation_angle_rad"]),
        cpu_workers=int(numerics["cpu_workers"]),
        progress=bool(run["progress"]),
        skip_postprocessing=bool(layout["skip_postprocessing"]),
        skip_sweep=bool(layout["skip_sweep"]),
        skip_inductance=bool(layout["skip_inductance"]),
        inductance_source=str(layout["inductance_source"]),
        advanced_parameters={
            "parallel_max_chunk_memory_mb": float(numerics["max_chunk_memory_mb"]),
            "parallel_min_work_size": int(numerics["parallel_min_work_size"]),
            "sensitivity_chunks_per_worker": int(numerics["sensitivity_chunks_per_worker"]),
            "progress_min_interval_seconds": float(numerics["progress_min_interval_seconds"]),
            "progress_percent_step": float(numerics["progress_percent_step"]),
            "contour_symmetry_validation": str(layout["contour_symmetry_validation"]),
            "zero_crossing_level_strategy": str(layout["zero_crossing_level_strategy"]),
            "field_evaluation_current_source": str(target["field_evaluation_current_source"]),
        },
    )


def build_gradient_project(parameters: Mapping[str, Any]) -> CoilProject:
    parameters = dict(parameters)
    geometry = _section(parameters, "geometry")
    target = _section(parameters, "target")
    optimization = _section(parameters, "optimization")
    layout = _section(parameters, "layout")
    conductor = _section(parameters, "conductor")
    numerics = _section(parameters, "numerics")
    system = _section(parameters, "system")
    run = _section(parameters, "run")
    output = _section(parameters, "output")
    plots = _section(parameters, "plots")

    axes = _normalize_axes(run["axes"])
    project_name = _safe_stem(output["name"])
    template = CylinderCoilDesign(
        name=f"{project_name}_template",
        advanced_parameters={
            "parallel_max_chunk_memory_mb": float(numerics["max_chunk_memory_mb"]),
            "parallel_min_work_size": int(numerics["parallel_min_work_size"]),
            "sensitivity_chunks_per_worker": int(numerics["sensitivity_chunks_per_worker"]),
            "progress_min_interval_seconds": float(numerics["progress_min_interval_seconds"]),
            "progress_percent_step": float(numerics["progress_percent_step"]),
            "contour_symmetry_validation": str(layout["contour_symmetry_validation"]),
            "zero_crossing_level_strategy": str(layout["zero_crossing_level_strategy"]),
            "field_evaluation_current_source": str(target["field_evaluation_current_source"]),
        },
    )
    project = CoilProject(
        name=project_name,
        template=template,
        project_root=REPOSITORY_ROOT,
        output_dir=OUTPUT_ROOT / project_name,
        save_reports=bool(output["save_reports"]),
        save_gallery=bool(output["save_svg_gallery"]),
        open_plot_book=False,
        close_previous_plots=bool(plots["close_previous"]),
        plot_include=_with_standard_active_plot_keys(plots.get("include")),
        max_plot_parts=_optional_int(plots.get("max_parts")),
    )
    project.parameters.mesh.radius_m = float(geometry["radius_m"])
    project.parameters.mesh.length_m = float(geometry["length_m"])
    project.parameters.mesh.angular_divisions = int(geometry["angular_divisions"])
    project.parameters.mesh.length_divisions = int(geometry["length_divisions"])
    project.parameters.mesh.rotation_axis = tuple(float(value) for value in geometry["rotation_axis"])
    project.parameters.mesh.rotation_angle_rad = float(geometry["rotation_angle_rad"])
    project.parameters.target.scale = float(target["gradient_strength_T_per_m"])
    project.parameters.target.region_radius_m = float(target["dsv_radius_m"])
    project.parameters.target.region_resolution = int(target["dsv_resolution"])
    project.parameters.optimization.method = str(optimization["method"])
    project.parameters.optimization.regularization_method = str(optimization["regularization_method"])
    project.parameters.optimization.regularization_factor = float(optimization["regularization_factor"])
    project.parameters.optimization.l_curve_candidate_count = int(optimization["l_curve_candidate_count"])
    project.parameters.optimization.l_curve_smoothing_window = int(optimization["l_curve_smoothing_window"])
    project.parameters.optimization.target_error_tolerance_percent = float(
        optimization["target_error_tolerance_percent"]
    )
    project.parameters.optimization.target_error_tolerance_absolute_T = float(
        optimization["target_error_tolerance_absolute_T"]
    )
    project.parameters.optimization.target_error_constraint_candidate_count = int(
        optimization["target_error_constraint_candidate_count"]
    )
    project.parameters.optimization.contour_discretization_method = str(
        layout.get("contour_discretization_method", "level_count")
    )
    project.parameters.optimization.contour_levels = int(layout.get("contour_levels", 8))
    project.parameters.optimization.engineering_contour_step_safety_factor = float(
        layout.get("engineering_contour_step_safety_factor", 1.0)
    )
    project.parameters.optimization.engineering_contour_max_levels = int(
        layout.get("engineering_contour_max_levels", 0)
    )
    project.parameters.optimization.min_turn_spacing_m = _layout_min_wire_distance_m(layout)
    project.parameters.optimization.potential_offset_factor = float(layout.get("potential_offset_factor", 0.25))
    project.parameters.optimization.min_contour_points = int(layout.get("min_contour_points", 4))
    project.parameters.optimization.skip_postprocessing = bool(layout["skip_postprocessing"])
    project.parameters.optimization.skip_sweep = bool(layout["skip_sweep"])
    project.parameters.optimization.skip_inductance = bool(layout["skip_inductance"])
    project.parameters.optimization.inductance_source = str(layout["inductance_source"])
    project.parameters.conductor.width_m = float(conductor["width_m"])
    project.parameters.conductor.height_m = float(conductor["height_m"])
    project.parameters.conductor.resistivity_ohm_m = float(conductor["resistivity_ohm_m"])
    project.parameters.conductor.max_current_A = float(
        layout.get("max_current_A", layout.get("max_wire_current_A", 0.0))
    )
    project.parameters.conductor.min_wire_distance_m = _layout_min_wire_distance_m(layout)
    project.parameters.conductor.surface_current_density_limit_A_per_m = float(
        layout.get("surface_current_density_limit_A_per_m", 0.0)
    )
    project.parameters.system.main_field_strength_T = float(system["main_field_strength_T"])
    project.parameters.system.b0_direction = tuple(float(value) for value in system["b0_direction"])
    project.parameters.numerics.cpu_workers = int(numerics["cpu_workers"])
    project.parameters.numerics.progress = bool(run["progress"])
    for axis in axes:
        axis_label = f"G{axis.upper()}"
        project.channel(
            axis_label,
            label=axis_label,
            target_bz_expression=axis,
            color=_axis_color(axis),
            design_overrides={"name": f"{project_name}_g{axis}"},
            metadata={"axis": axis},
        )
    return project


def run_gradient_system(
    parameters: Mapping[str, Any],
    *,
    project: CoilProject | None = None,
    project_run: CoilProjectRun | None = None,
) -> GradientSystemRun:
    parameters = dict(parameters)
    output = _section(parameters, "output")
    plots = _section(parameters, "plots")

    project = build_gradient_project(parameters) if project is None else project
    output_dir = project.output_dir if project.output_dir is not None else OUTPUT_ROOT / _safe_stem(output["name"])
    output_dir = Path(output_dir)
    project_run = project.solve(open_plot_book=False) if project_run is None else project_run

    axis_runs: list[AxisRun] = []
    summary_rows: list[dict[str, Any]] = []
    saved_files: list[Path] = list(project_run.files)
    plot_book_scope = _plot_book_scope(plots)
    for channel_run in project_run.channel_runs:
        axis = _axis_from_channel_run(channel_run)
        if channel_run.artifacts is None:
            continue
        print(f"\n=== Completed G{axis.upper()} cylindrical gradient ===")
        axis_files = list(channel_run.files)

        figures = []
        if (
            bool(plots["show_plot_book"])
            and plot_book_scope == "axis"
            and axis in _normalize_axes(plots["plot_book_axes"])
        ):
            figures = project.plot(
                project_run,
                channels=channel_run.channel.name,
                close_previous=bool(plots["close_previous"]),
                include=_with_standard_active_plot_keys(plots.get("include")),
                max_parts=_optional_int(plots.get("max_parts")),
                mode=str(plots["mode"]),
            )

        axis_run = AxisRun(
            axis=axis,
            design=channel_run.design,
            artifacts=channel_run.artifacts,
            files=tuple(axis_files),
            figures=tuple(figures),
        )
        axis_runs.append(axis_run)
        summary_rows.append(axis_summary_row(axis_run, parameters))
        print_result_summary(channel_run.artifacts.result, title=f"G{axis.upper()} cylindrical gradient")
        if axis_files:
            print_saved_files(axis_files, title=f"G{axis.upper()} saved files:")

    summary_files = write_system_summary(summary_rows, output_dir, prefix=_safe_stem(output["name"]))
    saved_files.extend(summary_files)
    print_saved_files(summary_files, title="3-axis system summary files:")
    system_figures = []
    if bool(plots["show_plot_book"]) and plot_book_scope == "system":
        selected_project_run = _filter_project_run_for_axes(
            project_run, _normalize_axes(plots.get("plot_book_axes", AXES))
        )
        system_figures = project.plot(
            selected_project_run,
            include=_with_standard_active_plot_keys(plots.get("include")),
            max_parts=_optional_int(plots.get("max_parts")),
            close_previous=bool(plots["close_previous"]),
            show=True,
            block=False,
        )
    if system_figures:
        print("Spyder variables available after run: system_run, results; active plot book: system_run.figures[0]")
    else:
        print("Spyder variables available after run: system_run, results")
    return GradientSystemRun(
        output_dir=output_dir,
        axis_runs=tuple(axis_runs),
        summary_rows=tuple(summary_rows),
        files=tuple(saved_files),
        figures=tuple(system_figures),
        project_run=project_run,
    )


def axis_summary_row(axis_run: AxisRun, parameters: Mapping[str, Any]) -> dict[str, Any]:
    result = axis_run.artifacts.result
    summary = result_summary_dict(result)
    quality = result_quality_status(result)
    target = _section(parameters, "target")
    geometry = _section(parameters, "geometry")
    return {
        "axis": f"G{axis_run.axis.upper()}",
        "target_bz_expression": axis_run.axis,
        "quality_status": quality.status,
        "target_gradient_T_per_m": float(target["gradient_strength_T_per_m"]),
        "radius_m": float(geometry["radius_m"]),
        "length_m": float(geometry["length_m"]),
        "dsv_radius_m": float(target["dsv_radius_m"]),
        "target_points": summary.get("dsv_target_point_count", ""),
        "contours": summary.get("contour_count", ""),
        "needed_current_A": summary.get("needed_current_A", ""),
        "layout_needed_current_A": summary.get("layout_needed_current_A", ""),
        "gradient_efficiency_mT_per_m_per_A": summary.get("layout_gradient_efficiency_mT_per_m_per_A", ""),
        "layout_mean_rel_error_percent": summary.get("layout_mean_rel_error_percent", ""),
        "layout_max_rel_error_percent": summary.get("layout_max_rel_error_percent", ""),
        "coil_length_total_m": summary.get("coil_length_total_m", ""),
        "coil_resistance_total_ohm": summary.get("coil_resistance_total_ohm", ""),
        "coil_inductance_same_current_H": summary.get("coil_inductance_same_current_H", ""),
        "min_contour_distance_m": summary.get("min_contour_distance_m", ""),
        "min_wire_wire_distance_m": summary.get("min_wire_wire_distance_m", ""),
        "route_statuses": summary.get("route_statuses", ""),
    }


def _axis_from_channel_run(channel_run: Any) -> str:
    axis = channel_run.channel.metadata.get("axis", "")
    if axis:
        return _normalize_axis(axis)
    name = str(channel_run.channel.name).strip()
    if len(name) >= 2 and name[0].lower() == "g":
        return _normalize_axis(name[1])
    return _normalize_axis(name)


def _filter_project_run_for_axes(project_run: CoilProjectRun, axes: Sequence[str]) -> CoilProjectRun:
    selected = {f"G{axis.upper()}" for axis in _normalize_axes(axes)}
    channel_runs = tuple(
        channel_run for channel_run in project_run.channel_runs if channel_run.channel.name in selected
    )
    summary_rows = tuple(row for row in project_run.summary_rows if row.get("channel") in selected)
    return CoilProjectRun(
        project=project_run.project,
        output_dir=project_run.output_dir,
        channel_runs=channel_runs,
        summary_rows=summary_rows,
        files=project_run.files,
        figures=project_run.figures,
    )


def _axis_color(axis: str) -> str:
    return {
        "x": "tab:blue",
        "y": "tab:green",
        "z": "tab:red",
    }.get(_normalize_axis(axis), "tab:blue")


def write_system_summary(rows: Sequence[Mapping[str, Any]], output_dir: str | Path, *, prefix: str) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{prefix}_3axis_summary.csv"
    json_path = output / f"{prefix}_3axis_summary.json"
    md_path = output / f"{prefix}_3axis_summary.md"
    _write_rows_csv(rows, csv_path)
    json_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_summary_markdown(rows), encoding="utf-8")
    return [csv_path, json_path, md_path]


def _summary_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "axis",
        "quality_status",
        "target_gradient_T_per_m",
        "layout_needed_current_A",
        "gradient_efficiency_mT_per_m_per_A",
        "layout_mean_rel_error_percent",
        "layout_max_rel_error_percent",
        "coil_inductance_same_current_H",
    )
    lines = [
        "# Standard Cylindrical 3-Axis Gradient Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row.get(column, "")) for column in columns) + " |")
    lines.extend(
        (
            "",
            "Each axis is an independent unshielded cylindrical CoilGen design on the same support geometry.",
            "",
        )
    )
    return "\n".join(lines)


def _write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _normalize_axes(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "all":
            return AXES
        values = tuple(part.strip() for part in text.replace(",", " ").split() if part.strip())
    else:
        values = tuple(value)
    axes = tuple(_normalize_axis(axis) for axis in values)
    if not axes:
        raise ValueError("At least one gradient axis must be selected.")
    return axes


def _normalize_axis(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("g") and len(text) == 2:
        text = text[1]
    if text not in AXES:
        raise ValueError("Gradient axis must be 'x', 'y', 'z', 'Gx', 'Gy', 'Gz', or 'all'.")
    return text


def _plot_book_scope(plots: Mapping[str, Any]) -> str:
    text = str(plots.get("plot_book_scope", "system")).strip().lower()
    if text in {"system", "combined", "three_axis", "3axis", "all"}:
        return "system"
    if text in {"axis", "per_axis", "separate"}:
        return "axis"
    raise ValueError("plots.plot_book_scope must be 'system' or 'axis'.")


def _with_standard_active_plot_keys(include: Any) -> Sequence[str] | None:
    if include is None:
        return None
    keys = [str(key) for key in include]
    for key in STANDARD_ACTIVE_PLOT_KEYS:
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def _layout_min_wire_distance_m(layout: Mapping[str, Any]) -> float:
    return float(layout.get("min_wire_distance_m", layout.get("min_turn_spacing_m", 0.0)))


def _section(parameters: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = parameters.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"PARAMETERS.{name} must be a mapping/object section.")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "all"}:
        return None
    return int(value)


def _safe_stem(value: Any) -> str:
    text = "".join(char if char.isalnum() else "_" for char in str(value).strip())
    text = "_".join(part for part in text.split("_") if part)
    return text or "standard_cylindrical_3axis_gradient"


def _markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|")
