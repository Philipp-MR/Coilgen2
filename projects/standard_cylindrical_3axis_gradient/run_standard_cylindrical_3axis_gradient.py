from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parents[1]

for import_path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

from coilgen import (
    CoilProject,
    CylinderCoilDesign,
    GradientLinearitySpec,
    GradientSliceMapSpec,
    manufacturing_diagnostics_rows,
    save_manufacturing_diagnostics,
    save_swept_bodies_as_stl,
)
from projects._shared.example_utils import (
    interactive_plots_enabled,
    native_thread_budget,
    open_project_plot_book_after_run,
)
from projects._shared.paths import project_output_root


OUTPUT_ROOT = project_output_root("standard_cylindrical_3axis_gradient")


################################################################################
# USER SETTINGS
################################################################################
# Open this file in Spyder, edit these objects, then press Run.
# All lengths are metres. A gradient strength of 0.010 means 10 mT/m.

PROJECT_NAME = "standard_cylindrical_3axis_gradient"
SHOW_PLOT_BOOK = True  # Open after solving; --headless or MPLBACKEND=Agg suppresses it.
NATIVE_THREADS = 1  # BLAS threads inside each CoilGen worker; increase deliberately for large serial solves.
EXPORT_STL = True
EXPORT_WIRE_TRACE_CSV = True
ENABLE_SWEEP = True

BASE_SURFACE_DIAMETER_M = 0.130
SURFACE_DIAMETER_STEP_M = 0.002
SURFACE_DIAMETERS_M = {
    "x": BASE_SURFACE_DIAMETER_M,
    "y": BASE_SURFACE_DIAMETER_M + SURFACE_DIAMETER_STEP_M,
    "z": BASE_SURFACE_DIAMETER_M + 2.0 * SURFACE_DIAMETER_STEP_M,
}
COIL_LENGTH_M = 0.250
DSV_DIAMETER_M = 0.050
DSV_RADIUS_M = DSV_DIAMETER_M / 2.0
WIRE_DIAMETER_M = 0.0014


def _round_wire_cross_section_points(diameter_m: float, point_count: int = 64) -> tuple[tuple[float, float], ...]:
    radius = float(diameter_m) / 2.0
    return tuple(
        (
            radius * math.cos(2.0 * math.pi * point_index / point_count),
            radius * math.sin(2.0 * math.pi * point_index / point_count),
        )
        for point_index in range(point_count)
    )


ROUND_WIRE_CROSS_SECTION_POINTS = _round_wire_cross_section_points(WIRE_DIAMETER_M)

TARGET_GRADIENT_STRENGTH_T_PER_M = 0.010
STREAM_FUNCTION_OPTIMIZATION_METHOD = "constrained_quadratic"
GRADIENT_STRENGTH_TOLERANCE_PERCENT = 1.0
GRADIENT_LINEARITY = GradientLinearitySpec(
    region="dsv",
    reference_radius_m=DSV_RADIUS_M,
    rms_tolerance_percent=1.0,
    peak_tolerance_percent=5.0,
    cross_gradient_tolerance_percent=1.0,
    normalization="requested_gradient_times_radius",
    fit_offset=True,
)
GRADIENT_SLICE_MAPS = GradientSliceMapSpec(
    enabled=True,
    axis="z",
    slice_count=7,
    grid_size=64,
    domain="dsv",
    source="layout",
    current="target",
    field_components=("Bx", "By", "Bz"),
    gradient_components=("dBz_dx", "dBz_dy", "dBz_dz", "gradBz_mag", "linearity_residual"),
    concomitant_components=("concomitant_Bz", "concomitant_grad_mag"),
    overlay=("wire_path", "contours"),
    overlay_mode="slice_slab",
)
MAX_WIRE_CURRENT_A = 500.0
MIN_WIRE_DISTANCE_M = 0.005
SURFACE_CURRENT_DENSITY_LIMIT_A_PER_M = 0.0
ENGINEERING_CONTOUR_STEP_SAFETY_FACTOR = 1.0
ENGINEERING_CONTOUR_MAX_LEVELS = 80
CONTOUR_LEVELS = 8
GRADIENT_HARDWARE_INFEASIBLE_POLICY = "scale_to_limit"

# Allow raised returns to clear the routing spacing below.
# Channel-to-channel assembly needs its own clearance audit.
RETURN_LIFT_LIMIT_M = 0.006
# Contour generation keeps 5 mm spacing. Diagonal turn splices use a separate
# routing target, still above the independent 1.4 mm conductor-body minimum.
ROUTING_CLEARANCE_M = 0.003
# Keep fixed-current changes bounded while repairing the inherited layout.
# Passing field limits remain hard; inherited failures may only improve.
ROUTING_MAX_FIELD_CHANGE_REL = 0.02
ROUTING_FIELD_ACCEPTANCE = "non_regression"
# Y's combined returns need a larger transverse-field allowance. Keep an
# independent B0-component bound; original gradient failures stay reported.
GY_ROUTING_MAX_FIELD_CHANGE_REL = 0.30
GY_ROUTING_MAX_LONGITUDINAL_FIELD_CHANGE_REL = 0.05
GY_ROUTING_FIELD_ACCEPTANCE = "budget_only"
# Realize inherited Y contour-tip kinks at finite radius without changing the
# saved nominal contours or current. The source corridor is measured on the
# declared analytic cylinder; full conductor and field checks remain hard.
GY_CORNER_CLUSTER_SOURCE_DEVIATION_M = 0.001
# Combine returns at the original contour openings used in this example.
ROUTING_SEARCH_CUTS = False

GRADIENT_TEMPLATE = CylinderCoilDesign(
    name=f"{PROJECT_NAME}_template",
    design_context="mr_gradient",
    radius_m=BASE_SURFACE_DIAMETER_M / 2.0,
    length_m=COIL_LENGTH_M,
    angular_divisions=28,
    length_divisions=20,
    rotation_axis=(1.0, 0.0, 0.0),
    rotation_angle_rad=0.0,
    target_scale=TARGET_GRADIENT_STRENGTH_T_PER_M,
    target_gradient_normalization="linear_fit",
    target_gradient_axis="auto",
    target_region_radius_m=DSV_RADIUS_M,
    target_region_resolution=15,
    optimization_method=STREAM_FUNCTION_OPTIMIZATION_METHOD,
    regularization_method="fixed",
    regularization_factor=50.0,
    l_curve_candidate_count=21,
    l_curve_smoothing_window=5,
    target_error_tolerance_percent=5.0,
    target_error_tolerance_absolute_T=0.0,
    target_error_constraint_candidate_count=121,
    contour_discretization_method="engineering",
    max_current_A=MAX_WIRE_CURRENT_A,
    max_wire_current_A=MAX_WIRE_CURRENT_A,
    min_wire_distance_m=MIN_WIRE_DISTANCE_M,
    min_turn_spacing_m=MIN_WIRE_DISTANCE_M,
    surface_current_density_limit_A_per_m=SURFACE_CURRENT_DENSITY_LIMIT_A_PER_M,
    engineering_contour_step_safety_factor=ENGINEERING_CONTOUR_STEP_SAFETY_FACTOR,
    engineering_contour_max_levels=ENGINEERING_CONTOUR_MAX_LEVELS,
    gradient_hardware_infeasible_policy=GRADIENT_HARDWARE_INFEASIBLE_POLICY,
    contour_levels=CONTOUR_LEVELS,
    potential_offset_factor=0.25,
    min_contour_points=4,
    skip_postprocessing=False,
    skip_sweep=not ENABLE_SWEEP,
    skip_inductance=False,
    inductance_source="auto",
    save_wire_trace_csv=EXPORT_WIRE_TRACE_CSV,
    conductor_width_m=WIRE_DIAMETER_M,
    conductor_height_m=WIRE_DIAMETER_M,
    conductor_resistivity_ohm_m=0.018e-6,
    main_field_strength_T=1.5,
    b0_direction=(0.0, 0.0, 1.0),
    field_evaluation_current_source="stream_function",
    cpu_workers=0,
    progress=True,
    advanced_parameters={
        "cross_sectional_points": ROUND_WIRE_CROSS_SECTION_POINTS,
        "normal_shift_length": RETURN_LIFT_LIMIT_M,
        "joint_group_routing_clearance_m": ROUTING_CLEARANCE_M,
        "joint_group_routing_max_field_change_rel": ROUTING_MAX_FIELD_CHANGE_REL,
        "joint_group_routing_field_acceptance": ROUTING_FIELD_ACCEPTANCE,
        "joint_group_routing_search_cuts": ROUTING_SEARCH_CUTS,
    },
)

PLOT_INCLUDE = (
    "wire",
    "wire_sequence",
    "contour_current_orientation",
    "swept_body",
    "contour_groups",
    "layout_field_overview",
    "field_pointclouds",
    "spherical_harmonics",
    "gradient_maps",
    "field_slice_mosaic",
    "gradient_slice_mosaic",
    "concomitant_slice_mosaic",
    "error_maps",
    "regularization",
    "performance_table",
)


PROJECT = CoilProject(
    name=PROJECT_NAME,
    template=GRADIENT_TEMPLATE,
    project_root=PROJECT_ROOT,
    output_dir=OUTPUT_ROOT / PROJECT_NAME,
    save_reports=True,
    save_gallery=False,
    open_plot_book=SHOW_PLOT_BOOK,
    close_previous_plots=False,
    plot_include=PLOT_INCLUDE,
    design_context="mr_gradient",
)

PROJECT.gradient_channel(
    "GX",
    axis="x",
    gradient_strength_T_per_m=TARGET_GRADIENT_STRENGTH_T_PER_M,
    gradient_strength_tolerance_percent=GRADIENT_STRENGTH_TOLERANCE_PERCENT,
    linearity=GRADIENT_LINEARITY,
    slice_maps=GRADIENT_SLICE_MAPS,
    max_wire_current_A=MAX_WIRE_CURRENT_A,
    min_wire_distance_m=MIN_WIRE_DISTANCE_M,
    surface_current_density_limit_A_per_m=SURFACE_CURRENT_DENSITY_LIMIT_A_PER_M,
    engineering_contour_step_safety_factor=ENGINEERING_CONTOUR_STEP_SAFETY_FACTOR,
    engineering_contour_max_levels=ENGINEERING_CONTOUR_MAX_LEVELS,
    contour_levels=CONTOUR_LEVELS,
    gradient_hardware_infeasible_policy=GRADIENT_HARDWARE_INFEASIBLE_POLICY,
    design_overrides={
        "name": f"{PROJECT_NAME}_gx",
        "radius_m": SURFACE_DIAMETERS_M["x"] / 2.0,
    },
    label="GX",
    color="tab:blue",
    metadata={"axis": "x"},
)
PROJECT.gradient_channel(
    "GY",
    axis="y",
    gradient_strength_T_per_m=TARGET_GRADIENT_STRENGTH_T_PER_M,
    gradient_strength_tolerance_percent=GRADIENT_STRENGTH_TOLERANCE_PERCENT,
    linearity=GRADIENT_LINEARITY,
    slice_maps=GRADIENT_SLICE_MAPS,
    max_wire_current_A=MAX_WIRE_CURRENT_A,
    min_wire_distance_m=MIN_WIRE_DISTANCE_M,
    surface_current_density_limit_A_per_m=SURFACE_CURRENT_DENSITY_LIMIT_A_PER_M,
    engineering_contour_step_safety_factor=ENGINEERING_CONTOUR_STEP_SAFETY_FACTOR,
    engineering_contour_max_levels=ENGINEERING_CONTOUR_MAX_LEVELS,
    contour_levels=CONTOUR_LEVELS,
    gradient_hardware_infeasible_policy=GRADIENT_HARDWARE_INFEASIBLE_POLICY,
    design_overrides={
        "name": f"{PROJECT_NAME}_gy",
        "radius_m": SURFACE_DIAMETERS_M["y"] / 2.0,
        "advanced_parameters": {
            **GRADIENT_TEMPLATE.advanced_parameters,
            "sweep_bend_mode": "fillet",
            "sweep_corner_cluster_max_source_deviation_m": GY_CORNER_CLUSTER_SOURCE_DEVIATION_M,
            "joint_group_routing_max_field_change_rel": GY_ROUTING_MAX_FIELD_CHANGE_REL,
            "joint_group_routing_max_longitudinal_field_change_rel": GY_ROUTING_MAX_LONGITUDINAL_FIELD_CHANGE_REL,
            "joint_group_routing_field_acceptance": GY_ROUTING_FIELD_ACCEPTANCE,
        },
    },
    label="GY",
    color="tab:green",
    metadata={"axis": "y"},
)
PROJECT.gradient_channel(
    "GZ",
    axis="z",
    gradient_strength_T_per_m=TARGET_GRADIENT_STRENGTH_T_PER_M,
    gradient_strength_tolerance_percent=GRADIENT_STRENGTH_TOLERANCE_PERCENT,
    linearity=GRADIENT_LINEARITY,
    slice_maps=GRADIENT_SLICE_MAPS,
    max_wire_current_A=MAX_WIRE_CURRENT_A,
    min_wire_distance_m=MIN_WIRE_DISTANCE_M,
    surface_current_density_limit_A_per_m=SURFACE_CURRENT_DENSITY_LIMIT_A_PER_M,
    engineering_contour_step_safety_factor=ENGINEERING_CONTOUR_STEP_SAFETY_FACTOR,
    engineering_contour_max_levels=ENGINEERING_CONTOUR_MAX_LEVELS,
    contour_levels=CONTOUR_LEVELS,
    gradient_hardware_infeasible_policy=GRADIENT_HARDWARE_INFEASIBLE_POLICY,
    design_overrides={
        "name": f"{PROJECT_NAME}_gz",
        "radius_m": SURFACE_DIAMETERS_M["z"] / 2.0,
    },
    label="GZ",
    color="tab:red",
    metadata={"axis": "z"},
)


def export_project_stls(project_run) -> tuple[Path, ...]:
    """Save available swept meshes and report manufacturing readiness separately."""
    if not EXPORT_STL:
        return ()
    saved_files: list[Path] = []
    export_report = []
    for channel_run in project_run.channel_runs:
        entry = {"channel": channel_run.channel.name, "status": "unavailable", "files": []}
        export_report.append(entry)
        result = channel_run.result
        if result is None:
            entry["reason"] = "No solved result is available."
            continue
        output_dir = project_run.output_dir / channel_run.channel.safe_name / "stl"
        prefix = f"{getattr(channel_run.design, 'name', channel_run.channel.safe_name)}_swept_body"
        files = list(save_manufacturing_diagnostics(result, output_dir, prefix=prefix))
        rows = manufacturing_diagnostics_rows(result)
        stl_files = save_swept_bodies_as_stl(result, output_dir, prefix=prefix, fail_on_violations=False)
        files.extend(stl_files)
        if stl_files:
            entry["status"] = "exported"
            missing_parts = {
                part_index
                for part_index, part in enumerate(result.coil_parts, start=1)
                if not getattr(part, "swept_bodies", ()) and part.swept_body is None
            }
            warnings = [
                {
                    "part_index": row["part_index"],
                    "body_index": row.get("body_index"),
                    "reasons": (
                        "no_swept_body"
                        if row["part_index"] in missing_parts
                        else str(row.get("manufacturing_failure_reasons", "")).strip() or "unspecified"
                    ),
                }
                for row in rows
                if row["part_index"] in missing_parts or not row.get("manufacturing_ok", False)
            ]
            entry["manufacturing_ready"] = not warnings
            if warnings:
                entry["manufacturing_warnings"] = warnings
                print(
                    f"STL exported for {channel_run.channel.name} with manufacturing warnings; "
                    f"inspect diagnostics in {output_dir}"
                )
        else:
            entry["reason"] = "No swept bodies are available. Enable sweeping to create them."
        entry["files"] = [path.relative_to(project_run.output_dir).as_posix() for path in files]
        channel_run.files = tuple(dict.fromkeys([*channel_run.files, *files]))
        saved_files.extend(files)
    project_run.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = project_run.output_dir / "stl_export_report.json"
    report_path.write_text(json.dumps(export_report, indent=2) + "\n", encoding="utf-8")
    saved_files.append(report_path)
    project_run.files = tuple(dict.fromkeys([*project_run.files, *saved_files]))
    print(f"STL export report: {report_path}")
    return tuple(saved_files)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main(project: CoilProject | None = None):
    """Solve the configured project; callers may pass an explicit study variant."""
    project = PROJECT if project is None else project
    print(project.describe())
    with native_thread_budget(NATIVE_THREADS):
        project_run = project.solve(open_plot_book=interactive_plots_enabled(project.open_plot_book))
    export_project_stls(project_run)
    return project_run


if __name__ == "__main__":
    project = PROJECT
    project_run = main(project)
    stl_files = tuple(path for path in project_run.files if path.suffix.lower() == ".stl")
    system_run = project_run
    results = project_run.results
    plot_book = open_project_plot_book_after_run(
        project_run, figures=project_run.figures, include=PLOT_INCLUDE, enabled=SHOW_PLOT_BOOK
    )
