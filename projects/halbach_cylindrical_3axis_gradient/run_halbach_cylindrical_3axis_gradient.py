from __future__ import annotations

import math
import json
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
    GradientLinearitySpec,
    GradientSliceMapSpec,
    ImportedMeshCoilDesign,
    save_manufacturing_diagnostics,
    save_swept_bodies_as_stl,
)
from coilgen.manufacturing import SweptBodyExportError
from projects._shared.example_utils import (
    interactive_plots_enabled,
    native_thread_budget,
    open_project_plot_book_after_run,
)
from projects.halbach_cylindrical_3axis_gradient.paths import HALBACH_DATA_ROOT, halbach_output_dir


OUTPUT_ROOT = halbach_output_dir()


################################################################################
# USER SETTINGS
################################################################################
# Open this file in Spyder, edit these objects, then press Run.
# All lengths are metres. A gradient strength of 0.010 means 10 mT/m.

PROJECT_NAME = "halbach_cylindrical_3axis_gradient"
SHOW_PLOT_BOOK = True
NATIVE_THREADS = 1  # BLAS threads inside each CoilGen worker; increase deliberately for large serial solves.
EXPORT_STL = True
EXPORT_WIRE_TRACE_CSV = True
ENABLE_SWEEP = True
# "warn" always writes visualization STLs and labels rejected bodies clearly.
# "skip" preserves the solved reports/plots but omits rejected STL files.
# Use "raise" for a strict manufacturing/CI run that must stop on rejection.
STL_EXPORT_REJECTION_POLICY = "warn"  # choices: warn | skip | raise

# Smooth only the extracted, unconnected contour loops. Circular cylinders are
# developed exactly to 2D; other meshes use a local harmonic chart. A periodic
# cubic B-spline is sampled at regular physical arc length and mapped back with
# exact triangle provenance. Connected routes, return paths, and inter-group
# connectors are not passed through it.
ENABLE_UNCONNECTED_CONTOUR_SMOOTHING = True
# "mesh_coupled" attenuates one-element-wavelength contour motion by the
# requested gain. Refining the support mesh therefore reduces the physical
# smoothing wavelength automatically. "legacy" retains the SciPy residual-
# bound behavior controlled by CONTOUR_SMOOTHING_FACTOR.
CONTOUR_SMOOTHING_SCALE_MODE = "mesh_coupled"  # choices: mesh_coupled | legacy
CONTOUR_SMOOTHING_MESH_GAIN = 0.10  # Retain 10% of a one-mesh-length oscillation.
CONTOUR_SMOOTHING_WORK_SAMPLES_PER_MESH_LENGTH = 6
CONTOUR_SMOOTHING_OUTPUT_SAMPLES_PER_MESH_LENGTH = 8
CONTOUR_SMOOTHING_FACTOR = 0.25  # Used only when SCALE_MODE is "legacy".
ENABLE_CONTOUR_SMOOTHING_DISPLACEMENT_CAP = True
CONTOUR_SMOOTHING_MAX_DISPLACEMENT_M = 0.001
CONTOUR_SMOOTHING_TARGET_SEGMENT_LENGTH_M = 0.001
CONTOUR_SMOOTHING_PATCH_RINGS = 4

# Lift routed return/crossing segments away from the support along its mesh
# normals. This does not offset the unconnected contours or the complete coil.
ENABLE_NORMAL_SHIFT = True
NORMAL_SHIFT_LENGTH_M = 0.003  # 3 mm maximum shift per routing level.
NORMAL_SHIFT_TRANSITION_FACTOR = 5.0  # Transition length = factor * shift length.

# Axis-specific octagonal current-carrying surfaces exported in millimetres.
# CoilGen converts their coordinates to metres as they are loaded.
SURFACE_MESH_FILES = {
    "x": "Octagonal Gradients - Gx Surface_remeshed.stl",
    "y": "Octagonal Gradients - Gy Surface_remeshed.stl",
    "z": "Octagonal Gradients - Gz Surface_remeshed.stl",
}
SURFACE_MESH_LENGTH_SCALE_TO_M = 1.0e-3
DSV_DIAMETER_M = 0.050
DSV_RADIUS_M = DSV_DIAMETER_M / 2.0
WIRE_DIAMETER_M = 0.0014  # Used when SWEEP_CROSS_SECTION is "round".
SWEEP_CROSS_SECTION = "trapezoid"  # "rectangle", "round", or "trapezoid"
RECTANGLE_WIDTH_M = 0.0017
RECTANGLE_HEIGHT_M = 0.0017
SWEEP_MIN_BEND_RADIUS_M = 0.0025
TRAPEZOID_BOTTOM_WIDTH_M = 0.00196
TRAPEZOID_TOP_WIDTH_M = 0.00160
TRAPEZOID_HEIGHT_M = 0.00180
TRAPEZOID_TOP_WIDTH_RATIO = TRAPEZOID_TOP_WIDTH_M / TRAPEZOID_BOTTOM_WIDTH_M


def _round_wire_cross_section_points(diameter_m: float, point_count: int = 64) -> tuple[tuple[float, float], ...]:
    radius = float(diameter_m) / 2.0
    return tuple(
        (
            radius * math.cos(2.0 * math.pi * point_index / point_count),
            radius * math.sin(2.0 * math.pi * point_index / point_count),
        )
        for point_index in range(point_count)
    )


def _rectangle_cross_section_points(width_m: float, height_m: float) -> tuple[tuple[float, float], ...]:
    width = float(width_m)
    height = float(height_m)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width_m and height_m must be positive")
    half_width = width / 2.0
    half_height = height / 2.0
    return (
        (-half_width, -half_height),
        (+half_width, -half_height),
        (+half_width, +half_height),
        (-half_width, +half_height),
    )


def _trapezoid_cross_section_points(
    width_m: float,
    height_m: float | None = None,
    top_width_ratio: float = 0.9,
) -> tuple[tuple[float, float], ...]:
    """
    Create a trapezoid centered at (0, 0).

    The bottom has the full width_m.
    The top width is top_width_ratio * width_m.

    Points are returned counter-clockwise.
    """
    width = float(width_m)
    height = width if height_m is None else float(height_m)

    if width <= 0.0 or height <= 0.0:
        raise ValueError("width_m and height_m must be positive")

    if not 0.0 < top_width_ratio <= 1.0:
        raise ValueError("top_width_ratio must be between 0 and 1")

    bottom_half_width = width / 2.0
    top_half_width = width * top_width_ratio / 2.0
    half_height = height / 2.0

    return (
        (-bottom_half_width, -half_height),  # bottom left
        (+bottom_half_width, -half_height),  # bottom right
        (+top_half_width, +half_height),  # top right
        (-top_half_width, +half_height),  # top left
    )


def _sweep_cross_section_points() -> tuple[tuple[float, float], ...]:
    shape = SWEEP_CROSS_SECTION.strip().lower().replace("-", "_")
    if shape in {"rectangle", "rectangular", "square"}:
        return _rectangle_cross_section_points(RECTANGLE_WIDTH_M, RECTANGLE_HEIGHT_M)
    if shape == "round":
        return _round_wire_cross_section_points(WIRE_DIAMETER_M)
    if shape == "trapezoid":
        return _trapezoid_cross_section_points(
            TRAPEZOID_BOTTOM_WIDTH_M,
            height_m=TRAPEZOID_HEIGHT_M,
            top_width_ratio=TRAPEZOID_TOP_WIDTH_RATIO,
        )
    raise ValueError(
        f"Unsupported SWEEP_CROSS_SECTION {SWEEP_CROSS_SECTION!r}; expected 'rectangle', 'round', or 'trapezoid'"
    )


def _active_conductor_dimensions() -> tuple[float, float]:
    shape = SWEEP_CROSS_SECTION.strip().lower().replace("-", "_")
    if shape in {"rectangle", "rectangular", "square"}:
        return RECTANGLE_WIDTH_M, RECTANGLE_HEIGHT_M
    if shape == "round":
        return WIRE_DIAMETER_M, WIRE_DIAMETER_M
    if shape == "trapezoid":
        # Use the maximum width for clearance and the physical depth for area.
        return TRAPEZOID_BOTTOM_WIDTH_M, TRAPEZOID_HEIGHT_M
    raise ValueError(
        f"Unsupported SWEEP_CROSS_SECTION {SWEEP_CROSS_SECTION!r}; expected 'rectangle', 'round', or 'trapezoid'"
    )


SWEEP_CROSS_SECTION_POINTS = _sweep_cross_section_points()
CONDUCTOR_WIDTH_M, CONDUCTOR_HEIGHT_M = _active_conductor_dimensions()

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
# Balance ampere-turn efficiency on the transverse-field cylinder while keeping
# the same 10 mT/m target and the shared 5 mm template/manufacturing default.
AXIS_MIN_WIRE_DISTANCE_M = {
    "x": 0.0050,
    "y": 0.0038,
    "z": 0.0060,
}
SURFACE_CURRENT_DENSITY_LIMIT_A_PER_M = 0.0
ENGINEERING_CONTOUR_STEP_SAFETY_FACTOR = 1.0
ENGINEERING_CONTOUR_MAX_LEVELS = 80
CONTOUR_LEVELS = 8
GRADIENT_HARDWARE_INFEASIBLE_POLICY = "scale_to_limit"

GRADIENT_TEMPLATE = ImportedMeshCoilDesign(
    name=f"{PROJECT_NAME}_template",
    design_context="mr_gradient",
    mesh_file=SURFACE_MESH_FILES["x"],
    geometry_source_path=HALBACH_DATA_ROOT,
    mesh_length_scale_to_m=SURFACE_MESH_LENGTH_SCALE_TO_M,
    # The imported meshes are axially extruded octagons with two open ends.
    surface_is_cylinder=True,
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
    min_wire_distance_m=AXIS_MIN_WIRE_DISTANCE_M["x"],
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
    conductor_width_m=CONDUCTOR_WIDTH_M,
    conductor_height_m=CONDUCTOR_HEIGHT_M,
    conductor_resistivity_ohm_m=0.018e-6,
    main_field_strength_T=1.5,
    b0_direction=(0.0, 0.0, 1.0),
    field_evaluation_current_source="stream_function",
    cpu_workers=0,
    progress=True,
    advanced_parameters={
        "cross_sectional_points": SWEEP_CROSS_SECTION_POINTS,
        "contour_path_regularization_mode": (
            "parameterized_bspline" if ENABLE_UNCONNECTED_CONTOUR_SMOOTHING else "off"
        ),
        "contour_resample_target_segment_length": CONTOUR_SMOOTHING_TARGET_SEGMENT_LENGTH_M,
        "contour_smoothing_factor": CONTOUR_SMOOTHING_FACTOR,
        "contour_smoothing_scale_mode": CONTOUR_SMOOTHING_SCALE_MODE,
        "contour_smoothing_mesh_gain": CONTOUR_SMOOTHING_MESH_GAIN,
        "contour_smoothing_work_samples_per_mesh_length": (CONTOUR_SMOOTHING_WORK_SAMPLES_PER_MESH_LENGTH),
        "contour_smoothing_output_samples_per_mesh_length": (CONTOUR_SMOOTHING_OUTPUT_SAMPLES_PER_MESH_LENGTH),
        "contour_smoothing_patch_rings": CONTOUR_SMOOTHING_PATCH_RINGS,
        # A zero cap explicitly disables displacement limiting in the smoother.
        # Spacing preservation remains active independently.
        "contour_smoothing_max_displacement": (
            CONTOUR_SMOOTHING_MAX_DISPLACEMENT_M if ENABLE_CONTOUR_SMOOTHING_DISPLACEMENT_CAP else 0.0
        ),
        "contour_smoothing_preserve_min_turn_distance": True,
        # Remove only redundant, nearly collinear rings before triangulation.
        # This prevents micron-scale lateral edges without changing the route.
        "sweep_path_simplification": "exact",
        # Bound both the longitudinal facet length and the support chord error.
        # Inserted samples retain any routed normal lift above the cylinder.
        "sweep_max_segment_length": 0.005,
        "sweep_max_chord_error": 0.00005,
        "sweep_max_frame_rotation_deg": 5.0,
        # Resolve sharp return/connector corners into tangent-continuous bends.
        # 2.5 mm keeps the 1.202 mm square-profile envelope at kappa*r < 0.5.
        "sweep_bend_mode": "fillet",
        "sweep_min_bend_radius": SWEEP_MIN_BEND_RADIUS_M,
        "sweep_corner_detection_angle_deg": 15.0,
        "sweep_max_curvature_envelope_ratio": 0.5,
        "sweep_self_intersection_check": "capsule_fast",
        "skip_normal_shift": not ENABLE_NORMAL_SHIFT,
        "normal_shift_length": NORMAL_SHIFT_LENGTH_M,
        "normal_sweep_smooth_factor": NORMAL_SHIFT_TRANSITION_FACTOR,
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
    output_dir=OUTPUT_ROOT,
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
        "mesh_file": SURFACE_MESH_FILES["x"],
        "min_turn_spacing_m": AXIS_MIN_WIRE_DISTANCE_M["x"],
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
    min_wire_distance_m=AXIS_MIN_WIRE_DISTANCE_M["y"],
    surface_current_density_limit_A_per_m=SURFACE_CURRENT_DENSITY_LIMIT_A_PER_M,
    engineering_contour_step_safety_factor=ENGINEERING_CONTOUR_STEP_SAFETY_FACTOR,
    engineering_contour_max_levels=ENGINEERING_CONTOUR_MAX_LEVELS,
    contour_levels=CONTOUR_LEVELS,
    gradient_hardware_infeasible_policy=GRADIENT_HARDWARE_INFEASIBLE_POLICY,
    design_overrides={
        "name": f"{PROJECT_NAME}_gy",
        "mesh_file": SURFACE_MESH_FILES["y"],
        "min_turn_spacing_m": AXIS_MIN_WIRE_DISTANCE_M["y"],
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
    min_wire_distance_m=AXIS_MIN_WIRE_DISTANCE_M["z"],
    surface_current_density_limit_A_per_m=SURFACE_CURRENT_DENSITY_LIMIT_A_PER_M,
    engineering_contour_step_safety_factor=ENGINEERING_CONTOUR_STEP_SAFETY_FACTOR,
    engineering_contour_max_levels=ENGINEERING_CONTOUR_MAX_LEVELS,
    contour_levels=CONTOUR_LEVELS,
    gradient_hardware_infeasible_policy=GRADIENT_HARDWARE_INFEASIBLE_POLICY,
    design_overrides={
        "name": f"{PROJECT_NAME}_gz",
        "mesh_file": SURFACE_MESH_FILES["z"],
        "min_turn_spacing_m": AXIS_MIN_WIRE_DISTANCE_M["z"],
    },
    label="GZ",
    color="tab:red",
    metadata={"axis": "z"},
)


def export_project_stls(project_run) -> tuple[Path, ...]:
    if not EXPORT_STL:
        return ()
    rejection_policy = str(STL_EXPORT_REJECTION_POLICY).strip().lower()
    if rejection_policy not in {"warn", "skip", "raise"}:
        raise ValueError("STL_EXPORT_REJECTION_POLICY must be 'warn', 'skip', or 'raise'.")
    saved_files: list[Path] = []
    for channel_run in project_run.channel_runs:
        result = channel_run.result
        if result is None:
            continue
        output_dir = project_run.output_dir / channel_run.channel.safe_name / "stl"
        prefix = f"{getattr(channel_run.design, 'name', channel_run.channel.safe_name)}_swept_body"
        diagnostic_files = save_manufacturing_diagnostics(result, output_dir, prefix=prefix)
        channel_run.files = tuple(dict.fromkeys([*channel_run.files, *diagnostic_files]))
        saved_files.extend(diagnostic_files)
        status_path = output_dir / f"{prefix}_export_status.json"

        def save_status(status, reason, paths=()):
            record = {
                "channel": channel_run.channel.name,
                "status": status,
                "policy": rejection_policy,
                "manufacturing_ready": status == "exported",
                "reason": reason,
                "stl_files": [path.name for path in paths],
                "manufacturing_failure_reasons": [
                    part.manufacturing_diagnostics.get("manufacturing_failure_reasons", "")
                    for part in result.coil_parts
                ],
            }
            status_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            channel_run.files = tuple(dict.fromkeys([*channel_run.files, status_path]))
            saved_files.append(status_path)

        status, reason = "exported", ""
        try:
            stl_files = save_swept_bodies_as_stl(result, output_dir, prefix=prefix, fail_on_violations=True)
        except SweptBodyExportError as exc:
            # Readiness checks may add export-specific diagnostics. Persist the
            # final values even when strict export raises or the mesh is skipped.
            save_manufacturing_diagnostics(result, output_dir, prefix=prefix)
            reason = str(exc)
            if rejection_policy == "raise":
                save_status("blocked", reason)
                project_run.files = tuple(dict.fromkeys([*project_run.files, *saved_files]))
                raise
            if rejection_policy == "skip":
                print(f"WARNING: STL export skipped for {channel_run.channel.name}: {exc}")
                save_status("blocked", reason)
                continue
            print(
                f"WARNING: {channel_run.channel.name} is not manufacturing-ready: {exc} "
                "Writing visualization STL files because STL_EXPORT_REJECTION_POLICY='warn'."
            )
            stl_files = save_swept_bodies_as_stl(
                result,
                output_dir,
                prefix=prefix,
                fail_on_violations=False,
            )
            status = "diagnostic_only"
        if not stl_files:
            status, reason = "unavailable", "No swept bodies were available."
        save_manufacturing_diagnostics(result, output_dir, prefix=prefix)
        save_status(status, reason, stl_files)
        channel_run.files = tuple(dict.fromkeys([*channel_run.files, *stl_files]))
        saved_files.extend(stl_files)
    project_run.files = tuple(dict.fromkeys([*project_run.files, *saved_files]))
    return tuple(dict.fromkeys(saved_files))


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main(project: CoilProject | None = None):
    """Validate every axis before starting the imported-surface study."""
    project = PROJECT if project is None else project
    reports = project.validate()
    invalid = [f"{name}: {report.format()}" for name, report in reports.items() if not report.ok]
    root = Path(project.project_root) if project.project_root is not None else PROJECT_ROOT
    for name, design in project.resolved_designs().items():
        config = design.to_config()
        source = str(config.coil_mesh_file)
        if source.lower().endswith(".stl"):
            path = config.resolved_geometry_source_path(root) / Path(source).expanduser()
            if not path.is_file():
                invalid.append(f"{name}: input surface STL not found: {path.resolve()}")
    if invalid:
        raise ValueError("Halbach project input/configuration preflight failed:\n" + "\n".join(invalid))
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
