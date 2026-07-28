from __future__ import annotations

from projects._shared.object_parameters import object_parameters

import math
import time
from pathlib import Path

from projects._shared.example_utils import (
    PROJECT_ROOT,
    example_output_dir,
    open_project_plot_book_after_run,
    parse_example_parameters,
    print_result_summary,
    print_saved_files,
    save_standard_reports,
)

from coilgen import CoilGenConfig, CoilGenConfigDesign, CoilProject, save_plot_gallery
from coilgen.validation import (
    apply_ordered_solenoid_wire_path,
    validate_solenoid_against_analytical_formulas,
)


from projects.general_examples.parameters import SOLENOID_CONSTANT_Z_VALIDATION_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
################################################################################
# Run this file directly in Spyder. Edit values in this object, then run again.
PARAMETERS = object_parameters(
    {
        "coilgen": {
            # Constant requested Bz field in tesla. This should produce circular turns
            # on the cylinder, i.e. a standard finite solenoid layout.
            "field_shape_function": "1",
            "target_gradient_strength": 1.0e-4,
            # Generated cylinder: height [m], radius [m], circular divisions, height divisions,
            # rotation axis x/y/z, rotation angle [rad].
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_mesh_parameter_list": (0.8, 0.08, 48, 48, 1, 0, 0, 0),
            # Fit a small central sphere so the validation corresponds to the
            # analytical center-field solenoid formulas.
            "target_region_radius": 0.03,
            "target_region_resolution": 5,
            # Number of stream-function levels. For this constant-field case this is
            # also the intended number of solenoid turns.
            "levels": 14,
            "pot_offset_factor": 0.2,
            "min_point_loop_number": 8,
            "gauss_order": 1,
            # Stream-function optimization strategy.
            # Keep "tikhonov" for this validation because it fixes the inverse
            # problem while the analytical comparison tests the extracted solenoid.
            # Uncomment the second line only for experimentation.
            "sf_opt_method": "tikhonov",
            # "sf_opt_method": "target_error_constrained_efficiency",
            "stream_function_cost_function": "field_error_plus_resistance",
            # Fixed-error optimizer settings. Used only by
            # sf_opt_method="target_error_constrained_efficiency".
            "target_error_tolerance_percent": 5.0,
            "target_error_tolerance_absolute_T": 0.0,
            "target_error_constraint_factor_count": 61,
            # Fixed regularization keeps this validation independent from L-curve
            # model-selection behavior.
            "tikonov_reg_factor": 20.0,
            "regularization_selection_method": "fixed",
            "l_curve_scan_method": "normal_equations",
            "l_curve_curvature_smoothing_window": 5,
            "l_curve_factor_count": 21,
            "l_curve_factor_min_multiplier": 1e-4,
            "l_curve_factor_max_multiplier": 1e4,
            "l_curve_regularization_factors": (),
            # Parallel CPU usage: 0 uses all logical cores, 1 is serial, -1 keeps one core free.
            "parallel_worker_count": 0,
            "parallel_min_work_size": 50000,
            "sensitivity_chunks_per_worker": 4,
            "sensitivity_calculation_method": "vectorized",
            "biot_savart_chunk_size": 512,
            "inductance_chunk_size": 256,
            "wire_distance_method": "auto",
            "wire_distance_chunk_size": 512,
            "wire_distance_spatial_min_segments": 3000,
            "parallel_max_chunk_memory_mb": 64.0,
            "optimization_solver": "auto",
            "optimization_solver_residual_tolerance": 1e-8,
            "progress_flag": True,
            "progress_min_interval_seconds": 1.0,
            "progress_percent_step": 5.0,
            "inter_group_connection_path_method": "direct_3d",
            "inter_group_connection_candidate_count": 16,
            "within_group_cut_candidate_count": 12,
            # This validation uses the extracted circular contours and then replaces
            # the generic loop-group routing with a deterministic solenoid-style
            # ordered path. That is the physically correct route for this special
            # textbook case.
            "skip_postprocessing": True,
            # Sweep is not required for this physics validation. Inductance stays enabled.
            "skip_sweep": True,
            "skip_inductance_calculation": False,
            "skip_calculation_min_winding_distance": True,
            # Rectangular conductor dimensions used by resistance and the regularized
            # Neumann inductance estimate.
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
            "specific_conductivity_conductor": 0.018e-6,
            "inductance_source": "auto",
            "plot_flag": False,
            "save_stl_flag": False,
        },
        "validation": {
            # The analytical comparisons use the field at this point.
            "center_point_m": (0.0, 0.0, 0.0),
            # Build a single ordered solenoid path from the extracted circular rings.
            "build_ordered_solenoid_wire_path": True,
            # Azimuth at which each turn is opened before connecting to the next turn.
            "wire_path_cut_angle_rad": 0.0,
            # Quadrature order for complete elliptic integrals in the analytical
            # coaxial-loop inductance formula.
            "elliptic_quadrature_order": 96,
            # Set warn_only=False if this example should raise on failed tolerances.
            "warn_only": True,
            "max_center_field_error_percent": 10.0,
            "max_inductance_error_percent": 25.0,
        },
        "output": {
            "name": "solenoid_constant_z_validation",
            "save_svg_gallery": False,
        },
        "plots": {
            "show_active": True,
            "close_previous": True,
            "mode": "book",
            "include": (
                "wire",
                "raw_contours",
                "field_pointclouds",
                "error_maps",
            ),
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="solenoid_constant_z_validation")


def build_config(parameters: dict | None = None) -> CoilGenConfig:
    parameters = get_parameters() if parameters is None else parameters
    return CoilGenConfig(**parameters["coilgen"])


def print_solenoid_validation(validation: dict[str, float | int], parameters: dict) -> None:
    max_field_error = float(parameters["validation"]["max_center_field_error_percent"])
    max_inductance_error = float(parameters["validation"]["max_inductance_error_percent"])

    print("Solenoid constant-Bz validation")
    print(f"turn count: {validation['turn_count']}")
    print(f"cylinder radius: {validation['cylinder_radius_m']:.6g} m")
    print(f"cylinder height: {validation['cylinder_height_m']:.6g} m")
    print(f"effective turn radius: {validation['effective_radius_m']:.6g} m")
    print(f"effective turn z span: {validation['effective_height_m']:.6g} m")
    print(f"equivalent round wire radius: {validation['wire_radius_m']:.6g} m")
    print(f"needed current: {validation['needed_current_a']:.6g} A")
    print("center Bz:")
    print(f"  requested: {validation['requested_center_bz_t']:.6g} T")
    print(f"  numerical wire path: {validation['numerical_center_bz_t']:.6g} T")
    print(f"  analytical circular-loop stack: {validation['circular_loop_stack_center_bz_t']:.6g} T")
    print(f"  finite solenoid formula: {validation['finite_solenoid_center_bz_t']:.6g} T")
    print(f"  numerical per ampere: {validation['numerical_center_bz_per_a']:.6g} T/A")
    print(f"  circular-loop stack per ampere: {validation['circular_loop_stack_center_bz_per_a']:.6g} T/A")
    print(f"  finite solenoid per ampere: {validation['finite_solenoid_center_bz_per_a']:.6g} T/A")
    print(f"  long solenoid per ampere: {validation['long_solenoid_center_bz_per_a']:.6g} T/A")
    print(f"  wire path vs circular-loop stack error: {validation['center_vs_circular_loop_stack_error_percent']:.3g}%")
    print(f"  wire path vs requested center error: {validation['center_vs_requested_error_percent']:.3g}%")
    print("inductance:")
    print(f"  numerical Neumann estimate: {validation['numerical_inductance_h']:.6g} H")
    print(f"  analytical circular-loop stack: {validation['circular_loop_stack_inductance_h']:.6g} H")
    print(f"  long solenoid formula: {validation['long_solenoid_inductance_h']:.6g} H")
    print(
        f"  numerical vs circular-loop stack error: {validation['inductance_vs_circular_loop_stack_error_percent']:.3g}%"
    )
    _print_tolerance_result(
        "center field",
        float(validation["center_vs_circular_loop_stack_error_percent"]),
        max_field_error,
        bool(parameters["validation"]["warn_only"]),
    )
    _print_tolerance_result(
        "inductance",
        float(validation["inductance_vs_circular_loop_stack_error_percent"]),
        max_inductance_error,
        bool(parameters["validation"]["warn_only"]),
    )


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    config = build_config(parameters)
    output = parameters["output"]
    output_dir = example_output_dir(output["name"])
    project = CoilProject(
        output["name"],
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
        save_reports=False,
        write_summary=False,
    )
    project.channel("main", design=CoilGenConfigDesign(name=output["name"], config=config))
    project_run = project.solve()
    result = project_run.channel_runs[0].result
    if parameters["validation"]["build_ordered_solenoid_wire_path"]:
        result = apply_ordered_solenoid_wire_path(
            result,
            config,
            cut_angle_rad=float(parameters["validation"]["wire_path_cut_angle_rad"]),
        )
        project_run.set_channel_result("main", result)
    validation = validate_solenoid_against_analytical_formulas(
        result,
        config,
        center_point_m=parameters["validation"]["center_point_m"],
        elliptic_quadrature_order=int(parameters["validation"]["elliptic_quadrature_order"]),
    ).as_dict()
    print_solenoid_validation(validation, parameters)

    files: list[Path] = list(project_run.write_summary())
    if output["save_svg_gallery"]:
        start = time.perf_counter()
        gallery_files = save_plot_gallery(result, output_dir, prefix=output["name"])
        files.extend(gallery_files)
        elapsed = time.perf_counter() - start
        print(f"Saved SVG gallery in {elapsed:.3f} s ({len(gallery_files)} file(s)).")
    files.extend(save_standard_reports(result, output_dir, prefix=output["name"]))
    if files:
        print_saved_files(files, title="Saved diagnostics, plots, and reports:")

    plots = parameters["plots"]
    figures = (
        project.plot(
            include=plots["include"],
            close_previous=plots["close_previous"],
            mode=plots.get("mode", "book"),
        )
        if plots["show_active"]
        else []
    )
    print_result_summary(result, title="Constant-Bz solenoid design")
    print("Spyder variables available after run: project, project_run, result, validation, files, figures")
    return project, project_run, result, validation, files, figures


def _print_tolerance_result(name: str, error_percent: float, limit_percent: float, warn_only: bool) -> None:
    if not math.isfinite(error_percent):
        message = f"{name} validation error is not finite."
        if warn_only:
            print(f"  WARNING: {message}")
            return
        raise ValueError(message)
    if error_percent <= limit_percent:
        print(f"  PASS: {name} error <= {limit_percent:.3g}%")
        return
    message = f"{name} error {error_percent:.3g}% exceeds {limit_percent:.3g}%."
    if warn_only:
        print(f"  WARNING: {message}")
        return
    raise ValueError(message)


if __name__ == "__main__":
    project, project_run, result, validation, files, figures = main()
    plot_book = open_project_plot_book_after_run(
        project_run, figures=figures, include=get_parameters()["plots"].get("include")
    )
