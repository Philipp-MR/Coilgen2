from __future__ import annotations

# Support direct terminal/Spyder execution from any working directory.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    for _path in (_Path(__file__).resolve().parents[3], _Path(__file__).resolve().parents[3] / "src"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

from projects._shared.object_parameters import object_parameters

from projects._shared.example_utils import (
    interactive_plots_enabled,
    native_thread_budget,
    PROJECT_ROOT,
    example_output_dir,
    open_project_plot_book_after_run,
    parse_example_parameters,
    print_result_summary,
    print_saved_files,
)

from coilgen import CoilGenConfig, CoilGenConfigDesign, CoilProject


from projects.general_examples.parameters import SPYDER_DESIGN_EXAMPLE_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
# Runtime/chunking defaults are shared in parameters/defaults.py.
################################################################################
# This is the recommended Spyder entry point for interactive inspection.
PARAMETERS = object_parameters(
    {
        "coilgen": {
            # Bz target field: "x", "y", "z", or an expression such as "x + 0.5*y".
            "field_shape_function": "y",
            # Dense cylinder: length [m], radius [m], angular/axial divisions, rotation axis x/y/z, angle [rad].
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_mesh_parameter_list": (0.5, 0.25, 40, 40, 1, 0, 0, 0),
            # Spherical target cloud used for fitting the requested field.
            "target_region_radius": 0.08,
            "target_region_resolution": 5,
            # Number of stream-function levels converted to contours.
            "levels": 8,
            # Relative contour offset between min/max stream-function values.
            "pot_offset_factor": 0.25,
            # Small contours with fewer points are removed.
            "min_point_loop_number": 4,
            # Quadrature order for the Biot-Savart sensitivity calculation.
            "gauss_order": 1,
            # Stream-function optimization scheme.
            # Choose exactly one active line:
            #   "tikhonov" keeps the standard CoilGen inverse solve and selects regularization below.
            #   "target_error_constrained_efficiency" minimizes current/resistance effort while enforcing
            #   target_error_tolerance_percent. This is useful for comparing different support geometries
            #   at the same allowed field error.
            "sf_opt_method": "tikhonov",
            # "sf_opt_method": "target_error_constrained_efficiency",
            # Objective used by the current solvers. Keep this unless you are testing new cost functions.
            "stream_function_cost_function": "field_error_plus_resistance",
            # L-curve regularization scan. Used by sf_opt_method="tikhonov";
            # tikonov_reg_factor is the scan center for automatic searches.
            "tikonov_reg_factor": 50.0,
            "regularization_selection_method": "l_curve",
            # Use exact normal-equation candidate solves and smooth curvature to avoid underregularized local knees.
            "l_curve_scan_method": "normal_equations",
            "l_curve_curvature_smoothing_window": 5,
            "l_curve_factor_count": 21,
            "l_curve_factor_min_multiplier": 1e-4,
            "l_curve_factor_max_multiplier": 1e4,
            "l_curve_regularization_factors": (),
            # Fixed-error efficiency optimizer settings. Used only when
            # sf_opt_method="target_error_constrained_efficiency".
            # Example: 5.0 means accept at most 5 percent weighted target-field residual,
            # then choose the lowest-current solution that satisfies that budget.
            "target_error_tolerance_percent": 5.0,
            # Optional absolute residual allowance in tesla. Usually keep 0 for gradient studies.
            "target_error_tolerance_absolute_T": 0.0,
            # Number of candidate regularization factors scanned before local refinement.
            "target_error_constraint_factor_count": 61,
            # Manufacturing turn-spacing threshold in meters. If > 0, contour levels are reduced as needed.
            "min_turn_distance": 0.0,
            # Optional extra stream-function smoothing for spacing; 0 keeps only the hard level-count threshold.
            "turn_spacing_regularization_factor": 0.0,
            "turn_spacing_max_iterations": 4,
            "turn_spacing_penalty_growth": 10.0,
            # Print progress percentages for long calculation stages.
            "progress_flag": True,
            # Inter-group connector method: "direct_3d" or "geodesic".
            "inter_group_connection_path_method": "direct_3d",
            # Keep postprocessing and sweep enabled for a complete layout.
            "skip_postprocessing": False,
            "skip_sweep": False,
            "skip_inductance_calculation": False,
            # Rectangular conductor dimensions used for resistance/sweep estimates.
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
            "specific_conductivity_conductor": 0.018e-6,
            "inductance_source": "auto",
            # Internal MATLAB-style plotting/STL flags remain disabled in the Python examples.
            "plot_flag": False,
            "save_stl_flag": False,
        },
        "output": {
            # Output folder below outputs/ and filename prefix if SVG export is enabled.
            "name": "spyder_design_example",
            # Keep SVG export off by default so interactive Spyder runs stay quick.
            "save_svg_gallery": False,
        },
        "plots": {
            # Open active Matplotlib windows in Spyder.
            "show_active": True,
            # Close figures from previous runs before plotting this result.
            "close_previous": True,
            # "book" opens one paged window; "windows" opens separate interactive figures.
            "mode": "book",
            # Plot groups to include in the active Matplotlib view.
            "include": (
                "wire",
                "raw_contours",
                "contour_groups",
                "stream_function",
                "current_density",
                "interconnect_debug",
                "field_pointclouds",
                "gradient_maps",
                "error_maps",
                "regularization",
                "field_cross_sections",
                "gradient_cross_sections",
                "error_cross_sections",
            ),
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="spyder_design_example")


def build_config(parameters: dict | None = None) -> CoilGenConfig:
    parameters = get_parameters() if parameters is None else parameters
    return CoilGenConfig(**parameters["coilgen"])


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    output = parameters["output"]
    project = CoilProject(
        output["name"],
        output_dir=example_output_dir(output["name"]),
        project_root=PROJECT_ROOT,
    )
    project.channel(
        "main",
        design=CoilGenConfigDesign(name=output["name"], config=build_config(parameters)),
        save_gallery=bool(output["save_svg_gallery"]),
    )
    project_run = project.solve()
    result = project_run.channel_runs[0].result
    files = list(project_run.files)
    if files:
        print_saved_files(files, title="Saved project outputs:")
    plots = parameters["plots"]
    figures = (
        project.plot(
            include=plots["include"],
            close_previous=plots["close_previous"],
            mode=plots.get("mode", "book"),
        )
        if interactive_plots_enabled(plots["show_active"])
        else []
    )
    print_result_summary(result, title="Spyder design example")
    print("Spyder variables available after run: project, project_run, result, files, figures")
    return project, project_run, result, files, figures


if __name__ == "__main__":
    with native_thread_budget():
        project, project_run, result, files, figures = main()
        plot_settings = get_parameters()["plots"]
        plot_book = open_project_plot_book_after_run(
            project_run, figures=figures, include=plot_settings.get("include"), enabled=plot_settings["show_active"]
        )
