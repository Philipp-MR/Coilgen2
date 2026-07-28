from __future__ import annotations

from projects._shared.object_parameters import object_parameters

from projects._shared.example_utils import (
    PROJECT_ROOT,
    example_output_dir,
    open_project_plot_book_after_run,
    parse_example_parameters,
    print_result_summary,
    print_saved_files,
)

from coilgen import CoilGenConfig, CoilGenConfigDesign, CoilProject


from projects.general_examples.parameters import SPLIT_BORE_FLANGE_SMOKE_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
################################################################################
# This example designs on a generated flanged split-bore surface.
PARAMETERS = object_parameters(
    {
        "coilgen": {
            # Bz target field: "x", "y", "z", or an expression such as "x + 0.5*y".
            "field_shape_function": "y",
            # Generated flanged split-bore surface and its shape parameters.
            "coil_mesh_file": "create flanged split bore mesh",
            "flanged_split_bore_mesh_parameter_list": (0.12, 0.2, 0.28, 0.08, 0.2),
            # Treat the generated surface as a cylindrical/split-bore family for topology handling.
            "surface_is_cylinder_flag": True,
            # Spherical target cloud used for fitting the requested field.
            "target_region_radius": 0.05,
            "target_region_resolution": 5,
            # Number of stream-function levels converted to contours.
            "levels": 4,
            # Relative contour offset between min/max stream-function values.
            "pot_offset_factor": 0.25,
            # Small contours with fewer points are removed.
            "min_point_loop_number": 3,
            # Quadrature order for the Biot-Savart sensitivity calculation.
            "gauss_order": 1,
            # Stream-function optimization strategy.
            # Choose exactly one active line:
            #   "tikhonov" uses the regularization scan below.
            #   "target_error_constrained_efficiency" finds the lowest-current solution
            #   that stays inside target_error_tolerance_percent.
            "sf_opt_method": "tikhonov",
            # "sf_opt_method": "target_error_constrained_efficiency",
            "stream_function_cost_function": "field_error_plus_resistance",
            # Fixed-error optimizer settings. Used only by
            # sf_opt_method="target_error_constrained_efficiency".
            "target_error_tolerance_percent": 5.0,
            "target_error_tolerance_absolute_T": 0.0,
            "target_error_constraint_factor_count": 61,
            # tikonov_reg_factor is the L-curve scan center.
            "tikonov_reg_factor": 20.0,
            "regularization_selection_method": "l_curve",
            "l_curve_factor_count": 21,
            "l_curve_factor_min_multiplier": 1e-4,
            "l_curve_factor_max_multiplier": 1e4,
            "l_curve_regularization_factors": (),
            # Parallel CPU usage: 0 uses all logical cores, 1 is serial, -1 keeps one core free.
            "parallel_worker_count": 0,
            # Minimum estimated operation count before parallel workers are used.
            "parallel_min_work_size": 50000,
            # Sensitivity-matrix chunking; higher values create more, smaller worker tasks.
            "sensitivity_chunks_per_worker": 4,
            "sensitivity_calculation_method": "vectorized",
            # Segment chunk sizes for Biot-Savart field evaluation and Neumann inductance.
            "biot_savart_chunk_size": 512,
            "inductance_chunk_size": 256,
            "wire_distance_method": "auto",
            "wire_distance_chunk_size": 512,
            "wire_distance_spatial_min_segments": 3000,
            # Approximate memory cap per numerical chunk.
            "parallel_max_chunk_memory_mb": 64.0,
            "optimization_solver": "auto",
            "optimization_solver_residual_tolerance": 1e-8,
            # Print progress percentages for long calculation stages.
            "progress_flag": True,
            "progress_min_interval_seconds": 1.0,
            "progress_percent_step": 5.0,
            # Inter-group connector method: "direct_3d" or "geodesic".
            "inter_group_connection_path_method": "direct_3d",
            # Keep final electrical estimates active. They use a thin-wire
            # Neumann/Biot-Savart inductance model and copper resistance below.
            "skip_inductance_calculation": False,
            "inductance_source": "auto",
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
            "specific_conductivity_conductor": 0.018e-6,
            # Internal MATLAB-style plotting/STL flags remain disabled in the Python examples.
            "plot_flag": False,
            "save_stl_flag": False,
        },
        "output": {
            # Output folder below outputs/ and filename prefix for saved SVG diagnostics.
            "name": "split_bore_flange",
            # Save the dependency-free SVG plot gallery.
            "save_svg_gallery": False,
        },
        "plots": {
            # Open active Matplotlib windows in Spyder.
            "show_active": False,
            # Close figures from previous runs before plotting this result.
            "close_previous": True,
            # "book" opens one paged window; "windows" opens separate interactive figures.
            "mode": "book",
            # Plot groups to include if active plotting is enabled.
            "include": ("wire", "contour_groups", "field_pointclouds", "error_maps"),
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="split_bore_flange")


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
        print_saved_files(files, title="Saved split-bore outputs:")
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
    print_result_summary(result, title="Split-bore flange example")
    return project, project_run, result, files, figures


if __name__ == "__main__":
    project, project_run, result, files, figures = main()
    plot_book = open_project_plot_book_after_run(
        project_run, figures=figures, include=get_parameters()["plots"].get("include")
    )
