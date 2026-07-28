from __future__ import annotations

from projects._shared.object_parameters import object_parameters

from pathlib import Path

import numpy as np

from projects._shared.example_utils import (
    PROJECT_ROOT,
    example_output_dir,
    open_project_plot_book_after_run,
    parse_example_parameters,
    print_result_summary,
    print_saved_files,
)

from coilgen import CoilGenConfig, CoilGenConfigDesign, CoilProject, save_route_diagnostics


from projects.general_examples.parameters import CSV_TARGET_FIELD_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
################################################################################
# This example writes a small CSV target field, imports it, and designs a coil.
PARAMETERS = object_parameters(
    {
        "csv_target": {
            # CSV filename written below outputs/<output.name>/.
            "filename": "y_gradient_target.csv",
            # Cartesian grid spans -axis_limit to +axis_limit in x/y/z.
            "axis_limit": 0.04,
            # Number of samples per Cartesian axis before spherical filtering.
            "axis_points": 5,
            # Only grid points inside this radius are written to the target CSV.
            "sphere_radius": 0.04,
        },
        "coilgen": {
            # Generated cylinder: radius, height, circular divisions, height divisions, normal sign, x/y/z rotation.
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_mesh_parameter_list": (0.25, 0.1, 8, 4, 1, 0, 0, 0),
            # Named scalar Bz map in the CSV columns, e.g. lin_y. Bx and By are set to zero.
            "target_field_definition_field_name": "lin_y",
            # Scale factor applied to imported target field values.
            "target_gradient_strength": 1.0,
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
            "tikonov_reg_factor": 10.0,
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
            # Skip swept solid-body generation, but keep electrical estimates active.
            "skip_sweep": True,
            "skip_inductance_calculation": False,
            "inductance_source": "auto",
            # Rectangular copper conductor for final resistance and equivalent
            # round-wire radius in the Neumann/Biot-Savart inductance estimate.
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
            "specific_conductivity_conductor": 0.018e-6,
            # Internal MATLAB-style plotting/STL flags remain disabled in the Python examples.
            "plot_flag": False,
            "save_stl_flag": False,
        },
        "output": {
            # Output folder below outputs/.
            "name": "csv_target_field",
            # Filename prefix for saved diagnostics and SVG plots.
            "file_prefix": "csv_target",
            # Save route diagnostics JSON/CSV.
            "save_route_diagnostics": True,
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
            "include": ("wire", "field_pointclouds", "gradient_maps", "error_maps", "field_cross_sections"),
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="csv_target_field")


def build_config(target_file: Path, parameters: dict | None = None) -> CoilGenConfig:
    parameters = get_parameters() if parameters is None else parameters
    config_values = dict(parameters["coilgen"])
    config_values["target_field_definition_file"] = str(target_file)
    return CoilGenConfig(**config_values)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    output = parameters["output"]
    output_dir = example_output_dir(output["name"])
    target_file = output_dir / parameters["csv_target"]["filename"]
    write_y_gradient_csv(target_file, parameters)
    project = CoilProject(
        output["name"],
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
    )
    project.channel(
        "main",
        design=CoilGenConfigDesign(name=output["file_prefix"], config=build_config(target_file, parameters)),
        save_gallery=bool(output["save_svg_gallery"]),
    )
    project_run = project.solve()
    result = project_run.channel_runs[0].result
    files = list(project_run.files)
    if output["save_route_diagnostics"]:
        files.extend(save_route_diagnostics(result, output_dir, prefix=output["file_prefix"]))
    print_saved_files(files, title="Saved CSV-target outputs:")
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
    print_result_summary(result, title="CSV target-field example")
    print(f"target CSV: {target_file}")
    return project, project_run, result, files, figures


def write_y_gradient_csv(path: Path, parameters: dict | None = None) -> None:
    parameters = get_parameters() if parameters is None else parameters
    csv_target = parameters["csv_target"]
    axis = np.linspace(-csv_target["axis_limit"], csv_target["axis_limit"], csv_target["axis_points"])
    lines = ["x_m,y_m,z_m,lin_x,lin_y,lin_z,radial_xy,weight,suppressed,group,dbz_dx,dbz_dy,dbz_dz"]
    for x in axis:
        for y in axis:
            for z in axis:
                if np.linalg.norm([x, y, z]) <= csv_target["sphere_radius"]:
                    radial = float(np.hypot(x, y))
                    lines.append(f"{x:.12g},{y:.12g},{z:.12g},{x:.12g},{y:.12g},{z:.12g},{radial:.12g},1,false,1,0,1,0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    project, project_run, result, files, figures = main()
    plot_book = open_project_plot_book_after_run(
        project_run, figures=figures, include=get_parameters()["plots"].get("include")
    )
