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

from coilgen import CoilGenConfig, CoilGenConfigDesign, CoilProject
from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh
from coilgen.target import evaluate_field_expression


from projects.general_examples.parameters import PREOPTIMIZED_STREAM_FUNCTION_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
################################################################################
# This example writes readable CSV inputs and then uses them as imported data.
PARAMETERS = object_parameters(
    {
        "preoptimized": {
            # CSV filename for imported stream-function values below outputs/<output.name>/.
            "stream_function_filename": "preoptimized_stream_function.csv",
            # CSV filename for imported target-field maps below outputs/<output.name>/.
            "target_map_filename": "custom_target_map.csv",
            # Expression evaluated on coil mesh vertices to emulate an externally optimized stream function.
            "stream_function_expression": "z",
            # Cartesian grid used for writing the target map CSV.
            "target_axis_limit": 0.04,
            "target_axis_points": 5,
            "target_sphere_radius": 0.04,
        },
        "coilgen": {
            # Target field comes from target_field_definition_file; field_shape_function is ignored here.
            "field_shape_function": "none",
            # Generated cylinder: height, radius, circular divisions, height divisions, normal sign, x/y/z rotation.
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_mesh_parameter_list": (0.25, 0.1, 16, 8, 1, 0, 0, 0),
            # Selects one scalar Bz map column in the CSV: constant_bz, lin_y, or quadratic_xy.
            "target_field_definition_field_name": "constant_bz",
            "target_gradient_strength": 1.0,
            # Maximum requested contour levels. Imported stream function may still be level-limited by min_turn_distance.
            "levels": 8,
            "pot_offset_factor": 0.25,
            "min_point_loop_number": 4,
            # Sensitivity is still calculated so the stream-function field and error maps remain meaningful.
            "gauss_order": 1,
            # Stream-function optimization strategy. Normally ignored here because
            # preoptimized_stream_function_file is set in build_config().
            # If you remove the imported stream-function file, choose exactly one active line:
            #   "tikhonov" uses the regularized inverse solve.
            #   "target_error_constrained_efficiency" finds the lowest-current solution
            #   that stays inside target_error_tolerance_percent.
            "sf_opt_method": "tikhonov",
            # "sf_opt_method": "target_error_constrained_efficiency",
            "stream_function_cost_function": "field_error_plus_resistance",
            "target_error_tolerance_percent": 5.0,
            "target_error_tolerance_absolute_T": 0.0,
            "target_error_constraint_factor_count": 61,
            "regularization_selection_method": "fixed",
            # The inverse solve is bypassed when preoptimized_stream_function_file is set.
            "preoptimized_stream_function_field_name": "stream_function",
            "preoptimized_stream_function_match_tolerance": 1e-8,
            # Skip swept solid-body generation, but keep electrical estimates active.
            "skip_sweep": True,
            "skip_inductance_calculation": False,
            "inductance_source": "auto",
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
            "specific_conductivity_conductor": 0.018e-6,
            "skip_calculation_min_winding_distance": False,
            "plot_flag": False,
            "save_stl_flag": False,
        },
        "output": {
            "name": "preoptimized_stream_function",
            "save_svg_gallery": False,
        },
        "plots": {
            "show_active": False,
            "close_previous": True,
            "mode": "book",
            "include": ("wire", "raw_contours", "stream_function", "field_pointclouds", "error_maps"),
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="preoptimized_stream_function")


def build_config(stream_function_file: Path, target_map_file: Path, parameters: dict | None = None) -> CoilGenConfig:
    parameters = get_parameters() if parameters is None else parameters
    config_values = dict(parameters["coilgen"])
    config_values["preoptimized_stream_function_file"] = str(stream_function_file)
    config_values["target_field_definition_file"] = str(target_map_file)
    return CoilGenConfig(**config_values)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    output_dir = example_output_dir(parameters["output"]["name"])
    stream_function_file = output_dir / parameters["preoptimized"]["stream_function_filename"]
    target_map_file = output_dir / parameters["preoptimized"]["target_map_filename"]
    write_preoptimized_stream_function_csv(stream_function_file, parameters)
    write_custom_target_map_csv(target_map_file, parameters)
    output = parameters["output"]
    project = CoilProject(
        output["name"],
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
    )
    project.channel(
        "main",
        design=CoilGenConfigDesign(
            name=output["name"],
            config=build_config(stream_function_file, target_map_file, parameters),
        ),
        save_gallery=bool(output["save_svg_gallery"]),
    )
    project_run = project.solve()
    result = project_run.channel_runs[0].result
    files = list(project_run.files)
    print_saved_files(files, title="Saved preoptimized-stream-function outputs:")
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
    print_result_summary(result, title="Preoptimized stream-function example")
    print(f"stream-function CSV: {stream_function_file}")
    print(f"target-map CSV: {target_map_file}")
    return project, project_run, result, files, figures


def write_preoptimized_stream_function_csv(path: Path, parameters: dict | None = None) -> None:
    parameters = get_parameters() if parameters is None else parameters
    config = CoilGenConfig(**dict(parameters["coilgen"]))
    mesh = load_mesh_from_config(config, PROJECT_ROOT)
    expression = parameters["preoptimized"]["stream_function_expression"]
    lines = ["part,vertex_index,x_m,y_m,z_m,stream_function"]
    for part_ind, mesh_part in enumerate(split_disconnected_mesh(mesh), start=1):
        stream_function = evaluate_field_expression(expression, mesh_part.vertices)
        for vertex_ind, (vertex, value) in enumerate(zip(mesh_part.vertices, stream_function)):
            lines.append(
                f"{part_ind},{vertex_ind},{vertex[0]:.12g},{vertex[1]:.12g},{vertex[2]:.12g},{float(value):.12g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_custom_target_map_csv(path: Path, parameters: dict | None = None) -> None:
    parameters = get_parameters() if parameters is None else parameters
    settings = parameters["preoptimized"]
    axis = np.linspace(-settings["target_axis_limit"], settings["target_axis_limit"], settings["target_axis_points"])
    lines = ["x_m,y_m,z_m,constant_bz,lin_y,quadratic_xy,weight,suppressed,group"]
    for x in axis:
        for y in axis:
            for z in axis:
                if np.linalg.norm([x, y, z]) <= settings["target_sphere_radius"]:
                    lines.append(f"{x:.12g},{y:.12g},{z:.12g},1,{y:.12g},{x * y:.12g},1,false,1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    project, project_run, result, files, figures = main()
    plot_book = open_project_plot_book_after_run(
        project_run, figures=figures, include=get_parameters()["plots"].get("include")
    )
