from __future__ import annotations

from projects._shared.object_parameters import object_parameters

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

from projects._shared.example_utils import (
    PROJECT_ROOT,
    example_output_dir,
    open_project_plot_book_after_run,
    parse_example_parameters,
)

from coilgen import CoilGenConfig, CoilGenConfigDesign, CoilProject
from projects.general_examples.scripts.run_standard_examples import validate_generated_example_result


from projects.general_examples.parameters import GENERATED_PARAMETER_SWEEP_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
################################################################################
# Edit this object in Spyder to define the parameter sweep.
PARAMETERS = object_parameters(
    {
        "run": {
            # False runs the compact study; True uses the larger full_sweep axes below.
            "full_sweep": False,
        },
        "sweep": {
            # Number of stream-function contour levels to test in the compact sweep.
            "levels_values": (6, 12),
            # Relative contour offsets to test in the compact sweep.
            "offset_values": (0.25, 0.5),
            # Cylinder mesh densities as (circular divisions, height divisions).
            "mesh_values": ((12, 6), (20, 10)),
            # L-curve scan centers to test; each run reports the selected factor.
            "regularization_values": (50.0,),
            # Interconnection cut widths to test.
            "cut_width_values": (0.02,),
        },
        "full_sweep": {
            # Larger sweep axes used when run.full_sweep is True.
            "levels_values": (6, 10, 14),
            "offset_values": (0.25, 0.5, 0.8),
            "mesh_values": ((12, 6), (18, 8), (24, 10)),
            "regularization_values": (20.0, 50.0, 100.0),
            "cut_width_values": (0.01, 0.02, 0.035),
        },
        "coilgen": {
            # Bz target field and generated cylinder geometry shared by all sweep rows.
            "field_shape_function": "y",
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_radius": 0.5,
            "cylinder_height": 0.25,
            "cylinder_normal_sign": 1,
            "cylinder_rotation": (0, 0, 0),
            # Spherical target cloud used for fitting the requested field.
            "target_region_radius": 0.08,
            "target_region_resolution": 5,
            # Shared contour cleanup and field-solve settings.
            "min_point_loop_number": 4,
            "gauss_order": 1,
            "within_group_cut_candidate_count": 8,
            # Stream-function optimization strategy for every sweep row.
            # Choose exactly one active line:
            #   "tikhonov" uses the L-curve/fixed regularization settings below.
            #   "target_error_constrained_efficiency" compares sweep rows at a fixed
            #   allowed target-field error and reports the lowest-current solution.
            "sf_opt_method": "tikhonov",
            # "sf_opt_method": "target_error_constrained_efficiency",
            "stream_function_cost_function": "field_error_plus_resistance",
            # Used only by sf_opt_method="target_error_constrained_efficiency".
            "target_error_tolerance_percent": 5.0,
            "target_error_tolerance_absolute_T": 0.0,
            "target_error_constraint_factor_count": 61,
            # L-curve regularization scan controls.
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
            # Keep minimal winding-distance diagnostics enabled for the sweep table.
            "skip_calculation_min_winding_distance": False,
            # Internal MATLAB-style plotting/STL flags remain disabled in the Python examples.
            "plot_flag": False,
            "save_stl_flag": False,
            # Skip swept solid-body generation, but keep electrical estimates active.
            "skip_sweep": True,
            "skip_inductance_calculation": False,
            "inductance_source": "auto",
            # Rectangular copper conductor for resistance and equivalent round-wire
            # radius in the Neumann/Biot-Savart inductance estimate.
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
            "specific_conductivity_conductor": 0.018e-6,
        },
        "output": {
            # Output folder below outputs/.
            "name": "generated_parameter_sweep",
            # Result table filenames.
            "csv_filename": "generated_parameter_sweep.csv",
            "json_filename": "generated_parameter_sweep.json",
        },
        "plots": {
            # Open active Matplotlib windows for the last sweep result.
            "show_active_for_last_result": False,
            # Close figures from previous runs before plotting the last result.
            "close_previous": True,
            # "book" opens one paged window; "windows" opens separate interactive figures.
            "mode": "book",
            # Plot groups to include if active plotting is enabled.
            "include": ("wire", "field_pointclouds", "error_maps"),
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="generated_parameter_sweep")


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    output = example_output_dir(parameters["output"]["name"])
    rows, last_project = run_sweep(
        full=parameters["run"]["full_sweep"], return_last_project=True, parameters=parameters
    )
    csv_path = output / parameters["output"]["csv_filename"]
    json_path = output / parameters["output"]["json_filename"]
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} sweep rows to {csv_path}")
    figures = []
    if last_project is not None:
        plots = parameters["plots"]
        figures = (
            last_project.plot(
                include=plots["include"],
                close_previous=plots["close_previous"],
                mode=plots.get("mode", "book"),
            )
            if plots["show_active_for_last_result"]
            else []
        )
    return rows, last_project, figures


def run_sweep(
    *,
    full: bool | None = None,
    return_last_result: bool = False,
    return_last_project: bool = False,
    parameters: dict | None = None,
):
    parameters = get_parameters() if parameters is None else parameters
    full = parameters["run"]["full_sweep"] if full is None else full
    rows: list[dict[str, Any]] = []
    active_sweep = parameters["full_sweep"] if full else parameters["sweep"]
    levels_values = active_sweep["levels_values"]
    offset_values = active_sweep["offset_values"]
    mesh_values = active_sweep["mesh_values"]
    reg_values = active_sweep["regularization_values"]
    cut_width_values = active_sweep["cut_width_values"]
    last_result = None
    last_project = None
    output_root = example_output_dir(parameters["output"]["name"])

    for levels, offset, mesh_density, reg, cut_width in product(
        levels_values,
        offset_values,
        mesh_values,
        reg_values,
        cut_width_values,
    ):
        name = f"cyl_y_L{levels}_a{offset:g}_m{mesh_density[0]}x{mesh_density[1]}_r{reg:g}_w{cut_width:g}"
        config = _sweep_config(levels, offset, mesh_density, reg, cut_width, parameters)
        project = CoilProject(
            name,
            output_dir=output_root / name,
            project_root=PROJECT_ROOT,
            save_reports=False,
        )
        project.channel("main", design=CoilGenConfigDesign(name=name, config=config))
        project_run = project.solve()
        result = project_run.channel_runs[0].result
        last_result = result
        last_project = project
        failures, warnings = validate_generated_example_result(name, result)
        part = result.coil_parts[0]
        diag = part.route_diagnostics
        row = {
            "name": name,
            "levels": levels,
            "pot_offset_factor": offset,
            "mesh_circular_divisions": mesh_density[0],
            "mesh_longitudinal_divisions": mesh_density[1],
            "tikonov_reg_factor": reg,
            "regularization_factor": result.regularization_factor,
            "regularization_lambda": result.regularization_lambda,
            "interconnection_cut_width": cut_width,
            "contour_count": len(part.contour_lines),
            "group_count": len(part.groups),
            "wire_length": diag.get("length", 0.0),
            "minimal_contour_distance": part.minimal_contour_distance,
            "route_quality_status": diag.get("route_quality_status", "unknown"),
            "route_quality_failure_count": diag.get("route_quality_failure_count", 0),
            "route_quality_warning_count": diag.get("route_quality_warning_count", 0),
            "max_rel_error_layout_vs_target": result.error_vals.get("max_rel_error_layout_vs_target", 0.0),
            "mean_rel_error_layout_vs_target": result.error_vals.get("mean_rel_error_layout_vs_target", 0.0),
            "max_rel_error_unconnected_contours_vs_target": result.error_vals.get(
                "max_rel_error_unconnected_contours_vs_target", 0.0
            ),
            "needed_current_layout": result.needed_current_layout,
            "failures": "; ".join(failures),
            "warnings": "; ".join(warnings),
        }
        rows.append(row)
        print(
            f"{name}: status={row['route_quality_status']} contours={row['contour_count']} groups={row['group_count']}"
        )
    if return_last_project:
        return rows, last_project
    if return_last_result:
        return rows, last_result
    return rows


def _sweep_config(
    levels: int,
    offset: float,
    mesh_density: tuple[int, int],
    regularization: float,
    cut_width: float,
    parameters: dict,
) -> CoilGenConfig:
    base = parameters["coilgen"]
    cylinder_rotation = base["cylinder_rotation"]
    return CoilGenConfig(
        field_shape_function=base["field_shape_function"],
        coil_mesh_file=base["coil_mesh_file"],
        cylinder_mesh_parameter_list=(
            base["cylinder_radius"],
            base["cylinder_height"],
            mesh_density[0],
            mesh_density[1],
            base["cylinder_normal_sign"],
            cylinder_rotation[0],
            cylinder_rotation[1],
            cylinder_rotation[2],
        ),
        target_region_radius=base["target_region_radius"],
        target_region_resolution=base["target_region_resolution"],
        levels=levels,
        pot_offset_factor=offset,
        min_point_loop_number=base["min_point_loop_number"],
        gauss_order=base["gauss_order"],
        sf_opt_method=base["sf_opt_method"],
        stream_function_cost_function=base["stream_function_cost_function"],
        tikonov_reg_factor=regularization,
        regularization_selection_method=base["regularization_selection_method"],
        l_curve_factor_count=base["l_curve_factor_count"],
        l_curve_factor_min_multiplier=base["l_curve_factor_min_multiplier"],
        l_curve_factor_max_multiplier=base["l_curve_factor_max_multiplier"],
        l_curve_regularization_factors=base["l_curve_regularization_factors"],
        interconnection_cut_width=cut_width,
        within_group_cut_candidate_count=base["within_group_cut_candidate_count"],
        parallel_worker_count=base["parallel_worker_count"],
        parallel_min_work_size=base["parallel_min_work_size"],
        sensitivity_chunks_per_worker=base["sensitivity_chunks_per_worker"],
        sensitivity_calculation_method=base["sensitivity_calculation_method"],
        biot_savart_chunk_size=base["biot_savart_chunk_size"],
        inductance_chunk_size=base["inductance_chunk_size"],
        wire_distance_method=base["wire_distance_method"],
        wire_distance_chunk_size=base["wire_distance_chunk_size"],
        wire_distance_spatial_min_segments=base["wire_distance_spatial_min_segments"],
        parallel_max_chunk_memory_mb=base["parallel_max_chunk_memory_mb"],
        optimization_solver=base["optimization_solver"],
        optimization_solver_residual_tolerance=base["optimization_solver_residual_tolerance"],
        target_error_tolerance_percent=base["target_error_tolerance_percent"],
        target_error_tolerance_absolute_T=base["target_error_tolerance_absolute_T"],
        target_error_constraint_factor_count=base["target_error_constraint_factor_count"],
        progress_flag=base["progress_flag"],
        progress_min_interval_seconds=base["progress_min_interval_seconds"],
        progress_percent_step=base["progress_percent_step"],
        skip_calculation_min_winding_distance=base["skip_calculation_min_winding_distance"],
        plot_flag=base["plot_flag"],
        save_stl_flag=base["save_stl_flag"],
        skip_sweep=base["skip_sweep"],
        skip_inductance_calculation=base["skip_inductance_calculation"],
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows, last_project, figures = main()
    plot_book = open_project_plot_book_after_run(
        getattr(last_project, "last_run", None),
        figures=figures,
        include=get_parameters()["plots"].get("include"),
    )
