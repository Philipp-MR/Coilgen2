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

from coilgen import (
    CoilGenConfig,
    CoilGenConfigDesign,
    CoilProject,
    save_manufacturing_diagnostics,
    save_swept_bodies_as_step,
    save_swept_bodies_as_stl,
)


from projects.general_examples.parameters import EXPORT_SWEPT_BODY_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
################################################################################
# This example creates swept 3D conductor bodies and exports STL/STEP files.
PARAMETERS = object_parameters(
    {
        "coilgen": {
            # Four independent constant-Bz turns are a compact, valid CAD example.
            # Feed breaks, terminals and connections between turns are not designed.
            "field_shape_function": "1",
            "target_gradient_strength": 1e-4,
            # Length [m], radius [m], angular/axial divisions, rotation axis, angle.
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_mesh_parameter_list": (0.2, 0.08, 32, 12, 1, 0, 0, 0),
            "target_region_radius": 0.02,
            "target_region_resolution": 5,
            "levels": 4,
            "regularization_selection_method": "fixed",
            "tikonov_reg_factor": 50.0,
            "skip_postprocessing": True,
            "contour_only_sweep_mode": True,
            "skip_sweep": False,
            "skip_inductance_calculation": False,
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
            "specific_conductivity_conductor": 0.018e-6,
            # Subdivision keeps the swept triangles well proportioned.
            "sweep_max_segment_length": 0.004,
            "parallel_worker_count": 1,
            "progress_flag": True,
        },
        "output": {
            "name": "swept_body",
            "file_prefix": "independent_solenoid_turns",
            "save_stl_files": True,
            "save_step_files": True,
            "step_surface_mode": "advanced_brep",
            "save_diagnostics": True,
        },
        "plots": {
            "show_active": False,
            "close_previous": True,
            "mode": "book",
            "include": ("wire", "swept_body", "field_pointclouds"),
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="export_swept_body")


def build_config(parameters: dict | None = None) -> CoilGenConfig:
    parameters = get_parameters() if parameters is None else parameters
    return CoilGenConfig(**parameters["coilgen"])


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    output = parameters["output"]
    output_dir = example_output_dir(output["name"])
    project = CoilProject(output["name"], output_dir=output_dir, project_root=PROJECT_ROOT)
    project.channel(
        "main",
        design=CoilGenConfigDesign(name=output["file_prefix"], config=build_config(parameters)),
    )
    project_run = project.solve()
    result = project_run.channel_runs[0].result
    files = list(project_run.files)
    # Retain diagnostics even when an edited design fails the strict export gate.
    if output["save_diagnostics"]:
        files.extend(save_manufacturing_diagnostics(result, output_dir, prefix=output["file_prefix"]))
    if output["save_stl_files"]:
        files.extend(save_swept_bodies_as_stl(result, output_dir, prefix=output["file_prefix"]))
    if output["save_step_files"]:
        files.extend(
            save_swept_bodies_as_step(
                result,
                output_dir,
                prefix=output["file_prefix"],
                surface_mode=output["step_surface_mode"],
            )
        )
    if files:
        print_saved_files(files, title="Saved swept conductor files:")
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
    print_result_summary(result, title="Independent-turn swept-body export")
    print("Exported independent closed turns; feed breaks, terminals and interconnections remain undesigned.")
    return project, project_run, result, files, figures


if __name__ == "__main__":
    with native_thread_budget():
        project, project_run, result, files, figures = main()
        plot_settings = get_parameters()["plots"]
        plot_book = open_project_plot_book_after_run(
            project_run, figures=figures, include=plot_settings.get("include"), enabled=plot_settings["show_active"]
        )
