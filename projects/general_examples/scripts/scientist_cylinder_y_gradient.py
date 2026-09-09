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
    example_output_dir,
    open_project_plot_book_after_run,
    parse_example_parameters,
    print_result_summary,
    print_saved_files,
)

from coilgen import CoilProject, CylinderCoilDesign


from projects.general_examples.parameters import SCIENTIST_CYLINDER_Y_GRADIENT_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
# Runtime/chunking defaults are shared in parameters/defaults.py.
################################################################################
# Legacy compact cylinder-gradient entry point. New scripts should prefer
# conventional_cylindrical_mri_gradient.py. All design values use SI units.
PARAMETERS = object_parameters(
    {
        "design": {
            "name": "scientist_cylinder_y_gradient",
            # Target Bz map. For "x", "y", or "z", target_scale is the gradient in T/(A m).
            # For "1", target_scale is a constant Bz field in T/A.
            "target_bz_expression": "y",
            "target_scale": 1.0,
            # Cylindrical coil surface geometry.
            "radius_m": 0.25,
            "length_m": 0.5,
            "angular_divisions": 32,
            "length_divisions": 28,
            # Spherical target region.
            "target_region_radius_m": 0.08,
            "target_region_resolution": 9,
            # Contour and conductor settings.
            "contour_levels": 10,
            "min_turn_spacing_m": 0.0,
            "conductor_width_m": 0.001,
            "conductor_height_m": 0.001,
            # Copper resistivity in ohm metre; used by final wire resistance.
            "conductor_resistivity_ohm_m": 0.018e-6,
            # Stream-function optimization scheme.
            # Choose exactly one active line:
            #   "tikhonov" keeps the standard CoilGen inverse solve and selects regularization below.
            #   "target_error_constrained_efficiency" minimizes current/resistance effort while enforcing
            #   target_error_tolerance_percent. This is preferred for geometry-efficiency comparisons.
            "optimization_method": "tikhonov",
            # "optimization_method": "target_error_constrained_efficiency",
            # Fixed-error efficiency optimizer settings. Used only for
            # optimization_method="target_error_constrained_efficiency".
            "target_error_tolerance_percent": 5.0,
            "target_error_tolerance_absolute_T": 0.0,
            "target_error_constraint_candidate_count": 61,
            # Regularization. Used by optimization_method="tikhonov"; keep l_curve for normal design work.
            "regularization_method": "l_curve",
            "regularization_factor": 50.0,
            # Runtime controls.
            "cpu_workers": 1,
            "progress": True,
            "skip_sweep": True,
            "skip_inductance": False,
        },
        "output": {
            "save_svg_gallery": False,
            "save_design_report": True,
        },
        "plots": {
            "show_active": True,
            "close_previous": True,
            "mode": "book",
            "include": ("wire", "contour_groups", "field_pointclouds", "error_maps", "regularization"),
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="scientist_cylinder_y_gradient")


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    design = CylinderCoilDesign(**parameters["design"])
    project = CoilProject(
        name=design.name,
        template=design,
        output_dir=example_output_dir(design.name),
        save_reports=bool(parameters["output"]["save_design_report"]),
        save_gallery=bool(parameters["output"]["save_svg_gallery"]),
        open_plot_book=False,
        plot_include=parameters["plots"].get("include"),
    )
    project.channel(
        "main",
        label=design.name,
        design_overrides={"name": design.name},
    )
    project_run = project.solve()
    channel_run = project_run.channel_runs[0]
    solved_design = channel_run.design
    artifacts = channel_run.artifacts
    if artifacts is None:
        raise RuntimeError("scientist cylinder y-gradient channel did not produce artifacts.")
    result = artifacts.result
    files = list(project_run.files)
    print_saved_files(files, title="Saved design outputs:")
    plots = parameters["plots"]
    figures = (
        project.plot(
            include=plots["include"],
            close_previous=bool(plots["close_previous"]),
            mode=plots.get("mode", "book"),
        )
        if interactive_plots_enabled(plots["show_active"])
        else []
    )
    print_result_summary(result, title="Cylinder y-gradient design")
    print("Spyder variables available after run: project, project_run, design, result, files, figures")
    return project, project_run, solved_design, result, files, figures


if __name__ == "__main__":
    with native_thread_budget():
        project, project_run, design, result, files, figures = main()
        plot_settings = get_parameters()["plots"]
        plot_book = open_project_plot_book_after_run(
            project_run, figures=figures, include=plot_settings.get("include"), enabled=plot_settings["show_active"]
        )
