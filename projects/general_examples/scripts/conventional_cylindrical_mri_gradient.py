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
    print_result_summary,
    print_saved_files,
)

from coilgen import CoilProject, CylinderCoilDesign


################################################################################
# USER SETTINGS
################################################################################

PARAMETERS = object_parameters(
    {
        "design": {
            "name": "conventional_cylindrical_mri_gradient",
            "target_bz_expression": "y",
            "target_scale": 1.0,
            "radius_m": 0.26,
            "length_m": 0.55,
            "angular_divisions": 28,
            "length_divisions": 20,
            "target_region_radius_m": 0.08,
            "target_region_resolution": 7,
            "contour_levels": 8,
            "regularization_method": "l_curve",
            "regularization_factor": 50.0,
            "main_field_strength_T": 1.5,
            "b0_direction": (0.0, 0.0, 1.0),
            "progress": True,
            "skip_sweep": True,
            "skip_inductance": False,
        },
        "output": {
            "name": "conventional_cylindrical_mri_gradient",
            "open_plot_book": False,
        },
    }
)


def main():
    design = PARAMETERS.design
    template = CylinderCoilDesign(
        name=str(design.name),
        main_field_strength_T=float(design.main_field_strength_T),
        b0_direction=tuple(float(value) for value in design.b0_direction),
    )

    project = CoilProject(
        name=str(PARAMETERS.output.name),
        template=template,
        output_dir=example_output_dir(PARAMETERS.output.name),
        open_plot_book=interactive_plots_enabled(PARAMETERS.output.open_plot_book),
        save_reports=True,
    )
    project.parameters.mesh.radius_m = float(PARAMETERS.design.radius_m)
    project.parameters.mesh.length_m = float(PARAMETERS.design.length_m)
    project.parameters.mesh.angular_divisions = int(PARAMETERS.design.angular_divisions)
    project.parameters.mesh.length_divisions = int(PARAMETERS.design.length_divisions)
    project.parameters.target.expression = str(PARAMETERS.design.target_bz_expression)
    project.parameters.target.scale = float(PARAMETERS.design.target_scale)
    project.parameters.target.region_radius_m = float(PARAMETERS.design.target_region_radius_m)
    project.parameters.target.region_resolution = int(PARAMETERS.design.target_region_resolution)
    project.parameters.optimization.contour_levels = int(PARAMETERS.design.contour_levels)
    project.parameters.optimization.regularization_method = str(PARAMETERS.design.regularization_method)
    project.parameters.optimization.regularization_factor = float(PARAMETERS.design.regularization_factor)
    project.parameters.optimization.skip_sweep = bool(PARAMETERS.design.skip_sweep)
    project.parameters.optimization.skip_inductance = bool(PARAMETERS.design.skip_inductance)
    project.parameters.numerics.progress = bool(PARAMETERS.design.progress)
    project.channel(
        "main",
        label=template.name,
        design_overrides={"name": template.name},
    )
    project_run = project.solve()
    channel_run = project_run.channel_runs[0]
    solved_design = channel_run.design
    artifacts = channel_run.artifacts
    if artifacts is None:
        raise RuntimeError("conventional cylindrical gradient channel did not produce artifacts.")
    print_saved_files(project_run.files, title="Saved project outputs:")
    print_result_summary(artifacts.result, title="Conventional cylindrical gradient")
    print("Spyder variables available after run: project, project_run, design, artifacts")
    return project, project_run, solved_design, artifacts


if __name__ == "__main__":
    with native_thread_budget():
        project, project_run, design, artifacts = main()
        plot_book = open_project_plot_book_after_run(
            project_run, figures=getattr(artifacts, "figures", ()), enabled=PARAMETERS.output.open_plot_book
        )
