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

from coilgen import BiplanarCoilDesign, CoilProject


################################################################################
# USER SETTINGS
################################################################################

PARAMETERS = object_parameters(
    {
        "design": {
            "name": "open_biplanar_gradient",
            "target_bz_expression": "y",
            "target_scale": 1.0,
            "width_m": 0.38,
            "height_m": 0.38,
            "lateral_divisions": 18,
            "longitudinal_divisions": 18,
            "plane_distance_m": 0.26,
            "target_region_radius_m": 0.07,
            "target_region_resolution": 7,
            "contour_levels": 8,
            "regularization_method": "l_curve",
            "regularization_factor": 50.0,
            "main_field_strength_T": 1.5,
            "b0_direction": (0.0, 0.0, 1.0),
            "progress": True,
            # The connected circuit spans both planes; validate its fabricated
            # centroid and swept body before claiming a completed design.
            "skip_sweep": False,
            "skip_inductance": False,
        },
        "output": {
            "name": "open_biplanar_gradient",
            "open_plot_book": False,
        },
    }
)


def main():
    design = BiplanarCoilDesign(**PARAMETERS.design.to_dict())
    project = CoilProject(
        name=str(PARAMETERS.output.name),
        template=design,
        output_dir=example_output_dir(PARAMETERS.output.name),
        open_plot_book=interactive_plots_enabled(PARAMETERS.output.open_plot_book),
        save_reports=True,
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
        raise RuntimeError("open biplanar gradient channel did not produce artifacts.")
    print_saved_files(project_run.files, title="Saved project outputs:")
    print_result_summary(artifacts.result, title="Open biplanar gradient")
    print("Spyder variables available after run: project, project_run, design, artifacts")
    return project, project_run, solved_design, artifacts


if __name__ == "__main__":
    with native_thread_budget():
        project, project_run, design, artifacts = main()
        plot_book = open_project_plot_book_after_run(
            project_run, figures=getattr(artifacts, "figures", ()), enabled=PARAMETERS.output.open_plot_book
        )
