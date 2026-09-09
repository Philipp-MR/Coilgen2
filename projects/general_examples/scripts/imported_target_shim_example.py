from __future__ import annotations

# Support direct terminal/Spyder execution from any working directory.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    for _path in (_Path(__file__).resolve().parents[3], _Path(__file__).resolve().parents[3] / "src"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

import csv
from pathlib import Path

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
        "target": {
            "filename": "shim_target.csv",
            "axis_limit_m": 0.045,
            "axis_points": 5,
            "sphere_radius_m": 0.055,
            "field_name": "shim_bz",
            "amplitude_T": 1.0e-4,
        },
        "design": {
            "name": "imported_target_shim_example",
            "target_scale": 1.0,
            "radius_m": 0.22,
            "length_m": 0.35,
            "angular_divisions": 24,
            "length_divisions": 16,
            "target_region_radius_m": 0.055,
            "target_region_resolution": 5,
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
            "name": "imported_target_shim_example",
            "open_plot_book": False,
        },
    }
)


def write_target_csv(path: Path) -> Path:
    target = PARAMETERS.target
    path.parent.mkdir(parents=True, exist_ok=True)
    axis_limit = float(target.axis_limit_m)
    axis_points = int(target.axis_points)
    sphere_radius = float(target.sphere_radius_m)
    amplitude = float(target.amplitude_T)
    values = [(-axis_limit + 2.0 * axis_limit * i / max(axis_points - 1, 1)) for i in range(axis_points)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x_m", "y_m", "z_m", target.field_name, "weight"))
        for x in values:
            for y in values:
                for z in values:
                    if x * x + y * y + z * z > sphere_radius * sphere_radius:
                        continue
                    normalized = (z * z - 0.5 * (x * x + y * y)) / max(axis_limit * axis_limit, 1.0e-30)
                    writer.writerow((x, y, z, amplitude * normalized, 1.0))
    return path


def main():
    output_dir = example_output_dir(PARAMETERS.output.name)
    target_file = write_target_csv(output_dir / "inputs" / str(PARAMETERS.target.filename))
    design_values = PARAMETERS.design.to_dict()
    design_values.update(
        {
            "target_bz_expression": "none",
            "target_field_file": str(target_file),
            "target_field_name": str(PARAMETERS.target.field_name),
        }
    )
    design = CylinderCoilDesign(**design_values)
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
        raise RuntimeError("imported target shim channel did not produce artifacts.")
    print_saved_files((target_file, *project_run.files), title="Saved target and project outputs:")
    print_result_summary(artifacts.result, title="Imported target shim example")
    print("Spyder variables available after run: project, project_run, target_file, design, artifacts")
    return project, project_run, target_file, solved_design, artifacts


if __name__ == "__main__":
    with native_thread_budget():
        project, project_run, target_file, design, artifacts = main()
        plot_book = open_project_plot_book_after_run(
            project_run, figures=getattr(artifacts, "figures", ()), enabled=PARAMETERS.output.open_plot_book
        )
