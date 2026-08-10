# Getting Started

This Python CoilGen fork is intended for physicists, MR scientists, and coil
engineers who want to describe a magnetostatic coil design in SI units, run the
design, inspect plots, and export diagnostics without editing the numerical
core.

CoilGen is a general magnetostatic coil-design library. The general workflow is:

1. choose or import a current-carrying support surface;
2. define the useful magnetic-field target and optional suppression targets;
3. solve for a stream-function surface current;
4. convert the stream function into contours or routed conductor paths;
5. inspect field error, efficiency, current density, resistance, inductance,
   concomitant-field estimates, route quality, and manufacturing diagnostics.

Use it for early magnetostatic design and screening. Treat thermal, voltage,
PNS, acoustic, eddy-current, mechanical, and regulatory checks as downstream
engineering work unless a project script explicitly adds those checks.

## Recommended Spyder Workflow

1. Open `projects/general_examples/scripts/conventional_cylindrical_mri_gradient.py`.
2. Edit only the `PARAMETERS["design"]` values.
3. Run the file.
4. Inspect the active matplotlib figures.
5. Read the generated `*_design_report.md` in the output folder.

The example scripts intentionally keep only one editable parameter dictionary.
Advanced fallback values live in
`projects/general_examples/parameters/defaults.py`. If a known setting is
omitted from the script, the parser fills it from that defaults file and prints
the value it used.

The compact user-facing API is:

```python
from coilgen import CylinderCoilDesign

design = CylinderCoilDesign(
    target_bz_expression="y",
    target_scale=1.0,          # T/(A m) for x/y/z gradients
    radius_m=0.25,
    length_m=0.5,
    contour_levels=8,
    conductor_width_m=0.001,
    conductor_height_m=0.001,
)
result = design.run()
```

For scripts that should also save a standard Markdown report and summary CSV,
use `run_coil_design`:

```python
from coilgen import CoilDesignRunOptions, CylinderCoilDesign, run_coil_design

design = CylinderCoilDesign(target_bz_expression="y")
artifacts = run_coil_design(design, CoilDesignRunOptions(output_prefix="my_design"))
```

All geometry and conductor dimensions use meters. CoilGen still accepts the
expert `CoilGenConfig` API for low-level studies.

## Result Interpretation

Each standard example now writes:

- `*_design_report.md`: readable design summary, warnings, errors, timings.
- `*_design_summary.csv`: compact machine-readable summary.
- `*_wire_trace.csv` or `*_wire_path.csv`: routed XYZ centerline when `save_wire_trace_csv=True`.
- `*_contour_traces.step`: closed contour XYZ sequences as independent, unswept 1D curve paths.
- `*_wire_path.step`: the same points as an unswept, piecewise-linear STEP curve path.
- SVG/CSV diagnostics when enabled.

`needed_current_layout` is the signed current required to realize the requested
Bz target mode in the internal target region. For gradient targets such as
`Bz = G*y`, efficiency is the fitted `y`-mode amplitude per ampere. Field
components that are not part of the requested mode are not counted as useful
efficiency; they appear in the error metrics.

The report status is:

- `PASS`: no route/manufacturing failures and no configured quality warning.
- `WARNING`: result exists, but field error, route diagnostics, or spacing should
  be inspected.
- `FAIL`: route or manufacturing diagnostics indicate the layout is not ready.
