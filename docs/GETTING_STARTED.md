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

## First run in a GUI

On Windows, run `Setup CoilGen2.cmd` once. Choose one interface:

- Open `Start CoilGen2 GUI.cmd` for the desktop. Choose **Browse examples… → Gradients → Cylindrical Y gradient (Gy)**, then click **Compute full design**. When the job finishes, select it in **Jobs**, click
  **View results**, then **Run design choices**. [Desktop walkthrough](NATIVE_WORKBENCH.md)
- Open `Start CoilGen2 Browser.cmd` for the browser. Choose **Browse examples…**,
  select **Gradients · Cylindrical Y gradient (Gy)** and click **Load template**.
  Click **Compute full design** and follow the job in **Design plan**. When it
  finishes, open **Run design choices** above the result plot. [First-run screen](images/browser-workbench-start.jpg) · [Browser walkthrough](BROWSER_WORKBENCH.md)

Both interfaces use the same engine. The cylindrical Gy example designs a
connected winding for a 20 mT/m gradient along the world y axis. Gradient and shim
templates request smoothing, but a completed computation can still fail field, routing
or fabrication checks. The result reader shows the inputs actually submitted,
even if you have since changed the editor.

## Run a design from Spyder

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
    target_scale=1.0,          # Requested T/m for x/y/z gradients
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

## Export a GUI design as Python

In the desktop or browser workbench, choose **File → Export Python script…**.
The desktop defaults to **Editable Python script**; in the browser, click
**Download Python script** after the export is ready. The active channel's
pending valid setup edits are applied before export. Resolve any unfinished
specialist editor changes first.

The resulting `.py` file contains editable `CONFIG = CoilGenConfig(...)` and
`STUDY` settings, plus embedded copies of imported meshes and target maps. It
requires no companion project file or open GUI. Design choices appear first;
resolved defaults are also included so an updated default cannot silently
change the design. Run it in the Python environment installed for CoilGen2:

```shell
python coilgen_design.py
python coilgen_design.py --output new_results --threads 1
```

On Windows, setup's default interpreter is
`%USERPROFILE%\.coilgen2\venv\Scripts\python.exe`. For a separate environment,
install the same CoilGen checkout with `python -m pip install -e ".[full]"`.
Imported formats and selected solvers need their corresponding dependencies.

The script runs the exported stage and study, including validation and
fabrication choices, through the same headless service used by the GUI. Results
default to `<script-name>.results` beside the script. After changing settings,
choose a new output directory to preserve the previous run. Importing the script
does not compute anything; notebooks and Spyder can call `build_project()` or
`run(output_dir="new_results")`. For numerical replay, retain the same CoilGen
source revision, dependencies and thread count.

Existing direct calls to `run_coilgen(config)`, `CoilGenPipeline(config).run()`
and design objects such as `CylinderCoilDesign(...).run()` remain supported.
They run the numerical pipeline; the exported service also preserves GUI study
controls and writes standard result artifacts. The standalone export captures
the current editable model, not the currently displayed result or uncommitted
winding/PCB editor drafts. Assemblies export the active channel; the desktop's
**Export Python and run files → Enabled channels Python script…** retains the
existing independent-channel replay with a companion project.

Programmatic equivalent:

```python
project.export_python("coilgen_design.py", standalone=True)
```

Or convert a saved single-channel project:

```shell
coilgen export-script design.coilgen coilgen_design.py --standalone
```

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
