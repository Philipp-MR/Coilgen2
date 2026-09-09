# General CoilGen Examples

This project contains general, generated-data CoilGen scripts that can be run
directly from Spyder or a terminal.

Direct execution works from any working directory. Add `--headless` to suppress
interactive plots; `MPLBACKEND=Agg` also requests a noninteractive run. Output
folders are timestamped below `outputs/general_examples/`.

## Structure

- `scripts/`: runnable example scripts.
- `parameters/defaults.py`: shared fallback parameters used when a script omits
  an advanced setting.
- `../../outputs/general_examples/`: figures, CSV summaries, reports, and
  exported geometry written by the scripts.

## Spyder Use

1. Open a script from `scripts/`.
2. Edit the `PARAMETERS` object near the top of the file.
3. Run the script. Missing advanced parameters are filled from
   `parameters/defaults.py` and printed during parsing.

Each active example creates a `CoilProject`, adds one or more channels with
`project.channel(...)`, solves with `project.solve()`, and opens requested plots
with `project.plot(...)`.

New general scripts should be added under `projects/general_examples/scripts`.

## First-Class Examples

- `conventional_cylindrical_mri_gradient.py`: cylindrical gradient starting point.
- `open_biplanar_gradient.py`: open biplanar gradient starting point.
- `imported_target_shim_example.py`: generated CSV target/shim import example.
- `export_swept_body.py`: swept volumetric conductor export as STL and STEP
  advanced BREP solids in meter coordinates. The default is four independent
  constant-Bz turns with valid mesh diagnostics; feed breaks, terminals and
  interconnections remain undesigned.

Each writes one output folder with a Markdown report and summary CSV.

`simple_ygradient.py` is a compact y-gradient starting point; increase its mesh
and target-grid resolution for convergence studies. `export_visualizations.py`
saves the SVG gallery by default. Shared fallback runtime/chunking parameters
remain in `parameters/defaults.py`.

For quick checks, use `run_standard_examples.py --smoke --headless` or
`generated_parameter_sweep.py --smoke --headless`. Omit `--smoke` for their full
compact example sets. Native thread limits prevent nested BLAS oversubscription
when `threadpoolctl` is available.

## Scaling Benchmarks

- `benchmark_target_point_scaling.py`
- `benchmark_mesh_vertex_scaling.py`
- `benchmark_active_shield_scaling.py`
- `benchmark_sweep_case_cache_scaling.py`
