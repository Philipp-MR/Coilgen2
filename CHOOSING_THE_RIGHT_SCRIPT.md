# Choosing The Right Script

Start with the smallest script that matches the question you are asking.

## First-Class General Examples

- `projects/general_examples/scripts/conventional_cylindrical_mri_gradient.py`
  Use this for a conventional cylindrical gradient-coil starting point.

- `projects/standard_cylindrical_3axis_gradient/run_standard_cylindrical_3axis_gradient.py`
  Use this for a clean unshielded cylindrical `Gx/Gy/Gz` gradient set with
  one shared support geometry and plot-book inspection.

- `projects/general_examples/scripts/open_biplanar_gradient.py`
  Use this for open-access biplanar gradient feasibility.

- `projects/general_examples/scripts/imported_target_shim_example.py`
  Use this when the target field comes from a CSV file or shim-like imported
  field map.

Each script has one user-editable `PARAMETERS` block, one output folder, a
Markdown design report, and a summary CSV.

## Regression And Demonstration Scripts

- `projects/general_examples/scripts/run_standard_examples.py`
  Runs a compact generated example suite and checks basic routing expectations.

- `projects/general_examples/scripts/simple_ygradient.py`
  Runs a low-level expert-parameter y-gradient example.

- `projects/general_examples/scripts/preoptimized_stream_function.py`
  Starts from an imported stream function and continues through contour and
  layout evaluation.

## Scaling Benchmarks

- `benchmark_target_point_scaling.py`
  Varies target point count and records sensitivity-matrix size.

- `benchmark_mesh_vertex_scaling.py`
  Varies support mesh density and records runtime/memory scaling.

- `benchmark_active_shield_scaling.py`
  Varies generic two-surface active-shielded supports.

- `benchmark_sweep_case_cache_scaling.py`
  Exercises resumable parameter-sweep case caches.

## Project-Specific Scripts

Project folders under `projects/` may contain their own assumptions, metrics,
paper figures, and parameter names. Treat those scripts as project-specific
workflows, not as core CoilGen APIs.
