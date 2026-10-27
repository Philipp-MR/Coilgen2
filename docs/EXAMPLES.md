# Examples

The [current inventory](../validation/examples/current_inventory.json) records every
guarded runnable script and its source hash. Regenerate it after example changes
with `python validation/examples/inventory.py`. Inventory does not imply validation;
the [validation record](../validation/examples/README.md) separates full, reduced,
failed and time-limited runs.

## Run from a checkout

Install the dependencies once with `python -m pip install -e ".[full]"`.
The runnable scripts locate this checkout from their own filename, so direct
execution also works from another working directory. Input data remains under
`project_data/`; general-project outputs use a timestamped run directory under
`outputs/general_examples/`.

For a compact, noninteractive starting point:

```powershell
python projects/general_examples/scripts/simple_ygradient.py --headless
python projects/general_examples/scripts/run_standard_examples.py --smoke --headless
python projects/general_examples/scripts/generated_parameter_sweep.py --smoke --headless
```

General examples honor `--headless` and an explicitly selected static
`MPLBACKEND`, such as `Agg`. Plotting choices in `PARAMETERS` are respected.
The compact y-gradient uses a 24-by-16 cylinder mesh and a seven-point-per-axis
ROI grid; refine these settings explicitly for a convergence study. Ordinary
general examples use one pipeline worker, and limit native BLAS pools when
`threadpoolctl` is installed. The parallel benchmark explicitly compares workers.

## Choose a starting script

Start with the smallest script that answers the engineering question.

| Question | Script |
|---|---|
| Conventional cylindrical gradient | `projects/general_examples/scripts/conventional_cylindrical_mri_gradient.py` |
| Shared-support Gx/Gy/Gz system | `projects/standard_cylindrical_3axis_gradient/run_standard_cylindrical_3axis_gradient.py` |
| Open-access biplanar feasibility | `projects/general_examples/scripts/open_biplanar_gradient.py` |
| Imported measured field or shim target | `projects/general_examples/scripts/imported_target_shim_example.py` |
| Continue from an existing stream function | `projects/general_examples/scripts/preoptimized_stream_function.py` |

Use the generated smoke suite above to expose route regressions. A failed case
does not stop the remaining cases, and the suite returns a nonzero exit status
if any execution or regression check fails. Its connected defaults currently
include geometries rejected by strict fabrication checks.
Connected multipart examples enable conductor sweeping because their single-circuit
assembly checks require the final path and body. The routed `balanced`, `validation`
and `active_shielded_gradient` recommended settings do the same. Sweeping can change
the realized centroid field and expose geometric rejection; it does not relax
the solver constraints or fabrication limits.
Use the scaling benchmarks
below to isolate target-point, mesh-density, active-shield or cache costs.
Project-specific studies have their own assumptions and metrics; their support
modules are not core CoilGen APIs.

## Physical optimization and fabrication

These seven examples are headless. Each accepts `--output-dir PATH`; its default
output location is relative to the checkout. The two studies also accept
`--no-resume` to force recomputation. Imports define functions without solving,
creating benchmark caches or changing process environment.

| Command | Purpose |
|---|---|
| `python examples/research_basic_designs.py` | Cylinder and biplanar designs with independent-grid field acceptance. |
| `python examples/research_tolerance_study.py` | Power/energy trade-offs with fixed physical normalization and one reusable conic graph. |
| `python examples/research_parallel_study.py` | Bounded independent design processes and verified restart. |
| `python examples/optimization_minimax_comparison.py` | Compare minimum power with peak surface-current density under the same field constraints. |
| `python examples/research_valid_fabrication.py` | Checked single-turn geometry, field, electrical and winding artifacts. |
| `python examples/research_valid_gradient_fabrication.py` | Strict passing Maxwell-pair geometry and fixed-current gradient reference on 913 independent ROI points. |
| `python examples/research_geometry_benchmark.py` | Separate cold compilation, warm kernel and complete-sweep measurements. |

The tolerance example assembles its operators directly, avoiding an unused
preliminary optimization and contour extraction. Minimax and study summaries
write strict JSON, with unavailable results represented by `null`.

`coilgen.research.convergence.check_sheet_integration(result, points_m,
relative_tolerance=..., absolute_tolerance_T=..., orders=(2, 4), subdivisions=(0, 1))`
compares quadrature and uniform triangle subdivisions on a common world/SI grid.
It holds the same piecewise-linear sheet current and faceted support fixed,
with explicit resource budgets and no current refit. Agreement establishes only
the tested integration scope. It does not establish convergence of a newly
optimized mesh, or qualify a finite-cross-section conductor.
Samples on the singular sheet are rejected.

The Maxwell reference uses two independently closed turns at their original
signed currents. It is a benchmark for fabrication and thin-wire gradient
checks; feeds, cooling, coupling and construction qualification need separate
assessment. It does not replace the connected cylindrical or biplanar presets.

## General Magnetostatic Examples

- `projects/general_examples/scripts/spyder_design_example.py`: full interactive plotting example with
  more expert controls exposed.
- `projects/general_examples/scripts/simple_ygradient.py`: classic generated-cylinder y-gradient example.
- `projects/general_examples/scripts/csv_target_field.py`: imports a target-field map from CSV.
- `projects/general_examples/scripts/audit_csv_target_field_symmetry.py`: checks whether an imported
  target-field map is mirror-paired and even/odd enough for symmetry reduction.
- `projects/general_examples/scripts/preoptimized_stream_function.py`: imports a preoptimized
  stream-function CSV and bypasses the inverse solve.
- `projects/general_examples/scripts/export_visualizations.py`: writes the SVG gallery.
- `projects/general_examples/scripts/export_pcb_layout.py`: cylindrical PCB artwork.
- `projects/general_examples/scripts/export_swept_body.py`: exports four independent
  constant-Bz turns as checked STL and STEP solids. It does not design feed breaks,
  terminals or interconnections; strict manufacturing checks remain enabled.
- `projects/general_examples/scripts/solenoid_constant_z_validation.py`: constant-Bz solenoid validation
  against analytical field and inductance formulas.
- `projects/general_examples/scripts/benchmark_performance.py`: small timing benchmark for CPU
  parallelism and symmetry reduction.
- `projects/general_examples/scripts/benchmark_target_point_scaling.py`: target point count scaling.
- `projects/general_examples/scripts/benchmark_mesh_vertex_scaling.py`: support mesh vertex scaling.
- `projects/general_examples/scripts/benchmark_active_shield_scaling.py`: generic active-shielded
  two-surface scaling.
- `projects/general_examples/scripts/benchmark_sweep_case_cache_scaling.py`: resumable sweep/cache scaling.
- `projects/primitive_support_structures/scripts/run_primitive_support_designs.py`: compact project
  for circular disc, annular disc, partial cylinder, elliptical cylinder, and
  surface-of-revolution support surfaces.

All of these examples are generated-data examples. They do not require MATLAB
`.mat` files.

The swept-body default previously routed a coarse gradient whose folded/colliding
sweep failed export. The independent-turn default provides a valid CAD example.
Edited designs still undergo the same checks, and manufacturing diagnostics are
written before export. The solenoid validation separately checks numerical field
and inductance against analytical formulas.

## Project-Specific Studies

Project folders under `projects/` may define additional scripts, defaults,
figures, and interpretation rules. Read the project README before using those
outputs in a report.

## Spyder Workflow

Open an example file directly in Spyder, edit the top-level `PARAMETERS`
dictionary, and run the file. General scripts are in `projects/general_examples/scripts/`;
primitive support examples are in `projects/primitive_support_structures/scripts/`. The scripts add the
local `src` directory to `sys.path` automatically. Active rotation and zoom
require an interactive matplotlib backend, preferably Qt in Spyder's IPython
graphics settings.

Only the short `PARAMETERS` block is intended for normal editing. Shared
advanced defaults are stored in
`projects/general_examples/parameters/defaults.py` for general examples, so the
same fallback values are not duplicated across scripts.

Most examples expose:

- `PARAMETERS["design"]` or `PARAMETERS["coilgen"]`: target field, mesh,
  contour, regularization, and CPU settings.
- `PARAMETERS["geometry"]`: SI-unit geometry parameters where applicable.
- `PARAMETERS["plots"]`: active matplotlib windows and previous-plot cleanup.
- `PARAMETERS["output"]`: output folder and saved-figure flags.

If a known parameter is removed, the parser fills the default and prints it. If
an unknown parameter is added, the parser prints that it was ignored.

## Interpreting Outputs

- `*_design_report.md`: human-readable summary, warnings, symmetry diagnostics,
  field errors, timings, and route/manufacturing status.
- `*_design_summary.csv`: compact machine-readable design summary.
- `*_failure_report.json` and `*_failure_geometry.npz`: failed stage/operation,
  partial route and conductor geometry, and recorded engineering diagnostics.
  These are unqualified failure evidence, not completed results or CAD releases.
- `*_wire_trace.csv` or `*_wire_path.csv`: routed XYZ centerline when `save_wire_trace_csv=True`.
- `*_contour_traces.step`: closed contour XYZ sequences as independent, unswept 1D curve paths.
- `*_wire_path.step`: the same points as an unswept, piecewise-linear STEP curve path.
- `*_field_error_diagnostics.csv`: scalar field/error metrics.
- `*_shielding_effect_metrics.csv`: active-shielded vs unshielded fringe-field
  comparison on the same suppression surface.
- SVG/PNG galleries: fixed-view and active-plot visual checks.

For final layout review, use examples with `postprocess_layout=True`. For large
parameter sweeps, `skip_postprocessing=True` is appropriate because it evaluates
the stream-function contour set without spending time on interconnections.

## Template for New Geometries

When a new generated geometry is added in `coilgen.geometry`, run:

```powershell
python projects/general_examples/scripts/create_new_geometry_api_template.py
```

This writes a user-facing `CoilDesign` subclass template below
`outputs/general_examples/new_geometry_api_template/`. Replace the placeholder geometry fields
with SI-unit parameters and map them to the low-level mesh parameter list used
by `load_mesh_from_config`.

## Example Structure

Each example uses a top-level `PARAMETERS` dictionary. Missing values are filled
from `projects/general_examples/parameters/defaults.py` and printed. Unknown
values are ignored and printed. This keeps Spyder editing simple while avoiding
silent parameter mistakes.
