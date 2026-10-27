# CoilGen2

Created and maintained by **Philipp Amrein — MRI hardware engineer**.
[Project source](https://github.com/Philipp-MR/Coilgen2).
[Kontakt and project information](CONTACT.md).

This is the geodesic-topology fork of the class-based Python CoilGen
translation. CoilGen is intended to become a powerful, easy-to-use
magnetostatic coil-design tool: users describe a support surface, a target
magnetic field, conductor assumptions, and optional suppression regions; CoilGen
solves for a stream-function current distribution, extracts conductor contours,
and reports field, electrical, layout, and manufacturing diagnostics.

The library itself is application-neutral. It should support conventional MRI
gradients, portable/Halbach-style systems, shim coils, local nonlinear
gradients, active shielding, arbitrary imported target fields, and general
magnetostatic field-shaping studies.

The core layout pipeline does not build a global surface flattening; contour
topology is computed intrinsically on the triangle mesh.

The current implementation supports the first complete numerical path:

- generated cylinder, planar, and biplanar meshes;
- generated circular, shielded biplanar, split-bore suppression, and flanged split-bore meshes;
- binary/ASCII STL loading;
- disconnected mesh splitting;
- midpoint mesh refinement;
- target-field generation from a `Bz(x, y, z)` expression;
- imported target fields from `.npz`, `.json`, `.csv`, or `.tsv` files;
- one-ring stream-function basis construction;
- Biot-Savart sensitivity matrix construction;
- resistance regularization matrix construction;
- Tikhonov stream-function optimization and optional fixed-error constrained
  optimization via CVXPY;
- imported preoptimized stream-function values from `.csv`, `.tsv`, `.json`, or `.npz`;
- contour-level extraction with surface provenance;
- contour current-orientation correction from mesh normals and the stream-function-induced surface current;
- intrinsic loop-containment grouping using virtual cuts on the surface mesh;
- simple wire-path assembly;
- group-level layout interconnection with 3D sphere openings;
- return-path and 3D wire self-contact normal shifting;
- native Biot-Savart/Neumann layout inductance and DC resistance estimates;
- concomitant-field reports for MRI-style gradient screening from the fitted
  layout gradient and configured B0 assumptions;
- dependency-free SVG visualizations for wire paths, contour topology, stream function, current density, target-field maps, shimming residual mosaics, routing debug views, and field comparisons;
- swept 3D conductor body generation and ASCII STL export;
- swept-body manufacturing diagnostics and optional STL clearance gates;
- cylindrical PCB unrolling and SVG PCB artwork export;
- field/error evaluation.

The layout compiler now includes intrinsic grouping, within/across-group
interconnection, and normal shifting, but it is still not exact MATLAB
post-processing parity. Deeper 3D layout collision handling remains future
work. Inductance is estimated natively with a regularized thin-wire
Neumann/Biot-Savart integral rather than any FastHenry interface.

The recommended regression and inspection path uses generated meshes and
generated target fields only. No MATLAB `.mat` data is required for these
examples.

## Attribution and citation

CoilGen2 builds on the original CoilGen work by Philipp Amrein, Feng Jia,
Maxim Zaitsev and Sebastian Littin. Please cite the original publication when
using CoilGen and record the CoilGen2 version or commit used for reproducibility:

Amrein P, Jia F, Zaitsev M, Littin S. **CoilGen: Open-source MR coil layout
generator.** *Magnetic Resonance in Medicine.* 2022;88(3):1465–1479.
[doi:10.1002/mrm.29294](https://doi.org/10.1002/mrm.29294).

[CITATION.cff](../CITATION.cff) provides machine-readable software and publication
credits. The original paper describes CoilGen; the current CoilGen2 documentation
describes this implementation and its additional capabilities. Software credits
in reports and exports identify the tool's creator and maintainer; they do not
assign authorship of a user's coil design.

## User Documentation

The main guides are collected in `docs/`. Run the commands in these guides from
the repository root, where `pyproject.toml` and `Start CoilGen2 GUI.cmd` live.

- [Native scientific workbench](NATIVE_WORKBENCH.md): Qt/VTK desktop workflow,
  portable projects, staged execution, channel projects and script compatibility.
- [Installation](INSTALL.md): Python environment setup and dependency options.
- [Getting Started](GETTING_STARTED.md): recommended Spyder workflow and compact
  user-facing API.
- [Core Concepts](CORE_CONCEPTS.md): support surfaces, target fields, stream
  functions, contours, routed wire paths, and field sources.
- [Result Interpretation](RESULT_INTERPRETATION.md): efficiency, current, DSV
  error, active shielding, concomitant fields, and readiness caveats.
- [Examples](EXAMPLES.md): choose a starting script, benchmark or project study.
- [Parameters](PARAMETERS.md): SI-unit user parameters and validation.
- [Stream-function optimization](STREAM_FUNCTION_OPTIMIZATION.md): physical objectives,
  solver capabilities, minimax current density, and mathematical limitations.
- [Headless physical design](RESEARCH_PHYSICS_API.md): power/energy/current-density
  objectives, force/torque limits, reproducible studies and compiled fabrication checks.
- [Main integration and validation](MAIN_CORE_IMPROVEMENTS.md): changes, examples,
  compatibility decisions and remaining engineering limitations.
- [File Formats](FILE_FORMATS.md): CSV/TSV/JSON/NPZ target maps and
  preoptimized stream functions.
- [Diagnostics](DIAGNOSTICS.md): design reports, route diagnostics, and field
  error interpretation.
- [Physical Assumptions](PHYSICAL_ASSUMPTIONS.md): stream-function units,
  active-shielding assumptions, symmetry-reduction prerequisites, and field
  evaluation assumptions.
- [Quality and Extension API](QUALITY_AND_EXTENSION_API.md): regularization
  diagnostics, typed study results, user run helper, geometry factories,
  and plot-book extension points.
- [Object Model](OBJECT_MODEL.md): `CoilProject`, channels, shared project
  parameters, channel overrides, solving, and plotting.
- [Primitive Geometry Reference](PRIMITIVE_GEOMETRIES.md): generated support
  surfaces and their user-facing design classes.
- [Plot Book Extension Guide](PLOT_BOOK.md): developer notes for adding new
  active Matplotlib plot-book pages and backends.
- [Literature-Backed Design Insights](DESIGN_INSIGHTS.md): general CoilGen
  roadmap lessons from gradient-coil, open-source stream-function, and PNS
  literature.
- [Engineering Roadmap](IMPROVEMENT_ROADMAP.md): remaining physical-model,
  fabrication, validation and downstream-analysis work.
- [Coil Design Review](COIL_DESIGN_REVIEW.md): reproduced numerical/export
  errors, implemented corrections, and priorities for conductor sweeps,
  assembly clearance, terminals, and practical winding instructions.
- [Contributing](CONTRIBUTING.md): numerical-change rules and the automated
  quality checks used by the project.
- [Project Data Layout](PROJECT_DATA.md): local storage for meshes, measured
  field maps, and other large inputs excluded from version control.

## Installation

See [Installation](INSTALL.md) for the full setup guide. From a fresh clone,
create an environment and install the package plus the common
interactive/scientific extras:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` delegates to `pyproject.toml` and installs CoilGen in
editable mode with plotting, CVXPY, SciPy, and image-preview support.
For the dependency-light generated-example core, use:

```powershell
python -m pip install -e .
```

For local development and tests:

```powershell
python -m pip install -e ".[dev]"
python -m ruff check src tests projects
python -m ruff format --check src tests projects
python -m pytest -q
```

## Quick Run

For the native model editor, install `python -m pip install -e ".[gui,plot]"`
and run `python -m coilgen.gui`, or use `Start CoilGen2 GUI.cmd` on Windows.
Existing script workflows remain available independently of GUI dependencies.
Startup opens an empty workspace. Use **File → New project…** for a blank design,
**Open project…** for a saved JSON, or the upper-left **File → Load template** menu to
choose cylindrical or biplanar Gx/Gy/Gz, an actively shielded Gy, uniform or
second-order shims, or the complete biplanar gradient and shim system. Templates
create editable inputs for the full study and save as a self-contained JSON.

Use the project virtual environment after following the installation steps:

```powershell
.\.venv\Scripts\python.exe projects\general_examples\scripts\conventional_cylindrical_mri_gradient.py
```

The compact API used by that example is:

```python
from coilgen import CylinderCoilDesign

design = CylinderCoilDesign(target_bz_expression="y", radius_m=0.25, length_m=0.5)
result = design.run()
```

## Optional Solver Extra

The `constrained_quadratic` stream-function optimizer directly enforces the
weighted target-field residual as a solver constraint. It uses CVXPY when
available and falls back to the dependency-free Tikhonov multiplier scan when
CVXPY is not installed.

```powershell
pip install -e .[solver]
```

For the older expert-parameter example:

```powershell
.\.venv\Scripts\python.exe projects\general_examples\scripts\simple_ygradient.py
```

The example and project scripts locate the checkout when executed directly;
manual `PYTHONPATH` configuration is unnecessary. Install the required dependencies
in the Python environment used to run them. General scripts accept `--headless`
for terminal use without interactive plots. In Spyder, use that environment,
open a `projects/general_examples/scripts/*.py` file or the primitive geometry
project under `projects/primitive_support_structures/scripts/*.py`, edit the
`PARAMETERS` dictionary or object-style `PARAMETERS...` assignments in the
`USER SETTINGS` block at the top, and run the script directly. Each example
groups user-editable values into dictionary sections such as `coilgen`,
`output`, `plots`, and, where needed, `sweep` or `csv_target`. If a parameter
entry is removed, the script fills it from the central fallback dictionaries and
prints the default value that was used. General defaults live in
`projects/general_examples/parameters/defaults.py`.
Normal users should edit only the script-level `PARAMETERS` block; the defaults
files are for shared advanced settings and template maintenance. Project-specific
folders under `projects/` may add their own defaults and reports beside their
own scripts. For a clean unshielded cylindrical `Gx/Gy/Gz` starter project, use
`projects/standard_cylindrical_3axis_gradient/run_standard_cylindrical_3axis_gradient.py`.

## Scientific Performance Metrics

CoilGen is magnetostatic. A `PASS` or efficient design means the static field,
layout, and configured manufacturing diagnostics look acceptable inside this
model. It does not by itself prove thermal safety, gradient-amplifier margins,
PNS safety, acoustic behavior, Lorentz-force robustness, eddy-current behavior,
cryostat heating, or clinical/device readiness.

Concomitant-field values in the standard reports are screening estimates from
the fitted connected-layout gradient, configured `b_0_strength_T`, and
`b_0_direction`. Application-specific metrics are intentionally not part of the
general CoilGen core.

The central performance current is `result.needed_current_layout`. It is now
computed by fitting the requested Bz target mode in the internal target region,
not by a whole-field reciprocal projection. This matters for gradient coils:
orthogonal Bz content, uniform offsets, and shielding-region samples must not
make the useful gradient efficiency appear larger.

For a y-gradient target `Bz = G*y`, CoilGen fits the coefficient of the `y` mode
in the layout field per ampere. The reported gradient efficiency is the absolute
mode coefficient in `mT m^-1 A^-1`; the signed current tells the required current
polarity. Orthogonal field content is counted in the field-error metrics.

Active-shielded studies should report four distinct quantities:

- internal gradient efficiency: useful requested gradient mode per ampere;
- internal field error or linearity: residual Bz error after target-mode scaling;
- fringe-field suppression: shielded vs unshielded field on the external
  suppression surface;
- field-evaluation source: connected final wire path for final designs, or
  unconnected contours for fast grid sweeps with `skip_postprocessing=True`.

Strong shielding claims require an explicit suppression metric, a fair
quality-matched reference, and a connected-layout review. Too little
regularization can make a shield stream function oscillatory and fragment into
many small contour groups, while too much regularization can leave too few
shield turns.

For geometry-efficiency studies where the field error should be held fixed, set
`sf_opt_method = "target_error_constrained_efficiency"` and choose
`target_error_tolerance_percent`. This mode minimizes the resistance/current
norm subject to the specified weighted target-field residual budget, so support
mesh geometry can be compared at a common target deviation instead of at a
different L-curve compromise for each geometry.

## Minimum Wire-Turn Spacing

Set `min_turn_distance` in meters to request a minimum geodesic spacing between
neighboring contour turns. CoilGen estimates the local contour spacing as
`contour_step / |K|`, where `K` is the stream-function-induced surface current
density. If the requested number of levels would make the turns too dense,
contour-level selection reduces the effective number of levels before contour
extraction.

```python
"levels": 30,                 # requested maximum number of contour levels
"min_turn_distance": 0.003,   # 3 mm minimum turn spacing
```

The result reports the actual effective level count in `result.num_levels`.
When `skip_calculation_min_winding_distance = False`, the diagnostics also
include `measured_min_contour_distance` and
`turn_spacing_measured_threshold_met`.

An optional soft optimization term is also available:

```python
"turn_spacing_regularization_factor": 0.0,
```

Keep it at `0.0` for the default hard level-count threshold. Increase it only
when you explicitly want extra stream-function smoothing; this can change the
field fit and should be evaluated with the field/error plots.

## Parallel CPU Usage

The numerical kernels use portable thread-based parallelism. This works from a
normal Python shell and from Spyder on Windows because it does not require
`multiprocessing` process spawning or pickling large mesh arrays. The main
parallelized stages are sensitivity-matrix assembly, Biot-Savart field
evaluation for contours and connected wire paths, loop-significance filtering,
and native Neumann inductance evaluation.

Each Spyder example exposes the relevant settings in
`PARAMETERS["coilgen"]`:

```python
"parallel_worker_count": 0,        # 0 = all logical CPU cores, 1 = serial, -1 = keep one core free
"parallel_min_work_size": 50000,   # below this estimated size, run serial to avoid overhead
"sensitivity_chunks_per_worker": 4,
"sensitivity_calculation_method": "vectorized",  # "vectorized", "scalar", or "auto"
"biot_savart_chunk_size": 512,
"inductance_chunk_size": 256,
"wire_distance_method": "auto",  # "auto", "spatial", or "all_pairs"
"wire_distance_chunk_size": 512,
"wire_distance_spatial_min_segments": 3000,
"parallel_max_chunk_memory_mb": 64.0,
"optimization_solver": "auto",     # "auto", "solve", or "pinv"
"optimization_solver_residual_tolerance": 1e-8,
"progress_flag": True,
"progress_min_interval_seconds": 1.0,
"progress_percent_step": 5.0,
```

For dense designs, leave `parallel_worker_count = 0` to use all cores. If the
machine becomes unresponsive, use `-1` or an explicit worker count. Increasing
mesh density and target-point count also increases the sensitivity matrix size;
parallel workers reduce wall time for independent chunks, but they do not remove
the memory cost of storing and solving the dense optimization system. With
`progress_flag = True`, CoilGen prints percentage progress for the main pipeline
stages plus the long chunked kernels such as sensitivity assembly, contour field
evaluation, connected-wire field evaluation, and inductance evaluation. Each
completed progress task prints its execution time at the final `100%` line.
The default sensitivity backend uses a NumPy-vectorized local kernel rather
than the original scalar Python loop. The default optimizer tries a direct
linear solve and falls back to the pseudo-inverse when the residual check fails.
Wire clearance diagnostics use `wire_distance_method = "auto"`: moderate
compact paths use the vectorized all-pairs evaluator, while large paths above
`wire_distance_spatial_min_segments` attempt conservative AABB spatial pruning
and fall back when the candidate set is not small enough.

## Active Spyder Plots

For active matplotlib 2D/3D figures in Spyder, run:

```text
projects/general_examples/scripts/spyder_design_example.py
```

The script runs a self-contained y-gradient design and opens matplotlib figures
for the wire path, raw contours, contour groups, stream function, current
density, interconnect debug view, swept conductor body, field comparison,
field point clouds, gradient maps, error maps, central cross-section slices,
and error metrics. It also leaves `result` and `figures` in the Spyder
namespace for inspection.

Rotation and live zoom require an interactive matplotlib backend. The Spyder
example requests the Qt backend before plotting. If figures still appear as
static images in Spyder's Plots pane, set Spyder's graphics backend to Qt:
`Tools > Preferences > IPython console > Graphics > Backend > Qt`, then restart
the IPython kernel and run the example again.

Matplotlib is optional for the package because the SVG exporters do not need it.
Use a Spyder environment that already has matplotlib installed, or install the
plotting extra:

```powershell
python -m pip install -e ".[plot]"
```

For local development with plotting and pytest available:

```powershell
python -m pip install -e ".[dev]"
```

## Generated Examples

Run the generated example suite and save the full diagnostic/visual output set:

```powershell
.\.venv\Scripts\python.exe projects\general_examples\scripts\run_standard_examples.py
```

In Spyder, set `PARAMETERS["run"]["dense_examples"] = True` in
`projects/general_examples/scripts/run_standard_examples.py` to include denser generated cylinder and
planar cases. Set `PARAMETERS["run"]["selected_examples"]` to a tuple of
example names to run only a subset.

Outputs are written below
`outputs/general_examples/<date-time-run>/standard_examples/`. Each generated case
calls `save_plot_gallery`, which saves the compact wire/field plots plus raw
contours, contour groups, fixed isometric views, XY/XZ/YZ stream-function
surface maps, XY/XZ/YZ current-density maps, XY/XZ/YZ/ISO
target/layout/stream/error target-field maps, interconnect debug views,
target/layout/stream/error gradient maps, explicit layout/stream error maps,
within-group cut-choice views, route diagnostics, topology tree diagnostics,
within-group cut diagnostics, contour-orientation diagnostics, field-error
diagnostics, and inter-group diagnostics. The dense set also
includes a Halbach-style rotated cylinder and a generated flanged split-bore
surface. The example runner fails on topology/routing failures but reports
whole-wire near-contact as a warning because automatic whole-wire rerouting is
intentionally not implemented yet.

For a single result, the same full plot set is available directly:

```python
from coilgen import CoilGenConfig, run_coilgen, save_plot_gallery

result = run_coilgen(CoilGenConfig(field_shape_function="y"))
files = save_plot_gallery(result, "outputs/my_case", prefix="my_case")
```

For a compact thesis-style parameter study over generated cylinder examples:

```powershell
.\.venv\Scripts\python.exe projects\general_examples\scripts\generated_parameter_sweep.py
```

The default sweep writes
`outputs/general_examples/<date-time-run>/generated_parameter_sweep/` and varies
potential levels, potential offset, mesh density, regularization, and
interconnection width. In Spyder, set `PARAMETERS["run"]["full_sweep"] = True`
or edit `PARAMETERS["sweep"]`, `PARAMETERS["full_sweep"]`, and
`PARAMETERS["coilgen"]` at the top of `projects/general_examples/scripts/generated_parameter_sweep.py`.

## Validation Examples

The constant-field solenoid validation is the simplest whole-pipeline physics
check. It requests a constant `Bz` field on a generated cylinder, extracts the
circular stream-function contours, builds an ordered solenoid path, and compares
the center field and inductance against analytical coaxial circular-loop
formulas:

```powershell
.\.venv\Scripts\python.exe projects\general_examples\scripts\solenoid_constant_z_validation.py
```

The reusable validation functions live in `coilgen.validation`, including
`apply_ordered_solenoid_wire_path` and
`validate_solenoid_against_analytical_formulas`.

Additional generated validation/sanity cases are:

- `projects/general_examples/scripts/conventional_cylindrical_mri_gradient.py`: compact cylindrical
  gradient design and active plotting path.
- `projects/standard_cylindrical_3axis_gradient/run_standard_cylindrical_3axis_gradient.py`:
  standard unshielded cylindrical `Gx/Gy/Gz` project with one shared user
  settings block and optional plot books.
- `projects/general_examples/scripts/open_biplanar_gradient.py`: compact open biplanar
  gradient design.
- `projects/general_examples/scripts/imported_target_shim_example.py`: imported target/shim
  design using a generated CSV target map.
- `projects/general_examples/scripts/simple_ygradient.py`: expert-parameter y-gradient regression case.
- `projects/general_examples/scripts/spyder_design_example.py`: active plot smoke test for wire,
  contours, groups, fields, gradients, errors, and L-curve diagnostics.

Project-specific studies live in their own folders under `projects/`. Read each
project README before interpreting its figures or tables.

## Current Limitations

- Automatic whole-wire collision repair is not implemented. Near-contact is
  diagnosed and some return segments can be normal-shifted, but a final design
  still needs clearance review.
- The intrinsic topology path assumes orientable coil surfaces with meaningful
  mesh normals.
- Geodesic distances use robust Dijkstra-style graph distances on the surface
  mesh. Fast Marching or Heat Method distances are not implemented.
- Final contour smoothness is mesh-resolution limited. Finer meshes and more
  target samples improve visual smoothness and field evaluation quality but
  increase matrix size and runtime.

## CSV Target Fields

Set `target_field_definition_file` to a `.csv` or `.tsv` file to import target
points and field values directly. For a vector field, provide coordinates and
all three field components:

```text
x,y,z,bx,by,bz
```

For CoilGen's usual scalar target-map use case, a named `Bz` map is enough:

```text
x_m,y_m,z_m,lin_x,lin_y,lin_z,weight,suppressed,group
```

Set `target_field_definition_field_name = "lin_y"` to use the `lin_y` column as
`Bz`; `Bx` and `By` are then set to zero. Aliases such as `main_bz`,
`target_bz`, `field`, `bx_main`, and `main_bx/main_by/main_bz` are accepted.
Optional columns are `weight`, `suppressed`, `group`, and full gradient columns
`dbz_dx,dbz_dy,dbz_dz`.

Target field values are interpreted as tesla by default. For shim-field maps
stored in proton-frequency units, set `target_field_units = "Hz"`; CoilGen
converts those values internally using `42.57747892e6 Hz/T`. Set
`field_display_units = "Hz"` when reports and field-map plots should be labeled
and scaled in Hz instead of the default tesla units.

For shim-coil inspection, CoilGen also reports final-current residual
homogeneity metrics and saves a shimming-style slice mosaic in the plot gallery:

```python
from coilgen import save_shim_homogeneity_mosaic

save_shim_homogeneity_mosaic(result, "outputs/shim_residual_mosaic.svg")
```

The mosaic plots `target - connected-wire Bz` on representative target-region
slices with one symmetric color scale. This sign convention matches the
remaining requested correction after applying the coil field.

For low-level configuration dictionaries, the `local_b0_shim` preset applies
Hz target/display units and lightweight local-shim defaults:

```python
from coilgen import apply_recommended_settings

params = apply_recommended_settings(params, preset="local_b0_shim")
```

The runnable example is:

```powershell
.\.venv\Scripts\python.exe projects\general_examples\scripts\csv_target_field.py
```

## Preoptimized Stream Functions

Set `preoptimized_stream_function_file` to bypass the inverse stream-function
solve and continue directly with contour extraction, topology, routing, and
field/layout diagnostics. The MATLAB-style alias `sf_source_file` is also
accepted, but the explicit Python name is preferred.

Preferred CSV/TSV schema:

```text
part,vertex_index,x_m,y_m,z_m,stream_function
1,0,0.1,0.0,-0.125,-0.125
1,1,0.092,0.038,-0.125,-0.125
```

`part` is 1-based for readability. `vertex_index` may be 0-based or 1-based;
if zero appears anywhere, the column is interpreted as 0-based. The coordinate
columns are optional when indices are provided, but useful for inspection.
Without index columns, coordinate matching is supported for small files via
`preoptimized_stream_function_match_tolerance`.

JSON and NPZ may also provide a single `stream_function` vector in combined
mesh vertex order, or per-part vectors in JSON:

```json
{"parts": [{"part": 1, "stream_function": [0.0, 0.1, 0.2]}]}
```

The mesh used in the run must match the mesh used to create the imported stream
function. The runnable example writes both a target-map CSV and a
stream-function CSV, then imports them:

```powershell
.\.venv\Scripts\python.exe projects\general_examples\scripts\preoptimized_stream_function.py
```

## Diagnostics

Route diagnostics are stored on each `CoilPart.route_diagnostics` and can be
saved with `save_route_diagnostics`. The inter-group connector ratio threshold,
connector-to-contour clearance, opening-cut span threshold, and strict failure
mode are configurable through `CoilGenConfig`. Inter-group routing evaluates
the nearest `inter_group_connection_candidate_count` point-pair candidates per
group pair and prefers candidates that satisfy opening-span, connector-ratio,
and unrelated-contour clearance checks. If all candidates fail, the closest
candidate is still used and marked by `used_clearance_fallback`.

Within each loop group, opening cuts are selected as a consistent high or low
rail across all loops in the group. The default selector uses a bounded
candidate search over high/low rail positions and scores the resulting return
path by straightness, length, turn penalty, line deviation, and opening
validity. Explicit `force_cut_selection` values still constrain the side, but
the selector can still choose the best candidate position on that side.

Route diagnostics include `route_quality_status`: `pass`, `warn`, or `fail`.
Degenerate openings, suspicious inter-group connectors, and configured
inter-group clearance violations are failures. Whole-wire near-contact,
geodesic direct fallback, and clearance-candidate fallback are warnings.
Use `inter_group_connection_path_method="surface_geodesic"` for strict
surface-only routing. Use `"surface_geodesic_or_direct"` for design and
optimization runs that should prefer mesh-geodesic connectors but still finish
and report a warning when an inter-group route has to fall back to a direct 3D
connector.

Topology diagnostics export the group tree, parent/child/sibling relations,
loop indices, and containment relations. Within-group cut diagnostics export
selected high/low side, selected candidate indices, candidate counts, and
return-path straightness metrics. Contour-orientation diagnostics report the
stream-current alignment of each extracted contour. During contour extraction,
each contour is reversed when necessary so its tangent agrees with the
piecewise-linear surface current `K = grad(stream_function) x mesh_normal`.
Surface diagnostics can also report disconnected face components, boundary
loops, provided-normal inconsistencies, non-manifold edges, and optional
obvious self-intersections.

Each `WirePath` interval carries an explicit `WirePathGeometry`: `surface`,
`offset_surface`, or `free_3d`. Only true surface intervals are adaptively
projected to the support; lifted returns and spatial connectors remain
authoritative 3D curves. The default sweep frame uses surface-constrained
parallel transport and fades the surface-normal guide out on raised/free
intervals. Bounded quality coarsening may remove at most
`ceil(0.02 * sample_count)` sweep samples, only within the configured chord
tolerance and only when the rebuilt mesh strictly improves fold/triangle
quality.

Swept-body diagnostics are stored on `CoilPart.manufacturing_diagnostics` and
can be saved with `save_manufacturing_diagnostics`. STL export writes available
swept meshes by default, even when manufacturing checks fail. Pass
`fail_on_violations=True` to require manufacturing readiness before writing.
The triangular sweep remains a visualization/mesh-export
backend rather than an analytic CAD solid. Each swept-body STL export also
writes the corresponding current-carrying surface mesh as
`<prefix>_partN_current_carrying_surface.stl` in the same directory.
