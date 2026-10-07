# Standard Cylindrical 3-Axis Gradient Project

Clean starting point for an unshielded cylindrical MRI gradient set. It builds
three independent layouts on nested cylinders (130, 132 and 134 mm diameter):

- `Gx`: target `Bz = x`
- `Gy`: target `Bz = y`
- `Gz`: target `Bz = z`

The project is intentionally small. Most users only edit
`run_standard_cylindrical_3axis_gradient.py`. The script builds a
`CoilProject` with one channel per gradient axis, calls `project.solve()`, and
opens the optional project plot book. Shared geometry, target, optimization,
conductor, system, and numerical settings live on the `GRADIENT_TEMPLATE`
object; channels set the target axis and support radius, with explicit Y routing
field budgets and physical corner realization described below.

## Run

From PowerShell:

```powershell
cd "<repo-root>"
.\.venv\Scripts\python.exe projects\standard_cylindrical_3axis_gradient\run_standard_cylindrical_3axis_gradient.py
```

From Spyder, open `run_standard_cylindrical_3axis_gradient.py`, edit the
top-level `SHOW_PLOT_BOOK`, `GRADIENT_TEMPLATE`, and `PROJECT` objects, and
run the file.

The plot book opens by default (`SHOW_PLOT_BOOK = True`). If the solve has already
finished in Spyder, open it without recomputing the coils:

```python
plot_book = project_run.show_plot_book()
```

If an older script stopped during STL export after solving, recover the completed
run from `project_run = PROJECT.last_run` before opening its plot book.

For a headless run, add `--headless` or set `MPLBACKEND=Agg`. `NATIVE_THREADS`
bounds BLAS threads inside CoilGen workers, independently of `cpu_workers`.
`main(project)` accepts an explicit project variant without changing the default
template. The default three-axis solve and all seven 64-by-64 diagnostic slices
were executed during the example audit with one worker and one native thread.

## Useful Settings

- `PROJECT.channels`
  Keep `GX`, `GY`, and `GZ`, or remove channels to solve only one or two axes.

- `GRADIENT_TEMPLATE.radius_m` and `GRADIENT_TEMPLATE.length_m`
  Shared unshielded cylindrical support.

- `GRADIENT_TEMPLATE.target_scale`
  Requested gradient amplitude. `0.010` means 10 mT/m.

- `GRADIENT_TEMPLATE.target_region_radius_m`
  Radius of the spherical design volume.

- `GRADIENT_TEMPLATE.contour_discretization_method`
  Defaults to `"engineering"`, deriving contours from `max_wire_current_A`
  and `min_wire_distance_m`. Set it to `"level_count"` to use the historical
  `contour_levels` and `potential_offset_factor` setup.

- `GRADIENT_TEMPLATE.max_current_A` and
  `GRADIENT_TEMPLATE.min_wire_distance_m`
  Engineering contour specs. In engineering mode CoilGen limits the solved
  surface current density, then chooses the contour current from the requested
  minimum wire spacing.

- `SHOW_PLOT_BOOK`
  Opens the interactive CoilGen project plot book.

- `ENABLE_SWEEP` and `EXPORT_STL`
  Create the full conductor triangle mesh for the `swept_body` inspector and
  export STL meshes. Manufacturing failures remain visible in diagnostics and
  the export report; they do not suppress the mesh files.
  Routing first realizes the physical centerlines and frames without conductor
  triangles. Full body previews are limited to finalist validation; the selected
  route is then swept for output, reusing an accepted candidate's validated mesh.
  The joint search reports cheap topology planning, staged curve shaping and
  finalist validation before final mesh generation. Each
  phase shows elapsed time and a rough remaining-time estimate once timings are
  available. Estimates reset between phases; connection coverage is an upper
  bound because pruning may finish the search sooner. ETA refers to the active
  phase, not the complete XYZ run.

- `GRADIENT_TEMPLATE.advanced_parameters["joint_group_routing_search_mode"]`
  Defaults to `"progressive"`: rank complete directed circuits with a geometric
  guide, then generate fine curves only for a small batch. A broader curve and
  topology batch follows when the first batch produces no accepted route.
  `"thorough"` retains the wider eager search for comparison. Both modes apply
  the same protected-contour, fixed-current field, bend, support and conductor
  clearance checks. Exhausting a batch means no valid improvement was found
  within that search budget; it does not prove infeasibility or optimality.
  Routing diagnostics also check the actual triangulated mesh, SF parity up to
  its additive gauge, and directed contours. Verified group correspondences
  can prioritize related circuit proposals; they never replace physical or
  field validation. Mesh triangulation, contour cuts or SF asymmetry can prevent
  use of an otherwise visually symmetric cylindrical design.

- `RETURN_LIFT_LIMIT_M`, `ROUTING_CLEARANCE_M`, and the routing field settings
  The example keeps the requested 5 mm contour spacing, permits a 6 mm return
  lift, and uses an explicit 3 mm centerline clearance for joint routing.
  Diagonal turn splices cannot retain the full contour pitch as their mutual
  clearance. Conductor-body and bend limits remain independently enforced.
  Joint routing combines group returns and prioritizes fewer
  continuous raised transfers, followed by longitudinal field distortion and
  conductor length. X/Z bound unchanged-current vector change by 2% of the
  peak sampled DSV target field. Their explicit `"non_regression"` acceptance
  preserves passing limits and admits inherited failures only when they do not
  worsen. Y uses explicit `"budget_only"` acceptance, a 30% full-vector cap,
  and a separate 5% pointwise cap along normalized B0. Its transverse-field
  change exceeds the tighter vector allowance even for routes with better
  longitudinal distortion. The original gradient/layout limits are still
  evaluated: inherited, worsened inherited and newly failed limits remain
  visible, and any absolute field failure makes the whole design report FAIL.
  Suppression nonregression, conductor clearance and bend constraints remain
  independently enforced. These routing budgets compare complete raw and
  fabricated routes at unchanged series current; they do not refit current.
  `ROUTING_SEARCH_CUTS=False` keeps the original contour openings while choosing
  the group order, spiral directions and combined return curves.
  Core routing defaults retain absolute field qualification, a 0.1% vector
  change budget, and the existing lift allowance. The separately solved
  channels still require an assembly clearance audit before fabrication.

- `GY_CORNER_CLUSTER_SOURCE_DEVIATION_M`
  Y uses physical fillets to realize inherited contour-tip kinks that are too
  tight for the 1.4 mm wire. Its explicit 1 mm two-way source corridor permits
  compound intrinsic fillets when separate corner trims cannot fit. The bound
  uses all original source samples after the declared analytic-cylinder
  retraction; that retraction is reported separately. Compound regions certify
  endpoint positions and intrinsic tangent matching, source ownership and bend
  radius. Ordinary fillets retain their independent bend and complete-frame
  checks; exact preservation of their original sampled tangent vectors is not
  asserted. The complete
  fabricated route and fixed-current fields must still pass before adoption.
  Saved nominal contours and series current remain unchanged. This allowance
  is disabled in core defaults and in the other example channels.

## Outputs

Results are written below:

```text
outputs/standard_cylindrical_3axis_gradient/
```

Each axis gets its own folder with a Markdown report, CSV summary, and optional
SVG plot gallery. SVG galleries are off by default to avoid accumulating plot
files. The root output folder also contains the CoilProject summary files:

- `standard_cylindrical_3axis_gradient_project_summary.csv`
- `standard_cylindrical_3axis_gradient_project_summary.json`
- `standard_cylindrical_3axis_gradient_project_summary.md`

When `EXPORT_STL` is enabled, `stl_export_report.json` records each channel as
`exported` or `unavailable`, with its generated files. Exported channels also
record `manufacturing_ready`; failing bodies have per-body
`manufacturing_warnings`.
Each solved channel's `stl/` folder contains manufacturing diagnostics in JSON
and CSV. The mesh inspector displays the stored
manufacturing failure reasons; watertightness alone does not establish clearance,
bend, or triangle-quality compliance. STLs with manufacturing warnings are for
geometry inspection, not fabrication approval. The report describes the latest
export attempt. If you manually re-export into an existing run folder, older
STL files are retained; use the report's `files` entries to identify files
produced by the latest attempt.

This is a contour-level electromagnetic design screen, not a final engineered
gradient insert. Thermal, acoustic, mechanical, PNS, driver, and cryostat
coupling checks are outside this starter project.
