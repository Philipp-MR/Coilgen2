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
object; the channel objects only override the requested target expression.

## Run

From PowerShell:

```powershell
cd "<repo-root>"
.\.venv\Scripts\python.exe projects\standard_cylindrical_3axis_gradient\run_standard_cylindrical_3axis_gradient.py
```

From Spyder, open `run_standard_cylindrical_3axis_gradient.py`, edit the
top-level `SHOW_PLOT_BOOK`, `GRADIENT_TEMPLATE`, and `PROJECT` objects, and
run the file.

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

This is a contour-level electromagnetic design screen, not a final engineered
gradient insert. Thermal, acoustic, mechanical, PNS, driver, and cryostat
coupling checks are outside this starter project.
