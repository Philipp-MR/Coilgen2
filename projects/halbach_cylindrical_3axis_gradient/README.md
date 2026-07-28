# Halbach Cylindrical 3-Axis Gradient Project

Clean starting point for an unshielded cylindrical MRI gradient set. It builds
three independent layouts on the same cylinder. The cylinder/bore axis is
global `-y`, orthogonal to the global `B0` direction `+z`, matching a
transverse-field Halbach geometry:

- `Gx`: target `Bz = x`
- `Gy`: target `Bz = y`
- `Gz`: target `Bz = z`

The project is intentionally small. Most users only edit
`run_halbach_cylindrical_3axis_gradient.py`. The script builds a
`CoilProject` with one channel per gradient axis, calls `project.solve()`, and
opens the optional project plot book. Shared geometry, target, optimization,
conductor, system, and numerical settings live on the `GRADIENT_TEMPLATE`
object; the channel objects only override the requested target expression.

## Run

From PowerShell:

```powershell
cd "<repo-root>"
.\.venv\Scripts\python.exe projects\halbach_cylindrical_3axis_gradient\run_halbach_cylindrical_3axis_gradient.py
```

From Spyder, open `run_halbach_cylindrical_3axis_gradient.py`, edit the
top-level `SHOW_PLOT_BOOK`, `GRADIENT_TEMPLATE`, and `PROJECT` objects, and
run the file.

## Inputs

The default design generates its cylindrical support and target field in
memory, so it requires no static input files. Future imported meshes, measured
fields, or preoptimized solutions belong under:

```text
project_data/halbach_cylindrical_3axis_gradient/
```

Configuration paths beginning with `inputs/` or `data/` resolve to that
directory through `paths.py`. The complete `project_data/` tree is excluded
from version control.

## Useful Settings

The supplied manufacturing geometry is configured as follows:

- Gx, Gy, and Gz current-carrying surface diameters: 168, 182, and 196 mm.
- Cylinder length: 270 mm; spherical DSV diameter: 50 mm.
- The first winding surface is 1 mm radially below the 170 mm outer surface
  of the first cylinder. Each gradient cylinder is specified as 4 mm thick.
- The active sweep is a square rectangular cross-section, 1.70 mm wide and
  1.70 mm deep. The retained alternatives are a 1.40 mm round wire and the
  previous 1.96/1.60 mm by 1.80 mm trapezoid.
- The visualization sweep uses exact-mode path simplification to remove only
  redundant, nearly collinear rings that would otherwise create micron-scale
  triangle edges. Manufacturing diagnostics are written before STL readiness
  is evaluated, so every warning remains inspectable.
- `STL_EXPORT_REJECTION_POLICY="warn"` always writes visualization STL files
  while printing a prominent warning for bodies that are not manufacturing
  ready. Use `"skip"` to omit rejected STL files, or `"raise"` for a CI or
  manufacturing-release run that must terminate on the first rejected part.

- `PROJECT.channels`
  Keep `GX`, `GY`, and `GZ`, or remove channels to solve only one or two axes.

- `GRADIENT_TEMPLATE.radius_m` and `GRADIENT_TEMPLATE.length_m`
  Shared unshielded cylindrical support.

- `GRADIENT_TEMPLATE.rotation_axis` and `rotation_angle_rad`
  The default `(1, 0, 0)` and `pi/2` rotate the native cylinder axis from
  global `+z` to global `-y`; keep `b0_direction=(0, 0, 1)` for the
  transverse Halbach field convention.

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

- `AXIS_MIN_WIRE_DISTANCE_M`
  Per-axis engineering contour spacing used to balance the required current
  and gradient efficiency of the transverse-field Halbach set. The defaults
  are 5.0 mm for GX, 3.8 mm for GY, and 6.0 mm for GZ. These are contour-design
  spacings; verify conductor clearance, insulation, and routing before release.

- `SHOW_PLOT_BOOK`
  Opens the interactive CoilGen project plot book.

- `ENABLE_UNCONNECTED_CONTOUR_SMOOTHING`, `CONTOUR_SMOOTHING_FACTOR`,
  `CONTOUR_SMOOTHING_MAX_DISPLACEMENT_M`,
  `CONTOUR_SMOOTHING_TARGET_SEGMENT_LENGTH_M`, and
  `CONTOUR_SMOOTHING_PATCH_RINGS`
  Control periodic B-spline smoothing of extracted contour loops before
  grouping and routing. Circular cylinders use exact developed coordinates;
  arbitrary surfaces fall back to a harmonic 2D parameterization of the local
  mesh patch. The result is mapped back to exact triangle barycentric
  coordinates. Connected paths and inter-group connectors are not smoothed.

- `ENABLE_NORMAL_SHIFT`, `NORMAL_SHIFT_LENGTH_M`, and
  `NORMAL_SHIFT_TRANSITION_FACTOR`
  Control the post-routing lift applied to return/crossing segments along the
  mesh normals. The shift length is the displacement per routing level; the
  transition length is the shift length multiplied by the transition factor.
  These settings do not offset the unconnected contours or the complete coil.

## Outputs

Results are written below:

```text
outputs/halbach_cylindrical_3axis_gradient/
```

Each axis gets its own folder with a Markdown report, CSV summary, and optional
SVG plot gallery. SVG galleries are off by default to avoid accumulating plot
files. The root output folder also contains the CoilProject summary files:

When STL export is enabled, each axis `stl` folder contains both the swept
conductor volume and the corresponding cylindrical current-carrying surface.
The two files share the same part number for manufacturing association.

- `halbach_cylindrical_3axis_gradient_project_summary.csv`
- `halbach_cylindrical_3axis_gradient_project_summary.json`
- `halbach_cylindrical_3axis_gradient_project_summary.md`

This is a contour-level electromagnetic design screen, not a final engineered
gradient insert. Thermal, acoustic, mechanical, PNS, driver, and cryostat
coupling checks are outside this starter project.
