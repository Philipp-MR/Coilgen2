# Bounded contour-order benchmark

This benchmark isolates **which contour to visit next** while preserving the
directed traversal of every retained contour. It does not optimize arbitrary
wire shapes, the cut positions, field error, or conductor clearance.

Seven concentric circles have radii 12, 18, 24, 30, 36, 42, and 48 mm. Each retains
its counterclockwise arc from `theta = +0.16` to `2*pi - 0.16` rad. The outer circle
is visited first and the inner circle last. The five remaining circles have
`5! = 120` possible orders, all enumerated.

Each connector is a cubic Bezier curve from the previous arc's end to the next
arc's start. Its endpoint tangents match the directed circle tangents; both
handle lengths are 0.35 times the port-to-port distance. No circle is reversed.

| Quantity | Adjacent inward order | Fixed scrambled order |
|---|---:|---:|
| Radii visited, mm | 48, 42, 36, 30, 24, 18, 12 | 48, 18, 42, 24, 36, 30, 12 |
| Connector length, mm | 70.755426 | 135.601851 |
| Pairwise connector intersections | 0 | 14 |
| Connector / retained-arc intersections | 0 | 0 |
| Radial travel between ports, mm | 36 | 108 |
| Reversals in radial visit order | 0 | 4 |

The adjacent inward order is the **unique shortest route among these 120
fixed-geometry routes** and the only one with no centerline intersection.
The next-shortest route has 78.794360 mm of connector length. The scrambled order
was declared in the code in advance; it was not selected to maximize crossings.
Its connectors are 91.65% longer. The retained contour length is the same in all
orders: 1252.268915 mm. The connector percentage must not be described as a
percentage change in total wire length.

Lengths use 96-point Gauss-Legendre quadrature of the analytic Bezier derivative.
Intersection counts are computed on polylines and deduplicated at sample
vertices. Intended endpoint joins are excluded only for the incident loop and
matching endpoint. Counts for every order agree at 161 and 321 samples per
connector, with three times as many samples per retained arc. These checks
support the finite numerical comparison; they are not an analytic proof about
all continuous curves. Isolated transverse intersections are counted; tangential
contacts and collinear overlaps would need separate collision handling in a
general manufacturing checker.

Zero centerline intersections alone does not establish wire clearance, bend
radius, acceptable field perturbation, or easy terminal access. These are open
routes with two free terminals. Any return lead or circuit closure is a separate
routing requirement. All lengths are measured on the planar surface `z = 0`; a
non-isometric lift to a curved surface changes their physical lengths.

Run from this directory with the existing project interpreter:

```powershell
& ../.venv/Scripts/python.exe ./benchmark.py
```

The script imports only NumPy and the Python standard library. It does not import
or edit CoilGen production code or alter its environment. The JSON output includes
all 120 rankings, plus `best` and `scrambled` objects with `arcs_mm`, `bridges_mm`,
and `bridge_control_points_mm` arrays for plotting. Coordinates are `(x, y)` in mm.
