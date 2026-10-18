# Front-page figures and numerical evidence

The GitHub welcome page uses actual CoilGen2 computations, with generated
geometry and analytic targets. There are no external meshes, measured data,
MATLAB dependencies or synthetic decorative field maps in these figures.

## Reproduce

From the repository root, in an activated Python 3.10+ environment:

```bash
python -m pip install -e ".[research,plot]"
python docs/frontpage/generate_figures.py
```

The script uses one numerical thread, a headless Matplotlib backend and strict
CVXPY constrained optimization. It recomputes both cases and writes the PNG/SVG
figures, compressed numerical arrays and JSON evidence next to the script.
Small solver/library differences can change the last reported digits; use the
recorded dependency versions when comparing a specific figure revision.

## What was computed

| Quantity | Cylinder | Biplanar |
| --- | --- | --- |
| Target | Bz = 0.01 y T | Bz = 0.01 z T |
| Support | 120 mm radius, 300 mm length | Two 300 × 300 mm planes, 200 mm separation |
| Triangle count | 576 | 576 total |
| Target region | 45 mm radius sphere, centered at the origin | Same |
| Solver | Constrained minimum continuous-sheet power | Same |
| Design budget | 3.2% relative Bz norm error | Same |
| Independent check budget | 4%, using a 0.8 design margin | Same |
| Contouring | 16 requested levels; original signed contour current | Same |
| Extracted contour count | 32 | 36 total |

The generator checks the actual mesh bounds, cylinder radius and plane
dimensions/separation before producing evidence. Its cylinder input compensates
the legacy builder's height convention to realize the stated 300 mm length.
Executed support extents and bounds are recorded in `metrics.json`.

The independent grid contains **912** cell-centered samples inside the target
sphere and shares no points with the design grid. Relative RMS error is
`100 * ||Bz − Bz_target||₂ / ||Bz_target||₂`, using equal sample weights.
This is an aggregate error, including target-zero locations without pointwise
division. Neither sheet nor contour current is refitted to the validation data.

The two continuous-sheet errors are 3.404% and 3.014%; the unconnected-contour
errors are 3.381% and 2.959%. The modest reduction after contouring is specific
to these examples and does not imply that discretization generally improves
a field. A field constraint enforced on the design grid need not hold on new
samples; the solve reserves margin explicitly.

The field map panels show actual samples in the `x = +3.75 mm` slice, with no
interpolation. Target panels share one color scale; all residual panels share
another. The geometry colors show relative stream function separately for each
case. Exact stream-function values in amperes are available in the array files.

## Files

- [generate_figures.py](generate_figures.py): complete configurations,
  computation, validation and figure generation.
- [metrics.json](metrics.json): executed configurations and mesh bounds, solver status, field
  errors, current values, dependency versions, source fingerprint and artifact
  checksums. The source fingerprint identifies the working tree used for the
  figures, which may include changes after the recorded base commit.
- [case_0.npz](case_0.npz), [case_1.npz](case_1.npz): SI mesh vertices,
  zero-based triangle indices, stream functions, contours and validation
  points/target/sheet/contour fields. Load with `numpy.load(path,
  allow_pickle=False)`.
- [coil_designs.png](coil_designs.png), [coil_designs.svg](coil_designs.svg):
  generated supports and iso-current contours.
- [field_validation.png](field_validation.png), [field_validation.svg](field_validation.svg):
  target and residual maps with full-volume error metrics.

## Interpretation limits

Independent sampling changes evaluation locations, while retaining the same
order-three sheet quadrature and support mesh. This is not a mesh or quadrature
convergence study. The contour field uses the analytical straight-segment
thin-wire Biot–Savart kernel, at the original contour current.

The illustrated contours are **unconnected**, with postprocessing, conductor
sweeps and inductance calculation deliberately skipped. They do not represent
a completed electrical circuit or fabrication-validated winding. Routing,
feeds, return conductors, conductor cross section and downstream engineering
can change the final field. Refer to [physical assumptions](../PHYSICAL_ASSUMPTIONS.md)
and [result interpretation](../RESULT_INTERPRETATION.md) before using a design.
