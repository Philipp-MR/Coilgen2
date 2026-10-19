# Front-page figures and numerical evidence

The GitHub welcome page uses actual CoilGen2 computations, with generated
geometry and analytic targets. There are no external meshes, measured data,
MATLAB dependencies or synthetic decorative field maps in these figures.

## Reproduce

From the repository root, in an activated Python 3.10+ environment:

```bash
python -m pip install -e ".[research,plot,gui]"
python docs/frontpage/generate_figures.py
```

The script uses one numerical thread, Matplotlib, the VTK/PyVista rendering
stack and strict CVXPY constrained optimization. It recomputes both cases,
smooths and filters their contours, evaluates the processed fields, builds
illustrative swept conductor bodies, and writes high-resolution
PNG/SVG figures, compressed numerical arrays and JSON evidence next to the script.
Small solver/library differences can change the last reported digits; use the
recorded dependency versions when comparing a specific figure revision.

To retain the exact saved numerical designs while regenerating the gallery:

```bash
python docs/frontpage/generate_figures.py --reuse-data
```

This mode checks the numerical source fingerprint, case configurations, array
checksums and recorded RMS errors before reusing the sheet solutions. Contour
conditioning is cached by its complete inputs and settings; dense field maps
are evaluated afresh. Raw and processed paths and their field errors remain
separate. No optimization or display-time current fit is made.

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
| Retained smoothed contours | 32 | 30 total |

The generator checks the actual mesh bounds, cylinder radius and plane
dimensions/separation before producing evidence. Its cylinder input compensates
the legacy builder's height convention to realize the stated 300 mm length.
Executed support extents and bounds are recorded in `metrics.json`.

The original solver-independent grid contains **912** cell-centered samples
inside the target sphere and shares no points with the design grid. It is also
used to gate contour smoothing/pruning, so it is a postprocessing acceptance
grid. An untouched **1,472-point** cell-centered holdout is evaluated afterwards,
with no points shared with either the solve grid or the acceptance grid.
The retained geometry is not tuned to that holdout. Relative RMS error is
`100 * ||Bz − Bz_target||₂ / ||Bz_target||₂`, using equal sample weights.
This is an aggregate error, including target-zero locations without pointwise
division. Neither sheet nor contour current is refitted to the validation data.

The two continuous-sheet errors are 3.404% and 3.014%. The original raw-contour
errors are 3.381% and 2.959%; after smoothing and filtering they are 3.900% and
3.798%, at unchanged currents. The displayed curves and all contour-field
figures use these processed paths. The raw solved arrays remain available for
comparison. A field constraint on the design grid need not hold at new samples.

## Smooth and filter the contours

The gallery uses the existing periodic cubic B-spline smoothing kernel in
physical cylinder/plane coordinates. The cylinder paths lie on the analytic
120 mm radius surface; planar paths retain their original z coordinates.
Closed-loop orientation and the original signed turn current are preserved.
Output points are spaced at approximately 0.75 mm along each curve.
Tight bends receive additional samples from the same spline so adjacent
polyline tangents turn by at most 4 degrees; this does not change the fitted curve.

Low-contribution loops are considered for removal in symmetric groups, with
a final **4% full-volume Bz RMS error limit** at the original current. Six
small central biplanar loops are removed. The smallest cylindrical loops are
retained because dropping them after smoothing would exceed that limit.
The settings, retained/dropped loop identities, displacements and before/after
fields are recorded in [contour_processing.json](contour_processing.json).

| Processed-contour check | Samples | Cylindrical Gy | Biplanar Gz |
| --- | ---: | ---: | ---: |
| Postprocessing acceptance grid | 912 | 3.8998% | 3.7976% |
| Untouched holdout | 1,472 | 3.8951% | 3.8845% |

The holdout changes neither the retained loops nor the smoothing parameters.
Its continuous-sheet errors are 3.4100% and 3.1388%, respectively. All values
are relative Bz norm errors at the original solved current.

The field maps evaluate the actual sheet and contour fields at 7,845 points
on a 101 × 101 grid inside the central `x = 0` circular cross-section. Neither
field is interpolated. Before that evaluation, the reconstructed sheet and
contour operators must reproduce all 912 saved volume samples to roundoff.
Field panels share one color scale; all residual panels share another. The
reported relative RMS errors continue to use the separate full-volume grid.

The current-distribution atlas shows the physical piecewise-linear stream
function in amperes. Cylinder triangles crossing the angular seam are unwrapped
locally and duplicated across the periodic chart boundary. The biplanar maps
retain their actual sheet positions and signs. Color scales differ by geometry.

The fidelity figure compares requested and calculated Bz at all 912 acceptance
samples, their signed residual distributions, and empirical cumulative absolute
errors. Distribution and coverage axes divide by the peak absolute target Bz
for that design; they do not divide pointwise by a target that crosses zero.
Those percentages differ from the relative RMS norm used in the table above.

The swept-body illustration uses the public conductor sweep on each retained
smoothed contour, with a **0.9 mm radius**, 32-sided circular profile centered
on its path. The field figures use these same processed centerlines. Actual
projection and displacement diagnostics are retained with the body meshes.
The VTK render uses smooth surface normals, lighting and a close-up of a
rounded bend; geometry and profile dimensions are not exaggerated.
Surface shading represents geometry, not current or field.
The selected conductor size is illustrative and does not establish clearance,
electrical connections, manufacturability or fabrication acceptance.

The retained boundaries and all their geometry diagnostics are saved in
`swept_bodies.json`. All **62** bodies pass the production sweep's
`validate=True` checks: watertight boundaries, positive volumes, and no
self-intersection, frame, curvature-envelope or profile-Jacobian violations.
The evaluated field paths and swept centerlines agree to roundoff.
Between-body clearance and a complete fabrication review remain separate
from this illustration. These are not released conductor designs.

## Files

- [generate_figures.py](generate_figures.py): complete configurations,
  computation, validation and figure generation.
- [presentation.py](presentation.py), [presentation.json](presentation.json):
  geometry, stream-function and full-volume fidelity plots, with input hashes
  and derived residual statistics.
- [contour_processing.py](contour_processing.py), [contour_processing.json](contour_processing.json):
  physical spline smoothing, contribution-based pruning, and fixed-current
  before/after field validation.
- [refined_case_0.npz](refined_case_0.npz), [refined_case_1.npz](refined_case_1.npz):
  retained smooth paths, original sheet solutions and reevaluated contour fields.
- [field_maps.py](field_maps.py), [field_maps.json](field_maps.json),
  [field_maps.npz](field_maps.npz): actual dense sheet/contour evaluation,
  roundoff reconstruction checks, points and SI fields.
- [swept_bodies.py](swept_bodies.py), [swept_bodies.json](swept_bodies.json),
  [swept_bodies.npz](swept_bodies.npz): conductor sweep inputs, triangle bodies
  and rendering provenance.
- [metrics.json](metrics.json): executed configurations and mesh bounds, solver status, field
  errors, current values, dependency versions, source fingerprint and artifact
  checksums. The source fingerprint identifies the working tree used for the
  figures, which may include changes after the recorded base commit.
- [case_0.npz](case_0.npz), [case_1.npz](case_1.npz): SI mesh vertices,
  zero-based triangle indices, stream functions, **raw** contours and validation
  points/target/sheet/contour fields. Load with `numpy.load(path,
  allow_pickle=False)`.
- [coil_designs.png](coil_designs.png), [coil_designs.svg](coil_designs.svg):
  generated supports and smoothed, filtered contours.
- [field_validation.png](field_validation.png), [field_validation.svg](field_validation.svg):
  densely evaluated central-plane fields and residuals, labeled separately
  from the full-volume error metrics.
- [stream_function.png](stream_function.png), [stream_function.svg](stream_function.svg):
  physical stream functions on the periodic cylinder chart and both planar supports.
- [validation_fidelity.png](validation_fidelity.png), [validation_fidelity.svg](validation_fidelity.svg):
  full-volume field agreement, residual distributions and error coverage.
- [swept_bodies.png](swept_bodies.png), [swept_bodies.svg](swept_bodies.svg):
  shaded conductor-boundary meshes and enlarged geometric details.

## Interpretation limits

Independent sampling changes evaluation locations, while retaining the same
order-three sheet quadrature and support mesh. This is not a mesh or quadrature
convergence study. The contour field uses the analytical straight-segment
thin-wire Biot–Savart kernel, at the original contour current.

The optimized examples deliberately skip routing, conductor sweeps and
inductance calculation. The separate swept-body gallery subsequently sweeps
their **unconnected** loops as an illustration, without a fabrication review.
They do not represent a completed electrical circuit or fabrication-validated winding. Routing,
feeds, return conductors, conductor cross section and downstream engineering
can change the final field. Refer to [physical assumptions](../PHYSICAL_ASSUMPTIONS.md)
and [result interpretation](../RESULT_INTERPRETATION.md) before using a design.
