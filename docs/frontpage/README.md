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
CVXPY constrained optimization. It recomputes both cases, evaluates the dense
field maps, builds illustrative swept conductor bodies, and writes high-resolution
PNG/SVG figures, compressed numerical arrays and JSON evidence next to the script.
Small solver/library differences can change the last reported digits; use the
recorded dependency versions when comparing a specific figure revision.

To retain the exact saved numerical designs while regenerating the gallery:

```bash
python docs/frontpage/generate_figures.py --reuse-data
```

This mode checks the numerical source fingerprint, case configurations, array
checksums and recorded RMS errors before reuse. The dense maps are still
evaluated afresh. The original numerical revision and current rendering sources
are recorded separately; no optimization or display-time current fit is made.

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

The fidelity figure compares requested and calculated Bz at all 912 volume
samples, their signed residual distributions, and empirical cumulative absolute
errors. Distribution and coverage axes divide by the peak absolute target Bz
for that design; they do not divide pointwise by a target that crosses zero.
Those percentages differ from the relative RMS norm used in the table above.

The swept-body illustration uses the public conductor sweep on each original
closed contour, with a **0.9 mm radius**, 16-sided circular profile. The bodies
are mounted outside the support with a nominal 0.9 mm centerline offset along
its normal and remain independent loops. Actual projection and displacement
diagnostics are retained with the body meshes. The field figures evaluate
the original thin-wire paths, not the displaced body centroids.
The close-up shows the resulting triangular surface
and cross-section; surface shading represents geometry, not current or field.
The selected conductor size is illustrative and does not establish clearance,
electrical connections, manufacturability or fabrication acceptance.

All **68** swept boundaries are watertight with positive signed volume and
zero open or nonmanifold edges. This does not establish geometric validity:
four planar bodies have local self-intersections, and 64 bodies fail at least
one per-body manufacturing diagnostic, including unresolved frame sampling.
Those failures are retained in `swept_bodies.json`; no loops or failed checks
are hidden. Between-body clearance and a complete fabrication review were
not performed. These are illustrative outputs of the sweep stage, not
released conductor designs.

## Files

- [generate_figures.py](generate_figures.py): complete configurations,
  computation, validation and figure generation.
- [presentation.py](presentation.py), [presentation.json](presentation.json):
  geometry, stream-function and full-volume fidelity plots, with input hashes
  and derived residual statistics.
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
  zero-based triangle indices, stream functions, contours and validation
  points/target/sheet/contour fields. Load with `numpy.load(path,
  allow_pickle=False)`.
- [coil_designs.png](coil_designs.png), [coil_designs.svg](coil_designs.svg):
  generated supports and iso-current contours.
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
