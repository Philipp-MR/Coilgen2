# Engineering Roadmap

This roadmap contains remaining engineering work. Implemented field-component
and region APIs, current-density/concomitant reports, physical conic objectives,
minimax optimization, finite scenarios and reusable study models are documented
in [Object Model](OBJECT_MODEL.md),
[Stream-function optimization](STREAM_FUNCTION_OPTIMIZATION.md) and
[Headless physical design](RESEARCH_PHYSICS_API.md).

Preserve existing mathematical contracts while extending the model. Compare
continuous-sheet, contour, routed-wire and fabricated-conductor results at the
same current and useful-field budgets; a successful solve does not establish
build readiness.

## Complete circuits and practical winding

Terminal strategy/count metadata and closed mathematical turns do not define a
complete supply/return circuit. Add terminal locations and polarity, lead pairs,
permitted return corridors, layer transitions and winding-order constraints.
Keep insulation, tolerance, former clearance and conductor dimensions distinct.

Validate continuity and current direction through every joint, evaluate fields
and electrical properties including leads, and reject cyclic winding orders or
inaccessible installation steps. Benchmark manufactured paths against accepted
contours without concealing added field error or clearance failures. See
[Coil Design Review](COIL_DESIGN_REVIEW.md) for the detailed fabrication issues.

## Thermal operation

`CurrentDensityReport` reports conductor loading; `ThermalScreeningReport`
currently records unevaluated thermal status. Extend this with explicit current
waveforms or duty cycle, RMS current, average Joule loss, material temperature
dependence, thermal boundary conditions and a stated cooling model.

Validate a lumped model against analytic heating/cooling cases and measured
steady/transient temperatures before using it for temperature limits. Missing
pulse or cooling inputs must remain unevaluated. Peak current density alone does
not establish permissible operation.

## Component, region and comparison consistency

The optimizer supports Cartesian, B0-projected and vector targets. Audit
component conventions across gradient fits, efficiency, concomitant estimates,
exports and GUI labels; some downstream reports still fit Cartesian Bz.
Validate jointly rotated geometry/targets/B0 and verify reported units and axes.

Typed named regions and custom physical field constraints already exist. Ensure
each region's component, sampling, weighting and hard-limit semantics survive
conversion, execution and independent validation. Test that diagnostic-only
regions cannot influence a solve. Compare active/unshielded systems on the same
external samples and matched useful-field budgets, reporting effective numerical
allowances and unavailable constraints explicitly.

Add convergence studies for support meshes, contour density and independent
field grids. Finite scenario constraints cover their supplied operators; broader
uncertainty claims require a specified uncertainty set and validation beyond the
optimization scenarios. Keep fixed-current and adjustable-current robustness
contracts distinct.

## Downstream gradient engineering

Single-axis drive estimates and field/electrical artifacts provide a starting
point. Extend interoperable exports with the complete circuit mutual-inductance
matrix, coupled drive assumptions, prescribed waveforms and coordinate metadata.
Add vector-potential/response-operator interfaces for downstream PNS and eddy-
current tools when their physical models are available.

Validate voltage/current predictions against known coupled circuits and actual
waveforms. Distinguish exported inputs from completed analysis. Net force/torque
bounds do not establish local stress, acoustic response or vibration; static
suppression targets do not establish dynamic shielding. Keep those analyses
external until their response models are validated.

## Scalable energy and topology completeness

Surface magnetic energy currently uses dense quadrature with a face-count guard.
Benchmark memory, quadrature error and repeated solves before choosing a sparse,
hierarchical or accelerated implementation. Preserve self/cross-part terms,
symmetry, positive semidefiniteness and physical normalization. Compare against
analytic coils and the dense reference over coordinate/mesh scales.

Positive-genus harmonic circulation modes are currently unsupported and rejected.
Add explicit harmonic modes only when needed, preserving boundary constants and
gauge handling. Validate independent circulation, current continuity and field
maps on known multiply connected supports before accepting that topology.

## Literature traceability

These references motivate the engineering work; implementation status is defined
by the linked API and validation documentation.

- Stream-function mesh core: Zetter et al.,
  [arXiv:2005.10056](https://arxiv.org/abs/2005.10056).
- Peak/minimax current density: Poole et al.,
  [arXiv:1002.3413](https://arxiv.org/abs/1002.3413).
- Constraint-based local gradient design: Jia et al.,
  [arXiv:2012.12813](https://arxiv.org/abs/2012.12813).
- B0-relative targets: de Vos et al.,
  [arXiv:1912.08500](https://arxiv.org/abs/1912.08500).
- PNS/vector-potential modeling: Roemer et al.,
  [arXiv:2012.08694](https://arxiv.org/abs/2012.08694).
- Waveform-dependent engineering constraints: McCready et al.,
  [arXiv:2505.07117](https://arxiv.org/abs/2505.07117).
