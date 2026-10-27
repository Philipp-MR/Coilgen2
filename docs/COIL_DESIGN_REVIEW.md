# CoilGen review: numerical reliability, conductor sweeps, and winding

Review date: 2026-09-07. This review preserves the pre-existing work in the
checkout. It covers the project architecture and regression suite, with deeper
inspection and independent reviews of the numerical core, manufacturing sweeps,
interconnection, PCB export, provenance, and parameter studies. It is not a
claim that every project-specific dataset or every code path was exercised.

Update 2026-09-09: the mounted fabrication path, cross-body checks, physical
solver and runtime improvements have been integrated into main without the GUI.
This document retains the original findings and priorities; consult
[MAIN_CORE_IMPROVEMENTS.md](MAIN_CORE_IMPROVEMENTS.md) for current implementation
status, validation and remaining limitations.

## Main engineering conclusion

The next major improvement should be a fabrication path that is shared by
geometry export, field evaluation, electrical estimates, and winding instructions.
CoilGen already has useful surface topology, contour provenance, adaptive
sampling, transported sweep frames, and geometric diagnostics. The remaining
integration gaps can produce a plausible-looking conductor that differs from
the path used to report its performance.

For the designer, the desired outcome is: **a known conductor on a known former,
with reachable terminals, a feasible winding order, and validated fields from
the complete built circuit**. The most useful next work is ordered below.

## Corrections made in this review

| Area | Confirmed error and correction | Engineering impact |
| --- | --- | --- |
| Sweep clearance | Insulation/tolerance envelope violations were computed but omitted from `manufacturing_ok`; now included. | A copper-only gap cannot masquerade as sufficient insulated-wire spacing. |
| Fine geometry | Segment-distance and triangle-intersection predicates mixed length, area, and volume tolerances; fixed dimensionally inconsistent cutoffs. | Intersections remain detectable for micrometre-scale segments and triangles. |
| Mounted geometry | Capsule checks used the unmounted route; they now follow the mounted profile origin. | Inner/outer mounting offsets participate in collision screening. |
| Surface routing | A point could have a positive distance to itself, and two points on one triangle routed through a mesh vertex; added exact local paths. | Removes avoidable connector detours without crossing holes or disconnected surfaces. |
| Circuit closure | Greedy fusion bypassed strict surface routing for its final return; closure now uses the selected routing method. | The last bridge receives the same surface constraint as earlier bridges. |
| Provenance | Geometric nearest-segment matching confused coincident/retraced segments after duplicate removal. | Segment ownership survives unchanged/reversed paths and exact duplicate removal. |
| PCB units | DXF coordinates were in mm while `$INSUNITS` specified metres; corrected the unit declaration. | Eliminates a 1000-fold CAD scale error. |
| PCB geometry | Short cylinders could acquire a radial axis from positional PCA; use side-wall normals where applicable. A length heuristic also deleted valid long connectors; removed that split. | Improves cylinder unwrapping and preserves artwork continuity. |
| Shim optimization | Absolute solver tolerances acted on very small tesla-valued objectives; normalize the entire least-squares system by one shared scalar. Eliminate fixed-current variables exactly. | Identical physical fits retain their optimum when field magnitudes change; disabled channels are supported. |
| Electrical area | MATLAB-style `2 x N` profile coordinates were ignored before transposition; corrected area extraction. | Custom conductors have consistent resistance/power estimates in both array conventions. |
| Acceptance | Non-finite active layout/target samples were silently dropped; they now fail an enabled layout target check. Missing required hardware measurements and empty part lists now create violations. | Missing results cannot satisfy a requested engineering limit. |
| Parameter studies | Failed/malformed cache entries were reused; cache identity omitted resolved designs and input contents; mixed cached/new rows lost grid order. Corrected these and excluded failed/unknown designs from default `best_by`. | Studies retry failed cases, invalidate changed standard inputs, preserve order, and avoid recommending known failures. |
| Validation environment | The local editable install pointed to a sibling checkout. Reinstalled this repository and added `pythonpath = ["src"]` to pytest. | The test command now exercises the code being edited. |

The numerical normalization preserves the least-squares objective up to a
positive scalar; it does not introduce a different optimization objective.
One reproducer previously changed a bounded optimum from -2 A to -1 A simply
by scaling the system to microtesla values. A power-limited -0.7071 A optimum
was reported near -0.0000144 A with an optimal status. The new regressions
exercise these cases and fixed-channel bounds.

## Highest-priority remaining findings

### 1. Swept bodies can disagree with the reported conductor path

In `manufacturing.py::_sweep_wire_path_with_diagnostics`, fillets, surface
projection, adaptive sampling, and mounting produce the swept geometry, but
the function returns a mesh and diagnostics. It does not preserve that final
centerline as a separate fabrication object. `save_wire_trace_as_csv`, STEP
centerline export, and the layout field/electrical calculations still consume
the earlier `part.wire_path`.

**Required change:** preserve a `FabricationPath` with the actual conductor
centroid, profile orientation, units, coordinate frame, segment provenance, and
mounting/layer assignment. Keep the optimized and routed paths available for
comparison. Generate manufacturing exports from the fabrication object and
recompute field, resistance, and inductance from it.

**Acceptance:** export and reopen the path; match its coordinates to the sweep
centroid and recalculate the field at the same operating current. Report the
incremental field changes from contouring, interconnection, smoothing, mounting,
and lead routing. Any current refit should be shown separately so it cannot hide
a geometric change.

### 2. Independent bodies are not validated as a complete assembly

`create_swept_contour_meshes` validates each body separately;
`_aggregate_swept_body_diagnostics` aggregates their checks. It does not test
all contour/body pairs or all coil parts against each other. A reproducer with
concentric 1.000 m and 1.010 m radius loops and 50 mm-wide conductors gives
`manufacturing_ok=True` despite their overlapping volumes.

**Required change:** one assembly validator spanning independent turns,
primary/shield parts, coil channels, return conductors, insulation envelopes,
former surfaces, and keep-out regions. Begin with bounding-volume screening,
then use profile/triangle tests for candidate pairs. Include containment cases,
where one closed volume overlaps another without intersecting its boundary.
Report contact coordinates and body/segment identifiers.

**Acceptance:** crossing, touching, nested, and nearly separated bodies at
multiple length scales; insulation-only overlap; disconnected parts; and a
three-axis assembly. Per-body validity must remain distinct from assembly
clearance validity.

### 3. Terminals and winding order need an explicit circuit model

The layout cleanup closes wire paths. Existing terminal descriptions/counts
do not place terminals or generate a complete supply/return pair. A closed
mathematical route is insufficient to describe how wire reaches the amplifier
or how the assembly is wound.

**Required change:** model terminal locations, polarity, lead stubs, permitted
return corridors, and segment roles. Preserve contour current direction while
choosing cuts. Evaluate the complete current circuit, including paired leads,
at the requested operating current.

For manual winding, I would use this build-up:

1. Specify wire stock, insulation, minimum bend radius, former clearance, and
   available layers. Distinguish conductor dimensions from groove/tool dimensions.
2. Choose a reachable terminal zone and a reserved return corridor before
   opening contours. Protect mounting holes, service access, and cable exits.
3. Form continuous spirals within compatible contour groups, preserving their
   current orientation. Prefer cuts sharing an accessible corridor where the
   resulting field error and bend geometry are acceptable.
4. Compare a small number of complete interconnection alternatives. Apply hard
   clearance, bend, continuity, and field-error limits first; then compare
   crossover count, extra wire length, assembly effort, and repair access.
5. Convert over/under crossover relations into a winding-order graph. Reject a
   cycle that would require installing a segment beneath already fixed wire;
   reroute, change layer, or introduce a deliberately specified joint.
6. Export numbered turns and groups, direction arrows, IN/OUT polarity, layer
   and crossover heights, cumulative wire length, and accessible fixation points.
   Include a dimensioned flat template for cylinders and a stepwise 3D view for
   arbitrary supports.

An accessible crossover or joint can be preferable to a slightly shorter route
that cannot be wound. Which joints are permitted should be a designer choice.
Spacing-based routing exemptions should cover only the local contact interval;
excluding an entire endpoint contour can hide a collision with its remote side.

### 4. PCB output needs a defined seam and electrical connectivity

`generate_cylindrical_pcb_print` unwraps unordered mesh vertices as a continuous
path to compute the boundary. On a 150 mm-radius cylinder with 32 angular and
six axial cells, the resulting boundary width was 6.5679 m; one circumference
is 0.94248 m. Correcting the DXF units and short-cylinder axis does not fix this
boundary/seam problem.

The default two-layer export uses group spirals, but does not represent all
final inter-group bridges as nets with vias and pads. It should be treated as
artwork for inspection until that circuit is implemented.

**Required change:** a fixed angular seam, periodic clipping with explicit
edge connections, a boundary obtained from support topology, consistent layer
coordinates, and a net/via/pad model. Verify electrical continuity after export
and after importing into CAD. Profile offsets at sharp track corners also need
robust polygon handling.

## What makes a high-quality conductor sweep

The existing transported frames and adaptive sampling are a useful foundation.
Increasing point count alone cannot fix an infeasible bend or a bad circuit.

| Requirement | Implementation direction | Validation |
| --- | --- | --- |
| Controlled path accuracy | Bound deviation from the accepted fabrication curve in metres; preserve cuts, terminals, and crossover anchors. | Compare a finer discretization and report maximum path deviation and added field error. |
| Bend feasibility | Enforce the actual wire/strip bend radius before meshing; distinguish bending from twist for rectangular stock. | Report minimum radius and required/local radius ratio by segment; reject unresolved corners. |
| Stable profile orientation | Preserve a continuous frame and periodic closure; bound frame rotation and twist per length. | Straight, circular, helical, saddle, and closed nonplanar paths; check orientation at the closing seam. |
| Valid solid boundary | Keep oriented manifold faces, nondegenerate triangles, closed ends where appropriate, and positive volume. | Reimport the exported mesh; inspect topology and volume conservation under refinement. |
| Physical clearance | Separate copper, insulation, manufacturing tolerance, and former/groove clearances. Use profile-aware narrow checks after capsule screening. | Distinguish conservative capsule overlap from confirmed conductor collision; validate all bodies together. |
| Electrical fidelity | Calculate from the same mounted path used to build the solid, including leads. | Compare B, R, L, and required current before and after fabrication conditioning. |
| Actionable diagnostics | Store violating segment/ring IDs, world coordinates, severity, and required correction. | A designer can select the failure in the viewer and locate it in the winding instructions. |

For asymmetric profiles, the profile origin and current centroid are different
concepts. Clearance envelopes can be based on an enclosing origin-centred
radius, while thin-wire electrical evaluation should use the physical current
centroid appropriate to its assumptions. A faceted STEP export also remains a
faceted boundary; file extension alone does not establish an analytic CAD model.

## Parameter studies and numerical confidence

The sweep cache now fingerprints core source, resolved settings, and standard
mesh/target/preoptimized input contents. Only successful execution rows resume;
engineering quality is still a separate status. Set `cache_revision` for custom
metric/factory dependencies and external solver changes. Anonymous in-memory
registrations with new UUIDs conservatively miss the cache; use stable registry
keys when repeated registration is intended. Cached rows do not recreate
missing exported artifacts or retain full result objects.

For useful design studies:

- Compare at matched useful-field error, shielding limits, target samples,
  conductor assumptions, and operating conditions. Keep optimization samples
  separate from a denser validation set, especially near ROI boundaries.
- Use cheap continuous/contour screening, then rerun shortlisted candidates with
  interconnections, mounted geometry, all channels, and leads. Label the stage
  in every row; do not compare a contour-only field with a complete circuit
  as if both are equally realizable.
- Report feasible candidates and Pareto comparisons across field error, power,
  current, clearance, winding complexity, and efficiency. Use `best_by` only
  after those requirements are applied. Its default admits warnings, which
  still need inspection; `allowed_statuses=("PASS",)` is stricter.
- Check mesh, quadrature, contour-level, and conductor-path convergence
  separately. Increasing all four together obscures the source of error.
- Save effective inputs, source revision, input hashes, solver/backend versions,
  accepted constraints and units with finalists. Preserve failed cases and
  reasons; a sweep with absent cases is not complete evidence of feasibility.

Further numerical findings needing deliberate follow-up:

1. `optimization.py::_target_error_solution_meets_constraint` only reads target
   residual acceptance, yet contributes to `constrained_solver_contract_met`.
   The complete acceptance predicate should also recompute every enabled
   current-density, stream-function-range, and shield constraint in physical
   units. A residual pass is not proof of the full solver contract.
2. `_target_projection_current` ignores sample weights, although the mean-free
   shim projection has a weighted path. With `[1,1]` microtesla/A fields,
   `[0,10]` microtesla targets, and weights `[100,1]`, unweighted projection
   gives 5 A while weighted least squares gives 0.0990099 A. Define one clear
   fitting contract across modes before changing existing output conventions.
3. Hardware aggregation can ignore non-finite entries when other parts have
   finite values. The new checks reject wholly unavailable required metrics;
   full validation still needs complete part coverage, including leads.
4. Self-inductance needs a convergence-qualified discretization. For a 100 mm
   radius loop with 1 mm wire radius, the current implementation gave roughly
   0.5665, 0.5310, and 0.5502 microhenry at 32, 512, and 1024 segments. Its
   softened midpoint interaction and segment self-term require particular care
   when segment lengths approach the conductor radius. Do not rank close
   candidates by this estimate without a convergence study and stated model.

## Coverage and implementation order

The review followed geometry/coordinate handling, basis and field operators,
optimization/constraints, contour extraction and sparsification, topology,
interconnection, manufacturing/export, evaluation, engineering reports,
project APIs, and study persistence. Generated examples and regression tests
provide broad coverage. Large measured datasets, full-resolution
production designs, and downstream CAD/manufacturing were not run.

Preserve the existing library/project separation. Three small domain objects
would support the next stage without introducing a large framework:
`WindingSpec` for build constraints, `FabricationPath` for accepted geometry and
provenance, and `WindingPlan` for the circuit and ordered assembly instructions.

Recommended implementation order:

1. Persist fabrication paths and perform assembly-wide clearance checks.
2. Recompute field/electrical metrics from that geometry and unify acceptance.
3. Add physical terminals, paired leads, and winding-order constraints.
4. Generate fabrication instructions and correct periodic PCB/net export.
5. Add bounded routing alternatives and Pareto selection on validated designs.

The existing regression suite should remain the baseline throughout. Add
generated buildability cases with known turn directions, lead locations, and
crossovers so improvements are reproducible without private input data.

## Validation of this change set

- Full suite on this checkout: **669 passed, 1 skipped, 26 subtests passed**
  in 56.04 seconds. The skipped module requires unavailable optional `h5py`.
- `ruff check src tests projects`: passed.
- `git diff --check`: passed; Git emitted Windows line-ending notices.
- `ruff format --check src tests projects`: still reports the same 27 files
  needing formatting as at the start of review. Broad formatting was left
  separate from the numerical and manufacturing corrections.
- Ten existing Matplotlib pending-deprecation warnings remain.
- The earlier test run imported a sibling checkout and is not used as evidence
  for this change set. The editable environment now resolves to this repository,
  and pytest explicitly includes this repository's `src` directory.
- Large production builds, physical winding, and CAD round-trip validation
  remain outside this code-review run. The remaining findings above are not
  presented as repaired or validated fabrication capabilities.
