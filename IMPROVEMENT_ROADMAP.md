# Detailed Improvement Roadmap

This roadmap turns the literature-backed design insights into implementation
work packages. The goal is to make CoilGen a powerful, easy-to-use
magnetostatic coil design tool with project-specific studies kept outside the
core package.

## Priority Order

1. Preserve current behavior while improving reports.
   Add clearer feasibility/readiness labels, current-density metrics, and
   constraint-usage reporting before changing solver behavior.

2. Generalize target and diagnostic regions.
   Keep the current Bz-first path working, then add vector components, B0
   direction, named target groups, and general diagnostic regions.

3. Make comparisons physically fair.
   Extend the constrained optimization/reporting model so geometry sweeps are
   compared at matched useful-field error and matched secondary constraints.

4. Add engineering-screening layers.
   Add thermal/current-density screening, concomitant-field reports,
   vector-potential exports, and layout-readiness checks as explicit layers
   above the magnetostatic solve.

5. Add new optimization objectives.
   Add minimax or peak-current-density objectives after the current-density maps,
   target groups, and constraint reports are stable.

## Work Package 1: Readiness Levels And Report Language

Purpose:
Make every output clear about what has and has not been proven. A static
magnetostatic result should not read like a finished MR gradient insert.

Current code touchpoints:

- `src/coilgen/reporting.py`
- `src/coilgen/diagnostics.py`
- `src/coilgen/study.py`

Implementation details:

- Add report fields for:
  - `magnetostatic_feasibility_ready`
  - `layout_feasibility_ready`
  - `engineering_feasibility_ready`
  - `clinical_device_ready`
- Keep `clinical_device_ready = False` unless an external project explicitly
  supplies safety/regulatory evidence.
- Report missing checks as named reasons, not as a generic warning.
- In study CSV/JSON rows, keep separate flags for field quality, layout quality,
  shielding usefulness, current-density severity, and active/no-shield fairness.

Acceptance checks:

- Existing examples still produce the same core numeric results.
- Reports explain that `PASS` is a CoilGen static/layout quality pass.
- Study rows cannot be mistaken for final MR-gradient device approvals.

## Work Package 2: Current-Density And Thermal Screening

Purpose:
The minimax-current-density literature makes peak current density a first-class
engineering quantity. CoilGen already has surface current and conductor
dimensions, so the first useful step is reporting before optimization.

Current code touchpoints:

- `src/coilgen/basis.py`
- `src/coilgen/contours.py`
- `src/coilgen/visualization.py`
- `src/coilgen/engineering_metrics.py`
- `src/coilgen/reporting.py`

Implementation details:

- Report conductor current density:
  - `current_density_A_per_mm2 = abs(needed_current_layout) / area_mm2`
  - peak, median, and percentile surface-current density before contouring
  - estimated local turn spacing from `contour_step / |K|`
- Add optional user inputs:
  - `duty_cycle`
  - `pulse_duration_s`
  - `repetition_time_s`
  - `cooling_model = "none" | "lumped_proxy"`
  - warning thresholds in `A/mm^2` and average `I^2R`
- Add report values:
  - RMS current
  - peak power
  - average power
  - stored energy
  - simple temperature-rise proxy when enough inputs are present
- Add current-density plot-book pages that show hot spots on the support surface.

Acceptance checks:

- For a 4 mm x 4 mm conductor, 200 A reports 12.5 A/mm2.
- Missing duty cycle produces a clear "pulsed screening only" warning instead of
  pretending to know thermal safety.
- Current-density metrics appear in both single-case reports and study tables.

## Work Package 3: General Target Components

Purpose:
Conventional MRI often uses Bz as the useful component, but Halbach and
transverse-B0 systems need targets defined relative to the actual static-field
direction. Non-MRI field-shaping problems may need arbitrary vector targets.

Current code touchpoints:

- `src/coilgen/models.py`
- `src/coilgen/target.py`
- `src/coilgen/problem.py`
- `src/coilgen/conversion.py`
- `src/coilgen/config.py`
- `src/coilgen/config_validation.py`
- `src/coilgen/designs.py`
- `src/coilgen/evaluation.py`
- `src/coilgen/reporting.py`

Implementation details:

- Keep `target_bz_expression` as the simple default.
- Add a `FieldComponentSpec` concept:
  - `component = "Bz" | "Bx" | "By" | "BdotB0" | "vector"`
  - `b0_direction = (bx, by, bz)` for `BdotB0`
  - optional physical encoding direction for gradient linearity
- Extend imported target maps to accept vector columns:
  - `bx`, `by`, `bz`
  - `target_bx`, `target_by`, `target_bz`
  - group and weight columns as already supported
- Use the full sensitivity matrix already available in the pipeline instead of
  discarding non-Bz components during target assembly.
- Report the target convention explicitly: global Cartesian Bz, B dot b0_hat, or
  full vector residual.

Acceptance checks:

- Existing Bz examples are numerically unchanged.
- A rotated-Halbach-style test gives the same result when the geometry and
  `b0_direction` are rotated together.
- A vector imported target can constrain at least two components at the same
  sample points.

## Work Package 4: Named Target, Suppression, And Diagnostic Regions

Purpose:
The software needs a general way to describe not just "the DSV" and "the
suppression surface", but many named regions with different roles. This supports
active shielding, local nonlinear gradients, shim targets, and non-optimized
diagnostic regions through one data model.

Current code touchpoints:

- `src/coilgen/models.py`
- `src/coilgen/problem.py`
- `src/coilgen/target.py`
- `src/coilgen/optimization.py`
- `src/coilgen/reporting.py`
- `src/coilgen/study.py`

Implementation details:

- Add named region roles:
  - `useful_field`
  - `suppression`
  - `diagnostic_only`
  - `engineering_limit`
- Store per-region:
  - points
  - target field or target component
  - weights
  - hard constraint limits where applicable
  - metric names and units
- Export region-specific metrics:
  - RMS residual
  - peak residual
  - percentile residual
  - achieved/allowed constraint ratio
  - whether the region was optimized or diagnostic-only

Acceptance checks:

- Active shielding reports DSV error and suppression RMS as separate region
  metrics.
- A diagnostic-only region can be evaluated without affecting the solve.
- Study tables can compare designs using region-specific constraint usage.

## Work Package 5: Fair Constraint-Based Optimization

Purpose:
For geometry sweeps, a design should not look better simply because it uses more
of the allowed field-error budget. Comparisons should be made at matched DSV
error and matched secondary constraints.

Current code touchpoints:

- `src/coilgen/optimization.py`
- `src/coilgen/optimization_schemes.py`
- `src/coilgen/config_validation.py`
- `src/coilgen/reporting.py`
- `src/coilgen/study.py`

Implementation details:

- Keep `constrained_quadratic` as the preferred mode for full geometry studies.
- Generalize the current separate DSV/suppression model to any named region:
  - useful-field residual limit
  - suppression RMS limit
- Report active constraint usage:
  - `dsv_error_usage_percent`
  - `suppression_rms_usage_percent`
- Add an active/no-shield fairness flag based on matched useful-field error.
- Full-study entry points should default to fixed-error constrained comparison,
  not arbitrary L-curve comparison.

Acceptance checks:

- Geometry sweeps state whether active and no-shield rows are in the same DSV error
  band before comparing efficiency.
- Rows that meet an absolute suppression limit but are worse than the unshielded
  case are labeled as limit-only, not useful shielding.
- Failed constraints are recorded as structured failure reasons in CSV/JSON.

## Work Package 6: Concomitant Field Reports

Purpose:
MRI gradients produce Maxwell-required concomitant fields. Core CoilGen should
report these as general MR-gradient screening quantities from the fitted
gradient, configured B0 strength, and B0 direction. Application-specific metrics
belong in project folders, not core CoilGen.

Current code touchpoints:

- `src/coilgen/engineering_metrics.py`
- `src/coilgen/reporting.py`
- `src/coilgen/config.py`
- `src/coilgen/config_validation.py`
- `src/coilgen/designs.py`
- `src/coilgen/study.py`

Implementation details:

- Add a meaningful object-oriented report class with attributes and methods:
  - `ConcomitantFieldReport.from_linear_gradient(...)`
  - `ConcomitantFieldReport.field_values(...)`
  - `ConcomitantFieldReport.as_dict(...)`
- Use the fitted connected-layout gradient at `needed_current_layout`.
- Use configured `b_0_strength_T` and `b_0_direction`.
- Report RMS, 95th percentile, and peak concomitant field over the DSV.
- Report ppm relative to B0 and percent relative to target peak field.
- State the model assumption in the report.

Acceptance checks:

- A pure x-gradient with samples at `z = +/-0.1 m`, `Gx = 2 T/m`, and `B0 = 1 T`
  reports a peak concomitant field of `0.02 T`.
- Existing Bz examples remain numerically unchanged except for added report
  fields.
- The Markdown report and CSV summary contain the same concomitant-field
  metrics.

## Work Package 7: Minimax Or Peak-Current-Density Optimization

Purpose:
Resistance minimization can reduce total power while still creating local hot
spots. A minimax objective directly penalizes the worst current-density region
and should produce more manufacturable, thermally balanced layouts.

Current code touchpoints:

- `src/coilgen/basis.py`
- `src/coilgen/optimization.py`
- `src/coilgen/optimization_schemes.py`
- `src/coilgen/config.py`
- `src/coilgen/config_validation.py`
- `src/coilgen/reporting.py`

Implementation details:

- Add an optimization scheme such as:
  - `peak_current_density_constrained`
  - `minimax_current_density`
- Start with a CVXPY-backed formulation:
  - minimize peak surface-current norm or a weighted combination of peak current
    density and resistance
  - constrain useful-field residual and optional suppression regions
- Add a dependency-free fallback only if it can be made honest, for example a
  scan over Tikhonov/smoothness factors with peak-density reporting.
- Report whether the peak-density objective was exact, approximate, or fallback.

Acceptance checks:

- On a test mesh, the minimax mode lowers peak current density compared with
  pure resistance minimization at matched target error.
- If CVXPY is missing, the user gets a clear message and the report does not
  claim minimax optimization was performed.
- Peak-density improvements are reported together with any efficiency,
  resistance, or inductance penalty.

## Work Package 8: MR Gradient Operation Exports

Purpose:
CoilGen should remain a static magnetostatic design tool, but it should export
enough data for downstream MR-gradient engineering: waveform design, PNS,
voltage, acoustic, force, and eddy-current analysis.

Current code touchpoints:

- `src/coilgen/field_maps.py`
- `src/coilgen/engineering_metrics.py`
- `src/coilgen/inductance.py`
- `src/coilgen/reporting.py`
- `src/coilgen/visualization.py`

Implementation details:

- Export standard tables for:
  - conductor path points and segment current
  - field maps per ampere
  - gradient impulse response placeholders when external data exists
  - resistance
  - inductance
  - stored energy per target gradient
  - voltage estimate for target slew rate
- Add vector-potential map export hooks for PNS tools.
- Keep PNS as an external/downstream result until a validated E-field model is
  implemented.
- Add report fields for "export available" vs "engineering analysis completed".

Acceptance checks:

- A downstream tool can consume field and electrical CSV/NPZ outputs without
  parsing Markdown reports.
- Reports do not call PNS, acoustic, or eddy-current behavior complete unless
  those analyses are actually supplied.

## Work Package 9: Layout And Manufacturability Closure

Purpose:
A contour-level design is not a build-ready layout. The routing, terminal,
clearance, and mechanical assumptions need stronger checks before final design
claims.

Current code touchpoints:

- `src/coilgen/layout.py`
- `src/coilgen/routing.py`
- `src/coilgen/manufacturing.py`
- `src/coilgen/pcb.py`
- `src/coilgen/reporting.py`

Implementation details:

- Add explicit terminal-pad and cable-exit definitions.
- Validate terminal-wire fields separately from closed-loop contour fields.
- Add minimum bend radius and connector clearance checks.
- Report which field source was used:
  - stream-function continuous surface current
  - unconnected contour loops
  - connected final wire path
  - connected final wire path plus terminal leads
- Add Lorentz-force screening hooks when B0 direction and conductor path exist.

Acceptance checks:

- A final design cannot be `layout_feasibility_ready` unless routed terminals are
  included or explicitly waived.
- Study sweeps that use unconnected contours are labeled as contour-level
  screening.
- Layout snapshots and route diagnostics identify the exact reason for warnings.

## Work Package 10: Application Templates

Purpose:
Users should see CoilGen as a general tool with physically meaningful templates.

Current code touchpoints:

- `projects/general_examples/`
- `projects/primitive_support_structures/`
- `src/coilgen/designs.py`
- `src/coilgen/workflow.py`

Templates to add or strengthen:

- conventional cylindrical MRI gradient
- open biplanar gradient
- Halbach/transverse-B0 gradient
- shim coil with imported measured field target
- local nonlinear gradient
- active-shielded gradient with fixed suppression limit
- arbitrary imported STL support with imported target map
- project-specific extension examples outside the core package

Acceptance checks:

- Every template has one short user-editable script.
- Every template writes a design report and machine-readable summary.
- Template docs state whether the result is magnetostatic feasibility, layout
  feasibility, or a downstream engineering study.

## Suggested Milestones

### Milestone 1: Better Screening Reports

Deliver:

- readiness levels in reports and study rows
- current-density `A/mm2` metrics
- concomitant-field reports for MRI-gradient screening
- constraint-usage reporting
- clean full-study interpretation

Why first:
This improves technical honesty without changing the core inverse solver.

### Milestone 2: Region Model And Diagnostic Groups

Deliver:

- named target/suppression/diagnostic regions
- fair active/no-shield comparison flags

Why second:
This creates reusable infrastructure for project-specific applications while
keeping application-specific metrics outside the core package.

### Milestone 3: Vector Targets And B0 Direction

Deliver:

- `B dot b0_hat` target support
- vector imported target maps
- transverse-B0 and Halbach-style validation examples

Why third:
This makes CoilGen credible for portable MRI and nonstandard magnet geometries.

### Milestone 4: Engineering Exports

Deliver:

- field/electrical export package
- vector-potential export hooks
- voltage/slew tables
- downstream analysis status fields

Why fourth:
This keeps CoilGen static and focused while making it useful to MR-gradient
engineers.

### Milestone 5: Peak-Density Optimization

Deliver:

- CVXPY-backed minimax or peak-current-density mode
- peak-density plots and metrics
- comparison examples against resistance minimization

Why fifth:
This is solver-facing and should be built after target regions and current
density reporting are reliable.

## Literature Traceability

- Stream-function mesh core and general tool design: Zetter et al.,
  https://arxiv.org/abs/2005.10056.
- Peak/minimax current density: Poole et al.,
  https://arxiv.org/abs/1002.3413.
- Constraint-based local gradient design: Jia et al.,
  https://arxiv.org/abs/2012.12813.
- B0-relative targets for Halbach-style MRI: de Vos et al.,
  https://arxiv.org/abs/1912.08500.
- PNS/vector-potential direction for gradient engineering: Roemer et al.,
  https://arxiv.org/abs/2012.08694.
- Waveform-dependent voltage, PNS, acoustic, and mechanical constraints:
  McCready et al., https://arxiv.org/abs/2505.07117.
