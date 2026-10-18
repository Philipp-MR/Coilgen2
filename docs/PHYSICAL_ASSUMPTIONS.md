# CoilGen Python Physical Assumptions

This document summarizes the assumptions that matter when interpreting or extending
the Python CoilGen design workflow.

## Stream Function and Current

- The optimized stream function is defined on an orientable 3D coil surface.
- Stream-function values have units of ampere (A).
- Adjacent contour levels differ by the current carried by one wire turn.
- All primary and active-shield contour turns are driven with the same current
  magnitude. Active-shield layers may therefore receive fewer or more contour
  levels than the primary layer only when the shared current step is preserved.
- Contour current direction is validated against the mesh normal and local
  stream-function gradient. The direction is not arbitrary.

## Target Field

- Bz is the default target component. The core also supports Bx, By,
  `B dot b0_hat`, and vector-field targets through the field-component API.
- For transverse-B0 or Halbach systems, specify the useful component and B0
  direction explicitly. Some gradient-efficiency and engineering reports still
  fit Cartesian Bz; support in the optimizer does not establish that every
  downstream report uses the same component.
- Symbolic target expressions such as `x`, `y`, or `z` describe the requested Bz
  spatial dependence before scaling to `target_gradient_strength`.
- CSV/JSON/NPZ target-field imports are interpreted in SI units: coordinates in
  metres and magnetic field in tesla before the configured scale factor is
  applied.
- Secondary target samples are zero-field suppression targets. They are used for
  active shielding by assigning a nonzero `secondary_target_weight`.

## Active Shielding

- Shield surfaces are separate conductor layers outside the primary surface.
- The shield current is not independently scaled in postprocessing; it is part of
  the same stream-function optimization and shares the same contour current step.
- Suppression performance should be evaluated on the same external target surface
  for shielded and unshielded comparisons. In unshielded references, the same
  suppression samples may be present with zero weight so the field is measured
  but not optimized.
- Active shielding is sensitive to regularization. With same-current primary and
  shield channels, the shield strength is realized by the number of shield
  contours. Too little regularization can create oscillatory shield stream
  functions and fragmented contour topology. Too much can shrink the shield
  stream-function range, produce too few shield turns, and increase the realized
  fringe field even though a shield layer is present. Use quality-matched
  shielded/unshielded comparisons and inspect routed layouts before claiming a
  useful shield.

## Fixed-Error Stream-Function Optimization

For geometry-comparison studies with active shielding, the physically preferred
formulation separates useful-field accuracy from fringe-field suppression:

```text
minimize_x  x^T R x + eta ||L psi||_2^2
subject to  ||A_dsv x - b_dsv||_2 <= eps_dsv ||b_dsv||_2 + delta_dsv
            ||A_sup x||_2 / sqrt(N_sup) <= B_sup,rms,max
```

`x` is the reduced stream-function vector, `A_dsv` maps `x` to Bz on the
internal DSV target samples, `A_sup` maps `x` to Bz on the external suppression
surface, `R` is the quadratic resistance matrix, and `L` is the optional surface
smoothness operator. Using `||R x||^2` instead would apply `R^T R` and change the
physical objective. This example uses Bz; generalized targets use their selected
field component. The suppression bound is an RMS Bz amplitude in tesla and is
therefore independent of the number of suppression samples. If no RMS field
limit is configured, CoilGen keeps the older residual-norm form
`||A_sup x - b_sup||_2 <= eps_sup ||b_dsv||_2 + delta_sup` for compatibility.

The legacy weighted-combined residual
`||sqrt(W) (A x - b)||_2 <= epsilon ||sqrt(W) b||_2` is still available with
`target_error_constraint_mode="combined"`, but it should not be used for
paper-style geometry comparisons where DSV linearity and active shielding must
be interpreted separately.

## Symmetry

- Symmetry checks are diagnostic unless `contour_symmetry_validation="strict"` or
  `symmetry_reduction="strict"` is selected.
- Generated mirror-pair candidates are known for biplanar, circular biplanar,
  flanged split-bore, and their active-shielded variants.
- Symmetry reduction is used only after these checks pass:
  - generated geometry has a valid mirrored vertex mapping,
  - target samples, weights, groups, and suppression flags are mirror paired,
  - target Bz is mirror-even or mirror-odd,
  - sensitivity columns admit a consistent stream-function sign relation.
- Be careful with imported meshes, asymmetric target maps, asymmetric shielding
  weights, mesh rotations, and postprocessing choices. Any of these can break the
  assumptions needed for symmetry reduction.

## Contours and Topology

- Loop topology is built intrinsically on the 3D surface. UV coordinates are not
  used.
- Contours are extracted by linear interpolation across mesh triangle edges, so
  visual smoothness depends on mesh density and contour resampling.
- Minimal contour distance and inter-group connector selection can use surface
  geodesic distances, but direct 3D cuts remain the default where connector
  orientation matters for the field.

## Field Evaluation

- Thin-wire field evaluation uses straight wire segments.
- Long segments should be evaluated with the full finite straight-wire expression
  rather than a point-current approximation.
- Reported gradient efficiency is the fitted amplitude of the requested Bz
  target mode per ampere in the internal target region. For example, for
  `Bz = G*y`, CoilGen fits the coefficient of the `y` mode and reports it in
  `mT m^-1 A^-1`.
- Uniform Bz offsets and field components orthogonal to the requested target
  mode do not increase gradient efficiency. They are reported through the field
  error metrics.
- Suppression target samples are excluded from the internal gradient-efficiency
  fit. They are evaluated separately as fringe-field or active-shielding
  metrics.
- `needed_current_layout` is the signed current needed to reproduce the target
  mode. A negative value means the layout current polarity is opposite to the
  requested target convention.

## Active-Shielded Study Metrics

- Internal gradient efficiency quantifies useful target-mode amplitude per
  ampere inside the DSV or volume of interest.
- Internal field linearity/error quantifies residual deviation after scaling the
  field to the target mode. Orthogonal Bz components increase this error.
- Shielding efficiency compares shielded and unshielded fringe fields on the
  same external suppression surface, normally at each design's current required
  for the requested internal gradient.
- Connected-wire evaluation uses the final routed wire path. When
  `skip_postprocessing=True`, field and optional inductance are evaluated from
  unconnected contours; this is appropriate for fast parameter sweeps but not
  for final manufacturable layout review.

## Current Density, Heating, and Pulsed Operation

- Resistance and inductance estimates are static electrical estimates. They do
  not include duty cycle, cooling, insulation, conductor temperature, or
  proximity/skin effects.
- Current density can be estimated when the fitted current and conductor
  cross-section are known. High `A/mm^2` values should be interpreted as
  pulsed-gradient screening results unless a thermal model and duty cycle are
  supplied.
- Resistance minimization reduces global Ohmic power, but it does not guarantee
  that local current-density hot spots are acceptable. Facewise vector minimax
  objectives and current-density limits are available. These constrain the
  continuous sheet, not connector hot spots or final wire temperature.
- The opt-in [physical API](RESEARCH_PHYSICS_API.md) supports sheet magnetic
  energy and prescribed-background net force/torque constraints. Its physical
  current is preserved through contour/wire evaluation. Net load balance does
  not establish local stress, vibration or full-circuit construction readiness.

## MR Gradient Engineering Checks Outside The Static Model

CoilGen currently designs static conductor layouts and evaluates magnetostatic
fields. A finished MR gradient coil also needs time-domain and system checks:

- concomitant-field reports use a small-field Maxwell approximation from the
  fitted linear gradient and configured B0; they are screening diagnostics, not
  a replacement for sequence-specific correction or safety analysis;
- amplifier voltage and slew-rate margins;
- peripheral nerve stimulation from induced electric fields;
- acoustic noise and mechanical resonance;
- local Lorentz-force stress and structural response in the B0 field;
- eddy-current coupling into cryostat, magnet, RF shields, and patient-support
  structures;
- cryostat heating and conductive shielding losses;
- terminal routing, cable forces, insulation, cooling, and service access.

Reports should therefore distinguish a CoilGen static quality pass from a
complete engineering or clinical-device readiness claim.

## Current Limitations

- Automatic whole-wire rerouting is not implemented. CoilGen detects and
  reports close wire contacts and can normal-shift return segments, but final
  clearance violations still require inspection or manual parameter adjustment.
- Vector and B0-projected optimization are implemented, while many convenience
  APIs and gradient engineering reports remain Bz-first.
- Intrinsic contour topology assumes an orientable surface with usable mesh
  normals. Non-orientable surfaces are outside the intended design domain.
- Surface geodesic routing currently uses Dijkstra-style mesh-graph distances.
  This is robust and dependency-light, but not a continuous Fast Marching or
  Heat Method solver.
- Contour smoothness is limited by mesh resolution because contours are
  piecewise-linear intersections with mesh triangles. Use finer meshes for
  smoother final contours.

## Performance

- Parallelism is used where the work is naturally chunked, especially sensitivity
  and Biot-Savart evaluations.
- Symmetry reduction can reduce the optimization unknown count for validated
  symmetric generated problems, but it must not be forced on asymmetric geometry
  or asymmetric imported target fields.
