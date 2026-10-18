# Result Interpretation

CoilGen reports magnetostatic feasibility. A `PASS` means the configured static
field, routing, and manufacturing diagnostics passed the checks implemented in
the current model. It is not a finished hardware-readiness certificate.

## Efficiency

Gradient efficiency is the fitted useful target mode per ampere. For a linear
gradient target, the common unit is `mT m^-1 A^-1`. Higher efficiency means less
current is needed for the same target gradient, provided the field quality is
comparable.

## Needed Current

Needed current is the current required to reach the configured target scale
using the fitted target mode. It should be interpreted with the same field
source as the report row: stream function, unconnected contours, or connected
layout.

## DSV Error

DSV error reports how well the evaluated field matches the requested target
inside the design spherical volume. Do not compare efficiencies between designs
unless the DSV error or linearity metric is matched closely enough for the
study question.

## Active Shielding

Active shielding should be judged by an explicit fringe-field metric on the
suppression/evaluation surface. A design can have good internal efficiency and
still be a poor shield. If a residual fringe metric is greater than the
unshielded reference, the shield failed for that metric.

The suppression-clearance report is a geometry check on generated zero-field
target points. A `PASS` means the generated suppression points stayed at or
above the configured minimum distance from the source mesh. A `WARNING` or
`FAIL` means the active-shield target may be contaminated by near-field samples
too close to the conductor/support geometry.

## Concomitant Fields

The concomitant-field report is a Maxwell small-field screening estimate from
the fitted connected-layout gradient, configured main-field strength, and
configured main-field direction. It is useful for early MR-gradient screening,
but it does not replace sequence-specific image-error or safety analysis.

## Current Density

The current-density report divides the fitted current by the configured
rectangular conductor cross section. It does not include duty cycle, cooling,
insulation, temperature rise, or conductor material limits beyond the configured
geometry.

## Why PASS Is Not Finished Readiness

The standard report explicitly lists engineering checks not included. These
typically include thermal duty cycle, driver voltage/current margins, slew-rate
validation, PNS, acoustic response, Lorentz forces, eddy currents, heating in
nearby conductors, mechanical stress, and terminal routing validation.

Use `PASS` as a model-level screening result. Use the readiness section to see
what remains before claiming a final hardware design.
