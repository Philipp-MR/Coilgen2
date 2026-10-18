# Diagnostics

## Design Report

Standard examples write:

- `*_design_report.md`
- `*_design_summary.csv`

The report contains result status, warnings, failures, numerical summary, field
error, and timing.

## Route Diagnostics

Route diagnostics are stored on each `CoilPart.route_diagnostics`. Important
fields:

- `route_quality_status`: `pass`, `warn`, or `fail`.
- `min_wire_wire_distance`: closest whole-wire segment distance in meters.
- `degenerate_opening_cut_count`: invalid contour openings.
- `suspicious_inter_group_connector_count`: unusually long or distorted
  inter-group connectors.
- `wire_clearance_warning_count`: near contacts in the final full wire path.

## Swept-Body Manufacturing Diagnostics

`CoilPart.manufacturing_diagnostics` separates route, loft, mesh, collision,
and support failures. `manufacturing_ok` is true only when all required gates
pass. `manufacturing_failure_reasons` is a semicolon-separated summary suitable
for export logs.

Important groups include:

- adaptive sampling: inserted points, final maximum segment length, chord
  correction, frame rotation, and unresolved/depth-limit counts;
- bend construction: total detected/resolved corner counts, angle-, local
  radius-, and physical-window candidate counts, geometry-transition corners,
  trim conflicts, required/effective radius, undersized sampled curves, and
  curved-surface corners requiring intrinsic rerouting;
- section conservation: nominal and ring cross-sectional areas, maximum ring
  area error, expected `area * profile-centroid-path-length` volume, and final
  mesh volume error;
- loft quality: alternate diagonal count, folded quads, worst adjacent-face
  normal dot product, and maximum triangle aspect ratios;
- topology: short/open/non-manifold/misoriented edges and signed volume;
- collision: centerline capsule-BVH clearance plus final triangle-BVH
  self-intersection;
- support conformity: centerline support-domain residual and outer/inner
  conductor penetration.

Strict STL and STEP exports reject a part when `manufacturing_ok` is false.
Diagnostic JSON/CSV should therefore be saved before attempting the strict
geometry export.

## Field Diagnostics

`result.error_vals` reports relative percent errors for:

- scaled connected wire layout vs target
- scaled unconnected contours vs target
- scaled connected layout vs optimized stream-function field
- scaled unconnected contours vs optimized stream-function field

The scaling uses `result.needed_current_layout`, which is the signed current
from a target-mode fit over internal target points. Suppression samples are not
used to define this current.

Field maps and gradient maps include SI units in plot titles and color bars.

For active-shielded examples, `*_shielding_effect_metrics.csv` compares
shielded and unshielded fields on the same external suppression surface. These
fringe-field metrics are separate from the internal gradient efficiency and
linearity metrics.
