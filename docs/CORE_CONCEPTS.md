# Core Concepts

CoilGen is a magnetostatic coil-design tool. A design starts with a support
surface, a target magnetic field, conductor assumptions, and numerical settings.
The solver computes a stream function on the support surface, extracts contours,
routes a wire path when requested, and evaluates the resulting field.

## Support Surface

The support surface is the 3D surface on which current is allowed to flow. It
can be a generated cylinder, plane pair, disc pair, imported mesh, or another
registered geometry. The support mesh quality controls the available current
paths and the size of the inverse problem.

## Target Field

The target field is the magnetic field CoilGen tries to produce at target
points. The common scalar use case is a requested `Bz(x, y, z)` map, but target
points can also be imported from CSV, JSON, NPZ, or TSV files.

## Stream Function

The stream function is the continuous surface-current representation solved on
the support mesh. Contours of this scalar function become wire turns. The
stream-function solution is the magnetostatic inverse-design result before
discrete routing.

The numerical core uses nodal, piecewise-linear stream-function basis functions
on each vertex's incident triangles. Their surface currents are constant within
each triangle; Biot-Savart integration maps those currents to the target points.
The sensitivity operator uses configurable Gaussian triangle quadrature; mesh
refinement and quadrature convergence are separate numerical checks.
Boundary-loop constants and gauge reduction determine the allowed current paths
and independent unknowns, so changing them changes the physical problem.

Resistance regularization uses the quadratic sheet-loss form `psi.T R psi`.
Applying `R.T R` instead changes the objective. See
[Physical Assumptions](PHYSICAL_ASSUMPTIONS.md) and
[Stream-function optimization](STREAM_FUNCTION_OPTIMIZATION.md) for the actual
objectives, reduction and acceptance contracts.

## Contours

Contours are closed level sets of the stream function. Their spacing and number
control the discrete turn layout. Contours are still an idealized conductor
representation until they are grouped and connected into a routed path.

## Routed Wire Path

The routed wire path connects contour groups into one or more physical conductor
paths. Routing diagnostics check obvious topology, connection, spacing, and
manufacturing issues. A routed path is necessary for layout-level field,
resistance, and inductance estimates.

## Field Source

Reports identify which field source was evaluated:

- `stream_function`: continuous inverse-design current distribution.
- `unconnected_contours`: extracted contours before terminal routing.
- `connected_layout`: final routed wire path where available.

Use the connected-layout metrics for final layout review. Use stream-function or
unconnected-contour metrics for fast feasibility sweeps only.

## Suppression Surface

A suppression surface is an optional zero-field target surface used for active
shielding or fringe-field screening. CoilGen can derive this surface from the
coil/shield mesh topology, select all faces or interface-like faces, and apply a
geometric clearance check against the source mesh.

For non-smooth supports, a simple normal or paired-vertex offset can place some
target points too close to a neighboring face near corners. Use
`mesh_topology_suppression_min_clearance` with
`mesh_topology_suppression_clearance_action="filter_faces"` when the suppression
surface must maintain a minimum distance from the source geometry.
