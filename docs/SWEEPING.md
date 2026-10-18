# Swept-conductor geometry and inspection

Sweeping constructs a finite conductor around a typed route. A watertight mesh
alone does not establish a usable conductor: the centerline, material frame,
support contact, local profile geometry, and remote clearance must also pass.
The stream-function optimization is unchanged by these geometry checks.

## Lifted routes

Normal shifting interpolates the declared mesh-normal field barycentrically on
indexed support triangles. It does not choose a single nearest vertex or infer
the mounting side from the mesh centroid. Undefined or inconsistent normals
leave the affected lift unresolved.

Crossings use available segment ownership. Physical conductor proximity selects
over/under roles; a larger requested spacing warning cannot suppress a necessary
lift. Closest-point parameters locate the crossing constraints, so an underpass
at a fixed attachment does not flatten the entire neighboring connector segment.
Distinct local distance minima retain separate anchors, and contact clustering
does not merge different route priorities. Exact nonadjacent centerline contacts
are retained even inside the short-bend exclusion used for proximity warnings.
Contour points and shared connector endpoints remain fixed. The lift never
exceeds `normal_shift_length`. Arclength ramps use sorted nearest-anchor
queries with linear temporary storage instead of a path-by-anchor distance
matrix. An insufficient lift or incompatible ownership remains explicit.

A local connector spike may be corrected only when both adjacent intervals have
the same explicit connector owner and are offset/free geometry. Movement is
bounded by the smaller of the lift and cut width. Both replacement chords must
remain above the same planar support triangle, and no nonadjacent gap may be
reduced below its existing or required value. Other cases retain their geometry
and require a new route or mechanical transition. This correction is not a
finite-radius fillet.

Explicitly owned `FREE_3D` inter-group overpasses with deficient vertex ramps
are reconstructed as complete connectors. The repair trigger checks newly
introduced or worsened overlap of finite-radius bend trims, newly introduced
backtracking, and insufficient physical clearance at the selected underpass.
Thus even a two-point connector, with no interior vertex available for lifting,
can receive a bridge. A bounded search fits a cubic baseline to the fixed
attachments and neighboring contour tangents, then places a smooth normal arch
at the actual crossing. Planar proposals use eight handle lengths; spatial
proposals use at most 64 combinations of independent handles and arch heights.
Spatial proposals retain the full 3D endpoint tangents across support creases.
Dense analytic
curvature checks include the conductor envelope; adaptive sampling limits
chord error and tangent/frame changes. Infeasible candidates are reported,
without falling back to the sparse vertex spike.

Acceptance checks every output chord's footprint against the union of coplanar
support triangles, including holes and boundaries. Across different facets,
adaptive distance bounds and triangle-intersection checks certify the lift
limit, mounting side and support-boundary exclusion; exhausting that check's
budget rejects the proposal. Arch height above the fitted baseline is distinct
from distance to the actual support. The latter must obey the configured lift
limit too. Acceptance also checks the complete surrounding route.
Spacing-preserving proposals are tried first. For an originally intersecting
connector only, a final fallback may reduce an existing gap above the physical
conductor-envelope requirement when no spacing-preserving proposal succeeds.
It cannot worsen a gap below that physical requirement; all support, curvature,
new-intersection and ownership checks still apply. Its design-spacing losses
are recorded explicitly in `bridge_events`, and the design-spacing target
remains unmet. This avoids retaining a physical crossing merely to preserve a
larger, already infeasible spacing goal. New centerline intersections are
rejected independently.
This supports a single arch, including a connector touching the closed seam.
It is a local centerline correction, not a certificate for the finite swept conductor
or the entire coil. In particular, a 3 mm bridge does not satisfy a 6 mm
clearance target. The final sweep retains its independent physical checks.
An interior underpass remains protected and prevents a whole-span arch; an
underpass exactly at a fixed attachment is compatible with the bridge.

Within-group return approaches can also cross a conductor before their normal
lift reaches its full height. Deficient approaches are reconstructed between
the fixed attachment and an already lifted point, using explicit return-route
membership to protect contour intervals. A bounded set of 64 proposals preserves
both endpoint positions and tangents. Candidate ranking uses nearby underpasses
after the same support attachment and bend preparation used by sweeping: a raw
chord through a faceted support can substantially underestimate the eventual
underpass position. This local obstacle preparation includes closed-seam context
and leaves the original route unchanged. The complete final swept body still
needs its independent collision and bend checks.
An approach that itself wraps across the closed path's indexing seam is
currently reported as `unsupported_return_seam` and a routing failure, rather
than silently skipped. Inter-group bridges that terminate at that seam remain
supported; reconstructing a return across it requires cyclic resampling.

`normal_shift_physical_clearance_status` records the reference-path proximity
screen independently of the design-spacing target. These are unmounted route
coordinates: the final sweep must also check mounting offsets and material
orientation, even when this screen is clear. Remaining centerline intersections
are explicit routing failures.

`CoilPart.normal_shift_provenance` retains the pre-lift path, post-lift path,
post-repair path, normals, triangle/barycentric samples, owners, effective lift
route memberships, closest-point crossing anchors, crossing events,
and unresolved contacts. `save_manufacturing_diagnostics` writes these into an
additional `*_partNNN_normal_shift_provenance.npz` file when available. Arrays use
SI units; indices refer to the stage before route cleanup. Load with
`np.load(path, allow_pickle=False)` and parse `metadata_json` with `json.loads`.
Old exports without these records cannot establish which upstream operation
created a particular saved kink.

Resampled connectors use provenance schema 2: input points, normals, original
owners, and crossing events retain their original indices; output points,
lift factors, contacts, and output owners use the new indices. The arrays
`output_source_segment_indices` and `output_source_segment_fractions` provide
monotone construction correspondence between these domains. Correspondence
does not imply that intermediate connector coordinates remain unchanged.
`bridge_events` records each accepted or rejected proposal and its validation.
Stored connector paths use this mapping, including reversed paths and closed
seams, so diagnostic drawings include every inserted sample.
`lift_distance_convention` distinguishes original normal displacement,
baseline-relative inter-group arch displacement, and signed closest-support
distance on reconstructed return approaches. Bridge validation diagnostics
separately record support-distance bounds. A return approach's added arch height
is not its total height above the support.

Remote centerline distance searches use a bounded AABB hierarchy and batched
exact segment distances. Results retain the established pair order and distance
kernel. Dense cases that exceed the candidate budget fall back to the existing
chunked search instead of allocating an unbounded pair list. The saved GY
benchmark produces identical contact pairs and bitwise-identical distances in
about 2.2 seconds instead of 113--115 seconds; see the
[benchmark artifacts](../outputs/wire_distance_bvh_benchmark_20260916/).

## Frames and bends

Nonuniform three-point tangents account for unequal adjacent spacing. Free
spans use double-reflection rotation-minimizing transport. Attached spans solve
an arclength-weighted signed guide problem with a tridiagonal/cyclic system;
memory scales linearly with path size. Opposite endpoint directors on a free
span use a distributed material twist instead of normalizing a zero vector.

`sweep_surface_normal_constraint_strength` retains its range: zero selects free
transport, one fixes the signed guide at attached samples, and intermediate
values weight guidance on the physical profile-envelope scale. Inserting
collinear samples no longer repeatedly multiplies guide strength. Closed
attached paths do not receive an unrelated global seam twist.

Fillet tessellation respects the requested frame angle and an outer-profile
chord bound. Projected mesh fillets undergo an independent dense radius check,
including local join probes; making output triangles coarser cannot make the
same physical crease pass. A transverse path in exact contact with a sharp
octagonal edge generally needs a mechanical rounding or an explicitly designed
lift-off transition. Replacing that support with an analytic cylinder changes
the engineering problem and is not an automatic repair.

Bend analysis preserves curvature samples on free and offset paths, removing
only numerical collinearity. Linear subdivision cannot recover a free curve
after chord simplification. Ring coarsening also checks each resulting frame
interval against its own previous bound; a bad crease elsewhere cannot permit
a new frame-angle violation on a valid bridge.

In fillet mode, `sweep_corner_cluster_max_source_deviation_m` provides an
explicit fallback for unresolved corner clusters on a declared analytic
cylinder. Its default is zero, which retains the existing per-corner behavior.
A positive allowance permits a bounded intrinsic two-arc realization inside
one protected contour owner. Endpoints and intrinsic tangents are preserved;
the two-way source corridor is checked against all original contour samples
after the declared cylinder retraction, with a validation sampling margin.
The nominal contour arrays and current remain unchanged. The new realization
must retain the required physical bend radius, and neighboring corners are
replanned before the complete sweep passes its independent body checks.
Joint routing also checks the resulting fabricated centroid fields at the
unchanged series current. The cylinder retraction and extra corner displacement
are reported separately.

With that explicit allowance enabled, bend analysis also preserves the original
sampled curvature of owned SURFACE spiral joins and returns. It first verifies
the complete source sequence and its ownership and geometry contracts; missing
or ambiguous provenance retains the existing behavior. After surface sampling,
at most two further passes can resolve frame violations at unique original
source vertices, reusing the realized analysis and the same corner-search work
budget. All passes share the hard limits of 16 compound clusters, 368,640
proposals, 128 million distance pairs and two million curvature points. Compound
arc interiors obey the configured length, chord and frame bounds, with finer
endpoint samples for tangent matching. Final frame, bend and body checks still
apply to the complete realized path.

## Final checks and plotbook controls

The final complete-frame rotation includes tangent, normal, and transverse
axes on every interval, including free spans and the closing seam. The
`sweep_adaptive_max_frame_rotation_deg` compatibility field now reports this
actual output quantity. Support-normal trial and output rotations have separate
fields. Unresolved discontinuities are reported instead of recursively adding
linear samples that cannot remove a corner.

The finite-profile screen evaluates `J = 1 - curvature · profile_offset` at all
polygon vertices, including mounting offset. This is a sampled local inversion
check, not a continuous-curve proof or a global collision certificate. Bend
radius, folded quads, triangle quality, support constraints, and remote
intersection checks remain independent. Trapezoidal current-density reporting
uses the same polygon area as the resistance calculation.

In the plotbook's swept-mesh inspector, **Inspect issue** cycles stored fold,
profile-inversion, and frame-angle locations, marks the recorded ring, and fits
nearby actual mesh rings. It does not regenerate the sweep or rerun collisions.
Body isolation, support/path overlays, edges, opacity, and orthographic views
remain available. Older results without locations are identified explicitly.
See [PLOT_BOOK.md](PLOT_BOOK.md) for the plotting controls.

## Validation and remaining constraints

The latest [crossing replay report](../outputs/unshifted_crossings_20260916/REPLAY_REPORT.md)
records repairs and complete remote swept-triangle checks against the
September 16 saved source routes. It supersedes the crossing status in the
older reports below. It does not rerun electromagnetic optimization, and its
diagnostic meshes retain explicit manufacturing failures where other geometric
constraints remain unsatisfied.

Earlier validation stages:

The [saved GZ connector replay](../outputs/sweep_connector_fix_20260916/FINAL_REPORT.md)
inserts the production bridge stage into the exact saved post-lift surroundings
and runs cleanup and sweeping. The repaired connector's maximum actual frame
step falls from 94.88 to 2.14 degrees; its minimum profile-centroid radius is
2.84 mm, above the 2.72 mm requirement. All 160 bridge samples survive in the
actual STL, and the complete mesh has zero folded quads. The 6 mm clearance
target, another unsupported connector, and physical octagonal crease failures
remain unresolved. This replay does not rerun optimization or certify the
whole winding for manufacture.

The [saved-run replay report](../outputs/sweep_implementation_20260914/README.md)
records source/input hashes, numerical regressions, and timings. The saved GX
post-lift route still contains the original connector backtrack, physical
octagonal crease violations, and a confirmed remote branch intersection. It
correctly remains rejected. The upstream fixes are validated on independent
pre-lift fixtures; no new three-channel optimization or claim of a corrected
GX winding is made.

The geometric basis and alternatives are reviewed in the
[research report](../outputs/sweep_research_20260914/HALBACH_GX_SWEEP_REVIEW.md).
Free-frame transport follows [Wang et al., 2008](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Computation-of-rotation-minimizing-frames.pdf).
Material orientation relative to transported frames is informed by
[Bergou et al., 2008](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf); the guide
energy and conservative route-repair rules here are CoilGen-specific choices.
