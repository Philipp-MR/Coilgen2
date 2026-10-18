# Joint group returns

The engineering rationale and remaining search extensions are documented in
[ROUTING_RATIONALE.md](ROUTING_RATIONALE.md). The behavior below describes the
current implementation.

Joint routing plans the end of an intra-group return and the connection to the
next group as one transition. It searches group order, inward/outward spiral
order and bounded alternatives for contour openings. It never reverses the prescribed current
direction of a contour. The resulting route is one closed series circuit; this
does not design external leads, terminal pads, vias or PCB layers.

`CoilGenConfig.joint_group_routing` defaults to `True`. The existing route is
built first. A candidate replaces it only after independent validation on an
isolated part. Set the option to `False` to disable joint replacement. The
selected routing policy still controls construction of the initial route.

## Regular initial routes and curved supports

The regular policy orders verified essential rings monotonically along the
support axis, with axis sign determined by directed contour area. It requires
compatible boundary topology, once-encircling contours, a common current
orientation and disjoint complete axial bands. Nested saddle contours,
interleaving bands and ambiguous supports are left unchanged. Full contour
coordinates, potential, direction and multiplicity are preserved. Diagnostics
record the original and selected loop indices and axial travel.

Verified conical ring groups reuse the coherent seam and coupled cut-side
construction. Explicit alignment opt-out remains effective. Surface connector
seeds are refined inside a bounded strip of actual triangles, with crossings
parameterized on shared edges and exact terminal points. This minimizes strip
arc length without mesh-vertex zigzags; it does not prove a global geodesic,
obstacle clearance or a manufacturable bend. Distance-only queries retain the
vertex-distance behavior. The physical policy retains the historical ring
enumeration and connector seeds.

Outside/inside mounting now recognizes complete convex polygonal cylindrical
and conical-frustum side meshes using their actual oriented faces and two
circular, homothetic rims. Whole swept triangles are tested against the
verified mesh solid; exterior vertices alone cannot rule out midface
penetration. Signed lift, finite axial footprint, exact section/body frame
correspondence and bounded work remain required. This certificate concerns
the supplied mesh substrate, not an ideal analytic cylinder inferred from
metadata. Unknown shapes and unfinished audits remain unsupported.

Explicit within-group joins and returns are recorded separately from protected
turns. Winding instructions retain those roles and their source ownership.
Intervals lacking exact construction matches remain ambiguous; their
regularity metrics are reported as lower bounds. The joint search continues
to use tangent-constrained, radius-screened curve families and complete
circuit coupling, followed by the full physical and field gates below.

`joint_group_routing_search_mode="progressive"` is the default. Complete
directed circuits are ranked with cheap port geometry before detailed curves
are generated for a small batch. Initial curves retain compact, B0-oriented
and lifted representatives, including opposite cylindrical winding directions
and rail extremes. When the initial batch finds no accepted improvement,
broader curves and circuit orders use the remaining physical validation budget.
Search levels share cumulative work limits; earlier supported options remain
available. `"thorough"` retains the wider eager transition search and original
curve grids for comparison or known-pattern regressions. Neither mode proves
global optimality or infeasibility. Fine chord, bend, support and clearance
controls, unchanged-current field gates and final body audits remain identical.

Before planning, the actual mesh is checked for vertex bijection, triangle
connectivity, boundaries and oriented normals under world-axis mirrors and
half turns. SF checks fit both parities up to an additive gauge and use maximum
residuals. Contours must match their levels, physical turn multiplicity,
directed geometry and mesh provenance, with source direction checked against
the local SF current. Missing provenance or unresolved parity cannot certify
current symmetry. Verified group permutations prioritize existing circuit
proposals. They do not authorize discarding routes or reusing field/body
certificates: cuts, mounted profiles and the target grid can break source
symmetry. Diagnostics retain individual failed and approximate checks.

Progress counters describe the active phase: topology planning, completed
curve transitions, coupling and physical finalist validation. Timed phases
show a rough remaining-time estimate after two completed units and revise it
when the active unit is slower. Adaptive coupling and first-time baseline
certification explicitly report unavailable ETA when their total work is
unknown. No phase percentage is a whole-run completion estimate.

`routing_circuit_scope="single"` is the default for connected layouts. Grouped
source surfaces are assembled into one routing context with preserved mesh and
contour provenance and a common nonzero series current. Disconnected support
components do not acquire imaginary support across their gaps. The resulting
assembly must pass independent physical validation. `"per_part"` explicitly
retains independently closed circuits; contour-only modes remain independent.
Across multiple supports, single-circuit mode therefore requires the conductor
sweep: `skip_sweep=True` cannot certify the assembled physical conductor and
fails explicitly. Raw single-support and contour-only workflows retain their
existing behavior.

The candidate must satisfy all of these conditions:

- Every directed opened-contour segment in the selected cut context occurs
  exactly once, with unchanged coordinates. Alternative cuts are reconstructed
  from the same full directed contours: original coordinates, current direction,
  potential and physical turn multiplicity are preserved. Each opened contour
  is checked as one forward contiguous arc of its full source. Every required
  group occurs once in the closed circuit, including the closing connection.
- Mutable spiral joins and return transitions have tangent-matched curves.
  Lift obeys `normal_shift_length`; `skip_normal_shift=True` disallows new lift.
- The production sweep passes its existing spacing, bend, frame, mesh quality
  and solid checks. An additional bounded triangle audit checks nonincident
  pairs and shared-vertex/shared-edge pairs for overlap beyond their legitimate
  common boundary. An unfinished audit is a rejection. The prepared route and
  physical trace must remain closed; a capped open wire is not a series cycle.
- The reference path and final body satisfy the declared support/mounting
  contract. Centered mounting deliberately permits part of the section to lie
  inside the reference surface; it is not a substrate-exterior certificate.
- Fixed-current fields pass on the full active target and suppression grid,
  both before and after fabrication. No fitted current can hide a routing error.
  Enabled absolute target and Bz gradient-linearity/strength/cross-gradient limits
  still apply. The latter use the same reporting contract at the unchanged
  current and require enough independent spatial samples. Suppression RMS and
  peak may not worsen beyond numerical roundoff.
- A valid baseline must improve the routing objective. An invalid baseline may be replaced by
  a fully valid candidate. A failed baseline never relaxes candidate checks.

## Selection priorities, B0 guide and cut placement

`joint_group_routing_selection_policy="regular"` is the default. Feasible
fabricated candidates are compared lexicographically:

1. Fewest continuous raised transfers on the closed conductor.
2. Least backtracking along each explicitly owned added span's endpoint chord.
3. Least attachment turning at added/protected and declared span interfaces.
4. Least dimensionless bending energy, `L * sum(theta² / dual_length)`, within added spans.
5. Most consistent normalized bending descriptor across repeated joins in the same group.
6. Least added excess travel, then least total turning and shortest complete fabricated length.
7. Lowest weighted DSV RMS of `I * (B_candidate - B_original_contours) dot unit_B0`.

These geometric measures exclude retained winding intervals. Straight-interval
subdivision, rigid motion and current reversal leave them unchanged. Added spans
crossing the closed circuit's storage seam are joined before measurement. A
closed added span has no chord direction; its backtracking is reported as zero
and its full length is excess travel. Ambiguous legacy `contour_group` ownership
is excluded and reported as incomplete evidence. Such an incumbent uses the
historical physical comparison, while regular candidates require explicit
provenance. Total turning discourages repeated direction changes; minimum bend
radius and finite-body checks still determine physical feasibility. The bending
guide coalesces collinear subdivisions; it approximates `L * integral(kappa² ds)`
and discourages concentrated bends without charging retained winding curvature.
Proposal shortlists use the same attachment and curvature measures while retaining
historical length/B0 alternatives so a geometric preference cannot erase all
field-compatible families before validation.
Repeated-join consistency compares scale-invariant bending energies only for
explicit `joint_spiral_join` spans with the same group owner. Single joins,
returns and unrelated groups do not acquire symmetry constraints. Selected
family and consistency diagnostics expose this soft preference; it never
certifies feasibility or forces geometrically different joins to be identical.

Regular preference receives a separately recorded RMS allowance anchored to the
unchanged incumbent, never a pairwise epsilon. The default
`joint_group_routing_regular_field_tolerance_rel=0.001` allows at most 0.1% of
peak active DSV target as additional whole-coil longitudinal RMS. Each eligible
part receives an equal share. Zero forbids RMS growth. This extra limit never
relaxes absolute/non-regression field, suppression, gradient, vector-change,
support, clearance or manufacturing gates. Diagnostics record the policy,
per-part allowance in tesla, absolute RMS ceiling and measured regularity.

`joint_group_routing_selection_policy="physical"` retains the historical ranking:
jump count, longitudinal RMS, then fabricated length for numerical field ties.
It also retains historical contour traversal and mesh path seeding. Both policy
and RMS allowance are editable in the engineering settings and serialized in
the project configuration; changing them invalidates affected pipeline stages.
Authenticated version-2 pipeline checkpoints that predate these fields load
for inspection with the historical physical policy explicitly restored. Version-3
checkpoints retain their recorded policy. Both historical versions disable the
new port search on load; version 4 records its three controls explicitly.
The narrow migration retains source fingerprints
and does not permit resuming evidence created by a different numerical engine.

One uninterrupted raised transfer counts once regardless of crossed turns. A
span through the array's start/end is counted once. Counting removes mounting
and profile-centroid offsets and measures departure from the nominal support;
`FREE_3D` alone does not imply a jump. Nominal surface intervals remain attached.
`routing_jump_height_tolerance_m=1e-6` and
`routing_jump_sample_spacing_m=0.001` declare its resolution. Sampling work is
bounded and an unfinished measurement is not scored as zero. Collinear
resampling and the arbitrary cycle seam do not define new physical transfers.

The full original directed contours, including removed arcs, form the fixed
field reference. End bends, returns, all joins and closure are included in the
fabricated field. No mean offset or current refit is removed. Longitudinal peak
error is reported separately. The vector-change allowance and the configured
field acceptance policy remain hard constraints; these priorities do not waive
either check. Absolute qualification remains separate from explicit acceptance
of inherited failures under the non-regression policy described below.

`joint_group_routing_b0_weight=1.0` favors return/connection tangents along either
sign of `b_0_direction`. The direction is normalized; it is independent of the
support's axis. The cost in metres is
`length + weight * integral(1 - dot(tangent, unit_B0)**2) ds`.
Thus one metre perpendicular to B0 costs twice one metre parallel to B0 at the
default weight. Zero removes the alignment guide. This search guide is invariant under
current reversal, B0 reversal, rotation of both geometry and B0, and linear
resampling. It does not replace the vector-field or finite-body checks.

Topology and curve generation apply the directional guide to mutable joins and
returns, never to retained winding intervals. The thorough graph ranks jump
count first; progressive topology ranking uses only the geometric guide until
curves exist. After shaping, measured jump count determines finalist order.
The additive geometric guide breaks provisional ties before whole-circuit field
evaluation. It is not the final field objective. Final adoption measures the
fabricated centroid and re-counts physical jumps after shaping. A longer route
can win by reducing jumps, improving regularity within its explicit RMS allowance,
or reducing actual longitudinal error under the physical policy. Resistance and
inductance are recalculated downstream.
This is not an inductance/power/force optimizer.
New spiral joins carry explicit `joint_spiral_join` provenance through
fabrication. Their directional cost remains known even when a previously
accepted route is later retained without change.

`joint_group_routing_search_cuts=True` retains two isolated high/low rail
alternatives: flip all unforced groups with available cut states, then flip only the worst
scoring free group. Coupled axial ring groups flip together; a forced member
pins that cohort's automatic members. Explicit `force_cut_selection` choices are honored. Each
context retains the same baseline wire and swept body for fixed-current field
comparison; new groups/cuts are adopted only with the complete validated route.
Single-loop groups in a multi-group circuit receive explicit temporary openings;
a sole closed loop stays closed. Source winding direction is preserved.
Shaping shares its bounded work across contexts, then physical validation visits
the proposed jump budgets in ascending order. Set the option to `False` to keep
the existing multi-loop cuts; singleton internal ports remain necessary.
After unsuccessful regular shaping, bounded alternatives also sample continuous
directed arc-length positions on the original contours. The controls are
`joint_group_routing_port_search_count=8` (0–32 context limit; zero disables the
new alternatives), `joint_group_routing_cut_span_max_factor=4.0` (1–16 times the
nominal cut width), and `joint_group_routing_seam_search_extent_rel=0.1` (0–0.25
of each original perimeter). Actual removed arc spans and seam shifts are saved
per contour. Retained vertices and interpolated segment ports lie exactly on the
original directed polyline, with face/barycentric provenance where available.
Explicit forced groups are preserved. Interleaved axial bands are recorded for
planning; electromagnetic groups are neither repartitioned nor reversed.

On annular supports, lane direction follows the directed source opening chord
in the local circumferential/axial frame. An explicitly degenerate direction
is rejected. Axial projected loop area is not an orientation test for saddle
contours: it vanishes there and its roundoff sign changes under rigid motion.

Shared lane proposals account for profile clearance and required bend radius.
Returns use a continuous lane seed rather than visiting every intermediate
loop start. A bounded exact 3D audit ranks cross-group added seed corridors;
it excludes retained turns, does not prune crossings that could require lift,
and provides no finite-body or infeasibility certificate. Compact and thorough
regular searches can both try these repair contexts. Full curve/body work and
physical-finalist budgets remain bounded. This samples continuous port locations;
it does not solve the continuous global placement problem exhaustively.

For two groups, the search also proposes complete cycles using a prefix of one
directed return and a suffix of the other, with distinct interior return ports.
Both directed cycle edges are declared; every retained contour interval occurs
exactly once. Raw bridge pairs that intersect in 3D are rejected before shaping.
These seeds receive winding trials alongside the ordinary cycle family. Their
bridge separation alone does not certify the remaining returns, bend radius or
finite conductor: the entire shaped cycle still uses the original field
reference and full body validator. The baseline is never replaced by a raw seed.

Within each context, regular topology trials prefer an untried group order
before repeating its other winding states. A witnessed failed inter-group
return also softly lowers the priority of that directed cycle edge, including
the closing edge. Every proposal remains available later; a failure at one
port never certifies that edge infeasible at another. Attempted topology indices
and group sequences are recorded. Field-tolerance rejection is reported as a
physical acceptance limit, separately from computational work exhaustion and
bounded topology/shape coverage.

Verified polygonal frusta also admit nominal cubic and biarc transitions in an
isometric development of their actual facets. Exact facet-crossing samples keep
the mapped chords on the supplied mesh. Chart radius screens do not certify
the 3D bends at facet creases: independent support, swept-body, curvature,
clearance and unchanged-current field checks remain mandatory. Unknown, damaged
or nearly cylindrical frusta fall back to existing supported families.

For completed runs, **Results → Interconnections** or **View → Inspect interconnections…**
opens the immutable geometry inspection. Each actual support is shown separately,
including the biplanar supports. The full developed support uses millimetres,
with signed face height, directed current arrows and group ownership. Exact saved
profile rings can be shown as the conductor envelope. Selecting a recorded
engineering check locates its saved spatial witness; a projected X is never
reported as a physical collision by itself.

For cylindrical axial ring groups, baseline cut placement also preserves one
common corridor. This applies only when B0 and each individual loop normal are
parallel to the detected support axis and every closed loop winds once around
that axis. Local saddle loops and nonencircling or oblique contours keep their
existing cut method. The common plane passes through the support axis; its
radial convention is independent of contour starting vertex, winding direction,
sampling density, and group order. On an axisymmetric surface no intrinsic
azimuth exists, so the convention uses projected world X (world Y if necessary).
It is not a promise of rigid-rotation equivariance for the absolute seam angle.

Automatic ring sides are selected jointly from feasible high/low rails; their
costs are evaluated once and reused. A single explicitly forced side pins the
automatic members. Conflicting explicit side requests remain authoritative.
If no common feasible automatic side exists, routing fails explicitly rather
than silently separating the returns onto opposite sides of the cylinder.
`inter_group_cut_alignment="off"` disables this corridor policy.

Inter-group candidates compare both endpoints' corridor alignment. Matching
azimuths are interpolated on existing contour segments only where both source
vertices remain in the admissible search path; removed return neighborhoods
are excluded. Feasible paired cuts inside the existing near-shortest distance
allowance take precedence over vertex-only proposals. This avoids diagonal
connections caused solely by different upper/lower sampling phases. All usual
connector, normal-shift, field and swept-body validation still applies. A shared
cut corridor is not a certificate of exact whole-coil mirror symmetry or adequate
conductor clearance.

Crossing-bridge and return-ramp acceptance also checks mixed surface/free-space
attachments in the sweep's final support frame. On analytic cylinders, a shared
endpoint can move away from the faceted support while its free-space ramp stays
fixed, and resampling the neighboring surface chord can change the attachment
tangent even when the endpoint stays fixed. Mixed analytic attachments are
prepared together with their retained neighboring intervals; insufficient
realized bend radius or a reversal rejects the candidate, including the
design-spacing fallback. Mesh-only supports and attachments with exclusively
free-space neighbors skip this local preparation. Rejection preserves explicit unresolved
diagnostics; it does not certify the retained route or waive clearance checks.

The default field-change allowance is deliberately small:

```python
joint_group_routing_max_field_change_rel = 0.001  # 0.1%
joint_group_routing_max_candidates = 6
```

`joint_group_routing_field_acceptance="absolute"` remains the default. Explicit
`"non_regression"` allows a routing repair of a design whose field limits
already fail: each passing limit remains hard, and each inherited violation
must remain or improve within numerical roundoff at the same current. The
pointwise vector-change budget and suppression nonregression still apply.
Accepted routing and absolute field qualification are reported separately;
inherited absolute failures continue to make the whole-design report FAIL.
Explicit `"budget_only"` instead admits measured gradient/layout scalar failures
within the hard routing field-change budgets. Suppression RMS/peak nonregression
and enabled suppression limits remain hard. Diagnostics retain every absolute
failure and distinguish newly failed limits from inherited violations that
worsen. This policy does not qualify the gradient design; remaining field
failures still make its whole-design report FAIL.
An optional `joint_group_routing_max_longitudinal_field_change_rel` adds a hard
pointwise change bound along normalized B0, in both raw and fabricated geometry.
It uses the same peak sampled DSV target and unchanged-current normalization as
the full-vector budget, including the same allocation across eligible parts.
The default is `None`, preserving the existing field checks.
The standard cylindrical example explicitly permits 6 mm lift. X/Z use a 2%
full-vector budget and `"non_regression"`; Y uses `"budget_only"`, a 30%
full-vector budget and a separate 5% B0-component budget. The larger vector
allowance exposes Y's transverse-field change instead of silently excluding it.
The example keeps the original contour openings
(`joint_group_routing_search_cuts=False`) while optimizing the complete group
circuit. It also sets
`joint_group_routing_clearance_m=0.003`: contour generation retains its 5 mm
spacing target, while joint routing uses a separate 3 mm centerline target.
Diagonal turn splices do not retain the entire contour pitch as their mutual
clearance. The override applies to an isolated routing configuration, including
physical clearance checks, without changing the stream-function solve or source
current. It replaces both `min_wire_distance_m` and `min_turn_distance` in that
isolated configuration; both source contour-spacing aliases remain unchanged.
Conductor-envelope, insulation, tolerance and existing manufacturing-margin
floors remain hard. Reports distinguish the source spacing target, routing
centerline target and physical clearance evidence. Leaving it as `None`
preserves the original clearance settings.
These are example engineering choices,
not relaxed core defaults or a certificate of clearance between channels.

Intrinsic cylindrical curves are also generated when opposite endpoint
normals cancel, since a developed surface connection does not need one
ambient normal. Compact circular terminal bends joined by an axial span keep
return width independent of group separation; bounded middle-lift profiles
can clear the spiral joins as one raised transfer. Near-level opposite-side
ports with opposing circumferential tangents also receive four-bend axial
detours. Both perimeter winding choices remain available near a half turn,
and bounded shortlists preserve curve-family, lift and winding alternatives
through complete-circuit shaping. All proposals still pass
support, finite-wire clearance, bend and fixed-current checks. Cached
transitions are reused during complete-circuit shaping without charging a
second generation estimate; any new geometry and interaction work remains
bounded by the evaluator's consumed-work limits.

Raised circumferential detours follow the smooth analytic cylinder, with the
facet-to-cylinder radial difference included in the permitted lift. Their
mutable terminal surface arcs blend to that representation with a C2 radial
correction; the physical sweep constrains those arcs to the analytic support.
Original protected contour ports and directed tangents stay fixed in the
nominal route. Nominal zero-lift proposals retain the faceted representation.
This avoids artificial facet corners and attachment kinks without weakening
the independent support or fabricated bend checks.

At the original stream-function current, the Euclidean vector change at **each**
active sample is limited to this fraction of the peak active DSV target-vector
magnitude. This includes transverse components and suppression samples. For
multiple eligible parts, each receives an equal share of the whole-coil budget;
the triangle inequality bounds the accumulated change. Other parts' routed and
fabricated fields are tracked separately. Changing this allowance is an explicit
engineering tradeoff, not an optimization tolerance adjustment.

Production topology search scores complete transitions between retained contours;
it no longer ranks a provisional 75% local return plus a separate bridge. A
shared transition evaluator generates tangent-matched curves, screens support
and protected-contour clearance, and caches the same curves for final shaping.
Within-group joins are scored in the group states; inter-group transfers are
scored once on the graph edges. A state with an unsupported join stops evaluation
immediately. Its failure concerns the available curve family, not the winding's
global physical feasibility.

Direct lifted curves and actual 3D axial returns are both generated routinely.
Compact circular terminal bends with a straight middle span are also proposed
when both attachments share a support facet. Their optional raised middle has
C2 lift ramps and retains nominal terminal bends. On a verified cylindrical
polygon extrusion, the evaluator additionally develops the support into
perimeter/axial coordinates, shapes the connection there, and inserts every
crossed facet boundary before mapping it back. This lets short spiral joins
cross support facets without the radial bow of a Cartesian cubic. Signed radial
arches use the existing lift allowance; they receive no extra clearance.
Nominal intervals carry `SURFACE` geometry through analytic-cylinder sweep
realization. Raised intervals carry `FREE_3D` geometry. This distinction avoids
artificial jumps and mismatched surface/free-space attachments; every family
still passes independent support, spacing, bend, field and body checks.
Axial returns contain tangent-matched entry/exit bends around an exactly straight
span parallel to either sign of B0. If these cannot satisfy local constraints,
the evaluator also tries bounded piecewise cubic paths through support waypoints
and two B0-aligned corridor alternatives. The spans match directed tangents at
their joins; they do not promise continuous curvature. When locally supported
connections conflict in a complete circuit, only the implicated transitions
receive additional corridor alternatives before one further coupled ranking.
This is a finite curve family, not a general obstacle-avoiding path planner.

The default topology search admits 1–64 groups and physically expanded turns,
with at most 250,000 state expansions and 50,000 connector evaluations. Small
graphs are enumerated completely when their worst-case expansion count fits the
budget, independently of beam width. Four groups with two supported spiral
states each give 96 directed closed circuits per cut context, 252 attempted
edges and at most 48 distinct inter-group connector evaluations. Larger graphs
use fixed-order state dynamic programming, a bounded group-order beam and
further state refinement. Complete incumbents survive later work exhaustion.
Search pruning or exhaustion cannot establish physical infeasibility.
Optimality diagnostics refer only to the enumerated additive graph of available
local transition costs; remote transition interactions and final fabrication
can change the ranking. They never establish optimality of the manufactured coil.
A single multi-loop group can reshape its own closing return. A lone closed
contour without open return ports remains unchanged.

The old 24-transition rejection is removed. Curves are ranked using a conservative
spatial interaction graph: separated transition envelopes need no pair score;
nearby transitions still compete jointly for clearance. Selected connections
constrain future option domains before a partial circuit is pruned.
The beam uses an optimistic remaining-jump bound so slightly longer local joins
can survive when they permit a compatible combined return. Complete-family
representatives count actual group returns; short spiral joins cannot fill their
slots. These checks preserve the candidate caps and final physical objective.
Protected winding segments use a reusable spatial index without enumerating winding-to-winding
pairs for every proposed curve. SciPy accelerates these queries when present;
a bounded NumPy implementation remains available. Defaults bound analytic trials
to 32,768 and retained curve samples to 2,097,152 per shared transition evaluator
(one cut context). Geometry and support results are reused through final
shaping. Content-matched protected-contour clearance certificates avoid repeating
the same fixed-obstacle queries. Support screening and coupled connector ranking
have separate 8,000,000 candidate-pair budgets per evaluator, reported individually
and together (16,000,000). Coupled ranking also retains its 32,768 interaction-score
limit across shape attempts. Budget exhaustion is explicit and is never cached as an
unsupported geometric connection. These ranking checks are not finite-body
clearance certificates.

The configured conductor-preview budget remains 1–64 (default six), shared
across all cut contexts and shape alternatives. Route-field failures before
sweeping do not consume these slots. A separate per-part limit of 96 candidate
field screens bounds that work; exhaustion is reported as computational work,
never geometric infeasibility. Diagnostics retain field-screen and body-preview
counts separately. All unchanged-current route and fabricated field limits
remain mandatory; a candidate reaches sweeping only after its route field passes.
Shape attempts are bounded
separately to `max(6, 3 * candidate_budget)`. The best fully validated physical
objective reached within this budget is selected, rather than the first passing
candidate. All alternatives for one part use the same original baseline and
other-part field backgrounds. The winner is adopted once, then the backgrounds
are updated before considering the next part. Numerical ties preserve the first
deterministically encountered winner.
Large/unsupported cases, missing target or swept geometry, weighted turns that
have not been physically expanded, and PCB layer routing retain the baseline.
Budgets also bound route points, section points, target points and triangle pairs.

An invalid reference can also make local field-preserving repair insufficient.
On identical geometry evaluation, current, model and sample weights, a whole-coil
pointwise vector-change allowance `epsilon` implies
`RMS(Bnew_z - Btarget_z) >= max(0, RMS(Bold_z - Btarget_z) - epsilon)`.
This reverse-triangle bound can rule out a target-error threshold within that
specific reference-field neighborhood. It does not prove the complete design
infeasible or transfer automatically between different fabrication resweeps.
Retained contours can independently violate the required bend radius. Protected
contours and a disabled source-deviation option cannot silently be replaced by
smoothed turns merely to obtain a passing interconnection result.
The independent body audit admits at most 2,000,000 faces and streams triangle
pair work in batches of at most 4,096. Its separate ceilings are 128,000,000
broad-phase leaf pairs and 32,000,000 exact near pairs. These bounds admit the
standard 64-point round-wire sweeps while preserving every incident-overlap
and nonincident-clearance check. Exhaustion rejects the candidate.
Planar convex supports have a whole-body halfspace certificate. Other coherently
oriented supports, including cylinders, are admitted under centered mounting
with adaptive reference-path checks and a conservative finite-body boundary
margin. Curved `surface_outer`/`surface_inner` mounting remains unsupported by
this additional certificate and retains the baseline. Support checks assume an
ordinary geometrically embedded input mesh; they do not repair a self-intersecting
support. Exhausted geometric-certification budgets are explicit rejections.

For a declared analytic cylinder, nominal `SURFACE` intervals have a separate
reference-tessellation certificate. The mesh must match the oriented cylinder,
both chord endpoints must lie on its declared radius and domain, and the exact
radial deficit of the entire chord must stay within `sweep_max_chord_error`.
That chord-specific deficit accounts for finite tessellation of the analytic
reference under intentional centered embedding. It is not an exterior-to-mesh
certificate or a general support tolerance. Raw source vertices and free-space
chords retain the strict support predicates; actual lift, physical boundaries
and the complete body tube are still checked. An exact triangle AABB hierarchy
charges construction and traversal work while keeping the existing 50 million
support face/edge work budget and all exact distance queries.

The new route's sweep is retained, and downstream PCB, inductance and field
evaluation use the selected part. Rejection retains the original route, groups,
cuts, fabrication path and mesh objects. Decisions are available through
`part.joint_route_diagnostics` and the exported
`*_partNNN_joint_routing.json`, including numerical failure reasons and timing.
`baseline_retained` is a selection decision, not a success certificate:
`selected_route_status` separately records invalid, unverified, existing
manufacturing checks passed, or independently validated selection. The Halbach
example also writes `*_export_status.json`; files exported under its explicit
warning policy are labeled `diagnostic_only` when manufacturing checks fail.

Final field evaluation and slice maps use the fabricated conductor centroid
when it exists. Diagnostics identify that geometry and the reporting current.
Configured current fitting remains available for ordinary target reports;
`routing_distortion_*` always compares with the original contours at the
unchanged series current, including the complete closing return. RMS and peak
are reported in tesla. Numerical selection ties are quantized at 1 fT for RMS
and 0.1 nm for length, using one transitive ordering for all candidates.

No method can guarantee a feasible winding for contradictory geometry and
manufacturing constraints. In particular, a fixed surface contour crossing a
sharp physical support crease may require a rounded support or a separately
permitted lift-off transition. Moving cuts or refining STL triangles cannot
remove that limitation. External terminal leads, PCB layers, force/torque
optimization and simultaneous routing of independently solved gradient axes
remain outside this closed-series routing stage. Existing other coil parts
within the same solve are included in final collision and field validation.
Inter-part surface clearance includes both insulation layers, manufacturing
tolerance and bend-clearance margin. Bounded component containment checks
also reject nested conductor volumes that do not intersect at their surfaces.
Other parts must already pass their own manufacturing checks: this stage does
not simultaneously repair multiple invalid bodies or certify cavity semantics.

Small reproducible cases live in `tests/_group_routing_cases.py`: facing/shared
corridors, opposite prescribed circulation, a blocked corridor, insufficient
bend/lift space, and a three-group nearest-neighbor trap. Temporary geometry,
reports and before/after views are under
`outputs/group_return_routing_20260917/`. These cases distinguish a shorter
topological route, a manufacturable conductor, and a field-qualified adoption.
In particular, the first manufacturable facing-corridor proposal was correctly
rejected at an explicit 5% field budget; its best screened change was about 6.6%.
Passing a geometry test alone does not authorize replacing the coil.
The historical `outputs/spiral_routing_20260920/default_repair_validation.json`
experiment repairs a 1 micrometre transition-only perturbation of an independently
qualified route under the earlier length-based selection. The perturbed baseline remains
manufacturing-valid. At the default 0.1% field allowance, its physical objective
improves from 0.33256132274 m to 0.33256123479 m; routed and fabricated maximum
vector changes are both 6.29e-13 T/A against a 4.20e-9 T/A budget on nine samples.
All six candidates reach the independent validator; only one passes, and the
other five are rejected by the fixed-current field limit. This is a constructed
repair experiment, not evidence for the current jump/field ranking or a claim
that a production Halbach layout passed. Current regression tests evaluate
selection and field rejection against the new priorities.

The saved 20 September Halbach contour replays and two alternate cut contexts
per axis failed the then-tested transition family within the configured bend/lift/support
constraints. These nine failures do not establish global infeasibility and
do not authorize replacing any route or calling its STL manufacturing-ready.
See `outputs/spiral_routing_20260920/SAVED_GEOMETRY_REPLAY.md` for reproducible
scope, source snapshots/hashes and axis-specific results. Those nine contexts
produced no complete proposal and reached no full-body or field certificate.
They have not been requalified by this implementation's synthetic regressions.
