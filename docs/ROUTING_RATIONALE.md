# Rationale for global spiral and return routing

Design rationale, updated 22 September 2026. Whole-transition scoring, complete small
state-graph search, bounded order/state optimization and best validated
fabricated-path selection are implemented. Corridor generation remains a
bounded family; general obstacle-avoiding corridor search, arbitrary cut
relocation and the lower-bound-driven search extensions below are not claimed
implemented. This document does not establish that the saved Halbach layouts
have passed validation. Current behavior is described in
[JOINT_ROUTING.md](JOINT_ROUTING.md).

**Design decision:** choose the complete directed circuit, including each
group's spiral direction and connection corridor, before committing to local
return paths. First minimize physical raised transfers, then actual longitudinal
field distortion at fixed current, then conductor length.
Preserve winding circulation and treat field and manufacturing limits as hard
constraints.

## Why this is the right variable

Spiral-in and spiral-out change the order in which a group's contours are
visited. They must not reverse current around any individual contour. Each
choice presents different entry and exit ports, with different directed
tangents and access to neighboring groups. A group is therefore a small set
of directed routing states, rather than a closed object with an independently
chosen return.

A useful connection can run from the inner exit of an inward group to the
inner entrance of an outward group. Another useful combination connects two
outer ports. Neither is universally preferable: physical access, tangents,
support and the rest of the circuit decide. The integrated transition replaces
the old local return and inter-group bridge; neither removed path is retained
or counted in its cost.

"Shared return" means one series span fulfilling both functions. It does not
mean a conducting bus, a branch, overlapping wires, or traversing the same
segment twice. Separate wires may share a routing corridor only with the
required insulation and separation.

The original CoilGen method constructs within-group paths and then joins
groups through a topology-guided sequence. The proposal here is to couple
those decisions through entry/exit states and complete transition costs.
[CoilGen, sections 2.10–2.12](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29294).

## A physically motivated, lean objective

For a fully validated complete circuit use the lexicographic objective

\[
\min_{\rm lex}\left(N_{\rm jumps},\;
\operatorname{RMS}_{\rm DSV}\left[I(B_{\rm circuit}-B_{\rm contours})\cdot\hat b_0\right],\;
L_{\rm total}\right).
\]

Count one continuous raised transfer once, including across the cycle storage
seam, at the declared support-distance sampling resolution. The error includes
removed contour arcs and every added interval at unchanged current. A weighted
length/alignment expression remains only a cheap graph-search guide after jump
count. Field RMS is coupled across the whole circuit and is not an additive
independent-edge cost. The graph optimum is not a field or fabrication optimum.

- Total length discourages detours and additional resistance for a fixed
  material and section. It must include retained turns when comparing cut
  contexts, because the amount of retained contour can change with a cut.
- The directional term favors either sign of B0 and is unchanged by current
  reversal, rigid rotation with B0, or linear resampling.
- Each internal join is charged once in its group state. Each complete
  group-to-group transition is charged once on its connecting edge.
- Search uses shaped geometry as it becomes available. Final selection evaluates
  the fabricated conductor centroid path, with segment ownership mapped
  through fabrication. A raw-route score remains a labeled search estimate.
  Missing ownership must not silently classify uncertain intervals as fixed.

There is a physical reason for the B0 preference. For an ideal filamentary
element parallel to a fixed B0 direction, the Biot–Savart cross product gives
zero field component along that direction at off-wire observation points.
That is a deduction from the
[Biot–Savart law](https://openstax.org/books/university-physics-volume-2/pages/12-1-the-biot-savart-law).
Likewise, `dF = I dl × B0` gives zero local force from a parallel uniform
background field. [Magnetic force on a conductor](https://openstax.org/books/university-physics-volume-2/pages/11-4-magnetic-force-on-a-current-carrying-conductor).
The end transitions, transverse fields, spatially varying B0 and interactions
with other conductors remain relevant; this is not a force or torque certificate.

A return constrained to a surface cannot always follow B0. Its tangential
projection is useful for proposing corridors; if that projection vanishes,
there is no preferred in-surface direction from B0 alone. Score the actual
three-dimensional tangent and permit lift only under the configured contract.

Do not initially add a large collection of aesthetic weights. Record lifted
length, maximum height, bend radius, frame rotation and clearance margins.
Use well-defined secondary measures only for numerical ties, followed by a
stable deterministic identifier. Under the selected strict priority, one fewer
jump can justify a longer route only after all physical and field constraints
pass. Without an absolute longitudinal-error bound, minimizing jumps can favor
a counter-winding surface return that cancels much of the desired field. The
implementation retains existing field gates rather than inventing a new limit.

## Hard feasibility comes before preference

A candidate must preserve every required directed contour interval exactly
once, physical turn multiplicity and prescribed current. Alternative openings
must come from the same full directed contours; cut gaps are bounded design
variables, not permission to discard difficult turns.

Require a connected series circuit, including its closing link under the
current closed-circuit model. Future feed-terminal routing must explicitly
replace that closure constraint. The complete fabricated body must satisfy
section clearance, insulation, bend radius, twist/frame limits, support,
allowed lift and solid validity. Intersections with other transitions and coil
parts are included.

At unchanged current, check full routed and fabricated fields on the active
target and suppression grid. Retain the existing pointwise vector-change
allowance, absolute layout/gradient requirements and suppression checks.
Reordering unchanged contour arcs does not guarantee the field of the added
connections is acceptable. Neither improved geometry nor a fitted current
can bypass these checks.

The principle of including local motion constraints during path planning,
rather than hoping refinement will repair an arbitrary geometric path, is
also established in motion planning. The proposed application here is to
conductor tangents, bends and section orientation.
[LaValle, planning under differential constraints](https://lavalle.pl/planning/node648.html).

## State and transition model

For each permitted cut context, a group state contains:

- spiral-in or spiral-out order and the directed source identities;
- its retained contours and candidate internal joins;
- entry/exit coordinates, directed tangents and section-frame requirements;
- internal jump count, length/alignment guide and validation status.

Between two states, retain a small set of complete transition alternatives:
direct smooth connection, a permitted B0-oriented corridor, and alternative
surface corridors around obstacles when necessary. A corridor reserves space
for the conductor section and its clearance, not just its centerline. Interior
ports require an actual passage through the opened turns or permitted lift;
the point-to-point distance does not establish access.

The local connector joins the last retained contour of the first state directly
to the first retained contour of the next. Matching endpoints alone is
insufficient: directed tangents, bends and section orientation must be checked.
An obstacle may require several smooth spans. A single lifted polynomial arch
is a useful fast candidate family, not a general corridor planner.

Cache transitions using the actual endpoints/tangents, cut and obstacle
geometry, support, section, constraints and objective settings. Cache numerical
geometry and its scope of validation; do not reuse a certificate after any of
those dependencies changes.

## Search that spends work where it matters

For a fixed group order, optimize all spiral states together with dynamic
programming on the enumerated state graph. With group-state cost `c_i(s)` and
transition cost `t_ij(s,u)`, the recurrence for a fixed initial state is

\[
D_{i+1}(u)=c_{i+1}(u)+\min_s\left[D_i(s)+t_{i,i+1}(s,u)\right].
\]

Try each initial state and include the last-to-first transition. For binary
spiral choices this is linear work in group count for a fixed order. It is
exact only for the enumerated graph with additive independent costs. Keep
several candidate solutions because their remote transitions or fields can
interact.

For small group counts, enumerate orders or use subset dynamic programming
within an explicit state budget. In the four-group Halbach case there are only
`(4-1)! * 2^4 = 96` directed order/spiral combinations per fixed cut context
after eliminating cyclic rotation duplicates. This count excludes corridor
alternatives and physical validation. There is little reason to prune these
96 combinations using an inaccurate retained-return surrogate.

For larger cases, use deterministic bounded order search and re-optimize spiral
states after each change of order. Preserve the valid baseline as an incumbent.
Maintain diverse corridor/state alternatives, rather than spending the entire
budget on tiny variations of one pattern.

Generate expensive transition geometry lazily. Chord length is a safe lower
bound for a transition's length-plus-nonnegative-penalty objective. A direction-
weighted straight chord is not a universal lower bound for arbitrary B0
weights. Cheap bounds can prioritize work; only genuinely valid lower bounds
may justify optimality pruning. Budget-based pruning must be reported as such.

Pairwise transition costs are not the complete problem. Two separately valid
transitions may occupy the same corridor. Check interactions between selected
edges and between them and all selected group geometry. Record the specific
conflicting alternatives, then try other corridors, states or orders. Do not
ban a whole group pair because one local connector failed. Global field checks
also remain outside the additive graph cost.

Validate promising complete candidates using the existing independent sweep,
support and field checks. Keep the best fully validated candidate through the
declared budget. Do not stop at the first passing route. If a later shape or
validation changes its score, compare the final score again. Return the valid
incumbent if no candidate improves it; if the incumbent is invalid, report
that state explicitly rather than treating retention as success.

## Expected behavior on diagnostic cases

| Case | Expected decision |
|---|---|
| Two inner ports face an open common corridor | An in/out combination can use one integrated inner-to-inner transition. |
| Inner access is blocked but outer ports are adjacent | Prefer the viable outer-to-outer state combination; do not force alternation. |
| A short local link creates a long closing link | Compare the complete circuit and reject the locally attractive choice if its total cost is worse. |
| Odd number of groups in a closed cycle | Strict in/out alternation cannot hold on every edge; choose the best actual closing transition. |
| Two cheap transitions need the same space | Resolve the corridor conflict before acceptance; independent edge validity is insufficient. |
| Longer route follows B0 much better | Allow it only if its declared objective improves and all fixed-current field limits pass. |
| Smooth modest lift versus long surface detour | Apply the field and geometry constraints first; among feasible candidates, fewer raised transfers wins. |
| Sharp support or inadequate clearance prevents available curves | Report failed families and bounds; never infer global infeasibility from one unsuccessful family. |

## Evidence required for adoption

1. On tiny state graphs, compare with exhaustive enumeration, including closure.
   Separately test incompatible transition pairs; basic additive DP alone must
   not be claimed exact for the coupled problem.
2. On small physical fixtures, demonstrate both a successful integrated return
   and a case where a different state/order must win. Include opposing contour
   circulation, blocked inner ports and restrictive lift/bend limits.
3. Verify directed coverage exactly once, unchanged protected geometry, fixed
   current and rejection isolation. Check deterministic behavior under B0 sign
   reversal, rigid rotation with B0 and equivalent resampling, within specified
   numerical tolerances.
4. Exercise a case where final shaping reverses provisional candidate ranking,
   and one where the first valid candidate is worse than a later valid candidate.
5. Compare final geometry and field metrics to a certified baseline. Report
   graph expansions, connector solves, sweep/field evaluations, memory and time.
   A continued or warm-started search with a larger budget must preserve its
   already found valid incumbent. An independent restart with a different beam
   policy need not rediscover that candidate unless it is supplied explicitly.
6. Repeat the saved Halbach geometry replays and, before a production success
   claim, a full run with the original target/current context. Require complete
   body and fixed-current field validation. No synthetic passing example or
   prettier rendering establishes success on those layouts.

Diagnostics should expose group order, spiral states, cuts/corridors, baseline
and selected final cost components, fabrication margins, field deltas, work
limits and rejection witnesses. Distinguish a valid result, a valid unchanged
baseline, an invalid unchanged baseline, and search/validation exhaustion.

## Implementation sequence

1. Replace provisional return-prefix ranking with whole-transition scoring and
   compare the best validated candidate within budget. Preserve existing cut,
   geometry and field acceptance limits.
2. Add state optimization and complete small-order search using cached transition
   alternatives; keep the search bounded for larger group counts.
3. Expand corridor families where the saved geometry demonstrates that direct
   arches are insufficient. Couple corridor conflicts back into state/order
   selection. Widen cut-placement search only within explicit permitted gaps.

The intended guarantee is the **best fully validated route found within the
declared candidate family and work budget**. Global physical optimality or
feasibility for arbitrary supports and constraints is not claimed. The current
saved Halbach failures remain unresolved by this design document.
