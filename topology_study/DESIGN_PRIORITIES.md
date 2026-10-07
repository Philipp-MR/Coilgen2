# Interconnection priorities

User-specified study priorities, 22 September 2026:

1. **Few jumps**, to make the coil easier to wind.
2. **Low B0 distortion**, favoring straight return segments parallel to B0.

The user explicitly defined one jump as **one continuous raised transfer**, even when it crosses several turns. These priorities replace the earlier generic ranking discussion. They are requirements for subsequent comparisons; no production routing code or existing numerical result has been changed.

**Current closure requirement:** produce one continuous closed conductor with no prescribed terminals or final wire opening. Temporary contour cuts and group entry/exit ports remain internal construction variables and must all be reconnected. See [the revised closed-loop decision sequence](CLOSED_LOOP_RATIONALE.md). This supersedes earlier comparisons based on fixed external terminals.

## 1. Count physical raised transfers

Count maximal continuous raised-transfer spans along the ordered winding. A rise, one elevated span over several turns, and a landing constitute one jump. A later separate rise constitutes another. A smooth adjacent-turn connection that remains on the winding support is not a jump.

Count these spans cyclically. A continuous raised transfer that crosses the arbitrary start/end of the stored point array counts once, not twice. Closure and jump count must not depend on where traversal of the cycle begins.

The count must follow the physical conductor and its designated winding support, rather than the number of source contours, curve segments, mesh samples, or routing labels. Joining two raised transfers into one continuous elevated span can reduce the count; relabeling disconnected spans cannot. Retained winding arcs must remain on their prescribed support and preserve their directed current.

Report turns crossed, lift height, raised length, and rise/landing transitions separately. They affect physical feasibility, but do not redefine the agreed jump count. One long raised transfer can be preferable to several short transfers under this priority. Conductor support, clearance, allowed lift, bend radius, and electrical continuity still constrain the design.

## 2. Optimize the priorities in order

Use a lexicographic comparison: first minimize the jump count over feasible complete routes, then minimize their longitudinal field perturbation. There is no weighted exchange in which a small reduction of wire length buys an extra jump.

An explicit proposed field score is

\[
\min_{\rm feasible\ route}^{\rm lex}
\left(N_{\rm jump}, E_{\parallel,\rm RMS}\right),\qquad
E_{\parallel,\rm RMS}
=\sqrt{\frac{\sum_j w_j [\Delta\mathbf B(\mathbf r_j)\cdot\hat{\mathbf b}_0]^2}{\sum_j w_j}},
\]

where the current is unchanged and

\[
\Delta\mathbf B=\mathbf B_{\rm routed}-\mathbf B_{\rm original\ directed\ contours}.
\]

This RMS score is a proposed implementation of priority 2, not an additional user-specified convention. Report the peak absolute longitudinal perturbation as well. Evaluation points, weights, B0 direction, and permitted field errors have not yet been specified for this study. No numerical field tolerance is inferred here. Previously declared hard field limits remain constraints; a reduction in jumps cannot override them.

Length, resistance, maximum lift, and bend severity remain reported engineering quantities and any applicable constraints. They do not outrank either stated preference. Numerical ties should preserve a deterministic candidate order unless another tie-break rule is chosen.

## 3. Why straight returns along B0 help

For a uniform background direction and filamentary current, Biot–Savart gives

\[
d\mathbf B=\frac{\mu_0 I}{4\pi}\frac{d\boldsymbol\ell\times\mathbf R}{|\mathbf R|^3}.
\]

If \(d\boldsymbol\ell\parallel\pm\hat{\mathbf b}_0\), then
\(d\mathbf B\cdot\hat{\mathbf b}_0=0\) at every off-wire observation point. Thus a finite straight axial segment contributes exactly zero longitudinal field in this model. This is a direct deduction from the [Biot–Savart law](https://openstax.org/books/university-physics-volume-2/pages/12-1-the-biot-savart-law).

It can still produce transverse field. The rises, landings, transverse end bends and removed contour pieces can produce longitudinal error. Evaluate the entire closed conductor, including its closing connection, for every comparison. External feeds and a supply opening are outside the current scope. Do not assign zero error to an entire return merely because its long middle segment is axial.

If total field magnitude is relevant, evaluate
\(|\mathbf B_{\rm bg}+\mathbf B_{\rm routed}|-|\mathbf B_{\rm bg}+\mathbf B_{\rm original}|\).
Zero longitudinal field from one segment does not make this complete magnitude difference identically zero.

## 4. Choose the connection points to permit axial returns

For two ports \(\mathbf p\) and \(\mathbf q\), a straight segment parallel to B0 requires

\[
(\mathbf I-\hat{\mathbf b}_0\hat{\mathbf b}_0^T)(\mathbf q-\mathbf p)=0.
\]

Therefore cut positions, inward/outward group traversal, and group order should be chosen together to align suitable entry and exit ports in the plane perpendicular to B0. Merely selecting nearby ports and later straightening their connector can leave unavoidable transverse end sections.

For B0 along a cylinder axis, axial lines are natural return candidates. On a curved support, an exactly straight B0-parallel line may depart from the support or meet an obstacle. Its required raised transfer must then enter the primary jump count. Following the tangential projection of B0 is not generally an exactly B0-parallel straight return.

Parameterize the long return spans as straight segments along either sign of B0 where feasible, joined by clearance- and bend-constrained end transitions. The useful geometric guide

\[
A=\int_{\rm mutable\ connections}\left[1-(\hat{\mathbf t}\cdot\hat{\mathbf b}_0)^2\right]ds
\]

penalizes nonparallel path length. It is not a field-error calculation and does not itself impose straightness. Use it to generate or screen candidate connections when full field evaluation is not yet available; do not report its optimum as minimum B0 distortion.

## 5. Consequences for this study

- Prefer continuous local spirals without raised turn-to-turn joins.
- Choose complete intergroup transfers and returns together; consolidate necessary raised crossings into as few continuous spans as feasible.
- At the minimum feasible jump count, favor straight B0-parallel returns and assess their complete longitudinal field error.
- Require exactly one continuous closed cycle without prescribed terminals. Choose its closing connection jointly with all other interconnections.
- Report the minimum jump count actually found. Only an exhaustive search or a suitable lower bound can establish that fewer jumps are impossible.

The existing atlas does not yet rank candidates by these priorities: its raised paths are illustrative and its B0 direction is unspecified. The seven-loop benchmark compares open centerlines with fixed cuts and connectors, without a complete return or field calculation. Its previous length/crossing result remains valid within that scope; it does not establish the preferred complete winding under this revised objective.
