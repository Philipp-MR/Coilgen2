# Contour topology and winding interconnection: theoretical study

This study asks which contour arrangements occur on coil supports and which connection strategies are likely to produce orderly, accessible windings. The central distinction is between **topological feasibility**, **geometric manufacturability**, and **electromagnetic acceptability**. Topology rules out certain connections, but cannot select a universally optimal winding without dimensions, terminals, conductor properties, and field requirements. The accompanying 3D figures are schematic centerlines and connection candidates, not manufacturing drawings or verified coil designs.

## 1. Mathematical setting

Let \(S\subset\mathbb R^3\) be a smooth, compact, orientable, embedded surface, possibly with boundary. Choose a consistent normal \(\mathbf n\), a single-valued real stream function \(\psi:S\to\mathbb R\), and the convention

\[
\mathbf K=\mathbf n\times\nabla_S\psi.
\]

Reversing this convention reverses all current arrows and changes none of the topology. At a regular point, \(\nabla_S\psi\ne0\), current follows a contour. In the usual discretization, equal increments of stream function correspond to equal conductor currents: \(I=|\Delta\psi|\). Equal stream-function increments do **not** imply equal physical wire spacing. Locally, away from critical points and for sufficiently small increments, transverse spacing is approximately \(|\Delta\psi|/|\nabla_S\psi|\). The surface-current formulation and constant-boundary treatment are described by [Mäkinen et al., 2020](https://arxiv.org/abs/2005.10060).

For a regular level transverse to the boundary, its compact connected components are embedded circles or boundary-to-boundary arcs. A regular contour does not branch, self-cross, terminate in the interior, or intersect another level. A projected crossing in a 3D drawing does not necessarily indicate a physical intersection.

For an electrically insulating edge, \(\mathbf K\cdot\boldsymbol\nu=0\), where \(\boldsymbol\nu\) is the surface conormal. This requires \(\psi\) to be constant along each connected boundary component; different components may have different constants. Interior regular levels away from those boundary values consequently form loops. Boundary arcs require an explicitly different model: feeding through the edge, a cut coordinate chart, or an intentionally truncated support. Do not silently treat their endpoints as a closed-current design.

## 2. Ten representative motifs

These are an engineering atlas, not ten mutually exclusive mathematical types. In particular, flat and curved patches share topology, while an annulus and an open cylinder are homeomorphic. Their different geometry nevertheless changes connection access, bend radius, and winding procedure.

| Case | Representative 3D support and contour arrangement | Orderly candidate interconnection | Main qualification |
|---|---|---|---|
| 1. Nested flat loops | Concentric or smoothly distorted loops on a planar disk | Cut a consistent narrow sector; connect adjacent levels in radial order to make an open spiral | Retain the circulation of every turn. One terminal is internal; plan its lead explicitly. |
| 2. Nested curved loops | The same family on a dome or bent patch | Transport a coherent seam across the surface and connect adjacent contours | Intrinsic topology is unchanged. Surface curvature, normal curvature of the wire, and tool access can make a formerly easy seam impractical. |
| 3. Side-by-side, same circulation | Two separate contour nests around two maxima or two minima | Spiralize each nest locally, then connect compatible terminals through their shared free region | A short pair of facing bridges can preserve orientation when the facing tangents oppose. Terminal placement still controls the complete route. |
| 4. Side-by-side, opposite circulation | A maximum-type nest beside a minimum-type nest | Preserve the opposite circulating currents and select terminal exits jointly with the intergroup connection | The naive two-bridge connection across facing cuts can require a crossing; use different cuts, a detour, or an explicit layer transition. |
| 5. Saddle hierarchy | A larger enclosing contour family gives way to two child nests | Follow adjacent regular bands locally; allocate distinct connection corridors around the saddle region | A branching family relationship is not permission for an electrical three-way junction. Avoid placing finite-width wire on the singular contour. |
| 6. Annular hole | Nested contours encircle an excluded hole in a curved patch | Use a radial corridor between the two boundary components, with ordered connections and accessible terminal exits | These loops are noncontractible on the support. The hole blocks straight shortcuts; different constant boundary values can support their circulating current. |
| 7. Cylinder bands | Circumferential rings at successive axial positions | Use one axial seam and a consistent helical progression between adjacent rings | Treat angular coordinates periodically. A parameterization seam is not a physical gap or a free crossover. |
| 8. Torus essential pairs | Oppositely oriented essential contours bounding annular strips | Group compatible parallel loops and route connections in a consistent homotopy class | A global real stream function on the closed torus requires oriented homology balance; an isolated unbalanced family is a different current model. |
| 9. Boundary arcs | Open contours traverse a patch and meet designated boundary ports | Connect endpoints in a controlled boundary routing zone, respecting their directions | A simple serpentine works only when its alternating traversals match the prescribed current directions; it cannot reverse every second required arc. |
| 10. Disconnected supports | Contour groups occupy separate panels or shells | Produce local windings and join designated ports with a planned off-surface lead pair | A continuous path cannot stay on disconnected supports while visiting both. External interconnection or separate electrical circuits are necessary. |

For every case, the displayed arrows should be derived from the same normal convention and stream function, rather than assigned to make an attractive connector work.

## 3. Critical contours and the family graph

Near a nondegenerate extremum, local coordinates give \(\psi-\psi_c=\pm(u^2+v^2)\): small closed contours appear or disappear. Near a nondegenerate saddle, \(\psi-\psi_c=u^2-v^2\): the critical level has crossing branches, and neighboring regular levels reconnect differently. The crossing is a singular level-set feature with zero current density at the critical point, not a regular wire intersection. Choose contour levels away from critical values and test sensitivity to small level shifts.

A Reeb graph collapses each connected level component to a point. Edges describe regular contour families; vertices record critical changes. For simple Morse functions on closed orientable surfaces, extrema give degree-one vertices and saddles give degree-three vertices. With boundary arcs, additional boundary and saddle vertex types occur; a bare graph does not always determine the surface. See [Izosimov and Khesin, Classification of Casimirs in 2D hydrodynamics, §5.2](https://math.arizona.edu/~izosimov/papers/casimirs.pdf).

For a disk-like arrangement, a containment forest is often an adequate practical structure. On a cylinder or torus, “inside” is not a universal ordering: record noncontractible classes and periodic adjacency. A Reeb edge gives a natural place to seek consecutive-turn connections, but a Reeb branch is **not** an electrical branch. A single series wire has degree two internally and exactly two degree-one terminals.

Degenerate critical points, plateaus, boundaries at critical values, nonmanifold meshes, and coincident contour segments require separate handling. Tiny loops and near-saddles can be sensitive to mesh resolution. Their removal changes the current distribution and needs a field-error check rather than purely cosmetic justification.

## 4. Homology and the limits of a global stream function

Three notions must remain distinct: a curve may be contractible, separating, or homologically trivial. They coincide in some simple settings but not generally. A cylinder ring is noncontractible yet separates the cylinder. On a higher-genus closed surface, a separating loop can be noncontractible while remaining homologically trivial.

The following useful constraint follows directly from the sublevel region. On a closed surface and at a regular value \(c\), the **complete oriented level set** is the boundary of \(\{\psi\le c\}\), up to the chosen overall sign. Therefore

\[
\sum_{C\subset\psi^{-1}(c)}[C]=0\quad\text{in }H_1(S;\mathbb Z).
\]

An individual contour need not be null-homologous. Essential torus contours can occur as oppositely oriented parallel pairs. On a surface with boundary, physical boundary pieces enter the sublevel boundary, so the closed-surface statement cannot be applied unchanged: \(\psi=z\) on an open cylinder legitimately yields same-direction circumferential rings.

On a torus, essential simple curves have primitive integer winding classes \((p,q)\). Two classes \((p,q)\) and \((r,s)\) have minimum intersection number \(|ps-qr|\). Consequently, disjoint essential regular contours must share the same unoriented slope; independent toroidal and poloidal families cannot simultaneously cross each other as regular levels of one scalar function. See [Juvan, Malnič and Mohar, Systems of Curves on Surfaces](https://www.sfu.ca/~mohar/Reprints/1996/BM96_JCT68_Juvan_SystemsofCurves.pdf).

A divergence-free current on a surface with handles can contain global circulation components that a single-valued real stream function alone cannot represent. Such cases need a suitable harmonic/cohomology basis or a cut surface with stream-function jumps. This is a modeling extension, not a reason to discard the current. [Bettini and Specogna, 2015](https://www.comphys.com/pdf/ieee_tmag_bem.pdf) treats this topological obstruction in thin conductors.

## 5. Open series chains, closed circuits, and bridge parity

Opening each of \(N\) contours once and joining directed endpoints can form an open series chain. At least \(N-1\) intercomponent joins are needed to connect \(N\) initial components if each join merges two components. This elementary count does not minimize length, establish geometric feasibility, or imply that an arbitrary minimum spanning tree is a valid serial wire.

A different operation cuts two loops and reconnects them with **two** bridges to obtain one closed component. There is a local orientation constraint. In a narrow rectangular corridor between facing cuts, the two orientation-preserving bridge pairings are noncrossing when the corresponding contour tangents oppose. When those tangents are parallel, the pairing alternates around the corridor boundary and crosses. Thus side-by-side equally circulating loops and nested oppositely circulating loops admit the familiar local band connection; the other two orientation combinations do not admit that same local construction. This is a corridor-specific obstruction, not proof that every longer open-chain route is impossible.

An open spiral and a completed electrical circuit are also different objects. A nested same-direction spiral commonly has an internal terminal. A short radial return drawn across its turns needs physical clearance on another layer or an explicitly checked three-dimensional route. For a planar support, a simple closed curve has winding number only zero or plus/minus one about a point off the curve; it cannot realize the multiple same-direction turns of a coil without a crossing or departure from the plane. A long return following interturn gaps can avoid crossings, but can substantially alter the field. Return leads must therefore be modeled, not hidden. Practical spiral connections and raised return paths are discussed by [Littin et al., 2021](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.699468/full).

## 6. What should “optimal” mean?

The current requested result is one continuous closed conductor with no prescribed terminals or final opening. Optimize the closing connection jointly with the other transfers. Earlier open-chain examples remain illustrative; they do not impose terminal constraints on this closed-loop problem. The current sequence is recorded in [CLOSED_LOOP_RATIONALE.md](CLOSED_LOOP_RATIONALE.md).

First reject infeasible candidates: broken continuity, incorrect turn orientation, forbidden crossings, insufficient insulation clearance, excessive curvature, inaccessible winding steps, or unacceptable field error. Check three-dimensional conductor envelopes, including remote surface regions that approach one another. Parameter-space spacing alone is insufficient.

The user specified the ranking on 22 September 2026:

1. **Minimize the number of jumps.** One continuous raised transfer is one jump, even when it crosses several turns. Separate raised transfers count separately; ordinary surface joins do not.
2. **Minimize B0 distortion, favoring straight returns parallel to B0.** Choose cut positions, group traversal states, transfers, returns and the final closure jointly to make such paths possible. Evaluate bends, removed contour portions and every connection of the closed conductor at unchanged current.

Apply this order lexicographically among feasible complete routes. Wire length does not override either priority. Preserve the directed turns and all declared field/manufacturing bounds. See [DESIGN_PRIORITIES.md](DESIGN_PRIORITIES.md) for the physical jump definition, the proposed field score, and the exact longitudinal-field property of B0-parallel straight segments.

The resulting design preference is **continuous local spirals, as few continuous raised transfers as feasible, and straight B0-parallel return spans with controlled end transitions**. The atlas and fixed-cut benchmark remain illustrative evidence; neither has established the best complete route under this newly specified objective.
