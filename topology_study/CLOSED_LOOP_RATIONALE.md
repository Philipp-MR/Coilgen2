# Interconnection checks for one continuous closed coil

Current scope: one continuous closed conductor, with no prescribed terminals or final wire opening. This replaces the terminal-constrained assumption in the earlier walkthrough. The two priorities remain: first minimize continuous raised transfers; then minimize longitudinal field distortion, favoring straight returns along either sign of B0.

1. **Define the required closed conductor.** The result must be exactly one connected series cycle, with no free ends, electrical branches, or separate disconnected cycles. The start of a stored point array has no physical significance.

2. **Preserve the intended winding currents.** Retain every required directed contour interval and turn multiplicity. Interconnection must not reverse a turn or discard one to simplify the route.

3. **Identify contour families and access restrictions.** Establish nesting, separate groups, circulation, support boundaries and holes. Distinguish the contour-family graph from the electrical cycle.

4. **Generate local group states.** Consider inward and outward progression through neighboring contours and alternative seam/cut positions. Entry and exit ports are internal construction variables. Every temporary opening must ultimately be reconnected; no final terminal is introduced.

5. **Start with the smallest jump budget.** Investigate zero continuous raised transfers, or a larger lower bound only if established for the actual problem. Ordinary surface joins between contours do not count as jumps.

6. **Construct complete cyclic group orders.** Choose cuts, local states and group order jointly, including the connection from the last visited group back to the first. Require one global cycle: giving each group a predecessor and successor can otherwise leave several independent cycles. Do not select an open chain and append its closing connection afterward.

7. **Consolidate necessary raised transfers.** Prefer one uninterrupted raised span over several turns to several separate rises and landings when geometry and support allow. Do not skip required surface winding intervals or mandatory landings. Count physical raised spans on the closed cycle; a span crossing the arbitrary array start/end counts once.

8. **Arrange axial returns within that jump budget.** Choose internal ports that line up in the plane perpendicular to B0. Use long straight B0-parallel spans with controlled end transitions where possible. Include the closing connection in the same search and allow either current direction along B0. Alignment is a construction preference; the complete field calculation decides distortion.

9. **Reject physically invalid cycles.** Check closure, directed turn coverage, electrical continuity, conductor-envelope clearance, bend radius, lift height, support and winding access. An insulated overpass is not an electrical junction. A geometrically closed polyline alone is not a valid manufactured coil.

10. **Evaluate the field of the complete closed route.** At unchanged current, compare against the original oriented contour set, including all removed arcs, added joins, raised transitions, returns and the closing connection. Evaluate longitudinal error over the same ROI and other required regions. External feed wires or a supply opening are outside the current model.

11. **Resolve the current budget before accepting another jump.** Explore the declared cut/order/shape alternatives at fixed jump count. Save all feasible contenders and compare their field error. If none is feasible, broaden the alternatives or increase the budget. An unfinished finite search is not proof that a lower-jump physical route is impossible.

12. **Select and verify the final closed coil.** First choose the lowest feasible jump count established by the search; among those candidates choose the lowest agreed B0-error score. Length is secondary to those priorities. Recheck the complete cycle after smoothing and conductor generation, including its final-to-first join and cyclic jump count.

With strictly ordered priorities, field error only breaks ties at the minimum feasible jump count. A long surface return can avoid a lift while opposing much of the intended winding field. Such a route is not excluded unless it violates a specified field-error limit; no numerical limit has been set here. Nested contours alone therefore do not establish a universal positive jump lower bound when arbitrary surface returns are allowed.

The existing open-route atlas and benchmark retain their original scope; their results do not establish the optimum of this closed-loop problem. Earlier figures showing A/B terminals illustrate a superseded boundary condition and must not be treated as current terminal requirements.
