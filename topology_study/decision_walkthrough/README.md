# Twelve illustrated interconnection checks

**Scope update:** the user now requests one continuous closed coil with no prescribed terminals or wire opening. The A/B terminal assumptions in these earlier schematics are superseded. Use the [revised closed-loop rationale](../CLOSED_LOOP_RATIONALE.md) for current decisions; these figures have not yet been redrawn for that boundary condition.

These twelve schematics follow the decision sequence agreed in the study. Priority 1 is the number of continuous raised transfers. Priority 2 is routing-induced longitudinal field error, favoring straight returns along B0. A continuous raised span over several turns is one jump.

Teal denotes retained turns, orange denotes surface connections or end transitions, purple denotes raised transfers, and red denotes a rejected condition. Geometry, requirements and candidate scores shown symbolically are illustrative. There are no newly computed optimal coil layouts or field-error values in this walkthrough. Current arrows specify direction; A and B are terminals. In the geometric examples B0 lies in the support plane, along the world x axis.

## 1. Fix the requirements

Use the same terminal positions, field-evaluation region, B0 direction, conductor dimensions, and physical/field limits for all alternatives. The ROI is drawn below the support; its diagram is an evaluation region, not another wire.

![1. Fix requirements](figures/check_01.png)

## 2. Identify topology and circulation

Distinguish nested turns from separate families and preserve the prescribed arrows. A containment relation describes the contours; it is not an electrical connection or branch.

![2. Identify topology and direction](figures/check_02.png)

## 3. Find continuous local winding sequences

Connect neighboring opened turns through a coherent seam to form a spiral. Its remaining inner terminal must enter the complete return design. The depicted original circles and spiral retain the same circulation.

![3. Wind adjacent turns](figures/check_03.png)

## 4. Retain alternative ports

Vary the cut positions and whether the group is visited inward or outward. E and X mark entry and exit states. Reversing the radial visit order does not reverse current around a retained turn.

![4. Keep alternative ports](figures/check_04.png)

## 5. Start with the smallest jump budget

Investigate zero jumps first unless a valid lower bound says otherwise. The inset deliberately distinguishes three turn crossings from one continuous raised transfer. It does not assert that the zero-jump budget is impossible for every example.

![5. Set jump budget](figures/check_05.png)

## 6. Construct the complete interconnection

Include the entire A-to-B winding, both terminal leads, and all group returns while choosing order, cuts, and group traversal states. The dashed link represents the common external feed closure. This particular drawing is a topological routing candidate: corner rounding, spacing, support, and field checks have not been applied to certify it.

![6. Connect the complete route](figures/check_06.png)

## 7. Consolidate raised transfers

Compare separate optional rises/landings with one continuous elevated span between the same endpoints. Consolidation may reduce the agreed jump count, but cannot skip a required surface winding interval or a mandatory landing. Verify support and permitted height for the longer span.

![7. Consolidate jumps](figures/check_07.png)

## 8. Align returns with B0

Place cuts and ports so the long return can be straight and parallel to either sign of B0. For the ideal filament model that straight span contributes zero longitudinal field. The rises, landings and end bends remain part of the field evaluation; a transverse field contribution is still possible. The physics and scope are given in [the design priorities](../DESIGN_PRIORITIES.md).

![8. Arrange returns along B0](figures/check_08.png)

## 9. Reject invalid complete routes

Check the actual conductor envelope, its clearance, bend radius, continuity and prescribed circulation. A centerline without intersections is insufficient. Also verify that the chosen winding sequence and its tooling remain accessible as earlier turns are placed.

![9. Reject invalid routes](figures/check_09.png)

## 10. Evaluate the full field difference

Compare the complete routed conductor with the original directed contours at the same current. Both missing cut intervals and added connectors/leads enter the difference. Assess its projection along B0 over the ROI and other required regions, with declared RMS/peak criteria and the same feed closure. No numerical field errors are invented in this diagram.

![10. Evaluate full field error](figures/check_10.png)

## 11. Exhaust the current jump budget

Explore the declared family of cut positions, group orders, and port states at fixed J. A passing candidate is saved while its alternatives are considered. Raise J only after the current family's failure is established; an interrupted or bounded search is unresolved and cannot prove global impossibility.

![11. Exhaust the jump budget](figures/check_11.png)

## 12. Select and recheck

Among physically and electromagnetically feasible candidates, first minimize J, then minimize the agreed longitudinal field error. The symbolic candidate rows illustrate that order; they do not name a measured winning coil. Recheck continuity, turns, conductor geometry, jump count and fields after smoothing and conductor generation.

![12. Choose and recheck](figures/check_12.png)

## Reproduce the drawings

From `CoilGen2` with the existing environment:

```powershell
.\.venv\Scripts\python.exe topology_study\decision_walkthrough\build_walkthrough.py
```

This uses only NumPy, Matplotlib and the local drawing modules. All work and PNG/SVG outputs remain inside the study folder. The three overview sheets are `figures/checks_1.png`, `checks_2.png`, and `checks_3.png`; corresponding SVGs and twelve individual figures are also generated.
