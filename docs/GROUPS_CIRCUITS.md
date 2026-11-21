# Groups and circuits

Both workbenches provide **Groups & circuits** beside the central plot. The
editor uses the same contour snapshot and transactional plan in desktop and
browser; it does not introduce another stream-function optimization model.

1. Choose **Prepare groups**. The numerical worker computes through the contour
   stage, reusing compatible preparation and stream-function checkpoints. This
   temporary request preserves the project's selected computation stage.
2. Inspect the numbered, colored source groups in 3D or XY/XZ/YZ. Select a group
   in the plot or table. The table records support ownership, containment
   hierarchy and contour count; checkboxes select groups for assignment.
3. Use **Connect all**, **Separate per group**, **Join selected** or
   **Separate selected**. A shared circuit name joins those groups; assigning
   selected groups never removes the others. Set each group's spiral order and
   cut side, or leave either on **Auto**.
4. **Apply routing plan** commits one undoable project revision. Recompute
   routing or the full design, then inspect field and engineering checks.
   **Revert group edits** discards the uncommitted draft.

Opening the editor and accepting an unchanged initial draft leaves
`routing_group_plan=None`. Existing single-circuit/per-support behavior,
optimization defaults and field-current policies remain unchanged. An explicit
group plan takes precedence over the legacy `force_cut_selection` list.

Undo and Redo restore the group-table choices together with the project plan.
This reuse requires the same project owner and unchanged configuration and study
inputs apart from the plan itself. Other model changes clear a clean editor.
An uncommitted draft stays visible with Apply disabled until it is reverted and
the current groups are prepared again.

## Engineering contracts

- **Spiral order** changes the sequence of visiting contours, preserving their
  prescribed current circulation. It is not a current-polarity switch. Explicit
  spiral choices constrain both initial connections and joint routing proposals;
  explicit cut sides remain binding during cut search.
- A plan partitions every source group exactly once. All resulting circuits use
  the same finite, nonzero evaluation current. Independent current fitting per
  circuit is not introduced. External leads, terminals and supply connections
  still require their own design and review.
- The existing routing objective and acceptance limits remain in force. Regular
  routing prefers fewer continuous raised transfers among validated candidates.
  Search budgets do not prove a global minimum. Raised transfers are not PCB vias.
- Plans bind to exact source meshes, directed contours, current, multiplicity and
  group membership. Regeneration that changes this fingerprint rejects stale
  assignments. Preparing the new groups exposes a fresh draft for review; old
  assignments are not automatically matched to new group numbers.
- The route check verifies directed retained intervals outside recorded
  inter-group opening balls and requires every source contour to remain
  represented. Full field and fabricated-body checks remain separate.
- Fields sum the individual circuit paths. Electrical evaluation produces the
  full self/mutual inductance matrix and per-circuit resistance. Self-inductances
  are not summed into an inferred complete-system inductance.
- Swept circuits undergo the existing body-pair clearance audit, including pairs
  on the same physical support. Failed checks remain failures and preserve DRAFT
  export handling. Shared supports receive one former containing all grooves.

## Script and saved-result model

`CoilGenConfig.routing_group_plan` is optional JSON with schema
`coilgen.group-routing/1`, a snapshot `fingerprint`, a circuit `mode`, and one
`groups` entry per source group. Entries contain `key`, `circuit`, `spiral` and
`cut`. Supported modes are `existing`, `custom` and `per_group`; the graphical
assignment shortcuts produce explicit `custom` partitions.

```python
from dataclasses import replace
from coilgen.application.group_editor import GroupEditor
from coilgen.pipeline import CoilGenPipeline

contours = CoilGenPipeline(config).run_stage("contours")
editor = GroupEditor.from_parts(contours.parts, config)
editor.assign("per_group")
configured = replace(config, routing_group_plan=editor.plan)
result = CoilGenPipeline(configured).run_stage("evaluated", contours).result
```

With an explicit partition, `result.coil_parts` contains `RoutedCircuit`
numerical contexts using the existing single-path kernels. These carry
`circuit_id`, `source_group_keys` and `source_support_indices`; a numerical
circuit index is not a new physical support. The original combined mesh and
stream-function solution remain the source of the continuous surface field.
`result.source_parts` retains each original support once for independent field
validation and current-density export; circuit contexts do not duplicate its
FEM field operator.

Source keys use zero-based `s0:g0` indices; displayed `S1 / G1` labels are
one-based. Saved routing evidence and CAD package manifests retain the circuit
mapping. Detailed circuit CSV exports add circuit and source-ownership columns;
legacy exports retain their original columns. CSV/STEP paths stay separate and
never acquire a fictional connecting segment. STL coordinates and STEP units
retain the existing declared-unit contract.

Checkpoint schema 8 stores the optional plan, circuit identity, spiral
restrictions and original stream-function supports. Earlier supported schemas remain inspectable with defaults for
these additions; a numerical source change still prevents unverified resumption.

**Advanced: Routing** opens the current settings draft at its relevant section.
Desktop and browser searches accept native keys and category names, including
regularization. The Omega holder preset is omitted from both interfaces;
existing custom polygons and the scripted profile API are preserved.
