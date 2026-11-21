# Interconnection editor

Compute automatic routing, select its saved run in Results, then choose **Edit
interconnections**. The same command is available in the View menu. The existing
**Interconnections** inspection window remains available for saved evidence.

The editor opens the verified routing checkpoint and its recorded settings. It
does not reconstruct electrical connectivity from displayed polylines or alter
the original run. Each circuit keeps its source contour directions and original
automatic route.

## Browser workbench

Open a saved routed run, then choose **Results → Edit interconnections**. The
browser uses the same source-bound Python editing core and engineering checks as
the desktop. A drag previews locally and commits one transaction on release.

- Select a section in the 3D view or the searchable list. Drag its **CUT** or
  **GUIDE** handle; the inspector also accepts precise millimetre coordinates.
- Drag empty space to orbit, Shift-drag to pan, and scroll to zoom. View direction
  presets and **Fit view** make front and rear connections easier to reach.
  Surface picking stays on the side of the support nearest the original handle.
- **Save draft** saves a `.cgwinding` file in the browser workspace. Open **Saved
  drafts & import** to reopen it after restarting the workspace, or download it
  for import into another browser/desktop session with the same automatic route.
- **Back to workbench** retains the draft in the running session. Save before
  closing/restarting the server. Reopening a saved draft clears its validation
  receipt; run **Validate engineering** again before applying.
- **Download validation report** preserves the exact revision identity, fixed
  current field comparison, manufacturing checks, and rejection reasons as JSON.

Locks, cut/reconnect, per-circuit undo/redo, automatic routing through guides,
cooperative cancellation, and apply-to-model checks have the same semantics in
both interfaces. Stale browser messages cannot overwrite a newer edit. The
browser's transparent 3D drawing is an editing view; engineering acceptance uses
the full numerical geometry, not the displayed pixels.

## Editing a winding

Cutters are persistent objects. Select a cutter in the view or its compact
object selector to edit all associated openings together. The lock action on a
selected cutter locks its member cuts atomically. Group buttons isolate a group
and select it as the cutting target. **Place on coil** places a new cutter at a
clicked surface point; center, size and tilt handles remain available afterward.
Choose **Selected turn** to restrict a plane, sphere or box to one opening;
changing the scope recomputes the preview before Generate. Desktop edit handles
and warning markers keep a fixed screen size while zooming, and handles take
priority over warning markers when they overlap.

Cut and seam proposals now include the neighboring routed connections before
commit. Teal proposals are computed by the same routing core on a detached draft.
**Generate connections** commits the reviewed proposal; discard leaves the
winding unchanged. Pale alternative openings can be clicked to choose another
intersection of the same contour. Invalid or locked proposals cannot commit.

Drawing from an output endpoint to a compatible input also creates a proposal.
The input highlights when it will snap. Generate commits it and switches to
selection mode, exposing its simplified guide handles. Existing over/under
bridges have a height handle; moving it preserves attachments and ramp span.
On non-developable supports, crossing discovery uses projected physical facet
polylines, without substituting a flat chart.

**Hide rear wires**, group isolation and **Face selected area** reduce clutter.
**Local view** is a tangent-oriented orthographic projection, labelled as
distorted; it is not a metric-preserving unrolling of a hemisphere. Larger
picking targets do not change the numerical wire geometry.

**Winding preview** follows the directed circuit, with play/pause and previous/
next steps, a moving current marker and bridge-order notes. Incomplete circuits
cannot play. Playback is visual inspection, not an accessibility or fabrication
certificate.

The issues panel separates approximate centerline warnings, failed engineering
checks and unavailable certificates. Clicking a recorded bend, support or copper
clearance witness locates its verified conductor geometry. Global field-change
failures locate the worst active target sample when available. Checks without
recorded witnesses remain unlocated. Quick warnings never grant acceptance;
full validation and Apply are disabled while an uncommitted preview is displayed.

**Cut shapes and planes** defines contour openings with physical 3D volumes,
including on hemispherical supports. Choose **Cut plane**, **Sphere**, or **Box**,
then select the contour group. Drag the center to position the volume, white
handles to resize it, or the blue normal handle to tilt a plane or box. Axis
presets and dimensions in millimetres are available in the compact tool row.
The plane's **Gap** is the full slab thickness; sphere dimensions are diameters,
and box dimensions are full side lengths.

Red arcs preview the contour openings. **Generate connections** applies
all previewed cuts and reroutes their neighbors in one undoable transaction.
The × button discards the preview. If a shape crosses a turn several times,
the preview chooses the interval nearest that turn's existing opening; inspect
the highlighted arcs before applying. A cut must leave more than half of its
original contour. Locked cuts or neighboring connections reject the entire edit.
Shape definitions persist with their cuts in saved drafts and undo/redo.

Cut shapes operate on exact 3D contour segments and do not require a flat chart.
Flat views and freehand flat seams remain limited to planar, cylindrical and
conical supports. On a hemisphere, use 3D cut volumes and connection drawing.
Editing support does not imply engineering acceptance: the independent support
validator can still reject an unsupported surface, and existing conductor,
clearance and field constraints remain enforced.

- Select a connection in the 3D view or section list. Drag its handle to guide
  the route. Add, remove, or position guide points with the inspector; numeric
  coordinates are in millimetres. Drag empty space to orbit the camera.
- Select a contour with a complete, attributable opening. Drag its cut handle
  along the contour, or enter its directed arc-length position and opening
  length. Both neighboring connections update in the same operation.
- **Lock** a section to exclude it from edits and automatic routing. A cut cannot
  move while either neighboring connection is locked. Locks constrain the route
  centerline; physical conductor realization still performs its normal checks.
- **Generate connections** uses the existing surface geodesic solver through the
  specified guide points. It preserves locked sections and disconnected draft
  edges. These proposals are not clearance certificates.
- **Cut connection** opens the selected connection; **Reconnect** restores its
  directed adjacency. The status bar reports components and free ends. Arbitrary
  branch creation and reassignment between circuits are not part of this editor.
- **Undo/Redo** work per circuit, including Ctrl+Z/Ctrl+Y. A completed drag is one
  operation. Esc cancels an in-progress drag. A routing failure commits nothing.
- **Show automatic route** provides a reference overlay. **Save draft** writes a
  portable `.cgwinding` edit plan; **Open draft** requires the same source route.

Surface connections remain attached to the mesh. Raised-surface guides preserve
their selected connection's normal offset; the route interpolates endpoint
offsets along the support. Explicit free-space guides move in the camera plane
at the picked depth, with XYZ controls available for precise placement.

Ambiguous ownership and incomplete contour arcs remain visible but cannot be
treated as movable complete cuts. This includes contours split by inter-group
openings. The editor preserves them instead of guessing how to reconstruct a
turn. A closed automatic circuit remains closed unless a connection is explicitly
cut; the editor does not invent feed leads or turn geometric closure into a
claim about external electrical connections.

## Engineering validation and application

Every edit changes the revision identity. **Validate engineering** checks a
detached revision in a background worker, using:

1. Directed section continuity and immutable source contour geometry.
2. Production conductor realization, bend and profile checks, and the exact
   swept-triangle collision audit.
3. Independent support conformity and clearance between complete circuit bodies.
4. Route and physical-centerline fields on the recorded target grid at the
   recorded current of each circuit. No current or mean-offset refit is used.
5. Existing routing vector-change and optional longitudinal-change budgets,
   layout/gradient requirements, and suppression constraints.
6. The completed design review's norm/RMS field-error budget at the original
   current, using the budget ratio from the verified saved request. A small
   routing change cannot waive a failed absolute or relative field budget.

The progress display follows numerical work. **Cancel operation** requests
cooperative cancellation at the next numerical checkpoint.

The inspector reports fixed-current longitudinal RMS and peak changes plus
failed checks. The validation report must match the current revision before
**Apply to design** is enabled. Missing field/current context, disconnected
drafts, exhausted validation budgets, and failed manufacturing checks cannot
produce a passing report. A failing draft can still be saved.

When bounded corner repair is already enabled, failed exterior sweeps on
verified cylindrical, conical and spherical-cap supports can propose physical
bend smoothing and local crossing relief. A narrow radial transition is tried
first to avoid unnecessarily displacing long contour runs and attenuating their
field; difficult bends can use a broader candidate. Every proposal must pass the same bend,
frame, finite-profile, triangle-quality, copper-gap, lift and support checks. A
continuous directed correspondence bounds the complete repaired path against
the immutable source, including corners skipped by resampling. The configured
source-deviation bound is unchanged. Exact body checks use the actual support
triangles; a hemisphere is certified as a convex cap with its real rim, without
assuming that vertex clearance alone proves triangle clearance.

The accepted physical realization is shared by path-only field evaluation and
full sweeping. Path-only evaluation discards its temporary body and does not
grant export readiness. No current, field budget or manufacturing tolerance is
adjusted by the repair; the edited and automatic physical fields are evaluated
again on the same recorded target grid.

Apply creates a normal undoable project revision containing `routing_editor_plan`.
The active model must match the saved source settings and have no pending model
draft. Compute the winding again to update its fields, electrical estimates,
former and fabrication exports. Pipeline execution reproduces automatic routing,
checks its exact source fingerprint, applies the manual plan, and repeats the
engineering validation. A changed automatic route requires reconciliation;
manual coordinates are never silently transplanted onto different contours.

## Recomputed winding verification, 2026-11-21

The saved browser edits were applied to three real workbench designs and fully
recomputed at their original currents. The exported millimetre STL triangles
and CSV centroid paths were reopened and checked against the same bend, frame,
source-deviation, support, copper-clearance and fixed-current field constraints.
No constraint limits were relaxed and no current or field offset was refitted.

| Edited design | Current [A] | Exported field RMS error | Limit | Minimum bend radius [mm] | Outcome |
|---|---:|---:|---:|---:|---|
| Cylindrical Gy | 18.515568 | 4.70345% | 5% | 3.34214 | Passed |
| Hemispherical Gy | 16.082431 | 4.03094% | 5% | 3.67217 | Passed |
| Conical Z2 | 1.827142 | 36.68443% | 5% | 3.03463 | Field failed; geometry passed |

The Z2 design deliberately retains its exact saved edits. Its unconnected
contours already miss the final field budget; a small acceptable route change
does not establish overall acceptance. Its exports remain inspection-only.
Desktop and browser reopen the saved results with the matching winding and
field visible, including the failed verdict. These are numerical engineering
checks, not a physical construction release.

Large checkpoints store repeated segment provenance compactly and use stable
archive metadata so identical boundary/final checkpoints deduplicate in saved
projects. The numerical values, source fingerprints and verdicts are preserved;
the checkpoint and browser size limits remain unchanged.

Edited checkpoints retain `routing_editor_baseline` so later editing sessions
can reopen the same automatic baseline and saved edits. Disabling/removing the
manual plan restores the normal automatic-routing workflow.

Validation establishes the recorded digital checks only. Physical winding
accessibility, fit, insulation, and hardware commissioning still require the
usual engineering review.

## Regression checks

```powershell
python -m pytest tests/test_interconnection_editor.py tests/test_gui_interconnection_editor.py
python -m pytest tests/test_browser_interconnection_editor.py
node --test tests/browser_winding_geometry.test.cjs
```

The native OpenGL mouse-picking check is opt-in because Windows Qt's offscreen
OpenGL backend can terminate the process. Run it on a working desktop backend:

```powershell
$env:QT_QPA_PLATFORM = 'windows'
$env:COILGEN_NATIVE_GUI_TESTS = '1'
python -m pytest tests/test_gui_interconnection_editor.py -k native
```
