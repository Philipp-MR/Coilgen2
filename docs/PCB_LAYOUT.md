# PCB layout workspace

Open a saved routed result, then choose **Results → PCB layout…** in the desktop
or browser workbench. The browser also provides a PCB layout button beside
the winding editor. The File menu opens the same workspace for export.

PCB layouts are separate drafts bound to the saved winding. They do not alter
the stream function, routing result, active project, or original conductor export.
Install/update the `gui` or `web` extra to include Shapely, the copper geometry kernel.

## Working graphically

1. Select a **part** and a **layout source**: the saved winding, or the saved
   inward/outward group spirals on two copper layers. Switching the source clears
   that part's connection edits; Undo restores them.
2. On **Board**, move the opening seam or choose **Find seam between windings**.
   The proposed seam minimizes centerline cuts and then maximizes sampled angular
   distance to the winding. It does not guarantee that a seam without cuts exists.
   The outline comes from the support triangles, including openings.
3. On **Copper**, set trace width, copper gap, thickness and layer separation.
   Select traces directly in either preview or in the trace selector. The selection
   highlights in both views. Top/bottom visibility, zoom and Fit views help inspect
   dense winding patterns. Drag the installed view to rotate; right-drag to pan.
   On narrow screens controls collapse to leave room for the previews. The step
   buttons reopen them; **Hide controls** returns to the graphical view. Browser
   scrolling over a narrow preview scrolls the page; Ctrl+wheel zooms the preview.
4. On **Connect**, click two source endpoints in either preview (A, then B),
   or choose them from the selectors. Blue **A** and purple **B** markers appear in
   both previews. The purple proposed link reports length, layer transitions and
   local clearance concerns before you add it. Hover endpoints for their names.
   Add the straight connection on the selected copper layer. Where layers differ, the candidate contains explicit
   layer transitions and landing copper. Choose the two terminal endpoints, and
   remove unwanted connections individually or with Undo.
5. On **Review**, findings are grouped by cause, with seam and circuit problems
   first. Select a group to highlight its affected trace, use **Next finding** to
   inspect repeated witnesses, and follow the contextual repair action. Problems
   with spatial witnesses center and zoom both previews. **Compare saved path**
   adds the original path in purple to the installed view and marks the maximum
   saved-to-PCB displacement, including the copper-layer offset. Complete the terminal-to-terminal
   circuit before running the field/electrical evaluation.
6. The persistent review summary shows unresolved checks, circuit completeness and
   field-budget status. Export **SVG**, **DXF**, or both as a new ZIP package.
   The button says **Export inspection artwork** until checks and field budget pass.
   Both clients save a source-bound `.cgpcb` draft under the workspace's
   `pcb-drafts` directory after each edit and restore it after restart. A failed
   save is reported; an unreadable draft is never silently overwritten. Treat the
   workspace as a single-editor workspace rather than a collaborative document.
   Save/open draft copies to transfer work; the browser's **Prepare draft download**
   creates a clearly named **Download PCB draft** link. A draft is accepted only
   for its matching saved source. Field evaluation must be rerun after restart.

The supported developments are exact plane coordinates and circular-cylinder
coordinates. A cylinder is fitted from face normals and a radial circle fit, with
geometry checks before use. Other surfaces are rejected rather than projected
with undisclosed distortion. Faceted/lifted saved wires can move when developed
onto the ideal support; the maximum displacement is reported. A hemisphere is
not developable into a plane without distortion: its PCB evaluation and export
are disabled. A segmented or distortion-aware PCB workflow is not implemented.

## What the checks cover

- Copper polygons, support openings, board-edge clearance and out-of-board copper.
- Same-layer spacing between different traces and nonadjacent turns of one trace.
- Centerline self-crossings and copper near layer transitions.
- Real seam cuts, disconnected paths, unintended endpoint branches, and the two
  assigned terminal endpoints. Opposite board edges are never electrically joined
  just because the flex will wrap around a cylinder.
- Positive separation for two copper layers.

Checks apply to the selected part. The intended electrical topology is one series
circuit between its two terminals. Parallel circuits need a separate design model.
Clearance checks exclude only a local bend/join neighbourhood; they still inspect
long parallel return legs. A workload limit reports an incomplete check rather than
silently passing an oversized winding.

**Evaluate PCB coil** uses the assembled PCB path at the saved current. It reports
target RMS and field change with the selected part substituted; other parts retain
their saved paths. The source design's field budget and design margin are retained.
Curved PCB intervals are sampled with at most 1 µm cylindrical chord error. The
model is a thin wire; trace resistance and power exclude via/contact resistance.
Changing geometry or rules invalidates the evaluation. This is not a thermal,
RF, bending-fatigue or fabrication qualification.

## Export contract

The ZIP contains separate top copper, bottom copper and outline SVG/DXF files,
`layout.cgpcb`, `pcb-manifest.json`, and an import guide. Empty layers remain empty.
The manifest identifies the source and settings and records unresolved checks.

- All files use a common origin and 1:1 millimetre scale.
- Both copper layers are viewed from above: do **not** mirror a layer for PCB CAD.
- SVG copper uses filled polygons with holes, no white background or decorative
  board outline. SVG Y-down and DXF Y-up encode the same physical geometry.
- DXF exports closed copper outlines, not net-aware native PCB tracks. Convert or
  assign copper deliberately in the receiving CAD application.
- Support cutouts are retained in outline files.
- Layer transitions contain landing copper, but drill diameters, annular rings,
  plating, solder mask and a fabrication stackup must be defined in PCB CAD.

Export remains available for inspection when issues exist. The package identifies
itself as **PCB CAD artwork**, not a fabrication release. It never overwrites an
existing package. Arbitrary board tabs, pad editing, freehand copper rerouting and
Gerber/native-board generation are not part of this workspace.
