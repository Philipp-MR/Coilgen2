# CoilGen2 native scientific workbench

Created and maintained by **Philipp Amrein — MRI hardware engineer**.
**Help → About CoilGen2…** shows the software credits, project link and original
CoilGen publication with all coauthors. Newly generated reports and exports
include software provenance, separate from authorship of the coil design.
**Help → Contact / Kontakt…** opens the supplied email, GitHub profile and project
support links. **Help → Impressum…** shows the project notice. Both are available
before opening a project. Contact details are documented in [CONTACT.md](CONTACT.md).
The Impressum includes a separate Martinos Center acknowledgment expressing
Philipp Amrein's personal gratitude, with the official logo and website link.

The desktop workbench is an optional PySide6/Qt interface to the same Python
numerical engine used by existing CoilGen scripts. Qt owns the model editor and
graphics; isolated numerical workers own computation. No browser server is used.

Coil systems show a labeled **Editing channel** selector above the plot. Its
**Include in system compute** checkbox controls numerical participation.
The plot's **System meshes** menu independently shows or hides each channel's
support surface, with **Show all** and **Hide all** actions. Available supports
are visible together by default in the shared SI world frame. Visibility is
saved with the view; unavailable geometry is grayed out with its reason.
Saved results retain their stored support geometry, while any other-channel
overlays are identified as current model supports.

The right-hand **Design plan** panel shows every channel's requested computation
steps vertically. Pending steps are gray; recorded completed steps use the normal
black text. Running work is blue, execution failures red, and stopped or outdated
evidence explicitly labeled. **Cached** means a verified checkpoint can be reused;
steps beyond a channel's requested boundary are **Not requested**. Completion
describes computation, while engineering failures remain separate review findings.

For systems, the panel shows completed-step and channel counts, individual worker
percentages and excluded channels. It does not average percentages from unequal
independent studies. Double-click a channel header, or press Enter, to select its
editor; **Follow run** explicitly attaches the plot to the indicated worker.
**Compute enabled channels** submits each included channel's own study. **Jobs**
and **Recompute plan** open detailed evidence. Close the dock to give the plot more
space; **View → Design plan** restores it, and starting a computation opens it.
Its visibility is saved with the project view. The lower-left computing indicator
continues to cover other background operations as well.

The persistent activity strip also shows worker names, current operations and
individual percentages when the plan is closed or the plot shows an earlier run.
**Tasks…** opens a live list of every active or queued worker and local operation,
including field analysis, exports, model/system support previews and loading saved
stage data. Owned CoilGen editor and evidence windows mirror the same activity,
so a maximized analysis or settings window does not hide system computation.
Work without a measurable total shows a busy state. Synchronous project/result
loading, project saving and Python input exports paint their operation before
reading, displaying or writing large data.

## Install and start

From the repository environment:

```powershell
python -m pip install -e ".[gui,plot]"
python -m coilgen.gui
```

On Windows, double-click `Start CoilGen2 GUI.cmd`. It uses this checkout's
`.venv` when present, resolves its local source independently of the starting
directory, and keeps startup errors visible in the console. Progress messages
identify interpreter selection, library loading, Qt/3D initialization and when
the workspace is ready. To open a saved project:

```powershell
python -m coilgen.gui path/to/design.coilgen.json
```

**File → Open** reads the project, verifies embedded results and local history,
and checks reusable checkpoints in a background task. The status bar shows the
current operation and **Cancel opening** while the current project remains
usable. Editing inputs or notes, changing channel, closing the workspace, or
starting another open cancels the pending replacement. Invalid documents retain
the previous project and its drafts. Cancellation takes effect between verified
operations; it cannot interrupt an individual JSON decode or atomic archive write.
The synchronous Python `open_project(path)` API retains its existing contract.

Use `--background` to open without taking keyboard focus. Automated Qt/VTK tests
can exercise the interface without desktop mouse or keyboard input. Native
computer-use screenshots can inspect a covered window; the computer-use input
API activates its target, so interactive input is unsuitable while another
application needs to retain focus. Covered-window screenshots can include the
covering application's pixels; use accessibility-only capture in that situation.

`coilgen-gui` is also installed as a console entry point. The base package and
headless project API do not import Qt or VTK. Matplotlib remains available for
scientific figures and existing script plotbooks.

## Scientific workflow

### Coil results and field analysis

Selecting a saved result automatically displays an overview above the viewport:
RMS and peak field error, relative RMS error, recorded current, resistance,
inductance, mass and conductor volume. The source, field component, current policy,
sampling scope and engineering check status remain visible beside these values.
Missing quantities are identified explicitly. Original-current winding results are
preferred to refitted fields when that evidence exists.

The overview shows **Per body** when verified body properties are available but
no complete-circuit value is recorded. Open **Coil stats** for those rows; neither
self-inductances nor swept mesh volumes are combined into an inferred aggregate.
Field-budget cards use the actual positive-weight sample count and recorded
component. Skipped electrical calculations are unavailable, rather than zero.

Use the prominent **Coil results** toolbar button, the overview's direct actions,
or the entries under **Results and provenance** in the Model Builder to open
metrics, slices, errors, harmonics, coil properties or engineering checks. The
existing **View → Detailed field analysis…** and viewport **Inspect** actions are
also available. Target-only preparations disable field pages; saved electrical or
geometry statistics remain accessible without field samples. Failed evidence
verification disables result actions. Each modeless analysis window stays bound
to its verified run while the active model or channel changes; pending model edits
are not applied by opening it. Verification runs in the background, and opening
**Coil stats** does not start field calculations. Metrics and errors use native
samples first; physical slice evaluation starts when slices are requested.

Choose continuous-sheet, routed/contour, realized-conductor, independent-validation
or named region/pose samples where those fields were calculated. Recorded winding
current and original stream-function current are distinct choices when both exist.
Immutable stored field maps disable the wire-current selector and explain their
fixed policy. Returning to a routed or contour map retains the earlier selection.
The utility provides:

- A rotatable 3D point cloud in world metres, with Bx, By, Bz, projection along the
  recorded B0 direction, or field magnitude, plus numerical samples and statistics.
- Multiple XY, XZ or YZ slices on a regular grid, with common color limits and the
  target-volume section indicated. Authored spheres, ellipsoids, boxes, cylinders
  and verified enclosures retain their geometry and registration. An imported
  sample cloud shows its coverage hull rather than an invented target volume.
- Desired field, signed component residual, absolute component error and vector
  residual norm; histograms/profiles and RMS, peak, standard deviation and absolute
  95th percentile. Equal-sample and saved-weight statistics are explicit. Percent
  maps use one global desired RMS or peak; desired zero crossings do not create
  pointwise divisions. A zero desired field requires physical error units.
- Real solid spherical harmonics for scalar Cartesian components or B0 projection,
  with explicit center, reference radius and maximum degree. Actual, desired and
  residual coefficients use one common basis and sampling. Rank, singular values,
  conditioning and reconstruction error expose unsupported or poorly conditioned
  coefficients. Field magnitude and vector-error norms are excluded from this fit.
- PNG plots, CSV sample vectors/metrics and a complete JSON analysis bound to the
  displayed selection. JSON includes sampling, units, source/current provenance,
  slice grids, unavailable masks and harmonic diagnostics.
- **Coil stats** groups recorded circuit and per-body quantities: resistance,
  self-inductance, current, Joule loss, path length, cross-section, material and
  insulation settings, volume, contour/mesh counts and available loads or drive
  data. Complete-circuit estimates and per-body values retain their scope;
  per-body inductances are not summed without mutual coupling. Nominal
  cross-section-times-length volume and faceted mesh volume remain distinct.
  Rejected geometry is labelled as diagnostic evidence. Mass requires recorded
  mass or an explicitly entered density; no material density is inferred from
  resistivity. Density-based estimates state their assumptions and do not include
  unmodelled insulation, feeds or formers. JSON exports keep saved statistics and
  user density estimates separate.

Slices directly reevaluate the saved geometry and unchanged signed currents when
the source contract is available and reproduces the saved map. Otherwise the
display explicitly identifies linear interpolation, with blanks outside sample
support, across large gaps or at source singularities. New native snapshots record
the executed analytic target normalization, so a new grid does not rescale the
desired field. Older snapshots without that contract retain descriptive desired
interpolation. Progress and cancellation remain visible during background work.

These are diagnostic views, not new engineering acceptance tests. Saved weights
are not assumed to be volume quadrature, and a harmonic fit alone does not verify
a source-free homogeneous domain, Maxwell consistency or errors between samples.
The headless `coilgen.application.field_analysis` and `field_sampling` modules
expose the same operations to scripts without importing Qt or changing the design
pipeline.

### Research workflows

**Research studies** hosts the requirements and reviews for unusual designs:

| Design task | Authoring and evidence |
| --- | --- |
| Opposite gradients in separate ROIs, protected sensors | **Named targets, validation and poses** owns named regions, registered vector maps or analytic targets, independent desired datasets, component selection, separate RMS/peak budgets and fitting weights. Review preserves every region and shows failure locations, suppression and descriptive local gradient tensors. |
| Beam windows, obstacles, terminal locations | **Spatial scene and route anchors** owns closed imported meshes or boxes with explicit world registration and physical roles. Deliberate 3D picking binds terminals, cut intervals and waypoints to exact winding hashes. The circuit editor can initialize editable feeds from these anchors. |
| Motion and finite geometric manufacturing tolerances | Define separate design and held-out rigid poses, plus finite support dimension scales about an explicit center. Operators retain common nodal topology. Final wire reviews transform the same recorded paths at unchanged signed currents. **Queue / Review nominal and robust comparison** uses common evaluation datasets and poses. |
| Multichannel and cohort shim research | **Array currents and cohort design** fits signed channel currents jointly with individual bounds and a simultaneous resistive power budget, on reverified saved winding geometry. It reports residual fields, limiting samples, rank, conditioning and saturation. POD, adaptive sectorization and contour-family studies keep training and held-out cases separate; generated projects require final winding/feed review. |
| Unusual waveforms, shielding beside conducting structures | Author circuits and import time/current CSV in **Gradient engineering**. Import geometry-bound inductance/efficiency references in **Circuit references and coupled physics**. Review current, voltage and shared RC temperature, including finite pulse repetitions/rest. External complex frequency response, mechanical and spatial thermal records retain their own solver, excitation, material, environment, geometry and sampling contracts. |

These inputs are optional revisioned project study controls (`requirements`,
`scene`, `array_study`, `dynamic_analysis`). Owned target/map/mesh/table data
travels with project JSON and worker snapshots; the application modules expose
the same operations to ordinary Python callers without importing Qt. No-option
requests retain their existing numerical defaults and serialization.

Named regional field budgets replace the global target-error budget. Existing
custom research matrices and active gradient/linearity/shield contracts retain
explicit unknown blockers when their final wire counterparts are unassessed.
Independent validation uses the supplied desired datasets or held-out poses;
it never substitutes a legacy analytic grid or refits winding currents.
Incomplete electrical returns yield descriptive partial-path evidence.

Spatial review checks continuous segments, including modeled lifted returns,
supplies and drive closure. Its circumscribed conductor envelope is conservative:
overlap may require closer review, while a pass applies to that envelope and the
specified faceted exclusions. Available swept facets are checked as well,
including miter extensions and containment of disconnected solids. NPZ
`vertices_m` already uses metres. Scene
constraints gate acceptance of recorded geometry; they do not silently reroute
the sheet solution or certify an unswept feed. Finite pose/scale scenarios do not
prove continuous uncertainty coverage or material/cross-section tolerances.
Scene clearance applies to the recorded world geometry; posed field validation
does not qualify obstacle clearance in those additional poses.

Coupled-physics imports validate their numerical and source bindings, not the
accuracy of an external solver. The workbench does not add an eddy-current,
structural or spatial heat solver. Missing physics stays unassessed; static
magnetostatic fields, net force/torque and lumped temperature do not establish
those additional qualifications. Research review actions are disabled when
their selected run has no verified evidence.

Startup opens an empty welcome screen with **New project**, **Open project** and
**Load template**. **File → New project…** (Ctrl+N) creates a named design with no
support or target selected; it can be saved immediately as a self-contained JSON.
Templates are explicit starting choices under **File → Load template**. Passing
a project path at startup opens that saved design. **File → Close project**
returns to the welcome screen after checking pending changes.
Save includes pending design notes from every channel. Discard clears the
closed or replaced project's drafts; closing an individual popup or switching
channels retains them.

An open design shows the Model Builder and plot. The left Model Builder is the
main workflow navigation; there is no second workflow button bar. Click a model or study
item to open its settings; keyboard users can select it and press Enter.
Settings open in a floating inspector. Placement, mesh, harmonic basis/origin,
sheet material and numerical details open in smaller modeless windows from that
inspector. **Apply & preview** and **Revert** stay outside its scrolling area.
Closing an editor keeps its draft; applying refreshes the same logical editor.
Native drafts mark the document as modified immediately. Applying, undoing and
redoing acknowledge the applied revision; reverting or restoring the original
value clears the pending-edit message. Rejected edits retain their values and
show their error in the owning editor.

Inspectors present **numbered decision cards**, with the current choice and its
scientific icon. Open any card directly; numbers indicate a useful order without
locking navigation or claiming that a design passed a check. **Back** retains
pending edits and **Apply & next** validates and applies them before opening the
next decision. The final decision offers the next section, opening its first
choice directly; reaching Study never starts a computation automatically.
Each window also provides **Revert draft** and **Close**. An invalid
entry reopens its owning window, even if changing a mode hid or disabled that row.
Opening and closing decisions preserves exact saved SI values and makes no
model revision.

**Choose support** shows the complete primitive gallery. **Choose field** opens
linear gradients, analytic expressions, spherical harmonics and an existing
imported map as graphical tiles. Field regions and volume/boundary sampling
have their own graphical choices. Amplitude, dimensions, placement and mesh
remain separate windows. Inactive analytic controls are hidden for imported maps;
normalization and solve modes expose their applicable rows while preserving other
saved values. The insulation window contains the conductor-envelope preview.
Custom profile editing, explicit mounting/return-path displacement, engineering
current/spacing contouring and the full native Advanced editor remain available.

Under **Study**, **Choose output** opens graphical choices for preparation,
current-sheet optimization, contours, routing/sweeping or full evaluation.
**Requested operations** gives the detailed scope of the actual winding and
contouring policies. Changing an output selects the study boundary and is
undoable; computation still requires the explicit **Compute** action.

**View → Open evidence window** opens **Jobs**, **Results** and **History** in
separate reusable windows without shrinking the plot. **Detailed evidence** opens
checks, wire fields, comparisons and diagnostics. **Design status** and **Brief**
open pipeline evidence and physical assumptions on demand. Closing windows
preserves saved evidence and running jobs. Open evidence windows and inspector
visibility are saved in the single project JSON, including per-channel views.
One primary **Compute** button remains visible in the study toolbar. Select its
boundary in the Model Builder or **Study → Choose output**. Channel selection
appears for multiple channels, or to re-enable a disabled single channel.

The **Editing** row identifies the applied project/channel/revision and pending
changes. **Viewing** identifies the actual preview or immutable run, its recorded
date, sample scope and wire-current policy. Missing dates and data remain explicit.
Matching model/study parameters does not establish reusable checkpoints: reuse
also checks source, dependencies and saved artifact integrity.

Applying a target field or region selects **Support + target preview**, showing
the current support and colored target samples together. Component and units
identify the displayed field; this is the requested target rather than a solved
coil field. Unrelated edits preserve the chosen plot and camera.

The right-hand plan tracks **Prepare → Sheet → Contours → Winding → Evaluate**
for each channel. Black text means a recorded computation boundary; **Cached**
identifies verified reusable evidence for the applied model. Blue marks current
work, with the operation and its reported percentage shown separately. Completed
stages appear in the plot automatically,
including the current sheet, contours and connected/swept conductors when
requested. **Viewing** labels these as live previews of the submitted revision.
They do not establish final result integrity or engineering acceptance.
Automatic completion display follows only the exact current submitted revision.
Finishing an older revision leaves the selected view intact and makes its result
available through Jobs and Results. Stopped live snapshots retain their stopped status.
Selecting another plot pauses **Follow run**; enable it to resume. Jobs stays an
on-demand window, so submitting a computation keeps the graphics visible.
The lower-left status bar displays the current operation's measured percentage,
or **Computing…** with a busy bar when that operation has no known work total.
Mesh nodes, field chunks, contours, routing work, validation samples and saved
evidence advance it from actual completed work. Updates reach the GUI about five
times per second; worker messages are coalesced rather than writing each
numerical update to disk. Allocated overall work remains explicitly labelled in
**Tasks…** and the tooltip. Omitted or reused stages are excluded. These
percentages estimate work, not elapsed time or an ETA; an opaque factorization
or file compression does not invent solver convergence.
The plan shows the named operation, its current substep, elapsed worker time,
the age of the latest process-liveness observation and the age of the last
reported progress. Liveness observations never advance numerical progress.
Conductor preparation, bend resolution and provenance transfer report counted
work and check cancellation within their longer loops. Cancellation stays busy
and visibly says **Stopping** until the worker has stopped.

A blocked job-state write shows **Job state not saved · retrying**, including
the filesystem error and the limit on restart recovery. The monitor continues
draining events, supervising workers and verifying completed evidence. It retries
the latest state even after the worker exits; a successful save clears the
warning. Windows atomic replacements retry brief access/sharing errors within a
bounded interval, retaining the previous file if the denial persists. Explicit
submit and cancel calls still report failed durable writes to their caller.
The job reaches 100% only after saved evidence verifies. Automatic completed-run
verification, history restoration and reuse checks run in the background so the
workspace remains responsive while results become available. Integrity and
engineering acceptance checks remain mandatory and separate from completion.
Field analysis, harmonic fits, array studies and engineering tasks also feed
their measured progress into this indicator. It remains visible
while viewing an older run, switching channels or closing the project. Multiple
workers show a count and individual progress in the activity strip; **Tasks…**
lists every worker's revision, operation and measured work counters. Queued jobs,
cancellation and other background work have distinct labels; terminal jobs clear
the indicator. Status messages remain visible alongside it.
Queued jobs wait without an animated computation bar. Stopped jobs with no
reported progress show **Unreported**. If the runtime becomes unavailable,
activity freezes and is identified as last known until it reconnects.
Observed job outcomes remain in a dismissible notice below the workspace,
including when the project is closed. Each notice identifies its design and
revision, distinguishes execution from scoped engineering checks, and opens the
exact job or saved evidence on request. Finishing a background job preserves the
chosen plot and open editors. Multiple unresolved outcomes remain selectable;
previously completed jobs do not produce new startup notices.
Live snapshots remain separate from completed results and are not embedded as
results in project JSON or offered for CAD export.

Compute is named for the selected boundary; **Study → Requested operations**
lists its outputs.
Readiness checks use that boundary. Advanced settings also opens modelessly;
Compute stops if it has an unapplied draft. Valid expert drafts are included by
Save and script export; incomplete or conflicting drafts remain available for
correction. An expert draft cannot overwrite inputs changed in another editor.
Closing Advanced settings retains its draft and keeps it visible in Editing,
the modified-project indicator and the next action. Save, Python export and CAD
project companions include validated drafts and pending History notes for every channel.
Nonblocking preflight warnings remain accessible through **Modeling notes** beside
the compact design status. Its next-action button opens the relevant settings,
job or engineering evidence directly. Notes do not prevent computation.

The plot selector identifies the exact displayed dataset; **Inspect…** offers
target, field, wire and connection views in a single menu. The plot's **Layers**
and **Display** popups contain view options and sample-slab
controls. Field component, units, current selection and the displayed sampling
scope remain visible where applicable. These controls change the view only.
Drag in the 3D view to rotate the camera. A left click selects an object and
opens its settings only after release; camera drags and modified navigation
gestures do not open settings windows.
After camera navigation, its orientation reads **Custom**; choosing a preset,
including the same preset again, restores that world-coordinate view. A probe
identifies the selected layer and dataset. Field probes report the nearest stored
sample, distance and units without interpolation; contextual support, ROI and
witness layers do not borrow field values from another dataset. When the support
plot is hidden by its layer control, an explanation offers **Show support**.
Missing, empty or malformed datasets and unavailable current variants are grey
and disabled, with their reason and required step. Inspection uses the selected
current policy and does not silently switch to a different stored field. Comparison
offers common available datasets; each source retains its own evidence when none match.

**File → Load template** groups
editable starting designs into **Gradients**, **Shim coils**, **Coil systems**
and **Demonstrations**. Choose cylindrical or biplanar Gx/Gy/Gz, an actively
shielded Gy, uniform Z0, the five second-order modes, or conical Z2. Loading a
template protects unsaved edits and sets up a full study without starting a job.
All template settings are already applied. Loading selects full design evaluation
and leaves the settings windows closed; click **Compute full design** immediately.
Open a section in the Model Builder only when you want to inspect or change it.
The shielded template jointly optimizes primary and outer current supports
against the ROI target and a zero-vector-field constraint on a separate closed
external cylinder. Both support meshes and the suppression surface are owned
inputs embedded in the project JSON. Its sampled external-field budget and
independent ROI checks are separate evidence; inspect discrete-wire results
before exporting a shielded design.

**File → Load template → Coil systems → Cylindrical XYZ gradient system (Gx, Gy, Gz)**
creates the three independent gradient channels on coaxial 200 mm long supports
of 80, 86 and 92 mm radius. All use the same world-origin 40 mm spherical DSV,
B₀ along world z, and 20 mT/m Bz gradients. The complete sheet-to-conductor
studies are already configured. The supports have nominal 6 mm radial spacing;
routes, feeds, coupling, stray field and combined drive qualification require
system review. See the [cylindrical XYZ example](../examples/cylindrical_xyz_gradient_README.md)
for conductor settings, field definitions and script usage.

**File → Load template → Coil systems → Biplanar gradients + second-order shims** creates nine
independently driven channels: Gx/Gy/Gz, uniform Z0 and Z2/XZ/YZ/X²−Y²/XY.
Use **Channels → Channel overview** (or the channel icon beside Enabled) to
compare execution, parameter compatibility and scoped field/geometry evidence.
Use **Configuration…** for physical inputs, open a selected channel, or compute
the enabled channels independently. Each
channel retains its own inputs, study and completed results in the same JSON.
The example uses a shared 40 mm DSV, 20 mT/m gradients, 100 µT Z0 and quadratic
modes peaking at 10 µT on the reference sphere. Full evaluation requests
contours, routing, sweeping and strict review. See the
[example definitions and script](../examples/biplanar_gradient_shim_README.md) for
layer allocation, exact harmonic signs and recorded numerical checks.

The model tree separates editable **Model** inputs, **Study** controls and
immutable **Results**, following the
[Model Builder workflow](https://www.comsol.com/comsol-multiphysics/model-builder).
CoilGen keeps its surface-coil optimization and realization operations explicit:

1. Create a blank project, load a template, or open an owned project or
   independent-channel collection.
   Under **Model**, define support geometry/mesh, target field/ROI, conductor
   material and winding options. Physical values display their units; native
   configuration retains its declared units (including degree-valued options
   and T/Hz target fields). Inspect the geometry/target preview before solving.
2. Under **Study**, select the sheet objective, field budget and physical bounds,
   then configure independent evaluation and fabrication review. Choose a study
   boundary from preparation through full design evaluation. A saved project
   retains its own boundary and validation settings. Full evaluation executes
   the requested winding operations; disabled connection or sweeping options
   remain disabled and are described on the study page.
   The left Model Builder provides direct navigation to model, study, results
   and export settings without claiming completion or acceptance.
   Previous/Next section controls remain below the settings scroll area. Selecting
   the Study tree parent opens the currently requested computation boundary.
3. Review the pending-edit banner and apply a complete settings edit, or revert
   the draft. Invalid edits retain their values and show errors beside the
   affected controls. The current revision and undo history remain intact.
   Undo/redo restores full immutable revisions. Advanced settings opens searchable,
   typed expert controls and a synchronized native JSON editor.
4. Choose a computation boundary: prepare, sheet, contours, route/realize, or
   evaluate. Use the blue Compute button at the upper right, or Ctrl+Enter.
   Compute submits a detached input snapshot to a bounded worker.
   The Jobs panel's execution strip and table report activity, execution state,
   reported progress, elapsed time and cancellation. Completion does not move you
   away from pending edits; View results verifies the saved evidence and opens Results.
   Repeated clicks for the same active revision select its existing job.
   Missing or unreadable inputs, invalid settings and unavailable required
   solvers produce an actionable, nonmodal message before submission. Open
   settings takes you to the affected model section. Valid defaults and unused
   optional inputs do not create new requirements; numerical feasibility and
   manufacturing acceptance remain results of computation.
   The Jobs table identifies **This session**, **Earlier session** and **Other
   session** beside each design, with its absolute local submission date/time.
   This session means a job submitted from this workbench window, including a
   newly resumed study. Existing jobs retain their earlier-session label even
   when still running or finishing today. Unknown execution timestamps remain
   **Not recorded**; importing a result does not invent a run date. Hover for the
   exact UTC time and job ID. Use **Show → This session only** to focus on work
   launched here; filters preserve older evidence. Active jobs appear first,
   followed by this session's results and historical jobs ordered by submission.
5. Inspect the saved run and its physical scope. Stream functions, field maps
   and winding paths retain the geometry/current of that run. Editing a model
   marks incompatible evidence stale; previous runs remain inspectable.
   Full results initially show the exact saved pipeline sweep triangles when
   available. Strict reviewed results initially show their own recorded conductor
   path. The grouped dataset selector provides sheet, winding and field views.
   A labeled Plot header spans the viewport; field components, units and stored
   wire-current choices appear on their own row when relevant to the selected plot.
   Pipeline sweep geometry is separate from conductor centerlines and
   fabrication-review geometry; displaying a sweep does not certify its
   manufacturing checks.
6. Inspect the computation-stage tree. Compute automatically reuses the latest
   compatible saved boundary at or before the requested stage. Changing contour
   levels can reuse the optimized sheet, while upstream material changes rebuild
   it. Resume selected run restricts reuse to that run's compatible boundaries.
   Resume is enabled only when a verified compatible checkpoint exists.
   Completing a requested partial study offers **Continue** to the next boundary,
   without claiming that downstream field or manufacturing checks passed.
7. Review independent field and physical constraints before exporting. Engineering
   acceptance is separate from job completion and construction release.
   The banner and comparison table separate sheet checks, pipeline sweep checks
   and strict fabrication review. A sheet pass cannot hide a failed sweep; missing
   check evidence remains unverified. Wire field rows state their sampling scope
   and original/refitted current policy.
   **Inspect saved settings** opens the original verified request read-only beside
   the applied model snapshot. Editing current channel settings and running review
   explicitly target the active revision. Integrity failures remain visible across
   aliases of the same run and block inspection, export and reuse until reverified.
8. Open **Export** under Results. Select the actual saved conductor solid bodies
   and formats, choose explicit STL coordinate units (default mm), and create a
   new CAD export folder. STEP retains its declared source units. World bounds
   and physical extents display in mm. The package includes hashes, run revision,
   units and separate check scopes in `geometry_manifest.json`; the default
   companion `project.coilgen.json` contains model inputs and completed results.
   Export verifies the saved files and never regenerates a sweep or fits a new
   current. Failed checks label available geometry as inspection output; runs
   without saved solids explain the missing evidence and offer review or problems.
   Body/format selection, STL units and companion choice remain associated with
   the exact saved source across navigation and export retries. The export receipt
   shows the destination, committed formats/units and file count, with **Open
   export folder**. Changing the selected run does not relabel an ongoing export.

The persistent next-action strip guides the current revision from target to CAD.
Expand **Pipeline progress and evidence** to inspect Target, Support, Conductor, Solve, Winding,
Check and Export, then click a section to edit or inspect it. Input definition,
computation and scoped check acceptance are distinct states. Pending edits,
earlier jobs and stale saved results each retain their identity. Model changes
show their dependency plan; Compute verifies any compatible stage reuse.

Colors support the status text: blue means selection/computation, teal means a
check passed within its stated scope, amber means pending/stale/incomplete,
red means missing input or failed checks, and neutral means unverified/not run.
Foregrounds adapt to light and dark palettes. A green check or exported solid
does not establish a driven circuit or construction release.

Linear gradient targets expose direction (`Gx`, `Gy`, `Gz`), amplitude in mT/m,
ROI and sampling first. **Expert target controls** reveals the native expression
and scaling policy. Changing direction updates both the expression and its
normalization axis; opening the editor does not alter existing scientific values.
Analytic expressions, harmonics and imported maps keep their explicit definitions.
An undefined rotation axis with a nonzero angle produces an actionable GUI error
before submission; valid scientific inputs and the legacy script geometry API
retain their existing behavior.

### Support surfaces and field regions

In **Geometry**, click a surface tile to select Cylinder, Biplanar or Conical
frustum. **More surfaces…** opens the complete catalog: Rectangle, Disc, Annulus,
Partial cylinder, Elliptical cylinder, Superellipse, Circular biplanar, Spherical
cap, Ellipsoidal cap and Toroidal segment. Selecting a catalog shape displays its
tile and applicable dimensions; imported/custom supports keep their own tile.
All 13 choices use the existing generated-mesh methods and parameter fields.

Lengths display in mm and angles in degrees. Radii and ellipsoidal semi-axes
are labeled explicitly; a biplanar separation is the full gap. Rectangle height
follows local x and width follows local y, matching the native builder. Cap
opening is measured from local +z, with 90° giving a hemisphere. Superellipse
exponent 2 gives an ellipse and 4 a rounded rectangle. The editor explains the
local frame, translation, normal/rotation and open boundaries for each shape.
The legacy Cylinder editor displays its actual mesh length and preserves that
length when axial divisions change. Other cylinders use their full native axial
length directly.

Geometry checks reject unusable GUI meshes before Compute: zero normals,
zero-radius annular openings, collapsed 180° cap rims, coincident 360° open
segment end seams and undersampled discs. Use Disc for a surface without a
central opening. A toroidal tube can span 360° because its tube seam is welded;
its major segment remains open. Circular biplanar grid-step target determines
diameter divisions approximately and is not a guaranteed triangular edge length.
The GUI bounds that derived grid to 512 divisions. These editor checks preserve
the existing script-builder definitions.

In **Target**, choose the field definition and its **Field region** separately.
Sphere, Ellipsoid, Box and Cylinder have distinct dimension controls. Ellipsoid
controls show full diameters in mm; the stored definition uses semi-axes in m.
Box sizes and cylindrical axial length are full dimensions, with the cylinder
axis along local z. **Position and orientation** exposes the world center and
right-handed axis-angle rotation. Moving or rotating the region changes sampling
coordinates; the field expression stays in world x/y/z and the harmonic center
retains its own definition.

Choose **Volume samples** to constrain the region interior or **Surface samples**
to constrain its boundary. Surface-only checks do not establish interior field
accuracy. **Import region surface…** owns the imported SI world mesh in the
project. Volume sampling requires one watertight, consistently oriented,
connected enclosure and retains points inside its actual boundary, including
concave shapes. Surface sampling accepts an open valid triangular mesh. Imported
mesh coordinates remain authoritative; region center/rotation do not implicitly
transform them. Topological closure alone does not certify a shell free of
self-intersections or establish manufacturability.

Explicit region resolution is bins per local bounding-box axis for volume
sampling; shape filtering determines the retained point count. Surface sampling
uses proportional angular/face density; imported surfaces have a global
`6 × resolution²` area-distributed point budget. All retained points
carry equal objective weights, rather than integration weights. Independent
evaluation uses the same region definition with a distinct sampling phase.
The requested validation resolution remains authoritative and is not silently
increased. Preview reports the actual retained sample count.

An imported field map keeps its saved coordinates and values as the field
samples, so generated-region controls are disabled. Existing projects with no
explicit `target_region_geometry` retain their original sphere or legacy target
mesh sampling when opened. Choose a new region, sampling mode or explicit pose
to request the new definition. The script contract and native support parameter
fields are listed in [Parameters](PARAMETERS.md#native-generated-support-surfaces).

For explicit analytic regions, the optional spherical DSV/ROI gradient slice
diagnostic is unavailable. Optimization and independent evaluation use the
actual region correctly. Choose native slice-map domain `all`/`full` for a
separate saved-target-sample bounding-box diagnostic; its scope does not establish
coverage of the shaped region. Imported-map and legacy region diagnostics retain
their existing behavior.

The **Physical design brief** stays above the viewport. Expand it to inspect the
target, ROI, world frame, mechanical support, conductor, circuit intention,
active field and hardware budgets, objective and remaining engineering evidence.
Requirements, optimization preferences and model assumptions have explicit
labels. Click a brief row to open its existing editor. Pending editor values
are distinguished from the applied model; the brief never rewrites saved results.

Under **Winding**, choose a connected circuit, separate circuits per support,
independent closed turns, or contours only. Conductor solid generation is a
separate choice. These controls map to the existing script configuration; an
unrelated edit preserves inactive native preferences and all numerical values.
Feed and drive-terminal completeness still require their own evidence.

Open **Winding and routing → Contour smoothing** to optionally smooth closed
contours on their current-carrying surface before cutting and interconnection.
**Smooth on surface** exposes mesh-coupled strength, a displacement cap in mm,
sampling interval and centerline-spacing preservation. Smaller mesh gain means
stronger attenuation of mesh-scale oscillations. The option is off by default;
opening the editor does not change the model. Applying a change invalidates
contours and downstream geometry, while compatible current-sheet work remains
reusable. The field is evaluated again at the retained contour current.
**Resample only** adds points without rounding corners; connected routing
defers that operation. Advanced settings contain fit/output sampling density
and local surface-patch controls.

**Results → Design acceptance** presents sheet, contours, connected winding and
physical conductor evidence together. Saved requested/effective limits determine
the displayed margins. A descriptive wire-field value does not inherit the sheet
budget. Each row retains its current, sampling and geometry scope. Select a failed
check to open the relevant settings or locate an actual recorded spatial witness.
Missing evidence and undesigned feeds remain explicit.

In **Project history**, pin an owned verified baseline, record a design hypothesis,
and use **Create variant from run** to restore its original settings and owned
inputs as an undoable revision of the same project. Alternative candidates remain
separate from independent assembly channels. Compare the candidate to its baseline;
physical parameter differences retain exact native values in tooltips, and numerical
deltas appear only for compatible evidence. Comparisons and note drafts survive
history refreshes. Baseline evidence, design notes and decision rationale are embedded
in the single project JSON, with stable identities that survive relocation.
Verification-heavy GUI actions run in a background task and reject a returned
document if the model or editor draft changed during verification.

The viewport's **Inspect** shortcuts select existing evidence for Target and ROI,
Field accuracy, Wire fit, or Connections. Their messages state the actual displayed
scope and explain when solids or winding maps are unavailable. They do not compute
new fields or change the stored current policy. The expert Plot selector remains
available.

File → Load template provides small MRI insert models with explicit SI inputs:

| Example | Support | Target in spherical ROI | Winding request |
| --- | --- | --- | --- |
| Cylindrical Gx / Gy / Gz | 200 mm long, 80 mm radius | `ΔBz = (20 mT/m)x`, `y` or `z`; 40 mm ROI diameter | 16 contour levels, connected 2 × 2 mm winding and conductor solids |
| Biplanar Gx / Gy / Gz | Two 200 × 200 mm planes normal to world x, 100 mm apart | `ΔBz = (20 mT/m)x`, `y` or `z`; 40 mm ROI diameter | 16 contour levels, 2 × 2 mm circuits on each support and conductor solids |
| Actively shielded Gy | Two 300 mm long cylindrical supports, radii 80 / 110 mm | Gy = 20 mT/m, 40 mm DSV; external zero vector field, 20 µT area-weighted RMS budget on a closed 150 mm radius / 400 mm long cylinder | Joint sheet optimization, separate 2 × 2 mm support circuits and conductor solids |
| Cylindrical Z0 / Z2 / XZ / YZ / X²−Y² / XY | 200 mm long cylinder, 80 mm radius | Z0 = 100 µT; each quadratic mode peaks at 10 µT on the 20 mm reference sphere | 12 contour levels, connected 1 × 1 mm winding and conductor solids |
| Independent-turn sweep demo | 200 mm long cylinder, 80 mm radius | Uniform `Bz = 100 µT`; 20 mm ROI radius | Four separate closed turns and reviewed solids |
| Conical Z2 | 300 mm long frustum, 180/140 mm bottom/top radii | +10 µT poles / −5 µT equator; 50 mm ROI radius | 12 contour levels and a connected winding; challenging routing study |

These models use 18 nΩ m copper resistivity, a 1 mm homogenized sheet thickness,
0.1 mm insulation and 0.05 mm manufacturing tolerance. The Gy examples use
2 × 2 mm conducting copper (4 mm²), a nominal 2.2 × 2.2 mm insulated envelope,
and a 6 mm minimum centerline bend radius. The shim examples retain 1 × 1 mm
conductors and 3 mm minimum bend radius. The main field is explicitly 3 T along
world z. The sheet and physical-wire cross sections belong to separate models:
the Gy conductor upgrade does not change the effective sheet thickness or the
field optimization budget. With unchanged path length and current, four times
the copper area gives one-quarter the resistance and instantaneous Joule loss;
routing, mounting and bending can change the actual path and its field.

For a first Gy study:

1. Choose **File → Load template → Gradients → Cylindrical Y gradient (Gy)**. The support is
   centered on the origin with its axis along z. The spherical ROI has radius
   20 mm, so its diameter (DSV) is 40 mm.
2. Click **Compute full design**. The template already includes its target,
   conductor, optimization, winding and validation settings. Use **Plot** to
   compare the target, continuous-sheet solution, contours, connected winding
   and wire field. For wire maps, choose **Original contour current**, then
   inspect **Checks** and **Problems**.

Optional template inspection:

- Open **Target** to review the prescribed field. Gy means `dBz/dy`,
   the y dependence of the z field component.
   The expression `y`, linear-fit normalization and 20 mT/m give
   `ΔBz = 0.02 × y T`, with y in metres. At y = −20 / 0 / +20 mm, the prescribed
   perturbation is −0.4 / 0 / +0.4 mT; it is added to the separate main field.
   The interpretation panel updates when amplitude or ROI radius changes.
- Open **Material** and **Winding** to review 2 × 2 mm conducting copper, 0.1 mm
   insulation on each side, 6 mm minimum bend radius and 16 contour levels.
   A contour level can contain multiple loops; this is not a claim of 16 turns.

Define a custom conductor in **Conductor → Cross-section profile…**. Choose
**Polygon from straight segments**, edit the ordered **u [mm] / n [mm]** point
coordinates, or drag the numbered points in the preview. The boundary closes
automatically from the last point to the first. **Split segment** inserts a
midpoint after the selected point; move it to make a convex corner. **Remove**
deletes a point while keeping at least three. **Fit view** includes the origin
and uses the same physical scale on both axes.

The current sweep requires a strictly convex, nonintersecting polygon with
finite coordinates and positive area. Invalid drafts remain editable and block
Apply, Compute and Save; closing the window keeps pending edits. **Apply &
preview** creates one undoable model change; **Revert draft** restores the
applied profile. The project JSON stores the points in metres, including their
original order and offset. `u` is transverse to the wire in the support surface;
`n` follows its normal. Editing never recenters the profile. Mounting settings
can subsequently offset its path reference. Copper area comes from the polygon;
insulation remains a separate setting. **Rectangle from width / height** keeps
the existing script-compatible rectangle representation.
In polygon mode, **Nominal width / height** retain their existing values and
legacy report roles. Nominal width also controls PCB tracks and some routing
thresholds; choose it consistently with the polygon extent. The editor never
silently substitutes the bounding box into these settings.

The unshielded cylindrical gradients fit a 3% sheet RMS budget with a budget
ratio of 0.6; the other studies use 4% and 0.8. The spherical ROI uses 257 design
samples and a denser, different 3112-point independent sheet grid against a 5%
budget. The shielded template adds its separate external suppression samples
and area-weighted vector-field constraint. Strict fabrication review is
requested. In the cylindrical reference run, the independent sheet RMS error is
3.11%, and the 2 × 2 mm connected-wire Bz RMS error is 4.60% on the design grid
at the original 18.63 A current. The resulting path is 12.44 m long, with
0.05599 Ω resistance, 19.44 W instantaneous Joule loss and 4.66 A/mm² conducting
current density. These are electrical and field results, not a manufacturing pass:
the recorded clearance, bend and swept-body checks fail. Connected Gy examples
exercise the full pipeline and keep those failures visible. The separate
turns provide an inspectable shim geometry, while feeds, terminal breaks,
interconnections and assembly mutual inductance remain undesigned. Inspect the
current saved checks for each model; none is a released hardware design. Cooling,
voltage and duty-cycle limits require an explicit subsequent engineering study.

Imported STL geometry requires an explicit length scale. Owned SI NPZ meshes
retain float64 coordinates without STL quantization. Imported target maps retain
their units, weights, suppression labels and supported components. The generated
expression defines Bz; projection/evaluation components do not turn a Bz expression
into a Bx/By target. Use a vector map for such targets.

The cylinder **Length** control describes the actual mesh span. The GUI compensates
for the legacy cylinder generator's omitted final axial row, and keeps the
physical length fixed when axial divisions change. Original script and raw JSON
height parameters retain their previous semantics. Biplanar orientation uses
**Plane normal x/y/z** and **World center x/y/z**, rather than axis-angle rotation.
Negative-z normal directions use the same full-sphere rotation convention as
the circular biplanar generator.

Analytic target controls show mT/m for gradient normalization of tesla-valued
expressions and Hz/m for frequency-valued expressions. Amplitude scaling uses
a dimensionless multiplier. Absolute-field mode disables unused gradient
controls. Incomplete numeric notation blocks Apply, Save, navigation and Compute;
accepted text is committed before the study snapshot is submitted.

Generated support controls include a conical frustum with independent bottom and
top radii, axial length, mesh divisions, rotation and world center. Both radii
must be positive; the support is an open lateral surface. A tapered support uses
the finite-element surface for manufacturing; mesh refinement is needed before
claiming agreement with a smooth cone.

The analytic target editor also supports a real solid spherical harmonic. Its
coefficient is displayed in µT and multiplies `(r/R)^l * real_Ylm` using CoilGen's
existing normalized real basis and phase convention. The explicit reference
radius and world center persist independently of the sampling grid. Coefficients
are not pole amplitudes. Harmonic targets use absolute tesla and disable gradient
amplitude normalization. Arbitrary expressions and imported maps remain available.

Geometry and target preflight reports mesh counts and world bounds, actual ROI
samples and active weights, Bz range/RMS in tesla-derived display units, and the
resolved normalization. It uses the numerical target definition. Y20 additionally
shows the derived axial pole and equatorial amplitudes. Preflight is a model
inspection step; it does not establish field or manufacturing acceptance.

The Problems tab explains recorded failures and links to relevant settings.
Locate evidence is enabled only when saved diagnostics provide a supported spatial
witness, such as a recorded conductor interval or sweep section. Unsupported
locations stay unavailable. Locating a witness zooms to its saved bounds and fades
the surrounding conductor. Clear highlight restores the prior dataset and camera.
Selecting a problem does not infer a new collision or modify the numerical result.

Project history retains named runs, selected evidence and viewport controls across
reopening. Saving embeds the verified completed evidence, aliases and viewing
state in the single project JSON. An auxiliary local history cache supports the
active session and can be rebuilt from that JSON; it is not required to relocate
the project. Scientific outputs remain immutable. Missing or corrupt evidence
remains visible with an explanation. Histories remain separate for independent
channels, including after converting a standalone project to a collection.

Select two history rows to compare them. The comparison reports configuration and
engineering metric changes alongside linked 3D views, shared cameras and scalar
limits. Each field map retains its own saved sampling grid and current policy;
the viewer does not interpolate mismatched grids or fit currents to make results
comparable. Metric rows identify incompatible physical scopes.

The **Optimize current sheet** settings show the effective cost equation and
solver configuration above the decision cards. The display follows native
Tikhonov, field-budget regularization searches, native constrained objectives
and physical power, magnetic-energy, minimax or mixture objectives. It identifies
imported-pattern bypasses and parameters that the selected formulation ignores.
Pending edits update this display without changing the applied model.

The **Field and current limits** card exposes the relative field budget, absolute
budget [T] and metric-specific RMS/STD limits [T]. A positive RMS/STD limit replaces
the relative and absolute budgets for that metric; those inactive controls are
disabled. The card reports the active limit in nT. Disabling default physical
field constraints also disables these default budget controls, while explicitly
configured physical constraints retain their own scope. Native Tikhonov displays
these budgets as post-fit checks rather than hard optimization constraints.

Use **Solve strategy** to choose the formulation and objective, **Solver
configuration** for backends, tolerances, iteration limits, threads, verbosity
and fallback policy, and **Numerical settings** for regularization, penalties
and the stream-function basis. **All optimization parameters…** opens the
searchable advanced editor focused on optimization. It exposes every native
optimization setting and physical-objective parameter, including mixture weights
and scales, field constraints, hardware limits, search bounds and backend options.
The full configuration JSON remains available in the same editor. Optional and
structured values retain their native representation; numerical entries accept
scientific notation and preserve the precision of untouched loaded values.

Apply validates the complete draft and creates one undoable change. Switching
back to a native objective cannot silently discard configured physical bounds.
Viewing the settings does not change the configuration or numerical method.

The optimization allowance control edits the active solver's actual field
acceptance slack. Independent validation's allowance is shown separately: native
solver studies currently use the research checker's default independent allowance.
Selecting a physical solver makes both policies derive from its physical spec.

The **Design / validation budget ratio** preserves the existing `design_margin`
API. The independent validation limit is the configured design limit divided by
this ratio. A ratio of 1 uses the same field budget; 0.8 permits 1.25 times that
budget on the independent grid. This ratio does not tighten the optimization
tolerance. Set a tighter design field budget explicitly when reserving margin
for discretization or realization. Saved validation metadata records the actual
limit and acceptance allowance. Independent sheet sampling and design-grid wire
fields retain their separate physical scopes.

The independent-channel container retains separate channel targets, settings,
study controls and run identities in one project. It executes channels through
the same single-channel service. It does not infer electrical connections or
certify combined fields, mutual inductance, assembly clearance or joint limits.
Each independent channel owns a distinct project ID. Script callers can use
`project.duplicate(name)` to share immutable inputs while creating independent
revision and run ownership; ambiguous duplicate owners are rejected on load.

## Script and command-line use

Existing script APIs remain directly usable:

```python
from coilgen import CoilGenConfig, run_coilgen

config = CoilGenConfig(coil_mesh_file="create cylinder mesh")
result = run_coilgen(config)
```

To use owned, portable inputs and the application's validation evidence:

```python
from coilgen.application import DesignService, ProjectDocument
from coilgen.application.presets import preset_config

project = ProjectDocument.new("Gy", preset_config("cylinder"))
project = project.update_study({"pipeline_stage": "sheet"})
project.save("Gy.coilgen.json")
record = DesignService.run(project, "outputs/Gy")
```

A `CoilDesignProblem` can be captured with `ProjectDocument.from_problem(problem)`.
`ProjectAssembly.from_coil_project(project)` captures supported channel designs
from the existing script project API. Custom Python functions remain a script
extension path; arbitrary Python source is not parsed back into editable GUI data.

```powershell
python -m coilgen new --preset cylinder --name Gy --output Gy.coilgen.json
python -m coilgen inspect Gy.coilgen.json
python -m coilgen run Gy.coilgen.json --stage sheet --output outputs/Gy
python -m coilgen run Gy.coilgen.json --stage contours --resume outputs/Gy/pipeline_checkpoint.cgstage --output outputs/Gy_contours
python -m coilgen export-script Gy.coilgen.json replay_Gy.py
```

GUI script exports include an adjacent self-contained compact or JSON project
companion with its owned inputs. Completed evidence is optional for script
exports; full project Save retains that evidence.
The exported script executes the saved stage and validation controls through
`DesignService`, without a GUI. `--require-accepted` returns exit code 2 when
engineering acceptance is failed or unverified. Prepared, contour and routed
stages do not establish final winding acceptance.

Selected-run artifact export copies verified outputs on a background worker;
the GUI retains activity and a destination receipt, including inline retry on
failure. Copies preserve saved bytes and units, reverify source integrity and
never overwrite existing destination files. CAD and artifact exports verify all
recorded hashes without decompressing unrelated scientific arrays.
Project/Python export also
includes owned inputs and is the portable route for rerunning a design elsewhere.

## Ownership, persistence and reuse

`ProjectDocument` owns immutable input bytes, a resolved configuration and study
controls. Editing creates a new revision. Public configuration, arrays and run
metadata are detached copies. Display names do not substitute for persistent IDs.

Save project writes one self-contained JSON, including imported meshes/target
maps, configuration, study controls, completed run evidence, checkpoints and
viewing state. Independent-channel collections embed each channel in that same
file. Copy only the JSON to relocate the project. Saved run aliases, the selected
channel/run, workflow page and camera remain available after relocation.

Input assets and numerical result artifacts are stored as hash-verified base64
bytes. Original NPZ data types, floating-point precision, result/manifest bytes
and physical acceptance scopes are preserved. Saving atomically replaces the
JSON only after serialization succeeds. Reopening materializes verified worker
files into an application cache; that cache is rebuildable from the JSON. A large
study can produce a correspondingly large JSON because checkpoints and results
are included. Old v1 projects and collections still load; saving migrates them
to the self-contained v2 format.

Save As also offers **Compact CoilGen bundle (`.coilgen`)**. The single ZIP file
contains a readable JSON manifest and compressed, content-addressed input and
evidence bytes, deduplicated across channels. Original NPZ bytes, array types,
run owners, scientific fingerprints and artifact hashes stay unchanged. Opening
verifies every payload before interpreting evidence; it never extracts archive
paths or loads pickle. Save/open displays processed-byte progress. JSON remains
available and existing JSON files still load. In Python, use
`project.save("design.coilgen")` or `assembly.save("channels.coilgen")`.

Python export includes the owned model/study inputs and omits previous completed
results by default, since a fresh solve does not need those archives. Applied
design hypotheses and decisions remain in the companion; UI selection and view
state are omitted. Export does not read or capture prior result files. The export
dialog offers compact or JSON companions. To explicitly publish prior evidence,
use `project.export_python("design.py", compact=True, include_results=True)`;
the equivalent workbench export methods accept the same options. Full project
Save continues to retain completed evidence and viewing state.

New archives also retain actual job submission/completion timestamps, when the
saved job identifies the same request and project. These small provenance bytes
are covered by the archive content identity; original result and manifest bytes
remain unchanged. Reopening restores the dates separately from live worker IDs
or state. Older archives with no recorded timestamps continue to show an
unrecorded date; file modification and import times are never substituted.

Script callers can explicitly attach a completed run before saving:

```python
from coilgen.application import RunArchive

record = DesignService.load_run("outputs/Gy")
project = project.with_results((RunArchive.capture(record, alias="Baseline Gy"),))
project.save("Gy.coilgen.json")
restored = ProjectDocument.load("Gy.coilgen.json")
record = restored.saved_runs[0].materialize("local_run_cache")
```

Stage artifacts use an allowlisted JSON/NPZ format, not Python pickle. They store
typed numerical state, checksums and stage fingerprints. Reuse requires matching
upstream settings, asset content, engine source and numerical dependencies.
Each completed boundary is retained separately, so a later edit can reuse an
earlier stage without attempting to rewind a final checkpoint. The original final
`pipeline_checkpoint.cgstage` remains available to script callers. The interface
rechecks checkpoint identity and integrity before submission; workers validate
the full checkpoint again before using its numerical state.
GUI/plotting edits do not invalidate numerical caches. Unknown numerical controls
conservatively invalidate preparation. Wire dimensions, sheet material and linked
current/spacing constraints remain upstream whenever they affect optimization.

Workers own their submitted files and checkpoint. Native thread and worker counts
are bounded. Cancellation first requests a stop at a numerical boundary; a stuck
worker can be terminated. Interrupted work stays interrupted after restart;
partial files are never promoted to a completed result. Completed artifacts are
verified before inspection/reuse. New results also checksum their acceptance,
validation and provenance metadata; legacy results without this checksum remain
readable but only their listed artifacts receive the integrity check.
Stopped jobs can resume committed boundaries with a matching request, input and
dependency identity and per-checkpoint hashes. Jobs predating this recovery evidence
remain unavailable for recovery. An optional recovery-record write failure leaves
computation intact and reports which boundary cannot be resumed.

The application service reuses a completed output directory only for an identical,
verified numerical request. A different request or corrupt evidence requires a new
directory, so a parameter edit cannot replace historical results. The legacy script
runner retains its existing behavior.

For connected windings, the final conductor representation is shared by field
evaluation, electrical estimates and wire-path exports. Independent contour-body
studies require strict fabrication review for mounted-body fields and electrical
paths; ordinary contour fields remain source-loop fields. Use
`inductance_source="unconnected_contours"` for source-loop electrical estimates.
Reported source and circuit scope remain explicit; missing feeds or physical
validation are not silently inferred from closed paths.

The fabrication-review compiler defaults to realization on the finite-element
mesh; the ordinary pipeline can use analytic support realization. Their saved
centerlines are separate viewport datasets, and fabricated fields retain the
review's target coordinates. Independent sheet sampling verifies that sheet
model's field budget; it does not establish mesh convergence or independently
certify the finished winding. Legacy target-current fitting remains available;
fixed-current diagnostics and fabrication checks report their own scope.

Saved contour and evaluated-winding fields are available with signed residuals
and global weighted RMS errors. Select the reported evaluation current or the
original contour current; the viewer uses saved arrays and does not refit a
current. Signed component residuals use colour limits centred on zero. These maps
use the design grid and do not establish independent winding
validation. The evaluated geometry may be a mounted conductor centroid; a copied
contour field is not presented as an evaluated connected winding. The Wire fields
table states current, refitting policy, geometry and sampling scope.

## Conical harmonic workflow trial

Create a reproducible full-study project, open it, or compute it headlessly:

```powershell
python examples/conical_harmonic_workbench.py --output outputs/conical_z2
python -m coilgen.gui outputs/conical_z2/conical_z2.coilgen.json
python examples/conical_harmonic_workbench.py --output outputs/conical_z2_gui --gui
python examples/conical_harmonic_workbench.py --output outputs/conical_z2_script --compute
```

Each creation requires a fresh output directory. The example uses a 300 mm cone
with radii 180/140 mm, a 100 mm diameter spherical ROI, and
`Bz = 0.002*(2*z*z-x*x-y*y)` T. Its normalized Y20 coefficient is approximately
15.8533 µT at R=50 mm: +10 µT at the axial poles and −5 µT at the equator.
It requests a 4% sheet design budget, a 5% independent sheet budget, 12 contour
levels, mesh-geodesic connectors, 1×1 mm copper conductor and strict fabrication
review. It uses 257 fitting samples and a separate 3112-point evaluation grid;
the ROI and harmonic reference radius are both 50 mm, with independently saved
definitions.

This is a demanding shim-design study for inspecting the full connected-coil
workflow. The stored trial achieved the sheet budget but failed the fabricated
field budget and reported routing, clearance, support-penetration and sweep
failures. Inspect the current run's Problems and Wire fields tables at the
original current before considering export. A passing sheet alone does not
establish winding accuracy or construction release.

Use separate saved runs to investigate support-mesh convergence, contour
resolution, connector/return paths and manufacturing limits. Compare field maps
only where their recorded target grids and current policies match. The harmonic
editor and polynomial expression describe the same normalized Y20 field; the
explicit world center and reference radius remain independent of ROI sampling.

## Gradient engineering workflow

**Gradient engineering** provides focused editors and studies. The geometry,
target, winding and validation inspectors also link to the relevant tools.

- **Gradient requirements** edits the native strength, RMS/peak linearity and
  cross-gradient tolerances, normalization radius/region and offset convention.
  Zero disables a tolerance. These native gradient fits describe Cartesian Bz
  gradients in world coordinates; they do not qualify an arbitrary vector target.
- **Primary, shield and external region** creates owned concentric cylindrical
  meshes with explicit dimensions, sampling, center and axis. Known cylindrical
  templates reopen as editable dimensions. Imported geometry is replaced only by
  an explicit Apply. The external suppression cylinder is closed; its field bound
  must be supplied separately in optimization requirements.
- **Queue convergence study** varies native mesh refinement, source quadrature
  and independent target sampling in separate one-factor series. It retains the
  baseline inputs once and records candidate identities, jobs and user criteria
  in a convergence manifest. **Review convergence study** rechecks completed
  evidence and saves the changes and observed stable axes. Three distinct levels
  are required for a mesh/quadrature stability claim; changing target sampling
  does not establish either. Incompatible target/current scopes remain unverified.
- **Terminals, feeds and drive waveform** requires a verified fabrication review
  for the applied numerical model. Define named terminals, explicit current
  polarity, an indexed feed break for a closed turn, supply and return polylines,
  and any spatial drive closure in world-frame meters. No connector is inferred.
  Path hashes reject stale definitions. Piecewise-linear time/current waveforms
  provide I/V margins, exact waveform RMS current and average DC Joule loss.
  Voltage estimates include both endpoints of each ramp. Mutual terms require
  separate trusted geometry-matched evidence; multi-body self-only voltage
  estimates cannot qualify the amplifier. Optional measured lumped thermal
  parameters provide a scoped RC estimate, without a cooling or release claim.
- **Compare measured field** evaluates the saved winding at independent measured
  locations. Import numeric x,y,z,Bx,By,Bz rows with explicit units, or an SI CSV
  headed `x_m,y_m,z_m,Bx_T,By_T,Bz_T`. Declare the proper rigid transform, signed
  current for every displayed body and measurement/calibration source. There is
  no registration/current fitting or field interpolation. Optional RMS/peak vector
  error limits are separate from design acceptance. The exported archive owns the
  original data bytes, transformed values, predicted/residual fields and provenance.

Enable **Review fabricated winding on independent samples** in Gradient
requirements to request evaluated fabrication review. The independent winding
plots use recorded electrical paths and original signed contour currents, with
their own coordinates and targets. Repeated fitting grids, singular wire samples
and unsupported imported/custom/scenario/suppression contracts cannot pass as
independent evidence. Exact collinear-segment bisection checks exercise kernel
consistency; they do not check curve discretization or finite copper cross-section.

Results now present achieved/requested gradient, linearity, electrical path
length/current/R/L, gradient efficiency where a single complete body makes it
unambiguous, DC copper |I|/area and waveform margins. Compatible affine Bz targets
retain a descriptive gradient fit when tolerances are zero, with status not checked.
Surface K [A/m] has a saved face-sample plot. K, copper volume density, recorded
surface-geodesic contour spacing and physical envelope clearance retain distinct
scopes. Missing plots are disabled with their
required step. Feed closure does not certify unswept leads or a physical cut;
construction release, PNS, acoustic, eddy-current and structural qualification
remain separate engineering evidence.

The same operations are available without Qt through
`coilgen.application.gradient_geometry`, `coilgen.application.gradient_studies`,
`coilgen.application.gradient_review` and `coilgen.gradient_engineering`.
Optional `ResearchRequest.engineering` controls are saved in the project study,
advance its immutable input revision, and participate in request identity.
Omitting them preserves the existing numerical pipeline and defaults.

## Architecture and verification

The workbench uses repository-owned scientific vector icons for surfaces,
targets, conductor profiles, circuit topology, field inspection and engineering
checks. Icons follow the Qt palette and display scale; scientific controls retain
labels, values and units. Status badges distinguish evaluated evidence from
passed checks, failures, missing evidence, queued work and stale results.

In Geometry, click a Cylinder, Biplanar or Conical frustum tile; imported supports
retain their own source. The selection changes the same native geometry settings
as the former dropdown and remains undoable. In Winding, choose a circuit diagram
and apply the draft. The diagrams map to the existing connected, per-support,
independent-turn and contour-only modes without changing their mathematical
behavior. Additional backend surfaces remain available through Advanced settings.

Above the viewport, Target, Field, Wire and Connections buttons select available
scientific data. The scope note identifies fallbacks and missing evidence. The
selected task remains visible while changing units, components, current policy,
sample slabs or display layers; an explicit dataset or run change clears it.

The project/assets/service modules under `coilgen.application` are GUI-independent.
Scientific readiness and support parameter definitions live in `preflight` and
`support_parameters`; the Qt modules retain presentation adapters. Preparation
uses the same lightweight preflight for scripts and GUI requests. It checks
configured inputs, native settings and required dependencies without assembling
field operators or invoking a solver; readiness does not certify acceptance.

`ProjectDocument.update_inputs(config_changes, study_changes)` validates both
drafts before publishing one revision. Rejected edits publish neither half;
exact no-ops preserve the original document and native precision. An optional
`expected_fingerprint` rejects stale drafts. The GUI commits the same transaction
as one undo step. Snapshot controls and provenance are checked before writing
owned inputs.

The application `evidence` module owns finite-array availability, exact stored
current variants and source integrity. Missing arrays cannot become selectable
plots through metadata or stage names. `SourceHealth` shares failed verification
across canonical source paths while retaining cached snapshots for inspection;
only an explicit successful verification clears that failure. Verification
reports identify legacy results whose metadata lacks checksum protection.
Original request inspection checks selected identity and artifact bytes without
loading numerical maps.

`execution_reuse` combines verified saved and stopped checkpoints, excluding
live display snapshots. `DesignService.plan` pins owner, revision and study
controls; `submit_plan` rejects stale plans. For `JobManager`, active reuse occurs
under its queue lock and requires the same recorded controls and numerical
request digest, including source, dependency, input and checkpoint contents.
Unknown legacy control fields cannot prove a match. Ordinary `submit` keeps its
existing independent-job behavior. `run_actions` exposes the same recovery,
cancellation and viewing decisions to headless callers and the workbench.

```python
from coilgen.application import DesignService, execution_reuse

def queue_current_project(project, jobs, records=()):
    reuse = execution_reuse(
        project, records, jobs.list(), jobs_directory=jobs.directory,
        threads=jobs.policy.threads_per_worker,
    )
    plan = DesignService.plan(project, states=jobs.list(), reuse=reuse)
    return DesignService.submit_plan(project, jobs, plan)
```

`RunRecord.arrays` still returns detached writable copies. `array(name)` and
`select_arrays(names)` copy only selected arrays; `copy=False` explicitly borrows
read-only views for inspection. `load_run(directory, array_names=...)` verifies
all artifacts while decompressing only requested arrays. GUI display and
comparison avoid the redundant initial full-array copy. These are in-process
read-only contracts, not security isolation from hostile Python callers.

`RunRecord.metadata` returns detached dictionaries and lists. Read-only displays
can borrow `metadata_view`, whose nested mappings are immutable and sequences
are tuples, without copying unrelated winding and fabrication diagnostics on
every progress refresh. Metadata values retain their strict JSON representation.
Passive progress displays use captured canonical source identities and recorded
health; explicit loads, checks and evidence use still resolve and verify files.

Synchronous `DesignService.run(..., progress=callback, cancel_check=predicate)`
delivers typed `ExecutionEvent` objects using the engine's existing progress and
cancellation checks. Recorded boundaries remain distinct from percentages and
activity strings. Returning a cached result emits `result_available`, not a new
stage completion. Observer exceptions are logged without altering the numerical
result or masking its failure. Cancellation remains cooperative and cannot
interrupt an external solver at an arbitrary instruction.

The numerical pipeline exposes five stage boundaries while `run_coilgen()` retains
its original full-run entry point. `coilgen.research.jobs` handles process execution,
and `coilgen.gui` contains the optional Qt controller and PyVista/VTK renderer.

```powershell
python -m pytest -q tests/test_application_backend.py tests/test_native_workbench.py tests/test_pipeline_stages.py tests/test_research_runtime.py tests/test_cli.py
python -m pytest -q tests/test_imported_target_validation.py tests/test_result_metadata_integrity.py tests/test_research_scientific_datasets.py
python -m ruff check src tests projects
python -m ruff format --check src tests projects
```

Controller tests use an injected viewport/job service so they verify ownership,
selection, undo/save/reopen, cancellation and stale-result behavior without GPU
dependence. Native graphics smoke tests must use a supported Qt/OpenGL platform;
a headless numerical installation remains fully usable if graphics initialization
is unavailable.

Model-preview extraction runs in a coalesced background task; only the latest edit
is displayed. Numerical workers and VTK rendering stay separate, and all VTK/Qt
updates occur on the main application thread. CI installs the optional GUI stack to
exercise Qt controls and genuine VTK datasets without requiring an OpenGL context.

The obsolete research browser interface has been retired. Recovery copies of the
removed files and original edited references are under
`tmp/gui_migration/legacy-browser-gui-backup`; they are excluded from package and
test discovery. Its numerical fork, MATLAB reference and supported script
plotbooks/inspectors remain separate from the retired browser frontend.
