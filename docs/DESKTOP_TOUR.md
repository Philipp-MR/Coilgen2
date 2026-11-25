# CoilGen2 desktop image tour

[Back to the welcome page](../README.md) · [Desktop guide](NATIVE_WORKBENCH.md) ·
[Installation](INSTALL.md) · [Browser guide](BROWSER_WORKBENCH.md)

A visual walk through the major desktop workflows: define the support and field,
compute a stream function, refine the winding, evaluate the result and prepare
geometry or PCB artwork. All images use the same white background, light Qt
theme and typography. Open any image for its full-resolution PNG.

These are captures of production desktop widgets and their actual rendered
plots, at twice the normal pixel density. The winding and field views use a
saved cylindrical Gy example. The CAD and shielded-system views demonstrate
separate configured examples. Values, units, source labels and warnings remain
visible; the images do not establish fabrication or measured performance.

Jump to [project workflow](#1-start-and-organize-a-design),
[geometry and setup](#2-define-geometry-target-and-physical-assumptions),
[winding editing](#3-turn-the-stream-function-into-a-winding),
[PCB layout](#4-develop-pcb-and-flexible-pcb-artwork),
[field analysis](#5-evaluate-the-magnetic-field),
[checks and export](#6-review-engineering-checks-and-prepare-geometry), or
[multichannel systems](#7-work-with-multiple-channels-and-active-shielding).

## 1. Start and organize a design

### Welcome screen

Start a new design, browse configured examples or reopen a portable project.

![Native welcome screen with project and example entry points](images/desktop-tour/01-welcome.png)

### Main workbench

Common settings sit beside the 3D view. The workbench identifies the editing
revision and the saved result being displayed. The example below shows the
reviewed conductor field, its stored current convention and target samples.

![Design setup and saved cylindrical Gy result](images/desktop-tour/02-workbench.png)

### Design stages and saved choices

Use the design plan to follow the computation stages and inspect the inputs
and choices associated with a saved run.

![Design plan alongside the winding and saved run](images/desktop-tour/05-design-plan.png)

## 2. Define geometry, target and physical assumptions

### Geometry studio

Build CAD primitives, subtract openings, select retained faces and generate a
surface mesh. This example is a cylinder with a transverse opening, generated
from the repository's saved geometry recipe.

![Geometry studio with the windowed-cylinder surface mesh](images/desktop-tour/20-geometry-studio.png)

### Support and mesh

Choose the support, set dimensions and placement, then control mesh resolution.
Dimensions are displayed with their units.

![Support selection, dimensions, placement and mesh controls](images/desktop-tour/06-supports.png)

### Target field and region

Define the desired field and the physical region in which it is evaluated.
Additional region and suppression controls are available from the focused editor.

![Target field and region settings](images/desktop-tour/07-target.png)

### Optimization

Inspect the objective, regularization, constraints and solver choices before
computing. These controls affect the physical design, so the saved run retains
the submitted configuration.

![Optimization and constraint settings](images/desktop-tour/08-optimization.png)

### Conductor

Set the conductor cross section, dimensions and material assumptions used by
geometry generation and electrical estimates.

![Conductor profile and material controls](images/desktop-tour/09-conductor.png)

## 3. Turn the stream function into a winding

### Stream function

Inspect the optimized scalar stream function on the support before evaluating
the discrete winding. The stream-function solution and the realized conductor
remain distinct result representations.

![Stream function on the cylindrical support](images/desktop-tour/03-stream-function.png)

### Contours, smoothing and routing

Control the extracted contours and their conditioning, then configure routing.
Accepted smoothing decisions can be inspected with the saved design choices.

![Contour, smoothing and routing controls](images/desktop-tour/10-winding.png)

### Groups and circuits

Inspect the contour hierarchy, assign groups to circuits and choose spiral
order. Group data comes from the verified contour checkpoint.
This view inspects the original contour stage, before the later interconnection edits.

![Contour groups plotted above circuit assignment controls](images/desktop-tour/21-groups-circuits.png)

### Interconnection editor

Work with linked 3D and flat views to inspect seams, connect endpoints and shape
connections. Draft edits are reviewed before they are applied to the design.

![Interconnection editor with matching 3D and flat winding views](images/desktop-tour/11-interconnections.png)

## 4. Develop PCB and flexible-PCB artwork

[Read the PCB workflow](PCB_LAYOUT.md) for supported parameterizations and the
meaning of each check. Cylindrical and planar layouts can be flattened through
the supported charts. A hemisphere is not currently supported for distortion-free
PCB flattening and is rejected with an explanation.

### Linked flat and wrapped views

Choose the layout source, opening seam, copper width and layer visibility.
Selection links the flat artwork to the installed-coil view. Top and bottom
layers share one origin; their coordinates are not silently mirrored.

![PCB artwork and its installed cylindrical shape](images/desktop-tour/12-pcb-layout.png)

### Connections and layer transitions

Select endpoints in either view and inspect a proposed connection, bridge or
via before committing it to the draft.

![PCB connection controls and linked endpoint highlights](images/desktop-tour/13-pcb-connections.png)

### PCB checks and SVG/DXF export

Review grouped issues and their location in both views. Export SVG copper and
DXF outlines for inspection or downstream PCB design. This draft intentionally
shows unresolved connectivity and geometry issues; its field is not evaluated.
The export action therefore identifies the result as inspection artwork.

![Grouped PCB issues and SVG/DXF inspection export controls](images/desktop-tour/14-pcb-checks.png)

## 5. Evaluate the magnetic field

### Target, actual field and residual

Compare matched sections with a shared target/actual scale and a separate
residual scale. The source and current policy are explicit. Statistics identify
the saved samples on which they were calculated.

![Target, actual Bz and signed residual comparison](images/desktop-tour/15-field-maps.png)

### Slices through the target region

Inspect multiple sections with one color scale. The view distinguishes physical
reevaluation from interpolation of saved samples; unsupported locations remain
masked. The captured slice view uses interpolated saved samples, as labeled.

![Five field-error slices through the target region](images/desktop-tour/16-field-slices.png)

### Error distribution

Examine how residuals vary across samples and inspect the corresponding
statistics with the same selected source and current policy.

![Field error analysis for the selected saved conductor](images/desktop-tour/17-field-errors.png)

### Solid harmonics

Fit the field, target or residual to the selected real solid-harmonic basis.
Review sample count, rank, conditioning and fit residual alongside the terms.
The displayed fit does not by itself establish source-free validity.

![Solid-harmonic terms and numerical fit diagnostics](images/desktop-tour/18-harmonics.png)

### Coil statistics

Review the recorded current, conductor geometry and electrical estimates.
Unavailable quantities and the assumptions attached to estimates remain visible.

![Recorded coil statistics and conductor geometry](images/desktop-tour/19-coil-statistics.png)

## 6. Review engineering checks and prepare geometry

### Engineering evidence

Inspect saved acceptance checks before acting on the geometry. A passed saved
check applies to the recorded calculation and assumptions, not to subsequent
edits or physical commissioning.

![Saved engineering checks and acceptance evidence](images/desktop-tour/22-engineering-checks.png)

### Swept conductor

Inspect the 3D volume generated from the winding and conductor cross section.
This is the geometry used for volume inspection and the available conductor
export workflow.

![Swept conductor on the cylindrical support](images/desktop-tour/04-swept-conductor.png)

### Geometry export

Choose the available saved geometry and export format. Export status and
diagnostics remain associated with the selected result.

![Geometry export controls for a saved result](images/desktop-tour/23-geometry-export.png)

### Printable former

Configure the former and wire-groove settings for a winding support. The image
shows the setup controls; it does not claim that a former has been fabricated.

![Printable former and groove configuration](images/desktop-tour/24-print-former.png)

## 7. Work with multiple channels and active shielding

### Shielded XYZ example

Load the configured actively shielded XYZ foil-gradient system and inspect its
primary and shield supports. This screenshot shows the setup preview, before
computation. For computed leakage-field comparisons and numerical context, see
the [shielded XYZ example](../examples/shielded_xyz_gradient_README.md).

![Actively shielded XYZ gradient system setup](images/desktop-tour/25-shielded-system.png)

### Channel overview

Review each independent channel's target, support, conductor, computation state
and available evidence. The configured channels shown here have not been run.
Open **Channel configuration…** for the selected channel's physical settings.

![Independent-channel configuration and execution overview](images/desktop-tour/26-channel-overview.png)

## Capture details

The [capture script](frontpage/capture_desktop_tour.py) renders the production
Qt widgets and registers their VTK framebuffers at the widgets' native bounds.
It changes presentation settings only. It does not repaint controls, invent
results or remove warnings. The [capture manifest](images/desktop-tour/capture-manifest.json)
records dimensions, pixel density and image hashes.

See [capture instructions](frontpage/README.md#desktop-feature-tour) to regenerate
the images with a compatible saved project and run.
