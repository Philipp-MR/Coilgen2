# Cylindrical XYZ gradient system

Load **File → Load template → Coil systems → Cylindrical XYZ gradient system
(Gx, Gy, Gz)**. The template creates three independently driven, editable
channels in a single CoilGen2 project. Their complete studies and physical
settings are already applied; no settings windows must be visited before
computing. Choose the active channel to inspect its inputs or results, and use
**Channels → Compute enabled channels independently** to run all three studies.
The **Editing channel** selector above the plot switches between Gx, Gy and Gz.
All three support meshes appear together; **System meshes** in the plot toolbar
lets you show or hide individual supports. Hiding a mesh does not exclude its
channel from computation.

This is an unshielded small-bore gradient insert, with a common world-origin
**40 mm spherical DSV** and **B₀ = 3 T along world z**. The gradient axes describe
the spatial derivative of **Bz**; they do not change the field component or rotate
the support away from the common bore axis.

| Channel | Requested Bz in tesla | Cylindrical support radius |
|---|---|---|
| Gx | `0.02*x` | 80 mm |
| Gy | `0.02*y` | 86 mm |
| Gz | `0.02*z` | 92 mm |

Coordinates x, y and z are world metres. Each gradient is **20 mT/m**, giving
±400 µT at ±20 mm along its named axis. Every support is **200 mm long**,
centered at the world origin and coaxial with world z. The nominal radial pitch
is 6 mm, so support surfaces are separate rather than coincident. Each initial
mesh has 32 angular and 16 axial divisions, with open ends and an actual
200 mm axial span. The template compensates for the legacy generator's axial
span convention without changing that numerical generator.

Each channel uses **2 × 2 mm copper**, 18 nΩ·m resistivity, 0.1 mm insulation per
side, 50 µm manufacturing tolerance and 6 mm minimum bend radius. The **1 mm
homogenized sheet thickness** used for continuous-sheet power minimization is
separate from the physical conductor size. Each study requests a 3% RMS sheet
design budget, a distinct 5% independent sheet check, 16 contour levels, winding
connection, conductor sweeping, inductance calculation and strict fabrication
review. Fields are evaluated at the solved stream-function current. Failed
discrete-wire or geometry checks remain visible; the template contains no saved
acceptance claims.

The channels use independent power-minimizing solves. This example is ready for
computation, not released for construction. Nominal support spacing does not
establish conductor, connector or feed clearance. Combined field quality,
mutual inductance, external stray field, amplifier current/voltage, cooling,
duty cycle, mechanical loading and the full terminal/return arrangement must
be reviewed for the intended hardware. There is no active shield. Refine the
support and field grids to establish numerical convergence before accepting a
winding.

To save or open the same assembly from a script:

```powershell
.\.venv\Scripts\python.exe examples/cylindrical_xyz_gradient_workbench.py
.\.venv\Scripts\python.exe examples/cylindrical_xyz_gradient_workbench.py --gui --output outputs/cylindrical_xyz_gui
.\.venv\Scripts\python.exe examples/cylindrical_xyz_gradient_workbench.py --compute --output outputs/cylindrical_xyz_full
```

Every invocation requires a fresh output directory. Full computation requests
all three studies and embeds completed results into
`cylindrical_xyz_gradient.coilgen.json` after each channel. Failed channels are
recorded in `channel_execution.json`; earlier completed evidence is preserved.
For a shorter diagnostic, use `--compute --stage prepared` or `--stage sheet`.
Use `--channels gx gy` to select channels, and `--no-embed-results` to keep
completed results in their run folders instead of embedding them.

The reusable GUI-independent entry point is:

```python
from coilgen.application.system_presets import system_assembly

assembly = system_assembly("cylindrical_xyz_gradient", project_root="outputs/xyz")
assembly.save("outputs/cylindrical_xyz_gradient.coilgen.json")
```

## Recorded sheet smoke

The three default sheet studies completed with 257 design samples and 3112
distinct independent-grid samples. Independent Bz RMS error was **3.1089% for
Gx**, **3.1077% for Gy** and **3.1910% for Gz**, all within the saved 5% budget.
The three single-thread numerical runs totaled about 20 s locally. This checks
continuous-sheet feasibility at the initial discretization and original solved
current; it does not establish mesh convergence, discrete winding quality,
fabrication readiness or system qualification.

## Recorded full-study smoke

All three default full studies completed through contouring, connected winding,
conductor sweeping, field evaluation and strict fabrication review, with no
execution failures. Each completed run was embedded in the single editable
assembly JSON; the full three-channel artifact was approximately **316 MB**.

| Channel | Independent sheet RMS | Contour RMS | Native routed RMS | Fabrication-centroid RMS | Original current | Runtime |
|---|---|---|---|---|---|---|
| Gx | 3.1089% | 2.9892% | 4.5797% | 4.1120% | 18.6343 A | 108.6 s |
| Gy | 3.1077% | 2.9903% | 4.5755% | 4.0883% | 21.4784 A | 95.9 s |
| Gz | 3.1910% | 2.7134% | 4.6873% | 4.2202% | 11.6933 A | 72.9 s |

Wire values are Bz residuals on the 257 design samples at the original solved
current, with no refit. The native evaluated winding and the separately
mesh-realized fabrication profile centroid are distinct thin-wire geometries;
their stored maps retain that distinction. These values are not independent
winding validation or finite-cross-section field integration.

**Strict geometry review failed for every channel**, and construction release
remained false. Routing topology/clearance, sweep self-intersections,
minimum bend radius, curvature, frame sampling and support penetration need
improvement. Gx/Gy also reported folded sweep faces; Gy additionally reported
local profile inversion and poor triangle quality. All studies produced a
connected winding and conductor body, but those bodies remain engineering
review candidates rather than fabrication-ready solids. Inter-channel geometry
and electromagnetic coupling are still separate system checks.
