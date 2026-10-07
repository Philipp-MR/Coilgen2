# Biplanar gradients and second-order shims

This example contains nine independent channels in one editable, self-contained
CoilGen2 JSON: **Gx, Gy, Gz, Z0, Z2, XZ, YZ, X²−Y² and XY**. Gradients supply
the three first-order modes; Z0 supplies the zero-order uniform-field mode.
Their requested fields span all Cartesian Bz harmonics through degree two.

All target coordinates are world metres. B₀ is 3 T along world z. Every channel
uses the same world-origin spherical ROI with radius **R = 20 mm**, a 40 mm DSV,
the same 257-point design grid, and a distinct denser independent sheet grid.

| Channel | Requested Bz in tesla | Normalization |
|---|---|---|
| Gx | `0.02*x` | 20 mT/m; ±400 µT at x = ±20 mm |
| Gy | `0.02*y` | 20 mT/m; ±400 µT at y = ±20 mm |
| Gz | `0.02*z` | 20 mT/m; ±400 µT at z = ±20 mm |
| Z0 | `100e-6` | Uniform 100 µT perturbation |
| Z2 | `A*(2*z²-x²-y²)/(2*R²)` | +10 µT at either z pole; −5 µT at the equator |
| XZ | `A*2*x*z/R²` | ±10 µT at x = ±z = R/√2 |
| YZ | `A*2*y*z/R²` | ±10 µT at y = ±z = R/√2 |
| X²−Y² | `A*(x²-y²)/R²` | +10 µT on the x axis; −10 µT on the y axis |
| XY | `A*2*x*y/R²` | ±10 µT at x = ±y = R/√2 |

Here **A = 10 µT** is the maximum absolute quadratic Bz value on the ROI sphere,
not the real-Ylm coefficient. The program uses `SolidHarmonicTarget` for quadratic
modes: Y20 coefficient `A/sqrt(5/(4π))`; Y2±1 coefficients
`−A/sqrt(15/(16π))`; Y2±2 coefficients `A/sqrt(15/(16π))`.
The negative XZ/YZ coefficients compensate the evaluator's Condon–Shortley phase.
The fixed harmonic reference radius remains 20 mm when discretization changes.

The supports are 300 × 300 mm planes normal to world x. Channel index i = 0…8
uses x = ±(50 + 6i) mm: nominal pair separation 100, 112, …, 196 mm.
This avoids coincident support surfaces. It is an initial layer allocation;
three-dimensional routing, feeds and mounting can still violate stack clearance.

Gradient conductors are 2 × 2 mm copper, with 6 mm minimum bend radius; shim
conductors are 1 × 1 mm, with 3 mm minimum bend radius. All use 0.1 mm insulation
per side, 18 nΩ·m resistivity and 50 µm manufacturing tolerance. The 1 mm
homogenized optimization sheet thickness is a separate modeling assumption.
The initial support mesh uses 14 × 14 subdivisions on each plane. Refine it and
the field grids before claiming convergence; the starting discretization makes
the complete collection practical to investigate.

Each saved study requests a power-minimizing continuous sheet, a 4% design RMS
budget and 5% independent-sheet budget, 16 gradient or 12 shim contour levels,
routing within each support circuit, physical conductor sweeping, field
evaluation at the original solved current and strict fabrication review.
Contour/wire errors and manufacturing failures remain visible even when a sheet
passes. The default does not supply amplifier-current, voltage, cooling or
duty-cycle limits; those must be added for the intended hardware.

The channels are designed independently. Neither layer separation nor a
per-channel pass establishes inter-channel clearance, mutual inductance,
combined fields, complete feeds/returns, drive limits or construction release.
In particular, this example has two support circuits per channel; their external
series/parallel connection and drive arrangement remain to be designed.

```powershell
.\.venv\Scripts\python.exe examples/biplanar_gradient_shim_workbench.py
.\.venv\Scripts\python.exe examples/biplanar_gradient_shim_workbench.py --gui --output outputs/biplanar_gui
.\.venv\Scripts\python.exe examples/biplanar_gradient_shim_workbench.py --compute --output outputs/biplanar_full
```

Every invocation requires a fresh output directory. `--compute` embeds completed
results into the single JSON after each channel; a failed channel leaves earlier
completed evidence intact and records its error in `channel_execution.json`.
To investigate two channels first:

```powershell
.\.venv\Scripts\python.exe examples/biplanar_gradient_shim_workbench.py --compute --channels gy z2 --output outputs/biplanar_two
```

`--stage prepared` or `--stage sheet` performs an explicitly shortened diagnostic
study. `--no-embed-results` retains results in their run directories instead of
embedding them; the default includes them. No output directory is reused or
overwritten. Full computation can take several minutes depending on hardware
and winding complexity.

## Recorded numerical smoke

The nine default sheet studies completed with independent-grid RMS errors of
3.980–4.310%, within their saved 5% sheet budgets. Their aggregate numerical
runtime was about 40 s on the local single-thread smoke run. This checks sheet
feasibility on these grids; mesh convergence and realized-wire acceptance are
separate questions. The common target basis has rank nine and condition number
approximately one after column normalization. That measures target sampling,
not the conditioning of the surface-current optimization operator.

Two full studies also completed without weakening the geometry checks:

| Channel | Independent sheet RMS | Contour RMS | Routed-centroid RMS | Original evaluation current | Runtime | Strict geometry |
|---|---|---|---|---|---|---|
| Gy | 4.1071% | 4.0084% | 4.1161% | 20.6454 A | 172 s | Failed |
| Z2 | 3.9828% | 4.1146% | 6.8729% | 5.59844 A | 195 s | Failed |

Contour and routed values use the 257 design samples and the original stream
function current, with no refit; they are not independent winding validation.
The runs produced 64 Gy and 52 Z2 contour components, distinct from their 16
and 12 configured levels. Routing continuity, clearance, bend radius and sweep
quality require improvement. Both runs retain failed checks and an unreleased
construction state. A saved collection containing these two full results was
about 264 MB: conductor geometry and provenance make complete portable JSON
files substantial, and a full nine-channel archive will be larger.
