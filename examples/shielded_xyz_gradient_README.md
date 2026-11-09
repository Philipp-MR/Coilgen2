# Actively shielded XYZ foil gradient system

In the desktop workbench choose **File → Load template → Coil systems →
Actively shielded XYZ foil gradient system (Gx, Gy, Gz)**. The browser
template picker offers the same assembly. **Compute enabled channels
independently** runs one joint primary/shield solve per channel.

The recorded evaluation produced local artifacts under
`outputs/shielded_xyz_gradient/`. The evaluated project,
`shielded_xyz_gradient.coilgen`, opens in both workbenches and includes all
three computed studies and their checkpoints (491.6 MiB).
`shielded_xyz_inputs.coilgen` contains the inputs (50 KiB). Use the gallery
template or the commands below to create a project in a fresh checkout.

Open a completed channel result and select **View → Plot shielding surface…**
on desktop or **Shielding surface** in the browser. Compare the primary, shield
and combined fields on the same scale. Switch the source from **Stream function model** to
**Routed winding** to see how the realized winding changes the external field.
On desktop, select a **triangle-interior check** and choose **Compute interior
samples** to evaluate between the original optimization vertices.

| Channel | Target Bz (T), coordinates in world metres | Primary radius | Shield radius |
|---|---|---:|---:|
| Gx | `0.020*x` | 80 mm | 110 mm |
| Gy | `0.020*y` | 86 mm | 116 mm |
| Gz | `0.020*z` | 92 mm | 122 mm |

All six supports are 300 mm long, centered at the world origin and coaxial
with z. The 40 mm spherical DSV has a ±400 µT target range for each channel.
Each primary/shield pair shares a contour current. Separate routes on the two
supports leave the interlayer connection and external feeds to the designer.
The example assumes rectangular copper foil 2 mm wide × 0.20 mm thick,
18 nΩ·m resistivity, 0.10 mm insulation per side and a 6 mm minimum bend
radius. The continuous-sheet power objective uses a 0.20 mm effective thickness.

The secondary target is a closed observation cylinder, radius 150 mm and
length 400 mm, outside every winding. The stream function solve requests zero external
field with a 20 µT area-weighted vector RMS limit. For vertex fields
`B_i = B_primary,i + B_shield,i` and vertex areas `a_i`, the metric is
`sqrt(sum(a_i * |B_i|²) / sum(a_i))`. The reference primary field uses the
primary currents of the same joint design. The 24 × 12 enclosure mesh samples
13 sidewall rings and both cap centers. Refining the sidewall and end-cap
sampling is a separate convergence check.

The following commands use the shared Windows installation. Choose a fresh
output directory for each invocation.

```powershell
$env:PYTHONPATH = "src"
& "$env:USERPROFILE\.coilgen2\venv\Scripts\python.exe" examples/shielded_xyz_gradient_workbench.py
& "$env:USERPROFILE\.coilgen2\venv\Scripts\python.exe" examples/shielded_xyz_gradient_workbench.py --compute --output outputs/shielded_xyz_full
```

The first command writes a portable three-channel JSON with embedded meshes.
The second evaluates and saves the windings. Add `--stage sheet` to compute
the continuous design only. Per-channel studies omit mutual inductance,
eddy-current response and simultaneous drive. Qualification also requires
conductor and feed clearance, power and thermal analysis.

## Computed fields at the observation vertices

Values are area-weighted RMS vector magnitudes in µT on the same 314 vertices.
Each primary-only column uses that primary's current distribution from the
joint solve. The combined field includes its active shield at the same drive.

| Channel | Shared foil current (A) | Stream function primary | Stream function combined | Routed primary | Routed combined | Routed reduction |
|---|---:|---:|---:|---:|---:|---:|
| Gx | 19.03 | 127.61 | 20.00 | 127.77 | 28.61 | 4.47× |
| Gy | 22.80 | 175.91 | 20.00 | 176.16 | 20.76 | 8.49× |
| Gz | 15.50 | 210.05 | 20.00 | 208.35 | 14.54 | 14.33× |

Relative RMS error in routed Bz on the 257 design ROI samples was 3.53%,
4.25% and 4.71% for Gx, Gy and Gz. The Gx and Gy routed windings exceeded
the 20 µT exterior budget used in the optimization. The Bz errors from the stream function on the independent
3,112-point ROI grid were 4.06%, 4.04% and 4.23%. All three channels failed
strict fabrication review, including route clearance, sweep self-intersection
and minimum bend radius checks on both supports.

### Field between observation vertices

The triangle-interior check reevaluates the field from the same stream function or routed winding at
1,872 points on the 624 facets of the observation cylinder. It uses the three
barycentric permutations of (2/3, 1/6, 1/6), with one third of the triangle's
area per point. The end caps use the same rule as the sidewall. Reported peaks
are maxima over these points.

| Channel | Stream function net RMS (µT) | Stream function sampled peak (µT) | Routed net RMS (µT) | Routed sampled peak (µT) |
|---|---:|---:|---:|---:|
| Gx | 23.36 | 40.55 | 31.28 | 87.08 |
| Gy | 24.14 | 44.76 | 24.65 | 45.43 |
| Gz | 18.29 | 34.42 | 13.64 | 46.21 |

For Gx and Gy the stream function predicts more than 20 µT net RMS between the optimization vertices.
For a magnet-specific evaluation, place the observation mesh at the actual
conducting boundary and refine it until the leakage metrics of interest converge.
