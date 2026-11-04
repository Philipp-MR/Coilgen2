# Mounted-wire bends and return routes

The values below use the actual mounted conductor at each run’s recorded current. P95 describes typical tangent changes over 3 mm. The maximum over eight sampling phases picks up turns that P95 misses. Bend radii and clearance margins are the separate fabrication checks. Locations are world XYZ in millimetres. The compact traces, request hashes and engineering check outcomes are in the [per-channel reports](reviewed_centerlines/).

The selected gallery has 29 channels and 42 closed conductor bodies, matching the original body count and closure state in every channel. All 178,593 exported centerline points match their integrity-verified full run arrays exactly. No selected channel has construction release.

## Changes that required full routing comparisons

| Example | Measured outcome and witness |
|---|---|
| **Biplanar system Gy** | The 24×24 auto-side run (`4d03786c54c929…`) put an almost 178° reversal on plate 2 at **(−53.6, −68.7, 59.7)**. The route already turned ~150° before mounting. Forcing group 2 to its high side (`0d6782a29957c8…`) reduced its eight-phase maximum to **132.1°** versus the original gallery’s 137.6°. Mounted RMS stayed at 4.115%, and pooled P95 improved from 23.87° to 20.60°. Plate 1 still reaches **134.2°** at **(57.0, −68.7, 64.0)** versus the original 114.1°. Each plate’s minimum radius is **0.389 mm** versus 6 mm required. The witnesses are on loop 14 at **(57.0, −23.0, 10.5)** and loop 16 at **(−55.0, −23.0, 10.5)**. Both report folded sweep quads and clearance margin −3.078 mm. |
| **Biplanar system X²−Y²** | The 20×20 trial (`5f1fe4854c640b…`) improved contour RMS from 5.028% to 4.599% but worsened mounted RMS from 13.244% to 16.486%. The first plate contributed −0.113 µT of the −0.130 µT shift in mean DSV residual. The selected 14×14, local-fit-enabled run (`ea36b06bf5f68c…`) kept 72 turns, brought mounted RMS to **10.883%**, reduced mean residual to −0.283 µT and improved P95 from 33.00° to 30.89°. Minimum radius is **0.206 mm** against 3 mm. |
| **Biplanar system Z2** | The 24×24 trial (`cff3f67ac5a8fd…`) yielded 56 turns instead of 52. Mounted RMS increased from 6.277% to **8.737%**, and mean DSV residual shifted from +0.017 to −0.169 µT on a target RMS of 2.781 µT. The selected 52-turn run (`f095b6f4c12780…`) has **6.389%** RMS. P95 improved from 34.95° to 33.72° but the 99th percentile rose from ~52° to 58–60° across the plates. Inspect the contour bends near **(±74, 7.9, 42.8)**. |
| **Standalone biplanar Gy** | Historical / 24×24 (`28972c062c9545…`) / 16×16 fallback (`fc708957d36dc2…`) give mounted RMS **4.191 / 4.097 / 4.179%**, P95 **27.10 / 23.95 / 25.80°**, eight-phase peak **103.9 / 95.8 / 98.1°**, minimum bend **0.996 / 0.306 / 0.388 mm** against 6 mm. At 24×24 inspect loop 17 at **(51.0, 30.7, −5.7)**. Both updated meshes report folding and local profile inversion. All three miss the clearance requirement by 3.078 mm. |
| **Conical harmonic** | The one-ring coarse trial (`7a18c14a753d7b…`) had mounted RMS 20.728%, P95 8.97°, eight-phase peak 177.2°, minimum radius 1.145 mm. The selected 64×24 (`efbd16ff4caf93…`) gives **5.461%, 7.61°, 159.4°, 0.273 mm**. Loop 4’s bend at **(−62.3, −135.0, 87.6)** still misses the 3 mm requirement. Both runs have a route-topology failure and −1.664 mm clearance margin. The refined winding has capsule self-intersections on the assembled path. |

### Cylindrical Z2

| Full run and request | Mounted RMS | P95 at 3 mm | Maximum over 8 phases | Minimum bend, 3 mm required |
|---|---:|---:|---:|---:|
| Historical 32×16, `d6e05150c0a98f…` | 32.239% | 10.26° | 90.1° | 1.680 mm |
| **Selected 32×16**, guarded local fitting enabled, `8b0dda788f0f87…` | **32.237%** | 10.26° | **90.2°** | **1.680 mm** |
| 48×16, `feea9808d96bd1…` | 33.628% | 7.11° | 149.3° | 1.389 mm |
| 64×20, `68c09958a65c28…` | 32.447% | 5.40° | **177.2°** | 1.609 mm |
| 64×32, `e1b69ebdc2fac2…` | 34.013% | 5.41° | 147.0° | **0.170 mm** |

At 32×16 all 32 periodic fits pass. The maximum contour movement is 3.07 µm. At 48×16 the closing connector turns 149.3° near **(80.3, 3.0, 42.7)**, with the minimum bend 0.12 mm away, and the sweep adds a curvature-envelope failure. The 64×20 return near **(80.4, 2.9, 41.5)** almost reverses where the primary connector meets the group. The 64×32 group segment near **(80.4, −0.3, −19.3)** contains its minimum bend. Both 64-division sweeps report a folded quad. Check the area enclosed by the cut and return. The mounted mean DSV residuals are +0.889 and +0.932 µT on a 2.781 µT RMS target. Every candidate fails the bend requirement and has a −1.664 mm clearance margin.

### Conical and biplanar route details

The conical mesh refinement changes the extracted winding topology. The 32×12 solve has 21 positive and 12 negative world-azimuth wraps plus four zero-winding lobes. The 64×24 solve has 20 positive and 11 negative wraps plus four lobes, giving **35 loops instead of 37**. The lobes shift from z ≈ 21–29 mm near the diagonal azimuths to z ≈ 83–96 mm near the cardinal azimuths. They contribute to the measured field. Keep them in the comparison when investigating the remaining tight bends. [Mesh and contour sensitivity](MESH_SELECTION.md)

The refined cone’s fabrication report calls the route failure unspecified. Its saved pipeline route identifies an unresolved normal shift involving twelve centerline intersections, two backtracks and five crossing-ownership conflicts. Inspect these segments and the tightly turning lobes before reworking the conductor lift.

On system Gy, the 20×20 trial (`2b7169594c968e…`) exposed the high-side alternative. Group 2 on plate 2 was the only one of eight group-side choices that differed between the meshes. The high and low 24×24 rail scores were close, 0.1594 and 0.1573. The 20×20 run gave 4.092% RMS and 22.51° P95, but tightened both minimum radii to 0.295 mm. The selected high-side route still reports three inter-group bridge-pair intersections per plate and an unresolved normal shift.

## Other local regressions

| Example and request | Measured change | Design witness and next check |
|---|---|---|
| Cylindrical Gz, `a5ce0850f9042e…` | P95 10.32° → 7.14°, RMS 4.304% → 4.210%. Minimum radius 3.006 → **2.002 mm** against 6 mm. | Loop 1 meets the closing inter-group connector at **(72.0, 36.7, 9.9)**. Check its cut location and turning length. |
| Cylindrical system Gz, `f0c80f2c90fbb4…` | P95 10.16° → 7.04°, RMS 4.227% → 4.294%. Minimum radius 3.48 → **2.653 mm** against 6 mm. | Loop 22 meets the joint spiral at **(92.8, 3.1, −70.4)**. Check the splice and its local turning room. |
| Actively shielded Gy, `6c852eadbaa9ae…` | Primary minimum radius 1.233 → **1.064 mm** against 6 mm. Shield minimum stayed 0.891 mm. RMS on 314 original suppression samples was **29.545 µT** versus 20 µT required. The sheet reached 19.999 µT. | Inspect primary loop 17 near **(−80.1, 9.0, 10.6)** and the shield’s closing connector near **(108.7, −22.1, −42.7)**. Review the external field after changing either path. |
| Biplanar system Gz, `b60adcb949c35f…` | P95 19.28° → 13.85°. Plate radii 1.903 → **1.319 / 1.116 mm** against 6 mm. | Check loop 0 at **(63.0, 19.4, 0.0)** and the other plate’s return/group join at **(−61.0, 2.9, −81.9)**. |
| Cylindrical Z0, `0a6e41a396db10…` | P95 10.34° → 5.50°. Minimum radius 2.905 → **1.771 mm** against 3 mm. | Loop 11 meets the closing connector at **(80.5, −1.1, 82.1)**. Check terminal approach and fillet length. |
| Biplanar system Z0, `e76c196eabeb97…` | P95 21.08° → 13.17°. Minimum radius 2.63 → **1.65 mm** against 3 mm. | Plate 2’s return enters loop 5 near **(−67.5, −3.9, 3.2)**. Plate 1 also reports a clearance failure and sweep self-intersection. |
| Biplanar system XZ, `186ca0e4ebaa9a…` | P95 and peaks improve on both plates. Minimum radius 0.999 → **0.511 mm** against 3 mm. | Inspect loop 0 near **(80.5, 14.0, 0.0)** and **(−79.5, 14.0, 0.0)** before changing the return. |
| Biplanar system XY, `3042b1b963d4c2…` | P95 26.88° → 19.65°. Plate radii 1.806 → **0.606 / 0.743 mm** against 3 mm. | Loop 11 is tight near **(98.5, 6.7, −6.6)** and **(−97.5, 6.7, −6.6)**. A separate return at **(100.4, −65.8, −45.6)** now reaches 125° against 104° previously. |
| Standalone biplanar Gz, `471d57196b0586…` | P95 15.63° → 11.95°. Both minimum radii fell from 1.385 → **0.620 mm** against 6 mm. | Inspect loop 1 at **(51.0, 12.4, −16.7)** and loop 16 at **(−49.0, 12.4, −16.7)**. |

The standalone cylindrical Gx and the cylindrical-system Gx and Gy runs also added a route failure. The biplanar-system Gx and XZ sweeps report surface-mesh self-intersections on both plates. Investigate the relevant interconnect and swept conductor in **Engineering checks** before interpreting improved P95 as an improvement in clearance or conductor validity.

## Reproduce a local turn

Use a matching file from [the new reviewed centerlines](reviewed_centerlines/) and [the previous centerlines](../gui_review_2026-11-03/centerlines/). For these closed paths, the final point repeats the first. Sample each wire at stations separated by its length divided by the nearest integer to `length / 0.003 m`. The reported single-phase values use the first saved point as station zero. To test phase sensitivity, offset the stations by 0, ⅛, …, ⅞ of the station interval. Interpolate only to measure tangent changes; all plotted geometry and engineering checks use the original saved coordinates.

```python
import numpy as np

def local_turn(points_m, phase=0):
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points_m, axis=0), axis=1))]
    n = max(4, round(arc[-1] / 0.003))
    stations = (np.arange(n) + phase) * arc[-1] / n
    xyz = np.column_stack([np.interp(stations, arc, coordinate) for coordinate in points_m.T])
    tangent = np.roll(xyz, -1, axis=0) - xyz
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    dot = np.sum(tangent * np.roll(tangent, 1, axis=0), axis=1)
    angle_deg = np.degrees(np.arccos(dot.clip(-1, 1)))
    i = angle_deg.argmax()
    return angle_deg[i], xyz[i] * 1000, np.searchsorted(arc[1:], stations[i], side="right")
```

Select `review_conductor_0_points_m` for body 1, or index 1 for body 2. The returned segment index addresses `review_conductor_<i>_owner_kind` in the full run’s `scientific_result.npz`. The saved diagnostic `fabrication.paths[i].diagnostics.sweep_final_min_bend_point_index` addresses a vertex; inspect both adjacent segment owners. Each per-channel JSON identifies the model, request and compact centerline checksum. Field errors use each run’s own original drive current on its design samples.
