# Gallery support and winding comparison

The screenshot showed the **reviewed connected conductor** for the cylindrical X²−Y² shim. Its run requested mesh-scale smoothing before routing. On the 32 × 16 support, 24 of 48 closed turns accepted a periodic fit. Eight candidate fits exceeded the displacement cap and 16 retained their original contours for spacing. With 12.5 mm axial intervals and approximately 15.7 mm circumferential intervals, many visible corners originated in the piecewise-linear stream function on this mesh.

The updated model resolves the same 200 mm long, 80 mm radius support at 64 × 32. It retains all 48 turns, their circulation, the 10 µT target mode, the original study tolerances and 1 × 1 mm copper. All 48 periodic fits pass the 0.5 mm cap and spacing guard. Their maximum measured movement is 0.4153 mm. No local fallback is needed for this shim on the refined mesh.

| Cylindrical X²−Y² | 32 × 16, screenshot mesh | 48 × 24 trial | 64 × 32, new template |
|---|---:|---:|---:|
| Original contour-step current (A) | 1.93716 | 1.93403 | 1.92105 |
| Contours accepting a periodic fit | 24 / 48 | 48 / 48 | 48 / 48 |
| P95 turn angle at 3 mm stations, raw → fitted (°) | 26.89 → 25.19 | 19.72 → 12.13 | 15.77 → 11.23 |
| Raw → fitted contour RMS error, 257 design samples (%) | 3.908 → 3.916 | 3.926 → 3.933 | 3.912 → 3.916 |
| Reviewed connected winding RMS error, same current (%) | 5.041 | 5.06 | 5.019 |

The first column uses the earlier gallery's paired contour measurements and the actual screenshot run's reviewed error. The smoother works on the same solved stream function and current within each raw/fitted pair. Changing the support resolution requires a new sheet solve, so currents differ **between** columns. The independent sheet check at 64 × 32 gives 3.729% relative RMS. The final pipeline sweep still reports a minimum bend radius of 1.080 mm against the requested 3 mm.

The refined X²−Y² support has 2,112 vertices and 4,096 triangles, versus 544 vertices and 1,024 triangles in the earlier example. Its full background evaluation took about 646 seconds with one numerical thread on this system while several other jobs were running. This is an observed run time, not a controlled speed benchmark.

## Resolution choices

- Generated cylindrical gradients, including the rotated Halbach inserts and three-channel system, use 48 × 24. Refinement preserves the physical cylinder span including the legacy mesh builder's axial end convention.
- X²−Y², XY, XZ, YZ and Z0 use 64 angular and 32 axial divisions. Axisymmetric cylindrical Z2 retains its 32 × 16 ring mesh because its mounted connector performs better there. A 64 × 20 comparison kept all 32 contours, reduced their 3 mm P95 angle from 9.81° to 5.25°, and reduced fixed-current contour RMS from 13.15% to 12.97%. However, the mounted wire increased to 32.45% RMS, with a 177° near-reversing connector. The 48 × 16 intermediate increased mounted RMS to 33.63%, its worst turn reached 149°, and its tightest bend was 1.389 mm. The 32 × 16 control gives 32.24%, 90° and 1.680 mm. The winding field is sensitive to axial sampling (15.43% before smoothing at 64 × 24).
- Standalone biplanar gradients use 24 × 24 on each 200 mm plate. For the larger nine-channel system, Gx, Gy, Gz, Z0, XZ, YZ and XY use 24 × 24 on each 300 mm plate. The Z2 and X²−Y² layers use 14 × 14 after comparing the connected wires. The Z2 refinement adds four identifiable turns but worsens the mounted-wire field. For X²−Y², a 20 × 20 comparison retained all 72 and improved fitted contour RMS to 4.599%; its mounted RMS worsened to 16.486%. The selected 14 × 14 case with bounded local fits gives 10.883% mounted RMS versus 13.244% in the previous gallery. Its two mounted bodies have pooled eight-phase P95 angles of 31.05° and 30.76°, compared with 32.88° and 33.38° before. A 24 × 24 X²−Y² comparison lost 12 turns and gave 18.36% fitted contour error.
- Actively shielded Gy retains its jointly constrained primary and shield supports and its original external suppression samples. The improved bounded fit works on those same surfaces.
- Conical Z2 uses the 64 × 24 mesh. All 35 extracted contours accept bounded fits (28 periodic, seven local), and the signed winding families recur on finer meshes. At the new original current of 1.59844 A, fitted contour RMS is 4.89% and mounted-wire RMS is 5.46%. On the 37-turn 32 × 12 control, using one surrounding ring for the surface chart gave 20.73% mounted RMS. Increasing only angular divisions to 36, 40, 48, 56 or 64 on the original 12 axial intervals preserved 37 contours but worsened raw contour field error in each case.

The [mesh sensitivity table](mesh_trials.csv) reports the unprocessed contour field and the current actually obtained for each solved mesh. The [complete wire comparisons](route_trials.csv) record submitted-request hashes, fixed-current field errors and per-body turning, bend radius and length for the routed alternatives. The eight-phase columns repeat equal-arclength sampling with eight choices of the starting station within one 3 mm spacing. P95 and P99 pool all phases. Peak reports the maximum across those phases. These are path-direction changes measured directly on the saved mounted geometry. A lower residual on one design grid is insufficient to establish mesh convergence.

The [candidate evidence archive](candidate_evidence.zip) retains each measured alternative's request, manifest, result metadata and readable report. New alternatives also contain the inputs-only `source.coilgen` that owns the exact project ID, revision and fingerprints recorded in its result. Historical comparison rows reuse the text records from the previous gallery. `index.csv` maps the case and variant to its request hash and indicates whether an editable source is included. Historical stage caches and dense field files are excluded. [Archive and verify the receipts](archive_choices.py).

For the standalone biplanar Gy two-body winding, the bounded 16 × 16 control gave 4.179% reviewed RMS and 25.80° P95. The selected 24 × 24 example gives 4.097% and 23.95°. The peak over eight sampling phases also fell (98.1° to 95.8°), but the tighter bend deteriorated from 0.388 mm to 0.306 mm. Both fail the 6 mm request. These checks concern the standalone 200 mm plate pair. The larger nine-channel system uses a different support and has its own return geometry.

On the biplanar **system** Gy layer, the automatic 24 × 24 cut left a nearly reversing connector on plate 2. Selecting the high cut for zero-based group 2 reduces its 3 mm peak angle over eight station phases from 177.9° to 132.1°. Plate 1's peak remains 134.2°. Eight-phase P95 changes from 22.55° to 21.77° on plate 1 and 19.93° to 19.94° on plate 2. The stricter geometric failures remain: minimum bend is 0.389 mm and the clearance margin is −3.078 mm. Mounted RMS is 4.115% versus 4.099% with the automatic cut and 4.115% in the previous gallery. The three other indexed group cuts remain automatic. Automatic 14 × 14 and 20 × 20 alternatives were also fully evaluated.

In the selected biplanar system X²−Y², both the mean residual and the spatially varying residual improve. The mean changes from −0.338 to −0.283 µT. Relative RMS after subtracting that mean falls from 8.06% to 6.37% as a diagnostic. The reported **10.883%** mounted-wire RMS retains the uniform residual and uses the full error without fitting or subtracting an offset.

On the cone, the 32 × 12 solution has 21 forward wraps, 12 reverse wraps and four nonwrapping lobes. All sampled 64 × 16, 64 × 20, 64 × 24, 72 × 28 and 80 × 32 refinements have 20 forward wraps, 11 reverse wraps and four nonwrapping lobes. The four lobes move from level 4 near z = 25 mm to level 0 near z = 90 mm, counting ranks from zero. Their raw lengths decline from about 330 mm on the coarse mesh to 260, 225 and 221 mm on the three finest meshes. Their maximum 3 mm heading reversals are 174°, 159°, 143° and 120°, respectively. At 72 × 28 and 80 × 32 the original currents agree to 0.09%, with raw contour RMS errors of 5.99% and 6.13%. The lobes remain present on both meshes. [Per-contour signed winding and extent](cone_census.csv).

The cone choice has a spatially concentrated cost. From 32 × 12 with a one-ring chart to the selected 64 × 24 case, eight-phase mounted P95 fell from 8.96° to 7.61° and peak from 177.15° to 159.43°, but P99 increased from 18.37° to 21.36°. The smallest mounted bend radius decreased from 1.145 mm to 0.273 mm, below the 3 mm target. The tight bend is on a contour lobe near z = 88 mm. Both final reviews report an unspecified route-topology failure, self-contact and the same −1.664 mm clearance margin. The refined example removes the folded-quad failure of the coarse one. Neither layout has construction release.

The biplanar system Z2 acquires one additional small contour at each of its two middle levels on each support when the mesh changes from 14 × 14 to 16 × 16. The added components persist up to 24 × 24. Their perimeters are around 54 mm at 20 × 20 and 62 mm at 24 × 24, alongside contours of about 424 and 767 mm on the same levels. The full 24 × 24 review shows the tradeoff. Typical turning improves and its tightest reviewed bend increases to 0.403 mm, compared with 0.305 mm at 14 × 14, but the mounted fixed-current RMS error rises to 8.737% from 6.389%. Both fail the 3 mm bend requirement. [Per-contour planar area and extent](plate_z2_census.csv) · [Repeat the winding census](census.py).

## Supplementary mounted-wire field check

For 18 shortlisted complete runs, the saved electrical path of each modeled circuit was evaluated with the finite straight-segment Biot–Savart kernel and its recorded signed current. The reconstructed **three-component** magnetic field agreed exactly with the saved vector field at every original design point. The independent-grid comparison then excluded the eight positions shared with the 257-point design grid and evaluated the wires at the remaining **3,104** positions. All alternatives within each channel used identical coordinates and target vectors. No current, gain, offset or spatial mode was fitted. These values are supplementary calculations, separate from official run verdicts.

| Comparison | Mounted Bz relative RMS on the 3,104 separate points |
|---|---:|
| Cylindrical X²−Y² screenshot → recommended winding | 4.656% → 4.634% |
| Conical Z2, 37-turn one-ring control → 35-turn template | 19.103% → 5.307% |
| Biplanar system X²−Y², 20 × 20 trial → selected 14 × 14 | 15.243% → 10.086% |
| Biplanar system Z2, 24 × 24 trial → selected 14 × 14 | 8.325% → 6.225% |
| Biplanar system Gy, 24 × 24 automatic cut → selected group-2 high cut | 4.212% → 4.229% |
| Standalone biplanar Gy, 16 × 16 → selected 24 × 24 | 4.283% → 4.198% |
| Cylindrical Z2, 48 × 16 intermediate → selected 32 × 16 | 31.061% → 29.775% |

[Numerical results and request hashes](verified_wire_holdout.csv) · [Held-out positions, targets and B vectors](verified_wire_holdout.npz) · [Reproduce from the saved runs](check_wire_holdout.py).

## Repeating the contour measurements

The [paired gallery matrix](gallery.csv) records the physical model fingerprint, numerical source hash, current, bounded fits and field and shape measurements for each study. Use the CoilGen2 environment and set `PYTHONPATH=src` from the checkout. [Run the whole gallery or one named channel](compare_contours.py). The benchmark compares raw and fitted contours from the same solved sheet, verifies their closure, level, orientation, multiplicity and triangle/barycentric provenance, and checks the configured movement and spacing bounds. The P95 measure uses equal arclength stations approximately 3 mm apart on the closed physical curves. First and last tangents are treated as neighbors. Supplementary 1 mm measurements are available in the matrix. These turn angles depend on the sampling distance. For live checkpoints use a local unsynced directory with `--scratch`.

```powershell
python validation/gui_winding_quality_2026-11-03/compare_contours.py --workers 2
python validation/gui_winding_quality_2026-11-03/compare_contours.py --case shim_x2_y2__0 --mesh 48 24
python validation/gui_winding_quality_2026-11-03/compare_contours.py --case biplanar_gradient_shim__7 --mesh 24 24
```

The 3 mm station measurements describe pre-routing contour geometry. Changes introduced by cuts, interconnections, sweeping or mounting must be measured from the separately saved reviewed conductor. The earlier gallery's reference comes from its [paired study](../gui_review_2026-11-03/smoothing.csv).
