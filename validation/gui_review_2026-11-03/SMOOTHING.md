# Gallery contour smoothing

The 19 retained gallery examples contain 29 channel studies on 42 supports. All 29
completed with at least one accepted fit. Of 1,051 extracted contours, 482
took the fitted path. The engine kept 358 originals because the fit exceeded
the displacement limit, 24 because the fitted path revisited a surface face
after leaving it, and 187 to preserve turn spacing. Maximum measured movement
was 0.499718 mm. Reconstructing all endpoints from their recorded triangle
vertices and barycentric weights agreed within 1.46 × 10⁻¹⁶ m. All curves
kept their closure, potential, current orientation and turn multiplicity.

At approximately 3 mm arc-length intervals, the measured 95th-percentile
turn angle decreased by more than 0.01° in 24 channels and stayed within 0.01°
in four. It increased for the conical Z2 channel, from 8.462° to 9.216°.
Halbach Gx had the largest decrease, from 20.294° to 13.325°. Individual
contour-processing stages took 5.52–51.79 s during the concurrent benchmark.

| Paired case | Original | Smoothed | Quantity at the original current |
|---|---:|---:|---|
| Biplanar system X²−Y² | 4.709% | 5.028% | Unconnected contour DSV RMS, largest increase |
| Biplanar system X²−Y² | 13.729% | 13.244% | Fabricated DSV RMS, largest absolute change |
| Conical Z2 | 20.739% | 21.130% | Fabricated DSV RMS, largest increase |
| Standalone cylindrical Gy | 1.169 mm | 0.266 mm | Minimum pipeline bend radius, against a 6 mm limit |

All 29 sheet checks passed. All 29 pipeline sweeps and strict fabrication
reviews failed. Each case kept the same engineering
verdict as its original, and none received construction release.

Every new gallery example and every channel in the two systems requests a periodic
B-spline fit before cuts and interconnections. The fit uses a local mesh length
scale. It requests 10% retention of an oscillation at that wavelength and traces
the resulting polyline on the original surface triangles. The maximum permitted
sampled displacement is 0.5 mm. This is half the smallest gallery wire width.
The algorithm can weaken a fit, or keep the extracted contour when a fit fails
the displacement, topology, or turn-spacing checks. Straight segments on facets
of the original support can therefore remain visible.

The recorded results carry one count per support for accepted fits, fit
rejections and spacing rejections, plus the maximum measured displacement over
accepted fits. Rejections preserve the original turn. These are pre-routing
counts. Cuts, return paths, crossovers, bend treatment and conductor sweeps
affect the final wire positions.

Twenty-four paired runs reused their archived sheet checkpoints. For five
biplanar shims, the engine's source digest changed when the text-persistence
helper was updated. The new engine declined the older checkpoints and recomputed
their sheets. All 29 pairs had bit-identical stream functions, sheet fields,
independent sample coordinates, holdout targets and holdout fields, including
the same original drive currents.
All field RMS percentages in the CSV refer to their explicitly named geometry.
Wire and fabrication comparisons use the original stream-function current on
the design DSV, without refitting. The independent sheet samples are a separate
check. The turn-angle comparison measures the 95th percentile of changes in
direction along closed contour centerlines, sampled at approximately 3 mm
arclength intervals in world coordinates. It describes sampled turning on the
support, including mesh faceting. Manufacturing bend limits apply to the final
conductor and must be reviewed separately.

The biplanar system's Z2 channel accepted 2 of 52 fits. At the default six
working samples per mesh length, 46 fits exceeded the displacement limit and
four crossed a face repeatedly. A separate contour-only run with 18 working
samples per mesh length still accepted 2 of 52. Further improvement on this
channel requires investigating the support discretization or the allowable
physical change, then recomputing its wire field and clearances.

For conical Z2 we also repeated the contour fit on the same 37 extracted
turns, with the 0.5 mm cap and spacing guard active. The 0.1 trial reproduced
every stored fitted vertex. At gains 0.1, 0.3 and 0.6, the 3 mm P95 turn angle
was 9.216°, 9.198° and 9.208°, respectively, versus 8.462° with processing
off. Each accepted 16 fits. The fixed-current contour RMS was 14.968%,
14.971% and 14.970%, versus 15.016% with processing off. All four
measurements used 1.75156235 A. Neither gentler setting improved the turning
metric relative to the original, so the conical gallery setting remains 0.1.
These two sensitivity cases stopped at contours. [Measurements](cone-gains.csv)
and [reproduction script](check_cone_gains.py).

For actively shielded Gy, the area-weighted vector RMS on the 314 original
suppression-surface samples was 19.9987 µT for the sheet in both runs. The
fabricated winding changed from 29.5578 to 29.5561 µT, above the 20 µT limit.
These are design-surface measurements. They do not bound the external field
on an independent surface.

The [comparison matrix](smoothing.csv) and [benchmark script](check_smoothing.py)
contain results and reproduction instructions. Each matrix row links by filename
and SHA-256 to a compressed [centerline archive](centerlines/). Archives
retain the world-frame support vertices and faces, original and smoothed contour
samples, and the reviewed conductor centerlines. All coordinates are in metres.
The `original_` prefix selects the unprocessed comparison. The other arrays
come from the smoothed runs. If an archived checkpoint belongs to an earlier
engine digest, the benchmark computes a new sheet and compares the stream
function, design field and holdout arrays bit for bit. If they differ it also
runs the unprocessed geometry from the new sheet. On an installation without
the archived job directories, the benchmark computes an unprocessed example
first and reuses its saved sheet checkpoint for the smoothed run. Pass
`--fresh-baseline` to request fresh pairs explicitly.
